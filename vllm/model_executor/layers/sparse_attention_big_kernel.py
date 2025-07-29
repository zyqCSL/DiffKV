"""Multi-head attention."""
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
# from xformers import ops as xops
# from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
#                                          LowerTriangularMaskWithTensorBias)

from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.triton_flash_attention import triton_attention
from vllm.model_executor.layers.triton_fused_softmax_sum import triton_fused_softmax_sum
from vllm.utils import is_hip

import time

_SUPPORTED_HEAD_SIZES = [16, 64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in the attention kernel
_PARTITION_SIZE = 4800
_PARTITION_SIZE_PROMPT = 4000
# _PARTITION_SIZE_PROMPT = 8

# Use the qk products computed between the last few tokens and all tokens
# in the prompt to compute the compression metric.
# Note: this parameter should be consistent with the one in triton_flash_attention.py
NUM_TOKENS_SCORE = 64

class SparsePagedAttention(nn.Module):
    """MHA/MQA/GQA layer with PagedAttention.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Reshape and store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention using either
        xformers or the PagedAttention custom op.
    3. Return the output tensor.
    """

    def __init__(
        self,
        layer: int,
        num_heads: int,
        head_size: int,
        scale: float,
        kv_buffer_size: int,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        ''' Args
        local_buffer: number of most recent tokens that are never pruned
        '''
        super().__init__()
        self.layer = layer  # layer id
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.kv_buffer_size = kv_buffer_size    # size of local kv buffer

        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        # TODO (yanqi): modify save_kv_cache to support max_kv_slots and sliding_window
        assert self.sliding_window is None

        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_mapping = torch.repeat_interleave(
            torch.arange(self.num_kv_heads, dtype=torch.int32, device="cuda"),
            self.num_queries_per_kv)

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def set_attn_bias(
        self,
        input_metadata: InputMetadata,
        batch_size: int,
        seq_len: int,
        prompt_lens: List[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        '''
        TODO: this should be optimized (refer to the original implementation)
        '''
        if input_metadata.attn_bias is not None:
            # Already set by a previous layer.
            return
        input_metadata.attn_bias = _make_causal_mask(
            batch_size, seq_len, dtype, device, self.sliding_window)
        # ignore padding tokens for each sequence
        for batch_id in range(batch_size):
            prompt_len = prompt_lens[batch_id]
            input_metadata.attn_bias[batch_id, :, prompt_len:, :] = torch.finfo(dtype).min
        if self.alibi_slopes is not None:
            input_metadata.attn_bias += _make_alibi_bias(
                self.alibi_slopes, batch_size, seq_len, dtype)


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        input_positions: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> Tuple[torch.Tensor]:
        """PagedAttention forward pass.

        Args:
            query: shape = [total_num_tokens, num_heads * head_size]
            key: shape = [total_num_tokens, num_kv_heads * head_size]
            value: shape = [total_num_tokens, num_kv_heads * head_size]
            kv_cache: shape = [num_blocks, block_size]
            input_metadata: metadata for the inputs.
            Notes: positional embeddings are already applied to query & key
            Allocation of kv cache is in CacheEngine
        Returns:
            output: shape = [total_num_tokens, num_heads * head_size]
        """
        slot_ids = input_metadata.slot_ids
        block_tables = input_metadata.block_tables
        kv_len_tables = input_metadata.kv_len_tables
        compress_config_tables = input_metadata.compress_config_tables

        if input_metadata.is_prompt:
            # Naive prompt-phase attention implementation composed of small kernels.
            # Performance is bad, 10x slower than flash attention.
            # query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
            # key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
            # value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_size)

            # key_ = key.clone()
            # value_ = value.clone()

            # # TODO: remove repeat_interleave in the unified kernel
            # # [batch_size, seq_len, num_heads, head_size]
            # key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            # value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

            # query = query.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_size]
            # key = key.transpose(1, 2)      # [batch_size, num_heads, seq_len, head_size]
            # value = value.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_size]
            # score = torch.matmul(query, key.transpose(2, 3)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]

            # # attention mask
            # self.set_attn_bias(
            #     input_metadata,
            #     batch_size,
            #     seq_len,
            #     input_metadata.prompt_lens,
            #     query.dtype,
            #     query.device,
            # )

            # score = score + input_metadata.attn_bias
            # score = F.softmax(score, dim=-1, dtype=torch.float32)  # keep float32 for caching

            # # TODO: skip padded token in the kernel
            # # in the last layer we need to remove the padding and extract the last token in sequence as output
            # output = torch.matmul(score.to(query.dtype), value)  # [batch_size, num_heads, seq_len, head_size]
            # output = output.transpose(1, 2).contiguous()
            # output = output.view(batch_size, seq_len, self.num_heads * self.head_size)

            # # TODO(yuwei): move to cuda kernel
            # for batch_id in range(batch_size):
            #     seq_len = input_metadata.prompt_lens[batch_id]
            #     score[batch_id, :, seq_len:, :] = 0
            #     score[batch_id, :, :seq_len, seq_len:] = 0

            # score_slice = score[:, :, -NUM_TOKENS_SCORE:, :]
            # score_sum = torch.sum(score_slice, dim=2, dtype=score.dtype)

            # print('[DEBUG] SparsePagedAttention starts processing prompt')

            ############# Triton flash attention #################
            torch.cuda.synchronize()
            t0 = time.time()

            batch_size = len(input_metadata.prompt_lens)
            max_prompt_len = max(input_metadata.prompt_lens)
            seq_start_loc = input_metadata.seq_start_loc

            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)

            triton_out, qk_products = triton_attention(
                query,
                key,
                value,
                seq_start_loc,
                seq_start_loc,
                max_prompt_len,
                max_prompt_len,
                True,  # causal
                self.scale,
                None,  # bias
            )

            torch.cuda.synchronize()
            t1 = time.time()

            # print('[DEBUG] SparsePagedAttention prompt processed')

            triton_score_sum = triton_fused_softmax_sum(qk_products)

            torch.cuda.synchronize()
            t2 = time.time()

            del qk_products
            
            """
            The following is pytorch implementation of the same functionality as triton_fused_softmax_sum.
            It is less efficient, only for testing purposes.

            Shape of triton_score: [batch_size, num_heads, NUM_TOKENS_SCORE, max_prompt_len]
            FlashAttention-2 multiplies qk_products by log_2(e) and replaces exp(x) by 2^x when computing softmax. 2^(log_2(e)*x) = exp(x).
            To use standard softmax, we need to divide qk_products by log_2(e).

            Shape of triton_score_sum: [batch_size, num_heads, max_prompt_len]
            Why need torch.nan_to_num? If the prompt length < NUM_TOKENS_SCORE, triton_score will contain nan.
            """
            # triton_score_ = F.softmax(qk_products / 1.44269504089, dim=-1, dtype=torch.float32)
            # triton_score_sum_ = torch.sum(torch.nan_to_num(triton_score_, nan=0.0), dim=2, dtype=triton_score_.dtype)
            # assert torch.allclose(triton_score_sum, triton_score_sum_, atol=1e-5)

            # save pruned prompts to kv cache (key & value are before repeat_interleave)
            # if kv cache is not provided, nothing should be saved
            # this happens in the initial memory profiling run
            if kv_cache is not None and block_tables is not None and kv_len_tables is not None:
                if max_prompt_len <= _PARTITION_SIZE_PROMPT:
                    # NOTE: slot mapping should be computed within the kernel
                    cache_ops.compress_and_append_cache_prompt_phase(
                        key,
                        value,
                        triton_score_sum,
                        kv_cache,
                        slot_ids,
                        block_tables,
                        kv_len_tables,
                        seq_start_loc,
                        compress_config_tables,
                        self.kv_buffer_size,
                        self.layer,
                        input_metadata.num_bits_k_high,
                        input_metadata.num_bits_v_high,
                        input_metadata.num_bits_k_low,
                        input_metadata.num_bits_v_low,
                        input_metadata.num_chunks_k_high,
                        input_metadata.num_chunks_v_high,
                        input_metadata.num_chunks_k_low,
                        input_metadata.num_chunks_v_low,
                        input_metadata.key_vec_size,
                        input_metadata.val_vec_size,
                        input_metadata.num_tokens_per_block_high,
                        input_metadata.num_tokens_per_block_low)

                    torch.cuda.synchronize()
                    t3 = time.time()
                    
                    del triton_score_sum

                    # print(f'Triton flash attention: {(1000 * (t1 - t0)):.2f} ms')
                    # print(f'Softmax: {(1000 * (t2 - t1)):.2f} ms')
                    # print(f'compress_and_append_cache_prompt_phase: {(1000 * (t3 - t2)):.2f} ms')
                else:
                    print(f'sparse_attn_big_kernel::max_prompt_len {max_prompt_len} > _PARTITION_SIZE_PROMPT {_PARTITION_SIZE_PROMPT}')
                    
                    # process the scores in global memory instead of GPU shared memory in cuda kernels 
                    triton_score_sum = triton_score_sum.view(
                        batch_size, self.num_kv_heads, self.num_queries_per_kv, max_prompt_len)
                    # [batch_size, num_kv_heads, max_prompt_len]
                    max_score_sum = torch.max(triton_score_sum, dim=2).values
                    
                    # compute the mean score in global memory for sorting
                    indices = torch.tensor(range(max_prompt_len), dtype=torch.int, device=max_score_sum.device)
                    indices = indices.view(1, 1, max_prompt_len).expand(batch_size, self.num_kv_heads, max_prompt_len)
                    prompt_lens = torch.tensor(input_metadata.prompt_lens, dtype=torch.int, device=max_score_sum.device)
                    prompt_lens = prompt_lens.view(batch_size, 1, 1).expand(batch_size, self.num_kv_heads, max_prompt_len)
                    ideal_num_queries = prompt_lens - indices
                    # NOTE: we only dump scores of $NUM_TOKENS_SCORE tokens
                    # we need the > 0 condition here to avoid dividing by 0
                    real_num_queries = torch.where((ideal_num_queries < NUM_TOKENS_SCORE) & (ideal_num_queries > 0), 
                                                   ideal_num_queries, NUM_TOKENS_SCORE)
                    # [batch_size, num_kv_heads, max_prompt_len]                
                    max_score_mean = max_score_sum / real_num_queries
                    # set scores of tokens in kv_buffer to 1.0 and padded tokens to 10.0
                    # so recent tokens will be after older tokens, and padded tokens will be the last
                    for seq_idx, prompt_len in enumerate(input_metadata.prompt_lens):
                        max_score_mean[seq_idx, :, prompt_len:] = 10.0
                        max_score_mean[seq_idx, :, max(0, prompt_len - self.kv_buffer_size):prompt_len] = 1.0
                    # Sort the scores in aescending order
                    _, sorted_indices = torch.sort(max_score_mean, dim=-1)
                    sorted_indices = sorted_indices.to(torch.int32)
                    cache_ops.compress_and_append_cache_long_prompt_phase(
                        key,
                        value,
                        max_score_sum,
                        sorted_indices,
                        kv_cache,
                        slot_ids,
                        block_tables,
                        kv_len_tables,
                        seq_start_loc,
                        compress_config_tables,
                        self.kv_buffer_size,
                        self.layer,
                        input_metadata.num_bits_k_high,
                        input_metadata.num_bits_v_high,
                        input_metadata.num_bits_k_low,
                        input_metadata.num_bits_v_low,
                        input_metadata.num_chunks_k_high,
                        input_metadata.num_chunks_v_high,
                        input_metadata.num_chunks_k_low,
                        input_metadata.num_chunks_v_low,
                        input_metadata.key_vec_size,
                        input_metadata.val_vec_size,
                        input_metadata.num_tokens_per_block_high,
                        input_metadata.num_tokens_per_block_low)

                    torch.cuda.synchronize()
                    t3 = time.time()
                    
                    del triton_score_sum, max_score_sum, max_score_mean, real_num_queries, sorted_indices
                    
            # print('[DEBUG] SparsePagedAttention prompt KV saved')

            return triton_out.view(-1, self.num_heads * self.head_size)
        else:
            torch.cuda.synchronize()
            t0 = time.time()

            # print('[DEBUG] SparsePagedAttention starts processing decode')

            # Decoding run
            batch_size = query.size(0)
            query = query.view(batch_size, self.num_heads, self.head_size)
            key = key.view(batch_size, self.num_kv_heads, self.head_size)
            value = value.view(batch_size, self.num_kv_heads, self.head_size)

            output = torch.zeros((batch_size, self.num_heads, self.head_size),
                                  dtype=query.dtype, device=query.device)

            max_context_len = input_metadata.max_context_len[self.layer].item()

            assert self.alibi_slopes is None, "ALiBi is currently not supported"

            # Prune kv cache based on scores of each kv head and then update kv cache
            cache_ops.compress_and_append_cache_decode_phase(
                key,
                value,
                input_positions,
                kv_cache,
                slot_ids,
                block_tables,
                kv_len_tables,
                compress_config_tables,
                max_context_len,
                self.kv_buffer_size,
                self.layer,
                input_metadata.num_bits_k_high,
                input_metadata.num_bits_v_high,
                input_metadata.num_bits_k_low,
                input_metadata.num_bits_v_low,
                input_metadata.num_chunks_k_high,
                input_metadata.num_chunks_v_high,
                input_metadata.num_chunks_k_low,
                input_metadata.num_chunks_v_low,
                input_metadata.key_vec_size,
                input_metadata.val_vec_size,
                input_metadata.num_tokens_per_block_high,
                input_metadata.num_tokens_per_block_low)

            torch.cuda.synchronize()
            t1 = time.time()

            # print('[DEBUG] SparsePagedAttention decode KV cache saved')

            # key_cache_reshaped = torch.transpose(key_cache, 1, 2).contiguous().view(-1, block_size, self.head_size)
            # key_cache_np = key_cache_reshaped.cpu().numpy()
            # import numpy as np
            # np.save("big_kernel_key_cache_layer_" + str(self.layer) + ".npy", key_cache_np)

            # a new token might be appended in this forward pass
            max_context_len += 1
            # tmp_scores is a buffer to store the attention scores for each kv head
            # TODO: we should optimize this
            tmp_scores = torch.empty((batch_size, self.num_heads, max_context_len),
                                      dtype=torch.float32, device=query.device)

            if max_context_len <= _PARTITION_SIZE:
                num_partitions = 1
            else:
                num_partitions = ((max_context_len + _PARTITION_SIZE - 1) // _PARTITION_SIZE) + 1
            tmp_output = torch.empty(
                size=(batch_size, self.num_heads, num_partitions, self.head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(batch_size, self.num_heads, num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)

            # TODO(@yanqi): Pass the following parameters from the framework. Now they are hardcoded.
            ops.sparse_paged_attention(
                slot_ids,
                input_positions,
                output,
                exp_sums,
                max_logits,
                tmp_output,
                tmp_scores,
                query,
                kv_cache,
                self.layer,
                self.num_kv_heads,
                self.scale,
                block_tables,
                kv_len_tables,
                max_context_len,
                input_metadata.num_bits_k_high,
                input_metadata.num_bits_v_high,
                input_metadata.num_bits_k_low,
                input_metadata.num_bits_v_low,
                input_metadata.num_chunks_k_high,
                input_metadata.num_chunks_v_high,
                input_metadata.num_chunks_k_low,
                input_metadata.num_chunks_v_low,
                input_metadata.key_vec_size,
                input_metadata.val_vec_size,
                input_metadata.num_tokens_per_block_high,
                input_metadata.num_tokens_per_block_low,
                self.alibi_slopes,
            )

            torch.cuda.synchronize()
            t2 = time.time()

            # print('[DEBUG] SparsePagedAttention decode processed')

            # if self.layer == 1:
            #     print(f'Batch size: {batch_size}, Max context len: {max_context_len}')
            #     print(f'compress_and_append_cache_decode_phase: {(1000 * (t1 - t0)):.2f} ms')
            #     print(f'sparse_paged_attention: {(1000 * (t2 - t1)):.3f} ms')

            output = output.view(-1, self.num_heads * self.head_size)
            return output


def _make_causal_mask(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    assert seq_len > 1
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    mask = mask.to(dtype)

    # add lower triangular sliding window mask if necessary
    if sliding_window is not None:
        diagonal = - sliding_window + 1
        context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
        mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

    return mask[None, None, :, :].repeat(batch_size, 1, 1, 1)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    '''
    TODO: check if this bias works in autoregressive generation
    should we add a causal mask to alibi?
    '''
    bias = torch.arange(seq_len, dtype=dtype, device="cuda")
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]
    mask_cond = torch.arange(bias.size(-1), device=bias.device)
    bias.masked_fill_(mask_cond.view(bias.size(-1), 1) < mask_cond + 1, 0)

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    # TODO: or now ignore padding
    # padded_len = (seq_len + 7) // 8 * 8
    padded_len = seq_len
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    return bias


# make alibi bias of a single head during decode
def _make_decode_alibi_bias(
    alibi_slopes: torch.Tensor,
    head: int,
    position: torch.Tensor,
    indicies: torch.Tensor,
) -> torch.Tensor:
    bias = (indicies - position) * alibi_slopes[head]
    return bias
