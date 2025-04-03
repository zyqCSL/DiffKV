"""Multi-head attention."""
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# from xformers import ops as xops
# from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
#                                          LowerTriangularMaskWithTensorBias)

# from vllm._C import ops
# from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.utils import is_hip

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


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
        kv_score_thresh: float,
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
        assert kv_score_thresh >= 0 and kv_score_thresh <= 1
        self.kv_score_thresh = torch.tensor(kv_score_thresh,
                                            dtype=torch.float32, device='cuda')
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
            input_metadata.attn_bias = input_metadata.attn_bias + _make_alibi_bias(
                self.alibi_slopes, batch_size, seq_len, dtype)


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        score_cache: Optional[torch.Tensor],
        position_cache: Optional[torch.Tensor],
        input_positions: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PagedAttention forward pass.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: for now shape = [num_blocks, block_size, head_size]
                TODO: change shape to [num_blocks, head_size // x, block_size, x] to cope w. cuda kernel
            value_cache: shape = [num_blocks, block_size, head_size]
                TODO: change shape to [num_blocks, head_size, block_size] to cope w. cuda kernel
            score_cache: shape = [num_blocks, block_size]
            position_cache: shape = [num_blocks, block_size]
            input_metadata: metadata for the inputs.
            Notes: positional embeddings are already applied to query & key
            Allocation of kv cache is in CacheEngine
        Returns:
            output: shape = [batch_size, seq_len, num_heads * head_size]
            kv_len: shape = [batch_size, num_kv_heads]
        """
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        # TODO: rope should be applied before attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_size)

        # process each prompt sequentially
        if input_metadata.is_prompt:
            # Prompt run
            key_ = key
            value_ = value
            # TODO: remove repeat_interleave in the unified kernel
            # [batch_size, seq_len, num_heads, head_size]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

            query = query.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_size]
            key = key.transpose(1, 2)      # [batch_size, num_heads, seq_len, head_size]
            value = value.transpose(1, 2)   # [batch_size, num_heads, seq_len, head_size]
            score = torch.matmul(query, key.transpose(2, 3)) * self.scale    # [batch_size, num_heads, seq_len, seq_len]

            # attention mask
            self.set_attn_bias(
                input_metadata,
                batch_size,
                seq_len,
                input_metadata.prompt_lens,
                query.dtype,
                query.device,
            )

            score = score + input_metadata.attn_bias
            score = F.softmax(score, dim=-1, dtype=torch.float32)   # keep float32 for caching

            # prune prompt tokens
            saved_kv_len, saved_kv_indices, saved_kv_scores = \
                self._prune_prompt_kv_cache(score, input_metadata.prompt_lens)

            # TODO: debug info, remove later
            # print(f'[DEBUG] SparsePagedAttention, prompt, layer: {self.layer}, saved_kv_len: {saved_kv_len}')

            # save pruned prompts to kv cache (key & value are before repeat_interleave)
            # if kv cache is not provided, nothing should be saved
            # this happens in the initial memory profiling run
            if key_cache is not None and value_cache is not None and \
                position_cache is not None and score_cache is not None:
                self._save_prompt_kv_cache(
                    key_,
                    value_,
                    saved_kv_scores,
                    saved_kv_len,
                    saved_kv_indices,
                    key_cache,
                    value_cache,
                    score_cache,
                    position_cache,
                    input_metadata.slot_mapping[self.layer],
                )

            # TODO: skip padded token in the kernel
            # in the last layer we need to remove the padding and extract the last token in sequence as output
            output = torch.matmul(score.to(query.dtype), value) # [batch_size, num_heads, seq_len, head_size]
            output = output.transpose(1, 2).contiguous()
            output = output.view(batch_size, seq_len, hidden_size)  # [bsz, seq_len, num_heads * head_size]
            return output, saved_kv_len
        else:
            # Decoding run
            output = torch.zeros_like(query)    # [batch_size, seq_len, num_heads, head_size]
            saved_kv_len = torch.zeros((batch_size, self.num_kv_heads),
                                       dtype=torch.int, device=query.device)

            context_lens = input_metadata.context_lens[self.layer]  # [batch_size, num_kv_heads]
            block_tables = input_metadata.block_tables[self.layer]  # [batch_size, num_kv_heads, kv_len // block_size]
            block_size = input_metadata.block_size

            # TODO: process each sequence iteratively. Parallelize later
            for batch_id in range(batch_size):
                for kv_head in range(self.num_kv_heads):
                    kv_len = context_lens[batch_id, kv_head].item()
                    # we need to compute the effective block table length
                    # as sparse kv of each head can be of different lengths in the same sequence
                    num_blocks = (kv_len + block_size - 1) // block_size
                    block_ids = block_tables[batch_id, kv_head, :num_blocks]

                    # key, value, score & position caches should have identical block ids
                    num_slots = num_blocks * block_size
                    # read key cache
                    head_key = torch.zeros((kv_len + 1, self.head_size),
                                           dtype=key.dtype, device=key.device) # [kv_len + 1, head_size]
                    _key_cache = key_cache[block_ids].reshape(-1, self.head_size) # [num_blocks * block_size, head_size]
                    head_key[:kv_len] = _key_cache[:kv_len]

                    # read value cache
                    head_value = torch.zeros((kv_len + 1, self.head_size),
                                             dtype=value.dtype, device=value.device)  # [kv_len + 1, head_size]
                    _value_cache = value_cache[block_ids].reshape(-1, self.head_size)   # [num_blocks * block_size, head_size]
                    head_value[:kv_len] = _value_cache[:kv_len]

                    # read index of cached tokens
                    kv_indices = None
                    if self.alibi_slopes is not None:
                        kv_indices = torch.zeros((kv_len + 1),
                                                  dtype=position_cache.dtype, device=position_cache.device)
                        kv_indices[:kv_len] = position_cache[block_ids].reshape(-1)[:kv_len]

                    # init score, to be reduced across GQA
                    kv_score = torch.zeros(kv_len + 1, dtype=score_cache.dtype, device=score_cache.device)

                    # iterate over GQA
                    for head in range(kv_head * self.num_queries_per_kv,
                                      (kv_head + 1) * self.num_queries_per_kv):
                        # key of new token
                        head_key[kv_len] = key[batch_id, 0, kv_head]
                        # value of new token
                        head_value[kv_len] = value[batch_id, 0, kv_head]

                        # attention score
                        head_query = query[batch_id, 0, head]   # [head_size]
                        head_score = torch.matmul(head_query, head_key.transpose(0, 1)) * self.scale  # [kv_len + 1]
                        # alibi should be added directly to attn score
                        if self.alibi_slopes is not None:
                            kv_indices[-1] = input_positions[batch_id]
                            head_score += _make_decode_alibi_bias(
                                self.alibi_slopes, head, input_positions[batch_id], kv_indices)


                        head_score = F.softmax(head_score, dim=-1, dtype=torch.float32)
                        # aggregate scores of heads in the same GQA group
                        # TODO: experiment average later
                        kv_score = torch.maximum(kv_score, head_score) # [kv_len + 1]

                        # compute output
                        head_output = torch.matmul(head_score.to(query.dtype), head_value) # [head_size]
                        output[batch_id, 0, head, :] = head_output # [batch_size, seq_len, num_heads, head_size]

                    # prune kv cache based on scores of each kv head
                    pruned_kv_len, pruned_indices = self._prune_decode_kv_cache(
                        kv_len,
                        kv_score,
                        score_cache[block_ids].reshape(num_slots)[:kv_len],
                        position_cache[block_ids].reshape(num_slots)[:kv_len],
                        input_positions[batch_id])

                    saved_kv_len[batch_id, kv_head] = pruned_kv_len

                    # save pruned kv cache
                    self._save_decode_kv_cache(
                        key=key,
                        value=value,
                        score=kv_score,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        score_cache=score_cache,
                        position_cache=position_cache,
                        block_size=input_metadata.block_size,
                        block_ids=block_ids,
                        slot_mapping=input_metadata.slot_mapping[self.layer],
                        position=input_positions[batch_id],
                        batch_id=batch_id,
                        head=kv_head,
                        kv_len=kv_len,
                        victim_index=pruned_indices,
                    )

            # # TODO: debug info, remove later
            # print(f'decode, layer: {self.layer}, saved_kv_len: {saved_kv_len}')

            output = output.view(batch_size, seq_len, hidden_size)  # [bsz, seq_len, num_heads * head_size]
            return output, saved_kv_len


    def _prune_prompt_kv_cache(
        self,
        scores: torch.Tensor,   # [batch_size, num_heads, seq_len, seq_len]
        prompt_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ''' Prune the prompt tokens according to scores while keeping the local buffer
        Args
            scores: Attention scores of each token
                shape = [batch_size, num_heads, seq_len, seq_len]  (query, key)
        Return
            saved_kv_len: Effective kv len of each head, shape = [batch_size, num_kv_heads]
            saved_kv_indices: Index of saved tokens (in its sequence) of each head,
                shape = [batch_size, num_kv_heads, seq_len]
            kv_scores: accumulated scores of saved tokens, shape = [batch_size, num_kv_heads, seq_len]
        '''
        batch_size, _, max_seq_len = scores.shape[:3]
        # outputs
        saved_kv_len = torch.zeros((batch_size, self.num_kv_heads),
                                   dtype=torch.int, device=scores.device)
        saved_kv_indices = torch.zeros((batch_size, self.num_kv_heads, max_seq_len),
                                       dtype=torch.long, device=scores.device)

        # aggregate scores of GQA along the query dim
        if self.num_heads > self.num_kv_heads:
            # TODO: experiment other methods (mean, e.g.)
            scores = scores.view(
                batch_size, self.num_kv_heads, self.num_queries_per_kv, max_seq_len, max_seq_len)
            kv_scores, _ = torch.max(scores, dim=2)     # [batch_size, num_kv_heads, seq_len, seq_len]

            ''' The mean implementation
            score_head_mapping = self.head_mapping[None, :, None, None].expand(
                batch_size, self.num_heads, max_seq_len, max_seq_len,
            )
            kv_scores = torch.zeros(
                (batch_size, self.num_kv_heads, max_seq_len, max_seq_len),
                dtype=scores.dtype,
                device=scores.device)
            # add along the head dim
            kv_scores.scatter_add_(1, score_head_mapping, scores)
            '''
        else:
            kv_scores = scores

        # TODO: parallelize different batches
        # mask scores of padding tokens
        for batch_id in range(batch_size):
            seq_len = prompt_lens[batch_id]
            kv_scores[batch_id, :, seq_len:, :] = 0
            kv_scores[batch_id, :, :seq_len, seq_len:] = 0

        kv_scores = torch.sum(kv_scores, dim=2, dtype=scores.dtype)     # [batch_size, num_kv_heads, seq_len]
        # average scores with number of queries
        # we use another copy here as kv_scores is used as return variable
        mean_kv_scores = torch.zeros_like(kv_scores,
                                          device=kv_scores.device, dtype=kv_scores.dtype)
        # average scores with number of queries
        kv_index = torch.arange(max_seq_len, dtype=torch.long, device=scores.device)

        # TODO: iterate over each sequence and each head, parallelize later
        for batch_id in range(batch_size):
            seq_len = prompt_lens[batch_id]
            if seq_len <= self.kv_buffer_size:
                # save the entire kv cache as local buffer
                saved_kv_len[batch_id, :] = seq_len
                saved_kv_indices[batch_id, :, :seq_len] = kv_index[:seq_len]
                continue
            # amortize each token by times it is queried
            num_queries = seq_len - kv_index[:seq_len]
            # TODO: remove assert later
            assert torch.all(num_queries > 0)
            mean_kv_scores[batch_id, :, :seq_len] = kv_scores[batch_id, :, :seq_len] / num_queries
            # compute the average score of each token
            sum_kv_scores = torch.sum(
                mean_kv_scores[batch_id, :, :seq_len], dim=-1, keepdim=True)    # [num_kv_heads, 1]
            mean_kv_scores[batch_id, :, :seq_len] /= sum_kv_scores
            buffer_start = seq_len - self.kv_buffer_size
            # scores of non-evictable kv buffer
            buffer_sum_kv_scores = torch.sum(
                mean_kv_scores[batch_id, :, buffer_start:seq_len], dim=-1, keepdim=True)   # [num_kv_heads, 1]
            # sort the tokens outside non-evictable buffer by scores
            sorted_scores, sorted_indices = torch.sort(
                mean_kv_scores[batch_id, :, :buffer_start],
                dim=-1, descending=True)    # [num_kv_heads, seq_len - kv_buffer_size]

            sorted_scores = torch.cumsum(sorted_scores, dim=-1) # [num_kv_heads, seq_len - kv_buffer_size]
            sorted_scores += buffer_sum_kv_scores
            masked_scores = torch.where(
                sorted_scores >= self.kv_score_thresh,
                sorted_scores,
                0)
            for head in range(self.num_kv_heads):
                # process each head individually
                cut_indices = torch.nonzero(masked_scores[head])
                if cut_indices.shape[0] == 0:
                    # No element can be pruned, save all
                    saved_kv_len[batch_id, head] = seq_len
                    saved_kv_indices[batch_id, head, :buffer_start] = sorted_indices[head]
                    saved_kv_indices[batch_id, head, buffer_start:seq_len] = kv_index[buffer_start:seq_len]
                    # saved_kv_indices[batch_id, head, :seq_len] = kv_index[:seq_len]
                    continue
                # check which elements should be saved
                cut_index = cut_indices[0][0]
                saved_indices = sorted_indices[head, :cut_index + 1]
                saved_len = cut_index + 1 + self.kv_buffer_size
                saved_kv_len[batch_id, head] = saved_len
                # older tokens after pruning
                saved_kv_indices[batch_id, head, :cut_index + 1] = saved_indices
                # local kv buffer
                saved_kv_indices[batch_id, head, cut_index + 1:saved_len] = kv_index[buffer_start:seq_len]

        return saved_kv_len, saved_kv_indices, kv_scores

    # save pruned prompt kv cache
    # TODO (yanqi): add support for sliding window here
    def _save_prompt_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        score: torch.Tensor,
        saved_kv_len: torch.Tensor,
        saved_kv_indices: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        score_cache: torch.Tensor,
        position_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        ''' Args
        key: shape = [batch_size, seq_len, num_kv_heads, head_size]
        value: shape = [batch_size, seq_len, num_kv_heads, head_size]
        score: accmulated scores of each token, shape = [batch_size, num_kv_heads, seq_len]
        key_cache: for now shape = [num_blocks, block_size, head_size]
            TODO: change shape to [num_blocks, head_size // x, block_size, x] to cope w. cuda kernel
        value_cache: shape = [num_blocks, block_size, head_size]
            TODO: change shape to [num_blocks, head_size, block_size] to cope w. cuda kernel
        score_cache: shape = [num_blocks, block_size]
        position_cache: shape = [num_blocks, block_size]
        saved_kv_len: shape = [batch_size, num_kv_heads]
        saved_kv_indices: shape = [batch_size, num_kv_heads, max_seq_len]
        slot_mapping: shape = [batch_size, num_kv_heads, max_seq_len]
        '''
        batch_size = key.shape[0]
        # treat each cache as an array
        key_cache = key_cache.view(-1, self.head_size)      # [num_blocks * block_size, head_size]
        value_cache = value_cache.view(-1, self.head_size)  # [num_blocks * block_size, head_size]
        score_cache = score_cache.view(-1)      # [num_blocks * block_size]
        position_cache = position_cache.view(-1)      # [num_blocks * block_size]

        for batch_id in range(batch_size):
            for head in range(self.num_kv_heads):
                seq_len = saved_kv_len[batch_id, head]
                saved_indices = saved_kv_indices[batch_id, head, :seq_len]  # [seq_len]
                cache_slots = slot_mapping[batch_id][head][:seq_len]      # [seq_len]
                # save to cache
                key_cache[cache_slots, :] = key[batch_id, saved_indices, head, :]   # [seq_len, head_size]
                value_cache[cache_slots, :] = value[batch_id, saved_indices, head, :]
                score_cache[cache_slots] = score[batch_id, head, saved_indices]
                position_cache[cache_slots] = saved_indices


    # prune kv cache of one head during decode
    def _prune_decode_kv_cache(
        self,
        kv_len: int,
        score: torch.Tensor,
        cached_score: torch.Tensor,
        cached_index: torch.Tensor,
        position: torch.Tensor,
    ) -> Tuple[int, Optional[torch.Tensor]]:
        ''' Args
            score: shape = [kv_len + 1]
            cached_score: shape = [kv_len]
            cached_index: shape = [kv_len]
        return
            kv_len: the effective kv len after pruning
            prune_indices: index (in the kv cache) of tokens to be pruned, shape = [1]
        '''
        if kv_len + 1 <= self.kv_buffer_size:
            return kv_len + 1, None
        else:
            # create a new copy so as not to pollute the cache
            mean_score = torch.zeros(kv_len + 1,
                                    dtype=cached_score.dtype, device=cached_score.device)
            mean_score += score
            # sum up cached accumulated scores
            mean_score[:kv_len] += cached_score
            # compute number of queries of each token
            num_queries = position - cached_index + 1  # [kv_len]
            # TODO: remove assert later
            assert torch.all(num_queries > 1)

            mean_score[:kv_len].div_(num_queries) # [kv_len + 1]
            sum_score = torch.sum(mean_score)
            mean_score.div_(sum_score)

            # prune the kv cache
            prunable_bitmap = torch.where(
                num_queries > self.kv_buffer_size,
                True, False)
            prunable_indices = torch.nonzero(prunable_bitmap).reshape(-1)
            # TODO: remove assert later
            assert prunable_indices.shape[0] == kv_len + 1 - self.kv_buffer_size
            prunable_score = mean_score[:kv_len]   # the last token is the new token
            min_score, _min_index = torch.min(prunable_score[prunable_bitmap], dim=-1)
            min_index = prunable_indices[_min_index]
            if min_score >= 1 - self.kv_score_thresh:
                # min score is not small enough to be pruned
                return kv_len + 1, None
            else:
                return kv_len, min_index

    # update kv cache of one head during decode
    # TODO (yanqi): add support of sliding window here
    def _save_decode_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        score: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        score_cache: torch.Tensor,
        position_cache: torch.Tensor,
        block_size: int,
        block_ids: torch.Tensor,
        slot_mapping: Optional[torch.Tensor],
        position: torch.Tensor,
        batch_id: int,
        head: int,
        kv_len: int,
        victim_index: Optional[torch.Tensor],
    ) -> None:
        ''' Args:
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            score: shape = [kv_len + 1], scores of all attended tokens
            key_cache: for now shape = [num_blocks, block_size, head_size]
                TODO: change shape to [num_blocks, head_size // x, block_size, x] to cope w. cuda kernel
            value_cache: shape = [num_blocks, block_size, head_size]
                TODO: change shape to [num_blocks, head_size, block_size] to cope w. cuda kernel
            score_cache: shape = [num_blocks, block_size]
            position_cache: shape = [num_blocks, block_size]
            Notes: positional embeddings are already applied to query & key
                Allocation of kv cache is in CacheEngine
            position: index of the new token in the full sequence
            kv_len: effective kv len before the new token is added
            victim_index: index (in the pruned sequence) of the token to be replaced
        '''
        # treat each cache as an array
        key_cache = key_cache.view(-1, self.head_size)      # [num_blocks * block_size, head_size]
        value_cache = value_cache.view(-1, self.head_size)  # [num_blocks * block_size, head_size]
        position_cache = position_cache.view(-1)      # [num_blocks * block_size]
        score_cache = score_cache.view(-1)      # [num_blocks * block_size]

        if victim_index is None:
            # append the new token to key, value and position cache
            cache_slots = slot_mapping[batch_id, head, 0]
            key_cache[cache_slots, :] = key[batch_id, 0, head, :]   # [1, head_size]
            value_cache[cache_slots, :] = value[batch_id, 0, head, :]
            position_cache[cache_slots] = position
            # update scores of previous tokens
            prev_slots = _block_ids_to_slots(block_ids, block_size)[:kv_len]
            score_cache[prev_slots] += score[:kv_len]
            # update score of new token
            score_cache = score_cache.view(-1)      # [num_blocks * block_size]
            score_cache[cache_slots] = score[-1]
        else:
            # replace an existing token with new token
            prev_slots = _block_ids_to_slots(block_ids, block_size)[:kv_len]
            cache_slots = prev_slots[victim_index]
            # replace old key, value & position cache
            key_cache[cache_slots, :] = key[batch_id, 0, head, :]   # [1, head_size]
            value_cache[cache_slots, :] = value[batch_id, 0, head, :]
            position_cache[cache_slots] = position
            # update scores of previous tokens
            score_cache[prev_slots] += score[:kv_len]
            # override score of replaced token
            score_cache[cache_slots] = score[kv_len]


def _block_ids_to_slots(
    block_ids: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    ''' Args
    block_ids: shape = [num_blocks]
    '''
    slots = torch.arange(block_size,
                         dtype=block_ids.dtype, device=block_ids.device)
    slots = block_ids.view(-1, 1) * block_size + slots
    slots = slots.view(-1)
    return slots


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

    # return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)   # dim 1 reserved for num_heads
    # NOTE: We need to repeat on batch dimension instead of expand,
    # since they might be modified later according to their own seq length
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
