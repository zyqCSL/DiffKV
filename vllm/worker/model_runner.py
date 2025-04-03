import time
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, CacheConfig
from vllm.logger import init_logger
from vllm.model_executor import get_model, InputMetadata, SamplingMetadata
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.utils import in_wsl

logger = init_logger(__name__)

# KV caches unifed into a single tensor
KVCache = torch.Tensor

_PAD_SLOT_ID = -1
# Capture graphs for batch size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]


class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.cache_config: CacheConfig = None    # to be set later

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.model = None
        self.block_size = None  # Set after initial profiling.
        self.num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool = None  # Set during graph capture.

        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None else 0)
        # When using CUDA graph, the input block tables must be padded to
        # max_context_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = None  # Set after initial profiling.
        # cache in_wsl result
        self.in_wsl = in_wsl()

    def load_model(
        self,
        kv_buffer_size: Union[int, List[int]],
        max_kv_slots: Optional[int] = None,
        ) -> None:
        self.model = get_model(self.model_config, kv_buffer_size, max_kv_slots)

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        max_num_blocks = (self.max_context_len_to_capture + block_size -
                          1) // block_size
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        block_tables: Optional[torch.Tensor],
        kv_len_tables: Optional[torch.Tensor],
        sparsity_tables: Optional[torch.Tensor],
        compress_config_tables: Optional[torch.Tensor],
        key_vec_size: int,
        val_vec_size: int,
    ) -> Tuple[torch.Tensor, InputMetadata]:
        assert len(seq_group_metadata_list) > 0
        # batch_size = len(seq_group_metadata_list)
        prompt_lens: List[int] = []
        input_tokens: List[int] = []
        # input_positions required by rope, but can be inferred in place in attn
        input_positions: List[int] = []
        # get max prompt lengths to get padded shapes
        slot_ids = []
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(prompt_len)))
            input_tokens.extend(prompt_tokens)
            slot_ids.extend(seq_group_metadata.slot_ids)

            # TODO: determine when to skip for memory profiling
            # if seq_group_metadata.slot_ids is None:
            #     # During memory profiling, the block tables are not initialized
            #     # yet. In this case, we just use a dummy slot mapping.
            #     continue

        # print(f'[DEBUG] _prepare_prompt prompt_lens: {prompt_lens}')
        
        # TODO: the attention kernel should be aware of sliding_window

        # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
        # where start_idx is max(0, prompt_len - sliding_window).
        # For example, if the prompt len is 10, sliding window is 8, and
        # block size is 4, the first two tokens are masked and the slot
        # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
        # FIXME (yanqi): replace sliding_window w. max_kv_slots
        # NOTE: this should be done in the kernel

        # max_prompt_len = max(prompt_lens)
        # input_tokens = _make_tensor_with_pad(input_tokens,
        #                                      max_len=max_prompt_len,
        #                                      pad=0,
        #                                      dtype=torch.int,
        #                                      device='cuda')
        # input_positions = _make_tensor_with_pad(input_positions,
        #                                         max_len=max_prompt_len,
        #                                         pad=0,
        #                                         dtype=torch.int,
        #                                         device='cuda')

        input_tokens = torch.tensor(input_tokens, dtype=torch.int, device='cuda')
        input_positions = torch.tensor(input_positions, dtype=torch.int, device='cuda')
        slot_ids = torch.tensor(slot_ids, dtype=torch.int, device='cuda')

        quant_config_high = (
            seq_group_metadata_list[0].num_bits_k_high,
            seq_group_metadata_list[0].num_bits_v_high)
        quant_config_low = (
            seq_group_metadata_list[0].num_bits_k_low,
            seq_group_metadata_list[0].num_bits_v_low)

        # We don't need input_positions as it's obvious for prompt
        if self.cache_config is not None:
            num_tokens_per_block_high = self.cache_config.quantized_block_num_tokens[quant_config_high]
            num_tokens_per_block_low = self.cache_config.quantized_block_num_tokens[quant_config_low]
        else:
            # dummpy input for profiling run
            num_tokens_per_block_high = None
            num_tokens_per_block_low = None

        # The cumulative sequence lengths of the sequences in the batch, used to index into sequence.
        # E.g., if the sequence length is [4, 6], seq_start_loc is [0, 4, 10].
        seq_start_loc = torch.zeros(len(prompt_lens) + 1, dtype=torch.int32, device='cuda')
        torch.cumsum(torch.tensor(prompt_lens, dtype=torch.int32, device='cuda'),
                     dim=0, dtype=seq_start_loc.dtype, out=seq_start_loc[1:])

        input_metadata = InputMetadata(
            slot_ids=slot_ids,
            prompt_lens=prompt_lens,
            seq_start_loc=seq_start_loc,
            max_context_len=None,
            use_cuda_graph=False,
            block_size=self.block_size,
            block_tables=block_tables,
            kv_len_tables=kv_len_tables,
            sparsity_tables=sparsity_tables,
            attn_prune_thresh=seq_group_metadata_list[0].attn_prune_thresh,
            num_bits_k_high=seq_group_metadata_list[0].num_bits_k_high,
            num_bits_v_high=seq_group_metadata_list[0].num_bits_v_high,
            num_bits_k_low=seq_group_metadata_list[0].num_bits_k_low,
            num_bits_v_low=seq_group_metadata_list[0].num_bits_v_low,
            compress_config_tables=compress_config_tables,
            key_vec_size=key_vec_size,
            val_vec_size=val_vec_size,
            num_tokens_per_block_high=num_tokens_per_block_high,
            num_tokens_per_block_low=num_tokens_per_block_low,
        )
        return input_tokens, input_positions, input_metadata


    # TODO: try Tensor.to(non_blocking=True) for high layers
    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        block_tables: Optional[torch.Tensor],
        kv_len_tables: Optional[torch.Tensor],
        sparsity_tables: Optional[torch.Tensor],
        compress_config_tables: Optional[torch.Tensor],
        key_vec_size: int,
        val_vec_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        assert len(seq_group_metadata_list) > 0
        # get max context lens and batch size
        input_tokens: List[int] = []      # batch_id -> token
        input_positions: List[int] = []   # batch_id -> token index
        # NOTE: slot_mapping should be derived in the attn cuda kernel

        # NOTE: we still need input_positions as the full sequence length
        # can't be derived from kv_len_tables
        slot_ids = []
        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1, 'Beam search not supported yet'
            slot_ids.extend(seq_group_metadata.slot_ids)

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append(position)

        # take max over the batch & head dimensions
        # max_context_len = torch.amax(
        #     kv_len_tables[slot_ids], dim=(0, 2))
        max_context_len = torch.amax(
            torch.sum(kv_len_tables[slot_ids], dim=3), dim=(0, 2))

        # TODO: support cuda graph later
        use_captured_graph = False
        # When using CUDA graph, we don't need to make the tensors on the GPU
        # because they will be eventually copied to the designated GPU buffer.
        device = "cpu" if use_captured_graph else "cuda"
        # pin_memory = use_captured_graph and not self.in_wsl
        # input_tokens = _make_tensor_with_pad(input_tokens,
        #                                      max_len=1,
        #                                      pad=0,
        #                                      dtype=torch.int,
        #                                      device=device,
        #                                      pin_memory=pin_memory)
        # input_positions = _make_tensor_with_pad(input_positions,
        #                                         max_len=1,
        #                                         pad=0,
        #                                         dtype=torch.int,
        #                                         device=device,
        #                                         pin_memory=pin_memory)

        # print(f'[DEBUG] _prepare_decode input_positions: {input_positions}')
        
        input_tokens = torch.tensor(input_tokens, dtype=torch.int, device=device)
        input_positions = torch.tensor(input_positions, dtype=torch.int, device=device)
        slot_ids = torch.tensor(slot_ids, dtype=torch.int, device=device)

        quant_config_high = (
            seq_group_metadata_list[0].num_bits_k_high,
            seq_group_metadata_list[0].num_bits_v_high)
        quant_config_low = (
            seq_group_metadata_list[0].num_bits_k_low,
            seq_group_metadata_list[0].num_bits_v_low)

        if self.cache_config is not None:
            num_tokens_per_block_high = self.cache_config.quantized_block_num_tokens[quant_config_high]
            num_tokens_per_block_low = self.cache_config.quantized_block_num_tokens[quant_config_low]
        else:
            # dummpy input for profiling run
            num_tokens_per_block_high = None
            num_tokens_per_block_low = None
        
        input_metadata = InputMetadata(
            slot_ids=slot_ids,
            prompt_lens=[],
            seq_start_loc=None,
            max_context_len=max_context_len,
            use_cuda_graph=use_captured_graph,
            block_size=self.block_size,
            block_tables=block_tables,
            kv_len_tables=kv_len_tables,
            sparsity_tables=sparsity_tables,
            attn_prune_thresh=seq_group_metadata_list[0].attn_prune_thresh,
            num_bits_k_high=seq_group_metadata_list[0].num_bits_k_high,
            num_bits_v_high=seq_group_metadata_list[0].num_bits_v_high,
            num_bits_k_low=seq_group_metadata_list[0].num_bits_k_low,
            num_bits_v_low=seq_group_metadata_list[0].num_bits_v_low,
            compress_config_tables=compress_config_tables,
            key_vec_size=key_vec_size,
            val_vec_size=val_vec_size,
            num_tokens_per_block_high=num_tokens_per_block_high,
            num_tokens_per_block_low=num_tokens_per_block_low,
        )
        return input_tokens, input_positions, input_metadata

    # TODO (yanqi): prepare the ground truth token id here
    # and check how the vocabulary softmax should be accessed in
    # Sampler.forward (or _sample), what is the shape of logprobs?

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
    ) -> SamplingMetadata:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0
        truth_token_ids: List[int] = [] # ground truth token ids

        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1
                prompt_len = prompt_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += prompt_len - 1

                categorized_sample_indices[
                    sampling_params.sampling_type].append(
                        categorized_sample_indices_start_idx)
                categorized_sample_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + prompt_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              prompt_len - 1)
                selected_token_start_idx += prompt_len

                if sampling_params.model_seq:
                    assert sampling_params.truth_token_ids is not None
                    truth_token_ids.append(
                        sampling_params.truth_token_ids[prompt_len])
                elif sampling_params.emulate_seq:
                    assert sampling_params.truth_token_ids is not None
                    if len(sampling_params.truth_token_ids) > 0:
                        truth_token_ids.append(
                            sampling_params.truth_token_ids[prompt_len])
                    else:
                        print('Warning: _prepare_sample, emulate_seq with empty truth_token_ids')
                        truth_token_ids.append(None)
            else:
                num_seqs = len(seq_ids)
                selected_token_indices.extend(
                    range(selected_token_start_idx,
                          selected_token_start_idx + num_seqs))
                selected_token_start_idx += num_seqs

                categorized_sample_indices[
                    sampling_params.sampling_type].extend(
                        range(categorized_sample_indices_start_idx,
                              categorized_sample_indices_start_idx + num_seqs))
                categorized_sample_indices_start_idx += num_seqs

                if sampling_params.model_seq:
                    assert num_seqs == 1
                    assert sampling_params.truth_token_ids is not None
                    token_index = seq_group_metadata.seq_data[seq_ids[0]].get_len()
                    truth_token_ids.append(sampling_params.truth_token_ids[token_index])
                elif sampling_params.emulate_seq:
                    assert num_seqs == 1
                    assert sampling_params.truth_token_ids is not None
                    token_index = seq_group_metadata.seq_data[seq_ids[0]].get_len()
                    # print(f'token_idx = {token_index}, true prompt len = {len(sampling_params.truth_token_ids)}')
                    if token_index < len(sampling_params.truth_token_ids):
                        # the sequence is actually still processing prompt
                        truth_token_ids.append(sampling_params.truth_token_ids[token_index])
                    else:
                        # the prompt has finished
                        truth_token_ids.append(None)

        selected_token_indices = _async_h2d(selected_token_indices,
                                            dtype=torch.long,
                                            pin_memory=not self.in_wsl)
        categorized_sample_indices = {
            t: _async_h2d(seq_ids, dtype=torch.int, pin_memory=not self.in_wsl)
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            truth_token_ids=truth_token_ids,
        )
        return sampling_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: KVCache,
        block_tables: Optional[torch.Tensor],
        kv_len_tables: Optional[torch.Tensor],
        sparsity_tables: Optional[torch.Tensor],
        compress_config_tables: Optional[torch.Tensor],
        key_vec_size: Optional[int],
        val_vec_size: Optional[int],
    ) -> SamplerOutput:
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.

        t0 = time.time()

        if is_prompt:
            inputs = self._prepare_prompt(
                seq_group_metadata_list,
                block_tables,
                kv_len_tables,
                sparsity_tables,
                compress_config_tables,
                key_vec_size=key_vec_size,
                val_vec_size=val_vec_size)
            # print(f'_prepare_prompt = {(time.time() - t0) * 1000} ms')
            input_tokens, input_positions, input_metadata = inputs
        else:
            inputs = self._prepare_decode(
                seq_group_metadata_list,
                block_tables,
                kv_len_tables,
                sparsity_tables,
                compress_config_tables,
                key_vec_size=key_vec_size,
                val_vec_size=val_vec_size)
            # print(f'_prepare_decode = {(time.time() - t0) * 1000} ms')
            input_tokens, input_positions, input_metadata = inputs

        t1 = time.time()

        # Execute the model.
        if input_metadata.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model
        hidden_states = model_executable(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
        )

        t2 = time.time()

        sampling_metadata = self._prepare_sample(seq_group_metadata_list,
                                                 input_metadata.prompt_lens)
        # Sample the next token.
        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
        )

        t3 = time.time()
        # print(f'is_prompt: {is_prompt}, _prepare_X: {(1000 * (t1 - t0)):.2f} ms, '
        #       f'execute_model: {(1000 * (t2 - t1)):.2f} ms, sample: {(1000 * (t3 - t2)):.2f} ms')

        return output

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model_config.get_vocab_size()
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            # FIXME: temporary workaround for fitting larger batch size
            # seq_len = max(seq_len // 4, 1)
            seq_len = max(seq_len, 1)

            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                attn_prune_thresh=0.0,
                # dummpy inputs
                num_bits_k_high=None,
                num_bits_v_high=None,
                num_bits_k_low=None,
                num_bits_v_low=None,
            )
            seqs.append(seq)

        # key, value, score & index caches
        # all layers & heads unified to one cache
        kv_caches = None
        # dummpy inputs: key_vec_size & val_vec_size set to None
        self.execute_model(seqs, kv_caches, 
                           None, None, None, None, None, None)
        torch.cuda.synchronize()
        return

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[KVCache]) -> None:
        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode.")
        start_time = time.perf_counter()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        input_tokens = torch.zeros(max_batch_size, 1, dtype=torch.int).cuda()
        input_positions = torch.zeros(max_batch_size, 1,
                                      dtype=torch.int).cuda()
        slot_mapping = torch.empty(max_batch_size, 1, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()

        # NOTE: Capturing the largest batch size first may help reduce the
        # memory usage of CUDA graph.
        for batch_size in reversed(_BATCH_SIZES_TO_CAPTURE):
            # Create dummy input_metadata.
            input_metadata = InputMetadata(
                prompt_lens=[],
                slot_mapping=slot_mapping[:batch_size],
                max_context_len=self.max_context_len_to_capture,
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
                use_cuda_graph=True,
            )

            graph_runner = CUDAGraphRunner(self.model)
            graph_runner.capture(
                input_tokens[:batch_size],
                input_positions[:batch_size],
                kv_caches,
                input_metadata,
                memory_pool=self.graph_memory_pool,
            )
            self.graph_memory_pool = graph_runner.graph.pool()
            self.graph_runners[batch_size] = graph_runner

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info(f"Graph capturing finished in {elapsed_time:.0f} secs.")


class CUDAGraphRunner:

    def __init__(self, model: nn.Module):
        self.model = model
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        memory_pool,
    ) -> None:
        assert self.graph is None
        # Run the model once without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        self.model(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
        )
        torch.cuda.synchronize()

        # Capture the graph.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=memory_pool):
            hidden_states = self.model(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
            )
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": input_metadata.slot_mapping,
            "context_lens": input_metadata.context_lens,
            "block_tables": input_metadata.block_tables,
        }
        self.output_buffers = {"hidden_states": hidden_states}
        return

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(input_metadata.slot_mapping,
                                                 non_blocking=True)
        self.input_buffers["context_lens"].copy_(input_metadata.context_lens,
                                                 non_blocking=True)
        self.input_buffers["block_tables"].copy_(input_metadata.block_tables,
                                                 non_blocking=True)

        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["hidden_states"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return x + [pad] * (max_len - len(x))

def _pad_ndarray_to_max(
    x: np.ndarray,
    max_len: int,
    pad: int,
) -> np.ndarray:
    ''' Pad array along the last dimension
    '''
    assert x.shape[-1] <= max_len
    pad_dims = [(0, 0)] * (len(x.shape) - 1) + [(0, max_len - x.shape[-1])]
    return np.pad(x, pad_dims, constant_values=pad)

def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Union[str, torch.device] = "cuda",
    pin_memory: bool = False,
) -> torch.Tensor:
    padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x,
                        dtype=dtype,
                        device=device,
                        pin_memory=pin_memory and str(device) == "cpu")


def _get_graph_batch_size(batch_size: int) -> int:
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return (batch_size + 7) // 8 * 8


def _async_h2d(data: list, dtype, pin_memory):
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory)
    return t.to(device="cuda", non_blocking=True)
