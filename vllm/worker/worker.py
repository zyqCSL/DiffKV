"""A GPU worker class."""
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.core.block_manager import AllocStatus, BlockSpaceManager
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.cache_config = None    # to be set later
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.worker_id = None   # id in orchestrator's view
        
        assert self.model_config.enforce_eager, "Only eager mode is supported now"
        self.model_runner = ModelRunner(model_config, parallel_config,
                                        scheduler_config)
        
        # Uninitialized cache engine & block_manager. Will be initialized by
        # self.init_cache_engine().
        # block_manager keeps the block_table & kv_lens, and tracks available blocks
        # cache_engine creates the cache, and performs cache-related ops like swap-in & out.
        # block tables need to be per worker, due to the heterogeneous patterns across heads & layers
        self.block_manager = None
        self.cache_config = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None
  
        # memory config, set after profile_num_available_blocks
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None

    def set_worker_id(self, worker_id: int) -> None:
        self.worker_id = worker_id
        print(f'** WORKER_ID = {self.worker_id} **')
    
    def init_model(self) -> None:
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Env vars will be set by Ray.
        self.rank = self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)

        _check_if_gpu_supports_dtype(self.model_config.dtype)

        # Initialize the distributed environment.
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.distributed_init_method)

        # Initialize the model.
        set_random_seed(self.model_config.seed)

    def load_model(
        self,
        kv_buffer_size: Union[int, List[int]],
        max_kv_slots: Optional[int] = None,
    ) -> None:
        self.model_runner.load_model(kv_buffer_size, max_kv_slots)
    
    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_block_bytes: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = total_gpu_memory - free_gpu_memory
        
        # Calculate the gpu-native memory management overhead
        max_num_seqs = self.scheduler_config.max_num_seqs
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        
        # estimated block_size
        _min_block_size = 8
        # block_tables + kv_len_tables + block_num_tables, int32
        metadata_memory = (num_layers * num_kv_heads * self.model_config.max_model_len // _min_block_size +  
                           num_layers * num_kv_heads +  
                           num_layers * num_kv_heads) * max_num_seqs * 4
        peak_memory += metadata_memory
        
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_bytes)
        num_cpu_blocks = int(cpu_swap_space // cache_block_bytes)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        # used in init_cache_engine
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.model_runner.cache_config = cache_config
        # use per-device info in priority, otherwise use the minimum
        if not self.num_gpu_blocks:
            self.num_gpu_blocks = self.cache_config.num_gpu_blocks[self.worker_id]
        else:
            assert self.num_gpu_blocks == self.cache_config.num_gpu_blocks[self.worker_id]
        if not self.num_cpu_blocks:
            self.num_cpu_blocks = self.cache_config.num_cpu_blocks[self.worker_id]
        else:
            assert self.num_cpu_blocks == self.cache_config.num_cpu_blocks[self.worker_id]
            
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config, self.worker_id)
   
        self.block_manager = BlockSpaceManager(
            max_num_seqs=self.scheduler_config.max_num_seqs,
            block_bytes=self.cache_config.block_bytes,
            num_gpu_blocks=self.num_gpu_blocks,
            num_cpu_blocks=self.num_cpu_blocks,
            quantized_kv_bits=self.cache_config.quantized_kv_bits,
            quantized_block_num_tokens=self.cache_config.quantized_block_num_tokens,
            num_layers=self.model_config.get_num_layers(self.parallel_config),
            num_heads=self.model_config.get_num_heads(self.parallel_config),
            num_kv_heads=self.model_config.get_num_kv_heads(self.parallel_config),
            head_size=self.model_config.get_head_size(),
            max_model_len=self.model_config.max_model_len,
            sliding_window=self.cache_config.sliding_window,
            max_kv_slots=self.cache_config.max_kv_slots,
        )
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
    
    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[SamplerOutput, int, int]:
        is_prompt = seq_group_metadata_list[0].is_prompt
        
        # Set the block_table & kv_len of input sequences
        for seq_group_metadata in seq_group_metadata_list:
            self.block_manager.prepare_metadata(seq_group_metadata)
        
        # Issue cache operations.
        blocks_to_swap_in = self.block_manager.blocks_to_swap_in
        blocks_to_swap_out = self.block_manager.blocks_to_swap_out
        blocks_to_copy = self.block_manager.blocks_to_copy
        
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if cache_events is not None:
            for event in cache_events:
                event.wait()
        # clear finished memory ops
        self.block_manager.reset_pending_memory_ops()
        
        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            return {}

        output = self.model_runner.execute_model(
            seq_group_metadata_list, 
            self.gpu_cache,
            self.block_manager.block_tables,
            self.block_manager.kv_len_tables,
            self.block_manager.sparsity_tables,
            self.block_manager.compress_config_tables,
            self.cache_config.key_vec_size,
            self.cache_config.value_vec_size)
        
        # NOTE: We assume during decode, kv cache is only replaced or appended to,
        # so init_kv_len <= kv_len always holds during decode,
        # and we only need to free blocks for prompt
        
        seq_ids = []
        for seq_group_metadata in seq_group_metadata_list:
            assert len(seq_group_metadata.seq_data) == 1, \
                "Currently KV pruning doesn't support copy-on-write for shared prompts"
            seq_ids.extend(list(seq_group_metadata.seq_data.keys()))
        
        if is_prompt:
            metadata = seq_group_metadata_list[0]
            self.block_manager.free_prompt(
                seq_ids, 
                metadata.num_bits_k_high, 
                metadata.num_bits_v_high,
                metadata.num_bits_k_low,
                metadata.num_bits_v_low)
        
        # print(f'DEBUG: block_manager.start_block_pos = {self.block_manager.gpu_allocator.start_block_pos}, '
        #       f'end_block_pos = {self.block_manager.gpu_allocator.end_block_pos}')
        
        # return the available block numbers to host
        return (output, 
                self.block_manager.get_num_free_gpu_blocks(),
                self.block_manager.get_num_free_cpu_blocks())

    # memory management
    def allocate_seqs(
        self, 
        seq_ids: List[int], 
        num_prompt_tokens: int,
        kbits_high: int,
        vbits_high: int,
        kbits_low: int,
        vbits_low: int,
        compress_config: List[float],
    ) -> None:
        self.block_manager.allocate(
            batch_seq_ids=[seq_ids], 
            batch_num_prompt_tokens=[num_prompt_tokens],
            kbits_high=kbits_high,
            vbits_high=vbits_high,
            kbits_low=kbits_low,
            vbits_low=vbits_low,
            batch_compress_configs=[compress_config])
        
    def allocate_batch_seqs(
        self, 
        batch_seq_ids: List[List[int]], 
        batch_num_prompt_tokens: List[int],
        kbits_high: int,
        vbits_high: int,
        kbits_low: int,
        vbits_low: int,
        batch_compress_configs: List[List[float]],
    ) -> None:
        ''' Allocate the scheduled batch in one pass '''
        self.block_manager.allocate(
            batch_seq_ids, 
            batch_num_prompt_tokens,
            kbits_high,
            vbits_high,
            kbits_low,
            vbits_low, 
            batch_compress_configs)
    
    def append_slot_to_seqs(
        self, 
        seq_ids: List[int],
        kbits_high: int,
        vbits_high: int,
        kbits_low: int,
        vbits_low: int) -> bool:
        return self.block_manager.append_slot(
            seq_ids, kbits_high, vbits_high, kbits_low, vbits_low)
    
    def free_seq(self, seq_id: int, is_finished: bool) -> int:
        # TODO: update kv_len stats here
        return self.block_manager.free(seq_id, is_finished)
    
    def fork_seq(self, parent_seq_id: int, child_seq_id: int) -> None:
        self.block_manager.fork(parent_seq_id, child_seq_id)
    
    def can_swap_in_seqs(self, seq_ids: List[int]) -> bool:
        return self.block_manager.can_swap_in(seq_ids)
    
    def swap_in_seqs(self, seq_ids: List[int]) -> None:
        self.block_manager.swap_in(seq_ids)
    
    def can_swap_out_seqs(self, seq_ids: List[int]) -> bool:
        return self.block_manager.can_swap_out(seq_ids)

    def swap_out_seqs(self, seq_ids: List[int]) -> None:
        self.block_manager.swap_out(seq_ids)
        
    def log_stats(self, log_path: str) -> None:
        self.block_manager.log_stats(log_path, self.worker_id)
    
    def get_free_gpu_blocks(self) -> int:
        return self.block_manager.get_num_free_gpu_blocks()
        

def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}.")
