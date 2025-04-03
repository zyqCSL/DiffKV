"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple

import torch

from vllm._C import cache_ops
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl

logger = init_logger(__name__)

# unified cache for key, value, scale, offset, score and index
KVCache = torch.Tensor

class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        worker_id: int,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.worker_id = worker_id

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)
        # PyTorch older than version 2.3 does not support uint16 data type.
        # We use int16 and cast it to uint16 in cuda kernels.
        # KV cache is initialized as an empty tensor anyway.
        self.dtype = torch.int16

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks[self.worker_id]
        self.num_cpu_blocks = cache_config.num_cpu_blocks[self.worker_id]

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    def allocate_gpu_cache(self) -> KVCache:
        # TODO: we might need another cache for ref_count to support beam search
        # unified cache for all layers
        # each block includes key, val, scale, offset, score and index
        kv_blocks = torch.empty(
            size=(self.num_gpu_blocks, self.block_size),
            dtype=self.dtype,
            device='cuda',
        )
        return kv_blocks

    def allocate_cpu_cache(self) -> KVCache:
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        # unified cache for all layers
        kv_blocks = torch.empty(
            size=(self.num_cpu_blocks, self.block_size),
            dtype=self.dtype,
            pin_memory=pin_memory,
        )
        return kv_blocks

    def _swap(
        self,
        src_cache: KVCache,
        dst_cache: KVCache,
        src_to_dst: List[Dict[int, int]],
    ) -> None:
        # record swap events of each layer
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                layer_src_to_dst = src_to_dst[i]
                # Copy the kv blocks
                cache_ops.swap_blocks(src_cache, dst_cache, layer_src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: List[Dict[int, int]]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: List[Dict[int, int]]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        # TODO: modify the copy_blocks kernel to match current mem layout
        raise NotImplementedError
        # key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        # value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        # cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)
