from typing import Optional, Union, List, Dict, Tuple
import os

import torch
from transformers import PretrainedConfig

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_config
from vllm.utils import get_cpu_memory, is_hip

logger = init_logger(__name__)

_GB = 1 << 30


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        download_dir: Directory to download and load the weights, default to the
            default cache directory of huggingface.
        load_format: The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format.
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        download_dir: Optional[str],
        load_format: str,
        dtype: Union[str, torch.dtype],
        seed: int,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.download_dir = download_dir
        self.load_format = load_format
        self.seed = seed
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.enforce_eager = enforce_eager
        self.max_context_len_to_capture = max_context_len_to_capture

        if os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true":
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            from modelscope.hub.snapshot_download import snapshot_download  # pylint: disable=C
            model_path = snapshot_download(model_id=model,
                                           cache_dir=download_dir,
                                           revision=revision)
            self.model = model_path
            self.download_dir = model_path
            self.tokenizer = model_path

        self.hf_config = get_config(self.model, trust_remote_code, revision)
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.max_model_len = _get_and_verify_max_len(self.hf_config,
                                                     max_model_len)
        self._verify_load_format()
        self._verify_tokenizer_mode()
        self._verify_quantization()
        self._verify_cuda_graph()

    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        supported_load_format = [
            "auto", "pt", "safetensors", "npcache", "dummy"
        ]
        rocm_not_supported_load_format = []
        if load_format not in supported_load_format:
            raise ValueError(
                f"Unknown load format: {self.load_format}. Must be one of "
                "'auto', 'pt', 'safetensors', 'npcache', or 'dummy'.")
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f for f in supported_load_format
                if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(
                f"load format \'{load_format}\' is not supported in ROCm. "
                f"Supported load format are "
                f"{rocm_supported_load_format}")

        # TODO: Remove this check once HF updates the pt weights of Mixtral.
        architectures = getattr(self.hf_config, "architectures", [])
        if "MixtralForCausalLM" in architectures and load_format == "pt":
            raise ValueError(
                "Currently, the 'pt' format is not supported for Mixtral. "
                "Please use the 'safetensors' format instead. ")
        self.load_format = load_format

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto' or 'slow'.")
        self.tokenizer_mode = tokenizer_mode

    def _verify_quantization(self) -> None:
        # supported_quantization = ["awq", "gptq", "squeezellm"]
        supported_quantization = ["awq", "gptq", "fp8"]
        rocm_not_supported_quantization = ["awq"]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        hf_quant_config = getattr(self.hf_config, "quantization_config", None)
        if hf_quant_config is not None:
            hf_quant_method = str(hf_quant_config["quant_method"]).lower()
            if self.quantization is None:
                self.quantization = hf_quant_method
            elif self.quantization != hf_quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({hf_quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization}).")

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}.")
            if is_hip(
            ) and self.quantization in rocm_not_supported_quantization:
                raise ValueError(
                    f"{self.quantization} quantization is currently not supported "
                    f"in ROCm.")
            logger.warning(f"{self.quantization} quantization is not fully "
                           "optimized yet. The speed can be slower than "
                           "non-quantized models.")

    def _verify_cuda_graph(self) -> None:
        if self.max_context_len_to_capture is None:
            self.max_context_len_to_capture = self.max_model_len
        self.max_context_len_to_capture = min(self.max_context_len_to_capture,
                                              self.max_model_len)
        if (self.quantization in ["gptq", "squeezellm"]
                and not self.enforce_eager):
            # Related issue: https://github.com/vllm-project/vllm/issues/2147
            logger.warning(f"{self.quantization} does not support CUDA graph "
                           "yet. Disabling CUDA graph.")
            self.enforce_eager = True

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size}).")

        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({total_num_hidden_layers}) "
                "must be divisible by pipeline parallel size "
                f"({pipeline_parallel_size}).")

    def get_sliding_window(self) -> Optional[int]:
        return getattr(self.hf_config, "sliding_window", None)

    def get_vocab_size(self) -> int:
        return self.hf_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        # NOTE: Some configs may set head_dim=None in the config
        if getattr(self.hf_config, "head_dim", None) is not None:
            return self.hf_config.head_dim
        
        # FIXME(woosuk): This may not be true for all models.
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False))
        if not new_decoder_arch_falcon and getattr(self.hf_config,
                                                   "multi_query", False):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """Returns the number of KV heads per GPU."""
        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1,
                   total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_heads(self, parallel_config: "ParallelConfig") -> int:
        """Returns the number of attention heads per GPU."""
        # If tensor parallelism is used, we divide the number of attention heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1,
                   self.hf_config.num_attention_heads // parallel_config.tensor_parallel_size)

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size


class CacheConfig:
    """Configuration for the KV cache.

    Args:
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
        kv_buffer_size: number of most recent tokens that are never pruned
        kv_score_prune_thresh: fraction of total softmax score to keep, in [0, 1] and 1 for no pruning
        kv_score_quant_thresh: fraction of total softmax score to quantize, in [0, 1] and 0 for no quantization
        max_kv_slots: maximum number of entries in the KV cache
        key_vec_size: number of 16-bit vecs in a KV cache page's key part
        value_vec_size: number of 16-bit vecs in a KV cache page's value part
        memory_align_bytes: align each part of cache block to #memory_align_bytes boundary, by default 32
    """

    def __init__(
        self,
        gpu_memory_utilization: float,
        swap_space: int,
        kv_buffer_size: Union[int, List[int]],
        max_kv_slots: Optional[int] = None,
        sliding_window: Optional[int] = None,
        key_vec_size: int = 4,
        value_vec_size: int = 2,
        # value_vec_size: int = 4,
        num_thread_groups_k: int = 8,
        memory_align_bytes: int = 32,
        memory_align_bytes_kv: int = 128,
    ) -> None:
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GB
        self.sliding_window = sliding_window
        self._verify_args()

        # kv cache config
        self.memory_align_bytes = memory_align_bytes
        self.memory_align_bytes_kv = memory_align_bytes_kv
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
        self.num_thread_groups_k = num_thread_groups_k
        self.vec_bytes = 2 # each vec is 16 bits long
        # kv cache quantization (kbits, vbits, kmeta_num, vmeta_num)
        self.quantized_kv_bits = [
            (8, 8, 1, 1),
            (8, 4, 1, 2),
            (8, 4, 1, 1),
            (8, 2, 1, 1),
            (4, 4, 1, 1),
            (4, 2, 2, 4),
            (4, 2, 1, 1),
            (4, 1, 1, 1),
        ]
        # hetero kvs are used to compute page size
        self.hetero_kv_bits = [
            (8, 4, 1, 2),
            (8, 4, 1, 1),
            (4, 2, 2, 4),
            (4, 2, 1, 1),
        ]

        assert set(self.hetero_kv_bits).issubset(
            set(self.quantized_kv_bits))

        self.kv_buffer_size = kv_buffer_size
        self.max_kv_slots = max_kv_slots

        # Will be set after profiling.
        self._quantized_kv_head_bytes = {}
        self.quantized_block_num_tokens = {}
        self.block_bytes = None
        self.block_size = None    # number of 16-bit (model.dtype) elements
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None


    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (f"{cpu_memory_usage / _GB:.2f} GiB out of "
               f"the {total_cpu_memory / _GB:.2f} GiB total CPU memory is "
               "allocated for the swap space.")
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. " + msg)


    # utility functions for computing cache block size
    def _compute_quantized_kv_head_bytes(self, head_size: int) -> None:
        # key & val, scale & offset for both key & val ((2 + 2) * 2), score & index (4 bytes)
        assert head_size % 8 == 0
        for kbits, vbits, kmeta, vmeta in self.quantized_kv_bits:
            assert (kbits, vbits, kmeta, vmeta) not in self._quantized_kv_head_bytes
            self._quantized_kv_head_bytes[(kbits, vbits, kmeta, vmeta)] = [
                kbits * head_size // 8,   # kbytes
                vbits * head_size // 8,   # vbytes
                4 * kmeta,  # scaling & zero_point, both in FP16 -> 4bytes
                4 * vmeta,  
                8,  # index & score, in int32 & FP32
            ]

    def _block_bytes_from_tokens(
        self,
        num_tokens: int,
        head_size: int,
        kbits: int,
        vbits: int,
        kmeta: int,
        vmeta: int,
    ) -> int:
        # each pack is 16 bits (INT16)
        assert 16 % kbits == 0 and 16 % vbits == 0, (kbits, vbits)
        k_pack_size = 16 // kbits
        v_pack_size = 16 // vbits

        k_num_packs = head_size // k_pack_size
        v_num_packs = head_size // v_pack_size

        # NOTE: key shape = [NUM_PACKS/VEC_SIZE, NUM_TOKENS, VEC_SIZE]
        # make sure that keys do not need padding in the 1st dimension
        assert k_num_packs % self.key_vec_size == 0

        # effective data size
        k_bytes, v_bytes, k_quant_bytes, v_quant_bytes, meta_bytes = \
            self._quantized_kv_head_bytes[(kbits, vbits, kmeta, vmeta)]

        # data (padded) + quant_meta (padded)
        sum_kbytes = _divide_round_up(num_tokens * k_bytes, self.memory_align_bytes) * self.memory_align_bytes + \
                     _divide_round_up(num_tokens * k_quant_bytes, self.memory_align_bytes) * self.memory_align_bytes
        sum_kbytes = _divide_round_up(sum_kbytes, self.memory_align_bytes_kv) * self.memory_align_bytes_kv

        # NOTE: val shape [NUM_TOKENS/VEC_SIZE, NUM_PACKS, VEC_SIZE]
        padded_v_vecs = _divide_round_up(num_tokens, self.value_vec_size)
        padded_v_vec_bytes = padded_v_vecs * v_num_packs * self.value_vec_size * self.vec_bytes

        # data (padded) + quant_meta (padded)
        sum_vbytes = _divide_round_up(padded_v_vec_bytes, self.memory_align_bytes) * self.memory_align_bytes + \
                     _divide_round_up(num_tokens * v_quant_bytes, self.memory_align_bytes) * self.memory_align_bytes

        sum_meta_bytes = _divide_round_up(
            num_tokens * meta_bytes, self.memory_align_bytes) * self.memory_align_bytes

        sum_bytes = sum_kbytes + sum_vbytes + sum_meta_bytes
        sum_bytes = _divide_round_up(sum_bytes, self.memory_align_bytes_kv) * self.memory_align_bytes_kv

        assert sum_bytes % self.memory_align_bytes_kv == 0
        return sum_bytes

    def _block_tokens_from_bytes(
        self,
        block_bytes: int,
        head_size: int,
        kbits: int,
        vbits: int,
        kmeta: int,
        vmeta: int,
    ) -> int:
        assert block_bytes % self.memory_align_bytes == 0
        k_bytes, v_bytes, k_quant_bytes, v_quant_bytes, meta_bytes = \
            self._quantized_kv_head_bytes[(kbits, vbits, kmeta, vmeta)]

        num_tokens = block_bytes // (k_bytes + v_bytes +  k_quant_bytes + v_quant_bytes + meta_bytes)
        while num_tokens > 0:
            padded_block_bytes = self._block_bytes_from_tokens(num_tokens, head_size, 
                                                               kbits, vbits, kmeta, vmeta)
            # print(f'block_bytes = {block_bytes}, num_tokens = {num_tokens}, padded_block_bytes = {padded_block_bytes}')
            if padded_block_bytes <= block_bytes:
                break
            num_tokens -= 1
        assert num_tokens > 0, num_tokens
        return num_tokens

    def _get_residual_ratio(
        self,
        block_bytes: int,
        num_tokens: int,
        kbits: int,
        vbits: int,
        kmeta: int,
        vmeta: int,
    ) -> float:
        k_bytes, v_bytes, k_quant_bytes, v_quant_bytes, meta_bytes =\
            self._quantized_kv_head_bytes[(kbits, vbits, kmeta, vmeta)]
        total_bytes = num_tokens * (k_bytes + v_bytes + k_quant_bytes + v_quant_bytes + meta_bytes)
        return 1 - total_bytes / block_bytes

    def compute_cache_block_size(
        self,
        model_config: ModelConfig,
        min_block_bytes: int = 800,
        max_block_bytes: int = 8000,
    ) -> Tuple[int, int]:
        ''' Find the block size with the least fragmentation across all quantized configs
            return
                block_bytes, block_size (number of 16-bit elemets)
        '''
        
        print(f'[DEBUG] min_block_bytes = {min_block_bytes}, max_block_bytes = {max_block_bytes}')
        
        dtype_size = _get_dtype_size(model_config.dtype)
        assert dtype_size == 2 # FP16 or BF16
        head_size = model_config.get_head_size()
        self._compute_quantized_kv_head_bytes(head_size)
        
        # print(f'[DEBUG] head_size = {head_size}')

        # use heteo kv bits to compute the block size
        min_ratio = 500
        min_ratio_block_bytes = []
        for b in range(min_block_bytes // self.memory_align_bytes,
                       max_block_bytes // self.memory_align_bytes + 1):
            block_bytes = b * self.memory_align_bytes
            # print(f'** block_bytes = {_block_bytes}')
            res_ratio = 0
            # ignored = False
            for (kbits, vbits, kmeta, vmeta) in self.hetero_kv_bits:
                num_tokens = self._block_tokens_from_bytes(
                    block_bytes=block_bytes,
                    head_size=head_size,
                    kbits=kbits,
                    vbits=vbits,
                    kmeta=kmeta,
                    vmeta=vmeta,
                )
                # # if num_tokens % self.num_thread_groups_k < self.num_thread_groups_k // 2:
                # if num_tokens % self.num_thread_groups_k != 0:
                #     # print(f'{block_bytes} k{kbits}v{vbits} tokens={num_tokens} ignored due to excessive bubbles')
                #     # ignore this config due to excessive bubbles in attention Key processing
                #     ignored = True
                #     break
                r = self._get_residual_ratio(
                    block_bytes=block_bytes,
                    num_tokens=num_tokens,
                    kbits=kbits,
                    vbits=vbits,
                    kmeta=kmeta,
                    vmeta=vmeta,
                )
                # print(f'****** k{kbits}v{vbits}, num_tokens = {num_tokens}, res_ratio = {r}')
                res_ratio = max(r, res_ratio)
            # if ignored:
            #     # ignore this config due to excessive bubbles in attention Key processing
            #     continue
            if min_ratio > res_ratio:
                min_ratio = res_ratio
                min_ratio_block_bytes = [block_bytes]
            elif min_ratio == res_ratio:
                min_ratio_block_bytes.append(block_bytes)

        assert len(min_ratio_block_bytes) > 0

        # use the smallest feasible block size to avoid internal fragmentation
        opt_block_bytes = min(min_ratio_block_bytes)

        assert opt_block_bytes % dtype_size == 0
        opt_block_size = opt_block_bytes // dtype_size   # number of elements

        print(f'Optimal block size = {opt_block_bytes} bytes, '
              f'{opt_block_size} x {model_config.dtype}, '
              f'residual_ratio = {round(min_ratio * 100, 3)}%')

        self.block_bytes = opt_block_bytes
        self.block_size = opt_block_size

        # compute effective block size (number of elements in each block)
        for kbits, vbits, kmeta, vmeta in self.quantized_kv_bits:
            self.quantized_block_num_tokens[(kbits, vbits, kmeta, vmeta)] = self._block_tokens_from_bytes(
                block_bytes=self.block_bytes,
                head_size=head_size,
                kbits=kbits,
                vbits=vbits,
                kmeta=kmeta,
                vmeta=vmeta,
                )

        print(f'quantized tokens per block = {self.quantized_block_num_tokens}')
        return self.block_bytes, self.block_size

def _get_dtype_bits(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size() * 8

def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()

def _divide_round_up(x, y):
    return (x + y - 1) // y

class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        worker_use_ray: Whether to use Ray for model workers. Will be set to
            True if either pipeline_parallel_size or tensor_parallel_size is
            greater than 1.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        worker_use_ray: bool,
        enable_expert_parallel: bool = True,
        max_parallel_loading_workers: Optional[int] = None,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.worker_use_ray = worker_use_ray
        self.enable_expert_parallel = enable_expert_parallel
        self.max_parallel_loading_workers = max_parallel_loading_workers

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet.")


class SchedulerConfig:
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
        max_paddings: Maximum number of paddings to be added to a batch.
    """

    def __init__(
        self,
        max_num_batched_tokens: Optional[int],
        max_num_seqs: int,
        max_model_len: int,
        max_paddings: int,
    ) -> None:
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            # If max_model_len is too short, use 2048 as the default value for
            # higher throughput.
            self.max_num_batched_tokens = max(max_model_len, 2048)
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_paddings = max_paddings
        self._verify_args()

    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len.")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_ROCM_NOT_SUPPORTED_DTYPE = ["float", "float32"]


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: Union[str, torch.dtype],
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                # Following the common practice, we use float16 for float32
                # models.
                torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    if is_hip() and torch_dtype == torch.float32:
        rocm_supported_dtypes = [
            k for k, v in _STR_DTYPE_TO_TORCH_DTYPE.items()
            if (k not in _ROCM_NOT_SUPPORTED_DTYPE)
        ]
        raise ValueError(f"dtype \'{dtype}\' is not supported in ROCm. "
                         f"Supported dtypes are {rocm_supported_dtypes}")

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            pass
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning(f"Casting {config_dtype} to {torch_dtype}.")

    return torch_dtype


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    max_model_len: Optional[int],
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    for key in possible_keys:
        max_len_key = getattr(hf_config, key, None)
        if max_len_key is not None:
            derived_max_model_len = min(derived_max_model_len, max_len_key)
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            f"{possible_keys}. Assuming the model's maximum length is "
            f"{default_max_len}.")
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        # No need to consider "type" key because of patch_rope_scaling when
        # loading HF config
        rope_type = rope_scaling["rope_type"]

        if rope_type not in ("su", "longrope", "llama3"):
            # NOTE: rope_type == "default" does not define factor
            # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/modeling_rope_utils.py
            scaling_factor = rope_scaling.get("factor", 1.0)

            if rope_type == "yarn":
                derived_max_model_len = rope_scaling[
                    "original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    if max_model_len is None:
        max_model_len = derived_max_model_len
    elif max_model_len > derived_max_model_len:
        raise ValueError(
            f"User-specified max_model_len ({max_model_len}) is greater than "
            f"the derived max_model_len ({max_len_key}={derived_max_model_len}"
            " in model's config.json). This may lead to incorrect model "
            "outputs or CUDA errors. Make sure the value is correct and "
            "within the model context size.")
    return int(max_model_len)
