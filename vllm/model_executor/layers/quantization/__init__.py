from typing import Type, Optional, Tuple

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

_QUANTIZATION_CONFIG_REGISTRY = {
    "awq": AWQConfig,
    "gptq": GPTQConfig,
    "fp8": Fp8Config,
    "moe_wna16": MoeWNA16Config,
}

_MOE_ARCHITECTURES = [
    "Qwen3MoeForCausalLM",
    "MixtralForCausalLM",
]

def get_quantization_config(quantization: str, architecture: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_CONFIG_REGISTRY:
        raise ValueError(f"Invalid quantization method: {quantization}")
    if architecture in _MOE_ARCHITECTURES and quantization in ["awq", "gptq"]:
        quantization = "moe_wna16"
    return _QUANTIZATION_CONFIG_REGISTRY[quantization]

__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
]
