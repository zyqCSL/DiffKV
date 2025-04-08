# SPDX-License-Identifier: Apache-2.0

import importlib.util
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

# from vllm._C import ops
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase,
                                               LinearMethodBase,
                                               UnquantizedLinearMethod,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, all_close_1d,
    cutlass_fp8_supported, maybe_create_device_identity,
    per_tensor_dequantize)

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    apply_w8a8_block_fp8_linear,
)

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)

has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning("Detected fp8 checkpoint. Please note that the "
                           "format is experimental and subject to change.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []

        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now.")
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions")
            if activation_scheme != "dynamic":
                raise ValueError("The block-wise quantization only supports "
                                 "dynamic activation scheme for now, but got "
                                 f"{activation_scheme} activation scheme.")
        else:
            raise ValueError('Only block-wise fp8 quantization is supported for now')
        self.weight_block_size = weight_block_size

    def get_name(self) -> str:
        return "fp8"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @staticmethod
    def get_min_capability() -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = ("fp8" in quant_method)
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"],
                                                 None)
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
                   activation_scheme=activation_scheme,
                   ignored_layers=ignored_layers,
                   weight_block_size=weight_block_size)

    def get_scaled_act_names(self) -> List[str]:
        return []
    
    # def get_linear_method(self) -> LinearMethodBase:
    #     return Fp8LinearMethod(self)
    
    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Any:

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix=prefix,
                                ignored_layers=self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self)
        return None


def is_layer_skipped(
    prefix: str,
    ignored_layers: List[str],
    fused_mapping: Dict[str, List[str]] = {}
) -> bool:
    # prefix: model.layers.0.self_attn.q_proj
    # proj_name: q_proj
    proj_name = prefix.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = shard_prefix in ignored_layers

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision.")
    else:
        is_skipped = prefix in ignored_layers

    assert is_skipped is not None
    return is_skipped


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()

        self.block_quant = self.quant_config.weight_block_size is not None
        assert self.block_quant is not None
        assert self.quant_config.activation_scheme == "dynamic"
        self.fp8_linear = Fp8LinearOp(
            # Default to using per_token quantization if cutlass is supported
            use_per_token_if_dynamic=cutlass_fp8_supported())

    def create_weights(
        self,
        # layer: torch.nn.Module,
        input_size_per_partition: int,
        output_size_per_partition: int,
        # output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        # **extra_weight_attrs,
    ) -> Dict[str, Any]:
        maybe_create_device_identity()

        tp_size = get_tensor_model_parallel_world_size()
        assert self.quant_config.weight_block_size is not None
        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )
        # Required by row parallel
        if (tp_size > 1
                and input_size // input_size_per_partition == tp_size
                and input_size_per_partition % block_k != 0):
            raise ValueError(
                f"Weight input_size_per_partition = "
                f"{input_size_per_partition} is not divisible by "
                f"weight quantization block_k = {block_k}.")
        # Required by column parallel or enabling merged weights
        if (tp_size > 1
                and output_size // output_size_per_partition == tp_size
                and output_size_per_partition % block_n != 0):
            raise ValueError(
                f"Weight input_size_per_partition = "
                f"{output_size_per_partition} is not divisible by "
                f"weight quantization block_n = {block_n}.")

        # WEIGHT
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)

        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=weight_dtype
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
            }
        )

        all_weights = {
            "weight": weight,
        }

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            assert self.quant_config.activation_scheme == "dynamic"
            scale = Parameter(
                torch.empty(
                    (output_size_per_partition + block_n - 1) // block_n,
                    (input_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            scale[:] = torch.finfo(torch.float32).min
            set_weight_attrs(
                scale, {
                    "input_dim": 1,
                    "output_dim": 0,
                    "scale_type": "weight_scale",
                }
            )

            all_weights["weight_scale_inv"] = scale

        return all_weights


    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert self.quant_config.weight_block_size is not None
        return apply_w8a8_block_fp8_linear(
            input=x,
            weight=weights['weight'],
            block_size=self.quant_config.weight_block_size,
            weight_scale=weights['weight_scale_inv'],
            input_scale=weights.get('input_scale', None),
            bias=bias,
        )


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None

        assert self.block_quant is not None
        assert self.quant_config.activation_scheme == "dynamic"

        # Check for DeepGemm support.
        self.allow_deep_gemm = False

    def create_weights(
        self,
        # layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        extra_weight_attrs: Dict[str, Any],
    ) -> Dict[str, Any]:

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        assert self.quant_config.weight_block_size is not None
        tp_size = get_tensor_model_parallel_world_size()
        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )
        # NOTE: To ensure proper alignment of the block-wise quantization
        # scales, the output_size of the weights for both the gate and up
        # layers must be divisible by block_n.
        # Required by column parallel or enabling merged weights
        if intermediate_size_per_partition % block_n != 0:
            raise ValueError(
                f"The output_size of gate's and up's weight = "
                f"{intermediate_size_per_partition} is not divisible by "
                f"weight quantization block_n = {block_n}.")
        if (tp_size > 1
                and intermediate_size_per_partition % block_k != 0):
            # Required by row parallel
            raise ValueError(
                f"The input_size of down's weight = "
                f"{intermediate_size_per_partition} is not divisible by "
                f"weight quantization block_k = {block_k}.")

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype),
            requires_grad=False)
        # layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype),
            requires_grad=False)
        # layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) //
                     block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
        # extra_weight_attrs.update(
        #     {"quant_method": FusedMoeWeightScaleSupported.BLOCK.
        #      value} if self.block_quant else
        #     {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})

        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        return {
            "w13_weight": w13_weight,
            "w2_weight": w2_weight,
            "w13_weight_scale_inv": w13_weight_scale,
            "w2_weight_scale_inv": w2_weight_scale,
        }


    def apply_weights(
        self,
        # layer: torch.nn.Module,
        weights: Dict[str, torch.Tensor],
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )


        return fused_experts(
            x,
            # layer.w13_weight,
            # layer.w2_weight,
            w1=weights['w13_weight'],
            w2=weights['w2_weight'],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            use_fp8_w8a8=True,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            # w1_scale=(layer.w13_weight_scale_inv
            #           if self.block_quant else layer.w13_weight_scale),
            # w2_scale=(layer.w2_weight_scale_inv
            #           if self.block_quant else layer.w2_weight_scale),
            w1_scale=weights["w13_weight_scale_inv"],
            w2_scale=weights["w2_weight_scale_inv"],
            # a1_scale=layer.w13_input_scale,
            # a2_scale=layer.w2_input_scale,
            block_shape=self.quant_config.weight_block_size,
            # allow_deep_gemm=self.allow_deep_gemm,
        )
