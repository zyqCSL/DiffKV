# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Optional, Union

import torch

# from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)

from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs


class MoeWNA16Config(QuantizationConfig):
    """Config class for MOE WNA16 (W8A16/W4A16) quantization."""

    def __init__(self, linear_quant_method: str, weight_bits: int,
                 group_size: int, has_zp: bool, lm_head_quantized: bool,
                 modules_to_not_convert: Optional[List[str]],
                 full_config: Dict[str, Any]) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.has_zp = has_zp
        self.bit8_pack_factor = 8 // self.weight_bits
        self.lm_head_quantized = lm_head_quantized
        self.linear_quant_method = linear_quant_method
        self.full_config = full_config
        self.use_marlin = False
        # Avoid circular import
        from vllm.model_executor.layers.quantization.awq import AWQConfig
        if self.linear_quant_method == "gptq":
            pass
        elif self.linear_quant_method == "awq":
            capability = torch.cuda.get_device_capability()
            capability = capability[0] * 10 + capability[1]
            awq_min_capability = AWQConfig.get_min_capability()
            if capability < awq_min_capability:
                raise ValueError(
                    "The quantization method moe_wna16 + awq is not supported "
                    "for the current GPU. "
                    f"Minimum capability: {awq_min_capability}. "
                    f"Current capability: {capability}.")
            self.use_marlin = False
        else:
            raise ValueError("moe_wna16 only support gptq and awq.")

        if modules_to_not_convert is None:
            self.modules_to_not_convert = []
        else:
            self.modules_to_not_convert = modules_to_not_convert

    def get_name(self) -> str:
        return "moe_wna16"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @staticmethod
    def get_min_capability() -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MoeWNA16Config":
        linear_quant_method = cls.get_from_keys(config, ["quant_method"])
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        if linear_quant_method == "gptq":
            has_zp = not cls.get_from_keys(config, ["sym"])
            modules_to_not_convert = []
        elif linear_quant_method == "awq":
            has_zp = cls.get_from_keys(config, ["zero_point"])
            modules_to_not_convert = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None)
        else:
            raise ValueError("moe_wna16 only support gptq and awq.")

        return cls(linear_quant_method, weight_bits, group_size, has_zp,
                   lm_head_quantized, modules_to_not_convert, config)

    def override_quantization_method(
            self, hf_quant_cfg, user_quant) -> Optional[str]:
        can_convert = self.is_moe_wna16_compatible(hf_quant_cfg)
        if can_convert and user_quant == "moe_wna16":
            return self.get_name()
        return None

    def is_moe_wna16_compatible(self, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        desc_act = quant_config.get("desc_act")

        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]

        # Avoid circular import
        from vllm.model_executor.layers.quantization.awq import AWQConfig
        awq_min_capability = AWQConfig.get_min_capability()

        gptq_compatible = quant_method == "gptq" and \
                not desc_act and num_bits in [4, 8]
        awq_compatible = quant_method == "awq" and num_bits == 4 and \
            capability >= awq_min_capability

        return gptq_compatible or awq_compatible

    # def get_linear_method(self) -> FusedMoEMethodBase:
    #     return MoeWNA16Method(self)
    
    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Any:
        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            # Avoid circular import
            from vllm.model_executor.layers.quantization.awq import AWQConfig
            from vllm.model_executor.layers.quantization.gptq import GPTQConfig
            if self.linear_quant_method == "gptq":
                return GPTQConfig.from_config(
                    self.full_config).get_quant_method(layer, prefix)
            elif self.linear_quant_method == "awq":
                return AWQConfig.from_config(
                    self.full_config).get_quant_method(layer, prefix)
            else:
                raise ValueError("moe_wna16 only support gptq and awq.")
        elif isinstance(layer, FusedMoE):
            return MoeWNA16Method(self)
        return None
    
    def get_scaled_act_names(self) -> List[str]:
        return []


def is_layer_skipped_quant(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class MoeWNA16Method(FusedMoEMethodBase):
    """Linear method for MOE WNA16 (W8A16/W4A16) quantization.

    Args:
        quant_config: The MOE WNA16 (W8A16/W4A16) quantization config.
    """

    def __init__(self, quant_config: MoeWNA16Config):
        self.quant_config = quant_config

    def create_weights(self, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype,
                       extra_weight_attrs: Dict[str, Any]) -> Dict[str, Any]:
        all_weights = {}
        bit8_pack_factor = self.quant_config.bit8_pack_factor
        group_size = self.quant_config.group_size
        group_size_div_factor = 1

        # make intermediate_size and hidden_size diviable by group_size
        # we reduce the group size to ensure that
        # and we would repeat the loaded_weight later
        while intermediate_size_per_partition % group_size or \
                hidden_size % group_size:
            group_size = group_size // 2
            group_size_div_factor *= 2
            assert group_size >= 32
        
        self.group_size = group_size
        self.group_size_div_factor = group_size_div_factor

        strategy = FusedMoeWeightScaleSupported.GROUP.value

        extra_weight_attrs.update({
            "quant_method": strategy,
            "is_transposed": False
        })

        assert 'weight_loader' in extra_weight_attrs
        weight_loader = extra_weight_attrs['weight_loader']
        # wrapped_weight_loader = MoeWNA16Method.get_weight_loader(
        #     layer, weight_loader)
        wrapped_weight_loader = MoeWNA16Method.get_weight_loader(
            self.quant_config,
            group_size_div_factor,
            weight_loader)
        extra_weight_attrs['weight_loader'] = wrapped_weight_loader

        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // bit8_pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False
        )
        # layer.register_parameter("w13_qweight", w13_qweight)
        all_weights["w13_qweight"] = w13_qweight
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // bit8_pack_factor,
                dtype=torch.uint8
            ),
            requires_grad=False
        )
        # layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        all_weights["w2_qweight"] = w2_qweight

        w13_scales = torch.nn.Parameter(torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // group_size,
            dtype=params_dtype),
                                        requires_grad=False)
        # layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        all_weights["w13_scales"] = w13_scales

        w2_scales = torch.nn.Parameter(torch.zeros(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // group_size,
            dtype=params_dtype),
                                       requires_grad=False)
        # layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        all_weights["w2_scales"] = w2_scales

        if self.quant_config.has_zp:
            w13_qzeros = torch.nn.Parameter(torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition // bit8_pack_factor,
                hidden_size // group_size,
                dtype=torch.uint8),
                                            requires_grad=False)
            # layer.register_parameter("w13_qzeros", w13_qzeros)
            set_weight_attrs(w13_qzeros, extra_weight_attrs)
            all_weights["w13_qzeros"] = w13_qzeros

            w2_qzeros = torch.nn.Parameter(torch.zeros(
                num_experts,
                hidden_size // bit8_pack_factor,
                intermediate_size_per_partition // group_size,
                dtype=torch.uint8),
                                           requires_grad=False)
            # layer.register_parameter("w2_qzeros", w2_qzeros)
            set_weight_attrs(w2_qzeros, extra_weight_attrs)
            all_weights["w2_qzeros"] = w2_qzeros

        if self.quant_config.linear_quant_method == "gptq":
            # some param are unused, but we need to init them in order to
            # load weights
            invalid_param_keys = ["w13_g_idx", "w2_g_idx"]
            if not self.quant_config.has_zp:
                invalid_param_keys += ["w13_qzeros", "w2_qzeros"]
            for key in invalid_param_keys:
                param = torch.nn.Parameter(torch.empty((0, ),
                                                       dtype=torch.int32),
                                           requires_grad=False)
                # layer.register_parameter(key, param)
                set_weight_attrs(param, extra_weight_attrs)
                all_weights[key] = param

        return all_weights

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
        assert activation == "silu", "Only SiLU activation is supported."
        assert self.group_size is not None

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
            e_score_correction_bias=e_score_correction_bias)

        weight_bits = self.quant_config.weight_bits
        has_zp = self.quant_config.has_zp

        return fused_experts(
            x,
            # layer.w13_qweight,
            # layer.w2_qweight,
            w1=weights['w13_qweight'],
            w2=weights['w2_qweight'],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_int4_w4a16=weight_bits == 4,
            use_int8_w8a16=weight_bits == 8,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            # w1_scale=layer.w13_scales,
            # w2_scale=layer.w2_scales,
            w1_scale=weights["w13_scales"],
            w2_scale=weights["w2_scales"],
            # w1_zp=layer.w13_qzeros if has_zp else None,
            # w2_zp=layer.w2_qzeros if has_zp else None,
            w1_zp=weights["w13_qzeros"] if has_zp else None,
            w2_zp=weights["w2_qzeros"] if has_zp else None,
            block_shape=[0, self.group_size])

    
    @staticmethod
    # def get_weight_loader(layer, weight_loader):
    def get_weight_loader(
        quant_config: MoeWNA16Config,
        group_size_div_factor: int,
        weight_loader):

        def convert_awq_tensor(tensor, tensor_type):
            # convert awq qweight/qzeros to a standard format (assume int4)
            # qweight: (k, n // pack_factor_bit32) -> (n, k // pack_factor_bit8)
            # qzeros: (k // group_size, n // pack_factor_bit32) ->
            #         (n // pack_factor_bit8, k // group_size)
            # pack_factor_bit32 = 32 // weight_bits
            # pack_factor_bit8 = 8 // weight_bits

            # 0. suppose origin shape (a, b), dtype int32
            # 1. convert to uint8, shape (a, b) -> (a, 4 * b)
            size0 = tensor.size(0)
            tensor = tensor.view(torch.uint8)

            # 2. unpack to uint4 (only when weight_bits == 4)
            #    shape (a, 4 * b) -> (a, 4 * b, 2)
            shifter = torch.tensor([0, 4],
                                   dtype=torch.uint8,
                                   device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF

            # 3. change order, see
            # https://github.com/casper-hansen/AutoAWQ/blob/v0.2.8/awq/utils/quant_utils.py
            # shape -> (a, 4 * b * pack_factor_bit8)
            reverse_awq_pack_order = [0, 4, 1, 5, 2, 6, 3, 7]
            tensor = tensor.view(-1, 8)[:, reverse_awq_pack_order]
            tensor = tensor.view(size0, -1)

            # 4. transpose, shape -> (4 * b * pack_factor_bit8, a)
            tensor = tensor.T.contiguous()

            # 5. repack (only when weight_bits == 4)
            # qweight shape -> (4 * b * pack_factor_bit8, a // pack_factor_bit8)
            # qzeros shape -> (4 * b, a)

            if tensor_type == "qweight":
                tensor = tensor[:, 1::2] * 16 + tensor[:, ::2]
            elif tensor_type == "qzeros":
                tensor = tensor[1::2, :] * 16 + tensor[::2, :]
            return tensor

        def convert_gptq_int4_qzeros(tensor):
            tensor = tensor.view(torch.uint8)
            shifter = torch.tensor([0, 4],
                                   dtype=torch.uint8,
                                   device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF
            tensor = tensor + 1
            tensor = tensor[:, :, 0] + tensor[:, :, 1] * 16
            return tensor

        def moe_wna16_weight_loader(param: torch.nn.Parameter,
                                    loaded_weight: torch.Tensor,
                                    weight_name: str, shard_id: str,
                                    expert_id: int):
            if "g_idx" in weight_name:
                return
            if not quant_config.has_zp and "qzeros" in weight_name:
                return

            loaded_weight = loaded_weight.to('cuda')

            # convert gptq and awq weight to a standard format
            if quant_config.linear_quant_method == "awq":
                assert quant_config.weight_bits == 4
                if "weight" in weight_name:
                    loaded_weight = convert_awq_tensor(loaded_weight,
                                                       "qweight")
                elif "zeros" in weight_name:
                    loaded_weight = convert_awq_tensor(loaded_weight, "qzeros")
                else:
                    loaded_weight = loaded_weight.T
            elif quant_config.linear_quant_method == "gptq":
                assert quant_config.weight_bits in [4, 8]
                if "weight" in weight_name:
                    loaded_weight = loaded_weight.T.contiguous().view(
                        torch.uint8)
                elif "zeros" in weight_name:
                    # add 1 to gptq qzeros to align with awq
                    loaded_weight = loaded_weight.view(torch.uint8)
                    if quant_config.weight_bits == 4:
                        loaded_weight = convert_gptq_int4_qzeros(
                            loaded_weight).T
                    else:
                        loaded_weight = loaded_weight.T + 1
                else:
                    loaded_weight = loaded_weight.T
            else:
                raise ValueError(f"Unsupported quantization method: {quant_config.linear_quant_method}")
            
            # repeat the qzeros/scales to fit new group size
            if group_size_div_factor > 1 and \
                    "qzeros" in weight_name or "scales" in weight_name:
                loaded_weight = loaded_weight.repeat_interleave(
                    group_size_div_factor, 1)
            
            weight_loader(param, loaded_weight, weight_name, shard_id,
                              expert_id)

        return moe_wna16_weight_loader
