import torch
import torch.distributed

from typing import Optional

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    is_pipeline_model_parallel_first_rank,
    is_pipeline_model_parallel_last_rank,
)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation is applied in-place on the input tensor.
    """
    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    # All-reduce.
    torch.distributed.all_reduce(input_,
                                 group=get_tensor_model_parallel_group())
    return input_


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size, ) + input_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=get_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor


def tensor_model_parallel_reduce_scatter(input_: torch.Tensor,
                                         dim: int = -1) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()

    # Note: This will produce an incorrect answer if we don't make
    # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
    input_tensor = input_.movedim(0, dim).contiguous()
    assert input_tensor.shape[0] % world_size == 0
    output_shape = (input_tensor.shape[0] // world_size, ) + input_tensor.shape[1:]

    # Allocate output tensor.
    output_tensor = torch.empty(output_shape,
                                dtype=input_.dtype,
                                device=input_.device)
    # Reduce-scatter.
    torch.distributed.reduce_scatter(output_tensor,
                                     input_tensor,
                                     group=get_tensor_model_parallel_group())

    # Reshape before returning
    return output_tensor.movedim(0, dim).contiguous()


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """
    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    NOTE: `dst` is the local rank of the destination rank.
    """
    world_size = get_tensor_model_parallel_world_size()
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    assert dst < world_size, (
        f"Invalid dst ({dst}) for world size {world_size}")

    rank_in_group = get_tensor_model_parallel_rank()
    # Allocate output tensor.
    if rank_in_group == dst:
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        gather_list = None
    # Gather.
    tp_group = get_tensor_model_parallel_group()
    dst_rank = torch.distributed.get_process_group_ranks(tp_group)[dst]
    torch.distributed.gather(input_,
                             gather_list,
                             dst=dst_rank,
                             group=tp_group)
    if rank_in_group == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor


# TODO: add PP logic in model_executor
def tensor_pipeline_parallel_send(tensor: torch.Tensor) -> None:
    """Sends a tensor to the next rank in a non-blocking way"""
    """NOTE: `dst` is the local rank of the destination rank."""
    if is_pipeline_model_parallel_last_rank():
        return

    torch.distributed.send(tensor,
                           dst=get_pipeline_model_parallel_next_rank(),
                           group=get_pipeline_model_parallel_group())


def tensor_pipeline_parallel_recv(size: torch.Size,
                                  dtype: torch.dtype,
                                  device: torch.device) -> Optional[torch.Tensor]:
    """Receives a tensor from the previous rank."""
    """NOTE: `src` is the local rank of the source rank."""
    if is_pipeline_model_parallel_first_rank():
        return None

    tensor = torch.empty(size, dtype=dtype, device=device)
    torch.distributed.recv(tensor,
                           src=get_pipeline_model_parallel_prev_rank(),
                           group=get_pipeline_model_parallel_group())
    return tensor
