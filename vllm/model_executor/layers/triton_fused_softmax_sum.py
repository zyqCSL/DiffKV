import torch
import triton
import triton.language as tl


# Check if the input value is NaN.
@triton.jit
def isnan(x):
    return x != x


# Triton kernel for fused softmax, nan_to_zero, and sum computation
@triton.jit
def triton_fused_softmax_sum_kernel(qk_ptr, max_prompt_len, softmax_sum_ptr, NUM_TOKENS_SCORE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    softmax_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over NUM_TOKENS_SCORE sequences
    for i in tl.range(0, NUM_TOKENS_SCORE):
        # Load qk_products for the current sequence
        qk_products = tl.load(qk_ptr + offsets + (block_idx * NUM_TOKENS_SCORE + i) * max_prompt_len,
                              mask=offsets < max_prompt_len, other=-float('inf'))

        # Compute max for stability
        max_val = tl.max(qk_products, axis=0)
        qk_products -= max_val

        # Compute softmax
        exp_qk_products = tl.exp2(qk_products)
        exp_sum = tl.sum(exp_qk_products, axis=0)
        softmax_result = exp_qk_products / exp_sum

        # Replace NaNs with 0.0
        softmax_result = tl.where(isnan(softmax_result), 0.0, softmax_result)

        # Update the sum
        softmax_sum += softmax_result

    # Store the final result to softmax_sum_ptr
    tl.store(softmax_sum_ptr + offsets + block_idx * max_prompt_len, softmax_sum,
             mask=offsets < max_prompt_len)


# Wrapper function to launch the Triton kernel
def triton_fused_softmax_sum(qk_products):
    batch_size, num_heads, NUM_TOKENS_SCORE, max_prompt_len = qk_products.shape
    softmax_sum_output = torch.empty((batch_size, num_heads, max_prompt_len), dtype=torch.float32, device='cuda')
    grid = (batch_size * num_heads,)
    triton_fused_softmax_sum_kernel[grid](qk_products,
                                          max_prompt_len,
                                          softmax_sum_output,
                                          NUM_TOKENS_SCORE=NUM_TOKENS_SCORE,
                                          BLOCK_SIZE=triton.next_power_of_2(max_prompt_len))

    return softmax_sum_output
