#include "cache.h"
#include "mem_mgt.h"
#include "cuda_utils.h"
#include "ops.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // vLLM custom ops
  pybind11::module ops = m.def_submodule("ops", "vLLM custom operators");

  // Attention ops
  ops.def(
    "paged_attention_v1",
    &paged_attention_v1,
    "Compute the attention between an input query and the cached keys/values using PagedAttention.");
  ops.def(
    "paged_attention_v2",
    &paged_attention_v2,
    "PagedAttention V2.");

  // Sparse attention ops
  ops.def(
    "sparse_paged_attention",
    &sparse_paged_attention,
    "Compute the attention between an input query and the cached keys/values using class SparsePagedAttention.");

  // Activation ops
  ops.def(
    "silu_and_mul",
    &silu_and_mul,
    "Activation function used in SwiGLU.");
  ops.def(
    "gelu_new",
    &gelu_new,
    "GELU implementation used in GPT-2.");
  ops.def(
    "gelu_fast",
    &gelu_fast,
    "Approximate GELU implementation.");

  // Layernorm
  ops.def(
    "rms_norm",
    &rms_norm,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  ops.def(
    "fused_add_rms_norm",
    &fused_add_rms_norm,
    "In-place fused Add and RMS Normalization");

  // Rotary embedding
  ops.def(
    "rotary_embedding",
    &rotary_embedding,
    "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");

#ifndef USE_ROCM
  // Quantization ops
  ops.def("awq_gemm", &awq_gemm, "Quantized GEMM for AWQ");
#endif
  ops.def("gptq_gemm", &gptq_gemm, "Quantized GEMM for GPTQ");
  ops.def("gptq_shuffle", &gptq_shuffle, "Post processing for GPTQ");
  ops.def("squeezellm_gemm", &squeezellm_gemm, "Quantized GEMM for SqueezeLLM");

  // Cache ops
  pybind11::module cache_ops = m.def_submodule("cache_ops", "vLLM cache ops");
  cache_ops.def(
    "swap_blocks",
    &swap_blocks,
    "Swap in (out) the cache blocks from src to dst");
  cache_ops.def(
    "copy_blocks",
    &copy_blocks,
    "Copy the cache blocks from src to dst");
  cache_ops.def(
    "reshape_and_cache",
    &reshape_and_cache,
    "Reshape the key and value tensors and cache them");
  cache_ops.def(
    "compress_and_append_cache_prompt_phase",
    &compress_and_append_cache_prompt_phase,
    "Compress (prune + quantize) and append the cache in prompt phase");
  cache_ops.def(
    "compress_and_append_cache_long_prompt_phase",
    &compress_and_append_cache_long_prompt_phase,
    "Compress (prune + quantize) and append the cache in prompt phase for long prompts");
  cache_ops.def(
    "compress_and_append_cache_decode_phase",
    &compress_and_append_cache_decode_phase,
    "Compress (prune + quantize) and append the cache in decode phase");
  cache_ops.def(
    "gather_cached_kv",
    &gather_cached_kv,
    "Gather key and value from the cache into contiguous QKV tensors");

  // Memory management ops
  pybind11::module mem_mgt_ops = m.def_submodule("mem_mgt_ops", "GPU-native memory management ops");
  mem_mgt_ops.def(
    "allocate_seqs",
    &allocate_seqs,
    "Allocate memory for sequences in the prompt phase"
  );
  mem_mgt_ops.def(
    "prepare_free_prompt",
    &prepare_free_prompt,
    "Compute the offsets to store the released blocks in the prompt phase"
  );
  mem_mgt_ops.def(
    "free_prompt",
    &free_prompt,
    "Update memory metadata in the prompt phase with released blocks"
  );
  mem_mgt_ops.def(
    "prepare_append_seqs",
    &prepare_append_seqs,
    "Compute the block id to be appended (if needed) in the decode phase"
  );
  mem_mgt_ops.def(
    "append_seqs",
    &append_seqs,
    "Update memory metadata in the decode phase with memory layout computed by prepare_append_seqs"
  );
  mem_mgt_ops.def(
    "prepare_free_seqs",
    &prepare_free_seqs,
    "Compute the offsets to store the freed blocks when the sequence completes"
  );
  mem_mgt_ops.def(
    "free_seqs",
    &free_seqs,
    "Update memory metadata when the sequence completes with freed blocks"
  );

  // Cuda utils
  pybind11::module cuda_utils = m.def_submodule("cuda_utils", "vLLM cuda utils");
  cuda_utils.def(
    "get_device_attribute",
    &get_device_attribute,
    "Gets the specified device attribute.");
}
