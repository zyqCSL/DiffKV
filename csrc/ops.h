#pragma once

#include <torch/extension.h>

void paged_attention_v1(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes);

void paged_attention_v2(
  torch::Tensor& out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes);

void sparse_paged_attention(
  torch::Tensor& slot_ids,
  torch::Tensor& positions,
  torch::Tensor& out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& tmp_scores,
  torch::Tensor& query,
  torch::Tensor& kv_cache,
  int layer_idx,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& kv_len_tables,
  int max_context_len,
  int num_bits_k_high,
  int num_bits_v_high,
  int num_bits_k_low,
  int num_bits_v_low,
  int k_vec_size,
  int v_vec_size,
  int num_tokens_per_page_high,
  int num_tokens_per_page_low,
  const c10::optional<torch::Tensor>& alibi_slopes);

void rms_norm(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon);

void fused_add_rms_norm(
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& weight,
  float epsilon);

void rotary_embedding(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache,
  bool is_neox);

void silu_and_mul(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_new(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_fast(
  torch::Tensor& out,
  torch::Tensor& input);

#ifndef USE_ROCM
torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);
#endif

void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table);

torch::Tensor gptq_gemm(
  torch::Tensor a,
  torch::Tensor b_q_weight,
  torch::Tensor b_gptq_qzeros,
  torch::Tensor b_gptq_scales,
  torch::Tensor b_g_idx,
  bool use_exllama);

void gptq_shuffle(
  torch::Tensor q_weight,
  torch::Tensor q_perm);

void static_scaled_fp8_quant(
  torch::Tensor& out,
  torch::Tensor const& input,
  torch::Tensor const& scale);

void dynamic_scaled_fp8_quant(
  torch::Tensor& out,
  torch::Tensor const& input,
  torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scale,
    std::optional<torch::Tensor> const& scale_ub);
