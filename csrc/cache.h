#pragma once

#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping);

void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

void compress_and_append_cache_prompt_phase(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& score,
  torch::Tensor& kv_cache,
  torch::Tensor& slot_ids,
  torch::Tensor& block_tables,
  torch::Tensor& kv_len_tables,
  torch::Tensor& seq_start_loc,
  torch::Tensor& compress_config_tables,
  const int kv_buffer_size,
  const int layer_idx,
  const int num_bits_k_high,
  const int num_bits_v_high,
  const int num_bits_k_low,
  const int num_bits_v_low,
  const int k_vec_size,
  const int v_vec_size,
  const int num_tokens_per_page_high,
  const int num_tokens_per_page_low);

void compress_and_append_cache_long_prompt_phase(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& score,
  torch::Tensor& sorted_indices,
  torch::Tensor& kv_cache,
  torch::Tensor& slot_ids,
  torch::Tensor& block_tables,
  torch::Tensor& kv_len_tables,
  torch::Tensor& seq_start_loc,
  torch::Tensor& compress_config_tables,
  const int kv_buffer_size,
  const int layer_idx,
  const int num_bits_k_high,
  const int num_bits_v_high,
  const int num_bits_k_low,
  const int num_bits_v_low,
  const int k_vec_size,
  const int v_vec_size,
  const int num_tokens_per_page_high,
  const int num_tokens_per_page_low);

void compress_and_append_cache_decode_phase(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& positions,
  torch::Tensor& kv_cache,
  torch::Tensor& slot_ids,
  torch::Tensor& block_tables,
  torch::Tensor& kv_len_tables,
  torch::Tensor& compress_config_tables,
  const int max_context_len,
  const int kv_buffer_size,
  const int layer_idx,
  const int num_bits_k_high,
  const int num_bits_v_high,
  const int num_bits_k_low,
  const int num_bits_v_low,
  const int k_vec_size,
  const int v_vec_size,
  const int num_tokens_per_page_high,
  const int num_tokens_per_page_low);

void gather_cached_kv(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);
