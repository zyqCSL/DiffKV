#pragma once

#include <torch/extension.h>
#include <map>
#include <vector>

// prompt phase
void allocate_seqs(
  torch::Tensor& slot_ids,
  torch::Tensor& num_prompt_tokens,
  torch::Tensor& num_blocks,
  torch::Tensor& block_tables,
  torch::Tensor& kv_len_tables,
  torch::Tensor& block_num_tables,
  torch::Tensor& free_blocks,
  torch::Tensor& block_refs,
  const int start_block_pos,
  const int end_block_pos);

void prepare_free_prompt(
  torch::Tensor& slot_ids,
  torch::Tensor& kv_len_tables,
  torch::Tensor& block_num_tables,
  const int block_size_high,
  const int block_size_low,
  torch::Tensor& free_block_pos,
  torch::Tensor& block_nums);

void free_prompt(
  torch::Tensor& slot_ids,
  torch::Tensor& block_tables,
  torch::Tensor& block_num_tables,
  torch::Tensor& free_blocks,
  torch::Tensor& free_block_pos,
  torch::Tensor& block_nums,
  torch::Tensor& block_refs);

// decode phase
void prepare_append_seqs(
  torch::Tensor& slot_ids,
  torch::Tensor& kv_len_tables,
  torch::Tensor& block_num_tables,
  const int kbits_high,
  const int vbits_high,
  const int kbits_low,
  const int vbits_low,
  const int block_size_high,
  const int block_size_low,
  torch::Tensor& free_block_pos);

void append_seqs(
  torch::Tensor& slot_ids,
  torch::Tensor& block_tables,
  torch::Tensor& kv_len_tables,
  torch::Tensor& block_num_tables,
  torch::Tensor& free_blocks,
  torch::Tensor& block_refs,
  torch::Tensor& free_block_pos);

void prepare_free_seqs(
  torch::Tensor& slot_ids,
  torch::Tensor& block_num_tables,
  torch::Tensor& free_block_pos);

void free_seqs(
  torch::Tensor& slot_ids,
  torch::Tensor& block_tables,
  torch::Tensor& block_num_tables,
  torch::Tensor& free_blocks,
  torch::Tensor& free_block_pos,
  torch::Tensor& block_refs);