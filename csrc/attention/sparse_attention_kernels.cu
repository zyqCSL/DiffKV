/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "attention_dtypes.h"
#include "attention_utils.cuh"
#include "cache_utils.h"
#include "../quantization/kv_cache_quant/kv_cache_quant.h"

#include <algorithm>

#define DEBUG false
#ifndef USE_ROCM
#define WARP_SIZE 32
#else
#define WARP_SIZE warpSize
#endif
#define NUM_THREADS 256

#define _PARTITION_SIZE 4800

namespace vllm {

// Vectorized data types for quant_meta_type
template<>
struct Vec<quant_meta_type, 1> {
  struct Type {
    quant_meta_type data[1];
  };
};

template<>
struct Vec<quant_meta_type, 2> {
  struct Type {
    quant_meta_type data[2];
  };
};

template<>
struct Vec<quant_meta_type, 4> {
  struct Type {
    quant_meta_type data[4];
  };
};

// Utility function for attention softmax.
template<int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
  #pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
  #pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}

// For keys, one warp processes one unified page, one thread group processes one key.
// For values, one thread group processes one unified page, each thread performs reduction
// for a certain range of the embedding dim across all the tokens in the unified page.

// Grid: (num_heads, num_seqs, max_num_partitions).
template<
  typename scalar_t,
  int HEAD_SIZE,
  int NUM_QUERIES_PER_KV,
  int BITS_K_HIGH,
  int BITS_V_HIGH,
  int BITS_K_LOW,
  int BITS_V_LOW,
  int NUM_TOKENS_PER_PAGE_HIGH,
  int NUM_TOKENS_PER_PAGE_LOW,
  int THREAD_GROUP_SIZE_V,
  int PARTITION_SIZE>
__device__ void sparse_paged_attention_kernel(
  const int* __restrict__ slot_ids,        // [num_seqs]
  const int* __restrict__ positions,       // [num_seqs]
  float* __restrict__ exp_sums,            // [num_seqs, num_heads, max_num_partitions]
  float* __restrict__ max_logits,          // [num_seqs, num_heads, max_num_partitions]
  scalar_t* __restrict__ tmp_out,          // [num_seqs, num_heads, max_num_partitions, head_size]
  float* __restrict__ tmp_scores,          // [num_seqs, num_heads, max_context_len]
  // scalar_t* __restrict__ out,              // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ q,          // [num_seqs, num_heads, head_size]
  uint16_t* __restrict__ kv_cache,         // [num_blocks, unified_page_size]
  const int layer_idx,
  const int num_layers,
  const int num_kv_heads,
  const float scale,
  const int* __restrict__ block_tables,    // [num_slots, num_layers, num_kv_heads, max_num_blocks_per_seq]
  const int* __restrict__ kv_len_tables,   // [num_slots, num_layers, num_kv_heads, 2]
  uint64_t* __restrict__ sparsity_tables,       // [num_slots, num_layers, num_kv_heads]
  const int max_context_len,
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes,  // [num_heads]
  const int q_stride,
  const int unified_page_size,
  const float prune_thresh)
{
  // prune_thresh == 0 disables sparsity
  assert(prune_thresh >= 0);
  // assert(prune_thresh <= 1);
  const int partition_idx = blockIdx.z;
  const int seq_idx = blockIdx.y;
  const int slot_idx = slot_ids[seq_idx];
  const int max_num_partitions = gridDim.z;
  const int max_context_len_power2 = POWER2_ROUND_UP(max_context_len);
  const int position = positions[seq_idx];
  const float non_critical_thresh = 1.0 / position * prune_thresh;

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int kv_head_idx = head_idx / NUM_QUERIES_PER_KV;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  const int kv_len_tables_offset = slot_idx * num_layers * num_kv_heads * 2 +
                                   layer_idx * num_kv_heads * 2 + kv_head_idx * 2;
  const int context_len_left = kv_len_tables[kv_len_tables_offset];
  const int context_len_right = kv_len_tables[kv_len_tables_offset + 1];
  const int context_len = context_len_left + context_len_right;

  constexpr int K_PACK_SIZE_HIGH = 16 / BITS_K_HIGH;
  constexpr int V_PACK_SIZE_HIGH = 16 / BITS_V_HIGH;
  constexpr int K_PACK_SIZE_LOW = 16 / BITS_K_LOW;
  constexpr int V_PACK_SIZE_LOW = 16 / BITS_V_LOW;

  // assert(NUM_ELEMS_PER_THREAD % K_PACK_SIZE_HIGH == 0);
  // assert(NUM_ELEMS_PER_THREAD % K_PACK_SIZE_LOW == 0);

  constexpr int NUM_K_PACKS_HIGH = HEAD_SIZE / K_PACK_SIZE_HIGH;
  constexpr int NUM_V_PACKS_HIGH = HEAD_SIZE / V_PACK_SIZE_HIGH;
  constexpr int NUM_K_PACKS_LOW = HEAD_SIZE / K_PACK_SIZE_LOW;
  constexpr int NUM_V_PACKS_LOW = HEAD_SIZE / V_PACK_SIZE_LOW;

  // Align the starting address of each segment (key, key meta, val, val meta, score, pos) to 32 bytes.
  constexpr int KEY_BASE_HIGH = 0;
  constexpr int KEY_META_BASE_HIGH = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * NUM_K_PACKS_HIGH * 2, 32) * 32 / sizeof(uint16_t) + KEY_BASE_HIGH;
  constexpr int VAL_BASE_HIGH = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * 4 + KEY_META_BASE_HIGH * sizeof(uint16_t), 128) * 128 / sizeof(uint16_t);
  constexpr int PADDED_NUM_TOKENS_PER_PAGE_HIGH = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH, V_VEC_SIZE) * V_VEC_SIZE;
  constexpr int VAL_META_BASE_HIGH = DIVIDE_ROUND_UP(PADDED_NUM_TOKENS_PER_PAGE_HIGH * NUM_V_PACKS_HIGH * 2, 32) * 32 / sizeof(uint16_t) + VAL_BASE_HIGH;
  // constexpr int SCORE_BASE_HIGH = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * 4, 32) * 32 / sizeof(uint16_t) + VAL_META_BASE_HIGH;
  // constexpr int POS_BASE_HIGH = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * 4, 32) * 32 / sizeof(uint16_t) + SCORE_BASE_HIGH;

  constexpr int KEY_BASE_LOW = 0;
  constexpr int KEY_META_BASE_LOW = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * NUM_K_PACKS_LOW * 2, 32) * 32 / sizeof(uint16_t) + KEY_BASE_LOW;
  constexpr int VAL_BASE_LOW = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * 4 + KEY_META_BASE_LOW * sizeof(uint16_t), 128) * 128 / sizeof(uint16_t);
  constexpr int PADDED_NUM_TOKENS_PER_PAGE_LOW = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW, V_VEC_SIZE) * V_VEC_SIZE;
  constexpr int VAL_META_BASE_LOW = DIVIDE_ROUND_UP(PADDED_NUM_TOKENS_PER_PAGE_LOW * NUM_V_PACKS_LOW * 2, 32) * 32 / sizeof(uint16_t) + VAL_BASE_LOW;
  // constexpr int SCORE_BASE_LOW = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * 4, 32) * 32 / sizeof(uint16_t) + VAL_META_BASE_LOW;
  // constexpr int POS_BASE_LOW = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * 4, 32) * 32 / sizeof(uint16_t) + SCORE_BASE_LOW;

  const int num_pages_high = DIVIDE_ROUND_UP(context_len_left, NUM_TOKENS_PER_PAGE_HIGH);
  const int num_pages_low = DIVIDE_ROUND_UP(context_len_right, NUM_TOKENS_PER_PAGE_LOW);

  assert(PARTITION_SIZE % NUM_TOKENS_PER_PAGE_HIGH == 0);
  assert(PARTITION_SIZE % NUM_TOKENS_PER_PAGE_LOW == 0);
  constexpr int NUM_PAGES_HIGH_PER_PARTITION = PARTITION_SIZE / NUM_TOKENS_PER_PAGE_HIGH;
  constexpr int NUM_PAGES_LOW_PER_PARTITION = PARTITION_SIZE / NUM_TOKENS_PER_PAGE_LOW;

  // Configuration for no partitioning.
  int num_partitions = 1;
  int start_page_left = 0;
  int end_page_left = num_pages_high;
  int start_page_right = 0;
  int end_page_right = num_pages_low;
  int start_token_idx = 0;
  int num_tokens = context_len;

  const bool USE_PARTITIONING = context_len > PARTITION_SIZE;
  if (USE_PARTITIONING) {
    int num_partitions_high = DIVIDE_ROUND_UP(num_pages_high, NUM_PAGES_HIGH_PER_PARTITION);
    int num_partitions_low = DIVIDE_ROUND_UP(num_pages_low, NUM_PAGES_LOW_PER_PARTITION);
    num_partitions = num_partitions_high + num_partitions_low;
    assert(num_partitions > 1);
    assert(num_partitions <= max_num_partitions);
    if (partition_idx < num_partitions_high) {
      start_page_left = partition_idx * NUM_PAGES_HIGH_PER_PARTITION;
      end_page_left = MIN((partition_idx + 1) * NUM_PAGES_HIGH_PER_PARTITION, num_pages_high);
      start_page_right = -1;  // No work to do for the right side.
      end_page_right = -1;
      start_token_idx = start_page_left * NUM_TOKENS_PER_PAGE_HIGH;
      num_tokens = MIN(end_page_left * NUM_TOKENS_PER_PAGE_HIGH, context_len_left) - start_token_idx;
    } else if (partition_idx < num_partitions) {
      start_page_left = -1;  // No work to do for the left side.
      end_page_left = -1;
      start_page_right = (partition_idx - num_partitions_high) * NUM_PAGES_LOW_PER_PARTITION;
      end_page_right = MIN((partition_idx - num_partitions_high + 1) * NUM_PAGES_LOW_PER_PARTITION, num_pages_low);
      start_token_idx = context_len_left + start_page_right * NUM_TOKENS_PER_PAGE_LOW;
      num_tokens = MIN(end_page_right * NUM_TOKENS_PER_PAGE_LOW, context_len_right) + context_len_left - start_token_idx;
    }
  }

  if (partition_idx >= num_partitions) {
    // No work to do. Terminate the thread block.
    return;
  }

  if (DEBUG && thread_idx == 0 && layer_idx == 0 && head_idx == 0) {
    printf("[Debug info from sparse_paged_attention_kernel] seq_idx: %d, partition_idx: %d, PARTITION_SIZE: %d, NUM_PAGES_HIGH_PER_PARTITION: %d, NUM_PAGES_LOW_PER_PARTITION: %d, num_partitions: %d\n",
      seq_idx, partition_idx, PARTITION_SIZE, NUM_PAGES_HIGH_PER_PARTITION, NUM_PAGES_LOW_PER_PARTITION, num_partitions);
    printf("[Debug info from sparse_paged_attention_kernel] seq_idx: %d, partition_idx: %d, start_page_left: %d, end_page_left: %d, start_page_right: %d, end_page_right: %d, start_token_idx: %d, num_tokens: %d\n",
      seq_idx, partition_idx, start_page_left, end_page_left, start_page_right, end_page_right, start_token_idx, num_tokens);
  }

  constexpr int NUM_THREAD_GROUPS_K = WARP_SIZE / THREAD_GROUP_SIZE_K;
  constexpr int NUM_TOKENS_PER_THREAD_GROUP_HIGH = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH, NUM_THREAD_GROUPS_K);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP_LOW = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW, NUM_THREAD_GROUPS_K);

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE_K;
  constexpr int NUM_PACKS_PER_THREAD_K_HIGH = NUM_ELEMS_PER_THREAD / K_PACK_SIZE_HIGH;
  constexpr int NUM_VECS_PER_THREAD_K_HIGH = NUM_PACKS_PER_THREAD_K_HIGH / K_VEC_SIZE;
  constexpr int NUM_PACKS_PER_THREAD_K_LOW = NUM_ELEMS_PER_THREAD / K_PACK_SIZE_LOW;
  constexpr int NUM_VECS_PER_THREAD_K_LOW = NUM_PACKS_PER_THREAD_K_LOW / K_VEC_SIZE;

  // the 1st dimension key's layout
  assert(NUM_K_PACKS_HIGH % (K_VEC_SIZE * THREAD_GROUP_SIZE_K) == 0);
  assert(NUM_K_PACKS_LOW % (K_VEC_SIZE * THREAD_GROUP_SIZE_K) == 0);

  const int thread_group_idx_k = lane % NUM_THREAD_GROUPS_K;
  const int thread_group_offset_k = lane / NUM_THREAD_GROUPS_K;

  // const int thread_group_idx_k = lane / THREAD_GROUP_SIZE_K;
  // const int thread_group_offset_k = lane % THREAD_GROUP_SIZE_K;

  if (DEBUG && layer_idx == 0 && seq_idx == 0 && head_idx == 0 && thread_idx == 0) {
    printf("[Debug info from sparse_attention_kernels.cu] context_len_left: %d, context_len_right: %d, num_pages_high: %d, num_pages_low: %d\n",
      context_len_left, context_len_right, num_pages_high, num_pages_low);
  }

  // Load the query to registers.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  // Query elements are loaded to q_floats in a blocked manner.
  // For example, if the thread group size is 2, then q_floats[0] stores the first half (0-63) of the query,
  // and q_floats[1] stores the second half (64-127).
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ float q_floats[THREAD_GROUP_SIZE_K][NUM_ELEMS_PER_THREAD];
  #pragma unroll
  for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
    q_floats[i / NUM_ELEMS_PER_THREAD][i % NUM_ELEMS_PER_THREAD] = to_float(q_ptr[i]);
  }
  __syncthreads(); // TODO(naed90): possible speedup if this is replaced with a memory wall right before we use q_floats

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  float qk_max = -FLT_MAX;

  // Iterate over the keys.
  // One warp processes one unified page, one thread group processes one key at each iteration.
  // Each thread group in a warp fetches a key from the unified page, and computes dot product with the query.
  const int* block_table_left = block_tables +
                                (slot_idx * num_layers * num_kv_heads +
                                layer_idx * num_kv_heads +
                                kv_head_idx) * max_num_blocks_per_seq;
  const int* block_table_right = block_table_left + max_num_blocks_per_seq - 1;

  // clock_t t_0 = clock();

  // Process high-precision unified pages.
  for (int context_block_idx = warp_idx + start_page_left; context_block_idx < end_page_left; context_block_idx += NUM_WARPS) {
    int64_t physical_block_number = static_cast<int64_t>(*(block_table_left + context_block_idx));
    int64_t k_offset = physical_block_number * unified_page_size + KEY_BASE_HIGH;
    int64_t k_meta_offset = physical_block_number * unified_page_size + KEY_META_BASE_HIGH;

    for (int t = 0; t < NUM_TOKENS_PER_THREAD_GROUP_HIGH; t++) {
      // Load a key to registers.
      // Each thread in a thread group has a different part of the key.
      // For example, if the thread group size is 2, then the first thread in the group
      // has the first half (0-63) of the key, and the second thread has the second half (64-127).
      float k_floats[NUM_ELEMS_PER_THREAD];

      const int token_idx_within_the_page = t * NUM_THREAD_GROUPS_K + thread_group_idx_k;
      const int token_idx = context_block_idx * NUM_TOKENS_PER_PAGE_HIGH + token_idx_within_the_page;

      if (token_idx_within_the_page < NUM_TOKENS_PER_PAGE_HIGH && token_idx < context_len_left) {
        // TOOD: Each thread in a thread group needs to load k_quant_scale and k_quant_zero_point.
        //       The group size is typically 2. Can we optimize it?
        quant_meta_type k_quant_meta = *reinterpret_cast<quant_meta_type*>(&kv_cache[k_meta_offset + token_idx_within_the_page * 2]);
        const float k_quant_scale = to_float(k_quant_meta.scale);
        const float k_quant_zero_point = to_float(k_quant_meta.zero_point);

        #pragma unroll
        for (int i = 0; i < NUM_VECS_PER_THREAD_K_HIGH; i++) {
          // layout of keys within a unified page:
          // [NUM_PACKS/K_VEC_SIZE/THREAD_GROUP_SIZE_K, NUM_TOKENS_PER_PAGE, THREAD_GROUP_SIZE_K, K_VEC_SIZE]
          const int idx = (i * NUM_TOKENS_PER_PAGE_HIGH * THREAD_GROUP_SIZE_K // dim 0
                        + token_idx_within_the_page * THREAD_GROUP_SIZE_K // dim 1
                        + thread_group_offset_k) * K_VEC_SIZE;   // dim 2
          k_vec_type k_vec = *reinterpret_cast<k_vec_type*>(&kv_cache[k_offset + idx]);

          #pragma unroll
          for (int j = 0; j < K_VEC_SIZE; j++) {
            unpack_and_dequant(k_vec.data[j],
                               BITS_K_HIGH,
                               k_quant_scale,
                               k_quant_zero_point,
                               &k_floats[(i * K_VEC_SIZE + j) * K_PACK_SIZE_HIGH]);
          }
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<THREAD_GROUP_SIZE_K>::dot(q_floats[thread_group_offset_k], k_floats);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      // if (DEBUG && token_idx < context_len && layer_idx == 0 && seq_idx == 0 && kv_head_idx == 0 && thread_group_offset_k == 0) {
      //   printf("[Debug info from sparse_paged_attention_kernel] token_idx: %d, physical_block_number: %ld\n",
      //     token_idx, physical_block_number);
      //   printf("[Debug info from sparse_paged_attention_kernel] k_floats[0]: %f, k_floats[1]: %f, k_floats[2]: %f, k_floats[3]: %f, k_floats[4]: %f, k_floats[5]: %f, k_floats[6]: %f, k_floats[7]: %f\n",
      //     k_floats[0], k_floats[1], k_floats[2], k_floats[3], k_floats[4], k_floats[5], k_floats[6], k_floats[7]);
      // }

      if (token_idx_within_the_page < NUM_TOKENS_PER_PAGE_HIGH
          && token_idx < context_len_left
          && thread_group_offset_k == 0) {
        // Store the partial reductions to shared memory.
        logits[token_idx - start_token_idx] = qk;
        // Update the max value.
        qk_max = fmaxf(qk_max, qk);
      }
    }
  }

  // Process low-precision unified pages.
  for (int context_block_idx = warp_idx + start_page_right; context_block_idx < end_page_right; context_block_idx += NUM_WARPS) {
    int64_t physical_block_number = static_cast<int64_t>(*(block_table_right - context_block_idx));
    int64_t k_offset = physical_block_number * unified_page_size + KEY_BASE_LOW;
    int64_t k_meta_offset = physical_block_number * unified_page_size + KEY_META_BASE_LOW;

    for (int t = 0; t < NUM_TOKENS_PER_THREAD_GROUP_LOW; t++) {
      float k_floats[NUM_ELEMS_PER_THREAD];
      const int token_idx_within_the_page = t * NUM_THREAD_GROUPS_K + thread_group_idx_k;
      const int token_idx = context_len_left + context_block_idx * NUM_TOKENS_PER_PAGE_LOW + token_idx_within_the_page;

      if (token_idx_within_the_page < NUM_TOKENS_PER_PAGE_LOW && token_idx < context_len) {
        quant_meta_type k_quant_meta = *reinterpret_cast<quant_meta_type*>(&kv_cache[k_meta_offset + token_idx_within_the_page * 2]);
        const float k_quant_scale = to_float(k_quant_meta.scale);
        const float k_quant_zero_point = to_float(k_quant_meta.zero_point);

        #pragma unroll
        for (int i = 0; i < NUM_VECS_PER_THREAD_K_LOW; i++) {
          // layout of keys within a unified page:
          // [NUM_PACKS/K_VEC_SIZE/THREAD_GROUP_SIZE_K, NUM_TOKENS_PER_PAGE, THREAD_GROUP_SIZE_K, K_VEC_SIZE]
          const int idx = (i * NUM_TOKENS_PER_PAGE_LOW * THREAD_GROUP_SIZE_K // dim 0
                        + token_idx_within_the_page * THREAD_GROUP_SIZE_K // dim 1
                        + thread_group_offset_k) * K_VEC_SIZE;   // dim 2
          k_vec_type k_vec = *reinterpret_cast<k_vec_type*>(&kv_cache[k_offset + idx]);

          #pragma unroll
          for (int j = 0; j < K_VEC_SIZE; j++) {
            unpack_and_dequant(k_vec.data[j],
                               BITS_K_LOW,
                               k_quant_scale,
                               k_quant_zero_point,
                               &k_floats[(i * K_VEC_SIZE + j) * K_PACK_SIZE_LOW]);
          }
        }
      }

      float qk = scale * Qk_dot<THREAD_GROUP_SIZE_K>::dot(q_floats[thread_group_offset_k], k_floats);
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      if (token_idx_within_the_page < NUM_TOKENS_PER_PAGE_LOW
          && token_idx < context_len
          && thread_group_offset_k == 0) {
        // Store the partial reductions to shared memory.
        logits[token_idx - start_token_idx] = qk;
        // Update the max value.
        qk_max = fmaxf(qk_max, qk);
      }
    }
  }

  // clock_t t_1 = clock();

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
  #pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE_K; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  assert(WARP_SIZE >= NUM_WARPS);
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
  #pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // sparsity log
  int num_critical_keys = 0;
  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
    if (logits[i] >= non_critical_thresh) {
      num_critical_keys += 1;
    }
  }
  __syncthreads();

  num_critical_keys = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], num_critical_keys);
  if (thread_idx == 0) {
    const int sparsity_tables_offset = slot_idx * num_layers * num_heads +
                                       layer_idx * num_heads + head_idx;
    // We need atomicAdd because multiple cuda blocks may update the same location due to sequence partitioning.
    atomicAdd((unsigned long long*)&sparsity_tables[sparsity_tables_offset], num_critical_keys);
  }

  // Print the first 10 logits for debugging purpose.
  if (DEBUG && layer_idx == 0 && seq_idx == 0 && kv_head_idx == 0 && thread_idx == 0) {
    printf("[Debug info from sparse_paged_attention_kernel] logits: ");
    for (int i = 0; i < MIN(10, num_tokens); i++) {
      printf("%f ", logits[i]);
    }
    printf("\n");
  }

  // Write the softmax logits to tmp_scores.
  float* tmp_scores_ptr = tmp_scores + seq_idx * num_heads * max_context_len
                                      + head_idx * max_context_len;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    tmp_scores_ptr[i + start_token_idx] = logits[i];
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions
                                       + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                   + head_idx * max_num_partitions
                                   + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // layout of values within a unified page: [NUM_PACKS / THREAD_GROUP_SIZE_V,
  //                                          NUM_TOKENS_PER_PAGE / V_VEC_SIZE,
  //                                          THREAD_GROUP_SIZE_V,
  //                                          V_VEC_SIZE]
  // To support 1 bit quantization, NUM_ROWS_PER_THREAD should be 16 (set THREAD_GROUP_SIZE_V to 8)

  constexpr int NUM_THREAD_GROUPS_V = WARP_SIZE / THREAD_GROUP_SIZE_V;

  constexpr int NUM_ROWS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE_V;
  constexpr int NUM_PACKS_PER_THREAD_V_HIGH = NUM_ROWS_PER_THREAD / V_PACK_SIZE_HIGH;
  constexpr int NUM_PACKS_PER_THREAD_V_LOW = NUM_ROWS_PER_THREAD / V_PACK_SIZE_LOW;

  static_assert(HEAD_SIZE % THREAD_GROUP_SIZE_V == 0);
  static_assert(NUM_ROWS_PER_THREAD % V_PACK_SIZE_HIGH == 0);
  static_assert(NUM_ROWS_PER_THREAD % V_PACK_SIZE_LOW == 0);

  constexpr int NUM_VECS_HIGH = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH, V_VEC_SIZE);
  constexpr int NUM_VECS_LOW = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW, V_VEC_SIZE);
  constexpr int NUM_TOKEN_VECS_PER_THREAD_GROUP_HIGH = DIVIDE_ROUND_UP(NUM_VECS_HIGH, NUM_THREAD_GROUPS_V);
  constexpr int NUM_TOKEN_VECS_PER_THREAD_GROUP_LOW = DIVIDE_ROUND_UP(NUM_VECS_LOW, NUM_THREAD_GROUPS_V);

  const int thread_group_idx_v = lane / THREAD_GROUP_SIZE_V;
  const int thread_group_offset_v = lane % THREAD_GROUP_SIZE_V;

  // NOTE: the following config is slower
  // const int thread_group_idx_v = lane % THREAD_GROUP_SIZE_V;
  // const int thread_group_offset_v = lane / THREAD_GROUP_SIZE_V;

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  // tmp buffer for unpack_and_dequant, NUM_ROWS_PER_THREAD >= v_vec_size
  float v_floats[NUM_ROWS_PER_THREAD];

  // Workspace for quantization metadata.
  // Note: loading and accessing quantization metadata incurs less than 0.05 ms overhead, which is small.
  // constexpr int QUANT_META_SIZE = NUM_THREAD_GROUPS_V * MAX(NUM_TOKENS_PER_PAGE_HIGH, NUM_TOKENS_PER_PAGE_LOW);
  constexpr int QUANT_META_SIZE = NUM_WARPS * MAX(NUM_TOKENS_PER_PAGE_HIGH, NUM_TOKENS_PER_PAGE_LOW);
  __shared__ quant_meta_type v_quant_meta[QUANT_META_SIZE];

  // Create cooperative groups for sub-warp synchronization
  // auto group = cg::tiled_partition<THREAD_GROUP_SIZE_V>(cg::this_thread_block());

  // clock_t t_2 = clock();

  // Process high-precision unified pages.
  for (int context_block_idx = warp_idx + start_page_left; context_block_idx < end_page_left; context_block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(*(block_table_left + context_block_idx));
    const int64_t v_offset = physical_block_number * unified_page_size + VAL_BASE_HIGH;
    const int64_t v_meta_offset = physical_block_number * unified_page_size + VAL_META_BASE_HIGH;

    // Write the quantization metadata to shared memory.
    #pragma unroll
    for (int i = 0; i < DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH, WARP_SIZE); i++) {
      const int token_idx_within_the_page = i * WARP_SIZE + lane;
      if (token_idx_within_the_page < NUM_TOKENS_PER_PAGE_HIGH) {
        v_quant_meta[warp_idx * NUM_TOKENS_PER_PAGE_HIGH + token_idx_within_the_page] =
          *reinterpret_cast<quant_meta_type*>(&kv_cache[v_meta_offset + token_idx_within_the_page * 2]);
      }
    }
    // NOTE: __syncthreads() or __syncwarp() would cause deadlock, because some thread groups never enter the for loop.
    // group.sync();
    __syncwarp();

    #pragma unroll
    for (int p_idx = 0; p_idx < NUM_PACKS_PER_THREAD_V_HIGH; p_idx++) {
      #pragma unroll
      for (int v = 0; v < NUM_TOKEN_VECS_PER_THREAD_GROUP_HIGH; v++) {
        // layout of values within a unified page: [NUM_PACKS / THREAD_GROUP_SIZE_V,
        //                                          NUM_TOKENS_PER_PAGE / V_VEC_SIZE,
        //                                          THREAD_GROUP_SIZE_V,
        //                                          V_VEC_SIZE]
        const int vec_idx = v * NUM_THREAD_GROUPS_V + thread_group_idx_v;
        const int idx = p_idx * PADDED_NUM_TOKENS_PER_PAGE_HIGH * THREAD_GROUP_SIZE_V  // dim 0
                      + (vec_idx * THREAD_GROUP_SIZE_V + thread_group_offset_v) * V_VEC_SIZE;  // dim 1 & dim 2
        v_vec_type v_vec = *reinterpret_cast<v_vec_type*>(&kv_cache[v_offset + idx]);

        #pragma unroll
        for (int i = 0; i < V_VEC_SIZE; i++) {
          const int token_idx_within_the_page = vec_idx * V_VEC_SIZE + i;
          const int token_idx = context_block_idx * NUM_TOKENS_PER_PAGE_HIGH + token_idx_within_the_page;
          // const quant_meta_type meta = v_quant_meta[thread_group_idx_v * NUM_TOKENS_PER_PAGE_HIGH + token_idx_within_the_page];
          const quant_meta_type meta = v_quant_meta[warp_idx * NUM_TOKENS_PER_PAGE_HIGH + token_idx_within_the_page];
          unpack_and_dequant(v_vec.data[i],
                             BITS_V_HIGH,
                             to_float(meta.scale),
                             to_float(meta.zero_point),
                             v_floats);
          if (token_idx_within_the_page < NUM_TOKENS_PER_PAGE_HIGH && token_idx < context_len_left) {
            #pragma unroll
            for (int k = 0; k < V_PACK_SIZE_HIGH; k++) {
              if (logits[token_idx - start_token_idx] >= non_critical_thresh) {
                accs[p_idx * V_PACK_SIZE_HIGH + k] += logits[token_idx - start_token_idx] * v_floats[k];
              }
            }
          }
        }
      }
    }
  }

  __syncthreads(); // shared memory space for v_quant_meta is reused in the next loop.

  // Process low-precision unified pages.
  for (int context_block_idx = warp_idx + start_page_right; context_block_idx < end_page_right; context_block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(*(block_table_right - context_block_idx));
    const int64_t v_offset = physical_block_number * unified_page_size + VAL_BASE_LOW;
    const int64_t v_meta_offset = physical_block_number * unified_page_size + VAL_META_BASE_LOW;

    #pragma unroll
    for (int i = 0; i < DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW, WARP_SIZE); i++) {
      const int token_idx_within_the_page = i * WARP_SIZE + lane;
      if (token_idx_within_the_page < NUM_TOKENS_PER_PAGE_LOW) {
        v_quant_meta[warp_idx * NUM_TOKENS_PER_PAGE_LOW + token_idx_within_the_page] =
          *reinterpret_cast<quant_meta_type*>(&kv_cache[v_meta_offset + token_idx_within_the_page * 2]);
      }
    }
    // group.sync();
    __syncwarp();

    #pragma unroll
    for (int p_idx = 0; p_idx < NUM_PACKS_PER_THREAD_V_LOW; p_idx++) {
      #pragma unroll
      for (int v = 0; v < NUM_TOKEN_VECS_PER_THREAD_GROUP_LOW; v++) {
        // layout of values within a unified page: [NUM_PACKS / THREAD_GROUP_SIZE_V,
        //                                          NUM_TOKENS_PER_PAGE / V_VEC_SIZE,
        //                                          THREAD_GROUP_SIZE_V,
        //                                          V_VEC_SIZE]
        const int vec_idx = v * NUM_THREAD_GROUPS_V + thread_group_idx_v;
        const int idx = p_idx * PADDED_NUM_TOKENS_PER_PAGE_LOW * THREAD_GROUP_SIZE_V  // dim 0
                      + (vec_idx * THREAD_GROUP_SIZE_V + thread_group_offset_v) * V_VEC_SIZE;  // dim 1 & dim 2
        v_vec_type v_vec = *reinterpret_cast<v_vec_type*>(&kv_cache[v_offset + idx]);

        #pragma unroll
        for (int i = 0; i < V_VEC_SIZE; i++) {
          const int token_idx_within_the_page = vec_idx * V_VEC_SIZE + i;
          const int token_idx = context_block_idx * NUM_TOKENS_PER_PAGE_LOW + token_idx_within_the_page;
          // const quant_meta_type meta = v_quant_meta[thread_group_idx_v * NUM_TOKENS_PER_PAGE_LOW + token_idx_within_the_page];
          const quant_meta_type meta = v_quant_meta[warp_idx * NUM_TOKENS_PER_PAGE_LOW + token_idx_within_the_page];
          unpack_and_dequant(v_vec.data[i],
                             BITS_V_LOW,
                             to_float(meta.scale),
                             to_float(meta.zero_point),
                             v_floats);
          if (token_idx_within_the_page < NUM_TOKENS_PER_PAGE_LOW && token_idx < context_len_right) {
            #pragma unroll
            for (int k = 0; k < V_PACK_SIZE_LOW; k++) {
              if (logits[token_idx + context_len_left - start_token_idx] >= prune_thresh) {
                accs[p_idx * V_PACK_SIZE_LOW + k] += logits[token_idx + context_len_left - start_token_idx] * v_floats[k];
              }
            }
          }
        }
      }
    }
  }

  // clock_t t_3 = clock();

  // if (layer_idx == 0 && seq_idx == 0 && head_idx == 0 && thread_idx == 0) {
  //   printf("[Debug info from sparse_attention_kernels.cu] process keys: %f ms, process values: %f ms\n",
  //           (float)(t_1 - t_0) / CLOCKS_PER_SEC,
  //           (float)(t_3 - t_2) / CLOCKS_PER_SEC);
  // }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  const int thread_group_idx_within_block_v = warp_idx * NUM_THREAD_GROUPS_V + thread_group_idx_v;
  float* out_smem = reinterpret_cast<float*>(shared_mem);
  #pragma unroll
  for (int i = NUM_WARPS * NUM_THREAD_GROUPS_V; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (thread_group_idx_within_block_v >= mid && thread_group_idx_within_block_v < i) {
      float* dst = &out_smem[(thread_group_idx_within_block_v - mid) * HEAD_SIZE];
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = thread_group_offset_v * NUM_ROWS_PER_THREAD + i;
        dst[row_idx] = accs[i];
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (thread_group_idx_within_block_v < mid) {
      const float* src = &out_smem[thread_group_idx_within_block_v * HEAD_SIZE];
      #pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = thread_group_offset_v * NUM_ROWS_PER_THREAD + i;
        accs[i] += src[row_idx];
      }
    }
    __syncthreads();
  }

  // Write the output to tmp_out
  if (thread_group_idx_within_block_v == 0) {
    scalar_t* out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                                + head_idx * max_num_partitions * HEAD_SIZE
                                + partition_idx * HEAD_SIZE;

    #pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = thread_group_offset_v * NUM_ROWS_PER_THREAD + i;
      from_float(*(out_ptr + row_idx), accs[i]);
    }
  }
}

// Grid: (num_heads, num_seqs)
template<
  typename scalar_t,
  int HEAD_SIZE,
  int NUM_QUERIES_PER_KV,
  int BITS_K_HIGH,
  int BITS_V_HIGH,
  int BITS_K_LOW,
  int BITS_V_LOW,
  int NUM_TOKENS_PER_PAGE_HIGH,
  int NUM_TOKENS_PER_PAGE_LOW,
  int PARTITION_SIZE>
__global__ void reduce_kernel(
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const float* __restrict__ exp_sums,     // [num_seqs, num_heads, max_num_partitions]
  const float* __restrict__ max_logits,   // [num_seqs, num_heads, max_num_partitions]
  const scalar_t* __restrict__ tmp_out,   // [num_seqs, num_heads, max_num_partitions, head_size]
  float* __restrict__ tmp_scores,         // [num_seqs, num_heads, max_context_len]
  const int* __restrict__ block_tables,   // [num_slots, num_layers, num_kv_heads, max_num_blocks_per_seq]
  const int* __restrict__ kv_len_tables,  // [num_slots, num_layers, num_kv_heads, 2]
  const int* __restrict__ slot_ids,       // [num_seqs]
  uint16_t* __restrict__ kv_cache,        // [num_blocks, unified_page_size]
  const int unified_page_size,
  const int layer_idx,
  const int num_layers,
  const int num_kv_heads,
  const int max_context_len,
  const int max_num_blocks_per_seq,
  const int max_num_partitions)
{
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / NUM_QUERIES_PER_KV;
  const int seq_idx = blockIdx.y;
  const int slot_idx = slot_ids[seq_idx];

  constexpr int K_PACK_SIZE_HIGH = 16 / BITS_K_HIGH;
  constexpr int V_PACK_SIZE_HIGH = 16 / BITS_V_HIGH;
  constexpr int K_PACK_SIZE_LOW = 16 / BITS_K_LOW;
  constexpr int V_PACK_SIZE_LOW = 16 / BITS_V_LOW;

  constexpr int NUM_K_PACKS_HIGH = HEAD_SIZE / K_PACK_SIZE_HIGH;
  constexpr int NUM_V_PACKS_HIGH = HEAD_SIZE / V_PACK_SIZE_HIGH;
  constexpr int NUM_K_PACKS_LOW = HEAD_SIZE / K_PACK_SIZE_LOW;
  constexpr int NUM_V_PACKS_LOW = HEAD_SIZE / V_PACK_SIZE_LOW;

  constexpr int KEY_BASE_HIGH = 0;
  constexpr int KEY_META_BASE_HIGH = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * NUM_K_PACKS_HIGH * 2, 32) * 32 / sizeof(uint16_t) + KEY_BASE_HIGH;
  constexpr int VAL_BASE_HIGH = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * 4 + KEY_META_BASE_HIGH * sizeof(uint16_t), 128) * 128 / sizeof(uint16_t);
  constexpr int PADDED_NUM_TOKENS_PER_PAGE_HIGH = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH, V_VEC_SIZE) * V_VEC_SIZE;
  constexpr int VAL_META_BASE_HIGH = DIVIDE_ROUND_UP(PADDED_NUM_TOKENS_PER_PAGE_HIGH * NUM_V_PACKS_HIGH * 2, 32) * 32 / sizeof(uint16_t) + VAL_BASE_HIGH;
  constexpr int SCORE_BASE_HIGH = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * 4, 32) * 32 / sizeof(uint16_t) + VAL_META_BASE_HIGH;

  constexpr int KEY_BASE_LOW = 0;
  constexpr int KEY_META_BASE_LOW = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * NUM_K_PACKS_LOW * 2, 32) * 32 / sizeof(uint16_t) + KEY_BASE_LOW;
  constexpr int VAL_BASE_LOW = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * 4 + KEY_META_BASE_LOW * sizeof(uint16_t), 128) * 128 / sizeof(uint16_t);
  constexpr int PADDED_NUM_TOKENS_PER_PAGE_LOW = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW, V_VEC_SIZE) * V_VEC_SIZE;
  constexpr int VAL_META_BASE_LOW = DIVIDE_ROUND_UP(PADDED_NUM_TOKENS_PER_PAGE_LOW * NUM_V_PACKS_LOW * 2, 32) * 32 / sizeof(uint16_t) + VAL_BASE_LOW;
  constexpr int SCORE_BASE_LOW = DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * 4, 32) * 32 / sizeof(uint16_t) + VAL_META_BASE_LOW;

  const int* block_table_left = block_tables +
                                (slot_idx * num_layers * num_kv_heads +
                                layer_idx * num_kv_heads +
                                kv_head_idx) * max_num_blocks_per_seq;
  const int* block_table_right = block_table_left + max_num_blocks_per_seq - 1;

  const int kv_len_tables_offset = slot_idx * num_layers * num_kv_heads * 2 +
                                   layer_idx * num_kv_heads * 2 + kv_head_idx * 2;
  const int context_len_left = kv_len_tables[kv_len_tables_offset];
  const int context_len_right = kv_len_tables[kv_len_tables_offset + 1];
  const int context_len = context_len_left + context_len_right;

  int num_partitions = 1;  // No partitioning by default.
  const bool USE_PARTITIONING = context_len > PARTITION_SIZE;
  if (USE_PARTITIONING) {
    int num_partitions_high = DIVIDE_ROUND_UP(context_len_left, PARTITION_SIZE);
    int num_partitions_low = DIVIDE_ROUND_UP(context_len_right, PARTITION_SIZE);
    num_partitions = num_partitions_high + num_partitions_low;
    assert(num_partitions > 1);
    assert(num_partitions <= max_num_partitions);
  }

  if (!USE_PARTITIONING) {
    // No need to reduce. Only copy tmp_out to out.
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                                          + head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
      out_ptr[i] = tmp_out_ptr[i];
    }

    // The first head in NUM_QUERIES_PER_KV heads computes the max score and updates the score cache.
    if (head_idx % NUM_QUERIES_PER_KV == 0) {
      for (int token_idx = threadIdx.x; token_idx < context_len; token_idx += NUM_THREADS) {
        float score = 0.0f;
        #pragma unroll
        for (int j = 0; j < NUM_QUERIES_PER_KV; j++) {
          score = MAX(score, tmp_scores[seq_idx * num_heads * max_context_len
                                        + (kv_head_idx * NUM_QUERIES_PER_KV + j) * max_context_len
                                        + token_idx]);
        }
        int64_t score_offset;
        if (token_idx < context_len_left) {
          // Access block table from left
          int64_t physical_block_number = static_cast<int64_t>(*(block_table_left +
            (token_idx / NUM_TOKENS_PER_PAGE_HIGH)));
          score_offset = physical_block_number * unified_page_size + SCORE_BASE_HIGH
                        + token_idx % NUM_TOKENS_PER_PAGE_HIGH * 2;
        } else {
          // Access block table from right
          int64_t physical_block_number = static_cast<int64_t>(*(block_table_right -
            (token_idx - context_len_left) / NUM_TOKENS_PER_PAGE_LOW));
          score_offset = physical_block_number * unified_page_size + SCORE_BASE_LOW
                        + (token_idx - context_len_left) % NUM_TOKENS_PER_PAGE_LOW * 2;
        }
        *reinterpret_cast<float*>(&kv_cache[score_offset]) += score;
      }
    }

    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                           + head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += NUM_THREADS) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
  #pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
  #pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = VLLM_SHFL_SYNC(max_logit, 0);

  // Load rescale factors to shared memory.
  float* rescale_factors = reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += NUM_THREADS) {
    float l = shared_max_logits[i];
    global_exp_sum += exp_sums_ptr[i] * expf(l - max_logit);
    rescale_factors[i] = (exp_sums_ptr[i] + 1e-6) * expf(l - max_logit);
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                                        + head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
  #pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * rescale_factors[j] * inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }

  // Rescale tmp_scores
  float* tmp_scores_ptr = tmp_scores + seq_idx * num_heads * max_context_len + head_idx * max_context_len;
  #pragma unroll
  for (int token_idx = threadIdx.x; token_idx < context_len; token_idx += NUM_THREADS) {
    int partition_idx = token_idx / PARTITION_SIZE;
    tmp_scores_ptr[token_idx] *= rescale_factors[partition_idx] * inv_global_exp_sum;
  }

  // The first head in NUM_QUERIES_PER_KV heads computes the max score and updates the score cache.
  if (head_idx % NUM_QUERIES_PER_KV == 0) {
    for (int token_idx = threadIdx.x; token_idx < context_len; token_idx += NUM_THREADS) {
      float score = 0.0f;
      #pragma unroll
      for (int j = 0; j < NUM_QUERIES_PER_KV; j++) {
        score = MAX(score, tmp_scores[seq_idx * num_heads * max_context_len
                                      + (kv_head_idx * NUM_QUERIES_PER_KV + j) * max_context_len
                                      + token_idx]);
      }

      int64_t score_offset;
      if (token_idx < context_len_left) {
        // Access block table from left
        int64_t physical_block_number = static_cast<int64_t>(*(block_table_left +
          (token_idx / NUM_TOKENS_PER_PAGE_HIGH)));
        score_offset = physical_block_number * unified_page_size + SCORE_BASE_HIGH
                      + token_idx % NUM_TOKENS_PER_PAGE_HIGH * 2;
      } else {
        // Access block table from right
        int64_t physical_block_number = static_cast<int64_t>(*(block_table_right -
          (token_idx - context_len_left) / NUM_TOKENS_PER_PAGE_LOW));
        score_offset = physical_block_number * unified_page_size + SCORE_BASE_LOW
                      + (token_idx - context_len_left) % NUM_TOKENS_PER_PAGE_LOW * 2;
      }

      *reinterpret_cast<float*>(&kv_cache[score_offset]) += score;
    }
  }

  // // if NUM_QUERIES_PER_KV = 1, directly write the softmax logits to score cache.
  // for (int token_idx = threadIdx.x; token_idx < context_len; token_idx += NUM_THREADS) {
  //   float score = logits[token_idx];
  //   int64_t score_offset;
  //   if (token_idx < context_len_left) {
  //     // Access block table from left
  //     int64_t physical_block_number = static_cast<int64_t>(*(block_table_left +
  //       (token_idx / NUM_TOKENS_PER_PAGE_HIGH)));
  //     score_offset = physical_block_number * unified_page_size + SCORE_BASE_HIGH
  //                   + token_idx % NUM_TOKENS_PER_PAGE_HIGH * 2;
  //   } else {
  //     // Access block table from right
  //     int64_t physical_block_number = static_cast<int64_t>(*(block_table_right -
  //       (token_idx - context_len_left) / NUM_TOKENS_PER_PAGE_LOW));
  //     score_offset = physical_block_number * unified_page_size + SCORE_BASE_LOW
  //                   + (token_idx - context_len_left) % NUM_TOKENS_PER_PAGE_LOW * 2;
  //   }

  //   *reinterpret_cast<float*>(&kv_cache[score_offset]) += score;
  // }

}

// Grid: (num_heads, num_seqs, max_num_partitions)
template<
  typename scalar_t,
  int HEAD_SIZE,
  int NUM_QUERIES_PER_KV,
  int BITS_K_HIGH,
  int BITS_V_HIGH,
  int BITS_K_LOW,
  int BITS_V_LOW,
  int NUM_TOKENS_PER_PAGE_HIGH,
  int NUM_TOKENS_PER_PAGE_LOW,
  int THREAD_GROUP_SIZE_V,
  int PARTITION_SIZE>
__global__ void sparse_paged_attention_wrapper(
  const int* __restrict__ slot_ids,         // [num_seqs]
  const int* __restrict__ positions,        // [num_seqs]
  float* __restrict__ exp_sums,             // [num_seqs, num_heads, max_num_partitions]
  float* __restrict__ max_logits,           // [num_seqs, num_heads, max_num_partitions]
  scalar_t* __restrict__ tmp_out,           // [num_seqs, num_heads, max_num_partitions, head_size]
  float* __restrict__ tmp_scores,           // [num_seqs, num_heads, max_context_len]
  const scalar_t* __restrict__ q,           // [num_seqs, num_heads, head_size]
  uint16_t* __restrict__ kv_cache,          // [num_blocks, unified_page_size]
  const int layer_idx,
  const int num_layers,
  const int num_kv_heads,
  const float scale,
  const int* __restrict__ block_tables,     // [num_slots, num_layers, num_kv_heads, max_num_blocks_per_seq]
  const int* __restrict__ kv_len_tables,    // [num_slots, num_layers, num_kv_heads, 2]
  uint64_t* __restrict__ sparsity_tables,      // [num_slots, num_layers, num_heads]
  const int max_context_len,
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes,   // [num_heads]
  const int q_stride,
  const int unified_page_size,
  const float prune_thresh) {
  sparse_paged_attention_kernel<scalar_t, HEAD_SIZE, NUM_QUERIES_PER_KV, BITS_K_HIGH, BITS_V_HIGH, BITS_K_LOW, BITS_V_LOW,
                                NUM_TOKENS_PER_PAGE_HIGH, NUM_TOKENS_PER_PAGE_LOW, THREAD_GROUP_SIZE_V, PARTITION_SIZE>(
    slot_ids, positions, exp_sums, max_logits, tmp_out, tmp_scores,
    q, kv_cache, layer_idx, num_layers, num_kv_heads, scale,
    block_tables, kv_len_tables, sparsity_tables,
    max_context_len, max_num_blocks_per_seq, alibi_slopes, q_stride,
    unified_page_size, prune_thresh);
}

} // namespace vllm

#define LAUNCH_SPARSE_PAGED_ATTENTION(HEAD_SIZE, NUM_QUERIES_PER_KV)                                             \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                                                          \
    ((void*)vllm::sparse_paged_attention_wrapper<T, HEAD_SIZE, NUM_QUERIES_PER_KV, BITS_K_HIGH, BITS_V_HIGH,     \
                                                 BITS_K_LOW, BITS_V_LOW, NUM_TOKENS_PER_PAGE_HIGH,               \
                                                 NUM_TOKENS_PER_PAGE_LOW, THREAD_GROUP_SIZE_V, PARTITION_SIZE>), \
    shared_mem_size);                                                                                            \
  vllm::sparse_paged_attention_wrapper<T, HEAD_SIZE, NUM_QUERIES_PER_KV, BITS_K_HIGH, BITS_V_HIGH,               \
                                       BITS_K_LOW, BITS_V_LOW, NUM_TOKENS_PER_PAGE_HIGH,                         \
                                       NUM_TOKENS_PER_PAGE_LOW, THREAD_GROUP_SIZE_V, PARTITION_SIZE>             \
  <<<grid, block, shared_mem_size, stream>>>(                                                  \
     slot_ids_ptr,                                                                             \
     positions_ptr,                                                                            \
     exp_sums_ptr,                                                                             \
     max_logits_ptr,                                                                           \
     tmp_out_ptr,                                                                              \
     tmp_scores_ptr,                                                                           \
     query_ptr,                                                                                \
     kv_cache_ptr,                                                                             \
     layer_idx,                                                                                \
     num_layers,                                                                               \
     num_kv_heads,                                                                             \
     scale,                                                                                    \
     block_tables_ptr,                                                                         \
     kv_len_tables_ptr,                                                                        \
     sparsity_tables_ptr,                                                                      \
     max_context_len,                                                                          \
     max_num_blocks_per_seq,                                                                   \
     alibi_slopes_ptr,                                                                         \
     q_stride,                                                                                 \
     unified_page_size,                                                                        \
     prune_thresh);                                                                            \
  vllm::reduce_kernel<T, HEAD_SIZE, NUM_QUERIES_PER_KV, BITS_K_HIGH, BITS_V_HIGH,              \
                      BITS_K_LOW, BITS_V_LOW, NUM_TOKENS_PER_PAGE_HIGH,                        \
                      NUM_TOKENS_PER_PAGE_LOW, PARTITION_SIZE>                                 \
  <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                                    \
    out_ptr,                                                                                   \
    exp_sums_ptr,                                                                              \
    max_logits_ptr,                                                                            \
    tmp_out_ptr,                                                                               \
    tmp_scores_ptr,                                                                            \
    block_tables_ptr,                                                                          \
    kv_len_tables_ptr,                                                                         \
    slot_ids_ptr,                                                                              \
    kv_cache_ptr,                                                                              \
    unified_page_size,                                                                         \
    layer_idx,                                                                                 \
    num_layers,                                                                                \
    num_kv_heads,                                                                              \
    max_context_len,                                                                           \
    max_num_blocks_per_seq,                                                                    \
    max_num_partitions);

// TODO(yuwei): Tune THREAD_GROUP_SIZE_V according to quantization bits
template<
  typename T,
  int BITS_K_HIGH,
  int BITS_V_HIGH,
  int BITS_K_LOW,
  int BITS_V_LOW,
  int NUM_TOKENS_PER_PAGE_HIGH,
  int NUM_TOKENS_PER_PAGE_LOW,
  int THREAD_GROUP_SIZE_V>
void sparse_paged_attention_launcher(
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
  torch::Tensor& sparsity_tables,
  int max_context_len,
  float prune_thresh,
  const c10::optional<torch::Tensor>& alibi_slopes)
{
  // printf("[Debug info from sparse_attention_kernels.cu] max_context_len: %d\n", max_context_len);
  const int num_seqs = query.size(0);
  const int num_heads = query.size(1);
  const int head_size = query.size(2);
  const int num_layers = block_tables.size(1);
  const int max_num_blocks_per_seq = block_tables.size(3);
  const int q_stride = query.stride(0);
  const int unified_page_size = kv_cache.size(1);

  assert(head_size % THREAD_GROUP_SIZE_K == 0);
  assert(head_size % WARP_SIZE == 0);
  assert(WARP_SIZE % THREAD_GROUP_SIZE_K == 0);

  assert(num_heads % num_kv_heads == 0);
  const int num_queries_per_kv = num_heads / num_kv_heads;

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes ?
    reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
    : nullptr;

  const int* slot_ids_ptr = slot_ids.data_ptr<int>();
  const int* positions_ptr = positions.data_ptr<int>();
  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  float* tmp_scores_ptr = reinterpret_cast<float*>(tmp_scores.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  uint16_t* kv_cache_ptr = reinterpret_cast<uint16_t*>(kv_cache.data_ptr());
  const int* block_tables_ptr = block_tables.data_ptr<int>();
  const int* kv_len_tables_ptr = kv_len_tables.data_ptr<int>();
  uint64_t* sparsity_tables_ptr = sparsity_tables.data_ptr<uint64_t>();

  assert(WARP_SIZE % THREAD_GROUP_SIZE_V == 0);
  constexpr int NUM_THREAD_GROUPS_V = WARP_SIZE / THREAD_GROUP_SIZE_V;

  // Make PARTITION_SIZE a multiple of the page size.
  constexpr int PARTITION_SIZE = DIVIDE_ROUND_UP(_PARTITION_SIZE, (NUM_TOKENS_PER_PAGE_HIGH * NUM_TOKENS_PER_PAGE_LOW))
                                 * (NUM_TOKENS_PER_PAGE_HIGH * NUM_TOKENS_PER_PAGE_LOW);
  int max_num_partitions = 1;  // no partitioning by default
  if (max_context_len > PARTITION_SIZE) {
    max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE) + 1;
  }

  // Reserve enough (a bit more) shared memory space for logits.
  int logits_size = (PARTITION_SIZE + MAX(NUM_TOKENS_PER_PAGE_HIGH, NUM_TOKENS_PER_PAGE_LOW)
                    + MAX(WARP_SIZE / THREAD_GROUP_SIZE_K, V_VEC_SIZE)) * sizeof(float);
  // int outputs_size = (NUM_THREAD_GROUPS_V / 2) * head_size * sizeof(float);
  // each thread group writes output (of the entire hidden dimension) individually, so we need NUM_THREAD_GROUPS_V * NUM_WARPS entries
  int outputs_size = (NUM_THREAD_GROUPS_V * NUM_THREADS / WARP_SIZE / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  size_t max_shared_mem_size = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
  TORCH_CHECK(shared_mem_size < max_shared_mem_size,
              "Shared memory usage exceeds the limit: ", shared_mem_size, " > ", max_shared_mem_size);

  dim3 block(NUM_THREADS);
  dim3 grid(num_heads, num_seqs, max_num_partitions);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 reduce_grid(num_heads, num_seqs);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  // Dispatch the kernel based on the head size and the number of queries per kv.
  switch (head_size) {
    // case 64:
    //   switch (num_queries_per_kv) {
    //     case 1:
    //       LAUNCH_SPARSE_PAGED_ATTENTION(64, 1);
    //       break;
    //     case 2:
    //       LAUNCH_SPARSE_PAGED_ATTENTION(64, 2);
    //       break;
    //     case 4:
    //       LAUNCH_SPARSE_PAGED_ATTENTION(64, 4);
    //       break;
    //     case 8:
    //       LAUNCH_SPARSE_PAGED_ATTENTION(64, 8);
    //       break;
    //     default:
    //       TORCH_CHECK(false, "Unsupported num_queries_per_kv: ", num_queries_per_kv);
    //       break;
    //   }
    //   break;
    case 128:
      switch (num_queries_per_kv) {
        case 1:
          LAUNCH_SPARSE_PAGED_ATTENTION(128, 1);
          break;
        case 2:
          LAUNCH_SPARSE_PAGED_ATTENTION(128, 2);
          break;
        case 4:
          LAUNCH_SPARSE_PAGED_ATTENTION(128, 4);
          break;
        case 5:
          LAUNCH_SPARSE_PAGED_ATTENTION(128, 5);
          break;
        case 7:
          LAUNCH_SPARSE_PAGED_ATTENTION(128, 7);
          break;
        case 8:
          LAUNCH_SPARSE_PAGED_ATTENTION(128, 8);
          break;
        default:
          TORCH_CHECK(false, "Unsupported num_queries_per_kv: ", num_queries_per_kv);
          break;
      }
      break;
    // case 256:
    //   switch (num_queries_per_kv) {
    //     case 1:
    //       LAUNCH_SPARSE_PAGED_ATTENTION(256, 1);
    //       break;
    //     case 2:
    //       LAUNCH_SPARSE_PAGED_ATTENTION(256, 2);
    //       break;
    //     case 4:
    //       LAUNCH_SPARSE_PAGED_ATTENTION(256, 4);
    //       break;
    //     case 8:
    //       LAUNCH_SPARSE_PAGED_ATTENTION(256, 8);
    //       break;
    //     default:
    //       TORCH_CHECK(false, "Unsupported num_queries_per_kv: ", num_queries_per_kv);
    //       break;
    //   }
    //   break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_LAUNCHER(T, BITS_K_HIGH, BITS_V_HIGH, BITS_K_LOW, BITS_V_LOW,                               \
                      NUM_TOKENS_PER_PAGE_HIGH, NUM_TOKENS_PER_PAGE_LOW, THREAD_GROUP_SIZE_V)            \
  sparse_paged_attention_launcher<T, BITS_K_HIGH, BITS_V_HIGH, BITS_K_LOW, BITS_V_LOW,                      \
                                  NUM_TOKENS_PER_PAGE_HIGH, NUM_TOKENS_PER_PAGE_LOW, THREAD_GROUP_SIZE_V>(  \
    slot_ids,                                                   \
    positions,                                                  \
    out,                                                        \
    exp_sums,                                                   \
    max_logits,                                                 \
    tmp_out,                                                    \
    tmp_scores,                                                 \
    query,                                                      \
    kv_cache,                                                   \
    layer_idx,                                                  \
    num_kv_heads,                                               \
    scale,                                                      \
    block_tables,                                               \
    kv_len_tables,                                              \
    sparsity_tables,                                            \
    max_context_len,                                            \
    prune_thresh,                                               \
    alibi_slopes);

#define CALL_LAUNCHER_QUANT_CONFIG(T)                          \
  if (quant_config == std::vector<int>{8, 8, 8, 8}) {          \
    assert(num_tokens_per_page_high == 12);                    \
    assert(num_tokens_per_page_low == 12);                     \
    assert(thread_group_size_v == 8);                          \
    CALL_LAUNCHER(T, 8, 8, 8, 8, 12, 12, 8);                   \
  } else if (quant_config == std::vector<int>{8, 4, 8, 4}) {   \
    assert(num_tokens_per_page_high == 16);                    \
    assert(num_tokens_per_page_low == 16);                     \
    assert(thread_group_size_v == 8);                          \
    CALL_LAUNCHER(T, 8, 4, 8, 4, 16, 16, 8);                   \
  } else if (quant_config == std::vector<int>{8, 4, 8, 2}) {   \
    assert(num_tokens_per_page_high == 16);                    \
    assert(num_tokens_per_page_low == 19);                     \
    assert(thread_group_size_v == 8);                          \
    CALL_LAUNCHER(T, 8, 4, 8, 2, 16, 19, 8);                   \
  } else if (quant_config == std::vector<int>{8, 4, 4, 4}) {   \
    assert(num_tokens_per_page_high == 16);                    \
    assert(num_tokens_per_page_low == 22);                     \
    assert(thread_group_size_v == 8);                          \
    CALL_LAUNCHER(T, 8, 4, 4, 4, 16, 22, 8);                   \
  } else if (quant_config == std::vector<int>{8, 4, 4, 2}) {   \
    assert(num_tokens_per_page_high == 16);                    \
    assert(num_tokens_per_page_low == 30);                     \
    assert(thread_group_size_v == 8);                          \
    CALL_LAUNCHER(T, 8, 4, 4, 2, 16, 30, 8);                   \
  } else if (quant_config == std::vector<int>{4, 4, 4, 4}) {   \
    assert(num_tokens_per_page_high == 22);                    \
    assert(num_tokens_per_page_low == 22);                     \
    assert(thread_group_size_v == 8);                          \
    CALL_LAUNCHER(T, 4, 4, 4, 4, 22, 22, 8);                   \
  } else if (quant_config == std::vector<int>{4, 4, 4, 2}) {   \
    assert(num_tokens_per_page_high == 22);                    \
    assert(num_tokens_per_page_low == 30);                     \
    assert(thread_group_size_v == 8);                          \
    CALL_LAUNCHER(T, 4, 4, 4, 2, 22, 30, 8);                   \
  } else if (quant_config == std::vector<int>{4, 2, 4, 2}) {   \
    assert(num_tokens_per_page_high == 30);                    \
    assert(num_tokens_per_page_low == 30);                     \
    assert(thread_group_size_v == 8);                          \
    CALL_LAUNCHER(T, 4, 2, 4, 2, 30, 30, 8);                   \
  } else if (quant_config == std::vector<int>{4, 2, 4, 1}) {   \
    assert(num_tokens_per_page_high == 30);                    \
    assert(num_tokens_per_page_low == 35);                     \
    assert(thread_group_size_v == 8);                          \
    CALL_LAUNCHER(T, 4, 2, 4, 1, 30, 35, 8);                   \
  } else if (quant_config == std::vector<int>{4, 1, 4, 1}) {   \
    assert(num_tokens_per_page_high == 35);                    \
    assert(num_tokens_per_page_low == 35);                     \
    assert(thread_group_size_v == 8);                          \
    CALL_LAUNCHER(T, 4, 1, 4, 1, 35, 35, 8);                   \
  } else {                                                     \
    TORCH_CHECK(false, "Unsupported quant config: ", quant_config); \
  }

void sparse_paged_attention(
  torch::Tensor& slot_ids,        // [num_seqs]
  torch::Tensor& positions,       // [num_seqs]
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& exp_sums,        // [num_seqs, num_heads, max_num_partitions]
  torch::Tensor& max_logits,      // [num_seqs, num_heads, max_num_partitions]
  torch::Tensor& tmp_out,         // [num_seqs, num_heads, max_num_partitions, head_size]
  torch::Tensor& tmp_scores,      // [num_seqs, num_heads, max_context_len]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]
  torch::Tensor& kv_cache,        // [num_blocks, unified_page_size]
  int layer_idx,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,    // [num_slots, num_layers, num_kv_heads, max_num_blocks_per_seq]
  torch::Tensor& kv_len_tables,   // [num_slots, num_layers, num_kv_heads, 2]
  torch::Tensor& sparsity_tables, // [num_slots, num_layers, num_kv_heads]
  int max_context_len,
  float prune_thresh,
  int num_bits_k_high,
  int num_bits_v_high,
  int num_bits_k_low,
  int num_bits_v_low,
  int k_vec_size,
  int v_vec_size,
  int num_tokens_per_page_high,
  int num_tokens_per_page_low,
  const c10::optional<torch::Tensor>& alibi_slopes
) {
  TORCH_CHECK(k_vec_size == K_VEC_SIZE, "k_vec_size should be ", K_VEC_SIZE);
  TORCH_CHECK(v_vec_size == V_VEC_SIZE, "v_vec_size should be ", V_VEC_SIZE);

  std::vector<int> quant_config = {num_bits_k_high, num_bits_v_high, num_bits_k_low, num_bits_v_low};
  int lowest_bits = *std::min_element(quant_config.begin(), quant_config.end());
  int thread_group_size_v = get_thread_group_size_v(lowest_bits);
  if (query.dtype() == at::ScalarType::Half) {
    CALL_LAUNCHER_QUANT_CONFIG(uint16_t);
  } else if (query.dtype() == at::ScalarType::BFloat16) {
    CALL_LAUNCHER_QUANT_CONFIG(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

#undef WARP_SIZE
