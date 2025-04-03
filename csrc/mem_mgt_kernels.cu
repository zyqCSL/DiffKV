#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cuda_compat.h"
// #include "dispatch_utils.h"
#include "reduction_utils.cuh"

// #include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#define WARP_SIZE 32
#define NUM_THREADS 128
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define POWER2_ROUND_UP(a) (1 << (32 - __clz((a) - 1)))
#define POWER2_ROUND_UP_HOST(a) (1 << (32 - __builtin_clz((a) - 1)))

// prompt phase
namespace vllm {

__global__ void allocate_seqs_kernel(
  const int* __restrict__ slot_ids_ptr,
  const int* __restrict__ num_prompt_tokens_ptr,
  const int* __restrict__ num_blocks_ptr,
  int* __restrict__ block_tables_ptr,
  int* __restrict__ kv_len_tables_ptr,
  int* __restrict__ block_num_tables_ptr,
  const int* __restrict__ free_blocks_ptr,
  int* __restrict__ block_refs_ptr,
  const int batch_start_free_block_pos,
  const int batch_end_free_block_pos,
  const int max_num_blocks_per_head,
  const int num_gpu_blocks) {

  const int seq_idx = blockIdx.x;
  const int layer_idx = blockIdx.y;
  const int head_idx = blockIdx.z;

  // const int num_seqs = gridDim.x;
  const int num_layers = gridDim.y;
  const int num_kv_heads = gridDim.z;
  const int thread_idx = threadIdx.x;

  // NOTE: during prompt we only allocate for high precision
  // blocks used by low precision will be taken on demand from high precision
  const int num_blocks = num_blocks_ptr[seq_idx];
  assert(num_blocks > 0);
  // NOTE: we reserved additional block for low precision, so this is strict less than
  assert(num_blocks < max_num_blocks_per_head);
  const int num_prompt_tokens = num_prompt_tokens_ptr[seq_idx];
  const int head_offset = layer_idx * num_kv_heads + head_idx; // offset of the head in the flattened array

  // compute the first block that belongs to this sequence
  int used_num_blocks_per_head = 0;
  for(int batch_idx = thread_idx; batch_idx < seq_idx; batch_idx += NUM_THREADS) {
    used_num_blocks_per_head += num_blocks_ptr[batch_idx];
  }
  __syncthreads();
  used_num_blocks_per_head = blockReduceSum<int>(used_num_blocks_per_head);
  // if(thread_idx == 0 && layer_idx == 0 && head_idx == 0) {
  //   printf("seq_idx = %d, layer_idx = %d, head_idx = %d, allocated_blocks = %d\n",
  //     seq_idx, layer_idx, head_idx, used_num_blocks_per_head);
  // }
  // start position for high precision
  const int start_free_block_pos = batch_start_free_block_pos +
                                   used_num_blocks_per_head * num_layers * num_kv_heads +
                                   head_offset * num_blocks;
  const int slot_id = slot_ids_ptr[seq_idx];
  assert(slot_id >= 0);
  const int base_table_offset = slot_id * num_layers * num_kv_heads + head_offset;
  // NOTE: * 2 for two precisions
  const int kv_len_tables_offset = base_table_offset * 2;
  const int block_table_offset = base_table_offset * max_num_blocks_per_head;

  // fill block_tables for high precision & increment block_ref
  for(int block_pos = thread_idx; block_pos < num_blocks; block_pos += NUM_THREADS) {
    const int free_block_pos = (start_free_block_pos + block_pos) % num_gpu_blocks;
    const int block_id = free_blocks_ptr[free_block_pos];
    block_tables_ptr[block_table_offset + block_pos] = block_id;
    // NOTE: this assert assumes no block sharing, remove later
    assert(block_refs_ptr[block_id] == 0);
    block_refs_ptr[block_id] += 1;
  }

  // NOTE: we don't need a sync here,
  // because kv_len_tables & block_num_tables are not read beforehands
  // fill kv_len_tables and block_num_tables
  if(thread_idx == 0) {
    kv_len_tables_ptr[kv_len_tables_offset] = num_prompt_tokens;
    block_num_tables_ptr[kv_len_tables_offset] = num_blocks;
  }
}

__global__ void prepare_free_prompt_kernel(
  const int* __restrict__ slot_ids_ptr,
  const int* __restrict__ kv_len_tables_ptr,  // kv_len_tables should be updated during mem management
  int* __restrict__ block_num_tables_ptr,
  const int block_size_high,
  const int block_size_low,
  int* __restrict__ free_block_pos_ptr,
  int* __restrict__ block_nums_ptr,
  const int num_seqs){

  const int seq_group_idx = blockIdx.x;
  const int seq_idx = seq_group_idx * NUM_THREADS + threadIdx.x;
  if(seq_idx < num_seqs) {
    const int layer_idx = blockIdx.y;
    const int head_idx = blockIdx.z;

    // const int num_seqs = gridDim.x;
    const int num_layers = gridDim.y;
    const int num_kv_heads = gridDim.z;

    const int slot_id = slot_ids_ptr[seq_idx];

    // ---------- high precision
    // NOTE: * 2 for two precisions
    const int kv_len_tables_offset = (slot_id * num_layers * num_kv_heads +
                                      layer_idx * num_kv_heads + head_idx) * 2;
    // NOTE: +1 as the first position is used to store the start_pos of free blocks
    // NOTE: no x2 here because during prompt blocks are only allocated for high precision
    // and we only need to record released blocks of high precision
    const int free_block_pos_offset = seq_idx * num_layers * num_kv_heads +
                                      layer_idx * num_kv_heads + head_idx + 1;
    // handles one head
    const int kv_len_high = kv_len_tables_ptr[kv_len_tables_offset];
    const int kv_len_low = kv_len_tables_ptr[kv_len_tables_offset + 1];
    // blocks are only allocated to high precision during prompt
    const int num_allocated_blocks = block_num_tables_ptr[kv_len_tables_offset];
    const int num_blocks_high = DIVIDE_ROUND_UP(kv_len_high, block_size_high);
    const int num_blocks_low = DIVIDE_ROUND_UP(kv_len_low, block_size_low);
    const int num_blocks = num_blocks_high + num_blocks_low;

    assert(num_allocated_blocks >= num_blocks);
    assert(num_blocks > 0);
    free_block_pos_ptr[free_block_pos_offset] = num_allocated_blocks - num_blocks;

    // fill in block_nums
    const int block_nums_offset = (seq_idx * num_layers * num_kv_heads +
                                   layer_idx * num_kv_heads + head_idx) * 2;
    block_nums_ptr[block_nums_offset] = num_blocks_high;
    block_nums_ptr[block_nums_offset + 1] = num_blocks_low;
  }
}

__global__ void free_prompt_kernel(
  const int* __restrict__ slot_ids_ptr,
  int* __restrict__ block_tables_ptr,
  int* __restrict__ block_num_tables_ptr,
  int* __restrict__ free_blocks_ptr,
  const int* __restrict__ free_block_pos_ptr,
  const int* __restrict__ block_nums_ptr,
  int* __restrict__ block_refs_ptr,
  const int max_num_blocks_per_head,
  const int num_gpu_blocks){

  const int seq_idx = blockIdx.x;
  const int layer_idx = blockIdx.y;
  const int head_idx = blockIdx.z;

  // const int num_seqs = gridDim.x;
  const int num_layers = gridDim.y;
  const int num_kv_heads = gridDim.z;
  const int thread_idx = threadIdx.x;

  const int slot_id = slot_ids_ptr[seq_idx];

  // ---------- high precision
  const int table_base_offset = slot_id * num_layers * num_kv_heads +
                                layer_idx * num_kv_heads + head_idx;
  // NOTE: * 2 for two precisions
  const int block_num_tables_offset = table_base_offset * 2;
  // NOTE: index free_block_pos_offset stores the start_pos of released blocks
  // and index free_block_pos_offset + 1 stores the end_pos of released blocks (start_pos of next head)
  // NOTE: no x2 here because prompt blocks are only allocated for high precision
  const int free_block_pos_offset = seq_idx * num_layers * num_kv_heads +
                                    layer_idx * num_kv_heads + head_idx;
  const int block_nums_offset = 2 * free_block_pos_offset;
  const int num_blocks_high = block_nums_ptr[block_nums_offset];
  const int num_blocks_low = block_nums_ptr[block_nums_offset + 1];

  const int start_free_block_pos = free_block_pos_ptr[free_block_pos_offset];
  const int end_free_block_pos = free_block_pos_ptr[free_block_pos_offset + 1];

  const int num_released_blocks = end_free_block_pos - start_free_block_pos;
  assert(num_released_blocks >= 0);

  // // DEBUG
  // if(block_num_tables_ptr[block_num_tables_offset] <= num_released_blocks) {
  //   printf("Error: [%d, %d, %d], slot_id = %d, num_released_blocks = %d, start =%d, end = %d, offset = %d, block_num_tables_val = %d, kv_len = %d, num_released_blocks_debug = %d\n",
  //     seq_idx, layer_idx, head_idx,
  //     slot_id, num_released_blocks,
  //     start_free_block_pos, end_free_block_pos, free_block_pos_offset,
  //     block_num_tables_ptr[block_num_tables_offset]);
  // }

  // NOTE: IMPORTANT! We must release freed blocks before moving low precision blocks to the righthand side
  // otherwise for long sequences, the low precision blocks will overlap with the positions 
  // of the freed blocks in the block table, causing memory leaks and double freed blocks
  if(num_released_blocks > 0) {
    // compression required
    const int num_allocated_blocks = block_num_tables_ptr[block_num_tables_offset];
    const int num_blocks = num_blocks_high + num_blocks_low;
    
    // DEBUG
    // if(num_allocated_blocks != num_released_blocks + num_blocks) {
    //   printf("[Debug info from mem_mgt_kernels] free_prompt, num_allocated_blocks: %d, num_released_blocks: %d, num_blocks_high: %d, num_blocks_low: %d\n", 
    //       num_allocated_blocks, num_released_blocks, num_blocks_high, num_blocks_low);
    // }
    
    assert(num_allocated_blocks == num_released_blocks + num_blocks);
    // NOTE: blocks of high precision grows from left to right
    const int block_tables_offset = table_base_offset * max_num_blocks_per_head +
                                    num_allocated_blocks - num_released_blocks;
    // put all pruned blocks back to free_blocks
    // NOTE: here we assume that each block's ref counter equals 1 during prompt
    for(int released_block_offset = thread_idx;
      released_block_offset < num_released_blocks;
      released_block_offset += NUM_THREADS) {
      const int block_id = block_tables_ptr[released_block_offset + block_tables_offset];
      const int free_block_pos = (start_free_block_pos + released_block_offset) % num_gpu_blocks;
      free_blocks_ptr[free_block_pos] = block_id;

      // // DEBUG
      // if(block_id == 220865) {
      //   printf("[Debug info from mem_mgt_kernels] free_prompt release seq_idx: %d, layer_idx: %d, head_idx: %d, block_id: %d, ref: %d, released_block_offset: %d, num_blocks_high: %d, num_blocks_low: %d\n",
      //     seq_idx, layer_idx, head_idx, block_id, block_refs_ptr[block_id], released_block_offset, num_blocks_high, num_blocks_low);
      // }

      // update block reference counter
      assert(block_refs_ptr[block_id] == 1);
      block_refs_ptr[block_id] = 0;
    }
    // assert(block_num_tables_ptr[block_num_tables_offset] > 0);
  }

  // NOTE: we need a barrier here since afterwards we write the block table
  __syncthreads();

  // NOTE: move the quantized blocks to the righthand side after releasing freed blocks
  if (num_blocks_low > 0) {
    // move the low precision blocks to the righthand side of the block table
    // low precision grows from right to left
    const int block_tables_high_offset = table_base_offset * max_num_blocks_per_head;
    const int block_tables_low_offset = block_tables_high_offset + max_num_blocks_per_head - 1;

    // NOTE: IMPORTATANT! Here we swap the low precision blocks and the righthandside blocks symmetrically
    // We can't directly assign block_tables_ptr[dst_offset] = block_tables_ptr[src_offset],
    // as this will cause overlaps when sequence is long. 
    // Consider a block_table with 50 blocks, and the high precision has 5 blocks, low precision has 40 blocks
    // *** slot  0----4 5-----------------44 45----49
    // *** block 0----4 5-----------------44 ********
    // If we do block_tables_ptr[dst_offset] = block_tables_ptr[src_offset], we'd expect
    // *** slot  0----4 5-----------------44 45----49
    // *** block 0----4 ******** 5-----------------44  as the final layout.
    // Assume we have 30 threads moving the data in parallel, 
    // In the 1st round, we move blocks 5-34 to slots 10-44. 
    // However, after this round, the original blocks saved in slots 35-44 are overwritten and lost.
    // As a result, in the 2nd round, we can no longer move blocks 35-44 to the desired locations, 
    // since what we will move are actually blocks 25-34
    // To solve this issue, instead of moving the blocks, we swap the blocks symmetrically as follows
    // In the above synethetic example, the layout after swapping should be
    // *** slot  0----4 5-----------------44 45----49
    // *** block 0----4 ******* 44 -----------7, 6, 5.
    const int num_slots_to_mid = (max_num_blocks_per_head - num_blocks_high) / 2;
    const int num_swaps = MIN(num_blocks_low, num_slots_to_mid);
    for(int low_block_offset = thread_idx;
      low_block_offset < num_swaps;
      low_block_offset += NUM_THREADS) {
      // For example. max_num_blocks_per_head = k, and base ptr of the head's block table is p
      // the head has h blocks for high precision, and 3 blocks for low precision
      // Then the 0th, 1st, 2nd block of high precision,
      // at addr p + h, p + h + 1, p + h + 2 are moved to addr p + k - 1, p + k - 2, p + k - 3, resp.
      const int src_offset = block_tables_high_offset + num_blocks_high + low_block_offset;
      const int dst_offset = block_tables_low_offset - low_block_offset;  
      // swap src & dst    
      const int src_block_id = block_tables_ptr[src_offset];
      block_tables_ptr[src_offset] = block_tables_ptr[dst_offset];
      block_tables_ptr[dst_offset] = src_block_id;

      // // DEBUG
      // if(block_tables_ptr[src_offset] == 220865) {
      //   int _tmp_block_id = block_tables_ptr[src_offset];
      //   printf("[Debug info from mem_mgt_kernels] free_prompt to low precision seq_idx: %d, layer_idx: %d, head_idx: %d, block_id: %d, ref: %d, low_block_offset: %d, num_blocks_high: %d, num_blocks_low: %d\n",
      //     seq_idx, layer_idx, head_idx, _tmp_block_id, block_refs_ptr[_tmp_block_id], low_block_offset, num_blocks_high, num_blocks_low);
      // }

    }
  }

  

  // NOTE: we need wait for all threads before updating block_num_tables,
  // as block_num_tables are read beforehands
  __syncthreads();

  // update block_num_tables in the end
  if(thread_idx == 0) {
    block_num_tables_ptr[block_num_tables_offset] = num_blocks_high;
    block_num_tables_ptr[block_num_tables_offset + 1] = num_blocks_low;
  }
}

} // namespace vllm

void allocate_seqs(
  torch::Tensor& slot_ids,
  torch::Tensor& num_prompt_tokens,
  torch::Tensor& num_blocks,  // number of blocks per head
  torch::Tensor& block_tables,
  torch::Tensor& kv_len_tables,
  torch::Tensor& block_num_tables,
  torch::Tensor& free_blocks,
  torch::Tensor& block_refs,
  const int start_block_pos,
  const int end_block_pos) {

  const int num_seqs = slot_ids.size(0);
  const int num_gpu_blocks = free_blocks.size(0);
  const int num_layers = block_tables.size(1);
  const int num_kv_heads = block_tables.size(2);
  const int max_num_blocks_per_head = block_tables.size(3);

  dim3 grid(num_seqs, num_layers, num_kv_heads);
  dim3 block(NUM_THREADS);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::allocate_seqs_kernel<<<grid, block, 0, stream>>>(
    slot_ids.data_ptr<int>(),
    num_prompt_tokens.data_ptr<int>(),
    num_blocks.data_ptr<int>(),
    block_tables.data_ptr<int>(),
    kv_len_tables.data_ptr<int>(),
    block_num_tables.data_ptr<int>(),
    free_blocks.data_ptr<int>(),
    block_refs.data_ptr<int>(),
    start_block_pos,
    end_block_pos,
    max_num_blocks_per_head,
    num_gpu_blocks);
}

void prepare_free_prompt(
  torch::Tensor& slot_ids,
  torch::Tensor& kv_len_tables,   // kv_len_tables should be updated in the attention kernel
  torch::Tensor& block_num_tables,
  const int block_size_high,
  const int block_size_low,
  torch::Tensor& free_block_pos,
  torch::Tensor& block_nums) {

  const int num_seqs = slot_ids.size(0);
  const int num_seq_groups = DIVIDE_ROUND_UP(num_seqs, NUM_THREADS);
  const int num_layers = kv_len_tables.size(1);
  const int num_kv_heads = kv_len_tables.size(2);
  dim3 grid(num_seq_groups, num_layers, num_kv_heads);
  dim3 block(NUM_THREADS);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::prepare_free_prompt_kernel<<<grid, block, 0, stream>>>(
    slot_ids.data_ptr<int>(),
    kv_len_tables.data_ptr<int>(),
    block_num_tables.data_ptr<int>(),
    block_size_high,
    block_size_low,
    free_block_pos.data_ptr<int>(),
    block_nums.data_ptr<int>(),
    num_seqs);
}

void free_prompt(
  torch::Tensor& slot_ids,
  torch::Tensor& block_tables,
  torch::Tensor& block_num_tables,
  torch::Tensor& free_blocks,
  torch::Tensor& free_block_pos,
  torch::Tensor& block_nums,
  torch::Tensor& block_refs) {

  const int num_seqs = slot_ids.size(0);
  const int num_layers = block_tables.size(1);
  const int num_kv_heads = block_tables.size(2);
  const int max_num_blocks_per_head = block_tables.size(3);
  const int num_gpu_blocks = free_blocks.size(0);
  dim3 grid(num_seqs, num_layers, num_kv_heads);
  dim3 block(NUM_THREADS);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::free_prompt_kernel<<<grid, block, 0, stream>>>(
    slot_ids.data_ptr<int>(),
    block_tables.data_ptr<int>(),
    block_num_tables.data_ptr<int>(),
    free_blocks.data_ptr<int>(),
    free_block_pos.data_ptr<int>(),
    block_nums.data_ptr<int>(),
    block_refs.data_ptr<int>(),
    max_num_blocks_per_head,
    num_gpu_blocks);
}


// decode phase
namespace vllm {

__global__ void prepare_append_seqs_kernel(
  const int* __restrict__ slot_ids_ptr,
  const int* __restrict__ kv_len_tables_ptr,  // we don't update kv_len_tables during mem management
  const int* __restrict__ block_num_tables_ptr,  // we don't update block_num_tables during mem management
  const int kbits_high,
  const int vbits_high,
  const int kbits_low,
  const int vbits_low,
  const int block_size_high,
  const int block_size_low,
  int* __restrict__ free_block_pos_ptr,
  const int num_seqs) {

  const int seq_group_idx = blockIdx.x;
  const int seq_idx = seq_group_idx * NUM_THREADS + threadIdx.x;
  if(seq_idx < num_seqs) {
    const int num_layers = gridDim.y;
    const int num_kv_heads = gridDim.z;

    const int layer_idx = blockIdx.y;
    const int head_idx = blockIdx.z;

    const int slot_id = slot_ids_ptr[seq_idx];
    int kv_len_tables_offset = (slot_id * num_layers * num_kv_heads +
                                layer_idx * num_kv_heads + head_idx) * 2;
    // NOTE: +1 as the first position is used to store the start_pos of free blocks
    // NOTE: *2 for two precisions
    int free_block_pos_offset = (seq_idx * num_layers * num_kv_heads +
                                 layer_idx * num_kv_heads + head_idx) * 2 + 1;
    // handles one head
    // high precision
    int kv_len = kv_len_tables_ptr[kv_len_tables_offset];
    int num_allocated_blocks = block_num_tables_ptr[kv_len_tables_offset];
    int block_size = block_size_high;
    if(kv_len >= num_allocated_blocks * block_size) {
      // TODO: remove the assert later
      assert(kv_len < (num_allocated_blocks + 1) * block_size);
      free_block_pos_ptr[free_block_pos_offset] = 1;
    } else {
      free_block_pos_ptr[free_block_pos_offset] = 0;
    }

    // low precision
    kv_len_tables_offset += 1;
    free_block_pos_offset += 1;
    
    if(kbits_low > 0 && vbits_low > 0 && 
       (kbits_high != kbits_low || vbits_high != vbits_low)) {
      kv_len = kv_len_tables_ptr[kv_len_tables_offset];
      num_allocated_blocks = block_num_tables_ptr[kv_len_tables_offset];

      block_size = block_size_low;
      // if low precision is not enabled, kv_len should always be 0
      if(kv_len >= num_allocated_blocks * block_size) {
        // NOTE: this judgement must be conditioned on kbits_high != kbits_low || vbits_high != vbits_low
        // otherwise kv_len and num_allocated_blocks are both 0 when low precision is disabled
        // forcing a block to be allocated each time
        // TODO: remove the assert later
        assert(kv_len < (num_allocated_blocks + 1) * block_size);
        free_block_pos_ptr[free_block_pos_offset] = 1;
      } else {
        free_block_pos_ptr[free_block_pos_offset] = 0;
      }
    } else {
      free_block_pos_ptr[free_block_pos_offset] = 0;
    }

  }
}

__global__ void append_seqs_kernel(
  const int* __restrict__ slot_ids_ptr,
  int* __restrict__ block_tables_ptr,
  int* __restrict__ block_num_tables_ptr,
  const int* __restrict__ free_blocks_ptr,
  int* __restrict__ block_refs_ptr,
  const int* __restrict__ free_block_pos_ptr,
  const int max_num_blocks_per_head,
  const int num_gpu_blocks,
  const int num_seqs) {

  const int seq_group_idx = blockIdx.x;
  const int seq_id = seq_group_idx * NUM_THREADS + threadIdx.x;
  if(seq_id < num_seqs) {
    const int num_layers = gridDim.y;
    const int num_kv_heads = gridDim.z;

    const int layer_idx = blockIdx.y;
    const int head_idx = blockIdx.z;

    const int slot_id = slot_ids_ptr[seq_id];

    // high precision
    // index to the block_num_tables
    const int table_base_offset = slot_id * num_layers * num_kv_heads +
                                  layer_idx * num_kv_heads + head_idx;
    int block_num_tables_offset = table_base_offset * 2;
    int num_allocated_blocks = block_num_tables_ptr[block_num_tables_offset];
    // index to free_block_pos containing the start_pos and end_pos of free_blocks
    // NOTE: * 2 for two precisions
    int free_block_pos_offset = (seq_id * num_layers * num_kv_heads +
                                 layer_idx * num_kv_heads + head_idx) * 2;
    int start_free_block_pos = free_block_pos_ptr[free_block_pos_offset];
    int end_free_block_pos = free_block_pos_ptr[free_block_pos_offset + 1];
    // use free_block_pos to get the appended block id
    if(end_free_block_pos != start_free_block_pos) {
      assert(end_free_block_pos == start_free_block_pos + 1);
      // NOTE: end_free_block_pos might exceed the length of free_blocks
      const int block_id = free_blocks_ptr[start_free_block_pos % num_gpu_blocks];

      // // DEBUG
      // if(block_refs_ptr[block_id] != 0) {
      //   printf("[DEBUG info from mem_mgt_kernels] append_seqs_kernel seq_idx: %d, layer_idx: %d, head_idx: %d, block_id: %d, ref: %d, free_block_pos = %d\n", 
      //     seq_id, layer_idx, head_idx, block_id, block_refs_ptr[block_id], start_free_block_pos % num_gpu_blocks);
      // }

      // NOTE: newly added block should not be shared
      assert(block_refs_ptr[block_id] == 0);
      block_refs_ptr[block_id] += 1;
      block_num_tables_ptr[block_num_tables_offset] += 1;
      // NOTE: high precision grows from left to right
      const int block_tables_offset = table_base_offset * max_num_blocks_per_head +
                                      num_allocated_blocks;
      block_tables_ptr[block_tables_offset] = block_id;

      // // DEBUG
      // if(block_id == 127253) {
      //   printf("[Debug info from mem_mgt_kernels] append_seqs high_precision seq_idx: %d, layer_idx: %d, head_idx: %d, block_id: %d, ref: %d, block_nums_high: %d, block_nums_low: %d\n",
      //     seq_id, layer_idx, head_idx, block_id, block_refs_ptr[block_id], num_allocated_blocks, 
      //     block_num_tables_ptr[block_num_tables_offset], block_num_tables_ptr[block_num_tables_offset + 1]);
      // }

    }

    // low precision
    block_num_tables_offset += 1;
    num_allocated_blocks = block_num_tables_ptr[block_num_tables_offset];
    free_block_pos_offset += 1;
    start_free_block_pos = free_block_pos_ptr[free_block_pos_offset];
    end_free_block_pos = free_block_pos_ptr[free_block_pos_offset + 1];
    // use free_block_pos to get the appended block id
    if(end_free_block_pos != start_free_block_pos) {
      assert(end_free_block_pos == start_free_block_pos + 1);
      // NOTE: end_free_block_pos might exceed the length of free_blocks
      const int block_id = free_blocks_ptr[start_free_block_pos % num_gpu_blocks];
      
      // // DEBUG
      // if(block_refs_ptr[block_id] != 0) {
      //   printf("[DEBUG info from mem_mgt_kernels] append_seqs_kernel seq_idx: %d, layer_idx: %d, head_idx: %d, block_id %d, ref: %d, free_block_pos: %d\n", 
      //     seq_id, layer_idx, head_idx, block_id, block_refs_ptr[block_id], start_free_block_pos % num_gpu_blocks);
      // }

      // NOTE: newly added block should not be shared
      assert(block_refs_ptr[block_id] == 0);
      block_refs_ptr[block_id] += 1;
      block_num_tables_ptr[block_num_tables_offset] += 1;
      // NOTE: low precision grows from right to left
      const int block_tables_offset = table_base_offset * max_num_blocks_per_head +
                                      max_num_blocks_per_head - num_allocated_blocks - 1;
      block_tables_ptr[block_tables_offset] = block_id;

      // // DEBUG
      // if(block_id == 127253) {
      //   printf("[Debug info from mem_mgt_kernels] append_seqs low_precision seq_idx: %d, layer_idx: %d, head_idx: %d, block_id: %d, ref: %d, block_nums_high: %d, block_nums_low: %d\n\n",
      //     seq_id, layer_idx, head_idx, block_id, block_refs_ptr[block_id], 
      //     block_num_tables_ptr[block_num_tables_offset], block_num_tables_ptr[block_num_tables_offset + 1]);
      // }
    }
  }
}

} // namespace vllm

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
  torch::Tensor& free_block_pos) {

  const int num_seqs = slot_ids.size(0);
  const int num_seq_groups = DIVIDE_ROUND_UP(num_seqs, NUM_THREADS);
  const int num_layers = kv_len_tables.size(1);
  const int num_kv_heads = kv_len_tables.size(2);
  dim3 grid(num_seq_groups, num_layers, num_kv_heads);
  dim3 block(NUM_THREADS);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::prepare_append_seqs_kernel<<<grid, block, 0, stream>>>(
    slot_ids.data_ptr<int>(),
    kv_len_tables.data_ptr<int>(),
    block_num_tables.data_ptr<int>(),
    kbits_high,
    vbits_high,
    kbits_low,
    vbits_low,
    block_size_high,
    block_size_low,
    free_block_pos.data_ptr<int>(),
    num_seqs);
}

void append_seqs(
  torch::Tensor& slot_ids,
  torch::Tensor& block_tables,
  torch::Tensor& kv_len_tables,
  torch::Tensor& block_num_tables,
  torch::Tensor& free_blocks,
  torch::Tensor& block_refs,
  torch::Tensor& free_block_pos) {

  const int num_seqs = slot_ids.size(0);
  const int num_seq_groups = DIVIDE_ROUND_UP(num_seqs, NUM_THREADS);
  const int num_layers = block_tables.size(1);
  const int num_kv_heads = block_tables.size(2);
  const int max_num_blocks_per_head = block_tables.size(3);
  const int num_gpu_blocks = free_blocks.size(0);
  dim3 grid(num_seq_groups, num_layers, num_kv_heads);
  dim3 block(NUM_THREADS);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::append_seqs_kernel<<<grid, block, 0, stream>>>(
    slot_ids.data_ptr<int>(),
    block_tables.data_ptr<int>(),
    block_num_tables.data_ptr<int>(),
    free_blocks.data_ptr<int>(),
    block_refs.data_ptr<int>(),
    free_block_pos.data_ptr<int>(),
    max_num_blocks_per_head,
    num_gpu_blocks,
    num_seqs);
}


// sequence complete with block sharing disabled
namespace vllm {

__global__ void prepare_free_seqs_kernel(
  const int* __restrict__ slot_ids_ptr,
  const int* __restrict__ block_num_tables_ptr,
  int* __restrict__ free_block_pos_ptr,
  const int num_seqs){

  const int seq_group_idx = blockIdx.x;
  const int seq_idx = seq_group_idx * NUM_THREADS + threadIdx.x;
  if(seq_idx < num_seqs) {
    const int layer_idx = blockIdx.y;
    const int head_idx = blockIdx.z;

    // const int num_seqs = gridDim.x;
    const int num_layers = gridDim.y;
    const int num_kv_heads = gridDim.z;

    const int slot_id = slot_ids_ptr[seq_idx];

    // NOTE: *2 for two precisions
    const int block_num_tables_offset = (slot_id * num_layers * num_kv_heads +
                                         layer_idx * num_kv_heads + head_idx) * 2;
    // NOTE: +1 as the first position is used to store the start_pos of free blocks
    const int free_block_pos_offset = (seq_idx * num_layers * num_kv_heads +
                                       layer_idx * num_kv_heads + head_idx) * 2 + 1;
    // handles one head
    // high precision
    free_block_pos_ptr[free_block_pos_offset] = block_num_tables_ptr[block_num_tables_offset];
    // low precision, block_num_tables for low precision should be set 0 in allocate & attn kernels
    free_block_pos_ptr[free_block_pos_offset + 1] = block_num_tables_ptr[block_num_tables_offset + 1];
  }
}

__global__ void free_seqs_kernel(
  const int* __restrict__ slot_ids_ptr,
  int* __restrict__ block_tables_ptr,
  int* __restrict__ block_num_tables_ptr,
  int* __restrict__ free_blocks_ptr,
  const int* __restrict__ free_block_pos_ptr,
  int* __restrict__ block_refs_ptr,
  const int max_num_blocks_per_head,
  const int num_gpu_blocks){

  const int seq_idx = blockIdx.x;
  const int layer_idx = blockIdx.y;
  const int head_idx = blockIdx.z;

  // const int num_seqs = gridDim.x;
  const int num_layers = gridDim.y;
  const int num_kv_heads = gridDim.z;
  const int thread_idx = threadIdx.x;

  const int slot_id = slot_ids_ptr[seq_idx];
  const int table_base_offset = slot_id * num_layers * num_kv_heads +
                                layer_idx * num_kv_heads + head_idx;

  //---------- high precision
  // NOTE: *2 for two precisions
  const int block_num_tables_offset = table_base_offset * 2;
  // NOTE: index free_block_pos_offset stores the start_pos of pruned blocks
  // and index free_block_pos_offset + 1 stores the end_pos of pruned blocks
  const int free_block_pos_offset = (seq_idx * num_layers * num_kv_heads +
                                     layer_idx * num_kv_heads + head_idx) * 2;
  int start_free_block_pos = free_block_pos_ptr[free_block_pos_offset];
  // const int end_free_block_pos = free_block_pos_ptr[free_block_pos_offset + 1];
  int num_allocated_blocks = block_num_tables_ptr[block_num_tables_offset];
  // high precisions grows from left to right
  int block_tables_offset = table_base_offset * max_num_blocks_per_head;
  // put all freed blocks back to free_blocks
  for(int freed_block_offset = thread_idx;
      freed_block_offset < num_allocated_blocks;
      freed_block_offset += NUM_THREADS) {
      // read blocks from left to right
      const int block_id = block_tables_ptr[freed_block_offset + block_tables_offset];
      const int free_block_pos = (start_free_block_pos + freed_block_offset) % num_gpu_blocks;
      free_blocks_ptr[free_block_pos] = block_id;
      // update block reference counter
      // here we assume blocks are never shared
      assert(block_refs_ptr[block_id] == 1);
      block_refs_ptr[block_id] = 0;
  }

  //---------- low precision
  start_free_block_pos = free_block_pos_ptr[free_block_pos_offset + 1];
  num_allocated_blocks = block_num_tables_ptr[block_num_tables_offset + 1];
  // low precision grows from right to left
  block_tables_offset += max_num_blocks_per_head - 1;
  for(int freed_block_offset = thread_idx;
      freed_block_offset < num_allocated_blocks;
      freed_block_offset += NUM_THREADS) {
      // read blocks from right to left
      const int block_id = block_tables_ptr[block_tables_offset - freed_block_offset];
      const int free_block_pos = (start_free_block_pos + freed_block_offset) % num_gpu_blocks;
      free_blocks_ptr[free_block_pos] = block_id;

      // DEBUG
      // if(block_refs_ptr[block_id] != 1) {
      //   printf("[Debug info from mem_mgt_kernels] seq_idx: %d, layer_idx: %d, head_idx: %d, block_id: %d, ref: %d, freed_block_offset: %d, num_allocated_blocks: %d\n",
      //     seq_idx, layer_idx, head_idx, block_id, block_refs_ptr[block_id], freed_block_offset, num_allocated_blocks);
      // }

      // update block reference counter
      // here we assume blocks are never shared
      assert(block_refs_ptr[block_id] == 1);
      block_refs_ptr[block_id] = 0;
  }

}

} // namespace vllm


// sequence complete with block sharing enabled
namespace vllm {
__global__ void prepare_free_seqs_block_sharing_kernel(
  const int* __restrict__ slot_ids_ptr,
  const int* __restrict__ block_tables_ptr,
  const int* __restrict__ block_num_tables_ptr,
  const int* __restrict__ block_refs_ptr,
  int* __restrict__ free_block_pos_ptr,
  const int max_num_blocks_per_head,
  const int block_size){

  const int seq_idx = blockIdx.x;
  const int layer_idx = blockIdx.y;
  const int head_idx = blockIdx.z;

  // const int num_seqs = gridDim.x;
  const int num_layers = gridDim.y;
  const int num_kv_heads = gridDim.z;
  const int thread_idx = threadIdx.x;

  const int slot_id = slot_ids_ptr[seq_idx];
  const int table_base_offset = slot_id * num_layers * num_kv_heads +
                                layer_idx * num_kv_heads + head_idx;

  // ----------- high precision
  int block_num_tables_offset = table_base_offset * 2;
  // NOTE: +1 as the first position is used to store the start_pos of free blocks
  int free_block_pos_offset = (seq_idx * num_layers * num_kv_heads +
                              layer_idx * num_kv_heads + head_idx) * 2 + 1;
  int block_tables_offset = table_base_offset * max_num_blocks_per_head;
  int num_allocated_blocks = block_num_tables_ptr[block_num_tables_offset];

  int num_freed_blocks = 0;
  for(int block_offset = thread_idx; block_offset < num_allocated_blocks; block_offset += NUM_THREADS) {
    const int block_id = block_tables_ptr[block_tables_offset + block_offset];
    num_freed_blocks += (block_refs_ptr[block_id] == 1);
  }
  __syncthreads();
  num_freed_blocks = blockReduceSum<int>(num_freed_blocks);
  // handles one head
  free_block_pos_ptr[free_block_pos_offset] = num_freed_blocks;

  // ----------- low precision
  block_num_tables_offset += 1;
  free_block_pos_offset += 1;
  // low precision grows from right to left
  block_tables_offset += max_num_blocks_per_head - 1;
  num_allocated_blocks = block_num_tables_ptr[block_num_tables_offset];

  num_freed_blocks = 0;
  for(int block_offset = thread_idx; block_offset < num_allocated_blocks; block_offset += NUM_THREADS) {
    // read blocks from right to left
    const int block_id = block_tables_ptr[block_tables_offset - block_offset];
    num_freed_blocks += (block_refs_ptr[block_id] == 1);
  }
  __syncthreads();
  num_freed_blocks = blockReduceSum<int>(num_freed_blocks);
  // handles one head
  free_block_pos_ptr[free_block_pos_offset] = num_freed_blocks;
}

__global__ void free_seqs_block_sharing_kernel(
  const int* __restrict__ slot_ids_ptr,
  int* __restrict__ block_tables_ptr,
  int* __restrict__ block_num_tables_ptr,
  int* __restrict__ free_blocks_ptr,
  const int* __restrict__ free_block_pos_ptr,
  int* __restrict__ block_refs_ptr,
  const int max_num_blocks_per_head,
  const int num_gpu_blocks){

  const int thread_idx = threadIdx.x;
  // NOTE: we only use one thread to handle each head,
  // because blocks with reference counter > 1 might appear in arbitary positions in the seq
  // and having multiple threads need another sync to determine the memory region to write
  // free blocks each thread is responsbile for
  if(thread_idx == 0) {
    const int seq_idx = blockIdx.x;
    const int layer_idx = blockIdx.y;
    const int head_idx = blockIdx.z;

    // const int num_seqs = gridDim.x;
    const int num_layers = gridDim.y;
    const int num_kv_heads = gridDim.z;


    const int slot_id = slot_ids_ptr[seq_idx];
    const int table_base_offset = slot_id * num_layers * num_kv_heads +
                                  layer_idx * num_kv_heads + head_idx;

    // ------------ high precision
    int block_num_tables_offset = table_base_offset * 2;
    // NOTE: index free_block_pos_offset stores the start_pos of pruned blocks
    // and index free_block_pos_offset + 1 stores the end_pos of pruned blocks
    int free_block_pos_offset = (seq_idx * num_layers * num_kv_heads +
                                layer_idx * num_kv_heads + head_idx) * 2;
    int free_block_pos = free_block_pos_ptr[free_block_pos_offset];
    // const int end_free_block_pos = free_block_pos_ptr[free_block_pos_offset + 1];

    int num_allocated_blocks = block_num_tables_ptr[block_num_tables_offset];
    int block_tables_offset = table_base_offset * max_num_blocks_per_head;
    // put all freed blocks back to free_blocks
    for(int freed_block_offset = block_tables_offset;
        freed_block_offset < num_allocated_blocks;
        freed_block_offset += 1) {
        const int block_id = block_tables_ptr[freed_block_offset];
        if (block_refs_ptr[block_id] == 1) {
          // block can be freed
          free_blocks_ptr[free_block_pos % num_gpu_blocks] = block_id;
          free_block_pos += 1;
        } else {
          // block still referenced by other seq
          block_refs_ptr[block_id] -= 1;
        }
    }

    // ------------ low precision
    block_num_tables_offset += 1;
    // NOTE: index free_block_pos_offset stores the start_pos of pruned blocks
    // and index free_block_pos_offset + 1 stores the end_pos of pruned blocks
    free_block_pos_offset += 1;
    free_block_pos = free_block_pos_ptr[free_block_pos_offset];
    // const int end_free_block_pos = free_block_pos_ptr[free_block_pos_offset + 1];
    num_allocated_blocks = block_num_tables_ptr[block_num_tables_offset];
    block_tables_offset += max_num_blocks_per_head - 1;
    // put all freed blocks back to free_blocks
    for(int freed_block_offset = block_tables_offset;
        freed_block_offset > block_tables_offset - num_allocated_blocks;
        freed_block_offset -= 1) {
        const int block_id = block_tables_ptr[freed_block_offset];
        if (block_refs_ptr[block_id] == 1) {
          // block can be freed
          free_blocks_ptr[free_block_pos % num_gpu_blocks] = block_id;
          free_block_pos += 1;
        } else {
          // block still referenced by other seq
          block_refs_ptr[block_id] -= 1;
        }
    }

  }
}

} // namespace vllm


void prepare_free_seqs(
  torch::Tensor& slot_ids,
  torch::Tensor& block_num_tables,
  torch::Tensor& free_block_pos) {

  const int num_seqs = slot_ids.size(0);
  const int num_seq_groups = DIVIDE_ROUND_UP(num_seqs, NUM_THREADS);
  const int num_layers = block_num_tables.size(1);
  const int num_kv_heads = block_num_tables.size(2);
  dim3 grid(num_seq_groups, num_layers, num_kv_heads);
  dim3 block(NUM_THREADS);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::prepare_free_seqs_kernel<<<grid, block, 0, stream>>>(
    slot_ids.data_ptr<int>(),
    block_num_tables.data_ptr<int>(),
    free_block_pos.data_ptr<int>(),
    num_seqs);
}

void free_seqs(
  torch::Tensor& slot_ids,
  torch::Tensor& block_tables,
  torch::Tensor& block_num_tables,
  torch::Tensor& free_blocks,
  torch::Tensor& free_block_pos,
  torch::Tensor& block_refs) {

  const int num_seqs = slot_ids.size(0);
  const int num_layers = block_tables.size(1);
  const int num_kv_heads = block_tables.size(2);
  const int max_num_blocks_per_head = block_tables.size(3);
  const int num_gpu_blocks = free_blocks.size(0);
  dim3 grid(num_seqs, num_layers, num_kv_heads);
  dim3 block(NUM_THREADS);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::free_seqs_kernel<<<grid, block, 0, stream>>>(
    slot_ids.data_ptr<int>(),
    block_tables.data_ptr<int>(),
    block_num_tables.data_ptr<int>(),
    free_blocks.data_ptr<int>(),
    free_block_pos.data_ptr<int>(),
    block_refs.data_ptr<int>(),
    max_num_blocks_per_head,
    num_gpu_blocks);
}


#undef WARP_SIZE
#undef NUM_THREADS
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
#undef POWER2_ROUND_UP
#undef POWER2_ROUND_UP_HOST
#undef QUNAT_BLOCK_SIZE
