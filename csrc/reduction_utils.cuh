/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/reduce_kernel_utils.cuh
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
#pragma once

#include "cuda_compat.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += VLLM_SHFL_XOR_SYNC(val, mask);
  return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (lane < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    T tmp = VLLM_SHFL_XOR_SYNC(val, mask);
    val = MAX(val, tmp);
  }
  return val;
}

template<typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (lane < (blockDim.x / 32.f)) ? shared[lane] : (T)(-FLT_MAX);
  val = warpReduceMax<T>(val);
  return val;
}

template<typename T>
__inline__ __device__ T warpReduceMin(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    T tmp = VLLM_SHFL_XOR_SYNC(val, mask);
    val = MIN(val, tmp);
  }
  return val;
}

template<typename T>
__inline__ __device__ T blockReduceMin(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMin<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (lane < (blockDim.x / 32.f)) ? shared[lane] : (T)(FLT_MAX);
  val = warpReduceMin<T>(val);
  return val;
}

} // namespace vllm

#undef MAX
#undef MIN
