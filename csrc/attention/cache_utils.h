#pragma once

// attention kernels configs
#define K_VEC_SIZE 4
#define V_VEC_SIZE 2
#define LOG2_K_VEC_SIZE 2
#define LOG2_V_VEC_SIZE 1
#define THREAD_GROUP_SIZE_K 4

struct k_vec_type {
  uint16_t data[K_VEC_SIZE];
};

struct v_vec_type {
  uint16_t data[V_VEC_SIZE];
};

struct quant_meta_type {
  uint16_t scale;
  uint16_t zero_point;
};

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define POWER2_ROUND_UP(a) (1 << (32 - __clz((a) - 1)))
#define POWER2_ROUND_UP_HOST(a) (1 << (32 - __builtin_clz((a) - 1)))

// decide thread_group_size_v
__inline__ int get_thread_group_size_v(int lowest_bits) {
  int thread_group_size_v = 8;
  // if (lowest_bits == 1) {
  //   thread_group_size_v = 8;
  // }
  return thread_group_size_v;
}

constexpr int log2_of_pow2(const int x) {
  if (x == 1) {
    return 0;
  } else if (x == 2) {
    return 1;
  } else if (x == 4) {
    return 2;
  } else if (x == 8) {
    return 3;
  } else if (x == 16) {
    return 4;
  } else if (x == 32) {
    return 5;
  } else if (x == 64) {
    return 6;
  } else if (x == 128) {
    return 7;
  } else if (x == 256) {
    return 8;
  } else {
    return -1;
  }
}

// x % y, y is a power of two integer
__inline__ __device__ int mod_pow2(int x, int logy) {
  return x & ((1 << logy) - 1);
}

template<typename T>
__inline__ __device__ void swap(T* a, T* b) {
  T temp = *a;
  *a = *b;
  *b = temp;
}

// sort tokens by score
__inline__ __device__ void bitonic_sort_ascend(float *values, int* indices, int size) {
  int size_power2 = POWER2_ROUND_UP(size);

  // Padding
  for (int i = size + threadIdx.x; i < size_power2; i += blockDim.x) {
    values[i] = FLT_MAX;
    indices[i] = -1;
  }
  __syncthreads();

  for (int k = 2; k <= size_power2; k *= 2) {
    for (int j = k / 2; j > 0; j /= 2) {
      for (int i = threadIdx.x; i < size_power2; i += blockDim.x) {
        int ixj = i ^ j;
        if (ixj > i) {
          if ((i & k) == 0 && values[i] > values[ixj]) {
            swap<float>(&values[i], &values[ixj]);
            swap<int>(&indices[i], &indices[ixj]);
          }
          if ((i & k) != 0 && values[i] < values[ixj]) {
            swap<float>(&values[i], &values[ixj]);
            swap<int>(&indices[i], &indices[ixj]);
          }
        }
      }
      __syncthreads();
    }
  }
}