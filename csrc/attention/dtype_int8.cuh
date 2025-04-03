#pragma once

#include "attention_generic.cuh"
#include "dtype_float32.cuh"

#include <stdint.h>

namespace vllm {

template<>
struct Vec<uint8_t, 1> {
  using Type = uint8_t;
};
template<>
struct Vec<uint8_t, 2> {
  using Type = uint16_t;
};
template<>
struct Vec<uint8_t, 4> {
  using Type = uint32_t;
};

template<>
struct FloatVec<uint8_t> {
    using Type = float;
};

// From uint8 to float32
inline __device__ float to_float(uint8_t in) {
  return static_cast<float>(in);
}

} // namespace vllm
