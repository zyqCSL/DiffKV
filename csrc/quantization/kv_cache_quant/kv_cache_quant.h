#pragma once

#include "uint8_quant.cuh"
#include "uint4_quant.cuh"
#include "uint2_quant.cuh"
#include "uint1_quant.cuh"

namespace vllm {

inline __device__ float get_quant_range(int num_bits) {
  if (num_bits == 8) {
    return 255.0f;
  } else if (num_bits == 4) {
    return 15.0f;
  } else if (num_bits == 2) {
    return 3.0f;
  } else if (num_bits == 1) {
    return 1.0f;
  } else {
    // TODO: maybe we can do check on the host and remove this assert
    printf("Error: Invalid number of bits %d. Supported bits are 1, 2, 4, and 8.\n", num_bits);
    assert(false);
  }
  return 0.0f;
}

// inv_scale is 1.0 / scale. This is to avoid division.
template <typename scalar_t>
inline __device__ uint16_t quant_and_pack(const scalar_t* data, int quant_bits, float inv_scale, float zero_point) {
  if (quant_bits == 8) {
    const int vec_size = 2;
    using vec_t = typename Vec<scalar_t, vec_size>::Type;
    vec_t vec = *reinterpret_cast<const vec_t*>(data);
    return kv_cache_uint8::quant(to_float(vec), inv_scale, zero_point);
  } else if (quant_bits == 4) {
    const int vec_size = 4;
    using vec_t = typename Vec<scalar_t, vec_size>::Type;
    vec_t vec = *reinterpret_cast<const vec_t*>(data);
    return kv_cache_uint4::quant(to_float(vec), inv_scale, zero_point);
  } else if (quant_bits == 2) {
    const int vec_size = 8;
    using vec_t = typename Vec<scalar_t, vec_size>::Type;
    vec_t vec = *reinterpret_cast<const vec_t*>(data);
    return kv_cache_uint2::quant(to_float(vec), inv_scale, zero_point);
  } else if (quant_bits == 1) {
    const int vec_size = 16;
    using vec_t = typename Vec<scalar_t, vec_size>::Type;
    vec_t vec = *reinterpret_cast<const vec_t*>(data);
    return kv_cache_uint1::quant(to_float(vec), inv_scale, zero_point);
  } else {
    printf("Error: Invalid number of bits %d. Supported bits are 1, 2, 4, and 8.\n", quant_bits);
    assert(false);
  }
  return 0;
}

/**
 * @brief Unpacks and dequantizes a given source value and stores the results (a few float numbers) in the destination pointer.
 *
 * @param src The source value to be unpacked and dequantized.
 * @param quant_bits The number of quantization bits. Supported values are 2, 4, and 8.
 * @param scale Per-head scale factor.
 * @param zero_point Per-head zero point.
 * @param dst Pointer to the destination memory where the dequantized values will be stored.
 */
inline __device__ void unpack_and_dequant(uint16_t src,
                                          int quant_bits,
                                          float scale,
                                          float zero_point,
                                          float* dst) {
  if (quant_bits == 8) {
    float2 float_vec = kv_cache_uint8::dequant(src, scale, zero_point);
    *reinterpret_cast<float2*>(dst) = float_vec;
  } else if (quant_bits == 4) {
    Float4_ float_vec = kv_cache_uint4::dequant(src, scale, zero_point);
    *reinterpret_cast<Float4_*>(dst) = float_vec;
  } else if (quant_bits == 2) {
    Float8_ float_vec = kv_cache_uint2::dequant(src, scale, zero_point);
    *reinterpret_cast<Float8_*>(dst) = float_vec;
  } else if (quant_bits == 1) {
    Float16_ float_vec = kv_cache_uint1::dequant(src, scale, zero_point);
    *reinterpret_cast<Float16_*>(dst) = float_vec;
  } else {
    printf("Error: Invalid number of bits %d. Supported bits are 1, 2, 4, and 8.\n", quant_bits);
    assert(false);
  }
}

// inline __device__ uint16_t requant(uint16_t* data_high,
//                                    int quant_bits_high,
//                                    float scale_high,
//                                    float zero_point_high,
//                                    int quant_bits_low,
//                                    float scale_low,
//                                    float zero_point_low) {
//   if (quant_bits_high == quant_bits_low) {
//     return data_high[0];
//   } else if (quant_bits_high == 8 && quant_bits_low == 4) {
//     uint16_t x = data_high[0];
//     uint16_t y = data_high[1];
//     float2 float_vec_x = kv_cache_uint8::dequant(x, scale_high, zero_point_high);
//     float2 float_vec_y = kv_cache_uint8::dequant(y, scale_high, zero_point_high);
//     Float4_ xy = {float_vec_x.x, float_vec_x.y, float_vec_y.x, float_vec_y.y};
//     return kv_cache_uint4::quant(xy, scale_low, zero_point_low);
//   } else if (quant_bits_high == 4 && quant_bits_low == 2) {
//     uint16_t x = data_high[0];
//     uint16_t y = data_high[1];
//     Float4_ float_vec_x = kv_cache_uint4::dequant(x, scale_high, zero_point_high);
//     Float4_ float_vec_y = kv_cache_uint4::dequant(y, scale_high, zero_point_high);
//     Float8_ xy = {float_vec_x.x.x, float_vec_x.x.y, float_vec_x.y.x, float_vec_x.y.y,
//                   float_vec_y.x.x, float_vec_y.x.y, float_vec_y.y.x, float_vec_y.y.y};
//     return kv_cache_uint2::quant(xy, scale_low, zero_point_low);
//   } else {
//     printf("Error: Unsupported requantization from %d bits to %d bits.\n", quant_bits_high, quant_bits_low);
//     assert(false);
//   }
// }

// requant can be lossless if bit number of high precision is twice the bit number of low precision
// consider k-bit for high-precision and max-min=1 with loss of generality
// the quant ranges are [0, 1/(2^(k + 1) - 2)], [1/(2^(k + 1) - 2), 1/(2^(k + 1) - 2) + 1/(2^k - 1)], ...
// in other words, the width of the 1st and last intervals are 1/(2^(k + 1) - 2), and the rests are 1/(2^k - 1)
// assume the low precision is (k - s) bits
// in order to align the high precision quant range to the low precision quant range, two criterions need to be met
// 1. the 1st interval of low precision should be aligned with certain interval of high precision
//    which means
//    1/(2^(k + 1) - 2) + x * 1/(2^k - 1) = 1/(2^(k - s + 1) - 2) ->
//    x = 2^(k - s) * (2^s - 1) / {2 * [2^(k - s) - 1]}， x is an integer
// 2. after aligning with the 1st interval of low precision, later intervals also need to aligned
//    requiring normal intervals (except 1st and last) of low precisions need to multiples of high precision
//    which means
//    y = (2^k - 1) / (2 ^ (k - s) - 1), y is an interger
// Both 1 and 2 are met when k = 2s. Thus, requant from 8 to 4, 4 to 2 can be lossless
// When k = 2s
// in 1, x = 2^(s - 1), and the initial 1 + 2^(s - 1) high precision intervals align with the 1st low precision interval;
// In 2, y = 2^s + 1

inline __device__ uint16_t _requant_pow2(uint16_t quantized,
                                         uint16_t scale,
                                         uint16_t zero_point) {
  return quantized > zero_point ? (quantized - zero_point) / scale + 1 : 0;
}

// mid_point should be (2^k - 1) / 2
inline __device__ uint16_t _requant_1bit(uint16_t quantized, uint16_t mid_point) {
  return quantized > mid_point ? 1 : 0;
}

inline __device__ uint16_t requant(uint16_t* data_high,
                                   int quant_bits_high,
                                   int quant_bits_low) {
  if (quant_bits_high == quant_bits_low) {
    return data_high[0];
  } else if (quant_bits_high == 8 && quant_bits_low == 4) {
    uint16_t x = data_high[0];
    uint16_t y = data_high[1];
    uint16_t requant_zero_point = 9;  // 1 + 2^(4 - 1), s = 4
    uint16_t requant_scale = 17;   // 2^4 + 1, s = 4

    uint16_t uint16[4];
    uint16[0] = _requant_pow2(x & 0x00FF, requant_scale, requant_zero_point);
    uint16[1] = _requant_pow2((x >> 8) & 0x00FF, requant_scale, requant_zero_point);
    uint16[2] = _requant_pow2(y & 0x00FF, requant_scale, requant_zero_point);
    uint16[3] = _requant_pow2((y >> 8) & 0x00FF, requant_scale, requant_zero_point);
    return (uint16[3] << 12) | (uint16[2] << 8) | (uint16[1] << 4) | uint16[0];
  } else if (quant_bits_high == 4 && quant_bits_low == 2) {
    uint16_t x = data_high[0];
    uint16_t y = data_high[1];
    uint16_t requant_zero_point = 3;  // 1 + 2^(2 - 1), s = 2
    uint16_t requant_scale = 5;   // 2^2 + 1, s = 2

    uint16_t uint16[8];
    uint16[0] = _requant_pow2(x & 0x0F, requant_scale, requant_zero_point);
    uint16[1] = _requant_pow2((x >> 4) & 0x0F, requant_scale, requant_zero_point);
    uint16[2] = _requant_pow2((x >> 8) & 0x0F, requant_scale, requant_zero_point);
    uint16[3] = _requant_pow2((x >> 12) & 0x0F, requant_scale, requant_zero_point);
    uint16[4] = _requant_pow2(y & 0x0F, requant_scale, requant_zero_point);
    uint16[5] = _requant_pow2((y >> 4) & 0x0F, requant_scale, requant_zero_point);
    uint16[6] = _requant_pow2((y >> 8) & 0x0F, requant_scale, requant_zero_point);
    uint16[7] = _requant_pow2((y >> 12) & 0x0F, requant_scale, requant_zero_point);
    return (uint16[7] << 14) | (uint16[6] << 12) | (uint16[5] << 10) | (uint16[4] << 8) |
           (uint16[3] << 6) | (uint16[2] << 4) | (uint16[1] << 2) | uint16[0];
  } else if (quant_bits_high == 2 && quant_bits_low == 1) {
    uint16_t x = data_high[0];
    uint16_t y = data_high[1];
    uint16_t requant_mid_point = 1;  // (2^2 - 1) / 2

    uint16_t uint16[16];
    uint16[0] = _requant_1bit(x & 0x3, requant_mid_point);
    uint16[1] = _requant_1bit((x >> 2) & 0x3, requant_mid_point);
    uint16[2] = _requant_1bit((x >> 4) & 0x3, requant_mid_point);
    uint16[3] = _requant_1bit((x >> 6) & 0x3, requant_mid_point);
    uint16[4] = _requant_1bit((x >> 8) & 0x3, requant_mid_point);
    uint16[5] = _requant_1bit((x >> 10) & 0x3, requant_mid_point);
    uint16[6] = _requant_1bit((x >> 12) & 0x3, requant_mid_point);
    uint16[7] = _requant_1bit((x >> 14) & 0x3, requant_mid_point);
    uint16[8] = _requant_1bit(y & 0x3, requant_mid_point);
    uint16[9] = _requant_1bit((y >> 2) & 0x3, requant_mid_point);
    uint16[10] = _requant_1bit((y >> 4) & 0x3, requant_mid_point);
    uint16[11] = _requant_1bit((y >> 6) & 0x3, requant_mid_point);
    uint16[12] = _requant_1bit((y >> 8) & 0x3, requant_mid_point);
    uint16[13] = _requant_1bit((y >> 10) & 0x3, requant_mid_point);
    uint16[14] = _requant_1bit((y >> 12) & 0x3, requant_mid_point);
    uint16[15] = _requant_1bit((y >> 14) & 0x3, requant_mid_point);
    return (uint16[15] << 15) | (uint16[14] << 14) | (uint16[13] << 13) | (uint16[12] << 12) |
           (uint16[11] << 11) | (uint16[10] << 10) | (uint16[9] << 9) | (uint16[8] << 8) |
           (uint16[7] << 7) | (uint16[6] << 6) | (uint16[5] << 5) | (uint16[4] << 4) |
           (uint16[3] << 3) | (uint16[2] << 2) | (uint16[1] << 1) | uint16[0];
  } else if (quant_bits_high == 4 && quant_bits_low == 1) {
    uint16_t x = data_high[0];
    uint16_t y = data_high[1];
    uint16_t z = data_high[2];
    uint16_t w = data_high[3];
    uint16_t requant_mid_point = 7;  // (2^4 - 1) / 2

    uint16_t uint16[16];
    uint16[0] = _requant_1bit(x & 0x0F, requant_mid_point);
    uint16[1] = _requant_1bit((x >> 4) & 0x0F, requant_mid_point);
    uint16[2] = _requant_1bit((x >> 8) & 0x0F, requant_mid_point);
    uint16[3] = _requant_1bit((x >> 12) & 0x0F, requant_mid_point);
    uint16[4] = _requant_1bit(y & 0x0F, requant_mid_point);
    uint16[5] = _requant_1bit((y >> 4) & 0x0F, requant_mid_point);
    uint16[6] = _requant_1bit((y >> 8) & 0x0F, requant_mid_point);
    uint16[7] = _requant_1bit((y >> 12) & 0x0F, requant_mid_point);
    uint16[8] = _requant_1bit(z & 0x0F, requant_mid_point);
    uint16[9] = _requant_1bit((z >> 4) & 0x0F, requant_mid_point);
    uint16[10] = _requant_1bit((z >> 8) & 0x0F, requant_mid_point);
    uint16[11] = _requant_1bit((z >> 12) & 0x0F, requant_mid_point);
    uint16[12] = _requant_1bit(w & 0x0F, requant_mid_point);
    uint16[13] = _requant_1bit((w >> 4) & 0x0F, requant_mid_point);
    uint16[14] = _requant_1bit((w >> 8) & 0x0F, requant_mid_point);
    uint16[15] = _requant_1bit((w >> 12) & 0x0F, requant_mid_point);
    return (uint16[15] << 15) | (uint16[14] << 14) | (uint16[13] << 13) | (uint16[12] << 12) |
           (uint16[11] << 11) | (uint16[10] << 10) | (uint16[9] << 9) | (uint16[8] << 8) |
           (uint16[7] << 7) | (uint16[6] << 6) | (uint16[5] << 5) | (uint16[4] << 4) |
           (uint16[3] << 3) | (uint16[2] << 2) | (uint16[1] << 1) | uint16[0];
  } else {
    printf("Error: Unsupported requantization from %d bits to %d bits.\n", quant_bits_high, quant_bits_low);
    assert(false);
    return 0;
  }
}

} // namespace vllm
