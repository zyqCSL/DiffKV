#include "../../attention/attention_dtypes.h"

namespace vllm {
namespace kv_cache_uint8 {

// Convert two float values to two uint8 values packed into one uint16
// inv_scale is 1.0 / scale. This is to avoid division.
inline __device__ uint16_t quant(float2 in, const float inv_scale, const float zero_point)
{
  uint16_t uint16[2];
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[0]) : "f"((in.x - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[1]) : "f"((in.y - zero_point) * inv_scale));
  return (uint16[0] & 0x00FF) | (uint16[1] << 8);
}

// Convert two uint8 values (packed into one uint16) to two float values
inline __device__ float2 dequant(uint16_t in, const float scale, const float zero_point)
{
  union {
    uint8_t uint8[2];
    uint16_t uint16;
  };
  uint16 = in;

  float2 out;
  out.x = to_float(uint8[0]) * scale + zero_point;
  out.y = to_float(uint8[1]) * scale + zero_point;
  return out;
}

} // namespace kv_cache_uint8
} // namespace vllm
