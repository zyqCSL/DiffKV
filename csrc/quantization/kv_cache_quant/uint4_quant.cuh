#include "../../attention/attention_dtypes.h"

namespace vllm {
namespace kv_cache_uint4 {

// Convert two float values to two uint4 values packed into one uint8
// inv_scale is 1.0 / scale. This is to avoid division.
inline __device__ uint8_t quant(float2 in, const float inv_scale, const float zero_point)
{
  union {
    uint8_t uint8[4];
    uint16_t uint16[2];
  };

  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[0]) : "f"((in.x - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[1]) : "f"((in.y - zero_point) * inv_scale));
  uint8[0] = (uint8[0] & 0xF) | (uint8[2] << 4);
  return uint8[0];
}

// Convert four float values to four uint4 values packed into one uint16
inline __device__ uint16_t quant(float4 in, const float inv_scale, const float zero_point)
{
  union {
    uint8_t uint8[2];
    uint16_t uint16;
  };

  uint8[0] = quant(make_float2(in.x, in.y), inv_scale, zero_point);
  uint8[1] = quant(make_float2(in.z, in.w), inv_scale, zero_point);
  return uint16;
}

inline __device__ uint16_t quant(Float4_ in, const float inv_scale, const float zero_point)
{
  union {
    uint8_t uint8[2];
    uint16_t uint16;
  };

  uint8[0] = quant(in.x, inv_scale, zero_point);
  uint8[1] = quant(in.y, inv_scale, zero_point);
  return uint16;
}

// Convert eight float values to eight uint4 values packed into one uint32
inline __device__ uint32_t quant(Float8_ in, const float inv_scale, const float zero_point)
{
  union {
    uint8_t uint8[4];
    uint32_t uint32;
  };

  uint8[0] = quant(in.x, inv_scale, zero_point);
  uint8[1] = quant(in.y, inv_scale, zero_point);
  uint8[2] = quant(in.z, inv_scale, zero_point);
  uint8[3] = quant(in.w, inv_scale, zero_point);
  return uint32;
}

// Convert two uint4 values (packed into one uint8) to two float values
inline __device__ float2 dequant(uint8_t in, const float scale, const float zero_point)
{
  uint8_t in_low = in & 0x0F;
  uint8_t in_high = (in >> 4) & 0x0F;

  float2 out;
  out.x = to_float(in_low) * scale + zero_point;
  out.y = to_float(in_high) * scale + zero_point;
  return out;
}

// Convert four uint4 values (packed into one uint16) to four float values
inline __device__ Float4_ dequant(uint16_t in, const float scale, const float zero_point)
{
  union {
    uint8_t uint8[2];
    uint16_t uint16;
  };
  uint16 = in;

  Float4_ out;
  out.x = dequant(uint8[0], scale, zero_point);
  out.y = dequant(uint8[1], scale, zero_point);
  return out;
}

// Convert eight uint4 values (packed into one uint32) to eight float values
inline __device__ Float8_ dequant(uint32_t in, const float scale, const float zero_point)
{
  union {
    uint8_t uint8[4];
    uint32_t uint32;
  };
  uint32 = in;

  Float8_ out;
  out.x = dequant(uint8[0], scale, zero_point);
  out.y = dequant(uint8[1], scale, zero_point);
  out.z = dequant(uint8[2], scale, zero_point);
  out.w = dequant(uint8[3], scale, zero_point);
  return out;
}

} // namespace kv_cache_uint4
} // namespace vllm
