#include "../../attention/attention_dtypes.h"

namespace vllm {
namespace kv_cache_uint2 {

// Convert eight float values to eight uint2 values packed into one uint16
// inv_scale is 1.0 / scale. This is to avoid division.
inline __device__ uint16_t quant(Float8_ in, const float inv_scale, const float zero_point)
{
  uint16_t uint16[8];
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[0]) : "f"((in.x.x - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[1]) : "f"((in.x.y - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[2]) : "f"((in.y.x - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[3]) : "f"((in.y.y - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[4]) : "f"((in.z.x - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[5]) : "f"((in.z.y - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[6]) : "f"((in.w.x - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[7]) : "f"((in.w.y - zero_point) * inv_scale));
  return uint16[0] | (uint16[1] << 2) | (uint16[2] << 4) | (uint16[3] << 6) | (uint16[4] << 8) | (uint16[5] << 10) | (uint16[6] << 12) | (uint16[7] << 14);
}

// Convert eight uint2 values (packed into one uint16) to eight float values
inline __device__ Float8_ dequant(uint16_t in, const float scale, const float zero_point)
{
  Float8_ out;
  out.x.x = static_cast<float>(in & 0x3) * scale + zero_point;
  out.x.y = static_cast<float>((in >> 2) & 0x3) * scale + zero_point;
  out.y.x = static_cast<float>((in >> 4) & 0x3) * scale + zero_point;
  out.y.y = static_cast<float>((in >> 6) & 0x3) * scale + zero_point;
  out.z.x = static_cast<float>((in >> 8) & 0x3) * scale + zero_point;
  out.z.y = static_cast<float>((in >> 10) & 0x3) * scale + zero_point;
  out.w.x = static_cast<float>((in >> 12) & 0x3) * scale + zero_point;
  out.w.y = static_cast<float>((in >> 14) & 0x3) * scale + zero_point;
  return out;
}

} // namespace kv_cache_uint2
} // namespace vllm
