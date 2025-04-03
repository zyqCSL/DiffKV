#include "../../attention/attention_dtypes.h"

namespace vllm {
namespace kv_cache_uint1 {

// Convert 16 float values to 16 uint1 values packed into one uint16
// inv_scale is 1.0 / scale. This is to avoid division.
inline __device__ uint16_t quant(Float16_ in, const float inv_scale, const float zero_point)
{
  uint16_t uint16[16];
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[0]) : "f"((in.x.x - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[1]) : "f"((in.x.y - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[2]) : "f"((in.x.z - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[3]) : "f"((in.x.w - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[4]) : "f"((in.y.x - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[5]) : "f"((in.y.y - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[6]) : "f"((in.y.z - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[7]) : "f"((in.y.w - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[8]) : "f"((in.z.x - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[9]) : "f"((in.z.y - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[10]) : "f"((in.z.z - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[11]) : "f"((in.z.w - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[12]) : "f"((in.w.x - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[13]) : "f"((in.w.y - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[14]) : "f"((in.w.z - zero_point) * inv_scale));
  asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(uint16[15]) : "f"((in.w.w - zero_point) * inv_scale));
  return uint16[0] | (uint16[1] << 1) | (uint16[2] << 2) | (uint16[3] << 3) | (uint16[4] << 4) | (uint16[5] << 5) | (uint16[6] << 6) | (uint16[7] << 7) |
        (uint16[8] << 8) | (uint16[9] << 9) | (uint16[10] << 10) | (uint16[11] << 11) | (uint16[12] << 12) | (uint16[13] << 13) | (uint16[14] << 14) | (uint16[15] << 15);
}

// Convert 16 uint1 values (packed into one uint16) to 16 float values
inline __device__ Float16_ dequant(uint16_t in, const float scale, const float zero_point)
{
  Float16_ out;
  out.x.x = static_cast<float>(in & 0x1) * scale + zero_point;
  out.x.y = static_cast<float>((in >> 1) & 0x1) * scale + zero_point;
  out.x.z = static_cast<float>((in >> 2) & 0x1) * scale + zero_point;
  out.x.w = static_cast<float>((in >> 3) & 0x1) * scale + zero_point;
  out.y.x = static_cast<float>((in >> 4) & 0x1) * scale + zero_point;
  out.y.y = static_cast<float>((in >> 5) & 0x1) * scale + zero_point;
  out.y.z = static_cast<float>((in >> 6) & 0x1) * scale + zero_point;
  out.y.w = static_cast<float>((in >> 7) & 0x1) * scale + zero_point;
  out.z.x = static_cast<float>((in >> 8) & 0x1) * scale + zero_point;
  out.z.y = static_cast<float>((in >> 9) & 0x1) * scale + zero_point;
  out.z.z = static_cast<float>((in >> 10) & 0x1) * scale + zero_point;
  out.z.w = static_cast<float>((in >> 11) & 0x1) * scale + zero_point;
  out.w.x = static_cast<float>((in >> 12) & 0x1) * scale + zero_point;
  out.w.y = static_cast<float>((in >> 13) & 0x1) * scale + zero_point;
  out.w.z = static_cast<float>((in >> 14) & 0x1) * scale + zero_point;
  out.w.w = static_cast<float>((in >> 15) & 0x1) * scale + zero_point;
  return out;
}

} // namespace kv_cache_uint1
} // namespace vllm
