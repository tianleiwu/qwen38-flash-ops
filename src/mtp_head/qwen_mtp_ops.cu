// MTP head ops for the Qwen3.8-Flash-Next capture-lowered MTP head (domain qwen.mtp).
//
//   Bf16ArgMax(logits bf16[..., V]) -> int64[..., 1]      greedy draft token on device (lowest index among maxima)
//   MtpMoe(premix bf16[1,R,2560], router_logits bf16[1,R,512], gate_up bf16[512,1280,2560], down bf16[512,2560,640])
//       -> routed bf16[1,R,2560], scratch f32[R,..]
//        softmax -> top-10 -> normalised weights -> (gate_up GEMV + SwishMul) -> down GEMV -> bf16 weighted sum,
//        replacing Cast/Softmax/TopK/ReduceSum/Div/CastLike + Gather(expert weights) + two batched MatMuls + Swish/Mul +
//        Squeeze/Mul/ReduceSum of the exported BF16 MoE (the gather copied 98 MB of expert weights every call).
//   Rounding points follow the unfused chain (bf16 MatMul outputs, bf16(bf16(silu(g)) * u), bf16 weights, bf16 products,
//   fp32 sum in expert order, one bf16 rounding); the fp32 accumulation order of the GEMVs differs from cuBLAS.
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;

constexpr int kHidden = 2560;
constexpr int kExperts = 512;
constexpr int kTopK = 10;
constexpr int kFfn = 640;          // per-expert intermediate size
constexpr int kGateUp = 2 * kFfn;  // 1280
constexpr int kChunk = 8;          // bf16 per 16-byte load
constexpr int kScratchStride = 32 + kTopK * kFfn;  // route(idx[10], weight[10]) padded to 32 floats, then h[10][640]

__device__ __forceinline__ float R(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float Sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

void CheckLaunch() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

cudaStream_t Stream(const OrtApi& api, OrtKernelContext* context) {
  void* raw = nullptr;
  Ort::ThrowOnError(api.KernelContext_GetGPUComputeStream(context, &raw));
  return static_cast<cudaStream_t>(raw);
}

int64_t Count(const Ort::ConstValue& v) {
  int64_t n = 1;
  for (auto d : v.GetTensorTypeAndShapeInfo().GetShape()) n *= d;
  return n;
}

__device__ __forceinline__ void Unpack(const uint4& v, float* f) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ void UnpackFp8(const uint2& v, float* f) {
  const uint32_t w[2] = {v.x, v.y};
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const __half2_raw hr01 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w[i] & 0xffffu), __NV_E4M3);
    const __half2_raw hr23 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w[i] >> 16), __NV_E4M3);
    const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&hr01));
    const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&hr23));
    f[4 * i + 0] = f01.x;
    f[4 * i + 1] = f01.y;
    f[4 * i + 2] = f23.x;
    f[4 * i + 3] = f23.y;
  }
}

__device__ __forceinline__ float Fp8ByteToFloat(uint8_t b) {
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
  return __half2float(*reinterpret_cast<const __half*>(&hr));
}

__device__ __forceinline__ float E2M1ToFloat(uint32_t nib) {
  const uint32_t m = nib & 7u;
  const uint32_t exp_mant = (m == 0u) ? 0u : ((m == 1u ? 252u : (m + 252u)) << 22);
  return __uint_as_float(((nib & 8u) << 28) | exp_mant);
}

__device__ __forceinline__ void UnpackFp4(uint32_t pack8, uint8_t scale_fp8, float* f) {
  const float s = Fp8ByteToFloat(scale_fp8);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    f[i] = E2M1ToFloat((pack8 >> (4 * i)) & 0xfu) * s;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// argmax
// ---------------------------------------------------------------------------------------------------------------
constexpr int kArgMaxThreads = 256;
constexpr int kArgMaxBlocks = 124;  // 248320 / 8 = 31040 chunks = 124 * 250
constexpr int kArgMaxFinalThreads = 128;

__device__ __forceinline__ void MergeBest(float& best, int& best_i, float ov, int oi) {
  if (ov > best || (ov == best && oi < best_i)) {
    best = ov;
    best_i = oi;
  }
}

// stage 1: every block reduces a contiguous range of 16-byte chunks to one (value, index) partial.
__global__ void __launch_bounds__(kArgMaxThreads)
    Bf16ArgMaxPartialKernel(const bf16* __restrict__ x, float* __restrict__ partial, int n) {
  float best = -INFINITY;
  int best_i = 0x7fffffff;
  const uint4* x4 = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(blockIdx.y) * n);
  const int chunks = n / kChunk;
  const int per_block = (chunks + gridDim.x - 1) / gridDim.x;
  const int begin = blockIdx.x * per_block;
  const int end = min(chunks, begin + per_block);
  for (int c = begin + threadIdx.x; c < end; c += blockDim.x) {
    float f[8];
    Unpack(__ldg(x4 + c), f);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (f[k] > best) {
        best = f[k];
        best_i = c * kChunk + k;
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    MergeBest(best, best_i, __shfl_down_sync(0xffffffffu, best, offset), __shfl_down_sync(0xffffffffu, best_i, offset));
  }
  __shared__ float sv[kArgMaxThreads / 32];
  __shared__ int si[kArgMaxThreads / 32];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) {
    sv[warp] = best;
    si[warp] = best_i;
  }
  __syncthreads();
  if (warp == 0) {
    best = lane < (kArgMaxThreads / 32) ? sv[lane] : -INFINITY;
    best_i = lane < (kArgMaxThreads / 32) ? si[lane] : 0x7fffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
      MergeBest(best, best_i, __shfl_down_sync(0xffffffffu, best, offset), __shfl_down_sync(0xffffffffu, best_i, offset));
    }
    if (lane == 0) {
      float* out = partial + (static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x) * 2;
      out[0] = best;
      out[1] = __int_as_float(best_i);
    }
  }
}

// stage 2: one block per row merges the partials (value descending, index ascending).
__global__ void __launch_bounds__(kArgMaxFinalThreads)
    Bf16ArgMaxFinalKernel(const float* __restrict__ partial, int64_t* __restrict__ out, int blocks) {
  float best = -INFINITY;
  int best_i = 0x7fffffff;
  if (threadIdx.x < blocks) {
    const float* p = partial + (static_cast<int64_t>(blockIdx.x) * blocks + threadIdx.x) * 2;
    best = p[0];
    best_i = __float_as_int(p[1]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    MergeBest(best, best_i, __shfl_down_sync(0xffffffffu, best, offset), __shfl_down_sync(0xffffffffu, best_i, offset));
  }
  __shared__ float sv[kArgMaxFinalThreads / 32];
  __shared__ int si[kArgMaxFinalThreads / 32];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) {
    sv[warp] = best;
    si[warp] = best_i;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    best = sv[0];
    best_i = si[0];
    for (int w = 1; w < kArgMaxFinalThreads / 32; ++w) MergeBest(best, best_i, sv[w], si[w]);
    out[blockIdx.x] = best_i;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// MoE
// ---------------------------------------------------------------------------------------------------------------
// One warp per row: softmax (ORT warp-softmax order: per-lane sequential exp sum over elements lane + 32*i, butterfly),
// top-10 (value descending, lowest index first on ties), weights = bf16(p / sum_top10).
__global__ void RouterKernel(const bf16* __restrict__ logits, float* __restrict__ scratch) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  constexpr int kIter = kExperts / 32;
  float v[kIter];
  float mx = -INFINITY;
#pragma unroll
  for (int i = 0; i < kIter; ++i) {
    v[i] = __bfloat162float(logits[static_cast<int64_t>(row) * kExperts + i * 32 + lane]);
    mx = fmaxf(mx, v[i]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, offset));
  float e[kIter];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kIter; ++i) {
    e[i] = expf(v[i] - mx);
    sum += e[i];
  }
  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  float p[kIter];
#pragma unroll
  for (int i = 0; i < kIter; ++i) p[i] = e[i] / sum;

  float top_v[kTopK];
  int top_i[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float best = -1.0f;
    int best_i = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < kIter; ++i) {
      if (p[i] > best) {
        best = p[i];
        best_i = i * 32 + lane;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, offset);
      const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
      if (ov > best || (ov == best && oi < best_i)) {
        best = ov;
        best_i = oi;
      }
    }
    top_v[k] = best;
    top_i[k] = best_i;
    // remove the winner from its owner lane
#pragma unroll
    for (int i = 0; i < kIter; ++i) {
      if (best_i == i * 32 + lane) p[i] = -2.0f;
    }
  }
  if (lane == 0) {
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) total += top_v[k];
    float* out = scratch + static_cast<int64_t>(row) * kScratchStride;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      out[k] = __int_as_float(top_i[k]);
      out[kTopK + k] = R(top_v[k] / total);
    }
  }
}

// gate/up GEMV pair + SwishMul. grid (640 / 8, 10 experts, rows); each warp handles one (gate row j, up row 640 + j) pair.
constexpr int kWarpsPerBlock = 8;
constexpr int kKChunks = kHidden / kChunk;  // 320
constexpr int kPerLane = kKChunks / 32;     // 10

__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    GateUpKernel(const bf16* __restrict__ x, const bf16* __restrict__ gate_up, float* __restrict__ scratch) {
  const int row = blockIdx.z;
  const int slot = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float* base = scratch + static_cast<int64_t>(row) * kScratchStride;
  const int expert = __float_as_int(base[slot]);
  const int j = blockIdx.x * kWarpsPerBlock + warp;
  const uint4* xv = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(row) * kHidden);
  const uint4* wg = reinterpret_cast<const uint4*>(gate_up + (static_cast<int64_t>(expert) * kGateUp + j) * kHidden);
  const uint4* wu = reinterpret_cast<const uint4*>(gate_up + (static_cast<int64_t>(expert) * kGateUp + kFfn + j) * kHidden);
  uint4 g[kPerLane], u[kPerLane], xr[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    g[i] = __ldg(wg + i * 32 + lane);
    u[i] = __ldg(wu + i * 32 + lane);
    xr[i] = __ldg(xv + i * 32 + lane);
  }
  float acc_g = 0.0f, acc_u = 0.0f;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    float fg[8], fu[8], fx[8];
    Unpack(g[i], fg);
    Unpack(u[i], fu);
    Unpack(xr[i], fx);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      acc_g = fmaf(fg[k], fx[k], acc_g);
      acc_u = fmaf(fu[k], fx[k], acc_u);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc_g += __shfl_xor_sync(0xffffffffu, acc_g, offset);
    acc_u += __shfl_xor_sync(0xffffffffu, acc_u, offset);
  }
  if (lane == 0) {
    const float gv = R(acc_g);
    const float uv = R(acc_u);
    const float s = R(gv * Sigmoid(gv));
    // h is bf16 by construction; the down kernel reads it as packed bf16 (coalesced 16-byte loads)
    reinterpret_cast<bf16*>(base + 32)[slot * kFfn + j] = __float2bfloat16(s * uv);
  }
}

// Quantized GateUpKernel: mode == 1 (FP8 E4M3 rows of 2560 + 16 trailer with float32 scale),
// mode == 2 (NVFP4 block-16 rows of 1280 + 160 = 1440 bytes with global_scale).
template <int MODE>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    GateUpQuantKernel(const bf16* __restrict__ x, const uint8_t* __restrict__ gate_up, float* __restrict__ scratch,
                      float global_scale) {
  constexpr int kRowBytes = MODE == 1 ? (kHidden + 16) : (kHidden * 9 / 16);
  const int row = blockIdx.z;
  const int slot = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float* base = scratch + static_cast<int64_t>(row) * kScratchStride;
  const int expert = __float_as_int(base[slot]);
  const int j = blockIdx.x * kWarpsPerBlock + warp;
  const uint4* xv = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(row) * kHidden);
  const uint8_t* rg = gate_up + (static_cast<int64_t>(expert) * kGateUp + j) * kRowBytes;
  const uint8_t* ru = gate_up + (static_cast<int64_t>(expert) * kGateUp + kFfn + j) * kRowBytes;
  uint4 xr[kPerLane];
  float acc_g = 0.0f, acc_u = 0.0f;
  if constexpr (MODE == 1) {
    uint2 g2[kPerLane], u2[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int c = i * 32 + lane;
      g2[i] = __ldg(reinterpret_cast<const uint2*>(rg) + c);
      u2[i] = __ldg(reinterpret_cast<const uint2*>(ru) + c);
      xr[i] = __ldg(xv + c);
    }
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      float fg[8], fu[8], fx[8];
      UnpackFp8(g2[i], fg);
      UnpackFp8(u2[i], fu);
      Unpack(xr[i], fx);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        acc_g = fmaf(fg[k], fx[k], acc_g);
        acc_u = fmaf(fu[k], fx[k], acc_u);
      }
    }
  } else {
    uint32_t g4[kPerLane], u4[kPerLane];
    uint8_t gs[kPerLane], us[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int c = i * 32 + lane;
      g4[i] = __ldg(reinterpret_cast<const uint32_t*>(rg) + c);
      u4[i] = __ldg(reinterpret_cast<const uint32_t*>(ru) + c);
      gs[i] = __ldg(rg + (kHidden / 2) + (c >> 1));
      us[i] = __ldg(ru + (kHidden / 2) + (c >> 1));
      xr[i] = __ldg(xv + c);
    }
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      float fg[8], fu[8], fx[8];
      UnpackFp4(g4[i], gs[i], fg);
      UnpackFp4(u4[i], us[i], fu);
      Unpack(xr[i], fx);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        acc_g = fmaf(fg[k], fx[k], acc_g);
        acc_u = fmaf(fu[k], fx[k], acc_u);
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc_g += __shfl_xor_sync(0xffffffffu, acc_g, offset);
    acc_u += __shfl_xor_sync(0xffffffffu, acc_u, offset);
  }
  if (lane == 0) {
    if constexpr (MODE == 1) {
      acc_g *= *reinterpret_cast<const float*>(rg + kHidden);
      acc_u *= *reinterpret_cast<const float*>(ru + kHidden);
    } else {
      acc_g *= global_scale;
      acc_u *= global_scale;
    }
    const float gv = R(acc_g);
    const float uv = R(acc_u);
    const float s = R(gv * Sigmoid(gv));
    reinterpret_cast<bf16*>(base + 32)[slot * kFfn + j] = __float2bfloat16(s * uv);
  }
}

// down GEMV + weighted bf16 sum. grid (2560, rows), block = 10 warps: warp k owns expert slot k of output element
// blockIdx.x (one round trip of weight loads per block); the 10 weighted products are summed in expert order.
constexpr int kDownChunks = kFfn / kChunk;             // 80
constexpr int kDownPerLane = (kDownChunks + 31) / 32;  // 3

__global__ void __launch_bounds__(kTopK * 32)
    DownKernel(const bf16* __restrict__ down, const float* __restrict__ scratch, bf16* __restrict__ out) {
  const int row = blockIdx.y;
  const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int i = blockIdx.x;
  const float* base = scratch + static_cast<int64_t>(row) * kScratchStride;
  const int expert = __float_as_int(base[k]);
  const uint4* w = reinterpret_cast<const uint4*>(down + (static_cast<int64_t>(expert) * kHidden + i) * kFfn);
  const uint4* h = reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(base + 32) + k * kFfn);
  uint4 wv[kDownPerLane], hv[kDownPerLane];
#pragma unroll
  for (int t = 0; t < kDownPerLane; ++t) {
    const int c = t * 32 + lane;
    wv[t] = c < kDownChunks ? __ldg(w + c) : make_uint4(0, 0, 0, 0);
    hv[t] = c < kDownChunks ? h[c] : make_uint4(0, 0, 0, 0);
  }
  float sum = 0.0f;
#pragma unroll
  for (int t = 0; t < kDownPerLane; ++t) {
    const int c = t * 32 + lane;
    if (c < kDownChunks) {
      float fw[8], fh[8];
      Unpack(wv[t], fw);
      Unpack(hv[t], fh);
#pragma unroll
      for (int e = 0; e < 8; ++e) sum = fmaf(fw[e], fh[e], sum);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  __shared__ float weighted[kTopK];
  if (lane == 0) weighted[k] = R(R(sum) * base[kTopK + k]);
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int e = 0; e < kTopK; ++e) total += weighted[e];
    out[static_cast<int64_t>(row) * kHidden + i] = __float2bfloat16(total);
  }
}

template <int MODE>
__global__ void __launch_bounds__(kTopK * 32)
    DownQuantKernel(const uint8_t* __restrict__ down, const float* __restrict__ scratch, bf16* __restrict__ out,
                    float global_scale) {
  constexpr int kRowBytes = MODE == 1 ? (kFfn + 16) : (kFfn * 9 / 16);
  const int row = blockIdx.y;
  const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int i = blockIdx.x;
  const float* base = scratch + static_cast<int64_t>(row) * kScratchStride;
  const int expert = __float_as_int(base[k]);
  const uint8_t* rw = down + (static_cast<int64_t>(expert) * kHidden + i) * kRowBytes;
  const uint4* h = reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(base + 32) + k * kFfn);
  float sum = 0.0f;
  uint4 hv[kDownPerLane];
  if constexpr (MODE == 1) {
    uint2 w2[kDownPerLane];
#pragma unroll
    for (int t = 0; t < kDownPerLane; ++t) {
      const int c = t * 32 + lane;
      w2[t] = c < kDownChunks ? __ldg(reinterpret_cast<const uint2*>(rw) + c) : make_uint2(0, 0);
      hv[t] = c < kDownChunks ? h[c] : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int t = 0; t < kDownPerLane; ++t) {
      const int c = t * 32 + lane;
      if (c < kDownChunks) {
        float fw[8], fh[8];
        UnpackFp8(w2[t], fw);
        Unpack(hv[t], fh);
#pragma unroll
        for (int e = 0; e < 8; ++e) sum = fmaf(fw[e], fh[e], sum);
      }
    }
  } else {
    uint32_t w4[kDownPerLane];
    uint8_t ws[kDownPerLane];
#pragma unroll
    for (int t = 0; t < kDownPerLane; ++t) {
      const int c = t * 32 + lane;
      w4[t] = c < kDownChunks ? __ldg(reinterpret_cast<const uint32_t*>(rw) + c) : 0u;
      ws[t] = c < kDownChunks ? __ldg(rw + (kFfn / 2) + (c >> 1)) : 0u;
      hv[t] = c < kDownChunks ? h[c] : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int t = 0; t < kDownPerLane; ++t) {
      const int c = t * 32 + lane;
      if (c < kDownChunks) {
        float fw[8], fh[8];
        UnpackFp4(w4[t], ws[t], fw);
        Unpack(hv[t], fh);
#pragma unroll
        for (int e = 0; e < 8; ++e) sum = fmaf(fw[e], fh[e], sum);
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  __shared__ float weighted[kTopK];
  if (lane == 0) {
    if constexpr (MODE == 1) {
      sum *= *reinterpret_cast<const float*>(rw + kFfn);
    } else {
      sum *= global_scale;
    }
    weighted[k] = R(R(sum) * base[kTopK + k]);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int e = 0; e < kTopK; ++e) total += weighted[e];
    out[static_cast<int64_t>(row) * kHidden + i] = __float2bfloat16(total);
  }
}

struct ArgMaxState {
  const OrtApi& api;
  ArgMaxState(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
      const int64_t n = shape.back();
      if (n % kChunk != 0 || n >= (1 << 30)) throw std::invalid_argument("Bf16ArgMax: bad vocabulary size");
      const int64_t rows = Count(x) / n;
      shape.back() = 1;
      auto out = c.GetOutput(0, shape);
      std::vector<int64_t> scratch_shape = {rows, kArgMaxBlocks, 2};
      auto scratch = c.GetOutput(1, scratch_shape);
      cudaStream_t stream = Stream(api, raw);
      Bf16ArgMaxPartialKernel<<<dim3(kArgMaxBlocks, static_cast<unsigned>(rows)), kArgMaxThreads, 0, stream>>>(
          reinterpret_cast<const bf16*>(x.GetTensorData<Ort::BFloat16_t>()), scratch.GetTensorMutableData<float>(),
          static_cast<int>(n));
      Bf16ArgMaxFinalKernel<<<static_cast<unsigned>(rows), kArgMaxFinalThreads, 0, stream>>>(
          scratch.GetTensorData<float>(), out.GetTensorMutableData<int64_t>(), kArgMaxBlocks);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct MoeState {
  const OrtApi& api;
  float gate_up_global_scale = 1.0f;
  float down_global_scale = 1.0f;

  MoeState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    try {
      gate_up_global_scale = k.GetAttribute<float>("gate_up_global_scale");
    } catch (const std::exception&) {
    }
    try {
      down_global_scale = k.GetAttribute<float>("down_global_scale");
    } catch (const std::exception&) {
    }
  }

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      auto router = c.GetInput(1);
      auto gate_up = c.GetInput(2);
      auto down = c.GetInput(3);
      auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
      const int64_t rows = Count(x) / kHidden;
      const auto gu_info = gate_up.GetTensorTypeAndShapeInfo();
      const auto dn_info = down.GetTensorTypeAndShapeInfo();
      const auto gu_shape = gu_info.GetShape();
      const auto dn_shape = dn_info.GetShape();
      if (Count(x) != rows * kHidden || Count(router) != rows * kExperts || gu_shape.size() != 3 ||
          dn_shape.size() != 3 || gu_shape[0] != kExperts || gu_shape[1] != kGateUp || dn_shape[0] != kExperts ||
          dn_shape[1] != kHidden) {
        throw std::invalid_argument("MtpMoe: shape mismatch");
      }
      auto out = c.GetOutput(0, shape);
      std::vector<int64_t> scratch_shape = {rows, kScratchStride};
      auto scratch = c.GetOutput(1, scratch_shape);
      cudaStream_t stream = Stream(api, raw);
      float* scratch_ptr = scratch.GetTensorMutableData<float>();
      RouterKernel<<<static_cast<unsigned>(rows), 32, 0, stream>>>(
          reinterpret_cast<const bf16*>(router.GetTensorData<Ort::BFloat16_t>()), scratch_ptr);
      const auto wtype = gu_info.GetElementType();
      if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 && gu_shape[2] == kHidden && dn_shape[2] == kFfn) {
        GateUpKernel<<<dim3(kFfn / kWarpsPerBlock, kTopK, static_cast<unsigned>(rows)), kWarpsPerBlock * 32, 0, stream>>>(
            reinterpret_cast<const bf16*>(x.GetTensorData<Ort::BFloat16_t>()),
            reinterpret_cast<const bf16*>(gate_up.GetTensorData<Ort::BFloat16_t>()), scratch_ptr);
        DownKernel<<<dim3(kHidden, static_cast<unsigned>(rows)), kTopK * 32, 0, stream>>>(
            reinterpret_cast<const bf16*>(down.GetTensorData<Ort::BFloat16_t>()), scratch_ptr,
            reinterpret_cast<bf16*>(out.GetTensorMutableData<Ort::BFloat16_t>()));
      } else if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 && gu_shape[2] == kHidden + 16 &&
                 dn_shape[2] == kFfn + 16) {
        GateUpQuantKernel<1><<<dim3(kFfn / kWarpsPerBlock, kTopK, static_cast<unsigned>(rows)), kWarpsPerBlock * 32, 0, stream>>>(
            reinterpret_cast<const bf16*>(x.GetTensorData<Ort::BFloat16_t>()),
            static_cast<const uint8_t*>(gate_up.GetTensorRawData()), scratch_ptr, 1.0f);
        DownQuantKernel<1><<<dim3(kHidden, static_cast<unsigned>(rows)), kTopK * 32, 0, stream>>>(
            static_cast<const uint8_t*>(down.GetTensorRawData()), scratch_ptr,
            reinterpret_cast<bf16*>(out.GetTensorMutableData<Ort::BFloat16_t>()), 1.0f);
      } else if (wtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 && gu_shape[2] == (kHidden * 9) / 16 &&
                 dn_shape[2] == (kFfn * 9) / 16) {
        GateUpQuantKernel<2><<<dim3(kFfn / kWarpsPerBlock, kTopK, static_cast<unsigned>(rows)), kWarpsPerBlock * 32, 0, stream>>>(
            reinterpret_cast<const bf16*>(x.GetTensorData<Ort::BFloat16_t>()),
            static_cast<const uint8_t*>(gate_up.GetTensorRawData()), scratch_ptr, gate_up_global_scale);
        DownQuantKernel<2><<<dim3(kHidden, static_cast<unsigned>(rows)), kTopK * 32, 0, stream>>>(
            static_cast<const uint8_t*>(down.GetTensorRawData()), scratch_ptr,
            reinterpret_cast<bf16*>(out.GetTensorMutableData<Ort::BFloat16_t>()), down_global_scale);
      } else {
        throw std::invalid_argument("MtpMoe: unsupported expert weight layout");
      }
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

constexpr auto kBF = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
constexpr auto kF32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr auto kI64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
constexpr auto kAny = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

template <typename Derived, typename State>
struct MtpOp : Ort::CustomOpBase<Derived, State, true> {
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return Derived::kIn.size(); }
  ONNXTensorElementDataType GetInputType(size_t i) const { return Derived::kIn[i]; }
  size_t GetOutputTypeCount() const { return Derived::kOut.size(); }
  ONNXTensorElementDataType GetOutputType(size_t i) const { return Derived::kOut[i]; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try {
      *kernel = new State(api, info);
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what());
    }
  }
};

struct ArgMaxOp : MtpOp<ArgMaxOp, ArgMaxState> {
  static constexpr std::array<ONNXTensorElementDataType, 1> kIn = {kBF};
  static constexpr std::array<ONNXTensorElementDataType, 2> kOut = {kI64, kF32};
  const char* GetName() const { return "Bf16ArgMax"; }
};
struct MoeOp : MtpOp<MoeOp, MoeState> {
  static constexpr std::array<ONNXTensorElementDataType, 4> kIn = {kBF, kBF, kAny, kAny};
  static constexpr std::array<ONNXTensorElementDataType, 2> kOut = {kBF, kF32};
  const char* GetName() const { return "MtpMoe"; }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ORT API");
  Ort::InitApi(api);
  try {
    static ArgMaxOp argmax;
    static MoeOp moe;
    static Ort::CustomOpDomain domain("qwen.mtp");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&argmax);
      domain.Add(&moe);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_FAIL, e.what());
  }
}
