// Row-wise greedy token: idx[r] = argmax_v x[r, v] with numpy semantics (first maximum wins, i.e. the lowest index
// on ties; the first NaN wins over everything), optionally with the maximum value. One kernel, no workspace, no
// atomics: a thread-block cluster of C CTAs per row, partial (value, index) pairs are combined through DSMEM in a
// fixed order, so the result is deterministic and independent of the launch geometry.
#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <algorithm>
#include <cstdint>

namespace skinny {

namespace cg_ = cooperative_groups;

enum class LogitType { kBF16 = 0, kFP16 = 1, kFP32 = 2 };

namespace argmax_detail {

constexpr int kThreads = 1024;
constexpr int kMaxCluster = 16;  // <= 32 lanes of the final warp; > 8 is a non-portable cluster size

template <typename I>
__device__ __forceinline__ bool Better(float av, I ai, float bv, I bi) {
  if (av != av) return (bv == bv) || ai < bi;
  if (bv != bv) return false;
  return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float ToFloat(__half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat(float v) { return v; }

constexpr int kNone = -1;

// Elements are visited in increasing index order, so a strict comparison keeps the lowest index of equal values and
// the first NaN (a NaN is never replaced; it replaces any non-NaN).
template <typename T>
__device__ __forceinline__ void Consider(float& bv, int& bi, T raw, int idx) {
  const float v = ToFloat(raw);
  if (bi == kNone || v > bv || (v != v && bv == bv)) {
    bv = v;
    bi = idx;
  }
}

// Elements per 16-byte load: 8 (16-bit types) or 4 (fp32).
template <typename T>
constexpr int kVec = static_cast<int>(16 / sizeof(T));

template <typename T, bool VEC>
__global__ void __launch_bounds__(kThreads) ArgmaxKernel(const T* __restrict__ x, int64_t V, int64_t* __restrict__ idx_out,
                                                         float* __restrict__ val_out) {
  constexpr int kWarps = kThreads / 32;
  __shared__ float wv[kWarps];
  __shared__ int wi[kWarps];
  __shared__ float cv;
  __shared__ int ci;

  const int C = static_cast<int>(gridDim.x);
  const int c = static_cast<int>(blockIdx.x);
  const int64_t row = blockIdx.y;
  const T* xr = x + row * V;
  float bv = 0.f;
  int bi = kNone;

  // contiguous element range of this CTA (multiple of the vector width so loads stay aligned)
  constexpr int kV = VEC ? kVec<T> : 1;
  const int64_t units = (V + kV - 1) / kV;
  const int64_t per = (units + C - 1) / C;
  const int64_t ubeg = min(units, per * c);
  const int64_t uend = min(units, ubeg + per);
  constexpr int kU = 4;  // loads in flight per thread; elements are still visited in increasing index order
  for (int64_t u0 = ubeg + threadIdx.x; u0 < uend; u0 += static_cast<int64_t>(kThreads) * kU) {
    if constexpr (VEC) {
      uint4 q[kU];
#pragma unroll
      for (int i = 0; i < kU; ++i) {
        const int64_t u = u0 + static_cast<int64_t>(i) * kThreads;
        q[i] = u < uend ? __ldg(reinterpret_cast<const uint4*>(xr + u * kV)) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int i = 0; i < kU; ++i) {
        const int64_t u = u0 + static_cast<int64_t>(i) * kThreads;
        if (u < uend) {
          const T* v = reinterpret_cast<const T*>(&q[i]);
#pragma unroll
          for (int j = 0; j < kV; ++j) Consider(bv, bi, v[j], static_cast<int>(u * kV + j));
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kU; ++i) {
        const int64_t u = u0 + static_cast<int64_t>(i) * kThreads;
        if (u < uend) Consider(bv, bi, xr[u], static_cast<int>(u));
      }
    }
  }
  if (bi == kNone) {  // no elements: loses against every real element
    bv = -INFINITY;
    bi = INT_MAX;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
    if (Better(ov, oi, bv, bi)) {
      bv = ov;
      bi = oi;
    }
  }
  if ((threadIdx.x & 31) == 0) {
    wv[threadIdx.x >> 5] = bv;
    wi[threadIdx.x >> 5] = bi;
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  if (threadIdx.x < 32) {  // warp 0 combines the per-warp results (kWarps == 32)
    float v = wv[lane];
    int i = wi[lane];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, off);
      const int oi = __shfl_xor_sync(0xffffffffu, i, off);
      if (Better(ov, oi, v, i)) {
        v = ov;
        i = oi;
      }
    }
    if (lane == 0) {
      cv = v;
      ci = i;
    }
  }
  cg_::cluster_group cluster = cg_::this_cluster();
  cluster.sync();
  if (c == 0 && threadIdx.x < 32) {  // CTA 0 combines the CTAs of the cluster
    float v = -INFINITY;
    int i = INT_MAX;
    if (lane < C) {
      v = *cluster.map_shared_rank(&cv, lane);
      i = *cluster.map_shared_rank(&ci, lane);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, off);
      const int oi = __shfl_xor_sync(0xffffffffu, i, off);
      if (Better(ov, oi, v, i)) {
        v = ov;
        i = oi;
      }
    }
    if (lane == 0) {
      idx_out[row] = i;
      if (val_out != nullptr) val_out[row] = v;
    }
  }
  cluster.sync();
}

template <typename T>
cudaError_t LaunchT(const T* x, int64_t rows, int64_t V, int64_t* idx, float* val, int cluster, cudaStream_t s) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(cluster, static_cast<unsigned>(rows), 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cluster;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  if (cluster > 8) {
    static const cudaError_t a1 = cudaFuncSetAttribute(ArgmaxKernel<T, true>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    static const cudaError_t a2 = cudaFuncSetAttribute(ArgmaxKernel<T, false>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (a1 != cudaSuccess) return a1;
    if (a2 != cudaSuccess) return a2;
  }
  const bool vec = (V % kVec<T>) == 0 && (reinterpret_cast<uintptr_t>(x) % 16) == 0;
  if (vec) return cudaLaunchKernelEx(&cfg, ArgmaxKernel<T, true>, x, V, idx, val);
  return cudaLaunchKernelEx(&cfg, ArgmaxKernel<T, false>, x, V, idx, val);
}

}  // namespace argmax_detail

// x: [rows, V] row-major; idx: int64[rows]; val: float[rows] or nullptr.
inline cudaError_t LaunchArgmaxRows(LogitType type, const void* x, int64_t rows, int64_t V, int64_t* idx, float* val,
                                    cudaStream_t s, int cluster = 8) {
  if (rows < 1 || V < 1 || V >= INT_MAX || cluster < 1 || cluster > argmax_detail::kMaxCluster) return cudaErrorInvalidValue;
  switch (type) {
    case LogitType::kBF16:
      return argmax_detail::LaunchT(static_cast<const __nv_bfloat16*>(x), rows, V, idx, val, cluster, s);
    case LogitType::kFP16:
      return argmax_detail::LaunchT(static_cast<const __half*>(x), rows, V, idx, val, cluster, s);
    case LogitType::kFP32:
      return argmax_detail::LaunchT(static_cast<const float*>(x), rows, V, idx, val, cluster, s);
  }
  return cudaErrorInvalidValue;
}

}  // namespace skinny
