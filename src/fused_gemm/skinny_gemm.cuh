// Bandwidth-oriented BF16 skinny GEMM (M <= 4 rows): y[M,N] = x[M,K] @ W[K,N], W row-major.
//
// Each CTA owns a tile of NCOL = 8*TX output columns and one 1/C slice of K. A thread loads
// 16 bytes (8 columns) of one weight row at a time, U rows in flight, FP32 accumulation.
// Partial sums are reduced in shared memory across the K-lanes of the CTA and, for C > 1,
// across the C CTAs of a thread-block cluster through distributed shared memory in a fixed
// order, so results are deterministic and no workspace or second kernel is needed.
// The result is split over up to kMaxOut outputs (concatenated column ranges).
#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "common/sigmoid_acc.cuh"

#include <algorithm>
#include <cstdint>

namespace skinny {

using bf16 = __nv_bfloat16;
namespace cg = cooperative_groups;

#ifdef SKINNY_TRACE
__device__ unsigned long long g_trace[4096 * 4];
__device__ __forceinline__ unsigned long long Now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define TRACE(i) \
  if (threadIdx.x == 0) g_trace[(blockIdx.y * gridDim.x + blockIdx.x) * 4 + (i)] = Now()
#else
#define TRACE(i)
#endif

constexpr int kMaxOut = 8;
constexpr int kFuseMaxK = 1024;
constexpr int kMaxSeg = 4;

// A segment is one weight matrix W[K, n]; its columns are split over consecutive outputs.
// Segments with n % 8 != 0 (n % 4 == 0) use 8-byte loads and a dedicated CTA tile.
struct Params {
  const bf16* x;
  int K;
  int nseg;
  const bf16* w[kMaxSeg];
  const uint8_t* w_q[kMaxSeg];
  const float* w_scale[kMaxSeg];
  int w_mode[kMaxSeg];  // 0 = BF16, 1 = FP8 E4M3 [K, N], 2 = NVFP4 block-16 [K, N * 9 / 16]
  int n[kMaxSeg];
  int tile_begin[kMaxSeg + 1];  // first CTA tile (blockIdx.x) of each segment
  int out_begin[kMaxSeg + 1];   // first output of each segment
  bf16* out[kMaxOut];
  int col_begin[kMaxOut];  // first column of the output inside its segment
  int width[kMaxOut];
  // Shared-expert down projection (FUSE): x[k] = swish_mul(x[k], x2[k]) on the fly and
  // out = routed + bf16(bf16(acc) * bf16(sigmoid(bf16(dot(gx, gw))))) (see SharedExpertDown).
  const bf16* x2;
  const bf16* gx;
  const bf16* gw;
  const bf16* routed;
  int gk;   // gate vector length; row m of gx starts at gx + m * gk
  // programmatic dependent launch: 1 = launch_dependents at kernel start, 2 = after the K loop (0 = off). Only has an effect when the next
  // kernel in the stream was launched with the programmatic-serialization attribute (the qwen.fused_hc ops are).
  int trigger;
};

__device__ __forceinline__ void GridTrigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ float Round(float v) { return __bfloat162float(__float2bfloat16(v)); }
// BF16 value of the ORT Sigmoid node (accurate fp32 evaluation, one rounding; provider commit 453ac97a68)
__device__ __forceinline__ float SigmoidBf16(float v) { return qwen_acc::SigmoidBf16(v); }
// bf16(swish(g) * u) with swish(g) = bf16(g * bf16(sigmoid(g))): the rounding points of the unfused Swish -> Mul chain
__device__ __forceinline__ float SwishMulValue(bf16 g, bf16 u) {
  const float v = __bfloat162float(g);
  return Round(qwen_acc::SwishBf16(v) * __bfloat162float(u));
}

__device__ __forceinline__ uint4 LoadWeights(const bf16* p) {
  uint4 v;
#ifdef SKINNY_PLAIN_LOAD
  v = __ldg(reinterpret_cast<const uint4*>(p));
#else
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
#endif
  return v;
}

__device__ __forceinline__ uint2 LoadWeights4(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

__device__ __forceinline__ uint32_t LoadWeightsU32(const void* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ void FmaWord(float (&acc)[8], int base, uint32_t word, float xv) {
  acc[base] = fmaf(__uint_as_float(word << 16), xv, acc[base]);
  acc[base + 1] = fmaf(__uint_as_float(word & 0xffff0000u), xv, acc[base + 1]);
}

__device__ __forceinline__ uint32_t Fp8x2ToBf16x2Bits(uint16_t fp8x2) {
  const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(fp8x2), __NV_E4M3);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
  const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
  return *reinterpret_cast<const uint32_t*>(&b);
}

// Every finite E4M3FN value is 100% bit-exact in BFloat16.
__device__ __forceinline__ uint4 UnpackFp8U2ToBf16U4(const uint2& h) {
  uint4 out;
  out.x = Fp8x2ToBf16x2Bits(static_cast<uint16_t>(h.x & 0xffffu));
  out.y = Fp8x2ToBf16x2Bits(static_cast<uint16_t>(h.x >> 16));
  out.z = Fp8x2ToBf16x2Bits(static_cast<uint16_t>(h.y & 0xffffu));
  out.w = Fp8x2ToBf16x2Bits(static_cast<uint16_t>(h.y >> 16));
  return out;
}

__device__ __forceinline__ uint4 LoadWeightsFp8AsBf16(const uint8_t* p) {
  return UnpackFp8U2ToBf16U4(LoadWeights4(p));
}

__device__ __forceinline__ float E2M1ToFloat(uint32_t nib) {
  const uint32_t m = nib & 7u;
  const uint32_t exp_mant = (m == 0u) ? 0u : ((m == 1u ? 252u : (m + 252u)) << 22);
  return __uint_as_float(((nib & 8u) << 28) | exp_mant);
}

// Every product of E2M1 and E4M3FN has <= 6 significand bits and exponent in [-10, 10], so it is 100% bit-exact in BFloat16.
__device__ __forceinline__ uint4 UnpackFp4ToBf16U4(uint32_t pack8, uint8_t s_fp8) {
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(s_fp8), __NV_E4M3);
  const float s = __half2float(*reinterpret_cast<const __half*>(&hr));
  uint32_t words[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float f0 = E2M1ToFloat((pack8 >> (8 * i)) & 0xfu) * s;
    const float f1 = E2M1ToFloat((pack8 >> (8 * i + 4)) & 0xfu) * s;
    const __nv_bfloat162 b = __floats2bfloat162_rn(f0, f1);
    words[i] = *reinterpret_cast<const uint32_t*>(&b);
  }
  return make_uint4(words[0], words[1], words[2], words[3]);
}

__device__ __forceinline__ uint4 LoadWeightsFp4AsBf16(const uint8_t* row_p, int N, int col) {
  const uint32_t pack8 = LoadWeightsU32(row_p + (col >> 1));
  const uint8_t s_fp8 = row_p[(N >> 1) + (col >> 4)];
  return UnpackFp4ToBf16U4(pack8, s_fp8);
}

__device__ __forceinline__ void Store(const Params& p, int seg, int m, int n, float v) {
  int j = p.out_begin[seg];
  const int end = p.out_begin[seg + 1];
  for (int i = j + 1; i < end; ++i) {
    if (n >= p.col_begin[i]) j = i;
  }
  p.out[j][static_cast<size_t>(m) * p.width[j] + (n - p.col_begin[j])] = __float2bfloat16(v);
}

template <int M, bool FUSE>
__device__ __forceinline__ float GateOf(const float* gate, int m) {
  float g = gate[0];
  if constexpr (FUSE) {
#pragma unroll
    for (int j = 1; j < M; ++j) g = (m == j) ? gate[j] : g;
  }
  return g;
}

// NARROW: a 4-column segment; every thread is a K lane loading 8 bytes per row.
template <bool FUSE>
__device__ __forceinline__ void StoreOut(const Params& p, int seg, int m, int n, float v, float gate) {
  if (p.w_scale[seg] != nullptr) v *= p.w_scale[seg][n];
  if constexpr (FUSE) {
    const float shared = Round(Round(v) * gate);
    const size_t o = static_cast<size_t>(m) * p.n[seg] + n;
    p.out[0][o] = __float2bfloat16(__bfloat162float(p.routed[o]) + shared);
  } else {
    Store(p, seg, m, n, v);
  }
}

template <int M, int TX, int THREADS, int C, int U, bool NARROW, bool FUSE>
__device__ __forceinline__ void GemvBody(const Params& p, int seg, int tile) {
  constexpr int TXE = NARROW ? 1 : TX;
  constexpr int NCOL = NARROW ? 4 : TX * 8;
  constexpr int LANES = THREADS / TXE;
  constexpr int kWarps = THREADS / 32;
  __shared__ __align__(16) float red[kWarps][M][NCOL > 4 ? NCOL : 8];
  __shared__ __align__(16) float part[M * NCOL];
  __shared__ float gate_partial[FUSE ? M : 1][kWarps];
  __shared__ float xs[FUSE ? M * kFuseMaxK : 1];

  TRACE(0);
  const int tid = threadIdx.x;
  const int N = p.n[seg];
  const bf16* w = p.w[seg];
  const int tile_col = tile * NCOL;
  const int tx = NARROW ? 0 : tid % TX;
  const int lane_k = tid / TXE;
  const int col = tile_col + tx * 8;
  const bool valid = NARROW || col < N;
  const int kper = (p.K + C - 1) / C;
  const int kbeg = static_cast<int>(blockIdx.y) * kper;
  const int kend = min(p.K, kbeg + kper);

  float acc[M][8];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[m][i] = 0.f;

  const bf16* wp0 = w + col;
  const int w_mode = p.w_mode[seg];
  const uint8_t* wq = p.w_q[seg];
  const size_t fp4_row_bytes = static_cast<size_t>(N) * 9 / 16;
  uint4 vpre[FUSE ? U : 1];
  uint2 vpre_fp8[FUSE ? U : 1];
  uint32_t vpre_fp4[FUSE ? U : 1];
  uint8_t vpre_fp4_s[FUSE ? U : 1];
  static_assert(!FUSE || M * kFuseMaxK * 4 <= 16 * 1024, "fused xs buffer");
  if constexpr (FUSE) {
    // First weight loads go out before the prologue below so their latency overlaps it.
    if (w_mode == 0) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int kk = kbeg + lane_k + u * LANES;
        vpre[u] = (valid && kk < kend) ? LoadWeights(wp0 + static_cast<size_t>(kk) * N) : make_uint4(0, 0, 0, 0);
      }
    } else if (w_mode == 1) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int kk = kbeg + lane_k + u * LANES;
        vpre_fp8[u] = (valid && kk < kend) ? LoadWeights4(wq + static_cast<size_t>(kk) * N + col) : make_uint2(0, 0);
      }
    } else {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int kk = kbeg + lane_k + u * LANES;
        const uint8_t* row_p = wq + static_cast<size_t>(kk) * fp4_row_bytes;
        vpre_fp4[u] = (valid && kk < kend) ? LoadWeightsU32(row_p + (col >> 1)) : 0u;
        vpre_fp4_s[u] = (valid && kk < kend) ? row_p[(N >> 1) + (col >> 4)] : 0u;
      }
    }
    // x = swish_mul(g, u) for this CTA's K slice, and the gate-logit partial sums.
    // rows are interleaved in the loops (independent per-row chains, same arithmetic as the M == 1 kernel)
    for (int i = kbeg + tid; i < kend; i += THREADS) {
      bf16 gv[M], uv[M];
#pragma unroll
      for (int m = 0; m < M; ++m) {
        gv[m] = p.x[static_cast<size_t>(m) * p.K + i];
        uv[m] = p.x2[static_cast<size_t>(m) * p.K + i];
      }
#pragma unroll
      for (int m = 0; m < M; ++m) xs[m * kFuseMaxK + i - kbeg] = SwishMulValue(gv[m], uv[m]);
    }
    float g[M];
#pragma unroll
    for (int m = 0; m < M; ++m) g[m] = 0.f;
#pragma unroll 4
    for (int i = tid; i < p.gk; i += THREADS) {
      const float w = __bfloat162float(p.gw[i]);
#pragma unroll
      for (int m = 0; m < M; ++m) g[m] += __bfloat162float(p.gx[static_cast<size_t>(m) * p.gk + i]) * w;
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) g[m] += __shfl_xor_sync(0xffffffffu, g[m], off);
      if ((tid & 31) == 0) gate_partial[m][tid >> 5] = g[m];
    }
    __syncthreads();
  }
  if (valid) {
    const bf16* wp = w + col;
    const size_t row_stride = N;
    if (NARROW || w_mode == 0) {
      for (int k = kbeg + lane_k; k < kend; k += LANES * U) {
        uint4 v[U];
        float xv[U][M];
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const int kk = k + u * LANES;
          v[u] = make_uint4(0, 0, 0, 0);
#pragma unroll
          for (int m = 0; m < M; ++m) xv[u][m] = 0.f;
          if (FUSE && k == kbeg + lane_k) {
            v[u] = vpre[FUSE ? u : 0];
            if (kk < kend) {
#pragma unroll
              for (int m = 0; m < M; ++m) xv[u][m] = xs[m * kFuseMaxK + kk - kbeg];
            }
          } else if (kk < kend) {
            if constexpr (NARROW) {
              const uint2 h = LoadWeights4(wp + static_cast<size_t>(kk) * row_stride);
              v[u].x = h.x;
              v[u].y = h.y;
            } else {
              v[u] = LoadWeights(wp + static_cast<size_t>(kk) * row_stride);
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
              if constexpr (FUSE) {
                xv[u][m] = xs[m * kFuseMaxK + kk - kbeg];
              } else {
                xv[u][m] = __bfloat162float(p.x[static_cast<size_t>(m) * p.K + kk]);
              }
            }
          }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
#pragma unroll
          for (int m = 0; m < M; ++m) {
            FmaWord(acc[m], 0, v[u].x, xv[u][m]);
            FmaWord(acc[m], 2, v[u].y, xv[u][m]);
            if constexpr (!NARROW) {
              FmaWord(acc[m], 4, v[u].z, xv[u][m]);
              FmaWord(acc[m], 6, v[u].w, xv[u][m]);
            }
          }
        }
      }
    } else if (w_mode == 1) {
      const uint8_t* wp_fp8 = wq + col;
      for (int k = kbeg + lane_k; k < kend; k += LANES * U) {
        uint2 raw[U];
        float xv[U][M];
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const int kk = k + u * LANES;
          raw[u] = make_uint2(0, 0);
#pragma unroll
          for (int m = 0; m < M; ++m) xv[u][m] = 0.f;
          if (FUSE && k == kbeg + lane_k) {
            raw[u] = vpre_fp8[FUSE ? u : 0];
            if (kk < kend) {
#pragma unroll
              for (int m = 0; m < M; ++m) xv[u][m] = xs[m * kFuseMaxK + kk - kbeg];
            }
          } else if (kk < kend) {
            raw[u] = LoadWeights4(wp_fp8 + static_cast<size_t>(kk) * row_stride);
#pragma unroll
            for (int m = 0; m < M; ++m) {
              if constexpr (FUSE) {
                xv[u][m] = xs[m * kFuseMaxK + kk - kbeg];
              } else {
                xv[u][m] = __bfloat162float(p.x[static_cast<size_t>(m) * p.K + kk]);
              }
            }
          }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const uint4 v = UnpackFp8U2ToBf16U4(raw[u]);
#pragma unroll
          for (int m = 0; m < M; ++m) {
            FmaWord(acc[m], 0, v.x, xv[u][m]);
            FmaWord(acc[m], 2, v.y, xv[u][m]);
            FmaWord(acc[m], 4, v.z, xv[u][m]);
            FmaWord(acc[m], 6, v.w, xv[u][m]);
          }
        }
      }
    } else {
      for (int k = kbeg + lane_k; k < kend; k += LANES * U) {
        uint32_t raw_p[U];
        uint8_t raw_s[U];
        float xv[U][M];
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const int kk = k + u * LANES;
          raw_p[u] = 0u;
          raw_s[u] = 0u;
#pragma unroll
          for (int m = 0; m < M; ++m) xv[u][m] = 0.f;
          if (FUSE && k == kbeg + lane_k) {
            raw_p[u] = vpre_fp4[FUSE ? u : 0];
            raw_s[u] = vpre_fp4_s[FUSE ? u : 0];
            if (kk < kend) {
#pragma unroll
              for (int m = 0; m < M; ++m) xv[u][m] = xs[m * kFuseMaxK + kk - kbeg];
            }
          } else if (kk < kend) {
            const uint8_t* row_p = wq + static_cast<size_t>(kk) * fp4_row_bytes;
            raw_p[u] = LoadWeightsU32(row_p + (col >> 1));
            raw_s[u] = row_p[(N >> 1) + (col >> 4)];
#pragma unroll
            for (int m = 0; m < M; ++m) {
              if constexpr (FUSE) {
                xv[u][m] = xs[m * kFuseMaxK + kk - kbeg];
              } else {
                xv[u][m] = __bfloat162float(p.x[static_cast<size_t>(m) * p.K + kk]);
              }
            }
          }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const uint4 v = UnpackFp4ToBf16U4(raw_p[u], raw_s[u]);
#pragma unroll
          for (int m = 0; m < M; ++m) {
            FmaWord(acc[m], 0, v.x, xv[u][m]);
            FmaWord(acc[m], 2, v.y, xv[u][m]);
            FmaWord(acc[m], 4, v.z, xv[u][m]);
            FmaWord(acc[m], 6, v.w, xv[u][m]);
          }
        }
      }
    }
  }

  TRACE(1);
  if (p.trigger == 2) GridTrigger();
  // Lanes of a warp that share tx hold partial sums of the same columns: shuffle-reduce them
  // (fixed butterfly order), then add the per-warp results in warp order.
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int i = 0; i < (NARROW ? 4 : 8); ++i)
#pragma unroll
      for (int off = TXE; off < 32; off <<= 1) acc[m][i] += __shfl_xor_sync(0xffffffffu, acc[m][i], off);
  const int warp = tid >> 5;
  if ((tid & 31) < TXE) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float4* dst = reinterpret_cast<float4*>(&red[warp][m][(tid & 31) * 8]);
      if constexpr (NARROW) {
        dst[0] = make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
      } else {
        dst[0] = make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
        dst[1] = make_float4(acc[m][4], acc[m][5], acc[m][6], acc[m][7]);
      }
    }
  }
  __syncthreads();

  float gate[FUSE ? M : 1];
  if constexpr (FUSE) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float logit = 0.f;
#pragma unroll
      for (int t = 0; t < kWarps; ++t) logit += gate_partial[m][t];
      gate[m] = SigmoidBf16(Round(logit));
    }
  } else {
    gate[0] = 0.f;
  }
  for (int e = tid; e < M * NCOL; e += THREADS) {
    const int m = e / NCOL, c = e % NCOL;
    float sum = 0.f;
#pragma unroll
    for (int t = 0; t < kWarps; ++t) sum += red[t][m][c];
    if constexpr (C == 1) {
      if (tile_col + c < N) StoreOut<FUSE>(p, seg, m, tile_col + c, sum, GateOf<M, FUSE>(gate, m));
    } else {
      part[e] = sum;
    }
  }
  if constexpr (C > 1) {
    TRACE(2);
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    TRACE(3);
    const unsigned rank = cluster.block_rank();
    constexpr int kTotal = M * NCOL;
    constexpr int kChunk = (kTotal + C - 1) / C;
    for (int i = tid; i < kChunk; i += THREADS) {
      const int e = static_cast<int>(rank) * kChunk + i;
      if (e >= kTotal) break;
      float sum = 0.f;
#pragma unroll
      for (int r = 0; r < C; ++r) sum += cluster.map_shared_rank(part, r)[e];
      const int m = e / NCOL, c = e % NCOL;
      if (tile_col + c < N) StoreOut<FUSE>(p, seg, m, tile_col + c, sum, GateOf<M, FUSE>(gate, m));
    }
    cluster.sync();
  }
}

template <int M, int TX, int THREADS, int C, int U, bool FUSE = false>
__global__ void __launch_bounds__(THREADS) GemvKernel(Params p) {
  if (p.trigger == 1) GridTrigger();
  const int bx = blockIdx.x;
  int seg = 0;
#pragma unroll
  for (int i = 1; i < kMaxSeg; ++i) seg += (i < p.nseg && bx >= p.tile_begin[i]) ? 1 : 0;
  if constexpr (FUSE) {
    GemvBody<M, TX, THREADS, C, U, false, FUSE>(p, seg, bx - p.tile_begin[seg]);  // single segment, N % 8 == 0
  } else if ((p.n[seg] & 7) != 0) {
    GemvBody<M, TX, THREADS, C, U, true, FUSE>(p, seg, bx - p.tile_begin[seg]);
  } else {
    GemvBody<M, TX, THREADS, C, U, false, FUSE>(p, seg, bx - p.tile_begin[seg]);
  }
}

template <int M, int TX, int THREADS, int C, int U, bool FUSE = false>
cudaError_t LaunchT(Params p, cudaStream_t stream) {
  constexpr int NCOL = TX * 8;
  if constexpr ((THREADS / 32) * M * NCOL * 4 + (FUSE ? M * kFuseMaxK * 4 : 0) > 40 * 1024) {
    return cudaErrorInvalidValue;
  } else {
  int tiles = 0;
  for (int s = 0; s < p.nseg; ++s) {
    p.tile_begin[s] = tiles;
    tiles += (p.n[s] & 7) ? 1 : (p.n[s] + NCOL - 1) / NCOL;
  }
  p.tile_begin[p.nseg] = tiles;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(p.tile_begin[p.nseg], C, 1);
  cfg.blockDim = dim3(THREADS, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  int n_attr = 0;
  if (C > 1) {
    attr[n_attr].id = cudaLaunchAttributeClusterDimension;
    attr[n_attr].val.clusterDim.x = 1;
    attr[n_attr].val.clusterDim.y = C;
    attr[n_attr].val.clusterDim.z = 1;
    ++n_attr;
  }
  cfg.attrs = attr;
  cfg.numAttrs = n_attr;
  if constexpr (C > 8) {
    static const cudaError_t attr_status = cudaFuncSetAttribute(
        GemvKernel<M, TX, THREADS, C, U, FUSE>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (attr_status != cudaSuccess) return attr_status;
  }
  return cudaLaunchKernelEx(&cfg, GemvKernel<M, TX, THREADS, C, U, FUSE>, p);
  }
}

struct Config {
  int tx = 0;       // threads along N (x8 columns each): 4, 8, 16, 32
  int cluster = 1;  // K split across a cluster: 1, 2, 4, 8
  int threads = 256;
  int unroll = 8;
};

template <int M, int TX, int THREADS, int U>
cudaError_t DispatchC(const Params& p, int c, cudaStream_t s) {
  switch (c) {
    case 1: return LaunchT<M, TX, THREADS, 1, U>(p, s);
    case 2: return LaunchT<M, TX, THREADS, 2, U>(p, s);
    case 4: return LaunchT<M, TX, THREADS, 4, U>(p, s);
    case 8: return LaunchT<M, TX, THREADS, 8, U>(p, s);
    case 16: return LaunchT<M, TX, THREADS, 16, U>(p, s);
    default: return cudaErrorInvalidValue;
  }
}

template <int M, int THREADS, int U>
cudaError_t DispatchTX(const Params& p, const Config& cfg, cudaStream_t s) {
  switch (cfg.tx) {
    case 1: return DispatchC<M, 1, THREADS, U>(p, cfg.cluster, s);
    case 2: return DispatchC<M, 2, THREADS, U>(p, cfg.cluster, s);
    case 4: return DispatchC<M, 4, THREADS, U>(p, cfg.cluster, s);
    case 8: return DispatchC<M, 8, THREADS, U>(p, cfg.cluster, s);
    case 16: return DispatchC<M, 16, THREADS, U>(p, cfg.cluster, s);
    case 32: return DispatchC<M, 32, THREADS, U>(p, cfg.cluster, s);
    default: return cudaErrorInvalidValue;
  }
}

template <int M>
cudaError_t DispatchThreads(const Params& p, const Config& cfg, cudaStream_t s) {
  if (cfg.threads == 256 && cfg.unroll == 8) return DispatchTX<M, 256, 8>(p, cfg, s);
  if (cfg.threads == 256 && cfg.unroll == 4) return DispatchTX<M, 256, 4>(p, cfg, s);
  if (cfg.threads == 128 && cfg.unroll == 8) return DispatchTX<M, 128, 8>(p, cfg, s);
  if (cfg.threads == 512 && cfg.unroll == 4) return DispatchTX<M, 512, 4>(p, cfg, s);
  if (cfg.threads == 512 && cfg.unroll == 8) return DispatchTX<M, 512, 8>(p, cfg, s);
  if (cfg.threads == 1024 && cfg.unroll == 4) return DispatchTX<M, 1024, 4>(p, cfg, s);
  if (cfg.threads == 1024 && cfg.unroll == 8) return DispatchTX<M, 1024, 8>(p, cfg, s);
  return cudaErrorInvalidValue;
}

inline cudaError_t LaunchRows(int m, const Params& p, const Config& cfg, cudaStream_t s) {
  switch (m) {
    case 1: return DispatchThreads<1>(p, cfg, s);
#ifndef SKINNY_M1_ONLY
    case 2: return DispatchThreads<2>(p, cfg, s);
#endif
    default: return cudaErrorInvalidValue;
  }
}

// The reduction order of a row only depends on the config (never on the number of rows), so a row block
// that does not fit the static shared memory of a config can be split into smaller row blocks with
// bit-identical results (the weights are then streamed once per block).
inline bool RowsFit(int m, const Config& cfg) { return (cfg.threads / 32) * m * cfg.tx * 8 * 4 <= 40 * 1024; }

inline cudaError_t Launch(int m, const Params& p, const Config& cfg, cudaStream_t s) {
  if (m < 1) return cudaErrorInvalidValue;
  int done = 0;
  while (done < m) {
    int chunk = std::min(m - done, 2);
    while (chunk > 1 && !RowsFit(chunk, cfg)) --chunk;
    Params q = p;
    q.x = p.x + static_cast<size_t>(done) * p.K;
    if (done + chunk < m) q.trigger = 0;  // only the last row block may release PDL dependents
    for (int o = 0; o < p.out_begin[p.nseg]; ++o) q.out[o] = p.out[o] + static_cast<size_t>(done) * p.width[o];
    const cudaError_t err = LaunchRows(chunk, q, cfg, s);
    if (err != cudaSuccess) return err;
    done += chunk;
  }
  return cudaSuccess;
}

// Fused shared-expert down projection (M == 1): see Params::x2/gx/gw/routed.
template <int M, int THREADS, int C>
cudaError_t FusedTX(const Params& p, int tx, cudaStream_t s) {
  switch (tx) {
    case 4: return LaunchT<M, 4, THREADS, C, 8, true>(p, s);
    case 8: return LaunchT<M, 8, THREADS, C, 8, true>(p, s);
    case 16: return LaunchT<M, 16, THREADS, C, 8, true>(p, s);
    default: return cudaErrorInvalidValue;
  }
}

template <int M>
cudaError_t FusedRows(const Params& p, const Config& cfg, cudaStream_t s) {
  if (cfg.unroll != 8) return cudaErrorInvalidValue;
  if (cfg.threads == 256 && cfg.cluster == 1) return FusedTX<M, 256, 1>(p, cfg.tx, s);
  if (cfg.threads == 512 && cfg.cluster == 1) return FusedTX<M, 512, 1>(p, cfg.tx, s);
  if (cfg.threads == 256 && cfg.cluster == 2) return FusedTX<M, 256, 2>(p, cfg.tx, s);
  if (cfg.threads == 512 && cfg.cluster == 2) return FusedTX<M, 512, 2>(p, cfg.tx, s);
  return cudaErrorInvalidValue;
}

// m rows (1..4); p.x/x2 are [m, K], gx is [m, gk], routed/out are [m, N].
inline cudaError_t LaunchFused(int m, const Params& p, const Config& cfg, cudaStream_t s) {
  switch (m) {
    case 1: return FusedRows<1>(p, cfg, s);
#ifndef SKINNY_M1_ONLY
    case 2: return FusedRows<2>(p, cfg, s);
#endif
    default: return cudaErrorInvalidValue;
  }
}
inline cudaError_t LaunchFused(const Params& p, const Config& cfg, cudaStream_t s) { return LaunchFused(1, p, cfg, s); }

}  // namespace skinny
