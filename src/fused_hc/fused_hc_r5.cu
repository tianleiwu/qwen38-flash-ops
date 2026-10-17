// r3: fused hyperconnection (HC) block ops for the Qwen3.8 S=1 / S=2 (R in {1,2} rows) capture graphs. Same ops, op names, attributes and
// bit-exact results as fused_hc_r2.cu (HcDown / HcPostDown / HcUpPre / HcMix* / HcBlock / HcPostBlock, domain qwen.fused_hc); what changed is how
// the chain is executed (see fused_hc/REPORT_R3.md):
//   * no thread-block clusters: the down GEMV's 8 K-split partial sums go to a scratch row and are reduced (fixed order) by the consumer;
//   * pre-tiled weights (optimize_v12_hc.py): w_down / w_up are accepted as [160, 20480] tile images (one tile = the shared-memory image of
//     one CTA) and streamed with one bulk copy (cp.async.bulk + mbarrier) instead of 2560 scattered 16-byte cp.async per CTA; the original
//     [K, N] layout still works (cp.async path);
//   * HcBlock / HcPostBlock = ONE kernel per block for tiled weights and rows in {1, 2}: all weights of both phases are copied into shared
//     memory up front, the phases hand over through a data-carrying (value, launch-tag) table instead of a kernel boundary or grid barrier;
//   * activation loads are volatile asm (never hoisted above griddepcontrol.wait), so the op can be launched early (PDL) behind a producer that
//     calls launch_dependents early (libqwen_fused_gemm_r3.so `trigger`).
// Every intermediate is rounded to BF16 at the same points as the unfused chain; the fp32 reduction trees of the GEMVs are those of r2.
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"
#include "common/sigmoid_acc.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

constexpr int kBranches = 4;
constexpr int kHidden = 2560;
constexpr int kStreamDim = kBranches * kHidden;  // 10240
constexpr int kDown = 320;
constexpr int kGate = 4;
constexpr int kSplit = 8;                       // K splits of the down GEMV = cluster size
constexpr int kSliceK = kStreamDim / kSplit;    // 1280 rows of K per CTA
constexpr int kColSlice = 16;                   // down columns per CTA (32 bytes of a weight row)
constexpr int kColSlices = kDown / kColSlice;   // 20
constexpr int kDownGridX = kColSlices + 1;      // + one CTA column for the inject GEMV
constexpr int kThreads = 256;
constexpr int kBlockCtas = kDownGridX * kSplit;  // CTAs of the merged kernel (168)
constexpr int kUpSlice = 16;                    // h values per CTA in HcUpPre
constexpr int kUpBlocks = kHidden / kUpSlice;   // 160
// per-row scratch: SiLU'd hidden[320], bf16-rounded inject logits[4], inv_rms[4]
constexpr int kInjectOff = kDown;
constexpr int kTabRow = kSplit * (kDown + 8) + 8;  // u64 slots per row: [kSplit][328] partials, then 4 inv_rms
constexpr int kTabInv = kSplit * (kDown + 8);
constexpr int kInvOffF = kDown + kGate;  // final-image slot of inv_rms (test aid, see below)
// per-row scratch (floats). Every DN -> UP handoff store covers whole 32-byte L2 sectors (a partially written sector must be
// filled from DRAM when it is read back, ~1.5 us):
//   [0, 328)                       test-aid image of the former layout (hidden, inject logits, inv_rms); only written when final_out != 0
//   kInvOff   [4][8]               inv_rms of branch c at slot 0 of its own sector
//   kAlphaOff [8]                  silu_alpha at slot 0
//   kPartOff  [kSplit][kPartStride] raw per-split partial sums: 320 down + 4 inject (+ 4 zero pad)
constexpr int kInvOff = 328;
constexpr int kAlphaOff = kInvOff + kBranches * 8;
constexpr int kPartOff = kAlphaOff + 8;
constexpr int kPartStride = kDown + 8;
constexpr int kScratchStride = kPartOff + kSplit * kPartStride;
static_assert(kInvOff % 8 == 0 && kPartOff % 8 == 0 && kPartStride % 8 == 0 && kScratchStride % 8 == 0, "sector alignment");

static_assert(kSliceK % 2 == 0 && kHidden % kSliceK == 0, "K slice must live inside one branch");
static_assert(kThreads * 2 * 5 == kHidden, "norm loop expects 5 pair-iterations per thread");
static_assert(kThreads * 5 == kSliceK, "five rows of K per thread");

__device__ __forceinline__ float R(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float U2F(unsigned short u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }

// Matches onnxruntime's OP_Sigmoid<BFloat16> after the accurate-sigmoid fix (provider commit 453ac97a68): fp32 evaluation, one BF16 rounding.
__device__ __forceinline__ float SigmoidBf16(float a) { return qwen_acc::SigmoidBf16(a); }
// Matches the ScaledSiLU contrib kernel's float sigmoid.
__device__ __forceinline__ float SigmoidSilu(float v) {
  if (v >= 0.0f) return 1.0f / (1.0f + expf(-v));
  const float e = expf(v);
  return e / (1.0f + e);
}
// Matches the DivSigmoidMul fused kernel (qwen.fused r5) = ORT Div -> Sigmoid: the caller rounds the result to BF16, so return the BF16 value.
__device__ __forceinline__ float SigmoidGate(float v) { return qwen_acc::SigmoidBf16(v); }

__device__ __forceinline__ void Unpack(const uint4& u, float (&f)[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ void UnpackFp8(const uint2& u, float (&f)[8]) {
  const __half2_raw h0 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u.x & 0xffffu), __NV_E4M3);
  const __half2_raw h1 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u.x >> 16), __NV_E4M3);
  const __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u.y & 0xffffu), __NV_E4M3);
  const __half2_raw h3 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u.y >> 16), __NV_E4M3);
  const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0));
  const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1));
  const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&h2));
  const float2 f3 = __half22float2(*reinterpret_cast<const __half2*>(&h3));
  f[0] = f0.x;
  f[1] = f0.y;
  f[2] = f1.x;
  f[3] = f1.y;
  f[4] = f2.x;
  f[5] = f2.y;
  f[6] = f3.x;
  f[7] = f3.y;
}

// Data-carrying handoff of the merged kernel: a 64-bit slot = (fp32 value, 32-bit launch tag), single-copy atomic, so a consumer that sees
// the current tag sees the value (no flag, fence or atomic round trip). The launch tag = number of completed launches of this node + 1
// (floor(counter / 168) + 1; every CTA bumps the counter once at its very end).
__device__ __forceinline__ void StoreTagged(unsigned long long* p, float v, unsigned tag) {
  const unsigned long long x = (static_cast<unsigned long long>(tag) << 32) | __float_as_uint(v);
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(x) : "memory");
}
__device__ __forceinline__ unsigned long long LdSlot(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
// Loads of data produced by the previous op. `volatile` asm so the compiler can neither hoist them above griddepcontrol.wait (as it may for
// const __restrict__ loads) nor drop them; the L1 is bypassed (.cg).
__device__ __forceinline__ __nv_bfloat162 LdAct2(const bf16* p) {
  uint32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return *reinterpret_cast<__nv_bfloat162*>(&v);
}
__device__ __forceinline__ float LdAct1(const bf16* p) {
  unsigned short v;
  asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(v) : "l"(p) : "memory");
  return __uint_as_float(static_cast<uint32_t>(v) << 16);
}
// Exact butterfly sum over lane bits {first, ..} written as a reduce-scatter. The reference tree is
//   for off in (first, 2*first, .., 16): v[m] += shfl_xor(v[m], off)   for all 8 values m,
// i.e. at every level a lane adds its partner's partial sum of the same value. Here at every level the pair splits the remaining values
// between its two lanes (each keeps the half selected by its lane bit and receives the partner's partial sums of exactly those values), so
// every (value, level) addition is the same pair of operands as in the butterfly (IEEE addition is commutative) while the number of shuffles
// drops from 8 per level to 4, 2, 1, ... Returns the number of values kept per lane in `v` (their index m = see Select*).
template <int kFirstOff>
__device__ __forceinline__ void ReduceScatter8(float (&v)[8], int lane) {
  // level offs: kFirstOff, 2*kFirstOff, ... < 32 ; values held: 8 -> 4 -> 2 -> 1 (-> 1)
  int count = 8;
#pragma unroll
  for (int off = kFirstOff; off < 32; off <<= 1) {
    const bool hi = (lane & off) != 0;
    if (count > 1) {
      const int half = count >> 1;
#pragma unroll
      for (int k = 0; k < half; ++k) {
        const float send = hi ? v[k] : v[k + half];
        const float recv = __shfl_xor_sync(0xffffffffu, send, off);
        v[k] = (hi ? v[k + half] : v[k]) + recv;
      }
      count = half;
    } else {
      v[0] += __shfl_xor_sync(0xffffffffu, v[0], off);
    }
  }
}

__device__ __forceinline__ void GridWait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void GridTrigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ void CpAsync16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem))), "l"(gmem)
               : "memory");
}
__device__ __forceinline__ void CpAsync8(void* smem, const void* gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem))), "l"(gmem)
               : "memory");
}
__device__ __forceinline__ void CpCommit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void CpWait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ uint32_t Smem(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void MbarInit(uint64_t* b, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(Smem(b)), "r"(n) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void MbarExpectTx(uint64_t* b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(Smem(b)), "r"(bytes) : "memory");
}
// 1-D bulk copy global -> this CTA's shared memory, completion signalled on the mbarrier
__device__ __forceinline__ void BulkLoad(void* dst, const void* src, unsigned bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(Smem(dst)),
               "l"(src), "r"(bytes), "r"(Smem(b))
               : "memory");
}
__device__ __forceinline__ void MbarWait(uint64_t* b) {  // first phase only
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n @!p bra WAIT_%=;\n}" ::"r"(Smem(b))
      : "memory");
}

#ifdef HC_CLK  // per-stage SM clock deltas (thread 0), for bench_l2.cu
__device__ long long g_clk[2][512][12];
#define CLK(k, slot, blk) do { if (threadIdx.x == 0) g_clk[k][blk][slot] = clock64(); } while (0)
#else
#define CLK(k, slot, blk) do {} while (0)
#endif
#ifdef HC_TRACE
__device__ unsigned long long g_trace[2][8][512];
__device__ __forceinline__ unsigned long long Gt() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define TRACE(k, slot, blk) do { if (threadIdx.x == 0) g_trace[k][slot][blk] = Gt(); } while (0)
#else
#define TRACE(k, slot, blk) do {} while (0)
#endif

// grid: (kDownGridX, kSplit, rows / NR), cluster (1, kSplit, 1). NR rows (1 or 2) share one CTA, i.e. one stream of
// the weights; every row runs exactly the arithmetic (and reduction order) of the NR == 1 kernel.
// CTA (j, s, z) handles K rows [s*1280, s*1280+1280) of the normalised stream; j < 20 owns down
// columns [j*16, j*16+16), j == 20 owns the four inject columns. The cluster reduces the K splits
// through DSMEM in a fixed order and rank q finalises columns 2q, 2q+1 (down) or q (inject).
template <int NR, bool kFp8 = false>
struct DnSm {
  __align__(16) std::conditional_t<kFp8, uint2, uint4> wsm[10][kThreads];  // down weights (inject CTAs reuse it as uint2[5][256])
  __align__(16) unsigned short nws[kSliceK];        // norm weight slice (applied on the fly in the GEMV)
  float red2[NR][8][kColSlice];                     // GEMV reduction scratch (NR == 1: 4 columns of the inject CTAs use the first slots)
  unsigned short xs[NR][kSliceK];                   // BF16 bit patterns (the normalised stream is BF16)
  float inv_rms_s[NR];
  typename cub::BlockReduce<float, kThreads>::TempStorage reduce_storage;  // NR == 1 (the v10hc path)
  float wagg[NR][kThreads / 32];                    // NR == 2
  __align__(8) uint64_t mb_w, mb_n;
  unsigned tag;                                     // merged kernel: launch tag of the data-carrying handoff
};

template <int NR, bool kFp8 = false>
struct UpSm {
  __align__(16) std::conditional_t<kFp8, uint2, uint4> wsm[10][kThreads];
  float sf[NR][kDown];
  float red[NR][8][kBranches * kUpSlice];
  float sg[NR][kBranches * kUpSlice];
  __align__(16) unsigned short nws[kBranches][kUpSlice];
  __align__(8) uint64_t mb_w;
};

// start of the up phase that does not depend on the previous op: weight tile + norm slice into shared memory
template <int NR, bool kTiled, bool kFp8 = false>
__device__ __forceinline__ void UpPrefetch(UpSm<NR, kFp8>& sm, int hb, const bf16* norm_w, const void* w_up) {
  const int tid = threadIdx.x;
  const int h0 = hb * kUpSlice;
  if constexpr (kTiled) {  // w_up tile hb ([160][10 * 256 * 8] bf16 or fp8) is the shared-memory image of this CTA
    if (tid == 0) {
      constexpr unsigned kTxBytes = kFp8 ? (10 * kThreads * 8) : (10 * kThreads * 16);
      MbarInit(&sm.mb_w, 1);
      MbarExpectTx(&sm.mb_w, kTxBytes);
      BulkLoad(&sm.wsm[0][0], static_cast<const uint8_t*>(w_up) + static_cast<int64_t>(hb) * kTxBytes, kTxBytes, &sm.mb_w);
    }
  } else {
    const int g = tid & 7, c = g >> 1, half = g & 1, rp = tid >> 3;
    const bf16* wu_base = static_cast<const bf16*>(w_up) + static_cast<int64_t>(c) * kHidden + h0 + half * 8;
#pragma unroll
    for (int i = 0; i < 10; ++i) CpAsync16(&sm.wsm[i][tid], wu_base + static_cast<int64_t>(rp + 32 * i) * kStreamDim);
    CpCommit();
  }
  if (tid < kBranches * 2) CpAsync16(&sm.nws[tid >> 1][(tid & 1) * 8], norm_w + (tid >> 1) * kHidden + h0 + (tid & 1) * 8);
  CpCommit();
}

// grid: (kDownGridX, kSplit, rows / NR). NR rows (1 or 2) share one CTA, i.e. one stream of
// the weights; every row runs exactly the arithmetic (and reduction order) of the NR == 1 kernel.
// CTA (j, s, z) handles K rows [s*1280, s*1280+1280) of the normalised stream; j < 20 owns down
// columns [j*16, j*16+16), j == 20 owns the four inject columns. The K splits are written as raw
// partial sums to the scratch row; the up phase reduces them in a fixed order (s = 0..7).
// kMerged: the CTA also prefetches its up-phase weights (UpSm *up_sm, tile up_hb < kUpBlocks).
template <bool kPost, int NR, bool kTiled, bool kMerged, bool kFp8 = false>
__device__ __forceinline__ void DnPhase(DnSm<NR, kFp8>& sm, const bf16* __restrict__ streams, const bf16* __restrict__ branch,
                                        const bf16* __restrict__ gate, const bf16* __restrict__ norm_w,
                                        const void* __restrict__ w_down, const bf16* __restrict__ w_inject,
                                        bf16* __restrict__ streams_out, float* __restrict__ scratch, float epsilon,
                                        float silu_alpha, int trigger_at, UpSm<NR, kFp8>* up_sm, int up_hb, const void* w_up,
                                        unsigned long long* tab, const unsigned long long* ctr,
                                        const float* __restrict__ wd_scale = nullptr) {
  const bool early_w = (trigger_at & 16) != 0;
  unsigned long long c0 = 0;
  if (kMerged && threadIdx.x == 0) c0 = LdSlot(ctr);  // sampled up front; consumed once the stream loads have returned
  const int j = blockIdx.x, s = blockIdx.y;
  const int64_t r0 = static_cast<int64_t>(blockIdx.z) * NR;
  const int tid = threadIdx.x;
  const int c = s * kSliceK / kHidden;
  const int h0 = s * kSliceK - c * kHidden;
  [[maybe_unused]] const int blk = blockIdx.y * kDownGridX + blockIdx.x;
  TRACE(0, 0, blk);
  CLK(0, 0, blk);

  auto& xs = sm.xs;
  auto& inv_rms_s = sm.inv_rms_s;
  auto& wsm = sm.wsm;
  auto& nws = sm.nws;
  auto& red2 = sm.red2;
  using BlockReduce = cub::BlockReduce<float, kThreads>;
  auto& reduce_storage = sm.reduce_storage;
  auto& wagg = sm.wagg;
  uint64_t& mb_w = sm.mb_w;
  uint64_t& mb_n = sm.mb_n;

  // kTiled: w_down is [8 * 20 tiles][10 * 256 * 8] bf16 or fp8, tile (s * 20 + j) is the shared-memory image of CTA (j, s); the bulk copy engine
  // streams it (and the norm slice) without occupying the load/store pipe. w_inject is already contiguous per split.
  if constexpr (kTiled) {
    if (tid == 0) {
      MbarInit(&mb_w, 1);
      MbarInit(&mb_n, 1);
    }
    __syncthreads();
    if (early_w) {  // weights do not depend on the previous op: issue the bulk copies before the dependency resolves
      if (tid == 0) {
        MbarExpectTx(&mb_n, kSliceK * 2);
        BulkLoad(nws, norm_w + c * kHidden + h0, kSliceK * 2, &mb_n);
        if (j < kColSlices) {
          constexpr unsigned kTxBytes = kFp8 ? (10 * kThreads * 8) : (10 * kThreads * 16);
          MbarExpectTx(&mb_w, kTxBytes);
          BulkLoad(&wsm[0][0], static_cast<const uint8_t*>(w_down) + static_cast<int64_t>(s * kColSlices + j) * kTxBytes, kTxBytes, &mb_w);
        } else {
          MbarExpectTx(&mb_w, 5 * kThreads * 8);
          BulkLoad(&wsm[0][0], w_inject + static_cast<int64_t>(s * kSliceK) * kGate, 5 * kThreads * 8, &mb_w);
        }
      }
      if constexpr (kMerged) {
        if (up_hb < kUpBlocks) UpPrefetch<NR, true, kFp8>(*up_sm, up_hb, norm_w, w_up);
      }
    }
  }
  trigger_at &= 15;
  // data produced by the previous op: request it first so it does not queue behind the weight stream
  if (trigger_at == 0) GridTrigger();
  GridWait();
  TRACE(0, 1, blk);
  CLK(0, 1, blk);
  __nv_bfloat162 sv[NR][5], bv[NR][5];
  float gc[NR];
#pragma unroll
  for (int q = 0; q < NR; ++q) {
    const bf16* sp = streams + (r0 + q) * kStreamDim + c * kHidden;
#pragma unroll
    for (int i = 0; i < 5; ++i) sv[q][i] = LdAct2(sp + tid * 2 + 512 * i);
    gc[q] = 0.0f;
    if (kPost) {
      gc[q] = LdAct1(gate + (r0 + q) * kBranches + c);
      const bf16* bp = branch + (r0 + q) * kHidden;
#pragma unroll
      for (int i = 0; i < 5; ++i) bv[q][i] = LdAct2(bp + tid * 2 + 512 * i);
    }
  }

  const int cg = tid & 1, rp = tid >> 1;  // down: 2 x 16B per weight row, 128 row phases
  // weights do not depend on the previous op: stream them into shared memory while the data arrives
  if constexpr (kTiled) {
    if (!early_w) {
      if (tid == 0) {
        MbarExpectTx(&mb_n, kSliceK * 2);
        BulkLoad(nws, norm_w + c * kHidden + h0, kSliceK * 2, &mb_n);
        if (j < kColSlices) {
          constexpr unsigned kTxBytes = kFp8 ? (10 * kThreads * 8) : (10 * kThreads * 16);
          MbarExpectTx(&mb_w, kTxBytes);
          BulkLoad(&wsm[0][0], static_cast<const uint8_t*>(w_down) + static_cast<int64_t>(s * kColSlices + j) * kTxBytes, kTxBytes, &mb_w);
        } else {
          MbarExpectTx(&mb_w, 5 * kThreads * 8);
          BulkLoad(&wsm[0][0], w_inject + static_cast<int64_t>(s * kSliceK) * kGate, 5 * kThreads * 8, &mb_w);
        }
      }
    }
  } else {
  if (tid < kSliceK / 8) CpAsync16(&nws[tid * 8], norm_w + c * kHidden + h0 + tid * 8);
  CpCommit();
  if (j < kColSlices) {
    const bf16* wd_base = static_cast<const bf16*>(w_down) + static_cast<int64_t>(s * kSliceK) * kDown + j * kColSlice + cg * 8;
#pragma unroll
    for (int i = 0; i < 10; ++i) CpAsync16(&wsm[i][tid], wd_base + static_cast<int64_t>(rp + 128 * i) * kDown);
  } else {
    uint2* wi_s = reinterpret_cast<uint2*>(&wsm[0][0]);
#pragma unroll
    for (int i = 0; i < 5; ++i)
      CpAsync8(&wi_s[i * kThreads + tid], w_inject + static_cast<int64_t>(s * kSliceK + tid + 256 * i) * kGate);
  }
  CpCommit();
  }


  float sum_sq[NR];
#pragma unroll
  for (int q = 0; q < NR; ++q) {
    sum_sq[q] = 0.0f;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const int h = tid * 2 + 512 * i;
      float v0 = __bfloat162float(sv[q][i].x), v1 = __bfloat162float(sv[q][i].y);
      if (kPost) {
        v0 = R(v0 + gc[q] * __bfloat162float(bv[q][i].x));
        v1 = R(v1 + gc[q] * __bfloat162float(bv[q][i].y));
      }
      sum_sq[q] += v0 * v0;
      sum_sq[q] += v1 * v1;
      if (h >= h0 && h < h0 + kSliceK) {
        xs[q][h - h0] = __bfloat16_as_ushort(__float2bfloat16(v0));
        xs[q][h - h0 + 1] = __bfloat16_as_ushort(__float2bfloat16(v1));
        if (kPost && j == 0) {
          __nv_bfloat162 o;
          o.x = __float2bfloat16(v0);
          o.y = __float2bfloat16(v1);
          *reinterpret_cast<__nv_bfloat162*>(streams_out + (r0 + q) * kStreamDim + c * kHidden + h) = o;
        }
      }
    }
  }
  if (kMerged && tid == 0) sm.tag = static_cast<unsigned>(c0 / kBlockCtas) + 1;  // published by the barriers below
  TRACE(0, 2, blk);
  CLK(0, 2, blk);
  if constexpr (NR == 1) {
    const float total = BlockReduce(reduce_storage).Sum(sum_sq[0]);
    if (tid == 0) inv_rms_s[0] = rsqrtf(total / static_cast<float>(kHidden) + epsilon);
  } else {
    // cub::BlockReduce<float, 256>::Sum for all rows at once: per-warp shfl_down tree (offsets 1,2,4,8,16), then the warp
    // aggregates are added in warp order 0..7. Same association as cub (bit-identical), one barrier for all rows.
#pragma unroll
    for (int q = 0; q < NR; ++q) {
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) sum_sq[q] += __shfl_down_sync(0xffffffffu, sum_sq[q], off);
      if ((tid & 31) == 0) wagg[q][tid >> 5] = sum_sq[q];
    }
    __syncthreads();
    if (tid < NR) {
      float total = wagg[tid][0];
#pragma unroll
      for (int w = 1; w < kThreads / 32; ++w) total += wagg[tid][w];
      inv_rms_s[tid] = rsqrtf(total / static_cast<float>(kHidden) + epsilon);
    }
  }
  __syncthreads();
  float inv_rms[NR];
#pragma unroll
  for (int q = 0; q < NR; ++q) inv_rms[q] = inv_rms_s[q];
  if constexpr (kMerged) {
    if (j == 0 && h0 == 0 && tid < NR) StoreTagged(tab + (r0 + tid) * kTabRow + kTabInv + c, inv_rms_s[tid], sm.tag);
  } else if (j == 0 && h0 == 0 && tid < 8 * NR) {
    scratch[(r0 + tid / 8) * kScratchStride + kInvOff + c * 8 + (tid & 7)] = (tid & 7) == 0 ? inv_rms_s[tid / 8] : 0.0f;
  } else if (j == 0 && s == 0 && tid >= 64 && tid < 64 + 8 * NR) {
    scratch[(r0 + (tid - 64) / 8) * kScratchStride + kAlphaOff + (tid & 7)] = (tid & 7) == 0 ? silu_alpha : 0.0f;
  }
  TRACE(0, 3, blk);
  CLK(0, 3, blk);
  if constexpr (kTiled) {
    MbarWait(&mb_n);
  } else {
    CpWait<1>();  // norm_w slice landed (this thread's copy); the barrier below publishes it
    __syncthreads();
  }
  // normalised value of slice element kk (BF16-rounded as in the unfused chain), computed where it is consumed
  auto xn = [&](int q, int kk) { return U2F(__bfloat16_as_ushort(__float2bfloat16(U2F(xs[q][kk]) * inv_rms[q] * U2F(nws[kk])))); };
  if constexpr (kTiled) MbarWait(&mb_w); else CpWait<0>();
  if constexpr (kMerged) {
    // up-phase weights: queued behind the down weights (not competing with them); already issued when the weights were early
    if (!early_w && up_hb < kUpBlocks) UpPrefetch<NR, true, kFp8>(*up_sm, up_hb, norm_w, w_up);
  }
  if (trigger_at == 1) GridTrigger();
  TRACE(0, 4, blk);
  CLK(0, 4, blk);

  if (j < kColSlices) {
    float acc[NR][8];
#pragma unroll
    for (int q = 0; q < NR; ++q)
#pragma unroll
      for (int m = 0; m < 8; ++m) acc[q][m] = 0.0f;
#pragma unroll
    for (int i = 0; i < 10; ++i) {
      float wf[8];
      if constexpr (kFp8) UnpackFp8(wsm[i][tid], wf);
      else Unpack(wsm[i][tid], wf);
#pragma unroll
      for (int q = 0; q < NR; ++q) {
        const float xv = xn(q, rp + 128 * i);
#pragma unroll
        for (int m = 0; m < 8; ++m) acc[q][m] = fmaf(xv, wf[m], acc[q][m]);
      }
    }
    // lanes of a warp share cg = lane & 1 and differ in rp: reduce across lane bits 1..4
#pragma unroll
    for (int q = 0; q < NR; ++q) {
      ReduceScatter8<2>(acc[q], tid & 31);
      // kept value: column m = (b3) + 2 * (b2) + 4 * (b1) of this lane's 8-column half; lanes with lane bit 4 clear write
      if (((tid >> 4) & 1) == 0) {
        const int m = ((tid >> 3) & 1) + 2 * ((tid >> 2) & 1) + 4 * ((tid >> 1) & 1);
        red2[q][tid >> 5][cg * 8 + m] = acc[q][0];
      }
    }
    __syncthreads();
    if (tid < NR * kColSlice) {
      const int q = tid / kColSlice, col = tid % kColSlice;
      float v = 0.0f;
#pragma unroll
      for (int w = 0; w < 8; ++w) v += red2[q][w][col];
      if constexpr (kFp8) {
        if (wd_scale != nullptr) v *= wd_scale[j * kColSlice + col];
      }
      if constexpr (kMerged) StoreTagged(tab + (r0 + q) * kTabRow + s * (kDown + 8) + j * kColSlice + col, v, sm.tag);
      else scratch[(r0 + q) * kScratchStride + kPartOff + s * kPartStride + j * kColSlice + col] = v;
    }
  } else {
    float a[NR][kGate];
    const uint2* wi_s = reinterpret_cast<const uint2*>(&wsm[0][0]);
#pragma unroll
    for (int q = 0; q < NR; ++q)
#pragma unroll
      for (int m = 0; m < kGate; ++m) a[q][m] = 0.0f;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
      const uint2 w2 = wi_s[i * kThreads + tid];
#pragma unroll
      for (int q = 0; q < NR; ++q) {
        const float xv = xn(q, tid + 256 * i);
        a[q][0] = fmaf(xv, __uint_as_float(w2.x << 16), a[q][0]);
        a[q][1] = fmaf(xv, __uint_as_float(w2.x & 0xffff0000u), a[q][1]);
        a[q][2] = fmaf(xv, __uint_as_float(w2.y << 16), a[q][2]);
        a[q][3] = fmaf(xv, __uint_as_float(w2.y & 0xffff0000u), a[q][3]);
      }
    }
#pragma unroll
    for (int q = 0; q < NR; ++q) {
#pragma unroll
      for (int m = 0; m < kGate; ++m) {
        float v = a[q][m];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if ((tid & 31) == 0) red2[q][tid >> 5][m] = v;
      }
    }
    __syncthreads();
    if (tid < NR * 8) {  // 8 floats = one full sector (4 inject columns + zero pad)
      const int q = tid >> 3, col = tid & 7;
      float v = 0.0f;
      if (col < kGate) {
#pragma unroll
        for (int w = 0; w < 8; ++w) v += red2[q][w][col];
      }
      if constexpr (kMerged) {
        if (col < kGate) StoreTagged(tab + (r0 + q) * kTabRow + s * (kDown + 8) + kDown + col, v, sm.tag);
      } else {
        scratch[(r0 + q) * kScratchStride + kPartOff + s * kPartStride + kDown + col] = v;
      }
    }
  }
  TRACE(0, 5, blk);
  CLK(0, 5, blk);
  if (trigger_at == 2) GridTrigger();
  TRACE(0, 6, blk);
}

template <bool kPost, int NR, bool kTiled>
__global__ void __launch_bounds__(kThreads)
    HcDownKernel(const bf16* __restrict__ streams, const bf16* __restrict__ branch, const bf16* __restrict__ gate,
                 const bf16* __restrict__ norm_w, const bf16* __restrict__ w_down, const bf16* __restrict__ w_inject,
                 bf16* __restrict__ streams_out, float* __restrict__ scratch, float epsilon, float silu_alpha, int trigger_at) {
  extern __shared__ __align__(128) unsigned char smem_raw[];  // dynamic: DnSm<2> exceeds the 48 KB static limit
  DnSm<NR>& sm = *reinterpret_cast<DnSm<NR>*>(smem_raw);
  DnPhase<kPost, NR, kTiled, false, false>(sm, streams, branch, gate, norm_w, w_down, w_inject, streams_out, scratch, epsilon, silu_alpha,
                                           trigger_at, nullptr, 0, nullptr, nullptr, nullptr);
}

// grid: (kUpBlocks, 1, rows / NR). Block hb owns h in [hb*16, hb*16+16) of every branch; NR rows share the weight stream.
// UpBody runs after the dependency on the down phase resolved (grid wait or software barrier).
// kTagged (merged kernel): the K-split partials / inv_rms are polled from the data-carrying table `tab` instead of read from scratch,
// and the post-mixed stream values are recomputed from the op inputs (branch / gate when kPost) instead of read from streams_out.
template <int NR, bool kTiled, bool kTagged = false, bool kPost = false, bool kFp8 = false>
__device__ __forceinline__ void UpBody(UpSm<NR, kFp8>& sm, int hb, const bf16* __restrict__ streams, float* __restrict__ scratch,
                                       bf16* __restrict__ pre, bf16* __restrict__ gate_out, float reduction_scale,
                                       float gate_div, float gate_mul, int final_out, const unsigned long long* tab = nullptr,
                                       unsigned tag = 0, float silu_alpha = 0.0f, const bf16* __restrict__ branch = nullptr,
                                       const bf16* __restrict__ gate_in = nullptr,
                                       const float* __restrict__ wu_scale = nullptr) {
  const int64_t r0 = static_cast<int64_t>(blockIdx.z) * NR;
  const int tid = threadIdx.x;
  const int h0 = hb * kUpSlice;
  const int g = tid & 7, rp = tid >> 3;
  auto& sf = sm.sf;
  auto& red = sm.red;
  auto& sg = sm.sg;
  auto& wsm = sm.wsm;
  auto& nws = sm.nws;

  // the first NR*16 threads each own one (row, h) output of this block
  const int orow = tid / kUpSlice, ocol = tid % kUpSlice;
  float xv[kBranches] = {0, 0, 0, 0}, inv[kBranches] = {0, 0, 0, 0};
  if constexpr (kTagged) {
    if (tid < NR * kUpSlice) {
#pragma unroll
      for (int b = 0; b < kBranches; ++b) {
        float v = LdAct1(streams + (r0 + orow) * kStreamDim + b * kHidden + h0 + ocol);
        if constexpr (kPost) v = R(v + LdAct1(gate_in + (r0 + orow) * kBranches + b) * LdAct1(branch + (r0 + orow) * kHidden + h0 + ocol));
        xv[b] = v;
      }
    }
    // all rows' slots are requested together (one round trip when everything is there); otherwise re-poll until every slot carries this
    // launch's tag
    const bool need2 = tid < kDown - kThreads;
    const bool needg = hb == 0 && tid < kGate && gate_out != nullptr;
    const bool needi = tid < NR * kUpSlice;
    float p[NR][kSplit], p2[NR][kSplit], lg[NR][kSplit];
    for (;;) {
      unsigned long long u[NR][kSplit], u2[NR][kSplit], ug[NR][kSplit], ui[kBranches];
#pragma unroll
      for (int q = 0; q < NR; ++q) {
        const unsigned long long* row = tab + (r0 + q) * kTabRow;
#pragma unroll
        for (int t = 0; t < kSplit; ++t) u[q][t] = LdSlot(row + t * (kDown + 8) + tid);
        if (need2) {
#pragma unroll
          for (int t = 0; t < kSplit; ++t) u2[q][t] = LdSlot(row + t * (kDown + 8) + kThreads + tid);
        }
        if (needg) {
#pragma unroll
          for (int t = 0; t < kSplit; ++t) ug[q][t] = LdSlot(row + t * (kDown + 8) + kDown + tid);
        }
      }
      if (needi) {
#pragma unroll
        for (int b = 0; b < kBranches; ++b) ui[b] = LdSlot(tab + (r0 + orow) * kTabRow + kTabInv + b);
      }
      bool ok = true;
#pragma unroll
      for (int q = 0; q < NR; ++q) {
#pragma unroll
        for (int t = 0; t < kSplit; ++t) {
          p[q][t] = __uint_as_float(static_cast<unsigned>(u[q][t]));
          ok = ok && static_cast<unsigned>(u[q][t] >> 32) == tag;
        }
        if (need2) {
#pragma unroll
          for (int t = 0; t < kSplit; ++t) {
            p2[q][t] = __uint_as_float(static_cast<unsigned>(u2[q][t]));
            ok = ok && static_cast<unsigned>(u2[q][t] >> 32) == tag;
          }
        }
        if (needg) {
#pragma unroll
          for (int t = 0; t < kSplit; ++t) {
            lg[q][t] = __uint_as_float(static_cast<unsigned>(ug[q][t]));
            ok = ok && static_cast<unsigned>(ug[q][t] >> 32) == tag;
          }
        }
      }
      if (needi) {
#pragma unroll
        for (int b = 0; b < kBranches; ++b) {
          inv[b] = __uint_as_float(static_cast<unsigned>(ui[b]));
          ok = ok && static_cast<unsigned>(ui[b] >> 32) == tag;
        }
      }
      if (ok) break;
    }
#pragma unroll
    for (int q = 0; q < NR; ++q) {
      float v = 0.0f;
#pragma unroll
      for (int t = 0; t < kSplit; ++t) v += p[q][t];
      {
        const float z = R(R(v) * silu_alpha);
        sf[q][tid] = R(z * R(SigmoidSilu(z)));
        if (final_out && hb == 0) scratch[(r0 + q) * kScratchStride + tid] = sf[q][tid];
      }
      if (need2) {
        float v2 = 0.0f;
#pragma unroll
        for (int t = 0; t < kSplit; ++t) v2 += p2[q][t];
        const float z = R(R(v2) * silu_alpha);
        sf[q][tid + kThreads] = R(z * R(SigmoidSilu(z)));
        if (final_out && hb == 0) scratch[(r0 + q) * kScratchStride + tid + kThreads] = sf[q][tid + kThreads];
      }
      if (final_out && hb == 0 && tid < kBranches) {
        scratch[(r0 + q) * kScratchStride + kInvOffF + tid] =
            __uint_as_float(static_cast<unsigned>(LdSlot(tab + (r0 + q) * kTabRow + kTabInv + tid)));
        if (gate_out == nullptr) scratch[(r0 + q) * kScratchStride + kInjectOff + tid] = 0.0f;
      }
      if (needg) {
        float l = 0.0f;
#pragma unroll
        for (int t = 0; t < kSplit; ++t) l += lg[q][t];
        l = R(l);
        if (final_out) scratch[(r0 + q) * kScratchStride + kDown + tid] = l;
        const float t = R(l / gate_div);
        const float u = R(SigmoidGate(t));
        gate_out[(r0 + q) * kGate + tid] = __float2bfloat16(u * gate_mul);
      }
    }
  } else {
  if (tid < NR * kUpSlice) {
#pragma unroll
    for (int b = 0; b < kBranches; ++b) {
      xv[b] = LdAct1(streams + (r0 + orow) * kStreamDim + b * kHidden + h0 + ocol);
      inv[b] = __ldcg(scratch + (r0 + orow) * kScratchStride + kInvOff + b * 8);
    }
  }
#pragma unroll
  for (int q = 0; q < NR; ++q) {
    // fixed-order reduction of the K-split partials (same order as the former cluster / DSMEM reduction)
    const float* row = scratch + (r0 + q) * kScratchStride;
    const float alpha = row[kAlphaOff];
    float p[kSplit], p2[kSplit];
#pragma unroll
    for (int t = 0; t < kSplit; ++t) p[t] = __ldcg(row + kPartOff + t * kPartStride + tid);
    if (tid < kDown - kThreads) {
#pragma unroll
      for (int t = 0; t < kSplit; ++t) p2[t] = __ldcg(row + kPartOff + t * kPartStride + kThreads + tid);
    }
    float v = 0.0f;
#pragma unroll
    for (int t = 0; t < kSplit; ++t) v += p[t];
    {
      const float z = R(R(v) * alpha);
      sf[q][tid] = R(z * R(SigmoidSilu(z)));
      if (final_out && hb == 0) scratch[(r0 + q) * kScratchStride + tid] = sf[q][tid];
    }
    if (tid < kDown - kThreads) {
      float v2 = 0.0f;
#pragma unroll
      for (int t = 0; t < kSplit; ++t) v2 += p2[t];
      const float z = R(R(v2) * alpha);
      sf[q][tid + kThreads] = R(z * R(SigmoidSilu(z)));
      if (final_out && hb == 0) scratch[(r0 + q) * kScratchStride + tid + kThreads] = sf[q][tid + kThreads];
    }
    if (final_out && hb == 0 && tid < kBranches) {
      scratch[(r0 + q) * kScratchStride + kInvOffF + tid] = __ldcg(row + kInvOff + tid * 8);
      if (gate_out == nullptr) scratch[(r0 + q) * kScratchStride + kInjectOff + tid] = 0.0f;
    }
    if (hb == 0 && tid < kGate && gate_out != nullptr) {
      float lg = 0.0f;
#pragma unroll
      for (int t = 0; t < kSplit; ++t) lg += __ldcg(row + kPartOff + t * kPartStride + kDown + tid);
      lg = R(lg);
      if (final_out) scratch[(r0 + q) * kScratchStride + kDown + tid] = lg;
      const float t = R(lg / gate_div);
      const float u = R(SigmoidGate(t));
      gate_out[(r0 + q) * kGate + tid] = __float2bfloat16(u * gate_mul);
    }
  }
  }
  __syncthreads();
  TRACE(1, 2, hb);
  CLK(1, 2, hb);

  if constexpr (kTiled) MbarWait(&sm.mb_w);
  CpWait<0>();
  CLK(1, 3, hb);
  float acc[NR][8];
#pragma unroll
  for (int q = 0; q < NR; ++q)
#pragma unroll
    for (int m = 0; m < 8; ++m) acc[q][m] = 0.0f;
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    float wf[8];
    if constexpr (kFp8) UnpackFp8(wsm[i][tid], wf);
    else Unpack(wsm[i][tid], wf);
#pragma unroll
    for (int q = 0; q < NR; ++q) {
      const float sv = sf[q][rp + 32 * i];
#pragma unroll
      for (int m = 0; m < 8; ++m) acc[q][m] = fmaf(sv, wf[m], acc[q][m]);
    }
  }
  CLK(1, 4, hb);
  // lanes of a warp: g = lane & 7, rp differs in lane bits 3..4
#pragma unroll
  for (int q = 0; q < NR; ++q) {
    // two butterfly levels (lane bits 3, 4) as a reduce-scatter: 8 -> 4 -> 2 values per lane
    const int lane = tid & 31;
    {
      const bool hi = (lane & 8) != 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float send = hi ? acc[q][k] : acc[q][k + 4];
        const float recv = __shfl_xor_sync(0xffffffffu, send, 8);
        acc[q][k] = (hi ? acc[q][k + 4] : acc[q][k]) + recv;
      }
    }
    {
      const bool hi = (lane & 16) != 0;
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const float send = hi ? acc[q][k] : acc[q][k + 2];
        const float recv = __shfl_xor_sync(0xffffffffu, send, 16);
        acc[q][k] = (hi ? acc[q][k + 2] : acc[q][k]) + recv;
      }
    }
    const int m0 = 2 * ((lane >> 4) & 1) + 4 * ((lane >> 3) & 1);
    red[q][tid >> 5][g * 8 + m0] = acc[q][0];
    red[q][tid >> 5][g * 8 + m0 + 1] = acc[q][1];
  }
  CLK(1, 5, hb);
  __syncthreads();
  CLK(1, 6, hb);
  if (tid < NR * 64) {
    const int q = tid >> 6, col = tid & 63;
    float v = 0.0f;
#pragma unroll
    for (int t = 0; t < 8; ++t) v += red[q][t][col];
    if constexpr (kFp8) {
      if (wu_scale != nullptr) v *= wu_scale[(col >> 4) * kHidden + hb * kUpSlice + (col & 15)];
    }
    sg[q][col] = SigmoidBf16(R(v));
  }
  __syncthreads();
  TRACE(1, 3, hb);
  CLK(1, 7, hb);
  if (tid < NR * kUpSlice) {
    float sum = 0.0f;
#pragma unroll
    for (int b = 0; b < kBranches; ++b) {
      const float n = R(xv[b] * inv[b] * U2F(nws[b][ocol]));
      sum = fmaf(n, sg[orow][b * kUpSlice + ocol], sum);
    }
    pre[(r0 + orow) * kHidden + h0 + ocol] = __float2bfloat16(sum * reduction_scale);
  }
  TRACE(1, 4, hb);
  CLK(1, 8, hb);
}

template <int NR, bool kTiled>
__global__ void __launch_bounds__(kThreads)
    HcUpPreKernel(const bf16* __restrict__ streams, const bf16* __restrict__ norm_w, const bf16* __restrict__ w_up,
                  float* __restrict__ scratch, bf16* __restrict__ pre, bf16* __restrict__ gate_out,
                  float reduction_scale, float gate_div, float gate_mul, int final_out) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  UpSm<NR>& sm = *reinterpret_cast<UpSm<NR>*>(smem_raw);
  const int hb = blockIdx.x;
  TRACE(1, 0, hb);
  CLK(1, 0, hb);
  UpPrefetch<NR, kTiled>(sm, hb, norm_w, w_up);
  GridTrigger();
  GridWait();
  TRACE(1, 1, hb);
  CLK(1, 1, hb);
  UpBody<NR, kTiled>(sm, hb, streams, scratch, pre, gate_out, reduction_scale, gate_div, gate_mul, final_out);
}

// ---------------------------------------------------------------- merged single-kernel block (tiled weights, one row group)
//
// Both phases in one launch (grid (21, 8, 1)): the weights of BOTH phases are bulk-copied into shared memory at kernel start (before the
// dependency on the previous op resolves when HC_EARLYW is set), the down phase writes the K-split partials, a software grid barrier
// (monotonic arrival counter, 168 arrivals per launch) replaces the second kernel boundary, and CTAs 0..159 then run the up phase.
// All 168 CTAs must be co-resident (checked on the host); never launched with gridDim.z > 1.
template <int NR, bool kFp8 = false>
__host__ __device__ constexpr size_t BlockSmemUpOff() { return (sizeof(DnSm<NR, kFp8>) + 127) / 128 * 128; }
template <int NR, bool kFp8 = false>
__host__ __device__ constexpr size_t BlockSmemBytes() { return BlockSmemUpOff<NR, kFp8>() + sizeof(UpSm<NR, kFp8>); }

template <bool kPost, int NR, bool kFp8 = false>
__global__ void __launch_bounds__(kThreads, 2)
    HcBlockKernel(const bf16* __restrict__ streams, const bf16* __restrict__ branch, const bf16* __restrict__ gate,
                  const bf16* __restrict__ norm_w, const void* __restrict__ w_down, const bf16* __restrict__ w_inject,
                  const void* __restrict__ w_up, bf16* __restrict__ streams_out, float* __restrict__ scratch,
                  bf16* __restrict__ pre, bf16* __restrict__ gate_out, float epsilon, float silu_alpha, float reduction_scale,
                  float gate_div, float gate_mul, unsigned long long* __restrict__ tab, unsigned long long* __restrict__ ctr,
                  int trigger_at, int final_out, const float* __restrict__ wd_scale = nullptr,
                  const float* __restrict__ wu_scale = nullptr) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  DnSm<NR, kFp8>& sd = *reinterpret_cast<DnSm<NR, kFp8>*>(smem_raw);
  UpSm<NR, kFp8>& su = *reinterpret_cast<UpSm<NR, kFp8>*>(smem_raw + BlockSmemUpOff<NR, kFp8>());
  const int tid = threadIdx.x;
  const int b = blockIdx.y * kDownGridX + blockIdx.x;  // CTAs b < kUpBlocks also own up tile b
  DnPhase<kPost, NR, true, true, kFp8>(sd, streams, branch, gate, norm_w, w_down, w_inject, streams_out, scratch, epsilon, silu_alpha,
                                       trigger_at, &su, b, w_up, tab, ctr, wd_scale);
  TRACE(1, 0, b);
  TRACE(1, 1, b);
  CLK(1, 0, b);
  CLK(1, 1, b);
  if (b < kUpBlocks) {
    UpBody<NR, true, true, kPost, kFp8>(su, b, streams, scratch, pre, gate_out, reduction_scale, gate_div, gate_mul, final_out, tab, sd.tag,
                                        silu_alpha, branch, gate, wu_scale);
  }
  __syncthreads();
  if (tid == 0) asm volatile("red.relaxed.gpu.global.add.u64 [%0], 1;" ::"l"(ctr) : "memory");  // launch counter, see StoreTagged
}

// ---------------------------------------------------------------- host side

void CheckLaunch() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

cudaStream_t Stream(const OrtApi& api, OrtKernelContext* context) {
  void* raw = nullptr;
  Ort::ThrowOnError(api.KernelContext_GetGPUComputeStream(context, &raw));
  return static_cast<cudaStream_t>(raw);
}

// Programmatic dependent launch lets a kernel's weight prefetch overlap the previous kernel's tail.
// HC_PDL=0 disables it.
bool g_pdl = [] {
  const char* v = std::getenv("HC_PDL");
  return v == nullptr || std::string(v) != "0";
}();
bool PdlEnabled() { return g_pdl; }
// HC_SCRATCH_FINAL=1: the up kernel also writes the reduced hidden / inject logits / inv_rms into the scratch row (unit-test aid)
int FinalOut() {
  static const int v = [] {
    const char* e = std::getenv("HC_SCRATCH_FINAL");
    return e == nullptr ? 0 : std::atoi(e);
  }();
  return v;
}
int TriggerAt() {
  static const int v = [] {
    const char* e = std::getenv("HC_TRIGGER");
    return e == nullptr ? 1 : std::atoi(e);
  }();
  return v;
}

// raises the dynamic shared memory limit of a kernel once
template <typename Kernel>
void EnsureSmem(Kernel kernel, size_t bytes) {
  static std::mutex mutex;
  static std::vector<const void*> done;
  std::lock_guard<std::mutex> lock(mutex);
  const void* key = reinterpret_cast<const void*>(kernel);
  if (std::find(done.begin(), done.end(), key) != done.end()) return;
  cudaError_t error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  done.push_back(key);
}

template <typename Kernel, typename... Args>
void Launch(Kernel kernel, dim3 grid, size_t smem, cudaStream_t stream, Args... args) {
  EnsureSmem(kernel, smem);
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
  config.numAttrs = PdlEnabled() ? 1 : 0;
  cudaError_t error = cudaLaunchKernelEx(&config, kernel, args...);
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <typename Kernel, typename... Args>
void LaunchSmem(Kernel kernel, dim3 grid, size_t smem, cudaStream_t stream, Args... args) {
  EnsureSmem(kernel, smem);
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
  config.numAttrs = PdlEnabled() ? 1 : 0;
  cudaError_t error = cudaLaunchKernelEx(&config, kernel, args...);
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// Rows that share one CTA (one weight stream): 2 for an even row count, else 1. HC_ROWS_PER_CTA=1 disables sharing.
int RowsPerCta(int64_t rows) {
  static const int cap = [] {
    const char* e = std::getenv("HC_ROWS_PER_CTA");
    return e == nullptr ? 2 : std::atoi(e);
  }();
  return (cap >= 2 && rows % 2 == 0) ? 2 : 1;
}

template <bool kPost, bool kInject, typename... Args>
void LaunchDown(int64_t rows, bool tiled, cudaStream_t stream, Args... args) {
  const unsigned gx = kInject ? kDownGridX : kColSlices;  // the mixer (no inject branch) has no inject CTA column
  if (RowsPerCta(rows) == 2) {
    if (tiled) Launch(HcDownKernel<kPost, 2, true>, dim3(gx, kSplit, static_cast<unsigned>(rows / 2)), sizeof(DnSm<2>), stream, args...);
    else Launch(HcDownKernel<kPost, 2, false>, dim3(gx, kSplit, static_cast<unsigned>(rows / 2)), sizeof(DnSm<2>), stream, args...);
  } else {
    if (tiled) Launch(HcDownKernel<kPost, 1, true>, dim3(gx, kSplit, static_cast<unsigned>(rows)), sizeof(DnSm<1>), stream, args...);
    else Launch(HcDownKernel<kPost, 1, false>, dim3(gx, kSplit, static_cast<unsigned>(rows)), sizeof(DnSm<1>), stream, args...);
  }
}

template <typename... Args>
void LaunchUp(int64_t rows, bool tiled, cudaStream_t stream, Args... args) {
  if (RowsPerCta(rows) == 2) {
    if (tiled) Launch(HcUpPreKernel<2, true>, dim3(kUpBlocks, 1, static_cast<unsigned>(rows / 2)), sizeof(UpSm<2>), stream, args...);
    else Launch(HcUpPreKernel<2, false>, dim3(kUpBlocks, 1, static_cast<unsigned>(rows / 2)), sizeof(UpSm<2>), stream, args...);
  } else {
    if (tiled) Launch(HcUpPreKernel<1, true>, dim3(kUpBlocks, 1, static_cast<unsigned>(rows)), sizeof(UpSm<1>), stream, args...);
    else Launch(HcUpPreKernel<1, false>, dim3(kUpBlocks, 1, static_cast<unsigned>(rows)), sizeof(UpSm<1>), stream, args...);
  }
}

const bf16* In(const Ort::KernelContext& c, size_t i) {
  return reinterpret_cast<const bf16*>(c.GetInput(i).GetTensorData<Ort::BFloat16_t>());
}
bf16* Out(Ort::UnownedValue& v) { return reinterpret_cast<bf16*>(v.GetTensorMutableData<Ort::BFloat16_t>()); }

int64_t Rows(const Ort::ConstValue& streams) {
  auto shape = streams.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.empty() || shape.back() != kStreamDim) throw std::invalid_argument("Hc: streams last dim must be 10240");
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) rows *= shape[i];
  if (rows > 65535) throw std::invalid_argument("Hc: too many rows");
  return rows;
}

void CheckDims(const Ort::ConstValue& v, std::initializer_list<int64_t> expected, const char* what) {
  auto shape = v.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() != expected.size() || !std::equal(shape.begin(), shape.end(), expected.begin())) {
    throw std::invalid_argument(std::string("Hc: bad shape for ") + what);
  }
}

void CheckNumel(const Ort::ConstValue& v, int64_t expected, const char* what) {
  if (v.GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<size_t>(expected)) {
    throw std::invalid_argument(std::string("Hc: bad element count for ") + what);
  }
}

// Weight layout: the original [K, N] matrix, or the pre-tiled image [160 tiles][20480] (one tile per CTA, see optimize_v12_hc.py)
constexpr int64_t kTiles = kSplit * kColSlices;  // 160 == kUpBlocks
constexpr int64_t kTileElems = 10 * kThreads * 8;
static_assert(kTiles == kUpBlocks, "up and down tilings share the shape");
bool IsTiled(const Ort::ConstValue& v, std::initializer_list<int64_t> legacy, const char* what) {
  auto shape = v.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() == 2 && shape[0] == kTiles && shape[1] == kTileElems) return true;
  CheckDims(v, legacy, what);
  return false;
}

std::vector<int64_t> LeadingShape(const Ort::ConstValue& streams, int64_t last) {
  auto shape = streams.GetTensorTypeAndShapeInfo().GetShape();
  shape.back() = last;
  return shape;
}

template <bool kPost, bool kInject = true>
struct HcDownState {
  const OrtApi& api;
  float epsilon, silu_alpha;
  HcDownState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    epsilon = k.GetAttribute<float>("epsilon");
    silu_alpha = k.GetAttribute<float>("silu_alpha");
  }
  // HcDown:        streams, norm_w, w_down, w_inject               -> scratch
  // HcPostDown:    streams, branch, gate, norm_w, w_down, w_inject -> streams_out, scratch
  // HcMixDown / HcMixPostDown (kInject == false): the same without w_inject (final hyper_connection_mixer)
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      constexpr size_t o = kPost ? 3 : 1;
      auto streams = c.GetInput(0);
      const int64_t rows = Rows(streams);
      CheckDims(c.GetInput(o), {kStreamDim}, "norm_w");
      const bool tiled = IsTiled(c.GetInput(o + 1), {kStreamDim, kDown}, "w_down");
      if (kInject) CheckDims(c.GetInput(o + 2), {kStreamDim, kGate}, "w_inject");
      if (kPost) {
        CheckNumel(c.GetInput(1), rows * kHidden, "branch");
        CheckNumel(c.GetInput(2), rows * kGate, "gate");
      }
      bf16* streams_out = nullptr;
      size_t scratch_index = 0;
      if (kPost) {
        auto sout = c.GetOutput(0, streams.GetTensorTypeAndShapeInfo().GetShape());
        streams_out = Out(sout);
        scratch_index = 1;
      }
      auto scratch = c.GetOutput(scratch_index, std::vector<int64_t>{rows, kScratchStride});
      LaunchDown<kPost, kInject>(rows, tiled, Stream(api, raw),
             In(c, 0), kPost ? In(c, 1) : nullptr, kPost ? In(c, 2) : nullptr, In(c, o), In(c, o + 1),
             kInject ? In(c, o + 2) : nullptr, streams_out, scratch.GetTensorMutableData<float>(), epsilon, silu_alpha,
             TriggerAt());
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// kGate == false (HcMixUp): streams, norm_w, w_up, scratch -> pre
template <bool kGateOut = true>
struct HcUpPreState {
  const OrtApi& api;
  float reduction_scale, gate_div = 4.0f, gate_mul = 2.0f;
  HcUpPreState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    reduction_scale = k.GetAttribute<float>("reduction_scale");
    if (kGateOut) {
      gate_div = k.GetAttribute<float>("gate_div");
      gate_mul = k.GetAttribute<float>("gate_mul");
    }
  }
  // streams, norm_w, w_up, scratch -> pre, gate
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto streams = c.GetInput(0);
      const int64_t rows = Rows(streams);
      CheckDims(c.GetInput(1), {kStreamDim}, "norm_w");
      const bool tiled = IsTiled(c.GetInput(2), {kDown, kStreamDim}, "w_up");
      CheckDims(c.GetInput(3), {rows, kScratchStride}, "scratch");
      auto pre = c.GetOutput(0, LeadingShape(streams, kHidden));
      bf16* gate_ptr = nullptr;
      if (kGateOut) {
        auto gate = c.GetOutput(1, LeadingShape(streams, kGate));
        gate_ptr = Out(gate);
      }
      LaunchUp(rows, tiled, Stream(api, raw), In(c, 0), In(c, 1),
             In(c, 2), const_cast<float*>(c.GetInput(3).GetTensorData<float>()), Out(pre), gate_ptr, reduction_scale,
             gate_div, gate_mul, FinalOut());
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

bool MergeEnabled() {
  static const bool v = [] {
    const char* e = std::getenv("HC_MERGE");
    return e == nullptr || std::string(e) != "0";
  }();
  return v;
}
// bit 4 of the trigger argument: issue the weight bulk copies before the grid dependency resolves (HC_EARLYW=1; useful when the previous op
// launches its dependents early)
int EarlyW() {
  static const int v = [] {
    const char* e = std::getenv("HC_EARLYW");
    return e == nullptr ? 0 : std::atoi(e) * 16;
  }();
  return v;
}

// HcBlock:     streams, norm_w, w_down, w_inject, w_up                      -> pre, gate, scratch
// HcPostBlock: streams, branch, gate, norm_w, w_down, w_inject, w_up        -> streams_out, pre, gate, scratch
// tiled weights and rows in {1, 2}: the single merged kernel; everything else (and HC_MERGE=0): the two-kernel path
template <bool kPost>
struct HcBlockState {
  const OrtApi& api;
  float epsilon, silu_alpha, reduction_scale, gate_div, gate_mul;
  unsigned long long* tab = nullptr;  // data-carrying handoff table (2 rows) + launch counter, owned by this node
  unsigned long long* ctr = nullptr;
  float* d_wd_scale = nullptr;
  float* d_wu_scale = nullptr;
  int merged_ok = -1;  // -1 unknown, 0 no, 1 yes

  HcBlockState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    epsilon = k.GetAttribute<float>("epsilon");
    silu_alpha = k.GetAttribute<float>("silu_alpha");
    reduction_scale = k.GetAttribute<float>("reduction_scale");
    gate_div = k.GetAttribute<float>("gate_div");
    gate_mul = k.GetAttribute<float>("gate_mul");
    std::vector<float> wd_host, wu_host;
    try {
      wd_host = k.GetAttributes<float>("wd_scale");
    } catch (const std::exception&) {
    }
    try {
      wu_host = k.GetAttributes<float>("wu_scale");
    } catch (const std::exception&) {
    }
    if (!wd_host.empty()) {
      if (cudaMalloc(&d_wd_scale, wd_host.size() * sizeof(float)) != cudaSuccess ||
          cudaMemcpy(d_wd_scale, wd_host.data(), wd_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_wd_scale);
        d_wd_scale = nullptr;
        throw std::runtime_error("HcBlock: failed to upload wd_scale");
      }
    }
    if (!wu_host.empty()) {
      if (cudaMalloc(&d_wu_scale, wu_host.size() * sizeof(float)) != cudaSuccess ||
          cudaMemcpy(d_wu_scale, wu_host.data(), wu_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_wd_scale);
        cudaFree(d_wu_scale);
        d_wd_scale = nullptr;
        d_wu_scale = nullptr;
        throw std::runtime_error("HcBlock: failed to upload wu_scale");
      }
    }
  }
  ~HcBlockState() {
    if (tab != nullptr) cudaFree(tab);
    if (d_wd_scale != nullptr) cudaFree(d_wd_scale);
    if (d_wu_scale != nullptr) cudaFree(d_wu_scale);
  }

  template <int NR, bool kFp8 = false>
  bool CheckResident() {
    auto kernel = HcBlockKernel<kPost, NR, kFp8>;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(BlockSmemBytes<NR, kFp8>())) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    int per_sm = 0, sms = 0, dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, BlockSmemBytes<NR, kFp8>()) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return static_cast<long long>(per_sm) * sms >= kBlockCtas;
  }

  // Allocation and the co-residency check happen on the first (never capturing) call.
  bool EnsureMerged(cudaStream_t stream, bool fp8) {
    if (merged_ok >= 0) return merged_ok == 1;
    merged_ok = 0;
    if (!MergeEnabled()) return false;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess || status != cudaStreamCaptureStatusNone) {
      throw std::runtime_error("HcBlock: first execution must not happen during CUDA graph capture");
    }
    const bool ok = fp8 ? (CheckResident<1, true>() && CheckResident<2, true>())
                        : (CheckResident<1, false>() && CheckResident<2, false>());
    if (!ok) return false;
    const size_t bytes = (2 * static_cast<size_t>(kTabRow) + 8) * sizeof(unsigned long long);
    if (cudaMalloc(&tab, bytes) != cudaSuccess) return false;
    if (cudaMemset(tab, 0, bytes) != cudaSuccess) return false;
    ctr = tab + 2 * static_cast<size_t>(kTabRow);
    merged_ok = 1;
    return true;
  }

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      constexpr size_t o = kPost ? 3 : 1;
      auto streams = c.GetInput(0);
      const int64_t rows = Rows(streams);
      CheckDims(c.GetInput(o), {kStreamDim}, "norm_w");
      const bool tiled_d = IsTiled(c.GetInput(o + 1), {kStreamDim, kDown}, "w_down");
      CheckDims(c.GetInput(o + 2), {kStreamDim, kGate}, "w_inject");
      const bool tiled_u = IsTiled(c.GetInput(o + 3), {kDown, kStreamDim}, "w_up");
      const auto dtype_d = c.GetInput(o + 1).GetTensorTypeAndShapeInfo().GetElementType();
      const auto dtype_u = c.GetInput(o + 3).GetTensorTypeAndShapeInfo().GetElementType();
      const bool fp8 = (dtype_d == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN &&
                        dtype_u == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN);
      bf16* streams_out = nullptr;
      size_t next = 0;
      if (kPost) {
        auto sout = c.GetOutput(0, streams.GetTensorTypeAndShapeInfo().GetShape());
        streams_out = Out(sout);
        next = 1;
      }
      auto pre = c.GetOutput(next, LeadingShape(streams, kHidden));
      auto gate = c.GetOutput(next + 1, LeadingShape(streams, kGate));
      auto scratch = c.GetOutput(next + 2, std::vector<int64_t>{rows, kScratchStride});
      cudaStream_t stream = Stream(api, raw);
      float* scratch_ptr = scratch.GetTensorMutableData<float>();
      const bf16* br = kPost ? In(c, 1) : nullptr;
      const bf16* gt = kPost ? In(c, 2) : nullptr;
      const void* wd_ptr = c.GetInput(o + 1).GetTensorRawData();
      const void* wu_ptr = c.GetInput(o + 3).GetTensorRawData();
      if (tiled_d && tiled_u && (rows == 1 || rows == 2) && EnsureMerged(stream, fp8)) {
        if (fp8) {
          if (rows == 2) {
            LaunchSmem(HcBlockKernel<kPost, 2, true>, dim3(kDownGridX, kSplit, 1), BlockSmemBytes<2, true>(), stream, In(c, 0), br, gt,
                       In(c, o), wd_ptr, In(c, o + 2), wu_ptr, streams_out, scratch_ptr, Out(pre), Out(gate), epsilon, silu_alpha,
                       reduction_scale, gate_div, gate_mul, tab, ctr, TriggerAt() + EarlyW(), FinalOut(), d_wd_scale, d_wu_scale);
          } else {
            LaunchSmem(HcBlockKernel<kPost, 1, true>, dim3(kDownGridX, kSplit, 1), BlockSmemBytes<1, true>(), stream, In(c, 0), br, gt,
                       In(c, o), wd_ptr, In(c, o + 2), wu_ptr, streams_out, scratch_ptr, Out(pre), Out(gate), epsilon, silu_alpha,
                       reduction_scale, gate_div, gate_mul, tab, ctr, TriggerAt() + EarlyW(), FinalOut(), d_wd_scale, d_wu_scale);
          }
        } else {
          if (rows == 2) {
            LaunchSmem(HcBlockKernel<kPost, 2, false>, dim3(kDownGridX, kSplit, 1), BlockSmemBytes<2, false>(), stream, In(c, 0), br, gt,
                       In(c, o), wd_ptr, In(c, o + 2), wu_ptr, streams_out, scratch_ptr, Out(pre), Out(gate), epsilon, silu_alpha,
                       reduction_scale, gate_div, gate_mul, tab, ctr, TriggerAt() + EarlyW(), FinalOut(), nullptr, nullptr);
          } else {
            LaunchSmem(HcBlockKernel<kPost, 1, false>, dim3(kDownGridX, kSplit, 1), BlockSmemBytes<1, false>(), stream, In(c, 0), br, gt,
                       In(c, o), wd_ptr, In(c, o + 2), wu_ptr, streams_out, scratch_ptr, Out(pre), Out(gate), epsilon, silu_alpha,
                       reduction_scale, gate_div, gate_mul, tab, ctr, TriggerAt() + EarlyW(), FinalOut(), nullptr, nullptr);
          }
        }
      } else {
        if (fp8) throw std::invalid_argument("HcBlock: FP8 weights require tiled merged execution (rows in {1, 2})");
        LaunchDown<kPost, true>(rows, tiled_d, stream, In(c, 0), br, gt, In(c, o), In(c, o + 1), In(c, o + 2), streams_out,
                                scratch_ptr, epsilon, silu_alpha, 1);
        LaunchUp(rows, tiled_u, stream, kPost ? streams_out : In(c, 0), In(c, o), In(c, o + 3), scratch_ptr, Out(pre), Out(gate),
                 reduction_scale, gate_div, gate_mul, FinalOut());
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
constexpr auto kUNDEF = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

template <typename Derived, typename State>
struct HcOp : Ort::CustomOpBase<Derived, State, true> {
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

struct HcDownOp : HcOp<HcDownOp, HcDownState<false>> {
  static constexpr std::array<ONNXTensorElementDataType, 4> kIn = {kBF, kBF, kBF, kBF};
  static constexpr std::array<ONNXTensorElementDataType, 1> kOut = {kF32};
  const char* GetName() const { return "HcDown"; }
};
struct HcPostDownOp : HcOp<HcPostDownOp, HcDownState<true>> {
  static constexpr std::array<ONNXTensorElementDataType, 6> kIn = {kBF, kBF, kBF, kBF, kBF, kBF};
  static constexpr std::array<ONNXTensorElementDataType, 2> kOut = {kBF, kF32};
  const char* GetName() const { return "HcPostDown"; }
};
struct HcBlockOp : HcOp<HcBlockOp, HcBlockState<false>> {
  static constexpr std::array<ONNXTensorElementDataType, 5> kIn = {kBF, kBF, kUNDEF, kBF, kUNDEF};
  static constexpr std::array<ONNXTensorElementDataType, 3> kOut = {kBF, kBF, kF32};
  const char* GetName() const { return "HcBlock"; }
};
struct HcPostBlockOp : HcOp<HcPostBlockOp, HcBlockState<true>> {
  static constexpr std::array<ONNXTensorElementDataType, 7> kIn = {kBF, kBF, kBF, kBF, kUNDEF, kBF, kUNDEF};
  static constexpr std::array<ONNXTensorElementDataType, 4> kOut = {kBF, kBF, kBF, kF32};
  const char* GetName() const { return "HcPostBlock"; }
};
struct HcUpPreOp : HcOp<HcUpPreOp, HcUpPreState<true>> {
  static constexpr std::array<ONNXTensorElementDataType, 4> kIn = {kBF, kBF, kBF, kF32};
  static constexpr std::array<ONNXTensorElementDataType, 2> kOut = {kBF, kBF};
  const char* GetName() const { return "HcUpPre"; }
};
// final hyper_connection_mixer: same kernels without the inject / gate branch
struct HcMixDownOp : HcOp<HcMixDownOp, HcDownState<false, false>> {
  static constexpr std::array<ONNXTensorElementDataType, 3> kIn = {kBF, kBF, kBF};
  static constexpr std::array<ONNXTensorElementDataType, 1> kOut = {kF32};
  const char* GetName() const { return "HcMixDown"; }
};
struct HcMixPostDownOp : HcOp<HcMixPostDownOp, HcDownState<true, false>> {
  static constexpr std::array<ONNXTensorElementDataType, 5> kIn = {kBF, kBF, kBF, kBF, kBF};
  static constexpr std::array<ONNXTensorElementDataType, 2> kOut = {kBF, kF32};
  const char* GetName() const { return "HcMixPostDown"; }
};
struct HcMixUpOp : HcOp<HcMixUpOp, HcUpPreState<false>> {
  static constexpr std::array<ONNXTensorElementDataType, 4> kIn = {kBF, kBF, kBF, kF32};
  static constexpr std::array<ONNXTensorElementDataType, 1> kOut = {kBF};
  const char* GetName() const { return "HcMixUp"; }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ORT API");
  Ort::InitApi(api);
  try {
    static HcDownOp down;
    static HcPostDownOp post_down;
    static HcUpPreOp up_pre;
    static HcBlockOp block;
    static HcPostBlockOp post_block;
    static HcMixDownOp mix_down;
    static HcMixPostDownOp mix_post_down;
    static HcMixUpOp mix_up;
    // ORT rejects two libraries registering the same domain name in one session, and the
    // v9f ops already own "qwen.fused", so these ops live in a sibling domain.
    static Ort::CustomOpDomain domain("qwen.fused_hc");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&down);
      domain.Add(&post_down);
      domain.Add(&up_pre);
      domain.Add(&block);
      domain.Add(&post_block);
      domain.Add(&mix_down);
      domain.Add(&mix_post_down);
      domain.Add(&mix_up);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_FAIL, e.what());
  }
}
