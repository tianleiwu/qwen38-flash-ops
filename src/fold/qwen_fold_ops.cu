// Glue ops for the folded MTP(1) cycle graph (verify S=2 + head R=2 in ONE ORT graph), domain qwen.fold.
//
//   FoldMid(next_tokens i64[1,2], input_ids i64[1,2], status i32[N], ctl i64[8], ctx_commit i64[2], ctx_a i64[2], ctx_b i64[2],
//           ptrs i64[2])
//       -> head_ids i64[1,2], head_pos i64[1,2], head_c i32[1], sel i64[1]
//     accept test + commit bookkeeping of the verify cycle (the former k_post_tok kernel) and the head row inputs, all on device:
//       accept = next_tokens[0] == input_ids[1]
//       head rows [(accept ? draft : pred0, h0) @ L+1,  (pred1, h1) @ L+2], slots [c, c+1]; on reject row 1 is a never-committed filler.
//     ctl (i64[8]): [0] pending, [1] L, [2] draft, [4] sequence counter, [5] error bits, [6] accept flag, [3] mute flag (target prefill: no bookkeeping), [7] head cache length c.
//     ptrs: [0] = address of the pinned result ring (8 x i64 per record, record 0 word = sequence number, written last), [1] = ring size.
//   FoldMid2(... same inputs, + cos_cache, sin_cache bf16[N,32]) -> head_ids, head_c i32[1], slot_ids i64[1,2], seqlens i32[1], cos, sin
//            bf16[1,2,32] (rope rows of positions L+1, L+2), sel   (FoldMid with the position-derived head inputs folded in)
//   EmbedNorm(ids i64[1,R], table bf16[V,N], w f32[N]) -> bf16[1,R,N]  (attr epsilon) = the head's Gather + RMSNorm chain in one kernel
//   RowSelectBf16x3(sel i64[1], x0, x1, x2 bf16[1,2,...]) -> y0, y1, y2 bf16[1,1,...] = x_i[:, sel]
//   FoldTail(next_token i64[1,1,1], status i32[2], ctl i64[8]) -> done i32[1]
//     the draft of the next cycle goes to ctl[2]; a nonzero head status sets error bit 2.
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

typedef long long i64;
constexpr i64 kHeadCapacity = 33028;

void CheckLaunch(const char* = "") {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

cudaStream_t Stream(const OrtApi& api, OrtKernelContext* context) {
  void* raw = nullptr;
  Ort::ThrowOnError(api.KernelContext_GetGPUComputeStream(context, &raw));
  return static_cast<cudaStream_t>(raw);
}

__global__ void FoldMidKernel(const i64* tok01, const i64* input_ids, const int* status, int nst, i64* ctl, i64* ctx_commit,
                              const i64* ctx_a, const i64* ctx_b, const i64* ptrs, i64* head_ids, i64* head_pos,
                              int* head_c, i64* sel) {
  if (threadIdx.x != 0) return;
  const i64 pred0 = tok01[0];
  const i64 pred1 = tok01[1];
  const i64 draft = input_ids[1];
  const bool accept = pred0 == draft;
  i64 st = 0;
  for (int k = 0; k < nst; ++k) st |= static_cast<i64>(status[k]);
  const i64 L = ctl[1];
  const i64 c = ctl[7];
  head_ids[0] = accept ? draft : pred0;
  head_ids[1] = pred1;
  head_pos[0] = L + 1;
  head_pos[1] = L + 2;
  head_c[0] = static_cast<int>(c);
  sel[0] = accept ? 1 : 0;
  if (ctl[3] != 0) return;  // muted (target prefill): head row inputs only, no bookkeeping
  const i64 step = accept ? 2 : 1;
  const i64* src = accept ? ctx_b : ctx_a;
  ctx_commit[0] = src[0];
  ctx_commit[1] = src[1];
  i64* result = reinterpret_cast<i64*>(ptrs[0]);
  const i64 ring = ptrs[1];
  volatile i64* r = result + 8 * (ctl[4] % ring);
  ctl[0] = accept ? pred1 : pred0;
  ctl[1] = L + step;
  ctl[7] = c + step;
  ctl[4] += 1;
  ctl[5] |= (st ? 2 : 0) | ((c + 2 > kHeadCapacity) ? 4 : 0);
  ctl[6] = accept ? 1 : 0;
  r[1] = pred0;
  r[2] = pred1;
  r[3] = accept ? 1 : 0;
  r[4] = st;
  r[5] = accept ? pred1 : pred0;
  r[6] = L + step;
  __threadfence_system();
  r[0] = ctl[4];
}

// FoldMid + the position-derived head inputs (rope cos/sin gather, slot ids, DSA seqlens): saves the Gather/Cast/Add kernels of the head.
__global__ void FoldMid2Kernel(const i64* tok01, const i64* input_ids, const int* status, int nst, i64* ctl, i64* ctx_commit,
                               const i64* ctx_a, const i64* ctx_b, const i64* ptrs, const unsigned short* cos_cache,
                               const unsigned short* sin_cache, i64* head_ids, int* head_c, i64* slot_ids, int* seqlens,
                               unsigned short* cos_out, unsigned short* sin_out, i64* sel, int rows, int head_rows) {
  const bool muted = (ctl[3] != 0);
  const i64 L = muted ? 0 : ctl[1];
  const i64 c = muted ? 0 : ctl[7];
  if (threadIdx.x >= 32) {  // rope rows for positions L+1 .. L+head_rows (32 bf16 each)
    const int j = threadIdx.x - 32;
    if (j < 32 * head_rows) {
      const i64 pos = L + 1 + (j >> 5);
      cos_out[j] = cos_cache[pos * 32 + (j & 31)];
      sin_out[j] = sin_cache[pos * 32 + (j & 31)];
    }
    return;
  }
  if (threadIdx.x != 0) return;
  int a = 0;
  const int check_rows = rows <= 4 ? rows : head_rows;
  for (int j = 0; j + 1 < check_rows; ++j) {
    if (tok01[j] == input_ids[j + 1]) {
      ++a;
    } else {
      break;
    }
  }
  i64 st = 0;
  for (int k = 0; k < nst; ++k) st |= static_cast<i64>(status[k]);
  for (int j = 0; j < head_rows; ++j) {
    const int src_j = j < rows ? j : (rows - 1);
    head_ids[j] = (j < a && j + 1 < rows) ? input_ids[j + 1] : tok01[src_j];
    slot_ids[j] = c + j;
  }
  head_c[0] = static_cast<int>(c);
  seqlens[0] = static_cast<int>(c) + head_rows - 1;
  sel[0] = a < head_rows ? a : 0;
  if (ctl[3] != 0) return;
  const i64 step = a + 1;
  const i64* src = (a == 0) ? ctx_a : (a == 1 ? ctx_b : (ctx_a + 2 * a));
  ctx_commit[0] = src[0];
  ctx_commit[1] = src[1];
  i64* result = reinterpret_cast<i64*>(ptrs[0]);
  const i64 ring = ptrs[1];
  volatile i64* r = result + 8 * (ctl[4] % ring);
  const i64 new_pending = tok01[a];
  ctl[0] = new_pending;
  ctl[1] = L + step;
  ctl[7] = c + step;
  ctl[4] += 1;
  ctl[5] |= (st ? 2 : 0) | ((c + head_rows > kHeadCapacity) ? 4 : 0);
  ctl[6] = a;
  r[1] = tok01[0];
  r[2] = rows >= 2 ? tok01[1] : tok01[0];
  r[3] = a;
  r[4] = st;
  r[5] = new_pending;
  r[6] = L + step;
  if (rows >= 3) {
    const uint32_t p2 = static_cast<uint32_t>(tok01[2]);
    const uint32_t p3 = (rows >= 4) ? static_cast<uint32_t>(tok01[3]) : 0u;
    r[7] = (static_cast<i64>(p3) << 32) | static_cast<i64>(p2);
  }
  __threadfence_system();
  r[0] = ctl[4];
}

__global__ void CopyHeadHiddenKernel(const unsigned short* src, unsigned short* dst, int src_rows, int dst_rows, int width) {
  const int64_t total = static_cast<int64_t>(dst_rows) * width;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int r = static_cast<int>(idx / width);
    const int c = static_cast<int>(idx % width);
    const int sr = r < src_rows ? r : (src_rows - 1);
    dst[idx] = src[static_cast<int64_t>(sr) * width + c];
  }
}

// EmbedNorm: Gather(embed) + RMSNorm in one kernel, bit-identical to the ORT op chain at R=1 (Gather, Cast, Mul, ReduceMean, Add, Sqrt,
// Reciprocal, Mul, Mul, Cast): the fp32 sum of squares follows ORT's reduce_matrix_rows_kernel for [2560] (block of 512 threads: thread t sums
// x[t], x[t+512], ..., x[t+2048] in this order, then the halving tree s[t] += s[t+256], ... s[t] += s[t+1]); mean = sum / N; + eps; sqrt; 1/s;
// x * r; * w; bf16.  One block per row.  (The ORT chain at R=2 uses a different reduction order, reduce_matrix_columns_kernel: the R=1 order is
// kept for every row so that the head's rows equal the single-row chains of the reference head.)
__global__ void EmbedNormKernel(const i64* ids, const unsigned short* table, const float* w, float eps, unsigned short* out, int n,
                                const int* committed, i64* slot_ids, int* seqlens_k, int* cu_seqlens, int rows) {
  __shared__ float s[512];
  if (threadIdx.x == 0 && committed != nullptr) {
    const int c = committed[0];
    if (slot_ids != nullptr) slot_ids[blockIdx.x] = static_cast<i64>(c) + static_cast<i64>(blockIdx.x);
    if (blockIdx.x == 0) {
      if (seqlens_k != nullptr) seqlens_k[0] = c + rows - 1;
      if (cu_seqlens != nullptr) {
        cu_seqlens[0] = 0;
        cu_seqlens[1] = rows;
      }
    }
  }
  const i64 raw_tok = ids[blockIdx.x];
  const i64 tok = (raw_tok >= 0 && raw_tok < 248320) ? raw_tok : 0;
  const unsigned short* row = table + tok * n;
  float acc = 0.f;
  for (int k = threadIdx.x; k < n; k += 512) {
    const float x = __uint_as_float(static_cast<unsigned>(row[k]) << 16);
    acc += x * x;
  }
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int st = 256; st > 0; st >>= 1) {
    if (threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
    __syncthreads();
  }
  const float mean = s[0] / static_cast<float>(n);
  const float rs = 1.0f / sqrtf(mean + eps);
  for (int k = threadIdx.x; k < n; k += 512) {
    const float x = __uint_as_float(static_cast<unsigned>(row[k]) << 16);
    const float y = (x * rs) * w[k];
    out[blockIdx.x * n + k] = __bfloat16_as_ushort(__float2bfloat16_rn(y));
  }
}

__global__ void RowSelectKernel(const i64* sel, const unsigned short* x0, const unsigned short* x1, const unsigned short* x2,
                                unsigned short* y0, unsigned short* y1, unsigned short* y2, int n0, int n1, int n2, int rows) {
  const int which = blockIdx.y;
  const unsigned short* x = which == 0 ? x0 : which == 1 ? x1 : x2;
  unsigned short* y = which == 0 ? y0 : which == 1 ? y1 : y2;
  const int n = which == 0 ? n0 : which == 1 ? n1 : n2;  // bf16 elements per row
  const i64 s = sel[0];
  const i64 row = s < 0 ? 0 : (s >= rows ? rows - 1 : s);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = x[row * n + i];
}

struct DraftPrepBuffers {
  const unsigned short* buf[4];
  const int* len[4];
};

__global__ void FoldDraftPrepKernel(const i64* next_token, i64* ctl, const i64* sel, const unsigned short* cos_cache,
                                    const unsigned short* sin_cache, DraftPrepBuffers in_state, int stage,
                                    i64* head_ids, int* head_c, i64* slot_ids, int* seqlens,
                                    unsigned short* cos_out, unsigned short* sin_out,
                                    unsigned short* out_buf, int* out_len) {
  const i64 a = sel[0];
  const bool muted = (ctl[3] != 0);
  const i64 L_new = muted ? (ctl[1] + a + 1) : ctl[1];
  const i64 c_new = muted ? (ctl[7] + a + 1) : ctl[7];
  const i64 c_stage = c_new + (stage - 1);
  const i64 pos_stage = L_new + stage;

  if (threadIdx.x < 32) {
    cos_out[threadIdx.x] = cos_cache[pos_stage * 32 + threadIdx.x];
    sin_out[threadIdx.x] = sin_cache[pos_stage * 32 + threadIdx.x];
  }
  const int pick = (stage > 1 || a <= 0) ? 0
                   : (a == 1 && in_state.buf[1] ? 1
                   : (a == 2 && in_state.buf[2] ? 2
                   : (in_state.buf[3] ? 3 : (in_state.buf[2] ? 2 : (in_state.buf[1] ? 1 : 0)))));
  const unsigned short* src_buf = in_state.buf[pick];
  const int* src_len = in_state.len[pick];
  for (int i = threadIdx.x; i < 7 * 128; i += blockDim.x) {
    out_buf[i] = src_buf[i];
  }
  if (threadIdx.x < 2) {
    out_len[threadIdx.x] = src_len[threadIdx.x];
  }
  if (threadIdx.x == 0) {
    const i64 tok = next_token[0];
    head_ids[0] = tok;
    head_c[0] = static_cast<int>(c_stage);
    slot_ids[0] = c_stage;
    seqlens[0] = static_cast<int>(c_stage);
    if (!muted) {
      const int ctl_slot = (stage == 1) ? 2 : 8;
      ctl[ctl_slot] = tok;
      if (c_stage + 1 > kHeadCapacity) ctl[5] |= 4;
    }
  }
}

__global__ void FoldTailKernel(const i64* next_token, const int* status, int nst, i64* ctl, int* done, int ctl_slot) {
  if (threadIdx.x != 0) return;
  if (ctl[3] != 0) {
    done[0] = 0;
    return;
  }
  i64 st = 0;
  for (int k = 0; k < nst; ++k) st |= static_cast<i64>(status[k]);
  ctl[ctl_slot] = next_token[0];
  if (st) ctl[5] |= 2;
  done[0] = 1;
}

struct MidState {
  const OrtApi& api;
  MidState(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto tok = c.GetInput(0);
      auto ids = c.GetInput(1);
      auto status = c.GetInput(2);
      auto ctl = c.GetInput(3);
      auto ctx_commit = c.GetInput(4);
      auto ctx_a = c.GetInput(5);
      auto ctx_b = c.GetInput(6);
      auto ptrs = c.GetInput(7);
      const int nst = static_cast<int>(status.GetTensorTypeAndShapeInfo().GetElementCount());
      auto head_ids = c.GetOutput(0, std::vector<int64_t>{1, 2});
      auto head_pos = c.GetOutput(1, std::vector<int64_t>{1, 2});
      auto head_c = c.GetOutput(2, std::vector<int64_t>{1});
      auto sel = c.GetOutput(3, std::vector<int64_t>{1});
      FoldMidKernel<<<1, 32, 0, Stream(api, raw)>>>(
          tok.GetTensorData<i64>(), ids.GetTensorData<i64>(), status.GetTensorData<int>(), nst,
          const_cast<i64*>(ctl.GetTensorData<i64>()), const_cast<i64*>(ctx_commit.GetTensorData<i64>()),
          ctx_a.GetTensorData<i64>(), ctx_b.GetTensorData<i64>(), ptrs.GetTensorData<i64>(),
          head_ids.GetTensorMutableData<i64>(), head_pos.GetTensorMutableData<i64>(), head_c.GetTensorMutableData<int>(),
          sel.GetTensorMutableData<i64>());
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct Mid2State {
  const OrtApi& api;
  int fixed_head_rows = 2;
  Mid2State(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    try {
      Ort::ConstKernelInfo k(info);
      fixed_head_rows = static_cast<int>(k.GetAttribute<int64_t>("head_rows"));
    } catch (const std::exception&) {
    }
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto tok = c.GetInput(0);
      auto ids = c.GetInput(1);
      auto status = c.GetInput(2);
      auto ctl = c.GetInput(3);
      auto ctx_commit = c.GetInput(4);
      auto ctx_a = c.GetInput(5);
      auto ctx_b = c.GetInput(6);
      auto ptrs = c.GetInput(7);
      auto cos_cache = c.GetInput(8);
      auto sin_cache = c.GetInput(9);
      const int nst = static_cast<int>(status.GetTensorTypeAndShapeInfo().GetElementCount());
      const int64_t rows = tok.GetTensorTypeAndShapeInfo().GetElementCount();
      if (rows < 1) throw std::invalid_argument("FoldMid2: rows must be >= 1");
      const int64_t head_rows = (rows >= 2 && rows <= 4) ? rows : fixed_head_rows;
      auto head_ids = c.GetOutput(0, std::vector<int64_t>{1, head_rows});
      auto head_c = c.GetOutput(1, std::vector<int64_t>{1});
      auto slot_ids = c.GetOutput(2, std::vector<int64_t>{1, head_rows});
      auto seqlens = c.GetOutput(3, std::vector<int64_t>{1});
      auto cos_out = c.GetOutput(4, std::vector<int64_t>{1, head_rows, 32});
      auto sin_out = c.GetOutput(5, std::vector<int64_t>{1, head_rows, 32});
      auto sel = c.GetOutput(6, std::vector<int64_t>{1});
      cudaStream_t stream = Stream(api, raw);
      CheckLaunch("before FoldMid2");
      FoldMid2Kernel<<<1, static_cast<unsigned>(32 * (head_rows + 1)), 0, stream>>>(
          tok.GetTensorData<i64>(), ids.GetTensorData<i64>(), status.GetTensorData<int>(), nst,
          const_cast<i64*>(ctl.GetTensorData<i64>()), const_cast<i64*>(ctx_commit.GetTensorData<i64>()),
          ctx_a.GetTensorData<i64>(), ctx_b.GetTensorData<i64>(), ptrs.GetTensorData<i64>(),
          reinterpret_cast<const unsigned short*>(cos_cache.GetTensorData<Ort::BFloat16_t>()),
          reinterpret_cast<const unsigned short*>(sin_cache.GetTensorData<Ort::BFloat16_t>()),
          head_ids.GetTensorMutableData<i64>(), head_c.GetTensorMutableData<int>(), slot_ids.GetTensorMutableData<i64>(),
          seqlens.GetTensorMutableData<int>(), reinterpret_cast<unsigned short*>(cos_out.GetTensorMutableData<Ort::BFloat16_t>()),
          reinterpret_cast<unsigned short*>(sin_out.GetTensorMutableData<Ort::BFloat16_t>()), sel.GetTensorMutableData<i64>(),
          static_cast<int>(rows), static_cast<int>(head_rows));
      CheckLaunch("after FoldMid2Kernel");
      if (c.GetInputCount() > 10 && c.GetOutputCount() > 7) {
        const OrtValue* v_in = nullptr;
        Ort::ThrowOnError(api.KernelContext_GetInput(raw, 10, &v_in));
        if (v_in != nullptr) {
          Ort::ConstValue h_in(v_in);
          auto h_shape = h_in.GetTensorTypeAndShapeInfo().GetShape();
          const int width = static_cast<int>(h_shape.back());
          const std::vector<int64_t> out_h_shape = {1, head_rows, width};
          OrtValue* v_out = nullptr;
          Ort::ThrowOnError(api.KernelContext_GetOutput(raw, 7, out_h_shape.data(), out_h_shape.size(), &v_out));
          if (v_out != nullptr) {
            const void* src_p = nullptr;
            void* dst_p = nullptr;
            Ort::ThrowOnError(api.GetTensorData(v_in, &src_p));
            Ort::ThrowOnError(api.GetTensorMutableData(v_out, &dst_p));
            if (rows == head_rows) {
              cudaMemcpyAsync(dst_p, src_p, static_cast<size_t>(head_rows) * width * sizeof(unsigned short),
                              cudaMemcpyDeviceToDevice, stream);
            } else {
              const int blocks = static_cast<int>((head_rows * width + 255) / 256);
              CopyHeadHiddenKernel<<<blocks, 256, 0, stream>>>(
                  static_cast<const unsigned short*>(src_p), static_cast<unsigned short*>(dst_p),
                  static_cast<int>(rows), static_cast<int>(head_rows), width);
            }
          }
        }
      }
      CheckLaunch("after FoldMid2");
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct EmbedNormState {
  const OrtApi& api;
  float eps;
  EmbedNormState(const OrtApi& a, const OrtKernelInfo* info) : api(a), eps(1e-6f) {
    Ort::ConstKernelInfo k(info);
    eps = k.GetAttribute<float>("epsilon");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto ids = c.GetInput(0);
      auto table = c.GetInput(1);
      auto w = c.GetInput(2);
      auto tshape = table.GetTensorTypeAndShapeInfo().GetShape();
      auto ishape = ids.GetTensorTypeAndShapeInfo().GetShape();
      const int n = static_cast<int>(tshape[1]);
      const int rows = static_cast<int>(ids.GetTensorTypeAndShapeInfo().GetElementCount());
      std::vector<int64_t> oshape = ishape;
      oshape.push_back(n);
      auto out = c.GetOutput(0, oshape);
      const int* committed_ptr = nullptr;
      i64* slot_ids_ptr = nullptr;
      int* seqlens_k_ptr = nullptr;
      int* cu_seqlens_ptr = nullptr;
      if (c.GetInputCount() > 3) {
        const OrtValue* vc = nullptr;
        Ort::ThrowOnError(api.KernelContext_GetInput(raw, 3, &vc));
        if (vc != nullptr) {
          const void* p = nullptr;
          Ort::ThrowOnError(api.GetTensorData(vc, &p));
          committed_ptr = static_cast<const int*>(p);
        }
      }
      auto opt_out = [&](size_t idx, const std::vector<int64_t>& sh) -> void* {
        if (idx >= c.GetOutputCount()) return nullptr;
        OrtValue* vo = nullptr;
        Ort::ThrowOnError(api.KernelContext_GetOutput(raw, idx, sh.data(), sh.size(), &vo));
        if (vo == nullptr) return nullptr;
        void* p = nullptr;
        Ort::ThrowOnError(api.GetTensorMutableData(vo, &p));
        return p;
      };
      slot_ids_ptr = static_cast<i64*>(opt_out(1, {1, rows}));
      seqlens_k_ptr = static_cast<int*>(opt_out(2, {1}));
      cu_seqlens_ptr = static_cast<int*>(opt_out(3, {2}));
      CheckLaunch("before EmbedNorm");
      EmbedNormKernel<<<rows, 512, 0, Stream(api, raw)>>>(
          ids.GetTensorData<i64>(), reinterpret_cast<const unsigned short*>(table.GetTensorData<Ort::BFloat16_t>()),
          w.GetTensorData<float>(), eps, reinterpret_cast<unsigned short*>(out.GetTensorMutableData<Ort::BFloat16_t>()), n,
          committed_ptr, slot_ids_ptr, seqlens_k_ptr, cu_seqlens_ptr, rows);
      CheckLaunch("after EmbedNorm");
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct RowSelectState {
  const OrtApi& api;
  RowSelectState(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto sel = c.GetInput(0);
      const void* x[3];
      void* y[3];
      int n16[3];
      int rows = 2;
      for (int i = 0; i < 3; ++i) {
        auto in = c.GetInput(1 + i);
        auto shape = in.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() < 2 || shape[1] < 2 || shape[1] > 4) throw std::invalid_argument("RowSelect: expected [1,2..4,...]");
        rows = static_cast<int>(shape[1]);
        int64_t row_elems = 1;
        for (size_t d = 2; d < shape.size(); ++d) row_elems *= shape[d];
        if (shape[0] != 1) throw std::invalid_argument("RowSelect: bad row size");
        shape[1] = 1;
        auto out = c.GetOutput(i, shape);
        x[i] = in.GetTensorData<Ort::BFloat16_t>();
        y[i] = out.GetTensorMutableData<Ort::BFloat16_t>();
        n16[i] = static_cast<int>(row_elems);
      }
      CheckLaunch("before RowSelect");
      RowSelectKernel<<<dim3(8, 3), 256, 0, Stream(api, raw)>>>(
          sel.GetTensorData<i64>(), reinterpret_cast<const unsigned short*>(x[0]),
          reinterpret_cast<const unsigned short*>(x[1]), reinterpret_cast<const unsigned short*>(x[2]),
          reinterpret_cast<unsigned short*>(y[0]), reinterpret_cast<unsigned short*>(y[1]),
          reinterpret_cast<unsigned short*>(y[2]), n16[0], n16[1], n16[2], rows);
      CheckLaunch("after RowSelect");
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct DraftPrepState {
  const OrtApi& api;
  int stage = 1;
  DraftPrepState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    try {
      Ort::ConstKernelInfo k(info);
      stage = static_cast<int>(k.GetAttribute<int64_t>("stage"));
    } catch (const std::exception&) {
    }
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto tok = c.GetInput(0);
      auto ctl = c.GetInput(1);
      auto sel = c.GetInput(2);
      auto cos_cache = c.GetInput(3);
      auto sin_cache = c.GetInput(4);
      DraftPrepBuffers in_state{};
      const size_t n_in = c.GetInputCount();
      for (int m = 0; m < 4; ++m) {
        const size_t idx_buf = 5 + 2 * m;
        const size_t idx_len = 6 + 2 * m;
        if (idx_len < n_in) {
          const OrtValue* vb = nullptr;
          const OrtValue* vl = nullptr;
          Ort::ThrowOnError(api.KernelContext_GetInput(raw, idx_buf, &vb));
          Ort::ThrowOnError(api.KernelContext_GetInput(raw, idx_len, &vl));
          if (vb != nullptr && vl != nullptr) {
            const void* pb = nullptr;
            const void* pl = nullptr;
            Ort::ThrowOnError(api.GetTensorData(vb, &pb));
            Ort::ThrowOnError(api.GetTensorData(vl, &pl));
            in_state.buf[m] = static_cast<const unsigned short*>(pb);
            in_state.len[m] = static_cast<const int*>(pl);
          }
        }
      }
      auto head_ids = c.GetOutput(0, std::vector<int64_t>{1, 1});
      auto head_c = c.GetOutput(1, std::vector<int64_t>{1});
      auto slot_ids = c.GetOutput(2, std::vector<int64_t>{1, 1});
      auto seqlens = c.GetOutput(3, std::vector<int64_t>{1});
      auto cos_out = c.GetOutput(4, std::vector<int64_t>{1, 1, 32});
      auto sin_out = c.GetOutput(5, std::vector<int64_t>{1, 1, 32});
      auto out_buf = c.GetOutput(6, std::vector<int64_t>{1, 7, 128});
      auto out_len = c.GetOutput(7, std::vector<int64_t>{1, 2});
      CheckLaunch("before FoldDraftPrep");
      FoldDraftPrepKernel<<<1, 256, 0, Stream(api, raw)>>>(
          tok.GetTensorData<i64>(), const_cast<i64*>(ctl.GetTensorData<i64>()), sel.GetTensorData<i64>(),
          reinterpret_cast<const unsigned short*>(cos_cache.GetTensorData<Ort::BFloat16_t>()),
          reinterpret_cast<const unsigned short*>(sin_cache.GetTensorData<Ort::BFloat16_t>()),
          in_state, stage,
          head_ids.GetTensorMutableData<i64>(), head_c.GetTensorMutableData<int>(),
          slot_ids.GetTensorMutableData<i64>(), seqlens.GetTensorMutableData<int>(),
          reinterpret_cast<unsigned short*>(cos_out.GetTensorMutableData<Ort::BFloat16_t>()),
          reinterpret_cast<unsigned short*>(sin_out.GetTensorMutableData<Ort::BFloat16_t>()),
          reinterpret_cast<unsigned short*>(out_buf.GetTensorMutableData<Ort::BFloat16_t>()),
          out_len.GetTensorMutableData<int>());
      CheckLaunch("after FoldDraftPrep");
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct TailState {
  const OrtApi& api;
  int ctl_slot = 2;
  TailState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    try {
      Ort::ConstKernelInfo k(info);
      ctl_slot = static_cast<int>(k.GetAttribute<int64_t>("ctl_slot"));
    } catch (const std::exception&) {
    }
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto tok = c.GetInput(0);
      auto status = c.GetInput(1);
      auto ctl = c.GetInput(2);
      const int nst = static_cast<int>(status.GetTensorTypeAndShapeInfo().GetElementCount());
      auto done = c.GetOutput(0, std::vector<int64_t>{1});
      FoldTailKernel<<<1, 32, 0, Stream(api, raw)>>>(tok.GetTensorData<i64>(), status.GetTensorData<int>(), nst,
                                                      const_cast<i64*>(ctl.GetTensorData<i64>()),
                                                      done.GetTensorMutableData<int>(), ctl_slot);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

constexpr auto kBF = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
constexpr auto kI32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
constexpr auto kI64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;

template <typename Derived, typename State>
struct FoldOp : Ort::CustomOpBase<Derived, State, true> {
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

struct MidOp : FoldOp<MidOp, MidState> {
  static constexpr std::array<ONNXTensorElementDataType, 8> kIn = {kI64, kI64, kI32, kI64, kI64, kI64, kI64, kI64};
  static constexpr std::array<ONNXTensorElementDataType, 4> kOut = {kI64, kI64, kI32, kI64};
  const char* GetName() const { return "FoldMid"; }
};
struct Mid2Op : FoldOp<Mid2Op, Mid2State> {
  static constexpr std::array<ONNXTensorElementDataType, 11> kIn = {kI64, kI64, kI32, kI64, kI64, kI64, kI64, kI64, kBF, kBF, kBF};
  static constexpr std::array<ONNXTensorElementDataType, 8> kOut = {kI64, kI32, kI64, kI32, kBF, kBF, kI64, kBF};
  const char* GetName() const { return "FoldMid2"; }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t i) const {
    return i >= 10 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t i) const {
    return i >= 7 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
};
struct EmbedNormOp : FoldOp<EmbedNormOp, EmbedNormState> {
  static constexpr std::array<ONNXTensorElementDataType, 4> kIn = {kI64, kBF, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, kI32};
  static constexpr std::array<ONNXTensorElementDataType, 4> kOut = {kBF, kI64, kI32, kI32};
  const char* GetName() const { return "EmbedNorm"; }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t i) const {
    return i >= 3 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t i) const {
    return i >= 1 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
};
struct RowSelectOp : FoldOp<RowSelectOp, RowSelectState> {
  static constexpr std::array<ONNXTensorElementDataType, 4> kIn = {kI64, kBF, kBF, kBF};
  static constexpr std::array<ONNXTensorElementDataType, 3> kOut = {kBF, kBF, kBF};
  const char* GetName() const { return "RowSelectBf16x3"; }
};
struct DraftPrepOp : FoldOp<DraftPrepOp, DraftPrepState> {
  static constexpr std::array<ONNXTensorElementDataType, 13> kIn = {
      kI64, kI64, kI64, kBF, kBF, kBF, kI32, kBF, kI32, kBF, kI32, kBF, kI32};
  static constexpr std::array<ONNXTensorElementDataType, 8> kOut = {
      kI64, kI32, kI64, kI32, kBF, kBF, kBF, kI32};
  const char* GetName() const { return "FoldDraftPrep"; }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t i) const {
    return i >= 7 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
};
struct TailOp : FoldOp<TailOp, TailState> {
  static constexpr std::array<ONNXTensorElementDataType, 3> kIn = {kI64, kI32, kI64};
  static constexpr std::array<ONNXTensorElementDataType, 1> kOut = {kI32};
  const char* GetName() const { return "FoldTail"; }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ORT API");
  Ort::InitApi(api);
  try {
    static MidOp mid;
    static EmbedNormOp embed_norm;
    static Mid2Op mid2;
    static RowSelectOp row_select;
    static DraftPrepOp draft_prep;
    static TailOp tail;
    static Ort::CustomOpDomain domain("qwen.fold");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&mid);
      domain.Add(&embed_norm);
      domain.Add(&mid2);
      domain.Add(&row_select);
      domain.Add(&draft_prep);
      domain.Add(&tail);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_FAIL, e.what());
  }
}
