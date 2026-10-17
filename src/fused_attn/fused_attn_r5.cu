// Attention-side fused decode ops for the Qwen3.8 S=1 capture graph (domain qwen.fused_attn).
// Each kernel reproduces the arithmetic and BF16/FP32 rounding points of the ONNX chain it
// replaces (including ORT's reduction order and activation formulas) so results are bit-identical.
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"
#include "common/sigmoid_acc.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float F(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 B(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float R(float v) { return F(B(v)); }

// onnxruntime::cuda::OP_Sigmoid<BFloat16> after the accurate-sigmoid fix (provider commit 453ac97a68): fp32 evaluation, one BF16 rounding.
__device__ __forceinline__ float OrtSigmoid(float a) { return qwen_acc::SigmoidBf16(a); }

void CheckLaunch() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

cudaStream_t Stream(const OrtApi& api, OrtKernelContext* context) {
  void* raw = nullptr;
  Ort::ThrowOnError(api.KernelContext_GetGPUComputeStream(context, &raw));
  return static_cast<cudaStream_t>(raw);
}

template <typename T>
const T* In(const Ort::KernelContext& c, size_t i) {
  return reinterpret_cast<const T*>(c.GetInput(i).GetTensorRawData());
}
template <typename T>
T* Out(Ort::UnownedValue& v) {
  return reinterpret_cast<T*>(v.GetTensorMutableRawData());
}
int64_t Count(const Ort::ConstValue& v) {
  int64_t n = 1;
  for (auto d : v.GetTensorTypeAndShapeInfo().GetShape()) n *= d;
  return n;
}
std::vector<int64_t> Shape(const Ort::ConstValue& v) { return v.GetTensorTypeAndShapeInfo().GetShape(); }

// Rows per call (speculative verify): the state-carrying ops run a chain over at most this many rows.
constexpr int kMaxRows = 4;

// Raw device pointer of an optional trailing output (nullptr when the node does not connect it).
void* OptionalOut(const OrtApi& api, OrtKernelContext* raw, Ort::KernelContext& c, size_t index, const std::vector<int64_t>& shape) {
  if (index >= c.GetOutputCount()) return nullptr;
  OrtValue* value = nullptr;
  Ort::ThrowOnError(api.KernelContext_GetOutput(raw, index, shape.data(), shape.size(), &value));
  if (value == nullptr) return nullptr;
  void* data = nullptr;
  Ort::ThrowOnError(api.GetTensorMutableData(value, &data));
  return data;
}

// ---------------------------------------------------------------------------------------------
// GdnConvPrep / GdnProjPrep: Transpose + Concat/Slice(state) + CausalConvWithState(K=4, silu) +
// Transpose + Split(q,k,v) + Cast(b), Cast(a) of one GatedDeltaNet layer.
//   GdnConvPrep : qkv [B,1,C], state [B,C,4], weight [C,1,4], b [B,1,Hv], a [B,1,Hv] (bf16)
//   GdnProjPrep : merged projection output [B,1,N] (qkv | z | b | a slices) instead of the separate
//                 tensors; additionally emits z (replaces the merged-projection Split node).
//   -> q [B,1,Q], k [B,1,Kd], v [B,1,V], (z [B,1,Z]), state_out [B,C,4] bf16, b32/a32 [B,1,Hv] fp32
// ---------------------------------------------------------------------------------------------
struct GdnParams {
  const bf16* x; int64_t x_stride;
  const bf16* b_src; int64_t b_stride;
  const bf16* a_src; int64_t a_stride;
  const bf16* z_src; int64_t z_stride;
  const bf16* state; const bf16* weight;
  bf16 *q, *k, *v, *z, *state_out;
  bf16* mid_state_out[3];
  float *b32, *a32;
  int batch, rows, channels, q_size, k_size, heads, z_size;
};

// R rows (token = batch * rows + r) are a chain over the conv window: row r reads the 4-sample window that
// the S=1 op would have found in its state after r previous calls. state_out is the state after the last
// row; mid_state_out[m] (optional) the state after row m (m in {0, 1, 2}).
template <int ROWS>
__global__ void GdnPrepKernel(GdnParams p) {
  const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t conv_total = static_cast<int64_t>(p.batch) * p.channels;
  const int64_t tokens = static_cast<int64_t>(p.batch) * ROWS;
  if (t < conv_total) {
    const int bi = static_cast<int>(t / p.channels);
    const int c = static_cast<int>(t % p.channels);
    const bf16* ps = p.state + t * 4;
    const bf16* w = p.weight + static_cast<int64_t>(c) * 4;
    bf16 win[4 + ROWS];
#pragma unroll
    for (int i = 0; i < 4; ++i) win[i] = ps[i];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int64_t tok = static_cast<int64_t>(bi) * ROWS + r;
      const bf16 x = p.x[tok * p.x_stride + c];
      win[4 + r] = x;
      float sum = 0.0f;
      sum += F(w[0]) * F(win[r + 1]);
      sum += F(w[1]) * F(win[r + 2]);
      sum += F(w[2]) * F(win[r + 3]);
      sum += F(w[3]) * F(x);
      sum = sum / (1.0f + expf(-sum));
      const bf16 y = B(sum);
      if (c < p.q_size) {
        p.q[tok * p.q_size + c] = y;
      } else if (c < p.q_size + p.k_size) {
        p.k[tok * p.k_size + (c - p.q_size)] = y;
      } else {
        const int vs = p.channels - p.q_size - p.k_size;
        p.v[tok * vs + (c - p.q_size - p.k_size)] = y;
      }
    }
    bf16* po = p.state_out + t * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) po[i] = win[ROWS + i];
    if (p.mid_state_out[0] != nullptr) {
      bf16* pm0 = p.mid_state_out[0] + t * 4;
#pragma unroll
      for (int i = 0; i < 4; ++i) pm0[i] = win[1 + i];
    }
    if constexpr (ROWS >= 3) {
      if (p.mid_state_out[1] != nullptr) {
        bf16* pm1 = p.mid_state_out[1] + t * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) pm1[i] = win[2 + i];
      }
    }
    if constexpr (ROWS >= 4) {
      if (p.mid_state_out[2] != nullptr) {
        bf16* pm2 = p.mid_state_out[2] + t * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) pm2[i] = win[3 + i];
      }
    }
    return;
  }
  int64_t r = t - conv_total;
  const int64_t ba = tokens * p.heads;
  if (r < ba) {
    p.b32[r] = F(p.b_src[(r / p.heads) * p.b_stride + r % p.heads]);
    return;
  }
  r -= ba;
  if (r < ba) {
    p.a32[r] = F(p.a_src[(r / p.heads) * p.a_stride + r % p.heads]);
    return;
  }
  r -= ba;
  if (r < tokens * p.z_size) {
    p.z[r] = p.z_src[(r / p.z_size) * p.z_stride + r % p.z_size];
  }
}

__global__ void GdnPrepMultiRowKernel(GdnParams p) {
  const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t conv_total = static_cast<int64_t>(p.batch) * p.channels;
  const int64_t tokens = static_cast<int64_t>(p.batch) * p.rows;
  if (t < conv_total) {
    const int bi = static_cast<int>(t / p.channels);
    const int c = static_cast<int>(t % p.channels);
    const bf16* ps = p.state + t * 4;
    const bf16* w = p.weight + static_cast<int64_t>(c) * 4;
    bf16 w0 = ps[0], w1 = ps[1], w2 = ps[2], w3 = ps[3];
    for (int r = 0; r < p.rows; ++r) {
      const int64_t tok = static_cast<int64_t>(bi) * p.rows + r;
      const bf16 x = p.x[tok * p.x_stride + c];
      float sum = 0.0f;
      sum += F(w[0]) * F(w1);
      sum += F(w[1]) * F(w2);
      sum += F(w[2]) * F(w3);
      sum += F(w[3]) * F(x);
      sum = sum / (1.0f + expf(-sum));
      const bf16 y = B(sum);
      if (c < p.q_size) {
        p.q[tok * p.q_size + c] = y;
      } else if (c < p.q_size + p.k_size) {
        p.k[tok * p.k_size + (c - p.q_size)] = y;
      } else {
        const int vs = p.channels - p.q_size - p.k_size;
        p.v[tok * vs + (c - p.q_size - p.k_size)] = y;
      }
      w0 = w1;
      w1 = w2;
      w2 = w3;
      w3 = x;
      if (r < 3 && p.mid_state_out[r] != nullptr) {
        bf16* pm = p.mid_state_out[r] + t * 4;
        pm[0] = w0;
        pm[1] = w1;
        pm[2] = w2;
        pm[3] = w3;
      }
    }
    bf16* po = p.state_out + t * 4;
    po[0] = w0;
    po[1] = w1;
    po[2] = w2;
    po[3] = w3;
    return;
  }
  int64_t r = t - conv_total;
  const int64_t ba = tokens * p.heads;
  if (r < ba) {
    p.b32[r] = F(p.b_src[(r / p.heads) * p.b_stride + r % p.heads]);
    return;
  }
  r -= ba;
  if (r < ba) {
    p.a32[r] = F(p.a_src[(r / p.heads) * p.a_stride + r % p.heads]);
    return;
  }
  r -= ba;
  if (r < tokens * p.z_size) {
    p.z[r] = p.z_src[(r / p.z_size) * p.z_stride + r % p.z_size];
  }
}

// ---------------------------------------------------------------------------------------------
// QkNormRope / QsaProjPrep: Cast(fp32) + RMSNorm chain (Mul, ReduceMean, Add, Sqrt, Reciprocal, Mul,
// Mul, Cast) + RotaryEmbedding (non-interleaved, partial rotary) for the q and k heads of a QSA layer.
//   q_in: per token Hq groups of q_stride values (first head_dim = q, rest = gate); k_in: Hk*head_dim
//   cos/sin [B,1,rot/2] bf16, q_w/k_w [head_dim] fp32 -> q_out [B,1,Hq*head_dim], k_out [B,1,Hk*head_dim]
//   QkNormRope reads separate q/k tensors; QsaProjPrep reads both from the merged projection output
//   and copies two further slices (indexer qk, v) out (replaces the merged-projection Split node).
// Reduction order mirrors onnxruntime reduce_matrix_columns (64 threads, 4 elems/thread, shfl tree).
// ---------------------------------------------------------------------------------------------
constexpr int kNormThreads = 64;

struct CopySeg {
  const bf16* src;
  bf16* dst;
  int64_t src_stride;
  int size;
};

struct RopeParams {
  const bf16* q_in; int64_t q_batch_stride;
  const bf16* k_in; int64_t k_batch_stride;
  const float *q_w, *k_w;
  const bf16 *cos_c, *sin_c;
  bf16 *q_out, *k_out;
  CopySeg copy[2];
  int batch, hq, hk, head_dim, q_stride, rot, copies;
  float eps;
};

__global__ void QkNormRopeKernel(RopeParams p) {
  extern __shared__ unsigned char smem_raw[];
  float* red = reinterpret_cast<float*>(smem_raw);                 // [2]
  bf16* y = reinterpret_cast<bf16*>(smem_raw + 2 * sizeof(float));  // [head_dim]

  const int heads = p.hq + p.hk;
  const int tid = threadIdx.x;
  if (blockIdx.x >= p.batch * heads) {  // pass-through slices of the merged projection output
    // one block per (segment, token): the tail of this kernel is the longest copy
    const int copy_block = blockIdx.x - p.batch * heads;
    const CopySeg seg = p.copy[copy_block / p.batch];
    const int token = copy_block % p.batch;
    for (int i = tid; i < seg.size; i += blockDim.x) {
      seg.dst[token * seg.size + i] = seg.src[token * seg.src_stride + i];
    }
    return;
  }
  const int bi = blockIdx.x / heads;
  const int h = blockIdx.x % heads;
  const bool is_q = h < p.hq;
  const bf16* x = is_q ? p.q_in + bi * p.q_batch_stride + static_cast<int64_t>(h) * p.q_stride
                       : p.k_in + bi * p.k_batch_stride + static_cast<int64_t>(h - p.hq) * p.head_dim;
  const float* w = is_q ? p.q_w : p.k_w;
  bf16* out = is_q ? p.q_out + (static_cast<int64_t>(bi) * p.hq + h) * p.head_dim
                   : p.k_out + (static_cast<int64_t>(bi) * p.hk + (h - p.hq)) * p.head_dim;

  float v[4];
  float value = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int idx = tid + i * kNormThreads;
    v[i] = idx < p.head_dim ? F(x[idx]) : 0.0f;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (tid + i * kNormThreads < p.head_dim) value += v[i] * v[i];
  }
  for (int stride = 16; stride > 0; stride >>= 1) value += __shfl_down_sync(0xffffffff, value, stride);
  if ((tid & 31) == 0) red[tid >> 5] = value;
  __syncthreads();
  const float total = red[0] + red[1];
  const float mean = total / static_cast<float>(p.head_dim);
  const float inv = 1.0f / sqrtf(mean + p.eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int idx = tid + i * kNormThreads;
    if (idx < p.head_dim) y[idx] = B((v[i] * inv) * w[idx]);
  }
  __syncthreads();
  const int half_rot = p.rot / 2;
  const bf16* cs = p.cos_c + static_cast<int64_t>(bi) * half_rot;
  const bf16* sn = p.sin_c + static_cast<int64_t>(bi) * half_rot;
#pragma unroll
  for (int i4 = 0; i4 < 4; ++i4) {
    const int i = tid + i4 * kNormThreads;
    if (i >= p.head_dim) continue;
    if (i >= p.rot) {
      out[i] = y[i];
    } else {
      const int cache_idx = i % half_rot;
      const bf16 sign = B(i < half_rot ? -1.0f : 1.0f);
      const int j = (i + half_rot) % p.rot;
      out[i] = y[i] * cs[cache_idx] + sign * y[j] * sn[cache_idx];
    }
  }
}

// y = bf16(attn * bf16(sigmoid(gate))); gate of (token, head) sits at q_in[token*tok_stride + q_offset +
// head*q_stride + head_dim + d] (q projection output, or the merged projection output).
__global__ void SigmoidGateMulKernel(
    const bf16* attn, const bf16* q_in, bf16* y, int64_t n, int head_dim, int heads, int q_stride,
    int64_t tok_stride, int64_t q_offset) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const int64_t row = i / head_dim;  // (token, head)
  const int d = static_cast<int>(i % head_dim);
  const int64_t tok = row / heads;
  const int h = static_cast<int>(row % heads);
  const float g = F(q_in[tok * tok_stride + q_offset + static_cast<int64_t>(h) * q_stride + head_dim + d]);
  const float s = F(B(OrtSigmoid(g)));
  y[i] = B(F(attn[i]) * s);
}

// ---------------------------------------------------------------------------------------------
// GdnRecurrent: com.microsoft::GatedDeltaNet restricted to the configuration of this model
// (update_rule=gated_delta, gate_activation=qwen, beta_activation=sigmoid, qk_l2_norm, bf16 activations,
// fp32 state [B,Hv,V,K] V-major, K=V=128, no cu_seqlens) for R in {1,2} rows with an optional mid state.
// The per-row arithmetic is a verbatim copy of ORT's GatedDeltaNetDecodeWarpKernel<bf16,128,8> (the engine
// SelectPlan picks for bf16 and fewer than 32 tokens): the recurrent state lives in registers, so the
// row chain 0 -> 1 is exact w.r.t. two S=1 calls; the state after row 0 is additionally written to mid_state.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float GdnSoftPlus(float x) { return x > 20.0f ? x : __logf(1.0f + __expf(x)); }
__device__ __forceinline__ float GdnSigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float GdnWarpReduce(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
  return v;
}

struct GdnRecParams {
  const bf16 *q, *k, *v;
  const float *decay, *beta, *state, *a_log, *dt_bias;
  bf16* out;
  float *final_state;
  float* mid_state[3];
  int rows, num_heads_q, num_heads_k, num_heads_v, d_v;
  float scale;
};

constexpr int kGdnDK = 128;
constexpr int kGdnWarps = 8;

// Everything that does not depend on the recurrent state (q/k loads, L2 norms, decay, beta, v) is computed for all
// rows first, so the independent warp reductions of the rows overlap; the state chain per row is unchanged.
template <int ROWS>
__global__ __launch_bounds__(32 * kGdnWarps) void GdnRecurrentKernel(GdnRecParams p) {
  constexpr int DK = kGdnDK;
  constexpr int kRegs = DK / 32;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int b = blockIdx.x, hv = blockIdx.y;
  const int col = blockIdx.z * kGdnWarps + warp;
  if (col >= p.d_v) return;

  const int hq = hv * p.num_heads_q / p.num_heads_v;
  const int hk = hv * p.num_heads_k / p.num_heads_v;
  const int64_t tok_base = static_cast<int64_t>(b) * ROWS;
  const int64_t q_stride = static_cast<int64_t>(p.num_heads_q) * DK;
  const int64_t k_stride = static_cast<int64_t>(p.num_heads_k) * DK;
  const int64_t v_stride = static_cast<int64_t>(p.num_heads_v) * p.d_v;
  const int64_t st_off = (static_cast<int64_t>(b) * p.num_heads_v + hv) * static_cast<int64_t>(p.d_v) * DK +
                         static_cast<int64_t>(col) * DK;

  float s[kRegs];
#pragma unroll
  for (int i = 0; i < kRegs; ++i) s[i] = __ldcs(p.state + st_off + lane + i * 32);

  float kv[ROWS][kRegs], qv[ROWS][kRegs], decay_scalar[ROWS], beta_t[ROWS], vv[ROWS];
#pragma unroll
  for (int t = 0; t < ROWS; ++t) {
    const int64_t tok = tok_base + t;
    const int64_t gi = tok * p.num_heads_v + hv;
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      const int k = lane + i * 32;
      kv[t][i] = static_cast<float>(p.k[tok * k_stride + hk * DK + k]);
      qv[t][i] = static_cast<float>(p.q[tok * q_stride + hq * DK + k]);
    }
    float nq = 0.0f, nk = 0.0f;
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      nq += qv[t][i] * qv[t][i];
      nk += kv[t][i] * kv[t][i];
    }
    const float rq = rsqrtf(GdnWarpReduce(nq) + 1e-12f);
    const float rk = rsqrtf(GdnWarpReduce(nk) + 1e-12f);
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      qv[t][i] *= rq;
      kv[t][i] *= rk;
    }
    const float bias = p.dt_bias[hv];
    const float neg_rate = -__expf(p.a_log[hv]);
    decay_scalar[t] = __expf(neg_rate * GdnSoftPlus(p.decay[gi] + bias));
    beta_t[t] = GdnSigmoid(p.beta[gi]);
    vv[t] = static_cast<float>(p.v[tok * v_stride + hv * p.d_v + col]);
  }

#pragma unroll
  for (int t = 0; t < ROWS; ++t) {
    const int64_t tok = tok_base + t;
#pragma unroll
    for (int i = 0; i < kRegs; ++i) s[i] *= decay_scalar[t];

    float r = 0.0f;
#pragma unroll
    for (int i = 0; i < kRegs; ++i) r += s[i] * kv[t][i];
    r = GdnWarpReduce(r);

    const float delta = beta_t[t] * (vv[t] - r);

    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      s[i] += kv[t][i] * delta;
      o += s[i] * qv[t][i];
    }
    o = GdnWarpReduce(o);
    if (lane == 0) p.out[tok * v_stride + hv * p.d_v + col] = static_cast<bf16>(p.scale * o);
    if (t + 1 < ROWS && p.mid_state[t] != nullptr) {
#pragma unroll
      for (int i = 0; i < kRegs; ++i) __stcs(p.mid_state[t] + st_off + lane + i * 32, s[i]);
    }
  }
#pragma unroll
  for (int i = 0; i < kRegs; ++i) __stcs(p.final_state + st_off + lane + i * 32, s[i]);
}

__global__ __launch_bounds__(32 * kGdnWarps) void GdnRecurrentMultiRowKernel(GdnRecParams p) {
  constexpr int DK = kGdnDK;
  constexpr int kRegs = DK / 32;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int b = blockIdx.x, hv = blockIdx.y;
  const int col = blockIdx.z * kGdnWarps + warp;
  if (col >= p.d_v) return;

  const int hq = hv * p.num_heads_q / p.num_heads_v;
  const int hk = hv * p.num_heads_k / p.num_heads_v;
  const int64_t tok_base = static_cast<int64_t>(b) * p.rows;
  const int64_t q_stride = static_cast<int64_t>(p.num_heads_q) * DK;
  const int64_t k_stride = static_cast<int64_t>(p.num_heads_k) * DK;
  const int64_t v_stride = static_cast<int64_t>(p.num_heads_v) * p.d_v;
  const int64_t st_off = (static_cast<int64_t>(b) * p.num_heads_v + hv) * static_cast<int64_t>(p.d_v) * DK +
                         static_cast<int64_t>(col) * DK;

  float s[kRegs];
#pragma unroll
  for (int i = 0; i < kRegs; ++i) s[i] = __ldcs(p.state + st_off + lane + i * 32);

  const float bias = p.dt_bias[hv];
  const float neg_rate = -__expf(p.a_log[hv]);
  for (int t = 0; t < p.rows; ++t) {
    const int64_t tok = tok_base + t;
    const int64_t gi = tok * p.num_heads_v + hv;
    float qv[kRegs], kv[kRegs];
    float nq = 0.0f, nk = 0.0f;
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      const int k = lane + i * 32;
      kv[i] = static_cast<float>(p.k[tok * k_stride + hk * DK + k]);
      qv[i] = static_cast<float>(p.q[tok * q_stride + hq * DK + k]);
      nq += qv[i] * qv[i];
      nk += kv[i] * kv[i];
    }
    const float rq = rsqrtf(GdnWarpReduce(nq) + 1e-12f);
    const float rk = rsqrtf(GdnWarpReduce(nk) + 1e-12f);
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      qv[i] *= rq;
      kv[i] *= rk;
    }
    const float decay_scalar = __expf(neg_rate * GdnSoftPlus(p.decay[gi] + bias));
    const float beta_t = GdnSigmoid(p.beta[gi]);
    const float vv = static_cast<float>(p.v[tok * v_stride + hv * p.d_v + col]);
#pragma unroll
    for (int i = 0; i < kRegs; ++i) s[i] *= decay_scalar;

    float r = 0.0f;
#pragma unroll
    for (int i = 0; i < kRegs; ++i) r += s[i] * kv[i];
    r = GdnWarpReduce(r);

    const float delta = beta_t * (vv - r);

    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      s[i] += kv[i] * delta;
      o += s[i] * qv[i];
    }
    o = GdnWarpReduce(o);
    if (lane == 0) p.out[tok * v_stride + hv * p.d_v + col] = static_cast<bf16>(p.scale * o);
    if (t < 3 && t + 1 < p.rows && p.mid_state[t] != nullptr) {
#pragma unroll
      for (int i = 0; i < kRegs; ++i) __stcs(p.mid_state[t] + st_off + lane + i * 32, s[i]);
    }
  }
#pragma unroll
  for (int i = 0; i < kRegs; ++i) __stcs(p.final_state + st_off + lane + i * 32, s[i]);
}

// ---------------------------------------------------------------------------------------------
// PleMidState: the PLE conv state ([B,C,W], bf16, oldest first) and the PLE context ([B,2] int64: previous
// token ids) after row m (m in {0, 1, 2}) of an R-row call.
// ---------------------------------------------------------------------------------------------
struct PleMidOut {
  bf16* state[3];
  int64_t* ctx[3];
};

__global__ void PleMidStateKernel(const bf16* state, const bf16* x, const int64_t* ctx, const int64_t* ids,
                                  PleMidOut out, int batch, int channels, int width,
                                  int rows, int ctx_len) {
  const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const int64_t total = static_cast<int64_t>(batch) * channels;
  if (t < total) {
    const int bi = static_cast<int>(t / channels);
    const int c = static_cast<int>(t % channels);
    const bf16* ps = state + t * width;
    for (int m = 0; m < 3; ++m) {
      const int s = m + 1;
      if (out.state[m] != nullptr && s <= rows) {
        bf16* pm = out.state[m] + t * width;
        for (int j = 0; j < width; ++j) {
          pm[j] = (j + s < width)
                      ? ps[j + s]
                      : x[(static_cast<int64_t>(bi) * rows + (j + s - width)) * channels + c];
        }
      }
    }
    return;
  }
  const int64_t r = t - total;
  if (r < static_cast<int64_t>(batch) * ctx_len) {
    const int bi = static_cast<int>(r / ctx_len);
    const int j = static_cast<int>(r % ctx_len);
    for (int m = 0; m < 3; ++m) {
      const int s = m + 1;
      if (out.ctx[m] != nullptr && s <= rows) {
        out.ctx[m][r] = (j + s < ctx_len)
                            ? ctx[static_cast<int64_t>(bi) * ctx_len + j + s]
                            : ids[static_cast<int64_t>(bi) * rows + (j + s - ctx_len)];
      }
    }
  }
}

struct GdnRecurrentState {
  const OrtApi& api;
  float scale;
  GdnRecurrentState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    auto text = [&](const char* name, const char* fallback) {
      try { return k.GetAttribute<std::string>(name); } catch (...) { return std::string(fallback); }
    };
    auto integer = [&](const char* name, int64_t fallback) {
      try { return k.GetAttribute<int64_t>(name); } catch (...) { return fallback; }
    };
    if (text("update_rule", "gated_delta") != "gated_delta" || text("gate_activation", "none") != "qwen" ||
        text("beta_activation", "none") != "sigmoid" || integer("qk_l2_norm", 0) != 1) {
      throw std::invalid_argument("GdnRecurrent: only update_rule=gated_delta gate_activation=qwen beta_activation=sigmoid qk_l2_norm=1");
    }
    float s = 0.0f;
    try { s = k.GetAttribute<float>("scale"); } catch (...) {}
    scale = s != 0.0f ? s : 1.0f / std::sqrt(static_cast<float>(kGdnDK));
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      const auto qs = Shape(c.GetInput(0));
      const auto ks = Shape(c.GetInput(1));
      const auto vs = Shape(c.GetInput(2));
      const auto ss = Shape(c.GetInput(6));
      if (qs.size() != 4 || ks.size() != 4 || vs.size() != 4 || ss.size() != 4 || qs[1] < 1 ||
          qs[3] != kGdnDK || ks[3] != kGdnDK || vs[3] != kGdnDK || ss[2] != kGdnDK || ss[3] != kGdnDK) {
        throw std::invalid_argument("GdnRecurrent: expected q,k [B,R,Hq,128], v [B,R,Hv,128], state [B,Hv,128,128], R >= 1");
      }
      const int64_t batch = qs[0], rows = qs[1], hq = qs[2], hk = ks[2], hv = vs[2];
      if (ks[0] != batch || ks[1] != rows || vs[0] != batch || vs[1] != rows || ss[0] != batch || ss[1] != hv ||
          hv % hq != 0 || hq != hk || Count(c.GetInput(4)) != batch * rows * hv ||
          Count(c.GetInput(5)) != batch * rows * hv || Count(c.GetInput(7)) != hv || Count(c.GetInput(8)) != hv) {
        throw std::invalid_argument("GdnRecurrent: shape mismatch");
      }
      GdnRecParams p{};
      p.q = In<bf16>(c, 0); p.k = In<bf16>(c, 1); p.v = In<bf16>(c, 2);
      p.decay = In<float>(c, 4); p.beta = In<float>(c, 5); p.state = In<float>(c, 6);
      p.a_log = In<float>(c, 7); p.dt_bias = In<float>(c, 8);
      auto out = c.GetOutput(0, std::vector<int64_t>{batch, rows, hv, kGdnDK});
      auto fs = c.GetOutput(1, ss);
      p.out = Out<bf16>(out); p.final_state = Out<float>(fs);
      for (int m = 0; m < 3; ++m) {
        p.mid_state[m] = static_cast<float*>(OptionalOut(api, raw, c, 2 + m, ss));
        if (p.mid_state[m] != nullptr && p.mid_state[m] == p.state) {
          throw std::invalid_argument("GdnRecurrent: state outputs must not alias the input state");
        }
      }
      p.rows = static_cast<int>(rows); p.num_heads_q = static_cast<int>(hq); p.num_heads_k = static_cast<int>(hk);
      p.num_heads_v = static_cast<int>(hv); p.d_v = kGdnDK; p.scale = scale;
      if (p.final_state == p.state) {
        throw std::invalid_argument("GdnRecurrent: state outputs must not alias the input state");
      }
      const dim3 grid(static_cast<unsigned>(batch), static_cast<unsigned>(hv), kGdnDK / kGdnWarps);
      if (rows == 1) {
        GdnRecurrentKernel<1><<<grid, dim3(32, kGdnWarps, 1), 0, Stream(api, raw)>>>(p);
      } else if (rows == 2) {
        GdnRecurrentKernel<2><<<grid, dim3(32, kGdnWarps, 1), 0, Stream(api, raw)>>>(p);
      } else if (rows == 3) {
        GdnRecurrentKernel<3><<<grid, dim3(32, kGdnWarps, 1), 0, Stream(api, raw)>>>(p);
      } else if (rows == 4) {
        GdnRecurrentKernel<4><<<grid, dim3(32, kGdnWarps, 1), 0, Stream(api, raw)>>>(p);
      } else {
        GdnRecurrentMultiRowKernel<<<grid, dim3(32, kGdnWarps, 1), 0, Stream(api, raw)>>>(p);
      }
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct PleMidStateState {
  const OrtApi& api;
  PleMidStateState(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      const auto ss = Shape(c.GetInput(0));
      const auto xs = Shape(c.GetInput(1));
      const auto cs = Shape(c.GetInput(2));
      const auto is = Shape(c.GetInput(3));
      if (ss.size() != 3 || xs.size() != 3 || cs.size() != 2 || is.size() != 2 || xs[0] != ss[0] || xs[2] != ss[1] ||
          is[0] != ss[0] || cs[0] != ss[0] || xs[1] < 1 || xs[1] != is[1] || ss[2] < 2 || cs[1] < 1) {
        throw std::invalid_argument("PleMidState: expected state [B,C,W], x [B,R,C], ctx [B,L], ids [B,R]");
      }
      auto mid = c.GetOutput(0, ss);
      auto mid_ctx = c.GetOutput(1, cs);
      PleMidOut out{};
      out.state[0] = Out<bf16>(mid);
      out.ctx[0] = Out<int64_t>(mid_ctx);
      for (int m = 1; m < 3; ++m) {
        out.state[m] = static_cast<bf16*>(OptionalOut(api, raw, c, 2 * m, ss));
        out.ctx[m] = static_cast<int64_t*>(OptionalOut(api, raw, c, 2 * m + 1, cs));
      }
      const int64_t total = ss[0] * ss[1] + cs[0] * cs[1];
      PleMidStateKernel<<<(total + 255) / 256, 256, 0, Stream(api, raw)>>>(
          In<bf16>(c, 0), In<bf16>(c, 1), In<int64_t>(c, 2), In<int64_t>(c, 3), out,
          static_cast<int>(ss[0]), static_cast<int>(ss[1]), static_cast<int>(ss[2]),
          static_cast<int>(xs[1]), static_cast<int>(cs[1]));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ---------------------------------------------------------------------------------------------
template <bool kMerged>
struct GdnPrepState {
  const OrtApi& api;
  int64_t q_size, k_size, off_qkv = 0, off_z = 0, z_size = 0, off_b = 0, off_a = 0, merged_heads = 0;
  GdnPrepState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    q_size = k.GetAttribute<int64_t>("q_size");
    k_size = k.GetAttribute<int64_t>("k_size");
    if (kMerged) {
      off_qkv = k.GetAttribute<int64_t>("off_qkv");
      off_z = k.GetAttribute<int64_t>("off_z");
      z_size = k.GetAttribute<int64_t>("z_size");
      off_b = k.GetAttribute<int64_t>("off_b");
      off_a = k.GetAttribute<int64_t>("off_a");
      merged_heads = k.GetAttribute<int64_t>("heads");
    }
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto in0 = Shape(c.GetInput(0));
      auto state_shape = Shape(c.GetInput(1));
      const size_t state_idx = 1, weight_idx = 2;
      if (in0.size() != 3 || in0[1] < 1 || state_shape.size() != 3 || state_shape[2] != 4) {
        throw std::invalid_argument("GdnPrep: expected [B,R,*] activations (R >= 1) and state [B,C,4]");
      }
      const int64_t batch = in0[0], rows = in0[1], channels = state_shape[1];
      const int64_t tokens = batch * rows;
      if (state_shape[0] != batch || Count(c.GetInput(weight_idx)) != channels * 4) {
        throw std::invalid_argument("GdnPrep: shape mismatch");
      }
      const int64_t v_size = channels - q_size - k_size;
      if (v_size <= 0) throw std::invalid_argument("GdnPrep: bad split sizes");
      int64_t heads;
      GdnParams p{};
      p.state = In<bf16>(c, state_idx);
      p.weight = In<bf16>(c, weight_idx);
      std::vector<int64_t> bshape;
      if (kMerged) {
        const int64_t n = in0[2];
        heads = merged_heads;
        if (heads <= 0 || off_a + heads > n || off_b + heads > n || off_qkv + channels > n || off_z + z_size > n) {
          throw std::invalid_argument("GdnProjPrep: slice offsets exceed the merged projection width");
        }
        const bf16* m = In<bf16>(c, 0);
        p.x = m + off_qkv; p.x_stride = n;
        p.b_src = m + off_b; p.b_stride = n;
        p.a_src = m + off_a; p.a_stride = n;
        p.z_src = m + off_z; p.z_stride = n;
        bshape = {batch, rows, heads};
      } else {
        bshape = Shape(c.GetInput(3));
        heads = bshape.back();
        if (in0[2] != channels || Count(c.GetInput(3)) != tokens * heads || Count(c.GetInput(4)) != tokens * heads) {
          throw std::invalid_argument("GdnConvPrep: shape mismatch");
        }
        p.x = In<bf16>(c, 0); p.x_stride = channels;
        p.b_src = In<bf16>(c, 3); p.b_stride = heads;
        p.a_src = In<bf16>(c, 4); p.a_stride = heads;
      }
      auto q = c.GetOutput(0, std::vector<int64_t>{batch, rows, q_size});
      auto k = c.GetOutput(1, std::vector<int64_t>{batch, rows, k_size});
      auto v = c.GetOutput(2, std::vector<int64_t>{batch, rows, v_size});
      const size_t base = kMerged ? 4 : 3;
      auto so = c.GetOutput(base, state_shape);
      auto b32 = c.GetOutput(base + 1, bshape);
      auto a32 = c.GetOutput(base + 2, bshape);
      p.q = Out<bf16>(q); p.k = Out<bf16>(k); p.v = Out<bf16>(v);
      p.state_out = Out<bf16>(so); p.b32 = Out<float>(b32); p.a32 = Out<float>(a32);
      // optional trailing outputs: the conv state after rows 0, 1, 2
      for (int m = 0; m < 3; ++m) {
        p.mid_state_out[m] = static_cast<bf16*>(OptionalOut(api, raw, c, base + 3 + m, state_shape));
      }
      if (kMerged) {
        auto z = c.GetOutput(3, std::vector<int64_t>{batch, rows, z_size});
        p.z = Out<bf16>(z);
      }
      p.batch = static_cast<int>(batch); p.rows = static_cast<int>(rows); p.channels = static_cast<int>(channels);
      p.q_size = static_cast<int>(q_size); p.k_size = static_cast<int>(k_size);
      p.heads = static_cast<int>(heads); p.z_size = static_cast<int>(kMerged ? z_size : 0);
      const int64_t total = batch * channels + 2 * tokens * heads + tokens * p.z_size;
      if (rows == 1) {
        GdnPrepKernel<1><<<(total + 255) / 256, 256, 0, Stream(api, raw)>>>(p);
      } else if (rows == 2) {
        GdnPrepKernel<2><<<(total + 255) / 256, 256, 0, Stream(api, raw)>>>(p);
      } else if (rows == 3) {
        GdnPrepKernel<3><<<(total + 255) / 256, 256, 0, Stream(api, raw)>>>(p);
      } else if (rows == 4) {
        GdnPrepKernel<4><<<(total + 255) / 256, 256, 0, Stream(api, raw)>>>(p);
      } else {
        GdnPrepMultiRowKernel<<<(total + 255) / 256, 256, 0, Stream(api, raw)>>>(p);
      }
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

template <bool kMerged>
struct QkNormRopeState {
  const OrtApi& api;
  int64_t hq, hk, head_dim, q_stride, rot;
  int64_t off_q = 0, off_k = 0, copy_off[2] = {0, 0}, copy_size[2] = {0, 0};
  float eps;
  QkNormRopeState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    hq = k.GetAttribute<int64_t>("num_q_heads");
    hk = k.GetAttribute<int64_t>("num_k_heads");
    head_dim = k.GetAttribute<int64_t>("head_dim");
    q_stride = k.GetAttribute<int64_t>("q_stride");
    rot = k.GetAttribute<int64_t>("rotary_dim");
    eps = k.GetAttribute<float>("epsilon");
    if (kMerged) {
      off_q = k.GetAttribute<int64_t>("off_q");
      off_k = k.GetAttribute<int64_t>("off_k");
      for (int i = 0; i < 2; ++i) {
        copy_off[i] = k.GetAttribute<int64_t>(i == 0 ? "copy_off0" : "copy_off1");
        copy_size[i] = k.GetAttribute<int64_t>(i == 0 ? "copy_size0" : "copy_size1");
      }
    }
    if (head_dim != 256 || rot % 2 != 0 || rot > head_dim || q_stride < head_dim) {
      throw std::invalid_argument("QkNormRope: unsupported geometry (head_dim must be 256)");
    }
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto s0 = Shape(c.GetInput(0));
      const int64_t seq = s0.size() == 3 ? s0[1] : 1;
      if (seq < 1) throw std::invalid_argument("QkNormRope: rows per call must be >= 1");
      const int64_t out_batch = s0[0];
      const int64_t batch = s0[0] * seq;  // tokens: rows are independent work items
      const size_t w0 = kMerged ? 1 : 2;  // index of q_w
      if (Count(c.GetInput(w0)) != head_dim || Count(c.GetInput(w0 + 1)) != head_dim ||
          Count(c.GetInput(w0 + 2)) != batch * rot / 2 || Count(c.GetInput(w0 + 3)) != batch * rot / 2) {
        throw std::invalid_argument("QkNormRope: shape mismatch (decode S<=2 only)");
      }
      RopeParams p{};
      p.batch = static_cast<int>(batch); p.hq = static_cast<int>(hq); p.hk = static_cast<int>(hk);
      p.head_dim = static_cast<int>(head_dim); p.q_stride = static_cast<int>(q_stride);
      p.rot = static_cast<int>(rot); p.eps = eps;
      p.q_w = In<float>(c, w0); p.k_w = In<float>(c, w0 + 1);
      p.cos_c = In<bf16>(c, w0 + 2); p.sin_c = In<bf16>(c, w0 + 3);
      if (kMerged) {
        const int64_t n = Count(c.GetInput(0)) / batch;
        const bf16* m = In<bf16>(c, 0);
        if (off_q + hq * q_stride > n || off_k + hk * head_dim > n) {
          throw std::invalid_argument("QsaProjPrep: slice offsets exceed the merged projection width");
        }
        p.q_in = m + off_q; p.q_batch_stride = n;
        p.k_in = m + off_k; p.k_batch_stride = n;
        p.copies = 2;
        for (int i = 0; i < 2; ++i) {
          if (copy_off[i] + copy_size[i] > n) throw std::invalid_argument("QsaProjPrep: bad copy slice");
          auto o = c.GetOutput(2 + i, std::vector<int64_t>{out_batch, seq, copy_size[i]});
          p.copy[i] = CopySeg{m + copy_off[i], Out<bf16>(o), n, static_cast<int>(copy_size[i])};
        }
      } else {
        if (Count(c.GetInput(0)) != batch * hq * q_stride || Count(c.GetInput(1)) != batch * hk * head_dim) {
          throw std::invalid_argument("QkNormRope: shape mismatch");
        }
        p.q_in = In<bf16>(c, 0); p.q_batch_stride = hq * q_stride;
        p.k_in = In<bf16>(c, 1); p.k_batch_stride = hk * head_dim;
      }
      auto q = c.GetOutput(0, std::vector<int64_t>{out_batch, seq, hq * head_dim});
      auto k = c.GetOutput(1, std::vector<int64_t>{out_batch, seq, hk * head_dim});
      p.q_out = Out<bf16>(q); p.k_out = Out<bf16>(k);
      const size_t smem = 2 * sizeof(float) + head_dim * sizeof(bf16);
      const unsigned blocks = static_cast<unsigned>(batch * (hq + hk) + p.copies * batch);
      QkNormRopeKernel<<<blocks, kNormThreads, smem, Stream(api, raw)>>>(p);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct SigmoidGateMulState {
  const OrtApi& api;
  int64_t head_dim, heads, q_stride, tok_stride, q_offset;
  SigmoidGateMulState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    head_dim = k.GetAttribute<int64_t>("head_dim");
    heads = k.GetAttribute<int64_t>("num_heads");
    q_stride = k.GetAttribute<int64_t>("q_stride");
    tok_stride = k.GetAttribute<int64_t>("tok_stride");
    q_offset = k.GetAttribute<int64_t>("q_offset");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto attn = c.GetInput(0);
      const int64_t n = Count(attn);
      const int64_t tokens = n / (heads * head_dim);
      if (n % (heads * head_dim) != 0 ||
          Count(c.GetInput(1)) < (tokens - 1) * tok_stride + q_offset + heads * q_stride) {
        throw std::invalid_argument("SigmoidGateMul: shape mismatch");
      }
      auto out = c.GetOutput(0, Shape(attn));
      SigmoidGateMulKernel<<<(n + 255) / 256, 256, 0, Stream(api, raw)>>>(
          In<bf16>(c, 0), In<bf16>(c, 1), Out<bf16>(out), n, static_cast<int>(head_dim),
          static_cast<int>(heads), static_cast<int>(q_stride), tok_stride, q_offset);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

template <typename Derived, typename State>
struct AttnOp : Ort::CustomOpBase<Derived, State, true> {
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try {
      *kernel = new State(api, info);
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what());
    }
  }
};

constexpr auto kBF16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
constexpr auto kF32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

struct GdnConvPrepOp : AttnOp<GdnConvPrepOp, GdnPrepState<false>> {
  const char* GetName() const { return "GdnConvPrep"; }
  size_t GetInputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetInputType(size_t) const { return kBF16; }
  size_t GetOutputTypeCount() const { return 9; }  // [6..8] = optional conv state after rows 0, 1, 2
  ONNXTensorElementDataType GetOutputType(size_t i) const { return (i == 4 || i == 5) ? kF32 : kBF16; }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t i) const {
    return i >= 6 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
};
struct GdnProjPrepOp : AttnOp<GdnProjPrepOp, GdnPrepState<true>> {
  const char* GetName() const { return "GdnProjPrep"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t) const { return kBF16; }
  size_t GetOutputTypeCount() const { return 10; }  // [7..9] = optional conv state after rows 0, 1, 2
  ONNXTensorElementDataType GetOutputType(size_t i) const { return (i == 5 || i == 6) ? kF32 : kBF16; }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t i) const {
    return i >= 7 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
};
struct GdnRecurrentOp : AttnOp<GdnRecurrentOp, GdnRecurrentState> {
  const char* GetName() const { return "GdnRecurrent"; }
  size_t GetInputTypeCount() const { return 9; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    return i < 3 ? kBF16 : (i == 3 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 : kF32);
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t i) const {
    return i == 3 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
  size_t GetOutputTypeCount() const { return 5; }  // out, final_state, optional mid_state[0..2]
  ONNXTensorElementDataType GetOutputType(size_t i) const { return i == 0 ? kBF16 : kF32; }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t i) const {
    return i >= 2 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
};
struct PleMidStateOp : AttnOp<PleMidStateOp, PleMidStateState> {
  const char* GetName() const { return "PleMidState"; }
  size_t GetInputTypeCount() const { return 4; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i < 2 ? kBF16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; }
  size_t GetOutputTypeCount() const { return 6; }  // (mid_state[0..2], mid_ctx[0..2])
  ONNXTensorElementDataType GetOutputType(size_t i) const { return (i % 2 == 0) ? kBF16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t i) const {
    return i >= 2 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
};
struct QkNormRopeOp : AttnOp<QkNormRopeOp, QkNormRopeState<false>> {
  const char* GetName() const { return "QkNormRope"; }
  size_t GetInputTypeCount() const { return 6; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return (i == 2 || i == 3) ? kF32 : kBF16; }
  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return kBF16; }
};
struct QsaProjPrepOp : AttnOp<QsaProjPrepOp, QkNormRopeState<true>> {
  const char* GetName() const { return "QsaProjPrep"; }
  size_t GetInputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return (i == 1 || i == 2) ? kF32 : kBF16; }
  size_t GetOutputTypeCount() const { return 4; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return kBF16; }
};
struct SigmoidGateMulOp : AttnOp<SigmoidGateMulOp, SigmoidGateMulState> {
  const char* GetName() const { return "SigmoidGateMul"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const { return kBF16; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return kBF16; }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ORT API");
  Ort::InitApi(api);
  try {
    static GdnConvPrepOp gdn_conv_prep;
    static GdnProjPrepOp gdn_proj_prep;
    static QkNormRopeOp qk_norm_rope;
    static QsaProjPrepOp qsa_proj_prep;
    static SigmoidGateMulOp sigmoid_gate_mul;
    static GdnRecurrentOp gdn_recurrent;
    static PleMidStateOp ple_mid_state;
    static Ort::CustomOpDomain domain("qwen.fused_attn");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&gdn_conv_prep);
      domain.Add(&gdn_proj_prep);
      domain.Add(&qk_norm_rope);
      domain.Add(&qsa_proj_prep);
      domain.Add(&sigmoid_gate_mul);
      domain.Add(&gdn_recurrent);
      domain.Add(&ple_mid_state);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_FAIL, e.what());
  }
}
