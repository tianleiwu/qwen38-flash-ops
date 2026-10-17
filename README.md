# qwen38-flash-ops

CUDA custom-op libraries of the **Qwen3.8-Flash-Next** (NVFP4) decode stack for ONNX Runtime / onnxruntime-genai.
They use the plain ORT custom-op ABI (`RegisterCustomOps`, `Ort::CustomOpBase`, `onnxruntime_cxx_api.h`),
are compiled with `nvcc` for Hopper (`sm_90`) and are loaded next to the model package. One shared library per
domain; the file names, op names and op domains are the ones the existing model packages already reference, so a library
built here is a drop-in replacement of the shipped file.

The libraries are shipped next to the model package, not inside ORT (production custom-op sources do not belong in its tree) or
onnxruntime-genai (no CUDA op matrix in its CI). Ops that are generic are meant to be upstreamed to ORT as contrib ops instead.

| library (file name in the package `libs/`) | op domain | ops | source |
| --- | --- | --- | --- |
| `libqwen_fused_attn_r5.so` | `qwen.fused_attn` | `GdnConvPrep`, `GdnProjPrep`, `GdnRecurrent`, `PleMidState`, `QkNormRope`, `QsaProjPrep`, `SigmoidGateMul` | `src/fused_attn` |
| `libqwen_fused_hc_r5.so` | `qwen.fused_hc` | `HcBlock`, `HcPostBlock`, `HcDown`, `HcPostDown`, `HcUpPre`, `HcMixDown`, `HcMixPostDown`, `HcMixUp` | `src/fused_hc` |
| `libqwen_fused_gemm_r5.so` | `qwen.fused_gemm` | `SkinnyGemm`, `SharedExpertDown`, `ArgmaxRows`, `DequantWeightBf16`, `UntileHcBf16` | `src/fused_gemm` |
| `libqwen_fused_ops.so` | `qwen.fused` | `DivSigmoidMul`, `SwishMul`, `SharedGateMulAdd` | `src/fused_ops` |
| `libqwen_fold.so` | `qwen.fold` | `FoldMid`, `FoldMid2`, `FoldTail`, `EmbedNorm`, `RowSelectBf16x3` | `src/fold` |
| `libqwen_mtp_head.so` | `qwen.mtp` | `MtpMoe`, `Bf16ArgMax` | `src/mtp_head` |
| `libqwen_guarded_dsa_r2_v12.so` (fork, optional) | `qwen.static` | `GuardedDSA` (BF16, INT8 and FP8 per-channel KV cache) | `forks/guarded_dsa_r2` |
| `libqwen_packed_qsa_indexer_v11.so` (fork, optional) | `qwen.capture` | `PackedQSAIndexer` | `forks/packed_qsa_v11` |

All ops run on the CUDA execution provider (`GetExecutionProviderType() == "CUDAExecutionProvider"`) and use the
provider's compute stream, so they are CUDA-graph capturable. All tensors are BF16 unless stated.
Every op reproduces the BF16/FP32 rounding points of the ONNX chain it replaces (`docs`: the header comment of each source).

## Ops

### `qwen.fused_attn` (`src/fused_attn/fused_attn_r5.cu`)
Attention-side fusions of the S=1 decode graph; R in {1,2} rows per call (speculative verify) where noted.

| op | what it does | inputs -> outputs |
| --- | --- | --- |
| `GdnConvPrep` | Transpose + Concat/Slice(state) + `CausalConvWithState`(K=4, SiLU) + Transpose + Split(q,k,v) + Cast(b), Cast(a) of one GatedDeltaNet layer; R-row chain over the conv window | qkv `[B,R,C]`, state `[B,C,4]`, weight `[C,1,4]`, b, a `[B,R,Hv]` -> q, k, v, state_out, b32, a32 (fp32), optional mid state (after row 0) |
| `GdnProjPrep` | same, reading the merged projection output (qkv / z / b / a slices, attributes `off_*`) and also emitting z | merged `[B,R,N]`, state, weight -> q, k, v, z, state_out, b32, a32, optional mid state |
| `GdnRecurrent` | `com.microsoft::GatedDeltaNet` of this model (gated_delta, qwen gate, sigmoid beta, qk L2 norm, fp32 state `[B,Hv,V,K]`, K=V=128) for R in {1,2} rows; verbatim arithmetic of ORT's `GatedDeltaNetDecodeWarpKernel<bf16,128,8>` | 9 inputs: q, k `[B,R,Hq,128]`, v `[B,R,Hv,128]`, 3 more, state `[B,Hv,128,128]`, 2 per-head vectors `[Hv]` -> out, final_state, optional mid_state |
| `PleMidState` | PLE conv state / previous-token context after row 0 of an R-row call | state `[B,C,W]`, x `[B,R,C]`, ctx `[B,L]` (int64), ids `[B,R]` (int64) -> mid state, mid ctx |
| `QkNormRope` | Cast + RMSNorm chain + partial non-interleaved RoPE of the q and k heads of a QSA layer (attributes `num_q_heads`, `num_k_heads`, `head_dim`, `q_stride`, `rotary_dim`, `epsilon`) | q_in, k_in, cos, sin, q_w, k_w (fp32) -> q_out, k_out |
| `QsaProjPrep` | same, from the merged projection output, plus two copied slices (indexer qk, v) | merged, cos, sin, q_w, k_w -> q_out, k_out, 2 slices |
| `SigmoidGateMul` | `y = bf16(attn * bf16(sigmoid(gate)))`, gate read from the q projection (attributes `head_dim`, `num_heads`, `q_stride`, `tok_stride`, `q_offset`) | attn, q_in -> y |

### `qwen.fused_hc` (`src/fused_hc/fused_hc_r5.cu`)
Fused hyper-connection (HC) blocks for R in {1,2} rows. `HcBlock`/`HcPostBlock` are ONE kernel per block for pre-tiled weights
(all weights of both phases are bulk-copied to shared memory up front; the phases hand over through a data-carrying
value/launch-tag table); other layouts use the two-kernel `HcDown`/`HcUpPre` path.

| op | inputs -> outputs |
| --- | --- |
| `HcBlock` | streams, norm_w, w_down, w_inject, w_up -> pre, gate, scratch (fp32). Supports BF16 and per-column FP8 E4M3FN (`wd_scale`, `wu_scale` attributes) `[160, 20480]` tiled `w_down` / `w_up` |
| `HcPostBlock` | streams, branch, gate, norm_w, w_down, w_inject, w_up -> streams_out, pre, gate, scratch. Supports BF16 and per-column FP8 E4M3FN (`wd_scale`, `wu_scale` attributes) `[160, 20480]` tiled `w_down` / `w_up` |
| `HcDown` / `HcPostDown` | down GEMV (K-split partials to a scratch row) of a block / of the post-mix + next block -> scratch (and streams_out) |
| `HcUpPre` | up phase: scratch + inputs -> pre, gate |
| `HcMixDown`, `HcMixPostDown`, `HcMixUp` | the final hyper_connection_mixer: same kernels without the inject/gate branch |

Debug environment variables read by the library: `HC_PDL`, `HC_MERGE`, `HC_EARLYW`, `HC_ROWS_PER_CTA`, `HC_TRIGGER`, `HC_SCRATCH_FINAL`.

### `qwen.fused_gemm` (`src/fused_gemm/`)
| op | what it does |
| --- | --- |
| `SkinnyGemm` | Decode GEMV/GEMM, M <= 4 rows; `x [...,K] @ W_i [K,N_i]` for g weights, writing each split slice directly to its own output (attributes `seg_outs`, `widths`, optional `trigger` for programmatic dependent launch, optional per-column `w_scale`). Supports BF16 `[K,N]`, FP8 E4M3FN `[K,N]`, and block-16 NVFP4 `UINT8 [K, N*9/16]` weights. FP32 accumulation, fixed reduction order; row r of an M-row call is bit-identical to the M=1 result |
| `SharedExpertDown` | `y = routed + bf16(bf16(swish_mul(g,u) @ W) * bf16(sigmoid(bf16(dot(gx, gw)))))` per row, M in [1,4] (6 inputs -> 1 output); supports BF16, FP8 E4M3FN, and NVFP4 `W` (optional `w_scale` attribute) |
| `ArgmaxRows` | `idx[...] = argmax over the last axis` of BF16/FP16/FP32 logits (numpy semantics: lowest index on ties, first NaN wins); optional 2nd output: the maximum as FP32 |
| `DequantWeightBf16` | Dequantizes an FP8 E4M3FN `[K,N]` or block-16 NVFP4 `UINT8 [K, N*9/16]` weight tensor (with `n_cols`, optional `col_offset`, optional `widths`, and `w_scale` attributes, plus optional second input `x` to tie buffer lifetime to the consuming `MatMul`) into `[K,N]` BF16 |
| `UntileHcBf16` | Untiles (and optionally dequantizes from FP8 E4M3FN with `w_scale`) `[160, 20480]` pre-tiled hyper-connection weights (`hc_tiled.data`) into `[10240, 320]` (`kind=0`) or `[320, 10240]` (`kind=1`) BF16 so the dense prefill and head-KV graphs share `hc_tiled.data` with the decode graph |

### `qwen.fused` (`src/fused_ops/fused_ops.cu`)
Pointwise BF16 fusions: `DivSigmoidMul` (`y = bf16(bf16(sigmoid(bf16(x / div))) * mul)`), `SwishMul` (`y = bf16(bf16(silu(g)) * u)`),
`SharedGateMulAdd` (shared-expert gate + add, 4 inputs).

### `qwen.fold` (`src/fold/qwen_fold_ops.cu`)
Glue ops of the folded MTP(1) cycle graph (verify S=2 + draft head R=2 in one ORT graph), all bookkeeping on the device:

* `FoldMid(next_tokens i64[1,2], input_ids i64[1,2], status i32[N], ctl i64[8], ctx_commit, ctx_a, ctx_b i64[2], ptrs i64[2]) -> head_ids, head_pos, head_c, sel`: accept test + commit bookkeeping and the head row inputs. `FoldMid2` additionally takes the RoPE caches and emits the position-derived head inputs.
* `EmbedNorm(ids i64[1,R], table bf16[V,N], w f32[N]) -> bf16[1,R,N]` (attribute `epsilon`): Gather + RMSNorm of the head in one kernel.
* `RowSelectBf16x3(sel i64[1], x0, x1, x2 bf16[1,2,...]) -> y0, y1, y2 bf16[1,1,...]`.
* `FoldTail(next_token, status i32[2], ctl i64[8]) -> done i32[1]`: next draft into `ctl[2]`, error bits.

These encode the control-block ABI of the GenAI `qwen38_flash_graph` engine (`ctl`, pinned result ring) and are not general purpose.

### `qwen.mtp` (`src/mtp_head/qwen_mtp_ops.cu`)
* `Bf16ArgMax(logits bf16[..., V]) -> idx int64[..., 1], scratch f32[rows,124,2]`: greedy draft token on the device (lowest index among maxima); specialised for V = 248320.
* `MtpMoe(premix bf16[1,R,2560], router_logits bf16[1,R,512], gate_up, down) -> routed bf16[1,R,2560], scratch f32`: softmax, top-10, normalised weights, gate_up GEMV + SwishMul, down GEMV and the weighted sum of the MTP head MoE in one op (replaces the expert-weight Gather of the exported graph). Supports BF16 (`[512,1280,2560]` / `[512,2560,640]`), per-row FP8 E4M3FN (`UINT8 [512,1280,2576]` / `[512,2560,656]`), and block-16 NVFP4 (`UINT8 [512,1280,1440]` / `[512,2560,360]`, attributes `gate_up_global_scale` and `down_global_scale`).

### Forks of ORT kernels (`forks/`, optional)
`GuardedDSA` (guarded dynamic sparse attention with the 2-row chain and BF16 / per-channel INT8 / per-channel FP8 E4M3FN KV cache support via `k_scale` and `v_scale` attributes, domain `qwen.static`) and `PackedQSAIndexer` (packed QSA
indexer, domain `qwen.capture`) are forks of ORT CUDA kernels and need ORT private headers: see `forks/README.md`.

## Build

Requirements: CUDA toolkit 13.0 (what the shipped libraries use; 12.x should work for sm_90 but is untested), CMake >= 3.22,
a C++17 host compiler, and the **ORT public headers** (`onnxruntime_c_api.h`, `onnxruntime_cxx_api.h`, ...). Nothing is vendored. Get them from

* a release tarball: `onnxruntime-linux-x64-gpu-<ver>.tgz` from <https://github.com/microsoft/onnxruntime/releases> -> `<tarball>/include` (the pip wheels do not ship headers), or
* an ORT source checkout: `<ORT>/include/onnxruntime/core/session`.

Build against headers of an ORT version that is not newer than the runtime that will load the libraries: the libraries
request `ORT_API_VERSION` of the headers they were built with (`GetApi(ORT_API_VERSION)`). The libraries in the Engine v12b
package were built with the headers of the ORT 1.31.0-dev source (API 31); the runtime of that package is ORT 1.31.0. A build of the six public-API libraries
against the 1.24.1 release headers (API 24) also compiles.

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
cmake -S . -B build -DORT_INCLUDE_DIR=/path/to/onnxruntime/include      # or env ORT_INCLUDE_DIR / ORT_HOME
cmake --build build -j
ctest --test-dir build --output-on-failure                              # dlopen + RegisterCustomOps smoke
# optional, needs one GPU and python with onnxruntime-gpu, onnx, torch:
cmake -S . -B build -DQWEN38_PYTHON=$(which python) && CUDA_VISIBLE_DEVICES=0 ctest --test-dir build -R python_ops
```

Libraries are written to `build/lib/lib*.so`. Options: `-DQWEN38_CUDA_ARCH=90` (default; `fused_attn` is built for `sm_90a`
exactly as shipped), `-DQWEN38_BUILD_ORT_FORKS=ON` (see `forks/README.md`), `-DQWEN38_BUILD_TESTS=OFF`.
The compile flags are the ones of the original builds: `nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -arch=sm_90`.

`tools/make_manifest.py` writes `manifest.json` (SHA256 of each built library and of its sources);
`tools/compare_libs.py NEW_DIR REF_DIR` compares a build with another set of libraries (dynamic symbols, SASS, PTX).
A rebuild is not byte-identical to a shipped file (the compiler embeds path-derived names), but its symbols, SASS and PTX are.

## Using the libraries with GenAI

GenAI loads custom-op libraries from `model.decoder.session_options.custom_ops_libraries` of `genai_config.json`, as paths
**relative to the model folder** (an allow-list: only files inside the model folder are loaded), and registers them on the
session of that model. Example (the Engine v12b package, `libs/` is next to `genai_config.json`):

```json
"model": { "decoder": { "session_options": {
  "custom_ops_libraries": [
    "libs/libqwen_packed_qsa_indexer_v11.so", "libs/libqwen_fused_ops.so", "libs/libqwen_fused_attn_r5.so",
    "libs/libqwen_fused_hc_r5.so", "libs/libqwen_fused_gemm_r5.so", "libs/libqwen_guarded_dsa_r2_v12.so",
    "libs/libqwen_mtp_head.so", "libs/libqwen_fold.so" ] } } }
```

Copy the built `lib*.so` files into the package's `libs/` folder. With plain ORT use `SessionOptions.register_custom_ops_library(path)`.

**One process per GPU.** `HcBlock`/`HcPostBlock` use a software grid barrier and a spin-wait handoff between CTAs, which is only
safe when all 168 CTAs of the kernel are co-resident; the host checks that against the device's occupancy but cannot see other
processes. Never run a second process (or an MPS client) on the GPU of a process that has `libqwen_fused_hc_r5.so` loaded. (The
whole stack also needs ~113 GiB of a 141 GB H200 for the dense prefill warm-up.)

## Tests

`tests/test_load.py` registers every built library in an `InferenceSession` and runs one-node models on the CUDA EP against a
torch reference: `SigmoidGateMul`, `SwishMul`, `ArgmaxRows` (BF16/FP32, tie-break), `Bf16ArgMax`.
The model-level gates (token identity of a package built with these libraries) live in the GenAI repository tools
(`tools/qwen38_graph/engine_gate.py`); `VERIFICATION.md` summarises the run that checked this repository against the shipped libraries.

## Provenance

None of the source directories was under version control; the copies are verbatim (`sha256` of every source file is recorded in
`manifest.json`), except that the `#include "../sigmoid_acc.cuh"` of `fused_attn_r5.cu`, `fused_hc_r5.cu` and `skinny_gemm.cuh` became
`#include "common/sigmoid_acc.cuh"`.

| source here | original research run directory (`eval_runs/<dir>`, not in version control) |
| --- | --- |
| `src/fused_attn/fused_attn_r5.cu`, `src/fused_hc/fused_hc_r5.cu`, `src/fused_gemm/*`, `src/common/sigmoid_acc.cuh` | `qwen38_v12b_20261012_01/src/{attn_r5,hc_r5,gemm_r5}/`, `.../src/sigmoid_acc.cuh` (build: `qwen38_v12b_20261012_01/build_libs.sh`) |
| `src/fused_ops/fused_ops.cu` | `qwen38_cuda_graph_20261010_01/fused_ops/src/fused_ops.cu` (shipped library of Oct 11; the `*_r5` accurate-sigmoid variant `qwen38_v12b_20261012_01/src/fused_ops_r5` is not used by the v12 packages and not included) |
| `src/fold/qwen_fold_ops.cu` | `qwen38_cuda_graph_20261010_01/mtp_runner/fold/src/qwen_fold_ops.cu` (`build_fold_lib.sh`) |
| `src/mtp_head/qwen_mtp_ops.cu` | `qwen38_cuda_graph_20261010_01/mtp_head/src/qwen_mtp_ops.cu` (`build.sh`) |
| `forks/guarded_dsa_r2/` | `qwen38_cuda_graph_20261010_01/v12/dsa/dsa_src/` (`build_dsa.sh`; = `fused_attn/r2/dsa_src` + the `ZeroFlagsKernel` patch instead of a memset node) |
| `forks/packed_qsa_v11/` | `qwen38_cuda_graph_20261010_01/fused_attn/r2/packed_qsa_src_v11/` + `fused_attn/r2/src/{packed_qsa_op_v11.cu,packed_qsa.exports}` (`build_indexer_v11.sh`) |

The sigmoid used by the `*_r5` libraries is bit-identical to ORT's CUDA EP `OP_Sigmoid<BFloat16>` after the accurate-sigmoid fix
(ORT branch `qwen38-prefill-fix`, commit `453ac97a68`); the libraries have to be used with a provider that contains that fix.

## License

MIT, as ORT and onnxruntime-genai (`LICENSE`). The research sources carry no per-file headers; they are kept verbatim so that their
hashes stay comparable with the shipped libraries. The ORT-derived files under `forks/` keep Microsoft's MIT header.
