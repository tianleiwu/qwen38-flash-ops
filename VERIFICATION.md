# Verification of this repository against the shipped libraries (Oct 13 2026)

Toolchain: CUDA 13.0.48 (`nvcc`), GCC 13.2, CMake 3.x, H200 (sm_90), ORT headers of the 1.31.0-dev source (API 31), runtime ORT 1.31.0.
Reference: the nine libraries of the Engine v12b model package (`libs/`).

1. **Clean build of all 8 libraries from this repository** (`-DQWEN38_BUILD_ORT_FORKS=ON`, forks against a `git archive` of ORT
   `504238c6` + its Release build tree): no errors. The six public-API libraries also build against the headers of the
   `onnxruntime-linux-x64-gpu-1.24.1` release (API 24).
2. **Equivalence with the shipped libraries** (`tools/compare_libs.py`): for all 8 libraries the dynamic symbol tables, the SASS of
   all kernels and the PTX are identical. Files are not byte-identical (path-derived names of anonymous-namespace symbols; the
   shipped files additionally contain one empty cubin, 56 bytes; `libqwen_fused_ops.so`: 4152 bytes of fatbin metadata). The two forks have
   exactly the shipped size.
3. **ctest** (9 tests: 8 `dlopen`/`RegisterCustomOps` + `python_ops`): 100 % passed, including `SigmoidGateMul`, `SwishMul`,
   `ArgmaxRows`, `Bf16ArgMax` on the CUDA EP against a torch reference.
4. **Package gate with the rebuilt libraries** (copy of the Engine v12b package, only `libs/*.so` replaced by libraries built from this
   repository; one GPU, one process): GenAI Engine `engine_gate.py --prefill steps` (bit-exact prompt path, greedy, MTP(1) folded decode)
   on the 6 gate prompts: **GATE PASS**, 6/6 token streams identical to the Python v12b runner, cycle and accepted counts equal, and
   all token hashes, counts and cycle statistics equal to the run with the original libraries. A 128-token decode smoke
   (fast prefill) generates the expected answer at 251 tokens/s (original package: 234-251 tokens/s).

`manifest.json` holds the SHA256 of every library of this build (`sha256`), of the shipped libraries (`shipped_reference_sha256`) and of all sources.
