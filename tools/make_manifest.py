#!/usr/bin/env python3
"""Write manifest.json: SHA256 of every built library and of the source files it is built from.

usage: tools/make_manifest.py --lib-dir build/lib [--lib-dir build_forks/lib] [--ref-dir <package>/libs] [--out manifest.json]

--ref-dir lists the SHA256 of same-named libraries of a shipped model package for comparison (informational: a rebuild
is not byte-identical, because the compiler embeds path-derived names; tools/compare_libs.py compares code and symbols).
Libraries that are not found in any --lib-dir are listed with "built": false.
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent

# library -> (op domain, ops, sources relative to the repo root)
LIBRARIES = {
    "libqwen_fused_ops.so": ("qwen.fused", ["DivSigmoidMul", "SwishMul", "SharedGateMulAdd"], ["src/fused_ops/fused_ops.cu"]),
    "libqwen_fused_attn_r5.so": (
        "qwen.fused_attn",
        ["GdnConvPrep", "GdnProjPrep", "GdnRecurrent", "PleMidState", "QkNormRope", "QsaProjPrep", "SigmoidGateMul"],
        ["src/fused_attn/fused_attn_r5.cu", "src/common/sigmoid_acc.cuh"],
    ),
    "libqwen_fused_hc_r5.so": (
        "qwen.fused_hc",
        ["HcDown", "HcPostDown", "HcBlock", "HcPostBlock", "HcUpPre", "HcMixDown", "HcMixPostDown", "HcMixUp"],
        ["src/fused_hc/fused_hc_r5.cu", "src/common/sigmoid_acc.cuh"],
    ),
    "libqwen_fused_gemm_r5.so": (
        "qwen.fused_gemm",
        ["SkinnyGemm", "SharedExpertDown", "ArgmaxRows", "DequantWeightBf16", "UntileHcBf16"],
        ["src/fused_gemm/fused_gemm.cu", "src/fused_gemm/skinny_gemm.cuh", "src/fused_gemm/argmax_rows.cuh", "src/common/sigmoid_acc.cuh"],
    ),
    "libqwen_fold.so": ("qwen.fold", ["FoldMid", "FoldMid2", "FoldTail", "EmbedNorm", "RowSelectBf16x3"], ["src/fold/qwen_fold_ops.cu"]),
    "libqwen_mtp_head.so": ("qwen.mtp", ["Bf16ArgMax", "MtpMoe"], ["src/mtp_head/qwen_mtp_ops.cu"]),
    "libqwen_guarded_dsa_r2_v12.so": (
        "qwen.static",
        ["GuardedDSA"],
        [f"forks/guarded_dsa_r2/{n}" for n in ("capture_ops_r2.cu", "guarded_dsa_r2.cu", "private_dsa.h", "cuda_runtime_status.cc", "guarded.exports")],
    ),
    "libqwen_packed_qsa_indexer_v11.so": (
        "qwen.capture",
        ["PackedQSAIndexer"],
        [f"forks/packed_qsa_v11/{n}" for n in (
            "packed_qsa_op_v11.cu", "packed_sparse_attention_indexer_impl.cu", "packed_sparse_attention_indexer_impl.h",
            "packed_sparse_attention_indexer_common.h", "sparse_attention_indexer_common.h",
            "sparse_attention_indexer_device_math.cuh", "packed_qsa.exports")] + ["forks/guarded_dsa_r2/cuda_runtime_status.cc"],
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_version(cmd, key):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()
        return next(line.strip() for line in out if key in line)
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib-dir", action="append", required=True, type=pathlib.Path)
    ap.add_argument("--ref-dir", type=pathlib.Path)
    ap.add_argument("--nvcc", default="nvcc")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "manifest.json")
    args = ap.parse_args()

    libs = {}
    for name, (domain, ops, sources) in LIBRARIES.items():
        built = next((d / name for d in args.lib_dir if (d / name).exists()), None)
        entry = {"domain": domain, "ops": ops, "built": built is not None}
        if built:
            entry["sha256"] = sha256(built)
            entry["size"] = built.stat().st_size
        if args.ref_dir and (args.ref_dir / name).exists():
            entry["shipped_reference_sha256"] = sha256(args.ref_dir / name)
        entry["sources"] = {}
        for src in sources:
            entry["sources"][src] = sha256(ROOT / src)
        libs[name] = entry

    tree = hashlib.sha256()
    for src in sorted({s for e in libs.values() for s in e["sources"]}):
        tree.update(src.encode() + b"\0" + bytes.fromhex(sha256(ROOT / src)))
    manifest = {
        "schema": 1,
        "source_tree_sha256": tree.hexdigest(),
        "toolchain": {"nvcc": tool_version([args.nvcc, "--version"], "release"), "cxx": tool_version(["c++", "--version"], "c++")},
        "libraries": libs,
    }
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out}: {sum(e['built'] for e in libs.values())}/{len(libs)} libraries built")


if __name__ == "__main__":
    main()
