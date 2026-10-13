#!/usr/bin/env python3
"""Load every built library through the ORT custom-op API and check a few simple ops against a torch reference.

usage: CUDA_VISIBLE_DEVICES=0 python tests/test_load.py [--lib-dir build/lib]

Requires onnxruntime-gpu (CUDA EP), onnx and torch (CUDA). Libraries that are not present in --lib-dir are skipped
(the ORT forks are only built with -DQWEN38_BUILD_ORT_FORKS=ON).
"""
import argparse
import ctypes
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

LIBS = {
    "qwen_fused_ops": "qwen.fused",
    "qwen_fused_attn_r5": "qwen.fused_attn",
    "qwen_fused_hc_r5": "qwen.fused_hc",
    "qwen_fused_gemm_r5": "qwen.fused_gemm",
    "qwen_fold": "qwen.fold",
    "qwen_mtp_head": "qwen.mtp",
    "qwen_guarded_dsa_r2_v12": "qwen.static",
    "qwen_packed_qsa_indexer_v11": "qwen.capture",
}
BF, F32, I64 = TensorProto.BFLOAT16, TensorProto.FLOAT, TensorProto.INT64
ORT_ELEM = {BF: 16, F32: 1, I64: 7}  # ONNX element type ids used by IOBinding


def lib_path(lib_dir, name):
    return lib_dir / f"lib{name}.so"


def make_session(lib, domain, op, inputs, outputs, **attrs):
    """One-node model `op` of `domain`; inputs/outputs are (name, onnx type, shape) lists."""
    node = helper.make_node(op, [n for n, *_ in inputs], [n for n, *_ in outputs], domain=domain, **attrs)
    graph = helper.make_graph([node], "g", [helper.make_tensor_value_info(n, t, sh) for n, t, sh in inputs],
                              [helper.make_tensor_value_info(n, t, sh) for n, t, sh in outputs])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(domain, 1)])
    model.ir_version = 10
    so = ort.SessionOptions()
    so.register_custom_ops_library(str(lib))
    return ort.InferenceSession(model.SerializeToString(), so, providers=[("CUDAExecutionProvider", {"device_id": 0})])


def run(sess, feeds, out_specs):
    """feeds: name -> cuda torch tensor; out_specs: name -> (dtype, shape) of the outputs. Returns torch tensors."""
    binding = sess.io_binding()
    for name, tensor in feeds.items():
        elem = {torch.bfloat16: BF, torch.float32: F32, torch.int64: I64}[tensor.dtype]
        binding.bind_input(name, "cuda", 0, ORT_ELEM[elem], list(tensor.shape), tensor.data_ptr())
    outs = {}
    for name, (dtype, shape) in out_specs.items():
        outs[name] = torch.empty(shape, dtype=dtype, device="cuda")
        elem = {torch.bfloat16: BF, torch.float32: F32, torch.int64: I64}[dtype]
        binding.bind_output(name, "cuda", 0, ORT_ELEM[elem], list(shape), outs[name].data_ptr())
    sess.run_with_iobinding(binding)
    torch.cuda.synchronize()
    return outs


def bf16_ulp_close(actual, expected):
    """True when every element differs by at most one bf16 ulp (relative 2^-7)."""
    a, e = actual.float(), expected.float()
    return bool(torch.all((a - e).abs() <= 2.0**-7 * e.abs() + 1e-30))


def test_sigmoid_gate_mul(lib):
    tokens, heads, head_dim = 3, 4, 64
    q_stride, tok_stride, q_offset = 2 * head_dim, 2 * head_dim * heads + 16, 8
    gen = torch.Generator(device="cuda").manual_seed(1)
    attn = torch.randn(tokens, heads * head_dim, generator=gen, device="cuda").bfloat16()
    q_in = (torch.randn(tokens * tok_stride, generator=gen, device="cuda") * 3).bfloat16()
    sess = make_session(lib, "qwen.fused_attn", "SigmoidGateMul",
                        [("attn", BF, [tokens, heads * head_dim]), ("q_in", BF, [tokens * tok_stride])], [("y", BF, None)],
                        head_dim=head_dim, num_heads=heads, q_stride=q_stride, tok_stride=tok_stride, q_offset=q_offset)
    out = run(sess, {"attn": attn, "q_in": q_in}, {"y": (torch.bfloat16, [tokens, heads * head_dim])})["y"]
    gate = torch.empty(tokens, heads, head_dim, device="cuda")
    for t in range(tokens):
        for h in range(heads):
            start = t * tok_stride + q_offset + h * q_stride + head_dim
            gate[t, h] = q_in[start:start + head_dim].float()
    ref = (attn.float().view(tokens, heads, head_dim) * torch.sigmoid(gate).bfloat16().float()).bfloat16().view(tokens, -1)
    assert bf16_ulp_close(out, ref), "SigmoidGateMul differs from the reference"


def test_swish_mul(lib):
    gen = torch.Generator(device="cuda").manual_seed(2)
    g = (torch.randn(1, 2, 512, generator=gen, device="cuda") * 3).bfloat16()
    u = torch.randn(1, 2, 512, generator=gen, device="cuda").bfloat16()
    sess = make_session(lib, "qwen.fused", "SwishMul", [("g", BF, [1, 2, 512]), ("u", BF, [1, 2, 512])], [("y", BF, None)])
    out = run(sess, {"g": g, "u": u}, {"y": (torch.bfloat16, [1, 2, 512])})["y"]
    silu = (g.float() * torch.sigmoid(g.float())).bfloat16().float()
    ref = (silu * u.float()).bfloat16()
    assert bf16_ulp_close(out, ref), "SwishMul differs from the reference"


def test_argmax_rows(lib):
    rows, vocab = 5, 4099
    gen = torch.Generator(device="cuda").manual_seed(3)
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(rows, vocab, generator=gen, device="cuda").to(dtype)
        x[1, 17] = x[1, 3000] = x.max() + 1  # tie in row 1: the lowest index wins
        elem = BF if dtype == torch.bfloat16 else F32
        sess = make_session(lib, "qwen.fused_gemm", "ArgmaxRows", [("x", elem, [rows, vocab])], [("idx", I64, None), ("val", F32, None)])
        outs = run(sess, {"x": x}, {"idx": (torch.int64, [rows]), "val": (torch.float32, [rows])})
        assert outs["idx"][1].item() == 17, "ArgmaxRows tie-break"
        mx = x.float().max(dim=1)
        keep = torch.tensor([r != 1 for r in range(rows)], device="cuda")
        assert torch.equal(outs["idx"][keep], mx.indices[keep]), "ArgmaxRows indices"
        assert torch.equal(outs["val"], mx.values), "ArgmaxRows values"


def test_bf16_argmax(lib):
    rows, vocab, blocks = 2, 248320, 124  # the op is specialised for the Qwen3.8 vocabulary
    gen = torch.Generator(device="cuda").manual_seed(4)
    x = torch.randn(rows, vocab, generator=gen, device="cuda").bfloat16()
    x[0, 247999] = 100.0
    x[1, 5] = 100.0
    sess = make_session(lib, "qwen.mtp", "Bf16ArgMax", [("x", BF, [rows, vocab])], [("idx", I64, None), ("scratch", F32, None)])
    outs = run(sess, {"x": x}, {"idx": (torch.int64, [rows, 1]), "scratch": (torch.float32, [rows, blocks, 2])})
    assert outs["idx"].flatten().tolist() == [247999, 5], outs["idx"].flatten().tolist()
    x2 = torch.randn(rows, vocab, generator=gen, device="cuda").bfloat16()
    outs = run(sess, {"x": x2}, {"idx": (torch.int64, [rows, 1]), "scratch": (torch.float32, [rows, blocks, 2])})
    assert torch.equal(outs["idx"].flatten(), x2.float().argmax(dim=1)), "Bf16ArgMax random rows"


OP_TESTS = [
    ("qwen_fused_attn_r5", test_sigmoid_gate_mul),
    ("qwen_fused_ops", test_swish_mul),
    ("qwen_fused_gemm_r5", test_argmax_rows),
    ("qwen_mtp_head", test_bf16_argmax),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib-dir", default="build/lib")
    args = ap.parse_args()
    lib_dir = pathlib.Path(args.lib_dir).resolve()
    failures = 0

    present = [n for n in LIBS if lib_path(lib_dir, n).exists()]
    if not present:
        print(f"no libraries in {lib_dir}")
        return 2
    for name in present:
        path = lib_path(lib_dir, name)
        try:
            ctypes.CDLL(str(path)).RegisterCustomOps  # symbol present
            so = ort.SessionOptions()
            so.register_custom_ops_library(str(path))  # ORT accepts the library
            print(f"PASS register {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL register {name}: {exc}")

    for name, test in OP_TESTS:
        if name not in present:
            print(f"SKIP {test.__name__} ({name} not built)")
            continue
        try:
            test(lib_path(lib_dir, name))
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print("ALL PASSED" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
