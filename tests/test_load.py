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
BF, F32, I32, I64, U8, I8, FP8 = (
    TensorProto.BFLOAT16,
    TensorProto.FLOAT,
    TensorProto.INT32,
    TensorProto.INT64,
    TensorProto.UINT8,
    TensorProto.INT8,
    TensorProto.FLOAT8E4M3FN,
)
ORT_ELEM = {BF: 16, F32: 1, I32: 6, I64: 7, U8: 2, I8: 3, FP8: 17}  # ONNX element type ids used by IOBinding


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
    dtype_map = {
        torch.bfloat16: BF,
        torch.float32: F32,
        torch.int32: I32,
        torch.int64: I64,
        torch.uint8: U8,
        torch.int8: I8,
        torch.float8_e4m3fn: FP8,
    }
    binding = sess.io_binding()
    for name, tensor in feeds.items():
        elem = dtype_map[tensor.dtype]
        binding.bind_input(name, "cuda", 0, ORT_ELEM[elem], list(tensor.shape), tensor.data_ptr())
    outs = {}
    for name, (dtype, shape) in out_specs.items():
        outs[name] = torch.empty(shape, dtype=dtype, device="cuda")
        elem = dtype_map[dtype]
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


def test_mtp_moe_modes(lib):
    gen = torch.Generator(device="cuda").manual_seed(6)
    premix = (torch.randn(1, 2, 2560, generator=gen, device="cuda") * 0.1).bfloat16()
    router = torch.randn(1, 2, 512, generator=gen, device="cuda").bfloat16()
    gu_bf16 = (torch.randn(512, 1280, 2560, generator=gen, device="cuda") * 0.02).bfloat16()
    dn_bf16 = (torch.randn(512, 2560, 640, generator=gen, device="cuda") * 0.02).bfloat16()

    sess_bf = make_session(lib, "qwen.mtp", "MtpMoe",
                           [("premix", BF, [1, 2, 2560]), ("router", BF, [1, 2, 512]),
                            ("gu", BF, [512, 1280, 2560]), ("dn", BF, [512, 2560, 640])],
                           [("out", BF, [1, 2, 2560]), ("scratch", F32, [2, 6432])])
    out_bf = run(sess_bf, {"premix": premix, "router": router, "gu": gu_bf16, "dn": dn_bf16},
                 {"out": (torch.bfloat16, [1, 2, 2560]), "scratch": (torch.float32, [2, 6432])})["out"]

    def pack_fp8_rows(w, K):
        wf = w.float()
        s = (wf.abs().amax(dim=-1) / 448.0).clamp(min=1e-8)
        q = (wf / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        buf = torch.zeros((w.shape[0], w.shape[1], K + 16), dtype=torch.uint8, device="cuda")
        buf[:, :, :K] = q.view(torch.uint8)
        buf[:, :, K:K + 4] = s.contiguous().view(torch.uint8).view(w.shape[0], w.shape[1], 4)
        return buf

    gu_fp8 = pack_fp8_rows(gu_bf16, 2560)
    dn_fp8 = pack_fp8_rows(dn_bf16, 640)
    sess_fp8 = make_session(lib, "qwen.mtp", "MtpMoe",
                            [("premix", BF, [1, 2, 2560]), ("router", BF, [1, 2, 512]),
                             ("gu", U8, [512, 1280, 2576]), ("dn", U8, [512, 2560, 656])],
                            [("out", BF, [1, 2, 2560]), ("scratch", F32, [2, 6432])])
    out_fp8 = run(sess_fp8, {"premix": premix, "router": router, "gu": gu_fp8, "dn": dn_fp8},
                  {"out": (torch.bfloat16, [1, 2, 2560]), "scratch": (torch.float32, [2, 6432])})["out"]
    rel_bf = ((out_fp8.float() - out_bf.float()).norm() / out_bf.float().norm()).item()
    assert rel_bf < 0.05, f"MtpMoe FP8 error too high: vs_bf={rel_bf}"


def test_guarded_dsa_kv_modes(lib):
    gen = torch.Generator(device="cuda").manual_seed(7)
    B, R, S, num_heads, kv_heads, D, K_sel = 1, 2, 2048, 32, 2, 256, 2048
    q = (torch.randn(B, R, num_heads * D, generator=gen, device="cuda") * 0.1).bfloat16()
    k_new = (torch.randn(B, R, kv_heads * D, generator=gen, device="cuda") * 0.1).bfloat16()
    v_new = (torch.randn(B, R, kv_heads * D, generator=gen, device="cuda") * 0.1).bfloat16()
    pk = (torch.randn(B, kv_heads, S, D, generator=gen, device="cuda") * 0.1).bfloat16()
    pv = (torch.randn(B, kv_heads, S, D, generator=gen, device="cuda") * 0.1).bfloat16()
    indices = torch.full((R, K_sel), -1, dtype=torch.int32, device="cuda")
    indices[0, :8] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 8], dtype=torch.int32, device="cuda")
    indices[1, :8] = torch.tensor([0, 1, 2, 3, 4, 5, 8, 9], dtype=torch.int32, device="cuda")
    counts = torch.tensor([8, 8], dtype=torch.int32, device="cuda")
    total_seq_len = torch.tensor([9], dtype=torch.int32, device="cuda")
    past_seq_len = torch.tensor([8], dtype=torch.int32, device="cuda")

    k_sc = (torch.concat([pk.reshape(-1, kv_heads * D), k_new.reshape(-1, kv_heads * D)], dim=0).float().abs().amax(dim=0) / 127.0).clamp(min=1e-6)
    v_sc = (torch.concat([pv.reshape(-1, kv_heads * D), v_new.reshape(-1, kv_heads * D)], dim=0).float().abs().amax(dim=0) / 127.0).clamp(min=1e-6)
    pk_i8 = torch.round(pk.float() / k_sc.view(1, kv_heads, 1, D)).clamp(-127, 127).to(torch.int8)
    pv_i8 = torch.round(pv.float() / v_sc.view(1, kv_heads, 1, D)).clamp(-127, 127).to(torch.int8)

    def run_dsa(sess, kv_type, pk_buf, pv_buf):
        b = sess.io_binding()
        b.bind_input("q", "cuda", 0, ORT_ELEM[BF], [B, R, num_heads * D], q.data_ptr())
        b.bind_input("k", "cuda", 0, ORT_ELEM[BF], [B, R, kv_heads * D], k_new.data_ptr())
        b.bind_input("v", "cuda", 0, ORT_ELEM[BF], [B, R, kv_heads * D], v_new.data_ptr())
        b.bind_input("pk", "cuda", 0, ORT_ELEM[kv_type], [B, kv_heads, S, D], pk_buf.data_ptr())
        b.bind_input("pv", "cuda", 0, ORT_ELEM[kv_type], [B, kv_heads, S, D], pv_buf.data_ptr())
        b.bind_input("idx", "cuda", 0, ORT_ELEM[I32], [R, K_sel], indices.data_ptr())
        b.bind_input("cnt", "cuda", 0, ORT_ELEM[I32], [R], counts.data_ptr())
        b.bind_input("tot", "cuda", 0, ORT_ELEM[I32], [1], total_seq_len.data_ptr())
        b.bind_input("past", "cuda", 0, ORT_ELEM[I32], [1], past_seq_len.data_ptr())
        out = torch.empty((B, R, num_heads * D), dtype=torch.bfloat16, device="cuda")
        err = torch.zeros((1,), dtype=torch.int32, device="cuda")
        b.bind_output("out", "cuda", 0, ORT_ELEM[BF], [B, R, num_heads * D], out.data_ptr())
        b.bind_output("prk", "cuda", 0, ORT_ELEM[kv_type], [B, kv_heads, S, D], pk_buf.data_ptr())
        b.bind_output("prv", "cuda", 0, ORT_ELEM[kv_type], [B, kv_heads, S, D], pv_buf.data_ptr())
        b.bind_output("err", "cuda", 0, ORT_ELEM[I32], [1], err.data_ptr())
        sess.run_with_iobinding(b)
        torch.cuda.synchronize()
        assert int(err.item()) == 0, f"GuardedDSA error flag {err.item()}"
        return out

    sess_bf = make_session(lib, "qwen.static", "GuardedDSA",
                           [("q", BF, [B, R, num_heads * D]), ("k", BF, [B, R, kv_heads * D]), ("v", BF, [B, R, kv_heads * D]),
                            ("pk", BF, [B, kv_heads, S, D]), ("pv", BF, [B, kv_heads, S, D]),
                            ("idx", I32, [R, K_sel]), ("cnt", I32, [R]), ("tot", I32, [1]), ("past", I32, [1])],
                           [("out", BF, [B, R, num_heads * D]), ("prk", BF, [B, kv_heads, S, D]), ("prv", BF, [B, kv_heads, S, D]),
                            ("err", I32, [1])],
                           head_size=D, num_heads=num_heads, kv_num_heads=kv_heads, selected_capacity=K_sel, cache_capacity=S, scale=0.0625)
    out_bf = run_dsa(sess_bf, BF, pk.clone(), pv.clone())

    sess_i8 = make_session(lib, "qwen.static", "GuardedDSA",
                           [("q", BF, [B, R, num_heads * D]), ("k", BF, [B, R, kv_heads * D]), ("v", BF, [B, R, kv_heads * D]),
                            ("pk", I8, [B, kv_heads, S, D]), ("pv", I8, [B, kv_heads, S, D]),
                            ("idx", I32, [R, K_sel]), ("cnt", I32, [R]), ("tot", I32, [1]), ("past", I32, [1])],
                           [("out", BF, [B, R, num_heads * D]), ("prk", I8, [B, kv_heads, S, D]), ("prv", I8, [B, kv_heads, S, D]),
                            ("err", I32, [1])],
                           head_size=D, num_heads=num_heads, kv_num_heads=kv_heads, selected_capacity=K_sel, cache_capacity=S, scale=0.0625,
                           k_scale=k_sc.cpu().numpy().astype(np.float32).tolist(),
                           v_scale=v_sc.cpu().numpy().astype(np.float32).tolist())
    out_i8 = run_dsa(sess_i8, I8, pk_i8.clone(), pv_i8.clone())
    rel_err = ((out_i8.float() - out_bf.float()).norm() / out_bf.float().norm()).item()
    assert rel_err < 0.015, f"GuardedDSA INT8 relative error too high: {rel_err}"


def test_skinny_gemm_and_dequant_and_untile(lib):
    gen = torch.Generator(device="cuda").manual_seed(5)
    K, N = 256, 128
    widths = [64, 64]
    x = torch.randn(1, 2, K, generator=gen, device="cuda").bfloat16()
    w = (torch.randn(K, N, generator=gen, device="cuda") * 0.05).bfloat16()
    w_amax = w.float().abs().amax(dim=0).clamp(min=1e-8)
    w_scale = (w_amax / 448.0).cpu().numpy().astype(np.float32).tolist()
    w_fp8 = (w.float() / torch.tensor(w_scale, device="cuda")[None, :]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    # SkinnyGemm FP8 vs dequantized BF16
    sess_deq = make_session(lib, "qwen.fused_gemm", "DequantWeightBf16", [("w", FP8, [K, N]), ("x", BF, [1, 2, K])], [("o", BF, [K, N])],
                            n_cols=N, w_scale=w_scale)
    b = sess_deq.io_binding()
    b.bind_input("w", "cuda", 0, FP8, [K, N], w_fp8.view(torch.uint8).data_ptr())
    b.bind_input("x", "cuda", 0, BF, [1, 2, K], x.data_ptr())
    w_deq = torch.empty((K, N), dtype=torch.bfloat16, device="cuda")
    b.bind_output("o", "cuda", 0, BF, [K, N], w_deq.data_ptr())
    sess_deq.run_with_iobinding(b)
    ref_deq = (w_fp8.float() * torch.tensor(w_scale, device="cuda")[None, :]).bfloat16()
    assert torch.equal(w_deq, ref_deq), "DequantWeightBf16 FP8 mismatch"

    sess_fp8 = make_session(lib, "qwen.fused_gemm", "SkinnyGemm", [("x", BF, [1, 2, K]), ("w", FP8, [K, N])],
                            [("o0", BF, [1, 2, 64]), ("o1", BF, [1, 2, 64])], seg_outs=[2], widths=widths, w_scale=w_scale)
    b2 = sess_fp8.io_binding()
    b2.bind_input("x", "cuda", 0, BF, [1, 2, K], x.data_ptr())
    b2.bind_input("w", "cuda", 0, FP8, [K, N], w_fp8.view(torch.uint8).data_ptr())
    o0 = torch.empty((1, 2, 64), dtype=torch.bfloat16, device="cuda")
    o1 = torch.empty((1, 2, 64), dtype=torch.bfloat16, device="cuda")
    b2.bind_output("o0", "cuda", 0, BF, [1, 2, 64], o0.data_ptr())
    b2.bind_output("o1", "cuda", 0, BF, [1, 2, 64], o1.data_ptr())
    sess_fp8.run_with_iobinding(b2)
    y_fp8 = torch.cat([o0, o1], dim=-1)
    y_ref = (x.float() @ ref_deq.float()).bfloat16()
    assert ((y_fp8.float() - y_ref.float()).norm() / y_ref.float().norm()).item() < 0.01, "SkinnyGemm FP8 error too high"

    # UntileHcBf16 kind=0 ([160, 20480] -> [10240, 320])
    w_orig = torch.randint(0, 30000, (10240, 320), dtype=torch.int16, device="cuda").view(torch.bfloat16)
    w_tiled = w_orig.view(8, 10, 128, 20, 2, 8).permute(0, 3, 1, 2, 4, 5).contiguous().view(160, 20480)
    sess_ut = make_session(lib, "qwen.fused_gemm", "UntileHcBf16", [("w", BF, [160, 20480])], [("o", BF, [10240, 320])], kind=0)
    out_ut = run(sess_ut, {"w": w_tiled}, {"o": (torch.bfloat16, [10240, 320])})["o"]
    assert torch.equal(out_ut.view(torch.int16), w_orig.view(torch.int16)), "UntileHcBf16 kind=0 mismatch"


OP_TESTS = [
    ("qwen_fused_attn_r5", test_sigmoid_gate_mul),
    ("qwen_fused_ops", test_swish_mul),
    ("qwen_fused_gemm_r5", test_argmax_rows),
    ("qwen_fused_gemm_r5", test_skinny_gemm_and_dequant_and_untile),
    ("qwen_mtp_head", test_bf16_argmax),
    ("qwen_mtp_head", test_mtp_moe_modes),
    ("qwen_guarded_dsa_r2_v12", test_guarded_dsa_kv_modes),
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
