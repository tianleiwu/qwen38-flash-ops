#!/usr/bin/env python3
"""Compare freshly built custom-op libraries with a reference set (e.g. the libs of a shipped model package).

For every library name present in both directories the script compares
  * the dynamic symbol table (`nm -D`, addresses stripped),
  * the device code: SASS of every embedded cubin (`cuobjdump -sass`) and the PTX text (`cuobjdump -ptx`) when present,
  * the SHA256 of the files (informational; a bit-identical file is not expected because the link step embeds
    different build paths / ordering).
Exit status 0 iff symbols and SASS are identical for all compared libraries.

usage: compare_libs.py NEW_DIR REF_DIR [--cuobjdump PATH] [--only lib1.so lib2.so ...]
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def symbols(path):
    out = run(["nm", "-D", str(path)])
    return sorted(" ".join(line.split()[-2:]) if len(line.split()) > 2 else line.strip() for line in out.splitlines())


_ANON = re.compile(r"_(GLOBAL__N|INTERNAL)_+[0-9a-f]{8}_\d+_(\w+?)_cu_[0-9a-f]{8}")


def device_code(path, cuobjdump, flag):
    """SHA256 of the device code text with the path-derived hash of anonymous-namespace symbols removed.
    Header blocks ("Fatbin ... code for sm_XX") are dropped, so an empty extra cubin does not matter."""
    try:
        text = run([cuobjdump, flag, str(path)])
    except subprocess.CalledProcessError:
        return None
    keep = []
    for line in _ANON.sub(r"_\1__\2", text).splitlines():
        s = line.strip()
        if not s or s.startswith(("Fatbin", "=====", "arch =", "code version", "host =", "compile_size", "code for",
                                  "identifier", "ptxasOptions", ".target", "//")):
            continue
        keep.append(line)
    return hashlib.sha256("\n".join(keep).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_dir")
    ap.add_argument("ref_dir")
    ap.add_argument("--cuobjdump", default="cuobjdump")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--json", help="write the result here")
    args = ap.parse_args()
    new, ref = pathlib.Path(args.new_dir), pathlib.Path(args.ref_dir)
    names = args.only or sorted(p.name for p in new.glob("lib*.so") if (ref / p.name).exists())
    ok, result = True, {}
    for name in names:
        a, b = new / name, ref / name
        sym = symbols(a) == symbols(b)
        sass = device_code(a, args.cuobjdump, "-sass") == device_code(b, args.cuobjdump, "-sass")
        ptx_a, ptx_b = device_code(a, args.cuobjdump, "-ptx"), device_code(b, args.cuobjdump, "-ptx")
        ptx = ptx_a == ptx_b
        same_file = sha256(a) == sha256(b)
        result[name] = {"symbols_equal": sym, "sass_equal": sass, "ptx_equal": ptx, "file_identical": same_file,
                        "size_new": a.stat().st_size, "size_ref": b.stat().st_size}
        ok &= sym and sass and ptx
        print(f"{name:42s} symbols={sym} sass={sass} ptx={ptx} identical_file={same_file} "
              f"size={a.stat().st_size}/{b.stat().st_size}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(result, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
