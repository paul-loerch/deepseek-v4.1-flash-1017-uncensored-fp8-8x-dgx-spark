#!/usr/bin/env python3
"""Build a checkpoint directory = a base safetensors checkpoint with some tensors replaced from an overlay file.

Made for "official DeepSeek-V4.1-Flash-1017 + drowzeys Keys overlay" (26 attn.wo_b weights + 26 scales, layers 10-35),
but generic: every tensor in the overlay must exist in the base with the same dtype and shape, so the replaced bytes
have exactly the same size and every safetensors header stays byte-identical.

  * files of the base that hold no replaced tensor are HARDLINKED into the output (same filesystem; never written);
  * a shard that holds a replaced tensor is COPIED to a new inode (O_EXCL) and only the tensor's data bytes are
    overwritten with pwrite; the base files are opened read-only and never modified;
  * verification afterwards: every replaced tensor's bytes in the output equal the overlay's; every byte of a patched
    shard outside the replaced ranges equals the base shard; every hardlink points at the base inode;
  * a manifest (BUILD.json) records the base, the overlay, the replaced tensors and the sha256 of every patched shard.

usage: build_overlay_variant.py --base BASE_MODEL_DIR --overlay OVERLAY.safetensors --out OUT_DIR [--extra FILE ...]
       (--extra: small files copied next to the weights for provenance, e.g. the overlay's ABLIT_META.json)
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time

CHUNK = 64 << 20


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                return d.hexdigest()
            d.update(b)


def same_bytes(fa, fb, start, end):
    """Compare [start, end) of two open files chunk by chunk."""
    pos = start
    while pos < end:
        n = min(CHUNK, end - pos)
        fa.seek(pos)
        fb.seek(pos)
        if fa.read(n) != fb.read(n):
            return False
        pos += n
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    t0 = time.time()
    base, out = os.path.abspath(a.base), os.path.abspath(a.out)
    if os.path.exists(out):
        sys.exit(f"refusing: {out} exists")
    index = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    ov_start, ov = header(a.overlay)
    if not ov:
        sys.exit("overlay has no tensors")

    # plan: which base shard holds each overlay tensor; dtype/shape/size must match exactly
    plan = {}  # shard -> list of (name, base_abs_start, base_abs_end, ov_abs_start, ov_abs_end)
    base_headers = {}
    for name, meta in ov.items():
        shard = index.get(name)
        if shard is None:
            sys.exit(f"{name}: not in the base index")
        if shard not in base_headers:
            base_headers[shard] = header(os.path.join(base, shard))
        bstart, bh = base_headers[shard]
        bm = bh.get(name)
        if bm is None:
            sys.exit(f"{name}: index says {shard}, but the shard does not hold it")
        if bm["dtype"] != meta["dtype"] or bm["shape"] != meta["shape"]:
            sys.exit(f"{name}: base {bm['dtype']}{bm['shape']} != overlay {meta['dtype']}{meta['shape']}")
        b0, b1 = bm["data_offsets"]
        o0, o1 = meta["data_offsets"]
        if b1 - b0 != o1 - o0:
            sys.exit(f"{name}: byte size differs")
        plan.setdefault(shard, []).append((name, bstart + b0, bstart + b1, ov_start + o0, ov_start + o1))
    print(f"{len(ov)} overlay tensors in {len(plan)} base shards", flush=True)

    # build: hardlink everything except the patched shards, which are copied then patched
    os.makedirs(out)
    for root, dirs, files in os.walk(base):
        rel = os.path.relpath(root, base)
        dst_root = out if rel == "." else os.path.join(out, rel)
        os.makedirs(dst_root, exist_ok=True)
        for fn in files:
            src, dst = os.path.join(root, fn), os.path.join(dst_root, fn)
            relf = os.path.normpath(os.path.join(rel, fn))
            if relf in plan:
                with open(src, "rb") as fi, open(dst, "xb") as fo:
                    shutil.copyfileobj(fi, fo, CHUNK)
            else:
                os.link(src, dst)
    with open(a.overlay, "rb") as fov:
        for shard, items in plan.items():
            fd = os.open(os.path.join(out, shard), os.O_WRONLY)
            try:
                for name, b0, b1, o0, o1 in items:
                    fov.seek(o0)
                    data = fov.read(o1 - o0)
                    if os.pwrite(fd, data, b0) != len(data):
                        sys.exit(f"{name}: short write")
                os.fsync(fd)
            finally:
                os.close(fd)
    for extra in a.extra:
        shutil.copy2(extra, os.path.join(out, os.path.basename(extra)))
    print(f"built in {time.time() - t0:.0f} s; verifying", flush=True)

    # verify
    problems = 0
    with open(a.overlay, "rb") as fov:
        for shard, items in plan.items():
            with open(os.path.join(base, shard), "rb") as fb, open(os.path.join(out, shard), "rb") as fo:
                size = os.fstat(fb.fileno()).st_size
                if os.fstat(fo.fileno()).st_size != size:
                    problems += 1
                    print("SIZE", shard)
                    continue
                cuts = sorted((b0, b1) for _, b0, b1, _, _ in items)
                pos = 0
                for b0, b1 in cuts:
                    if not same_bytes(fb, fo, pos, b0):
                        problems += 1
                        print("OUTSIDE-DIFF", shard, pos, b0)
                    pos = b1
                if not same_bytes(fb, fo, pos, size):
                    problems += 1
                    print("OUTSIDE-DIFF", shard, pos, size)
                for name, b0, b1, o0, o1 in items:
                    fo.seek(b0)
                    fov.seek(o0)
                    if fo.read(b1 - b0) != fov.read(o1 - o0):
                        problems += 1
                        print("TENSOR-DIFF", name)
    for root, _, files in os.walk(base):
        rel = os.path.relpath(root, base)
        for fn in files:
            relf = os.path.normpath(os.path.join(rel, fn))
            if relf in plan:
                continue
            if os.stat(os.path.join(root, fn)).st_ino != os.stat(os.path.join(out, relf)).st_ino:
                problems += 1
                print("NOT-A-HARDLINK", relf)
    manifest = {
        "base": base, "overlay": os.path.abspath(a.overlay), "overlay_sha256": sha256_file(a.overlay),
        "replaced": sorted(ov), "patched_shards": {s: sha256_file(os.path.join(out, s)) for s in sorted(plan)},
        "built_unix": int(time.time()), "problems": problems,
    }
    json.dump(manifest, open(os.path.join(out, "BUILD.json"), "w"), indent=1)
    print(f"verify: {problems} problem(s); {len(plan)} shards patched, {len(ov)} tensors replaced; {time.time() - t0:.0f} s")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
