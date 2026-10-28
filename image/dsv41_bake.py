#!/usr/bin/env python3
"""GB10 bake for DeepSeek-V4.1-Flash-1017 on vLLM (nightly e4a5b55, FlashInfer 0.7.2.post1, arm64, CUDA 13).

Build-time source edits, applied once while the image is built. No network, no GPU, no imports of vllm or
torch: the files are located with importlib's finder (which does not execute the package) and edited as text.

Rules:
  * every edit must match exactly once, or the build fails (exit 1) and NOTHING is written
    (plan every file first, compile-check every patched .py, then write);
  * a file that already carries an edit (new text present once, old text absent) is left alone, so a
    re-run is a no-op; anything in between is an error;
  * the pristine sha256 of every edited file is pinned (fail closed; --allow-pin-mismatch overrides after
    you have read the diff yourself);
  * a unified diff per file is printed and a manifest is written for verify_bake.py.

usage: python3 -I dsv41_bake.py [--vllm-root DIR] [--dry-run] [--list] [--print-pins]
                                [--allow-pin-mismatch] [--manifest PATH]
"""
from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

LOG = "[dsv41-bake]"
BAKE_VERSION = 1
EXPECTED_BASE = (
    "vllm/vllm-openai:nightly-e4a5b5537344b103ed10de7764378ef09604ef3d (arm64 "
    "sha256:27aa6d814d1fc44ff8f13bb50c291cb23f54e15544ace8b699e0b29bdcc9451b)"
)
DEFAULT_MANIFEST = "/opt/dsv41-bake/bake-manifest.json"


@dataclasses.dataclass(frozen=True)
class Edit:
    id: str
    rel: str  # path relative to the vllm package root
    old: str
    new: str
    what: str


# Pristine sha256 of every edited file in the base image. Key: path relative to the vllm package root.
PINS = {
    "models/deepseek_v41/nvidia/flashinfer_sparse.py": "7926b4e3fa44fe0fc1ed3f93f0f292b4f56fa2103cf93c30a94d93958f84d682",
    "v1/attention/backends/mla/indexer.py": "6457fb5a755ca53addce7731cb56770493371ad9f3728251e7ff33515b67965c",
}

# The edits. Strings start at column 0 inside the triple quotes (they are file text).
EDITS = [
    # vLLM PR #60762 (open as of 2026-10-25): on SM12x the DeepSeek-V4.1 sparse-MLA backend and the indexer
    # declare 128-token kernel blocks, but the SM12x kernels (FlashInfer SM120 sparse MLA decode, DeepGEMM SM120
    # paged MQA logits) take 32/64-token pages, so the engine either fails the block-size negotiation ("No common
    # block size for 64") or asserts in DeepGEMM ("block_kv == 32 or block_kv == 64"). Declare 64 on family 120.
    Edit(
        id="dsv41_sparse_blocks",
        rel="models/deepseek_v41/nvidia/flashinfer_sparse.py",
        what="FLASHINFER_MLA_SPARSE_DSV41 declares 64-token kernel blocks on SM12x (PR #60762)",
        old="""\
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [128]
""",
        new="""\
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # GB10 bake (vLLM PR #60762): the SM12x sparse-MLA kernels take 64-token pages.
        from vllm.platforms import current_platform

        if current_platform.is_device_capability_family(120):
            return [64]
        return [128]
""",
    ),
    Edit(
        id="dsv41_indexer_blocks",
        rel="v1/attention/backends/mla/indexer.py",
        what="DeepseekV41IndexerBackend declares 64-token kernel blocks on SM12x (PR #60762)",
        old="""\
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [64 if current_platform.is_device_capability_family(90) else 128]
""",
        new="""\
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # GB10 bake (vLLM PR #60762): DeepGEMM's SM120 paged-MQA logits take 32/64-state pages.
        if current_platform.is_device_capability_family(120):
            return [64]
        return [64 if current_platform.is_device_capability_family(90) else 128]
""",
    ),
]


def _self_check_table() -> None:
    ids = set()
    for e in EDITS:
        assert e.id not in ids, f"duplicate edit id {e.id}"
        ids.add(e.id)
        assert e.old and e.new and e.old != e.new, e.id
        assert e.old not in e.new, f"{e.id}: old text is a substring of new text"
        assert e.new not in e.old, f"{e.id}: new text is a substring of old text"
        assert e.old.endswith("\n") and e.new.endswith("\n"), e.id
        assert e.rel in PINS, f"{e.id}: no pin for {e.rel}"
    for a in EDITS:
        for b in EDITS:
            if a is not b and a.rel == b.rel:
                for x in (a.old, a.new):
                    for y in (b.old, b.new):
                        assert x not in y, f"{a.id} and {b.id} overlap"


_self_check_table()


class BakeError(Exception):
    pass


def log(msg: str) -> None:
    print(f"{LOG} {msg}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def locate_vllm(override: str | None) -> Path:
    if override:
        root = Path(override)
    else:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            raise BakeError("vllm is not importable here (use --vllm-root)")
        root = Path(list(spec.submodule_search_locations)[0])
    if not (root / "__init__.py").is_file():
        raise BakeError(f"not a vllm package root: {root}")
    return root


def drop_stale_pyc(path: Path) -> None:
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(path.stem + ".*.pyc"):
            pyc.unlink()


def atomic_write(path: Path, data: bytes) -> None:
    mode = path.stat().st_mode & 0o7777
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


@dataclasses.dataclass
class FilePlan:
    rel: str
    path: Path
    before: bytes
    after: bytes
    applied: list
    pending: list


def plan(vllm_root: Path, *, allow_pin_mismatch: bool) -> list[FilePlan]:
    plans = []
    for rel in sorted({e.rel for e in EDITS}):
        path = vllm_root / rel
        if not path.is_file():
            raise BakeError(f"{rel}: file not found")
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        applied, pending = [], []
        for e in (x for x in EDITS if x.rel == rel):
            n_old, n_new = text.count(e.old), text.count(e.new)
            if n_old == 1 and n_new == 0:
                pending.append(e)
            elif n_old == 0 and n_new == 1:
                applied.append(e)
            else:
                raise BakeError(f"{e.id}: anchor found {n_old}x, edited text {n_new}x in {rel} (need 1/0 or 0/1)")
        if pending and applied:
            raise BakeError(f"{rel}: half-patched ({[e.id for e in applied]} applied, {[e.id for e in pending]} not)")
        if pending:
            pin = PINS[rel]
            got = sha256_bytes(raw)
            if got != pin:
                if not allow_pin_mismatch:
                    raise BakeError(f"{rel}: sha256 {got} != pinned {pin} (different base image?)")
                log(f"WARNING {rel}: sha256 {got} != pinned {pin}; continuing (--allow-pin-mismatch)")
            for e in pending:
                text = text.replace(e.old, e.new, 1)
        after = text.encode("utf-8")
        if pending:
            compile(after, str(path), "exec")
        plans.append(FilePlan(rel, path, raw, after, applied, pending))
    return plans


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm-root")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--print-pins", action="store_true")
    ap.add_argument("--allow-pin-mismatch", action="store_true")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    a = ap.parse_args(argv)
    if a.list:
        for e in EDITS:
            print(f"{e.id:24s} {e.rel}\n{'':24s} {e.what}")
        return 0
    try:
        root = locate_vllm(a.vllm_root)
        if a.print_pins:
            for rel in sorted(PINS):
                print(f'    "{rel}": "{sha256_bytes((root / rel).read_bytes())}",')
            return 0
        plans = plan(root, allow_pin_mismatch=a.allow_pin_mismatch)
    except BakeError as exc:
        log(f"FAILED: {exc}")
        return 1
    for p in plans:
        if p.pending:
            diff = difflib.unified_diff(
                p.before.decode().splitlines(True), p.after.decode().splitlines(True),
                fromfile="a/vllm/" + p.rel, tofile="b/vllm/" + p.rel, n=2)
            sys.stdout.writelines(diff)
    if a.dry_run:
        log("dry run: nothing written")
        return 0
    for p in plans:
        if p.pending:
            atomic_write(p.path, p.after)
            drop_stale_pyc(p.path)
    manifest = {
        "bake_version": BAKE_VERSION,
        "expected_base": EXPECTED_BASE,
        "vllm_root": str(root),
        "files": {p.rel: {"pristine_sha256": PINS[p.rel], "baked_sha256": sha256_bytes(p.after),
                          "edits": [e.id for e in p.applied + p.pending]} for p in plans},
    }
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    for p in plans:
        state = "applied now" if p.pending else "already applied"
        log(f"[OK] {p.rel}: {', '.join(e.id for e in p.applied + p.pending)} ({state})")
    log(f"manifest written to {a.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
