# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the CPU/static checks of the small patches 001, 003 and 012 (python3 >= 3.10 + stdlib; numpy only
where a check says so). No GPU. Read-only for the vendored sources: scratch roots are temp copies.

The vLLM tree the checks read is `--vllm-root`, else $VLLM_TREE, else the installed `vllm` package directory (found
with importlib's finder, which does not import vllm). Inside the recipe image that is the tree the patches edit."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The patch directory under test: <repo>/patches, or $PATCH_DIR. These checks live in <repo>/tests/patch-checks/.
PDIR = Path(os.environ.get("PATCH_DIR") or (HERE.parents[1] / "patches")).resolve()
FDIR = PDIR / "files"
RESULTS = {"pass": 0, "fail": 0, "skip": 0, "failed": []}


def default_vroot() -> str:
    env = os.environ.get("VLLM_TREE")
    if env:
        return env
    spec = importlib.util.find_spec("vllm")
    if spec is not None and spec.submodule_search_locations:
        return str(list(spec.submodule_search_locations)[0])
    return "vllm"


def check(name, ok, info=""):
    ok = bool(ok)
    RESULTS["pass" if ok else "fail"] += 1
    if not ok:
        RESULTS["failed"].append(name)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    return ok


def skip(name, why):
    RESULTS["skip"] += 1
    print(f"SKIP {name}  [{why}]", flush=True)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p) -> str:
    return sha_bytes(Path(p).read_bytes())


def pin(sh: Path, name: str) -> str:
    m = re.search(rf"^{name}=([0-9a-f]{{64}})$", sh.read_text(), re.M)
    if not m:
        raise SystemExit(f"no pin {name} (64 hex) in {sh}")
    return m.group(1)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def compiles_312(text: str, name: str = "<text>") -> bool:
    """compile() + a 3.12-grammar parse (the image runs python 3.12)."""
    try:
        compile(text, name, "exec")
        ast.parse(text, feature_version=(3, 12))
        return True
    except SyntaxError:
        return False


def raises(fn, exc=Exception, match=None) -> bool:
    try:
        fn()
    except exc as e:  # noqa: BLE001
        return match is None or re.search(match, str(e)) is not None
    return False


def make_root(vroot: Path, rels: list[str]) -> Path:
    """A scratch VLLM_ROOT holding copies of the given files (relative paths under the vllm package root)."""
    root = Path(tempfile.mkdtemp(prefix="smallchk_"))
    for rel in rels:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vroot / rel, dst)
    return root


def run_sh(sh: Path, root: Path, env: dict | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    e["VLLM_ROOT"] = str(root)
    e.update(env or {})
    return subprocess.run(["bash", str(sh)], env=e, capture_output=True, text=True, timeout=timeout)


def finish(tag: str) -> int:
    r = RESULTS
    print(f"\n== {tag}: {r['pass']} PASS, {r['fail']} FAIL, {r['skip']} SKIP"
          + (f"  FAILED: {r['failed']}" if r["failed"] else ""), flush=True)
    return 0 if r["fail"] == 0 else 1


def parse_args(argv, extra=()):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", default=default_vroot(), help="the vllm package directory the patches edit")
    for a in extra:
        ap.add_argument(a, action="store_true")
    return ap.parse_args(argv)
