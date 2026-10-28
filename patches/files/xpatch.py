#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""xpatch.py -- applier for the recipe's experimental patches (one module per patch: files/pNNN_*.py).

A patch module defines:
  ID     e.g. "013"
  TAG    short name, used in the marker comment "# [patch-ID TAG]"
  EDITS  list of (path relative to the vllm package, sha256 of the original file, old, new)
         `old` must occur exactly once in the original; `new` must contain the marker.

  xpatch.py apply MODULE.py [VLLM_ROOT]   apply all edits (atomic per file, fail-closed), verify every patched file parses
  xpatch.py check MODULE.py [VLLM_ROOT]   dry run

Fail-closed: a file whose sha256 is neither the pinned original nor already patched by this module (marker present and the
re-derived result matches) is refused. Idempotent: an already patched file is left alone.
"""
import ast
import hashlib
import importlib.util
import os
import sys
import tempfile


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load(module_path: str):
    spec = importlib.util.spec_from_file_location("xpatch_mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan(mod, root: str):
    marker = f"# [patch-{mod.ID} {mod.TAG}]"
    by_file: dict[str, list] = {}
    for rel, orig_sha, old, new in mod.EDITS:
        if marker not in new:
            raise SystemExit(f"[FAIL] [{mod.ID}] an edit for {rel} lacks the marker {marker!r}")
        by_file.setdefault(rel, []).append((orig_sha, old, new))
    out = []
    for rel, edits in by_file.items():
        path = os.path.join(root, rel)
        text = open(path).read()
        shas = {e[0] for e in edits}
        if len(shas) != 1:
            raise SystemExit(f"[FAIL] [{mod.ID}] {rel}: edits pin different original shas")
        orig_sha = shas.pop()
        now = sha(text)
        if now == orig_sha:
            result = text
            for _, old, new in edits:
                n = result.count(old)
                if n != 1:
                    raise SystemExit(f"[FAIL] [{mod.ID}] {rel}: anchor found {n} times (need 1): {old[:80]!r}")
                result = result.replace(old, new, 1)
            out.append((rel, path, "apply", result))
        elif marker in text:
            out.append((rel, path, "installed", text))
        else:
            raise SystemExit(f"[FAIL] [{mod.ID}] {rel}: sha256 {now[:16]}.. is not the pinned original "
                             f"{orig_sha[:16]}.. and carries no patch-{mod.ID} marker (different vLLM build?)")
    return out


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("apply", "check"):
        raise SystemExit(__doc__)
    cmd, module_path = sys.argv[1], sys.argv[2]
    root = sys.argv[3] if len(sys.argv) > 3 else "/usr/local/lib/python3.12/dist-packages/vllm"
    mod = load(module_path)
    for rel, path, action, result in plan(mod, root):
        try:
            ast.parse(result)
        except SyntaxError as e:
            raise SystemExit(f"[FAIL] [{mod.ID}] {rel}: result does not parse: {e}")
        if action == "installed":
            print(f" [skip] [{mod.ID}] {rel} already patched")
            continue
        if cmd == "check":
            print(f" [check] [{mod.ID}] {rel}: would apply, result sha256 {sha(result)[:16]}..")
            continue
        st = os.stat(path)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
        with os.fdopen(fd, "w") as f:
            f.write(result)
        os.chmod(tmp, st.st_mode & 0o7777)
        os.replace(tmp, path)
        print(f" [OK] [{mod.ID}] {rel} patched (sha256 {sha(result)[:16]}..)")


if __name__ == "__main__":
    main()
