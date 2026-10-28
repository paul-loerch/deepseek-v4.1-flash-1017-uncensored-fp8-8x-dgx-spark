#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 012 applier (vLLM nightly e4a5b55 port): two anchored edits to vllm/distributed/device_communicators/cuda_communicator.py.

  apply(text) -> text     nightly original -> + patch 012; fail-closed: both anchors must match exactly once, the hooks must
                          land where they are meant to (AST-checked: the creation right after the PyNcclCommunicator block of
                          CudaCommunicator.__init__ at method level; the dispatch as the FIRST statements of
                          CudaCommunicator.all_reduce), the result must carry N_MARKERS markers and compile
  revert(text) -> text    exact inverse (the tests rebuild the base from the patched file)

CLI (used by 012-gb10-oneshot-allreduce.sh):
  p012_apply.py apply  <cuda_communicator.py>   in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p012_apply.py check  <cuda_communicator.py>   dry run: print the sha256 the file would have after apply()
  p012_apply.py revert <cuda_communicator.py>   in-place inverse (tests / manual rollback)

Edits (nightly line numbers; identical text to the v0.32.0 port):
  1. CudaCommunicator.__init__, directly after
         if is_symmetric_memory_enabled():
             register_nccl_symmetric_ops(self.pynccl_comm)                                            (:112-113)
     (the end of the `if self.world_size > 1:` PyNcclCommunicator block), at method level:
         # [patch-012 gb10-oneshot-ar] ...
         from vllm._gb10_oneshot_ar import maybe_create as _gb10_os_create
         self._gb10_oneshot = _gb10_os_create(self.pynccl_comm, self.device, self.unique_name)
  2. CudaCommunicator.all_reduce(self, input_) (:362), first statements:
         # [patch-012 gb10-oneshot-ar]
         _gb10_os = getattr(self, '_gb10_oneshot', None)
         if _gb10_os is not None and _gb10_os.should_use(input_):
             return _gb10_os(input_)
     i.e. BEFORE the dispatch chain (NCCL symmetric memory, FlashInfer all-reduce, quick reduce, FlashInfer PCIe-IPC,
     AITER, custom all-reduce, torch symm mem, pynccl). maybe_create returns None unless VLLM_GB10_ONESHOT_AR=1 and the group
     is the TP group, so without the gate both hooks are inert.
Against v0.32.0 the nightly changed only the symmetric-memory scratch handling of this file (clone of the scratch output);
neither anchor nor the dispatch chain moved.
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

MARKER = "# [patch-012 gb10-oneshot-ar]"
N_MARKERS = 2
CLASS = "CudaCommunicator"

INIT_OLD = (
    "            if is_symmetric_memory_enabled():\n"
    "                register_nccl_symmetric_ops(self.pynccl_comm)\n"
)
INIT_NEW = INIT_OLD + (
    f"        {MARKER} opt-in: VLLM_GB10_ONESHOT_AR=1 -> one-shot all-reduce for small bf16 TP messages\n"
    "        from vllm._gb10_oneshot_ar import maybe_create as _gb10_os_create\n"
    "        self._gb10_oneshot = _gb10_os_create(self.pynccl_comm, self.device, self.unique_name)\n"
)
AR_OLD = "    def all_reduce(self, input_):\n"
AR_NEW = AR_OLD + (
    f"        {MARKER}\n"
    "        _gb10_os = getattr(self, '_gb10_oneshot', None)\n"
    "        if _gb10_os is not None and _gb10_os.should_use(input_):\n"
    "            return _gb10_os(input_)\n"
)
EDITS = [
    ("__init__ hook (after the PyNcclCommunicator block)", INIT_OLD, INIT_NEW),
    ("all_reduce dispatch (first statements)", AR_OLD, AR_NEW),
]


class Refused(ValueError):
    pass


def _method(tree: ast.Module, name: str):
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS]
    if len(cls) != 1:
        raise Refused(f"class {CLASS} found {len(cls)} times (expected 1)")
    fns = [n for n in cls[0].body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(fns) != 1:
        raise Refused(f"{CLASS}.{name} found {len(fns)} times (expected 1)")
    return fns[0]


def _assigns_attr(stmt, attr: str) -> bool:
    return isinstance(stmt, ast.Assign) and any(ast.unparse(t) == f"self.{attr}" for t in stmt.targets)


def check_structure(text: str, patched: bool) -> None:
    tree = ast.parse(text)
    init = _method(tree, "__init__")
    ar = _method(tree, "all_reduce")
    if [a.arg for a in ar.args.args] != ["self", "input_"]:
        raise Refused("CudaCommunicator.all_reduce is not (self, input_)")
    # the PyNcclCommunicator block: an `if` at method level whose body assigns self.pynccl_comm
    body = init.body
    blk = [i for i, s in enumerate(body) if isinstance(s, ast.If) and any(
        _assigns_attr(x, "pynccl_comm") for x in ast.walk(s) if isinstance(x, ast.Assign))]
    if len(blk) != 1:
        raise Refused(f"{len(blk)} method-level `if` blocks assign self.pynccl_comm in {CLASS}.__init__ (expected 1)")
    b = blk[0]
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if not patched:
        if "_gb10_oneshot" in text or "_gb10_os" in names:
            raise Refused("base already mentions the one-shot hooks")
        return
    # patched: [..., If(pynccl block), ImportFrom(vllm._gb10_oneshot_ar), Assign(self._gb10_oneshot = ...), ...]
    imp, asg = body[b + 1], body[b + 2]
    if not (isinstance(imp, ast.ImportFrom) and imp.module == "vllm._gb10_oneshot_ar"
            and [(a.name, a.asname) for a in imp.names] == [("maybe_create", "_gb10_os_create")]):
        raise Refused("the maybe_create import does not directly follow the PyNcclCommunicator block")
    if not (_assigns_attr(asg, "_gb10_oneshot")
            and ast.unparse(asg.value) == "_gb10_os_create(self.pynccl_comm, self.device, self.unique_name)"):
        raise Refused("self._gb10_oneshot is not assigned from _gb10_os_create(self.pynccl_comm, self.device, self.unique_name)")
    st = ar.body
    if len(st) < 3:
        raise Refused("all_reduce body too short")
    if not (isinstance(st[0], ast.Assign) and ast.unparse(st[0]) == "_gb10_os = getattr(self, '_gb10_oneshot', None)"):
        raise Refused("the first statement of all_reduce is not the _gb10_os lookup")
    s1 = st[1]
    if not (isinstance(s1, ast.If) and ast.unparse(s1.test) == "_gb10_os is not None and _gb10_os.should_use(input_)"
            and len(s1.body) == 1 and isinstance(s1.body[0], ast.Return) and ast.unparse(s1.body[0].value) == "_gb10_os(input_)"
            and not s1.orelse):
        raise Refused("the second statement of all_reduce is not the one-shot dispatch")
    if sum(1 for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "_gb10_os") != 4:
        raise Refused("_gb10_os is used by something other than the dispatch")


def apply(text: str) -> str:
    if MARKER in text:
        raise Refused("input already carries the patch-012 marker")
    bad = [(name, text.count(old)) for name, old, _ in EDITS if text.count(old) != 1]
    if bad:
        raise Refused("; ".join(f"anchor '{n}' found {c}x (expected 1)" for n, c in bad))
    check_structure(text, patched=False)
    out = text
    for name, old, new in EDITS:
        if out.count(old) != 1:
            raise Refused(f"anchor '{name}' changed by an earlier edit")
        out = out.replace(old, new, 1)
    if out.count(MARKER) != N_MARKERS:
        raise Refused(f"expected {N_MARKERS} markers after the edits, found {out.count(MARKER)}")
    compile(out, "cuda_communicator.py", "exec")
    check_structure(out, patched=True)
    return out


def revert(text: str) -> str:
    out = text
    for name, old, new in reversed(EDITS):
        if out.count(new) != 1:
            raise Refused(f"patched block '{name}' found {out.count(new)}x (expected 1)")
        out = out.replace(new, old, 1)
    if MARKER in out:
        raise Refused("marker still present after revert")
    compile(out, "cuda_communicator.py", "exec")
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("apply", "check", "revert"):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        print("usage: p012_apply.py {apply|check|revert} <cuda_communicator.py>", file=sys.stderr)
        return 2
    p = Path(argv[2])
    try:
        src = p.read_text(encoding="utf-8")
        new = revert(src) if argv[1] == "revert" else apply(src)
    except (Refused, SyntaxError, OSError, UnicodeDecodeError) as e:
        print(f"[FAIL] {p}: {e}", file=sys.stderr)
        return 1
    if argv[1] == "check":
        print(_sha(new.encode()))
        return 0
    tmp = p.with_name(p.name + ".p012.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    print(" [OK] wired maybe_create into CudaCommunicator.__init__ and the one-shot dispatch into all_reduce"
          if argv[1] == "apply" else " [OK] reverted the one-shot hooks")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
