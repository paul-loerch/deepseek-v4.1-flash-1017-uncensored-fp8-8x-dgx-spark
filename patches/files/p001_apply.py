#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch 001 applier (vLLM nightly e4a5b55 port): one anchored edit to vllm/distributed/device_communicators/shm_broadcast.py.

  apply(text) -> text     nightly original -> busy_loop_s default 1 -> 0.002; fail-closed: the anchor must match exactly
                          once, must sit inside SpinCondition.__init__ as the default of its `busy_loop_s` parameter
                          (AST-checked), the reader-side assignment must still consume it, the result must compile
  revert(text) -> text    exact inverse (the tests rebuild the base from the patched file)

CLI (used by 001-gb10-spin-wait.sh):
  p001_apply.py apply  <shm_broadcast.py>   in-place, atomic (temp file + os.replace, mode kept); exit 1 on any refusal
  p001_apply.py check  <shm_broadcast.py>   dry run: print the sha256 the file would have after apply()
  p001_apply.py revert <shm_broadcast.py>   in-place inverse (tests / manual rollback)

The edit (nightly line 141, `SpinCondition.__init__`): `busy_loop_s: float = 1,` -> `busy_loop_s: float = 0.002,`. Same value
and same default-argument site as the v0.32.0 port (upstream still has no knob for it); the line is unchanged between
v0.28.0 and this nightly, only its neighbours moved: the reader's wait moved into MessageQueue._wait_until_readable, and
SpinCondition.record_read() gained a periodic drain of the notify socket (SHM_NOTIFY_POLL_INTERVAL_S). Neither touches the
anchor or the reader-side `self.busy_loop_s = busy_loop_s` assignment that the structure check still requires.
"""
import ast
import hashlib
import os
import sys
from pathlib import Path

CLASS = "SpinCondition"
FUNC = "__init__"
PARAM = "busy_loop_s"
OLD_DEFAULT = 1
NEW_DEFAULT = 0.002

ANCHOR_OLD = "        busy_loop_s: float = 1,\n"
ANCHOR_NEW = "        busy_loop_s: float = 0.002,\n"
EDITS = [("SpinCondition.__init__ busy_loop_s default", ANCHOR_OLD, ANCHOR_NEW)]


class Refused(ValueError):
    pass


def _default_of(text: str):
    """The constant default of SpinCondition.__init__'s `busy_loop_s` parameter, or raise Refused."""
    tree = ast.parse(text)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS]
    if len(cls) != 1:
        raise Refused(f"class {CLASS} found {len(cls)} times (expected 1)")
    fns = [n for n in cls[0].body if isinstance(n, ast.FunctionDef) and n.name == FUNC]
    if len(fns) != 1:
        raise Refused(f"{CLASS}.{FUNC} found {len(fns)} times (expected 1)")
    args = fns[0].args
    names = [a.arg for a in args.args]
    if PARAM not in names:
        raise Refused(f"{CLASS}.{FUNC} has no parameter {PARAM}")
    k = names.index(PARAM) - (len(names) - len(args.defaults))
    if k < 0:
        raise Refused(f"{CLASS}.{FUNC}: {PARAM} has no default")
    d = args.defaults[k]
    if not isinstance(d, ast.Constant) or isinstance(d.value, bool) or not isinstance(d.value, (int, float)):
        raise Refused(f"{CLASS}.{FUNC}: {PARAM} default is not a numeric constant")
    return d.value


def check_structure(text: str, patched: bool) -> None:
    want = NEW_DEFAULT if patched else OLD_DEFAULT
    got = _default_of(text)
    if got != want:
        raise Refused(f"{CLASS}.{FUNC}: {PARAM} default is {got!r}, expected {want!r}")
    # the attribute assignment that consumes it must still be in place (reader branch): self.busy_loop_s = busy_loop_s
    if "            self.busy_loop_s = busy_loop_s\n" not in text:
        raise Refused("the reader-side `self.busy_loop_s = busy_loop_s` assignment is gone")
    # ... and wait() must still be the switch the value controls: sched_yield() while the last read is recent, else the poller
    if ("        if current_time <= self.last_read + self.busy_loop_s:\n"
            "            sched_yield()\n"
            "        else:\n") not in text:
        raise Refused("SpinCondition.wait() no longer switches between sched_yield() and the poller on busy_loop_s")


def apply(text: str) -> str:
    bad = [(name, text.count(old)) for name, old, _ in EDITS if text.count(old) != 1]
    if bad:
        raise Refused("; ".join(f"anchor '{n}' found {c}x (expected 1)" for n, c in bad))
    check_structure(text, patched=False)
    out = text
    for name, old, new in EDITS:
        if out.count(old) != 1:
            raise Refused(f"anchor '{name}' changed by an earlier edit")
        out = out.replace(old, new, 1)
    compile(out, "shm_broadcast.py", "exec")
    check_structure(out, patched=True)
    return out


def revert(text: str) -> str:
    out = text
    for name, old, new in reversed(EDITS):
        if out.count(new) != 1:
            raise Refused(f"patched block '{name}' found {out.count(new)}x (expected 1)")
        out = out.replace(new, old, 1)
    compile(out, "shm_broadcast.py", "exec")
    check_structure(out, patched=False)
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("apply", "check", "revert"):
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        print("usage: p001_apply.py {apply|check|revert} <shm_broadcast.py>", file=sys.stderr)
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
    tmp = p.with_name(p.name + ".p001.tmp")
    tmp.write_text(new, encoding="utf-8")
    os.chmod(tmp, os.stat(p).st_mode & 0o7777)
    os.replace(tmp, p)
    print(f" [OK] busy_loop_s 1 -> 0.002 (2ms grace before sleep)" if argv[1] == "apply"
          else " [OK] busy_loop_s 0.002 -> 1 (reverted)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
