#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch 001 gb10-spin-wait] CPU checks (any CPU: python3 >= 3.10, stdlib only; NO torch, NO vLLM import, no GPU).

  python3 -I -B tests/patch-checks/p001_cpu_check.py [--vllm-root <vllm package dir>]    (default: $VLLM_TREE, else the installed vllm package)

  A  pins + applier on the REAL nightly shm_broadcast.py: original == the .sh pin, apply() == the pinned result and differs
     in exactly one line, 3.12 grammar, revert() byte-exact, idempotence (a second apply is refused), vendored applier sha.
  B  refusals of the applier: anchor missing / duplicated / in the wrong class / default not a constant / reader-side
     assignment gone / wait() no longer switching on busy_loop_s -> Refused.
  C  the REAL .sh end to end in a scratch VLLM_ROOT: banner + [OK] lines (incl. the launch/readiness.txt `require` rule
     `busy_loop_s = 0\\.002`), result == pinned sha, second run = [skip] with the file untouched, unknown base / missing file
     / tampered applier -> non-zero exit with the file untouched; the [WARN] line for VLLM_USE_SPINLOOP_EXT=1.
  D  behaviour: the REAL patched SpinCondition (extracted by AST, run over stubs for zmq/time): a reader idle for > 2 ms goes
     to the poller (sleeps) instead of yielding; the unpatched class still yields for 1 s (the control); the nightly's new
     record_read() drain of the notify socket (every SHM_NOTIFY_POLL_INTERVAL_S) still works with the patched value.
  E  the reader path: the REAL MessageQueue._wait_until_readable (extracted by AST, run over stubs) reaches
     SpinCondition.wait(timeout_ms=...) when the block is not readable (the call whose grace the patch shortens); with the
     C spin-loop extension off (VLLM_USE_SPINLOOP_EXT default 0) it is the only wait.
Exit 0 = ALL PASS."""
from __future__ import annotations

import ast
import re
import shutil
import sys
import tempfile
import textwrap
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import psmall_lib as L  # noqa: E402

SH = L.PDIR / "001-gb10-spin-wait.sh"
APPLIER = L.FDIR / "p001_apply.py"
REL = "distributed/device_communicators/shm_broadcast.py"


def main(argv) -> int:
    args = L.parse_args(argv)
    vroot = Path(args.vllm_root)
    ap = L.load_module("p001_apply", APPLIER)
    orig_path = vroot / REL
    orig = orig_path.read_text()
    check = L.check

    # ---------------------------------------------------------------- A
    check("A1 nightly shm_broadcast.py == the SHM_ORIG pin", L.sha_file(orig_path) == L.pin(SH, "SHM_ORIG"))
    check("A2 vendored applier == the APPLIER_SHA pin", L.sha_file(APPLIER) == L.pin(SH, "APPLIER_SHA"))
    new = ap.apply(orig)
    check("A3 apply() == the SHM_P001 pin", L.sha_bytes(new.encode()) == L.pin(SH, "SHM_P001"), L.sha_bytes(new.encode())[:16])
    ol, nl = orig.splitlines(keepends=True), new.splitlines(keepends=True)
    diff = [(a, b) for a, b in zip(ol, nl) if a != b]
    check("A4 exactly one line differs: busy_loop_s: float = 1, -> 0.002,",
          len(ol) == len(nl) and diff == [("        busy_loop_s: float = 1,\n", "        busy_loop_s: float = 0.002,\n")], str(diff))
    check("A5 patched file compiles (also with the python 3.12 grammar)", L.compiles_312(new, REL))
    check("A6 revert(apply(x)) == x byte for byte", ap.revert(new) == orig)
    check("A7 a second apply() is refused (idempotence at the applier level)", L.raises(lambda: ap.apply(new), ap.Refused))
    check("A8 the old default is gone from the patched text and the new one occurs once",
          new.count("busy_loop_s: float = 0.002") == 1 and "busy_loop_s: float = 1," not in new)

    # ---------------------------------------------------------------- B
    a = ap.ANCHOR_OLD
    check("B1 anchor missing -> Refused", L.raises(lambda: ap.apply(orig.replace(a, "        busy_loop_s: float = 2,\n")), ap.Refused, "found 0x"))
    check("B2 anchor duplicated -> Refused", L.raises(lambda: ap.apply(orig + "\n# " + a), ap.Refused, "found 2x"))
    check("B3 class renamed -> Refused (AST)", L.raises(lambda: ap.apply(orig.replace("class SpinCondition:", "class SpinConditionX:")),
                                                       ap.Refused, "class SpinCondition found 0"))
    check("B4 default not a constant -> Refused", L.raises(lambda: ap.apply(orig.replace(a, "        busy_loop_s: float = float(1),\n")),
                                                          ap.Refused))
    check("B5 consumer assignment gone -> Refused", L.raises(
        lambda: ap.apply(orig.replace("            self.busy_loop_s = busy_loop_s\n", "            self.busy_loop_s = 1\n")), ap.Refused))
    check("B6 revert of an unpatched file is refused", L.raises(lambda: ap.revert(orig), ap.Refused))
    check("B7 wait() no longer switching on busy_loop_s -> Refused", L.raises(
        lambda: ap.apply(orig.replace("        if current_time <= self.last_read + self.busy_loop_s:\n",
                                      "        if current_time <= self.last_read + 1.0:\n")), ap.Refused, "wait\\(\\)"))

    # ---------------------------------------------------------------- C
    root = L.make_root(vroot, [REL])
    f = root / REL
    r1 = L.run_sh(SH, root)
    out = r1.stdout
    check("C1 .sh exit 0 on the nightly original", r1.returncode == 0 and "[WARN]" not in r1.stderr, r1.stderr[-300:])
    check("C2 banner / [OK] / verification lines",
          "=== Hotfix: GB10 vLLM IPC spin-wait (CPU load / SoC heat, issue #79) ===" in out
          and " [OK] busy_loop_s 1 -> 0.002 (2ms grace before sleep)" in out
          and "=== Verification ===" in out
          and "[OK] busy_loop_s = 0.002 — IPC wait sleeps after 2ms instead of spinning 1s." in out, out[-300:])
    ready = re.compile(r"busy_loop_s = 0\.002")
    check("C3 the launch/readiness.txt rule `require busy_loop_s = 0\\.002` matches", bool(ready.search(out)))
    check("C4 file == the pinned result", L.sha_file(f) == L.pin(SH, "SHM_P001"))
    before = f.read_bytes()
    r2 = L.run_sh(SH, root)
    check("C5 second run: [skip], exit 0, file untouched", r2.returncode == 0 and " [skip] busy_loop_s already set to 0.002" in r2.stdout
          and f.read_bytes() == before, r2.stdout[-200:])
    check("C6 the second run still prints the require-rule line", bool(ready.search(r2.stdout)))
    # unknown base
    root2 = L.make_root(vroot, [REL])
    f2 = root2 / REL
    f2.write_text(orig + "\n# local change\n")
    b2 = f2.read_bytes()
    r3 = L.run_sh(SH, root2)
    check("C7 unknown base -> exit 1, file untouched, no [OK] verification line",
          r3.returncode == 1 and f2.read_bytes() == b2 and "[OK] busy_loop_s = 0.002" not in r3.stdout, r3.stderr[-200:])
    # already carries the value but under another sha (e.g. a different build): refused, not skipped
    f2.write_text(orig.replace(a, ap.ANCHOR_NEW) + "\n# other build\n")
    b2 = f2.read_bytes()
    r4 = L.run_sh(SH, root2)
    check("C8 0.002 under an unpinned sha -> exit 1 (fail closed, not [skip])", r4.returncode == 1 and f2.read_bytes() == b2)
    # missing file
    root3 = Path(tempfile.mkdtemp(prefix="smallchk_"))
    r5 = L.run_sh(SH, root3)
    check("C9 missing shm_broadcast.py -> ERROR line, exit 1", r5.returncode == 1 and "ERROR: vLLM shm_broadcast.py not found" in r5.stderr)
    # tampered applier
    pdir = Path(tempfile.mkdtemp(prefix="smallchk_p_"))
    shutil.copy2(SH, pdir / SH.name)
    (pdir / "files").mkdir()
    (pdir / "files" / "p001_apply.py").write_text(APPLIER.read_text() + "\n# tampered\n")
    root4 = L.make_root(vroot, [REL])
    b4 = (root4 / REL).read_bytes()
    r6 = L.run_sh(pdir / SH.name, root4)
    check("C10 tampered applier -> exit 1 before any write", r6.returncode == 1 and (root4 / REL).read_bytes() == b4
          and "APPLIER" not in r6.stdout and "sha256" in r6.stderr, r6.stderr[-200:])
    root5 = L.make_root(vroot, [REL])
    r7 = L.run_sh(SH, root5, {"VLLM_USE_SPINLOOP_EXT": "1"})
    check("C11 VLLM_USE_SPINLOOP_EXT=1 -> [WARN] on stderr, still exit 0 and patched",
          r7.returncode == 0 and "[WARN] VLLM_USE_SPINLOOP_EXT='1'" in r7.stderr and L.sha_file(root5 / REL) == L.pin(SH, "SHM_P001"),
          r7.stderr[-200:])
    root6 = L.make_root(vroot, [REL])
    r8 = L.run_sh(SH, root6, {"VLLM_USE_SPINLOOP_EXT": "0"})
    check("C12 VLLM_USE_SPINLOOP_EXT=0 -> no [WARN]", r8.returncode == 0 and "[WARN]" not in r8.stderr)

    # ---------------------------------------------------------------- D
    def spin_class(text: str, label: str):
        tree = ast.parse(text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpinCondition")
        src = ast.get_source_segment(text, cls)
        const = next((n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                      and n.targets[0].id == "SHM_NOTIFY_POLL_INTERVAL_S"), None)
        clock = {"t": 1000.0}

        class FakeSocket:
            def __init__(self, kind):
                self.kind, self.opts, self.sent, self.recv_calls, self.poll_calls, self.pending = kind, {}, [], 0, [], False

            def setsockopt(self, k, v): self.opts[k] = v
            def setsockopt_string(self, k, v): self.opts[k] = v
            def connect(self, a): self.addr = a
            def bind(self, a): self.addr = a
            def send(self, b): self.sent.append(b)

            def poll(self, timeout=None):
                self.poll_calls.append(timeout)
                return self.pending

            def recv(self, flags=0, copy=True):
                self.recv_calls += 1
                self.pending = False

        class FakePoller:
            def __init__(self): self.polled, self.reg, self.ret = [], [], {}
            def register(self, s, ev): self.reg.append(s)
            def poll(self, timeout=None):
                self.polled.append(timeout)
                return dict(self.ret)

        zmq = types.SimpleNamespace(CONFLATE=1, SNDHWM=2, NOBLOCK=3, PAIR=4, Poller=FakePoller, POLLIN=5)
        ctx = types.SimpleNamespace(socket=lambda kind: FakeSocket(kind))
        yields = []
        ns = {"zmq": zmq, "SUB": "SUB", "PUB": "PUB", "SUBSCRIBE": "SUBSCRIBE", "time": types.SimpleNamespace(monotonic=lambda: clock["t"]),
              "sched_yield": lambda: yields.append(1), "get_open_zmq_inproc_path": lambda: "inproc://x",
              "logger": types.SimpleNamespace(debug=lambda *a, **k: None),
              "SHM_NOTIFY_POLL_INTERVAL_S": 2.0}
        if const is not None:
            exec(compile(ast.Module(body=[const], type_ignores=[]), "const", "exec"), ns)
        exec(compile(src, label, "exec"), ns)
        return ns["SpinCondition"], ctx, clock, yields

    for label, text, expect_poll in (("patched", new, True), ("unpatched control", orig, False)):
        cls, ctx, clock, yields = spin_class(text, label)
        sc = cls(True, ctx, "ipc://notify")
        check(f"D1 {label}: reader busy_loop_s", sc.busy_loop_s == (0.002 if expect_poll else 1), str(sc.busy_loop_s))
        clock["t"] += 0.010                       # idle for 10 ms since the last read
        sc.wait(timeout_ms=5)
        polled = bool(sc.poller.polled)
        check(f"D2 {label}: a reader idle for 10 ms {'sleeps in the poller' if expect_poll else 'still yields (spins)'}",
              polled == expect_poll and (len(yields) == (0 if expect_poll else 1)), f"polled={sc.poller.polled} yields={len(yields)}")
        sc.record_read()
        clock["t"] += 0.001                       # 1 ms since the last read: still inside the 2 ms grace
        yields.clear()
        sc.wait(timeout_ms=5)
        check(f"D3 {label}: a reader idle for 1 ms yields (inside the grace)", len(yields) == 1)
        w = cls(False, ctx, "ipc://notify")
        check(f"D4 {label}: the writer side is unaffected (busy_loop_s 0, publishes)", w.busy_loop_s == 0 and w.poller is None)
        # the nightly's record_read(): a reader that stays busy drains its notify socket every SHM_NOTIFY_POLL_INTERVAL_S
        sock = sc.local_notify_socket
        sock.poll_calls.clear()
        sock.pending = True
        clock["t"] += 0.5
        sc.record_read()
        check(f"D5 {label}: record_read() 0.5 s after the last drain does not touch the notify socket", sock.poll_calls == [])
        clock["t"] += 2.0
        sc.record_read()
        check(f"D6 {label}: record_read() 2 s after the last drain polls (timeout 0) and consumes the queued notification",
              sock.poll_calls == [0] and sock.recv_calls >= 1 and sock.pending is False, f"{sock.poll_calls} recv={sock.recv_calls}")
        # after 5 s without a read both classes are idle: wait() polls with the caller's timeout and consumes the notification
        sc.poller.ret = {sc.local_notify_socket: 5}
        sock.pending = True
        sock.recv_calls = 0
        clock["t"] += 5.0
        sc.wait(timeout_ms=1234)
        check(f"D7 {label}: idle for 5 s, wait() polls with the caller's timeout and receives the notification once",
              sc.poller.polled[-1] == 1234 and sock.recv_calls == 1, f"{sc.poller.polled[-1:]} recv={sock.recv_calls}")

    # ---------------------------------------------------------------- E
    tree = ast.parse(new)
    mq = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MessageQueue")
    fn = next(n for n in mq.body if isinstance(n, ast.FunctionDef) and n.name == "_wait_until_readable")
    src = textwrap.dedent(ast.get_source_segment(new, fn))
    trail = []

    def run_wait(spin_ext: bool, readable_after_spin: bool, flags):
        trail.clear()
        state = {"flags": list(flags)}

        def memory_fence():
            trail.append("fence")

        def spinloop(buf, check, timeout=None):
            trail.append(("spinloop", timeout))
            if readable_after_spin:
                state["flags"][0] = 1

        ns = {"memory_fence": memory_fence, "SPINLOOP_EXT_ENABLED": spin_ext, "spinloop": spinloop, "SPINLOOP_TIMEOUT_SECONDS": 0.1,
              "logger": types.SimpleNamespace(info=lambda *a, **k: trail.append("log")), "LONG_WAIT_TIME_LOG_MSG": "x",
              "VLLM_RINGBUFFER_WARNING_INTERVAL": 60, "ReadTimeoutWithWarnings": object}
        exec(compile(src, "_wait_until_readable", "exec"), ns)

        class Spin:
            def wait(self, timeout_ms=None):
                trail.append(("wait", timeout_ms))

        class RT:
            def timeout_ms(self):
                return 5000

            def should_warn(self):
                return False

        self_ = types.SimpleNamespace(local_reader_rank=0, _spin_condition=Spin(), shutting_down=False)
        buf = state["flags"]
        return ns["_wait_until_readable"](self_, buf, RT())

    r_not = run_wait(False, False, [0, 0])
    check("E1 block not readable, spin extension off: SpinCondition.wait(timeout_ms=5000) is the only wait, returns False",
          r_not is False and ("wait", 5000) in trail and not any(isinstance(t, tuple) and t[0] == "spinloop" for t in trail), str(trail))
    r_ok = run_wait(False, False, [1, 0])
    check("E2 block already readable: returns True without waiting", r_ok is True and not any(isinstance(t, tuple) for t in trail), str(trail))
    run_wait(True, False, [0, 0])
    check("E3 spin extension on (VLLM_USE_SPINLOOP_EXT=1): the C spin (100 ms) runs first, then the SpinCondition wait",
          [t for t in trail if isinstance(t, tuple)] == [("spinloop", 0.1), ("wait", 5000)], str(trail))

    for d in (root, root2, root3, root4, root5, root6, pdir):
        shutil.rmtree(d, ignore_errors=True)
    return L.finish("p001_cpu_check")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
