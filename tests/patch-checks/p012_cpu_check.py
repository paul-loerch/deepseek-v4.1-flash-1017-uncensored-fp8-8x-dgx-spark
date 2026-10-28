#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""[patch 012 gb10-oneshot-ar] CPU checks (any CPU: python3 >= 3.10 + numpy; the checks stub torch, triton and vLLM, no GPU).

  python3 -I -B tests/patch-checks/p012_cpu_check.py [--vllm-root <vllm package dir>]    (default: $VLLM_TREE, else the installed vllm package)

  A  pins + applier on the REAL nightly distributed/device_communicators/cuda_communicator.py: original == the .sh pin,
     apply() == the pinned result and adds exactly the 7 hook lines (2 markers), 3.12 grammar, revert() byte-exact, a second
     apply is refused, applier / module shas.
  B  refusals: an anchor missing / duplicated, the hooks displaced (AST), marker present -> Refused.
  C  the REAL .sh end to end in a scratch VLLM_ROOT: banner, module installed, [OK] lines, result == pinned sha, second run
     [skip], unknown base / stale edit / tampered applier or module -> exit 1 with the file untouched; the [WARN] line for
     VLLM_GB10_ONESHOT_AR=1 without VLLM_ALLREDUCE_USE_FLASHINFER=0 (stderr only, exit 0).
  D  the all-reduce PATH of v0.32.0 (the question 'does VLLM_ALLREDUCE_USE_FLASHINFER=0 keep the all-reduce on the path the
     hook sits on?'), on the REAL patched file: the `use_flashinfer_allreduce` expression of CudaCommunicator.__init__ run for
     the env values, FlashInferAllReduce constructed only under that flag, the REAL patched all_reduce() run over stubs: an
     eligible message returns from the hook before ANY backend (also with a FlashInfer communicator present), an ineligible
     one (or no hook object) falls through the chain to pynccl; with the flag 0 and no custom all-reduce the chain's last
     element is pynccl.all_reduce.
  E  the REAL module (vllm/_gb10_oneshot_ar.py) over a fake Triton (the REAL kernel source interpreted on numpy), a fake torch and
     a fake NCCL with grouped send/recv, 8 threaded ranks: maybe_create gating (env, group name tp:<n> exactly, disabled /
     single-rank comm, bad MAX_BYTES), the p2p schedule (group_start, send/recv to (r+d)%W / (r-d)%W, group_end), the result
     == a numpy fp32 rank-order sum with ONE bf16 rounding, bit-identical on all ranks, for sizes 1..max incl. +-0 / inf / NaN /
     subnormal inputs, should_use() rules, a wrong warm-up result -> RuntimeError (fail closed).
  F  reachability + sizes on the REAL nightly files (text level): every TP all-reduce of the DeepSeek-V4.1 stock path
     (embedding, RowParallelLinear, MoE runner) goes tensor_model_parallel_all_reduce -> GroupCoordinator.all_reduce ->
     CudaCommunicator.all_reduce (where the hook sits); the model's fused mHC all-reduce is limited to TP=4; the decode
     message sizes at hidden size 5120 (one request with 5 DSpark drafts = 6 tokens = 60 KiB) against the 64 KiB limit.
Exit 0 = ALL PASS."""
from __future__ import annotations

import ast
import re
import shutil
import sys
import tempfile
import threading
import types
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)      # numpy inf/NaN/overflow warnings of the special-value rows
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fake_torch_testlib as T  # noqa: E402  (numpy-backed fake torch; read-only)
import psmall_lib as L  # noqa: E402

SH = L.PDIR / "012-gb10-oneshot-allreduce.sh"
APPLIER = L.FDIR / "p012_apply.py"
MODULE = L.FDIR / "gb10_oneshot_ar.py"
REL = "distributed/device_communicators/cuda_communicator.py"


# ---------------------------------------------------------------------------------------------------------------------
# fake triton: the REAL kernel source is interpreted over numpy (one python call per program id)
# ---------------------------------------------------------------------------------------------------------------------
class Val(np.ndarray):
    """A loaded block. bf16 data stays as uint16 bit patterns until .to(tl.float32)."""

    def to(self, dt):
        a = np.asarray(self)
        if dt is TL.float32:
            return (T.bf16_bits_to_f32(a) if a.dtype == np.uint16 else a.astype(np.float32)).view(Val)
        if dt is TL.bfloat16:
            return T.f32_to_bf16_bits(a.astype(np.float32)).view(Val)
        raise TypeError(dt)


class Ptr:
    def __init__(self, arr, off):
        self.arr, self.off = arr, off

    def __add__(self, o):
        return Ptr(self.arr, self.off + (np.asarray(o) if isinstance(o, Val) else o))

    __radd__ = __add__


class _TL(types.ModuleType):
    float32, bfloat16 = "float32", "bfloat16"
    constexpr = object

    def __init__(self):
        super().__init__("triton.language")
        self._tls = threading.local()

    def program_id(self, axis):
        return self._tls.pid[axis]

    def arange(self, a, b):
        return np.arange(a, b, dtype=np.int64).view(Val)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=np.float32).view(Val)

    def static_range(self, n):
        return range(int(n))

    def load(self, ptr, mask=None, other=0.0, **kw):
        idx = np.asarray(ptr.off)
        m = np.ones(idx.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        m = np.broadcast_to(m, idx.shape)
        safe = np.where(m, idx, 0)
        out = ptr.arr[safe]
        fill = 0 if ptr.arr.dtype == np.uint16 and other == 0.0 else other
        return np.where(m, out, fill).astype(ptr.arr.dtype).view(Val)

    def store(self, ptr, val, mask=None, **kw):
        idx = np.asarray(ptr.off)
        m = np.ones(idx.shape, dtype=bool) if mask is None else np.broadcast_to(np.asarray(mask, dtype=bool), idx.shape)
        ptr.arr[idx[m]] = np.asarray(val)[m]


TL = _TL()


class FakeTriton(types.ModuleType):
    language = TL

    def __init__(self):
        super().__init__("triton")
        self.language = TL
        self.launches = []

    @staticmethod
    def cdiv(a, b):
        return (int(a) + int(b) - 1) // int(b)

    def jit(self, fn):
        outer = self

        class Jit:
            def __getitem__(self, grid):
                def launch(*args, **kw):
                    conv = [Ptr(a._a.reshape(-1), 0) if isinstance(a, T.Tensor) else a for a in args]
                    outer.launches.append(tuple(grid))
                    for pid in range(int(grid[0])):
                        TL._tls.pid = (pid, 0, 0)
                        fn(*conv, **kw)
                return launch
        return Jit()


# ---------------------------------------------------------------------------------------------------------------------
# fake NCCL: grouped send/recv with the matching semantics of ncclGroupStart/End
# ---------------------------------------------------------------------------------------------------------------------
class FakeComm:
    def __init__(self, rv, rank, ws, *, disabled=False, drop_all=False):
        self.rv, self.rank, self.world_size, self.disabled = rv, rank, ws, disabled
        self.seq, self.log, self.sends, self.recvs, self.in_group = 0, [], [], [], False
        self.drop_all = drop_all

    def group_start(self):
        assert not self.in_group
        self.in_group, self.sends, self.recvs = True, [], []
        self.log.append(("group_start",))

    def send(self, t, dst, stream=None):
        assert self.in_group and dst != self.rank
        self.sends.append((dst, t._a.copy()))
        self.log.append(("send", dst))

    def recv(self, t, src, stream=None):
        assert self.in_group and src != self.rank
        self.recvs.append((src, t))
        self.log.append(("recv", src))

    def group_end(self):
        assert self.in_group
        self.in_group = False
        vals = self.rv.exchange(self.rank, self.seq, ("p2p-group",), list(self.sends))
        self.seq += 1
        taken = {}
        for src, t in self.recvs:
            mine = [d for (dst, d) in vals[src] if dst == self.rank]
            k = taken.get(src, 0)
            taken[src] = k + 1
            t._a[...] = 0 if self.drop_all else mine[k]
        self.log.append(("group_end",))


def install_fakes():
    """fake torch + triton + the vLLM names the module imports. Returns the module under test (fresh)."""
    for k in [k for k in sys.modules if k in ("torch", "triton", "triton.language") or k == "vllm" or k.startswith("vllm.")]:
        del sys.modules[k]
    tm = T.make_fake_torch()
    tm.all = lambda t: t.all()
    tm.cuda.synchronize = T._cuda_sync
    sys.modules["torch"] = tm
    ftr = FakeTriton()
    sys.modules["triton"], sys.modules["triton.language"] = ftr, TL
    vl = types.ModuleType("vllm")
    vl.__path__ = []
    sys.modules["vllm"] = vl

    class _Log:
        def __init__(self):
            self.lines = []

        def info(self, fmt, *a):
            self.lines.append(fmt % a if a else fmt)

        warning = error = info

    lg = _Log()
    for name, attrs in (("vllm.logger", {"init_logger": lambda n: lg}), ("vllm.utils", {}),
                        ("vllm.utils.torch_utils", {"current_stream": lambda: object()})):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        if name in ("vllm.utils",):
            m.__path__ = []
        sys.modules[name] = m
    mod = L.load_module("vllm._gb10_oneshot_ar", MODULE)
    return mod, tm, ftr, lg


def bf16_rows(rng, n, kind="normal"):
    """[n] float32 values that are exactly representable in bf16 (so inputs and expectations share the bit patterns)."""
    if kind == "normal":
        v = rng.standard_normal(n).astype(np.float32) * 3
    elif kind == "wide":                               # magnitudes over 30 binades: bf16-vs-fp32 accumulation differs
        v = (rng.standard_normal(n) * np.exp2(rng.integers(-12, 18, n))).astype(np.float32)
    else:
        v = np.zeros(n, np.float32)
    return T.bf16_bits_to_f32(T.f32_to_bf16_bits(v))


def expected_oneshot(per_rank_bits):
    acc = np.zeros(per_rank_bits[0].shape, dtype=np.float32)
    for bits in per_rank_bits:
        acc = (acc + T.bf16_bits_to_f32(bits)).astype(np.float32)
    return T.f32_to_bf16_bits(acc)


def main(argv) -> int:
    args = L.parse_args(argv)
    vroot = Path(args.vllm_root)
    ap = L.load_module("p012_apply", APPLIER)
    orig = (vroot / REL).read_text()
    check = L.check

    # ---------------------------------------------------------------- A
    check("A1 nightly cuda_communicator.py == the COMM_ORIG pin", L.sha_file(vroot / REL) == L.pin(SH, "COMM_ORIG"))
    check("A2 applier == APPLIER_SHA, module == MOD_SHA", L.sha_file(APPLIER) == L.pin(SH, "APPLIER_SHA")
          and L.sha_file(MODULE) == L.pin(SH, "MOD_SHA"))
    new = ap.apply(orig)
    check("A3 apply() == the COMM_P012 pin", L.sha_bytes(new.encode()) == L.pin(SH, "COMM_P012"), L.sha_bytes(new.encode())[:16])
    ol, nl = orig.splitlines(), new.splitlines()
    check("A4 exactly 7 lines added (3 in __init__, 4 in all_reduce), 2 markers", len(nl) - len(ol) == 7 and new.count(ap.MARKER) == 2)
    check("A5 patched file compiles (also with the python 3.12 grammar)", L.compiles_312(new, REL))
    check("A6 revert(apply(x)) == x byte for byte", ap.revert(new) == orig)
    check("A7 a second apply() is refused", L.raises(lambda: ap.apply(new), ap.Refused))
    check("A8 the two anchors occur exactly once in the nightly", orig.count(ap.INIT_OLD) == 1 and orig.count(ap.AR_OLD) == 1)

    # ---------------------------------------------------------------- B
    check("B1 init anchor missing -> Refused", L.raises(lambda: ap.apply(orig.replace(ap.INIT_OLD, "")), ap.Refused, "found 0x"))
    check("B2 all_reduce anchor duplicated -> Refused", L.raises(lambda: ap.apply(orig + "\n" + ap.AR_OLD), ap.Refused, "found 2x"))
    check("B3 all_reduce signature changed -> Refused", L.raises(
        lambda: ap.apply(orig.replace(ap.AR_OLD, "    def all_reduce(self, input_, group=None):\n")), ap.Refused))
    check("B4 class CudaCommunicator renamed -> Refused (AST)", L.raises(
        lambda: ap.apply(orig.replace("class CudaCommunicator(", "class CudaCommunicatorX(")), ap.Refused))
    check("B5 marker present -> Refused", L.raises(lambda: ap.apply(orig + "\n" + ap.MARKER + "\n"), ap.Refused, "marker"))
    swapped = new.replace("        _gb10_os = getattr(self, '_gb10_oneshot', None)\n", "        pass\n        _gb10_os = getattr(self, '_gb10_oneshot', None)\n")
    check("B6 dispatch not among the first statements -> structure check fails", L.raises(
        lambda: ap.check_structure(swapped, patched=True), ap.Refused))
    nopynccl = orig.replace("            self.pynccl_comm = PyNcclCommunicator(", "            self.other_comm = PyNcclCommunicator(")
    check("B7 no pynccl_comm block -> Refused", L.raises(lambda: ap.apply(nopynccl), ap.Refused))
    check("B8 revert of an unpatched file is refused", L.raises(lambda: ap.revert(orig), ap.Refused))

    # ---------------------------------------------------------------- C
    roots = []

    def fresh(text=None):
        r = L.make_root(vroot, [REL])
        roots.append(r)
        if text is not None:
            (r / REL).write_text(text)
        return r

    root = fresh()
    r1 = L.run_sh(SH, root, {"VLLM_GB10_ONESHOT_AR": "1", "VLLM_ALLREDUCE_USE_FLASHINFER": "0"})
    out = r1.stdout
    check("C1 .sh exit 0, no [WARN] with VLLM_ALLREDUCE_USE_FLASHINFER=0", r1.returncode == 0 and "[WARN]" not in r1.stderr, r1.stderr[-300:])
    check("C2 banner (gate shown), module install, wiring, verification, final [OK] line",
          "=== Patch 012: GB10 one-shot all-reduce for small TP messages (opt-in VLLM_GB10_ONESHOT_AR=1; now: 1) ===" in out
          and " [OK] installed _gb10_oneshot_ar.py (sha256 " in out
          and " [OK] wired maybe_create into CudaCommunicator.__init__ and the one-shot dispatch into all_reduce" in out
          and "=== Verification ===" in out and "[OK] patch 012 in place (active only when VLLM_GB10_ONESHOT_AR=1)" in out, out[-400:])
    check("C3 file == the pinned result, module installed byte-identical",
          L.sha_file(root / REL) == L.pin(SH, "COMM_P012") and L.sha_file(root / "_gb10_oneshot_ar.py") == L.pin(SH, "MOD_SHA"))
    before = (root / REL).read_bytes()
    r2 = L.run_sh(SH, root)
    check("C4 second run: [skip], exit 0, file untouched", r2.returncode == 0 and " [skip] cuda_communicator.py already wired" in r2.stdout
          and (root / REL).read_bytes() == before)
    root2 = fresh()
    r3 = L.run_sh(SH, root2, {"VLLM_GB10_ONESHOT_AR": "1"})
    check("C5 VLLM_GB10_ONESHOT_AR=1 without VLLM_ALLREDUCE_USE_FLASHINFER=0 -> [WARN] on stderr, still exit 0",
          r3.returncode == 0 and "[WARN] VLLM_GB10_ONESHOT_AR=1 but VLLM_ALLREDUCE_USE_FLASHINFER=" in r3.stderr, r3.stderr[-300:])
    root3 = fresh(orig + "\n# local change\n")
    b3 = (root3 / REL).read_bytes()
    r4 = L.run_sh(SH, root3)
    check("C6 unknown base -> exit 1, file untouched, module not installed", r4.returncode == 1 and (root3 / REL).read_bytes() == b3
          and not (root3 / "_gb10_oneshot_ar.py").exists())
    root4 = fresh(new + "\n# edited after patching\n")
    b4 = (root4 / REL).read_bytes()
    r5 = L.run_sh(SH, root4)
    check("C7 stale patch-012 edit (marker, unpinned sha) -> exit 1 'recreate the container'",
          r5.returncode == 1 and (root4 / REL).read_bytes() == b4 and "recreate the container" in r5.stderr)
    for what, fn in (("applier", "p012_apply.py"), ("module", "gb10_oneshot_ar.py")):
        pdir = Path(tempfile.mkdtemp(prefix="smallchk_p_"))
        roots.append(pdir)
        shutil.copy2(SH, pdir / SH.name)
        (pdir / "files").mkdir()
        for f in ("p012_apply.py", "gb10_oneshot_ar.py"):
            shutil.copy2(L.FDIR / f, pdir / "files" / f)
        (pdir / "files" / fn).write_text((L.FDIR / fn).read_text() + "\n# tampered\n")
        rr = fresh()
        bb = (rr / REL).read_bytes()
        r6 = L.run_sh(pdir / SH.name, rr)
        check(f"C8 tampered {what} -> exit 1 before any write", r6.returncode == 1 and (rr / REL).read_bytes() == bb
              and not (rr / "_gb10_oneshot_ar.py").exists(), r6.stderr[-160:])

    # ---------------------------------------------------------------- D
    tree = ast.parse(new)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CudaCommunicator")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    arfn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "all_reduce")
    # D1: the flag expression of the tp branch, run for the env values
    sel = next(s for s in init.body if isinstance(s, ast.If) and "unique_name.split" in ast.unparse(s.test))
    check("D1 the group gate is the exact-name test unique_name.split(':')[0] != 'tp'", ast.unparse(sel.test) == "unique_name.split(':')[0] != 'tp'")

    def flag_value(env_fi: bool, unique_name: str, batch_invariant=False):
        ns = {"unique_name": unique_name,
              "envs": types.SimpleNamespace(VLLM_ALLREDUCE_USE_FLASHINFER=env_fi, VLLM_BATCH_INVARIANT=batch_invariant,
                                            VLLM_ALLREDUCE_USE_SYMM_MEM=True, VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=False),
              "rocm_aiter_ops": types.SimpleNamespace(is_custom_all_reduce_enabled=lambda: False)}
        stub = types.ModuleType("vllm.distributed.parallel_state")
        stub._ENABLE_CUSTOM_ALL_REDUCE = False
        saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.distributed", "vllm.distributed.parallel_state")}
        for k in ("vllm", "vllm.distributed"):
            m = types.ModuleType(k)
            m.__path__ = []
            sys.modules[k] = m
        sys.modules["vllm.distributed.parallel_state"] = stub
        try:
            exec(compile(ast.unparse(ast.Module(body=[sel], type_ignores=[])), "tp-branch", "exec"), ns)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        return ns["use_flashinfer_allreduce"], ns["use_custom_allreduce"]

    check("D2 tp group + VLLM_ALLREDUCE_USE_FLASHINFER=0 -> use_flashinfer_allreduce False", flag_value(False, "tp:0")[0] is False)
    check("D3 tp group + VLLM_ALLREDUCE_USE_FLASHINFER=1 (the default) -> True (so the recipe must set 0)", flag_value(True, "tp:0")[0] is True)
    check("D4 a non-tp group never enables FlashInfer all-reduce", flag_value(True, "pp:0")[0] is False and flag_value(True, "dp:0")[0] is False)
    fi_assigns = [n for n in ast.walk(init) if isinstance(n, ast.Assign) and any(ast.unparse(t) == "self.fi_ar_comm" for t in n.targets)
                  and "FlashInferAllReduce(" in ast.unparse(n.value)]
    guard = [n for n in ast.walk(init) if isinstance(n, ast.If) and any(a is fi_assigns[0] for a in n.body)] if fi_assigns else []
    check("D5 FlashInferAllReduce is constructed exactly once, only under `if self.use_flashinfer_allreduce and self.world_size > 1`",
          len(fi_assigns) == 1 and len(guard) == 1 and ast.unparse(guard[0].test) == "self.use_flashinfer_allreduce and self.world_size > 1")

    def run_all_reduce(self_attrs: dict, tensor):
        """The REAL patched CudaCommunicator.all_reduce over stubs. Returns (result, trail)."""
        trail = []
        src = ast.unparse(ast.Module(body=[arfn], type_ignores=[]))
        ns = {
            "torch": types.SimpleNamespace(ops=types.SimpleNamespace(vllm=types.SimpleNamespace(
                all_reduce_symmetric_with_copy=lambda t: trail.append("symm_mem_copy"))), distributed=None),
            "envs": types.SimpleNamespace(VLLM_BATCH_INVARIANT=False),
            "current_platform": types.SimpleNamespace(is_rocm=lambda: False),
            "should_nccl_symm_mem_allreduce": lambda ws, t: (trail.append("symm_mem_gate") or False),
        }
        exec(compile(src, "all_reduce(patched)", "exec"), ns)
        base = dict(fi_ar_comm=None, pynccl_comm=types.SimpleNamespace(world_size=8, disabled=False,
                                                                           all_reduce=lambda t: (trail.append("pynccl") or "pynccl")),
                    qr_comm=None, fi_pcie_ipc_ar_comm=None, use_aiter_allreduce=False, aiter_ar_comm=None, ca_comm=None,
                    symm_mem_comm=None, device_group=None)
        base.update(self_attrs)
        obj = types.SimpleNamespace(**base)
        return ns["all_reduce"](obj, tensor), trail

    class Fi:
        disabled = False

        def should_use_fi_ar(self, t):
            return True

        def all_reduce(self, t):
            return "flashinfer"

    class OS:
        def __init__(self, ok):
            self.ok = ok

        def should_use(self, t):
            return self.ok

        def __call__(self, t):
            return "oneshot"

    x = "tensor"
    check("D6 eligible message: returns the one-shot BEFORE any backend (flag 0)", run_all_reduce({"_gb10_oneshot": OS(True)}, x) == ("oneshot", []))
    check("D7 eligible message with a FlashInfer communicator present (flag 1): the hook still wins",
          run_all_reduce({"_gb10_oneshot": OS(True), "fi_ar_comm": Fi()}, x) == ("oneshot", []))
    check("D8 ineligible message, flag 0 (fi_ar_comm None), no custom all-reduce: the chain ends in pynccl",
          run_all_reduce({"_gb10_oneshot": OS(False)}, x) == ("pynccl", ["symm_mem_gate", "pynccl"]))
    check("D9 ineligible message, flag 1: the chain goes to FlashInfer (why the recipe sets the flag to 0)",
          run_all_reduce({"_gb10_oneshot": OS(False), "fi_ar_comm": Fi()}, x)[0] == "flashinfer")
    check("D10 no _gb10_oneshot attribute (object built before the patch): untouched chain", run_all_reduce({}, x)[0] == "pynccl")
    check("D11 _gb10_oneshot None (gate unset): untouched chain", run_all_reduce({"_gb10_oneshot": None}, x)[0] == "pynccl")

    # init hook run for real: creation args
    hook = [s for s in init.body if isinstance(s, (ast.ImportFrom, ast.Assign)) and "_gb10_os_create" in ast.unparse(s)]
    calls = []
    fake_mod = types.ModuleType("vllm._gb10_oneshot_ar")
    fake_mod.maybe_create = lambda comm, dev, name: (calls.append((comm, dev, name)) or "OBJ")
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm._gb10_oneshot_ar")}
    pk = types.ModuleType("vllm")
    pk.__path__ = []
    sys.modules["vllm"], sys.modules["vllm._gb10_oneshot_ar"] = pk, fake_mod
    pk._gb10_oneshot_ar = fake_mod
    try:
        obj = types.SimpleNamespace(pynccl_comm="COMM", device="DEV", unique_name="tp:0")
        exec(compile("def f(self):\n" + "\n".join("    " + ast.unparse(s) for s in hook) + "\n", "init-hook", "exec"), {})
        ns = {}
        exec(compile("def f(self):\n" + "\n".join("    " + ast.unparse(s).replace("\n", "\n    ") for s in hook) + "\n", "init-hook", "exec"), ns)
        ns["f"](obj)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    check("D12 the __init__ hook calls maybe_create(self.pynccl_comm, self.device, self.unique_name) and stores it",
          calls == [("COMM", "DEV", "tp:0")] and obj._gb10_oneshot == "OBJ", str(calls))

    # ---------------------------------------------------------------- E
    import os
    rng = np.random.default_rng(12)
    mod, tm, ftr, lg = install_fakes()
    os.environ.pop("VLLM_GB10_ONESHOT_AR", None)
    os.environ.pop("VLLM_GB10_ONESHOT_AR_MAX_BYTES", None)
    rv0 = T.Rendezvous(2)
    dev = T.device("cuda:0")
    comm = FakeComm(rv0, 0, 8)
    check("E1 gate unset -> maybe_create is None (nothing constructed)", mod.maybe_create(comm, dev, "tp:0") is None)
    os.environ["VLLM_GB10_ONESHOT_AR"] = "0"
    check("E2 gate '0' -> None", mod.maybe_create(comm, dev, "tp:0") is None)
    os.environ["VLLM_GB10_ONESHOT_AR"] = "1"
    check("E3 pynccl None / disabled / single rank -> None", mod.maybe_create(None, dev, "tp:0") is None
          and mod.maybe_create(FakeComm(rv0, 0, 8, disabled=True), dev, "tp:0") is None
          and mod.maybe_create(FakeComm(rv0, 0, 1), dev, "tp:0") is None)
    check("E4 only the group named exactly tp:<n> (pp:0, dp:0, etp:0, mtp:0, '', None -> None)",
          all(mod.maybe_create(comm, dev, n) is None for n in ("pp:0", "dp:0", "etp:0", "mtp:0", "", None)))
    for bad in ("0", "-2", "7", "x"):
        os.environ["VLLM_GB10_ONESHOT_AR_MAX_BYTES"] = bad
        check(f"E5 MAX_BYTES={bad!r} -> ValueError/refused", L.raises(lambda: mod.maybe_create(comm, dev, "tp:0"), ValueError))
    os.environ.pop("VLLM_GB10_ONESHOT_AR_MAX_BYTES", None)

    def run8(max_bytes="65536", drop_ranks=(), fn=None):
        os.environ["VLLM_GB10_ONESHOT_AR_MAX_BYTES"] = max_bytes
        rv = T.Rendezvous(8, timeout=20.0)
        comms = [FakeComm(rv, r, 8, drop_all=(r in drop_ranks)) for r in range(8)]
        ops = [None] * 8

        def body(r):
            ops[r] = mod.maybe_create(comms[r], dev, "tp:0")
            return fn(r, ops[r], comms[r]) if fn else None
        res, err = T.run_ranks(8, body, timeout=60)
        return ops, comms, res, err, rv

    ops, comms, res, err, rv = run8()
    check("E6 8 threaded ranks construct the op (collective self-checking warm-up passes)", all(e is None for e in err) and all(o is not None for o in ops),
          str([e[1][-200:] for e in err if e]))
    exp_log = [("group_start",)] + [x for d in range(1, 8) for x in (("send", (0 + d) % 8), ("recv", (0 - d) % 8))] + [("group_end",)]
    check("E7 rank 0 p2p schedule per call: group_start, (send (r+d)%8, recv (r-d)%8) for d=1..7, group_end; 3 warm-up calls",
          comms[0].log == exp_log * 3, str(comms[0].log[:6]))
    check("E8 the info line names the group and the limit", any("one-shot all-reduce ACTIVE for group tp:0 (world 8, bf16 <= 65536 bytes" in s
                                                              for s in lg.lines), str(lg.lines[-1:]))
    op0 = ops[0]
    xs_ok = T.Tensor(np.zeros(4, np.uint16), T.bfloat16, dev)
    check("E9 should_use: bf16 contiguous 0 < n <= 32768 only",
          op0.should_use(xs_ok) and not op0.should_use(T.Tensor(np.zeros(0, np.uint16), T.bfloat16, dev))
          and not op0.should_use(T.Tensor(np.zeros(32769, np.uint16), T.bfloat16, dev))
          and op0.should_use(T.Tensor(np.zeros(32768, np.uint16), T.bfloat16, dev))
          and not op0.should_use(T.Tensor(np.zeros(4, np.float32), T.float32, dev))
          and not op0.should_use(T.Tensor(np.zeros((4, 4), np.uint16)[:, ::2], T.bfloat16, dev)))

    sizes = (1, 7, 100, 4096, 12288, 32768)
    kinds = ("normal", "wide")
    cases = []
    for n in sizes:
        for kind in kinds:
            per_rank = [T.f32_to_bf16_bits(bf16_rows(rng, n, kind)) for _ in range(8)]
            cases.append((n, kind, per_rank))
    special = np.zeros((8, 16), np.float32)
    special[:, 0] = -0.0                                   # all ranks -0.0
    special[0, 1], special[1, 1] = np.inf, -np.inf          # inf - inf = NaN
    special[2, 2] = np.nan
    special[:, 3] = 1e-39                                  # fp32-subnormal bf16? (bf16 min subnormal ~9.2e-41; 1e-39 is bf16 subnormal range)
    special[3, 4], special[4, 4] = 3.0e38, 3.0e38          # fp32 sum 6e38 overflows bf16 max -> inf
    special[:, 5] = 1.0
    special[:, 6] = np.arange(8, dtype=np.float32) - 3.5   # exact small values
    cases.append((16, "special", [T.f32_to_bf16_bits(special[r]) for r in range(8)]))

    def fn(r, op, comm):
        outs = []
        for (n, kind, per_rank) in cases:
            x = T.Tensor(per_rank[r].copy(), T.bfloat16, dev)
            outs.append(op(x)._a.copy())
        return outs
    ops, comms, res, err, rv = run8(fn=fn)
    check("E10 the per-size runs finished on all ranks without error/hang", all(e is None for e in err) and not rv.hangs,
          str([e[1][-300:] for e in err if e]))
    if all(e is None for e in err):
        for ci, (n, kind, per_rank) in enumerate(cases):
            want = expected_oneshot(per_rank)
            got = [res[r][ci] for r in range(8)]
            same = all(np.array_equal(got[0], g) for g in got)
            exact = all(np.array_equal(want, g) for g in got)
            check(f"E11 n={n:<6d} {kind:<8s}: all 8 ranks bit-identical and == numpy fp32 rank-order sum, one bf16 rounding",
                  same and exact, "" if exact else f"mismatch elems {int((got[0] != want).sum())}")
        # the property that matters: differs from naive per-hop bf16 accumulation on wide-range data (documented numerics change)
        n, kind, per_rank = next(c for c in cases if c[1] == "wide" and c[0] == 32768)
        acc = np.zeros(n, np.uint16)
        for bits in per_rank:
            acc = T.f32_to_bf16_bits(T.bf16_bits_to_f32(acc) + T.bf16_bits_to_f32(bits))
        want = expected_oneshot(per_rank)
        exactsum = sum(T.bf16_bits_to_f32(b).astype(np.float64) for b in per_rank)
        e_one = np.abs(T.bf16_bits_to_f32(want).astype(np.float64) - exactsum)
        e_hop = np.abs(T.bf16_bits_to_f32(acc).astype(np.float64) - exactsum)
        check("E12 documented numerics: one rounding is at least as close to the exact sum as per-hop bf16 adds (sum of |err|)",
              e_one.sum() <= e_hop.sum() and not np.array_equal(want, acc), f"{e_one.sum():.4g} vs {e_hop.sum():.4g}")
        sp = res[0][-1]
        spw = expected_oneshot([T.f32_to_bf16_bits(special[r]) for r in range(8)])
        spf = T.bf16_bits_to_f32(sp)
        check("E13 special values == the numpy fp32 rank-order sum (bit for bit, NaN payload included)", np.array_equal(sp, spw))
        check("E13b ... in detail: all -0.0 inputs -> +0.0 bits; inf + -inf -> NaN; a NaN input -> NaN; 6e38 -> +inf; 8 x 1.0 -> 8.0; "
              "the exact alternating sum -> +0.0",
              sp[0] == 0x0000 and np.isnan(spf[1]) and np.isnan(spf[2]) and np.isposinf(spf[4]) and spf[5] == 8.0 and sp[6] == 0x0000,
              str([hex(int(v)) for v in sp[:8]]))
    # fail closed: a communicator that drops all data on rank 3 -> warm-up result wrong there -> RuntimeError on that rank
    ops, comms, res, err, rv = run8(drop_ranks=(3,))
    check("E14 a rank whose receives deliver zeros fails its warm-up with RuntimeError (fail closed), the others finish or time out",
          err[3] is not None and isinstance(err[3][0], RuntimeError) and "warm-up wrong on rank 3" in str(err[3][0]), str(err[3][1][-200:]) if err[3] else "no error")
    # MAX_BYTES honoured
    ops, comms, res, err, rv = run8(max_bytes="16384")
    check("E15 VLLM_GB10_ONESHOT_AR_MAX_BYTES=16384 -> max_elems 8192, should_use limit follows",
          all(e is None for e in err) and ops[0].max_elems == 8192
          and ops[0].should_use(T.Tensor(np.zeros(8192, np.uint16), T.bfloat16, dev)) and not ops[0].should_use(T.Tensor(np.zeros(8193, np.uint16), T.bfloat16, dev)))

    # DeepSeek-V4.1 decode shapes through the REAL module: [tokens, 5120] bf16, 1..6 tokens take the one-shot at the default
    # limit (6 x 5120 x 2 = 61,440 B), 7 and 12 tokens do not; with MAX_BYTES=131072 12 tokens do; results bit-identical on 8 ranks
    HID = 5120
    ops, comms, res, err, rv = run8()
    check("E16 default limit: should_use for [tokens, 5120] bf16 -> tokens 1..6 yes, 7 and 12 no (61,440 B vs 71,680 B / 122,880 B)",
          all(e is None for e in err) and all(
              ops[0].should_use(T.Tensor(np.zeros((t, HID), np.uint16), T.bfloat16, dev)) == (t <= 6) for t in (1, 2, 3, 4, 5, 6, 7, 8, 12, 48)))
    shape_cases = [(t, [T.f32_to_bf16_bits(bf16_rows(rng, t * HID, "wide")) for _ in range(8)]) for t in (1, 5, 6)]

    def fn_shape(r, op, comm):
        outs = []
        for t, per_rank in shape_cases:
            x = T.Tensor(per_rank[r].reshape(t, HID).copy(), T.bfloat16, dev)
            outs.append(op(x)._a.copy())
        return outs
    ops, comms, res, err, rv = run8(fn=fn_shape)
    ok16 = all(e is None for e in err) and not rv.hangs
    if ok16:
        for ci, (t, per_rank) in enumerate(shape_cases):
            want = expected_oneshot(per_rank).reshape(t, HID)
            got = [res[r][ci] for r in range(8)]
            check(f"E17 [{t}, {HID}] bf16 ({t * HID * 2} B): all 8 ranks bit-identical and == numpy fp32 rank-order sum, one rounding",
                  all(np.array_equal(want, g) for g in got))
    else:
        check("E17 [tokens, 5120] runs finished", False, str([e[1][-300:] for e in err if e]))
    ops, comms, res, err, rv = run8(max_bytes="131072")
    check("E18 VLLM_GB10_ONESHOT_AR_MAX_BYTES=131072 -> [12, 5120] (122,880 B) is eligible, [13, 5120] is not",
          all(e is None for e in err) and ops[0].should_use(T.Tensor(np.zeros((12, HID), np.uint16), T.bfloat16, dev))
          and not ops[0].should_use(T.Tensor(np.zeros((13, HID), np.uint16), T.bfloat16, dev)))

    # ---------------------------------------------------------------- F
    def text_of(rel: str):
        p = vroot / rel
        return p.read_text() if p.is_file() else None

    ps = text_of("distributed/parallel_state.py")
    if ps is None:
        L.skip("F1 GroupCoordinator.all_reduce -> _all_reduce_out_place -> device_communicator.all_reduce", "file not in this tree")
    else:
        check("F1 GroupCoordinator.all_reduce -> _all_reduce_out_place -> device_communicator.all_reduce",
              "            return self._all_reduce_out_place(input_)" in ps
              and "    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:\n"
                  "        if self.device_communicator is None:\n"
                  '            raise ValueError("No device communicator found")\n'
                  "        return self.device_communicator.all_reduce(input_)" in ps)
    comm_txt = text_of("distributed/communication_op.py")
    if comm_txt is not None:
        check("F2 tensor_model_parallel_all_reduce = get_tp_group().all_reduce(input_)",
              "def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:" in comm_txt
              and "return get_tp_group().all_reduce(input_)" in comm_txt)
    lin = text_of("model_executor/layers/linear.py")
    if lin is not None:
        check("F3 RowParallelLinear: output = tensor_model_parallel_all_reduce(output_parallel) under reduce_results and tp_size > 1",
              "if self.reduce_results and self.tp_size > 1:\n            output = tensor_model_parallel_all_reduce(output_parallel)" in lin)
    moe = text_of("model_executor/layers/fused_moe/runner/moe_runner.py")
    if moe is not None:
        check("F4 MoE runner reduces its combined output with tensor_model_parallel_all_reduce",
              moe.count("tensor_model_parallel_all_reduce(") >= 3 and "states = tensor_model_parallel_all_reduce(states)" in moe)
    vpe = text_of("model_executor/layers/vocab_parallel_embedding.py")
    if vpe is not None:
        check("F5 VocabParallelEmbedding.forward reduces with tensor_model_parallel_all_reduce (the embedding all-reduce)",
              "tensor_model_parallel_all_reduce(output_parallel)" in vpe)
    mhc = text_of("models/deepseek_v41/nvidia/ops/mhc.py")
    if mhc is not None:
        check("F6 the model's fused mHC all-reduce (which bypasses the communicator) needs TP == 4 -> off at TP=8",
              "def supports_mhc_all_reduce(vllm_config: VllmConfig) -> bool:" in mhc and "parallel.tensor_parallel_size != 4" in mhc)
    check("F7 decode message sizes at hidden size 5120 (bf16): 6 tokens = 61,440 B <= 65,536 < 7 tokens = 71,680 B",
          6 * 5120 * 2 == 61440 <= 65536 < 7 * 5120 * 2 == 71680)
    recipe_env = L.PDIR.parent / "config" / "recipe.env"
    if recipe_env.is_file():
        txt = recipe_env.read_text()
        m = re.search(r'\\?"num_speculative_tokens\\?":\s*(\d+)', txt)
        check("F8 the recipe's DSpark draft length gives a verification step of (drafts + 1) tokens that fits the default limit",
              bool(m) and (int(m.group(1)) + 1) * 5120 * 2 <= 65536, m.group(0) if m else "no num_speculative_tokens")
    else:
        L.skip("F8 the recipe's DSpark draft length fits the default limit", "config/recipe.env not next to patches/")

    for d in roots:
        shutil.rmtree(d, ignore_errors=True)
    return L.finish("p012_cpu_check")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
