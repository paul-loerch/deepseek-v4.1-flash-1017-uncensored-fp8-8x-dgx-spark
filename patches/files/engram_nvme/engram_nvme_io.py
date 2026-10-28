# SPDX-License-Identifier: Apache-2.0
"""ctypes binding of the native Engram NVMe reader (engram_nvme.c) and the safetensors table locator -- patch 016.

Standard library only (no torch, no triton, no vllm imports), so the reader can be tested on any Linux box. The GPU side
(kernels, pinned buffers, CUDA aliases) is in engram_nvme.py.

The library `libengram_nvme.so` is built from engram_nvme.c by the patch wrapper (gcc, raw syscalls only) and lives next to
this file.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import logging
import os
import struct
import threading
from dataclasses import dataclass, field

logger = logging.getLogger("vllm.engram_nvme")

ABI_VERSION = 1
ROW_V = 256  # fp8 value bytes of one row
ROW_S = 8  # e8m0 scale bytes of one row

# control page layout (must match struct en_ctrl; cross-checked against the library at load time)
OFF_REQ_SEQ = 0x000
OFF_REQ_NTOK = 0x040
OFF_DONE_SEQ = 0x080
OFF_STATUS = 0x0C0
OFF_GPU_ABORT = 0x0C4
OFF_HEARTBEAT = 0x100
OFF_SCRATCH = 0x200  # inside the padding of the first page: a scratch word for the fence variant of the doorbell
OFF_STALL = 0x1000
CTRL_BYTES = 0x2000
STALL_N = 512

BACKEND_NAMES = {0: "auto", 1: "aio", 2: "threads"}
STATUS_NAMES = {
    0: "ok",
    1: "I/O error",
    2: "short read",
    3: "protocol violation",
    4: "timeout",
    5: "GPU wait gave up",
    6: "verification mismatch",
    7: "shard file changed",
    8: "internal error",
}
EXIT_CODE_FATAL = 86

_BACKENDS = {"auto": 0, "aio": 1, "threads": 2}
_POLLS = {"auto": 0, "spin": 1, "sleep": 2}
FLAG_DEDUPE = 1
FLAG_NO_ABORT = 2


class EnCfg(ctypes.Structure):
    _fields_ = [
        ("abi", ctypes.c_int32),
        ("backend", ctypes.c_int32),
        ("queue_depth", ctypes.c_int32),
        ("nthreads", ctypes.c_int32),
        ("poll", ctypes.c_int32),
        ("spin_us", ctypes.c_int32),
        ("sleep_us", ctypes.c_int32),
        ("keepwarm_us", ctypes.c_int32),
        ("keepwarm_idle_ms", ctypes.c_int32),
        ("timeout_ms", ctypes.c_int32),
        ("verify_pct", ctypes.c_int32),
        ("cpu", ctypes.c_int32),
        ("flags", ctypes.c_int32),
        ("fault_every", ctypes.c_int32),
        ("fault_kind", ctypes.c_int32),
        ("delay_us", ctypes.c_int32),
        ("hang_at", ctypes.c_int32),
        ("max_rows", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


class EnChanSpec(ctypes.Structure):
    _fields_ = [
        ("path", ctypes.c_char_p),
        ("values_off", ctypes.c_uint64),
        ("scales_off", ctypes.c_uint64),
        ("num_rows", ctypes.c_uint64),
        ("row_lo", ctypes.c_uint64),
        ("row_hi", ctypes.c_uint64),
        ("ctrl", ctypes.c_void_p),
        ("req_idx", ctypes.c_void_p),
        ("stage_v", ctypes.c_void_p),
        ("stage_s", ctypes.c_void_p),
        ("cap_rows", ctypes.c_uint32),
        ("local_heads", ctypes.c_uint32),
    ]


class EnStats(ctypes.Structure):
    _fields_ = (
        [(n, ctypes.c_uint64) for n in (
            "requests", "rows", "unique_rows", "oob_rows", "dup_rows", "reads", "read_bytes", "keepwarm_reads", "verify_rows", "errors")]
        + [("lat_hist", ctypes.c_uint64 * 24), ("lat_sum_us", ctypes.c_uint64), ("lat_max_us", ctypes.c_uint64),
           ("stall_hist", ctypes.c_uint64 * 24), ("stall_n", ctypes.c_uint64),
           ("backend", ctypes.c_int32), ("nchan", ctypes.c_int32), ("qd_max", ctypes.c_uint32), ("align", ctypes.c_uint32)]
    )


_lib = None
_lib_lock = threading.Lock()


def load_library(path: str | None = None) -> ctypes.CDLL:
    """Load libengram_nvme.so (next to this file unless `path`), set the prototypes, check ABI and struct sizes."""
    global _lib
    with _lib_lock:
        if _lib is not None and path is None:
            return _lib
        so = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libengram_nvme.so")
        lib = ctypes.CDLL(so)
        lib.en_abi_version.restype = ctypes.c_int
        if lib.en_abi_version() != ABI_VERSION:
            raise RuntimeError(f"{so}: ABI {lib.en_abi_version()} != {ABI_VERSION}")
        layout = (ctypes.c_uint32 * 12)()
        lib.en_ctrl_layout(layout)
        expect = [OFF_REQ_SEQ, OFF_REQ_NTOK, OFF_DONE_SEQ, OFF_STATUS, OFF_GPU_ABORT, OFF_HEARTBEAT, OFF_STALL, CTRL_BYTES, STALL_N,
                  ctypes.sizeof(EnCfg), ctypes.sizeof(EnChanSpec), ctypes.sizeof(EnStats)]
        if list(layout) != expect:
            raise RuntimeError(f"{so}: struct layout differs from the Python binding: library {list(layout)} vs binding {expect}")
        lib.en_probe_aio.restype = ctypes.c_int
        lib.en_create.restype = ctypes.c_void_p
        lib.en_create.argtypes = [ctypes.POINTER(EnCfg), ctypes.c_char_p, ctypes.c_size_t]
        lib.en_add_channel.restype = ctypes.c_int
        lib.en_add_channel.argtypes = [ctypes.c_void_p, ctypes.POINTER(EnChanSpec), ctypes.c_char_p, ctypes.c_size_t]
        lib.en_start.restype = ctypes.c_int
        lib.en_start.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.en_stop.argtypes = [ctypes.c_void_p]
        lib.en_destroy.argtypes = [ctypes.c_void_p]
        lib.en_backend.restype = ctypes.c_int
        lib.en_backend.argtypes = [ctypes.c_void_p]
        lib.en_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(EnStats)]
        lib.en_test_gpu_publish.argtypes = [ctypes.POINTER(EnChanSpec), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64]
        lib.en_test_wait_done.restype = ctypes.c_int
        lib.en_test_wait_done.argtypes = [ctypes.POINTER(EnChanSpec), ctypes.c_uint64, ctypes.c_int]
        if path is None:
            _lib = lib
        return lib


@dataclass
class NvmeSettings:
    """Reader settings; `from_engram_config` maps the EngramConfig.nvme_* fields."""

    backend: str = "auto"  # auto | aio | threads
    queue_depth: int = 256
    threads: int = 64  # pread pool: only used when AIO is unavailable; 8 workers measured 0.67 s for a prefill-size request
    poll: str = "auto"  # auto | spin | sleep
    spin_us: int = 300
    sleep_us: int = 20
    keepwarm_us: int = 500
    keepwarm_idle_s: float = 5.0
    timeout_s: float = 30.0
    verify_pct: int = 0
    dedupe: bool = True
    cpu: int = -1
    no_abort: bool = False  # tests only: record the error instead of ending the process
    fault: str = ""  # tests only: "eio:N", "short:N", "delay_us:N", "hang:N" (also read from ENGRAM_NVME_FAULT)

    @classmethod
    def from_engram_config(cls, ec) -> "NvmeSettings":
        return cls(
            backend=ec.nvme_io_backend,
            queue_depth=ec.nvme_queue_depth,
            threads=ec.nvme_threads,
            poll=ec.nvme_poll,
            keepwarm_us=ec.nvme_keepwarm_us,
            keepwarm_idle_s=ec.nvme_keepwarm_idle_s,
            timeout_s=ec.nvme_timeout_s,
            verify_pct=ec.nvme_verify_pct,
            dedupe=ec.nvme_dedupe,
            cpu=ec.nvme_cpu,
        )

    def to_cfg(self, max_rows: int) -> EnCfg:
        cfg = EnCfg()
        cfg.abi = ABI_VERSION
        cfg.backend = _BACKENDS[self.backend]
        cfg.queue_depth = self.queue_depth
        cfg.nthreads = self.threads
        cfg.poll = _POLLS[self.poll]
        cfg.spin_us = self.spin_us
        cfg.sleep_us = self.sleep_us
        cfg.keepwarm_us = self.keepwarm_us
        cfg.keepwarm_idle_ms = int(self.keepwarm_idle_s * 1000)
        cfg.timeout_ms = int(self.timeout_s * 1000)
        cfg.verify_pct = self.verify_pct
        cfg.cpu = self.cpu
        cfg.flags = (FLAG_DEDUPE if self.dedupe else 0) | (FLAG_NO_ABORT if self.no_abort else 0)
        cfg.max_rows = max_rows
        fault = self.fault or os.environ.get("ENGRAM_NVME_FAULT", "")
        for item in filter(None, (x.strip() for x in fault.split(","))):
            kind, _, val = item.partition(":")
            n = int(val)
            if kind == "eio":
                cfg.fault_every, cfg.fault_kind = n, 1
            elif kind == "short":
                cfg.fault_every, cfg.fault_kind = n, 2
            elif kind == "delay_us":
                cfg.delay_us = n
            elif kind == "hang":
                cfg.hang_at = n
            else:
                raise ValueError(f"ENGRAM_NVME_FAULT: unknown fault {item!r}")
        return cfg


@dataclass(frozen=True)
class TableLoc:
    path: str
    values_off: int  # absolute file offset of row 0 of embed.weight
    scales_off: int  # absolute file offset of row 0 of embed.scale
    num_rows: int
    file_size: int


def locate_tables(model_dir: str, layer_id: int, num_rows: int, dim: int = ROW_V, quant_block: int = 32) -> TableLoc:
    """Find layers.<layer_id>.engram.embed.{weight,scale} in the checkpoint and return their absolute file offsets.

    Everything that would make a raw-offset read wrong is validated: both tensors in one shard, dtypes, shapes, byte lengths.
    """
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    wname, sname = f"layers.{layer_id}.engram.embed.weight", f"layers.{layer_id}.engram.embed.scale"
    if wname not in weight_map or sname not in weight_map:
        raise ValueError(f"{model_dir}: {wname} / {sname} not in model.safetensors.index.json")
    if weight_map[wname] != weight_map[sname]:
        raise NotImplementedError(f"layer {layer_id}: value and scale tensors are in different shards")
    path = os.path.join(model_dir, weight_map[wname])
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    base = 8 + hlen
    w, s = header[wname], header[sname]
    if w["dtype"] != "F8_E4M3" or s["dtype"] != "F8_E8M0":
        raise ValueError(f"{path}: unexpected dtypes {w['dtype']} / {s['dtype']} (want F8_E4M3 / F8_E8M0)")
    scale_dim = dim // quant_block
    if list(w["shape"]) != [num_rows, dim] or list(s["shape"]) != [num_rows, scale_dim]:
        raise ValueError(f"{path}: shapes {w['shape']} / {s['shape']} != [{num_rows}, {dim}] / [{num_rows}, {scale_dim}]")
    (w0, w1), (s0, s1) = w["data_offsets"], s["data_offsets"]
    if w1 - w0 != num_rows * dim or s1 - s0 != num_rows * scale_dim:
        raise ValueError(f"{path}: tensor byte lengths do not match the shapes")
    size = os.stat(path).st_size
    if base + max(w1, s1) > size:
        raise ValueError(f"{path}: truncated ({size} B, tensors end at {base + max(w1, s1)})")
    return TableLoc(path, base + w0, base + s0, num_rows, size)


def u32_at(addr: int, off: int) -> int:
    return ctypes.c_uint32.from_address(addr + off).value


def u64_at(addr: int, off: int) -> int:
    return ctypes.c_uint64.from_address(addr + off).value


@dataclass
class Channel:
    """One layer's table as registered with the engine. Keeps the spec (and so the buffer addresses) alive."""

    index: int
    spec: EnChanSpec
    ctrl_addr: int
    path: str = ""
    extra: dict = field(default_factory=dict)

    def status(self) -> int:
        return u32_at(self.ctrl_addr, OFF_STATUS)

    def gpu_abort(self) -> int:
        return u32_at(self.ctrl_addr, OFF_GPU_ABORT)

    def done_seq(self) -> int:
        return u64_at(self.ctrl_addr, OFF_DONE_SEQ)

    def req_seq(self) -> int:
        return u64_at(self.ctrl_addr, OFF_REQ_SEQ)


class NvmeEngine:
    """One native reader (threads + AIO context) serving any number of channels."""

    def __init__(self, settings: NvmeSettings, max_rows: int, lib_path: str | None = None):
        self.settings = settings
        self.lib = load_library(lib_path)
        self._cfg = settings.to_cfg(max_rows)
        err = ctypes.create_string_buffer(512)
        self._h = self.lib.en_create(ctypes.byref(self._cfg), err, len(err))
        if not self._h:
            raise RuntimeError(f"engram-nvme: engine creation failed: {err.value.decode()}")
        self.channels: list[Channel] = []
        self._started = False
        self._log_thread: threading.Thread | None = None
        self._log_stop = threading.Event()
        atexit.register(self.stop)

    @property
    def backend(self) -> str:
        return BACKEND_NAMES[self.lib.en_backend(self._h)]

    def add_channel(self, loc: TableLoc, row_lo: int, row_hi: int, ctrl_addr: int, req_idx_addr: int, stage_v_addr: int,
                    stage_s_addr: int, cap_rows: int, local_heads: int) -> Channel:
        spec = EnChanSpec()
        spec.path = loc.path.encode()
        spec.values_off, spec.scales_off, spec.num_rows = loc.values_off, loc.scales_off, loc.num_rows
        spec.row_lo, spec.row_hi = row_lo, row_hi
        spec.ctrl, spec.req_idx, spec.stage_v, spec.stage_s = ctrl_addr, req_idx_addr, stage_v_addr, stage_s_addr
        spec.cap_rows, spec.local_heads = cap_rows, local_heads
        err = ctypes.create_string_buffer(512)
        idx = self.lib.en_add_channel(self._h, ctypes.byref(spec), err, len(err))
        if idx < 0:
            raise RuntimeError(f"engram-nvme: cannot add channel for {loc.path}: {err.value.decode()}")
        ch = Channel(idx, spec, ctrl_addr, loc.path)
        self.channels.append(ch)
        return ch

    def start(self) -> None:
        if self._started:
            return
        err = ctypes.create_string_buffer(512)
        if self.lib.en_start(self._h, err, len(err)) != 0:
            raise RuntimeError(f"engram-nvme: cannot start the reader: {err.value.decode()}")
        self._started = True

    def stop(self) -> None:
        self._log_stop.set()
        if self._h:
            self.lib.en_stop(self._h)

    def close(self) -> None:
        self.stop()
        if self._h:
            self.lib.en_destroy(self._h)
            self._h = None

    def stats(self) -> dict:
        st = EnStats()
        self.lib.en_get_stats(self._h, ctypes.byref(st))
        d = {n: getattr(st, n) for n, _ in EnStats._fields_ if not n.endswith("hist")}
        d["lat_hist"] = list(st.lat_hist)
        d["stall_hist"] = list(st.stall_hist)
        d["backend"] = BACKEND_NAMES[st.backend]
        return d

    # test hooks: play the GPU's part with the same memory ordering as the Triton publish kernel
    def test_publish(self, ch: Channel, ids_addr: int, n_ids: int, ntok: int, seq: int) -> None:
        self.lib.en_test_gpu_publish(ctypes.byref(ch.spec), ids_addr, n_ids, ntok, seq)

    def test_wait_done(self, ch: Channel, seq: int, timeout_ms: int = 10000) -> int:
        return self.lib.en_test_wait_done(ctypes.byref(ch.spec), seq, timeout_ms)

    def start_stats_logger(self, interval_s: float = 60.0) -> None:
        """Log one `[engram-nvme]` line every `interval_s` (daemon thread; reads host memory only, no CUDA)."""
        if self._log_thread is not None:
            return

        def run() -> None:
            last = self.stats()
            while not self._log_stop.wait(interval_s):
                cur = self.stats()
                d = cur["requests"] - last["requests"]
                if d > 0:
                    lat = [a - b for a, b in zip(cur["lat_hist"], last["lat_hist"])]
                    stall = [a - b for a, b in zip(cur["stall_hist"], last["stall_hist"])]
                    logger.info(
                        "[engram-nvme] backend=%s requests=%d rows=%d unique=%d oob=%d reads=%d keepwarm=%d "
                        "io_p50<=%dus p99<=%dus max=%dus stall_polls_p50<=%d p99<=%d errors=%d",
                        cur["backend"], d, cur["rows"] - last["rows"], cur["unique_rows"] - last["unique_rows"],
                        cur["oob_rows"] - last["oob_rows"], cur["reads"] - last["reads"], cur["keepwarm_reads"] - last["keepwarm_reads"],
                        hist_pct(lat, 0.50), hist_pct(lat, 0.99), cur["lat_max_us"], hist_pct(stall, 0.50), hist_pct(stall, 0.99), cur["errors"],
                    )
                last = cur

        self._log_thread = threading.Thread(target=run, name="engram-nvme-stats", daemon=True)
        self._log_thread.start()


def hist_pct(hist: list[int], q: float) -> int:
    """Upper bound of the log2 bucket holding quantile q (bucket i covers [2^i, 2^(i+1)); bucket 0 covers 0..1)."""
    total = sum(hist)
    if total == 0:
        return 0
    acc = 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= q * total:
            return (1 << (i + 1)) - 1 if i else 1
    return (1 << len(hist)) - 1


_engine: NvmeEngine | None = None
_engine_lock = threading.Lock()


def get_engine(settings: NvmeSettings, max_rows: int) -> NvmeEngine:
    """The process-wide engine (one reader thread for all Engram layers of this rank)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = NvmeEngine(settings, max_rows)
            _engine.start()
            _engine.start_stats_logger()
            logger.info("[engram-nvme] reader started: backend=%s queue_depth=%d poll=%s keepwarm=%dus timeout=%ds dedupe=%s verify=%d%%",
                        _engine.backend, settings.queue_depth, settings.poll, settings.keepwarm_us, int(settings.timeout_s), settings.dedupe, settings.verify_pct)
        return _engine

