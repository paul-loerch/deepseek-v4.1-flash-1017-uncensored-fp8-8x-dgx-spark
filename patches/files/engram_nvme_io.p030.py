# SPDX-License-Identifier: Apache-2.0
"""ctypes binding of the native Engram NVMe reader (engram_nvme.c) and the safetensors table locator -- patch 016, packed rows -- patch 030.

Standard library only (no torch, no triton, no vllm imports), so the reader can be tested on any Linux box. The GPU side
(kernels, pinned buffers, CUDA aliases) is in engram_nvme.py.

The library `libengram_nvme.so` is built from engram_nvme.c by the patch wrapper (gcc, raw syscalls only) and lives next to
this file.

PATCH 030 (this file = patch 016's engram_nvme_io.py + the packed-rows selection; ABI 2; every 016 name and signature is kept, so the GPU-side
engram_nvme.py, which patch 028 pins, is unchanged): a channel may be served from a per-node PACKED file (tools/engram_pack.py: one 512-byte
record [256 B values | 8 B scales | 248 B zero] per row, so one NVMe read per unique row instead of two). `NvmeEngine.add_channel` (packed="auto",
the default) asks `select_packed` whether ENGRAM_NVME_PACKED_DIR holds a valid pack of exactly this channel's slice: found by content (layer and
row slice), manifest complete, source shard unchanged (size, mtime, sha256 of its header, tensor offsets), packed file complete and consistent
with its own header, a random sample of rows equal to the shard. When it does not, the channel is registered with the two-read layout and the
log says why. Either way the delivered bytes are the same. ENGRAM_NVME_PACKED = auto (default when the directory is set) | require (a
missing/invalid pack is an error, not a fallback) | off; ENGRAM_NVME_PACKED_SPOT = rows spot-checked at start (default 256, 0 = none).
ENGRAM_NVME_VERIFY_AFTER=1 (opt-in) moves the nvme_verify_pct re-reads behind the publication of done_seq: they cost ~0.3-0.4 ms per verified row (two
cold buffered reads), ~80-100 ms for 1 % of an 8,192-token prefill request.
"""
from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import logging
import os
import random
import struct
import threading
from dataclasses import dataclass, field

logger = logging.getLogger("vllm.engram_nvme")

ABI_VERSION = 2
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
FLAG_VERIFY_AFTER = 4  # patch 030


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
        ("packed_path", ctypes.c_char_p),  # patch 030: packed per-rank file, NULL = two-read layout
        ("packed_off", ctypes.c_uint64),  # patch 030: offset of record 0 (= row_lo) in the packed file
    ]


class EnStats(ctypes.Structure):
    _fields_ = (
        [(n, ctypes.c_uint64) for n in (
            "requests", "rows", "unique_rows", "oob_rows", "dup_rows", "reads", "read_bytes", "keepwarm_reads", "verify_rows", "errors", "packed_reads")]
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
    verify_after: bool = False  # patch 030: run the verify_pct re-reads after done_seq is published (also ENGRAM_NVME_VERIFY_AFTER=1)

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
        verify_after = self.verify_after or os.environ.get("ENGRAM_NVME_VERIFY_AFTER", "").strip().lower() in ("1", "on", "true", "yes")
        cfg.flags = (FLAG_DEDUPE if self.dedupe else 0) | (FLAG_NO_ABORT if self.no_abort else 0) | (FLAG_VERIFY_AFTER if verify_after else 0)
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
    layer_id: "int | None" = None  # patch 030: for log lines and the pack/layer cross-check (None: unknown)


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
    return TableLoc(path, base + w0, base + s0, num_rows, size, layer_id)


# --------------------------------------------------------------------------------------------------------------------
# packed rows (patch 030): selection and validation of a per-node packed file (written by tools/engram_pack.py)
# --------------------------------------------------------------------------------------------------------------------
PACK_FORMAT = "engram-packed-rows"
PACK_LAYOUT_VERSION = 1
PACK_MAGIC = b"ENGRPACK"
PACK_HEADER_BYTES = 4096
PACK_STRIDE = 512  # bytes per record: [ROW_V values | ROW_S scales | zero padding]
PACK_USED = ROW_V + ROW_S
ENV_PACKED_DIR = "ENGRAM_NVME_PACKED_DIR"
ENV_PACKED_MODE = "ENGRAM_NVME_PACKED"
ENV_PACKED_SPOT = "ENGRAM_NVME_PACKED_SPOT"
PACK_SPOT_DEFAULT = 256
_PACK_HDR = struct.Struct("<8s10I6Qq32s32sQ")  # same layout as tools/engram_pack.py (176 bytes, then zeros, then a sha256)


class PackedMismatch(Exception):
    """The packed file of a channel is missing, stale or inconsistent; str(e) says why."""


@dataclass(frozen=True)
class PackedLoc:
    path: str  # the packed file
    manifest_path: str
    data_off: int  # file offset of record 0 (= row_lo)
    rows: int
    data_sha256: str
    layer: int = -1  # what the pack was made for (from its manifest)
    tp: int = -1
    rank: int = -1


def decode_pack_header(raw: bytes) -> dict:
    if len(raw) < PACK_HEADER_BYTES:
        raise PackedMismatch(f"header block is {len(raw)} B, want {PACK_HEADER_BYTES}")
    raw = bytes(raw[:PACK_HEADER_BYTES])
    if raw[:8] != PACK_MAGIC:
        raise PackedMismatch("bad magic: not an engram packed-rows file")
    if hashlib.sha256(raw[:PACK_HEADER_BYTES - 32]).digest() != raw[PACK_HEADER_BYTES - 32:]:
        raise PackedMismatch("header checksum mismatch")
    (_m, ver, hbytes, data_off, stride, vb, sb, layer, rank, tp, flags, row_lo, row_hi, num_rows, values_off, scales_off, source_size,
     source_mtime_ns, hsha, dsha, data_bytes) = _PACK_HDR.unpack_from(raw)
    return dict(layout_version=ver, header_bytes=hbytes, data_off=data_off, row_stride=stride, value_bytes=vb, scale_bytes=sb, layer=layer,
                rank=rank, tp=tp, complete=bool(flags & 1), row_lo=row_lo, row_hi=row_hi, num_rows=num_rows, values_off=values_off,
                scales_off=scales_off, source_size=source_size, source_mtime_ns=source_mtime_ns, source_header_sha256=hsha.hex(),
                data_sha256=dsha.hex(), data_bytes=data_bytes)


def shard_header_sha256(path: str) -> str:
    """sha256 of the safetensors header bytes (the 8 length bytes plus the JSON header): identifies the tensor layout of a shard."""
    with open(path, "rb") as f:
        raw8 = f.read(8)
        if len(raw8) != 8:
            raise PackedMismatch(f"{path}: too short for a safetensors header")
        (hlen,) = struct.unpack("<Q", raw8)
        body = f.read(hlen)
    if len(body) != hlen:
        raise PackedMismatch(f"{path}: truncated safetensors header")
    return hashlib.sha256(raw8 + body).hexdigest()


def _fadvise_dontneed(fd: int) -> None:
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _spot_check(loc: TableLoc, pk_path: str, data_off: int, row_lo: int, row_hi: int, n: int, seed: "int | None") -> int:
    """Compare n records of the packed file (the slice's first and last row plus random ones) with the shard: values, scales, zero pad."""
    rng = random.Random(seed)
    ids = {row_lo, row_hi - 1}
    ids.update(rng.sample(range(row_lo, row_hi), min(row_hi - row_lo, max(0, n - 2))))
    sfd, pfd = os.open(loc.path, os.O_RDONLY), os.open(pk_path, os.O_RDONLY)  # buffered on purpose: an independent path from the reader's
    try:
        for g in sorted(ids):
            rec = os.pread(pfd, PACK_STRIDE, data_off + (g - row_lo) * PACK_STRIDE)
            v = os.pread(sfd, ROW_V, loc.values_off + g * ROW_V)
            sc = os.pread(sfd, ROW_S, loc.scales_off + g * ROW_S)
            if len(rec) != PACK_STRIDE or rec[:ROW_V] != v or rec[ROW_V:PACK_USED] != sc or rec[PACK_USED:] != bytes(PACK_STRIDE - PACK_USED):
                raise PackedMismatch(f"spot check: the packed record of row {g} differs from the shard (values/scales/zero padding)")
    finally:
        _fadvise_dontneed(sfd)
        _fadvise_dontneed(pfd)
        os.close(sfd)
        os.close(pfd)
    return len(ids)


def _read_manifest(mpath: str) -> dict:
    try:
        if os.path.getsize(mpath) > 8 << 20:
            raise PackedMismatch(f"manifest {mpath} is implausibly large")
        with open(mpath) as f:
            m = json.load(f)
    except FileNotFoundError:
        raise PackedMismatch(f"no manifest {mpath}")
    except (OSError, ValueError) as e:
        raise PackedMismatch(f"manifest {mpath} unreadable: {e}")
    if not isinstance(m, dict):
        raise PackedMismatch(f"manifest {mpath} is not a JSON object")
    return m


def validate_packed(loc: TableLoc, *, row_lo: int, row_hi: int, manifest_path: str, spot: int, seed: "int | None" = None) -> PackedLoc:
    """Everything that must hold for the pack described by `manifest_path` to deliver exactly the bytes the shard would for the slice
    [row_lo, row_hi) of this table. Raises PackedMismatch with the first thing that does not."""
    m = _read_manifest(manifest_path)
    try:
        if m.get("format") != PACK_FORMAT or m.get("layout_version") != PACK_LAYOUT_VERSION:
            raise PackedMismatch(f"manifest format {m.get('format')!r} layout {m.get('layout_version')!r}, this reader supports {PACK_FORMAT} {PACK_LAYOUT_VERSION}")
        if m.get("complete") is not True:
            raise PackedMismatch("the manifest does not say complete (pack unfinished?)")
        layer, tp, rank = m["layer"], m["tp"], m["rank"]
        if loc.layer_id is not None and layer != loc.layer_id:
            raise PackedMismatch(f"the pack is for layer {layer}, the channel is layer {loc.layer_id}")
        if (m["row_lo"], m["row_hi"]) != (row_lo, row_hi) or m["rows"] != row_hi - row_lo:
            raise PackedMismatch(f"row slice differs: manifest [{m['row_lo']}, {m['row_hi']}) vs the engine's [{row_lo}, {row_hi})")
        if m["num_rows"] != loc.num_rows:
            raise PackedMismatch(f"table size differs: manifest {m['num_rows']} rows vs the shard's {loc.num_rows}")
        if (m["row_stride"], m["value_bytes"], m["scale_bytes"]) != (PACK_STRIDE, ROW_V, ROW_S) or m["data_off"] % 4096 or m["data_off"] < PACK_HEADER_BYTES:
            raise PackedMismatch("manifest record geometry is not 512 B records of 256 + 8 bytes at a 4 KiB aligned data offset")
        if m["data_bytes"] != m["rows"] * PACK_STRIDE:
            raise PackedMismatch("manifest data_bytes is not rows * 512")
        name = m["file"]
        if os.path.basename(name) != name or not name.endswith(".pack") or os.path.basename(manifest_path) != name + ".json":
            raise PackedMismatch(f"manifest names another file ({name!r})")
        ppath = os.path.join(os.path.dirname(manifest_path), name)
        src = m["source"]
        st = os.stat(loc.path)
        if src["size"] != st.st_size or src["mtime_ns"] != st.st_mtime_ns:
            raise PackedMismatch(
                f"stale: the source shard changed since packing (size {src['size']} -> {st.st_size}, mtime_ns {src['mtime_ns']} -> {st.st_mtime_ns})")
        if (src["values_off"], src["scales_off"]) != (loc.values_off, loc.scales_off):
            raise PackedMismatch("stale: the tensor offsets in the source shard differ from the ones packed")
        if src["header_sha256"] != shard_header_sha256(loc.path):
            raise PackedMismatch("stale: the source shard's safetensors header differs from the one packed")
        want_size = m["data_off"] + m["rows"] * PACK_STRIDE
        try:
            pst = os.stat(ppath)
        except FileNotFoundError:
            raise PackedMismatch(f"packed file {ppath} is missing")
        if pst.st_size != want_size or m["file_size"] != want_size:
            raise PackedMismatch(
                f"packed file {ppath} is {pst.st_size} B, expected {want_size} B (manifest says {m['file_size']}): truncated or extended")
        with open(ppath, "rb") as f:
            hd = decode_pack_header(f.read(PACK_HEADER_BYTES))
        if not hd["complete"]:
            raise PackedMismatch("the packed file's own header says it is incomplete")
        for key, have, want in (
                ("layout_version", hd["layout_version"], PACK_LAYOUT_VERSION), ("layer", hd["layer"], layer), ("tp", hd["tp"], tp),
                ("rank", hd["rank"], rank), ("row_lo", hd["row_lo"], row_lo), ("row_hi", hd["row_hi"], row_hi), ("num_rows", hd["num_rows"], loc.num_rows),
                ("data_off", hd["data_off"], m["data_off"]), ("row_stride", hd["row_stride"], PACK_STRIDE), ("data_bytes", hd["data_bytes"], m["data_bytes"]),
                ("data_sha256", hd["data_sha256"], m["data_sha256"]), ("source_header_sha256", hd["source_header_sha256"], src["header_sha256"]),
                ("values_off", hd["values_off"], loc.values_off), ("scales_off", hd["scales_off"], loc.scales_off)):
            if have != want:
                raise PackedMismatch(f"the packed file's header and the manifest/engine disagree on {key} ({have} vs {want})")
        data_off, rows, data_sha = m["data_off"], m["rows"], m["data_sha256"]
    except (KeyError, TypeError) as e:
        raise PackedMismatch(f"manifest {manifest_path} is malformed ({type(e).__name__}: {e})")
    if spot > 0:
        _spot_check(loc, ppath, data_off, row_lo, row_hi, spot, seed)
    return PackedLoc(ppath, manifest_path, data_off, rows, data_sha, layer, tp, rank)


def _packed_mode() -> str:
    return os.environ.get(ENV_PACKED_MODE) or ("auto" if os.environ.get(ENV_PACKED_DIR) else "off")


def select_packed(loc: TableLoc, *, row_lo: int, row_hi: int, packed_dir: "str | None" = None, mode: "str | None" = None,
                  spot: "int | None" = None, seed: "int | None" = None) -> "tuple[PackedLoc | None, str]":
    """(PackedLoc, "ok") when `packed_dir` holds a valid pack of exactly this channel's slice, else (None, why).

    The pack is found by CONTENT, not by name: the manifest whose layer (if known), row slice and table size equal the channel's. A
    pack for another rank or layer therefore cannot be picked up by mistake, and nothing here needs the TP rank. Mode `require` raises
    RuntimeError instead of returning None."""
    env = os.environ
    packed_dir = env.get(ENV_PACKED_DIR) if packed_dir is None else packed_dir
    mode = _packed_mode() if mode is None else mode
    if mode not in ("auto", "require", "off"):
        raise ValueError(f"{ENV_PACKED_MODE} must be auto, require or off, not {mode!r}")
    if mode == "off":
        return None, f"packed rows disabled ({ENV_PACKED_MODE}=off or {ENV_PACKED_DIR} unset)"

    def no(why: str):
        if mode == "require":
            raise RuntimeError(f"engram-nvme: layer {loc.layer_id} rows [{row_lo}, {row_hi}): packed rows are required ({ENV_PACKED_MODE}=require) but unusable: {why}")
        return None, why

    if not packed_dir:
        return no(f"{ENV_PACKED_DIR} not set")
    if spot is None:
        spot = int(env.get(ENV_PACKED_SPOT, PACK_SPOT_DEFAULT))
    try:
        names = sorted(n for n in os.listdir(packed_dir) if n.endswith(".pack.json"))
    except OSError as e:
        return no(f"cannot list {packed_dir}: {e}")
    cands, notes = [], []
    for n in names:
        mpath = os.path.join(packed_dir, n)
        try:
            m = _read_manifest(mpath)
            if (m.get("row_lo"), m.get("row_hi"), m.get("num_rows")) != (row_lo, row_hi, loc.num_rows):
                notes.append(f"{n}: rows [{m.get('row_lo')}, {m.get('row_hi')}) of {m.get('num_rows')}")
            elif loc.layer_id is not None and m.get("layer") != loc.layer_id:
                notes.append(f"{n}: layer {m.get('layer')}")
            else:
                cands.append(mpath)
        except PackedMismatch as e:
            notes.append(f"{n}: {e}")
    if not cands:
        return no(f"no manifest in {packed_dir} matches layer {loc.layer_id}, rows [{row_lo}, {row_hi}) of {loc.num_rows}"
                  + (f" ({len(names)} seen: {'; '.join(notes[:4])})" if names else " (no *.pack.json there: nothing packed yet?)"))
    reasons = []
    for mpath in cands:
        try:
            return validate_packed(loc, row_lo=row_lo, row_hi=row_hi, manifest_path=mpath, spot=spot, seed=seed), "ok"
        except PackedMismatch as e:
            reasons.append(f"{os.path.basename(mpath)}: {e}")
    return no("; ".join(reasons))


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
    layout: str = "two-read"  # patch 030: "packed" (one 512 B record per row) or "two-read" (values and scales from the shard)
    layout_reason: str = ""  # why the layout is what it is (for the log)
    packed: "PackedLoc | None" = None

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
                    stage_s_addr: int, cap_rows: int, local_heads: int, packed: "PackedLoc | None | str" = "auto") -> Channel:
        """Register one layer's table.

        packed="auto" (the default, and what the unchanged patch-016 callers get): when ENGRAM_NVME_PACKED_DIR holds a valid pack of exactly
        this slice (see select_packed) the rows are read from it, one 512 B read per unique row; otherwise, and whenever the pack is
        stale, inconsistent or rejected by the reader, from the shard (two reads per row), and the log says which and why.
        packed=None forces the two-read layout, packed=PackedLoc forces that pack (errors are raised, not hidden)."""
        why = "two-read layout requested by the caller"
        auto = isinstance(packed, str)
        if auto:
            if packed != "auto":
                raise ValueError("packed must be 'auto', None or a PackedLoc")
            packed, why = select_packed(loc, row_lo=row_lo, row_hi=row_hi)  # raises RuntimeError in require mode
        elif packed is not None:
            why = "forced by the caller"
        base_spec = EnChanSpec()
        base_spec.path = loc.path.encode()
        base_spec.values_off, base_spec.scales_off, base_spec.num_rows = loc.values_off, loc.scales_off, loc.num_rows
        base_spec.row_lo, base_spec.row_hi = row_lo, row_hi
        base_spec.ctrl, base_spec.req_idx, base_spec.stage_v, base_spec.stage_s = ctrl_addr, req_idx_addr, stage_v_addr, stage_s_addr
        base_spec.cap_rows, base_spec.local_heads = cap_rows, local_heads
        spec = base_spec
        if packed is not None:
            spec.packed_path = packed.path.encode()
            spec.packed_off = packed.data_off
        err = ctypes.create_string_buffer(512)
        idx = self.lib.en_add_channel(self._h, ctypes.byref(spec), err, len(err))
        if idx < 0 and packed is not None and auto and _packed_mode() != "require":
            why = f"the reader rejected the packed file ({err.value.decode()})"
            packed = None
            spec = base_spec  # a fresh spec without the packed fields
            spec.packed_path, spec.packed_off = None, 0
            idx = self.lib.en_add_channel(self._h, ctypes.byref(spec), err, len(err))
        if idx < 0:
            raise RuntimeError(f"engram-nvme: cannot add channel for {loc.path}{' (packed ' + packed.path + ')' if packed else ''}: {err.value.decode()}")
        ch = Channel(idx, spec, ctrl_addr, loc.path, packed=packed, layout="packed" if packed else "two-read", layout_reason=why)
        self.channels.append(ch)
        tag = f"layer {loc.layer_id}" if loc.layer_id is not None else os.path.basename(loc.path)
        if packed is not None:
            logger.info("[engram-nvme] %s rows [%d, %d): layout=packed file=%s data_off=%d (one 512 B read per unique row; packed for rank %d of tp %d)",
                        tag, row_lo, row_hi, packed.path, packed.data_off, packed.rank, packed.tp)
        elif auto and os.environ.get(ENV_PACKED_DIR) and _packed_mode() != "off":
            logger.warning("[engram-nvme] %s rows [%d, %d): layout=two-read -- packed rows NOT used: %s", tag, row_lo, row_hi, why)
        else:
            logger.info("[engram-nvme] %s rows [%d, %d): layout=two-read (%s)", tag, row_lo, row_hi, why)
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
                        "io_p50<=%dus p99<=%dus max=%dus stall_polls_p50<=%d p99<=%d errors=%d packed_reads=%d",
                        cur["backend"], d, cur["rows"] - last["rows"], cur["unique_rows"] - last["unique_rows"],
                        cur["oob_rows"] - last["oob_rows"], cur["reads"] - last["reads"], cur["keepwarm_reads"] - last["keepwarm_reads"],
                        hist_pct(lat, 0.50), hist_pct(lat, 0.99), cur["lat_max_us"], hist_pct(stall, 0.50), hist_pct(stall, 0.99), cur["errors"],
                        cur["packed_reads"] - last["packed_reads"],
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
            logger.info("[engram-nvme] reader started: backend=%s queue_depth=%d poll=%s keepwarm=%dus timeout=%ds dedupe=%s verify=%d%%%s",
                        _engine.backend, settings.queue_depth, settings.poll, settings.keepwarm_us, int(settings.timeout_s), settings.dedupe, settings.verify_pct,
                        " (checked after the rows are handed to the GPU)" if (_engine._cfg.flags & FLAG_VERIFY_AFTER) and settings.verify_pct else "")
        return _engine

