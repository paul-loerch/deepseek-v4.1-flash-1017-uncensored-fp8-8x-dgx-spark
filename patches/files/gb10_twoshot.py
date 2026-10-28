# SPDX-License-Identifier: Apache-2.0
"""GB10 two-shot RoCE all-reduce for small tensor-parallel messages (reduce-scatter + all-gather over RC queue pairs).

WHY: RoCEnante (flashinfer.experimental.b12x.comm.roce) is a one-shot all-reduce: every rank RDMA-writes its WHOLE
message to all W-1 peers, so a rank sends (W-1)x the message and the time is bandwidth-bound from ~2 tokens up
(8 ranks, bf16, 5120 elements per token: 23 / 61 / 100 / 176 / 329 us at 1 / 6 / 12 / 24 / 48 tokens).  A two-shot sends
(W-1)/W of the message in each of two phases (about 1.75x instead of 7x at W = 8).

PROTOCOL (details in gb10_twoshot_proxy.c and tests/roce2/NOTES.md): the message is P 16-byte packs, rank q owns the
chunk [q*c, min((q+1)*c, P)) with c = ceil(P/W).  One fused GPU kernel stages the input in pinned host memory and rings
a doorbell; the rank's C proxy thread RDMA-writes chunk p to peer p (phase 1; a chunk is sent in pieces of about
``piece_bytes`` with a progress word after each), reduces each piece of its own range as soon as every peer's piece has
arrived (CPU, NEON, fp32 in rank order, one round-to-nearest-even to the element type; the same bits as RoCEnante's GPU
reduction), RDMA-writes the reduced piece to every peer (phase 2) while later pieces are still arriving, and releases
its own flag after the last piece; the GPU kernel waits for the W reduced chunks (its own included) and copies them to
the output.  ``piece_bytes=0`` is the plain two-phase protocol without pieces.

Derives from FlashInfer's b12x RoCEnante runtime (roce_oneshot.py, Apache-2.0): region/exchange/connect structure,
stream ordering, capture(), check_health() and the epoch/poison scheme are adapted from it; the HCA discovery, route
planning, traffic class and capture-id helpers are imported from it.

USE (collective construction; every rank of the exchange group):

    rt = GB10TwoShotAllReduce(exchange_group=gloo_group, device=torch.device("cuda", idx), max_bytes=1 << 20)
    rt.warmup()                              # optional collective self-test (raises on any wrong element)
    if rt.should_allreduce(x):               # depends only on dtype/shape/contiguity/size
        y = rt.all_reduce(x)                 # eager, or inside a CUDA graph capture (see capture())
    with rt.capture(): ...                   # around torch.cuda.graph(...) capturing all_reduce calls
    rt.check_health()                        # cheap host reads; call after graph replays

All collectives of one runtime execute in launch order on one stream (one epoch, one doorbell, two slots).  bf16, fp16 and
fp32 (fp32 sums in rank order; the fp16/bf16 results round once to the element type, NaN -> 0x7FFF like CUDA).
Supports 2..8 ranks, one RDMA device per rank (one rail).
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import logging
import os
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from flashinfer.experimental.b12x.comm.roce import _proxy as _roce_proxy
from flashinfer.experimental.b12x.comm.roce import _routes as _roce_routes
from flashinfer.experimental.b12x.comm.roce import roce_oneshot as _roce

try:  # installed next to this file (package or flat directory)
    from . import gb10_twoshot_cute as _cute
except ImportError:  # pragma: no cover - flat layout
    import gb10_twoshot_cute as _cute

logger = logging.getLogger(__name__)

API_VERSION = 1
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
SUPPORTED_WORLD_SIZES = tuple(range(2, 9))
PACK_BYTES = 16
DEFAULT_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_THREADS = 512
DEFAULT_BLOCKS = 8
DEFAULT_PACKS_PER_THREAD = 2
# Polls of a flag before the kernel gives up (each poll is a system-scope load of host memory, about a microsecond).
DEFAULT_SPIN_LIMIT = 20_000_000
DEFAULT_PROXY_TIMEOUT_S = 10.0
DEFAULT_PREFETCH_BYTES = 32768  # per source chunk and op, prefetched when the source's flag is seen (0 = off)
DEFAULT_IDLE_MS = 200  # the proxy spins this long after the last doorbell, then naps (0 = never nap)
DEFAULT_PIECE_BYTES = 4096  # phase 2: the reduced chunk is reduced and posted in pieces of about this size (0 = one piece)
_DT_CODE = {torch.bfloat16: 1, torch.float16: 2, torch.float32: 3}
_PROXY_SOURCE = Path(__file__).with_name("gb10_twoshot_proxy.c")
_LIB_LOCK = threading.Lock()
_LIB: Optional[ctypes.CDLL] = None
_LIB_DIGEST = ""

# control-record word indices (i32 view of the 256-byte record)
_CW_GPU_ERR_SEQ, _CW_GPU_ERR_PEER, _CW_GPU_ERR_RAIL, _CW_GPU_ERR_CODE = 2, 3, 6, 7
_CW_PROXY_ERR, _CW_PROXY_PEER, _CW_PROXY_SEQ, _CW_PROXY_DONE = 32, 33, 34, 35
_PROXY_ERRORS = {1: "queue-pair/completion error", 2: "timed out waiting for a peer's phase-1 flag",
                 3: "doorbell skipped too many ops", 4: "unsupported size or dtype"}


# ------------------------------------------------------------------------------------------------ proxy library

def _cache_dir() -> Path:
    override = os.getenv("GB10_TWOSHOT_CACHE_DIR")
    return Path(override) if override else _roce_proxy._cache_dir()


def _build_library() -> tuple[Path, str]:
    """Compile the proxy source (keyed by source hash and flags) into the cache directory; returns (.so, digest)."""
    source = _PROXY_SOURCE.read_bytes()
    flags = ["-O2", "-std=gnu11", "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC"]
    digest = hashlib.sha256(source + " ".join(flags).encode()).hexdigest()[:16]
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"gb10_twoshot_proxy-{digest}.so"
    if target.exists():
        return target, digest
    tmp = cache / f".gb10_twoshot_proxy-{digest}-{os.getpid()}.so"
    cmd = [_roce_proxy._compiler(), *flags, "-o", str(tmp), str(_PROXY_SOURCE), "-libverbs", "-lpthread"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("failed to build the GB10 two-shot proxy: " + " ".join(cmd) + "\n" + proc.stderr)
    os.replace(tmp, target)
    return target, digest


def _load_library() -> ctypes.CDLL:
    global _LIB, _LIB_DIGEST
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        path, digest = _build_library()
        lib = ctypes.CDLL(str(path), use_errno=True)
        u64, p, i32, u32 = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32
        lib.gb10ts_abi_version.restype = i32
        lib.gb10ts_abi_version.argtypes = []
        lib.gb10ts_tick_hz.restype = u64
        lib.gb10ts_tick_hz.argtypes = []
        lib.gb10ts_layout.restype = i32
        lib.gb10ts_layout.argtypes = [i32, u64, ctypes.POINTER(u64)]
        lib.gb10ts_blob_bytes.restype = u64
        lib.gb10ts_blob_bytes.argtypes = []
        lib.gb10ts_create.restype = p
        lib.gb10ts_create.argtypes = [i32, i32, ctypes.c_char_p, i32, i32, p, u64, u64, u64, ctypes.c_char_p, u64]
        lib.gb10ts_local_blob.restype = i32
        lib.gb10ts_local_blob.argtypes = [p, p, u64]
        lib.gb10ts_connect.restype = i32
        lib.gb10ts_connect.argtypes = [p, p, u64]
        lib.gb10ts_start.restype = i32
        lib.gb10ts_start.argtypes = [p, i32, i32, u64, u64, u64, u32]
        lib.gb10ts_stop.restype = None
        lib.gb10ts_stop.argtypes = [p]
        lib.gb10ts_failed.restype = i32
        lib.gb10ts_failed.argtypes = [p]
        lib.gb10ts_error.restype = ctypes.c_char_p
        lib.gb10ts_error.argtypes = [p]
        lib.gb10ts_stat.restype = u64
        lib.gb10ts_stat.argtypes = [p, i32]
        lib.gb10ts_prof_read.restype = u64
        lib.gb10ts_prof_read.argtypes = [p, ctypes.POINTER(u64), u64]
        lib.gb10ts_arrive_read.restype = u64
        lib.gb10ts_arrive_read.argtypes = [p, ctypes.POINTER(u64), u64]
        lib.gb10ts_reduce.restype = i32
        lib.gb10ts_reduce.argtypes = [u32, ctypes.POINTER(p), p, u64, i32]
        lib.gb10ts_destroy.restype = None
        lib.gb10ts_destroy.argtypes = [p]
        if lib.gb10ts_abi_version() != 1:
            raise RuntimeError("unexpected GB10 two-shot proxy ABI version")
        _LIB, _LIB_DIGEST = lib, digest
        return lib


def cpu_reduce(arrays: Sequence[torch.Tensor], out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The proxy's CPU reduction (test hook): fp32 sum in the order of ``arrays``, one rounding to the element type.

    ``arrays`` are contiguous CPU tensors of one dtype and equal size (a multiple of 16 bytes)."""
    lib = _load_library()
    first = arrays[0]
    if out is None:
        out = torch.empty_like(first)
    ptrs = (ctypes.c_void_p * len(arrays))(*[a.data_ptr() for a in arrays])
    if lib.gb10ts_reduce(_DT_CODE[first.dtype], ptrs, out.data_ptr(), first.numel() * first.element_size() // PACK_BYTES,
                         len(arrays)) != 0:
        raise ValueError("unsupported dtype or world size for the CPU reduction")
    return out


def fast_cpu_cores() -> list[int]:
    """The allowed cores whose cpu_capacity is within 10 % of the largest (Cortex-X925 on the GB10), ascending."""
    caps: dict[int, int] = {}
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        try:
            caps[int(path.name[3:])] = int((path / "cpu_capacity").read_text())
        except (OSError, ValueError):
            continue
    allowed = os.sched_getaffinity(0)
    caps = {c: v for c, v in caps.items() if c in allowed}
    if not caps:
        return []
    top = max(caps.values())
    return sorted(c for c, v in caps.items() if v >= 0.9 * top)


def _resolve_proxy_cpu(value: Any) -> int:
    """-1 (no pinning), an explicit core, or "auto" (the last of the fast cores)."""
    if value is None:
        value = os.getenv("GB10_TWOSHOT_PROXY_CPU", "-1")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "off", "none", "no", "-1"):
            return -1
        if text == "auto":
            cores = fast_cpu_cores()
            return cores[-1] if cores else -1
        return int(text)
    return int(value)


class _Layout:
    __slots__ = ("send_off", "recv1_off", "recv2_off", "flag1_off", "flag2_off", "ctrl_off", "total_bytes", "slot_bytes",
                 "cslot_bytes", "flag_stride", "slots", "rails", "ctrl_bytes")

    def __init__(self, world_size: int, max_bytes: int) -> None:
        out = (ctypes.c_uint64 * 13)()
        if _load_library().gb10ts_layout(int(world_size), int(max_bytes), out) != 0:
            raise ValueError(f"unsupported two-shot geometry: world_size={world_size} max_bytes={max_bytes}")
        for name, value in zip(self.__slots__, out):
            setattr(self, name, int(value))


def _device_pointer(host_ptr: int) -> int:
    from cuda.bindings import runtime as cudart

    err, ptr = cudart.cudaHostGetDevicePointer(host_ptr, 0)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
    return int(ptr)


class GB10TwoShotAllReduce:
    """Two-shot RDMA all-reduce over the DGX Spark RoCE fabric (bf16, fp16, fp32)."""

    algorithm = "gb10-twoshot"

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_bytes: int = DEFAULT_MAX_BYTES,
        hca_names: Optional[Sequence[str]] = None,
        gid_index: Optional[int] = None,
        threads: Optional[int] = None,
        blocks: Optional[int] = None,
        packs_per_thread: Optional[int] = None,
        proxy_cpu: Any = None,
        proxy_timeout_s: Optional[float] = None,
        spin_limit: Optional[int] = None,
        profile: bool = False,
        gpu_profile: Optional[bool] = None,
        prefetch_bytes: Optional[int] = None,
        piece_bytes: Optional[int] = None,
        idle_ms: Optional[int] = None,
        options: Optional[int] = None,
    ) -> None:
        """Allocate the pinned region, build and connect the proxy, compile the kernel and start the proxy thread."""
        self.device = _roce._normalize_device(device)
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self._group = exchange_group
        self._closed = False
        self._lock = threading.Lock()
        self._proxy_ctx = None
        self._lib: Optional[ctypes.CDLL] = None
        self._align_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._stream_event = torch.cuda.Event()
        self._last_stream: Optional[torch.cuda.Stream] = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._capture_id = 0
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"unsupported two-shot world size {self.world_size}; supported {SUPPORTED_WORLD_SIZES}")
        threads = _roce._env_int("GB10_TWOSHOT_THREADS", default=DEFAULT_THREADS) if threads is None else int(threads)
        blocks = _roce._env_int("GB10_TWOSHOT_BLOCKS", default=DEFAULT_BLOCKS) if blocks is None else int(blocks)
        if int(max_bytes) < PACK_BYTES or int(max_bytes) % PACK_BYTES:
            raise ValueError("max_bytes must be a positive multiple of 16")
        if int(threads) % 32 or not 32 <= int(threads) <= 1024:
            raise ValueError("threads must be a multiple of 32 between 32 and 1024")
        if int(threads) < self.world_size:
            raise ValueError("threads must be at least the world size (one flag-polling thread per rank)")
        if int(blocks) < 1 or int(blocks) & (int(blocks) - 1):
            # The kernel finds the last arriving block with a modulo of a free-running 32-bit counter, which survives
            # the counter wrap only for a power-of-two block count.
            raise ValueError("blocks must be a positive power of two")
        packs_per_thread = (_roce._env_int("GB10_TWOSHOT_PACKS_PER_THREAD", default=DEFAULT_PACKS_PER_THREAD)
                            if packs_per_thread is None else int(packs_per_thread))
        self.max_bytes = int(max_bytes)
        self._threads = int(threads)
        self._blocks = int(blocks)
        self._packs_per_thread = max(1, int(packs_per_thread))
        self._counter_classes = self._blocks.bit_length()
        self.gid_index = _roce.default_gid_index() if gid_index is None else int(gid_index)
        self.spin_limit = int(spin_limit) if spin_limit is not None else _roce._env_int(
            "GB10_TWOSHOT_SPIN_LIMIT", "B12X_ROCE_SPIN_LIMIT", default=DEFAULT_SPIN_LIMIT)
        self.proxy_timeout_s = float(proxy_timeout_s) if proxy_timeout_s is not None else float(
            os.getenv("GB10_TWOSHOT_PROXY_TIMEOUT_S", DEFAULT_PROXY_TIMEOUT_S))
        self.proxy_cpu = _resolve_proxy_cpu(proxy_cpu)
        self._profile = bool(profile) or os.getenv("GB10_TWOSHOT_PROFILE", "0") == "1"
        self.prefetch_bytes = int(prefetch_bytes) if prefetch_bytes is not None else _roce._env_int(
            "GB10_TWOSHOT_PREFETCH_BYTES", default=DEFAULT_PREFETCH_BYTES)
        self.piece_bytes = int(piece_bytes) if piece_bytes is not None else _roce._env_int(
            "GB10_TWOSHOT_PIECE_BYTES", default=DEFAULT_PIECE_BYTES)
        self.idle_ms = int(idle_ms) if idle_ms is not None else _roce._env_int(
            "GB10_TWOSHOT_IDLE_MS", default=DEFAULT_IDLE_MS)
        self.options = int(options) if options is not None else _roce._env_int("GB10_TWOSHOT_OPTIONS", default=0)
        self._gpu_profile = bool(gpu_profile) if gpu_profile is not None else os.getenv("GB10_TWOSHOT_GPU_PROFILE", "0") == "1"
        names = tuple(hca_names) if hca_names else _roce.discover_hcas(self.gid_index)
        if not names:
            raise RuntimeError("no active RDMA device found for the two-shot all-reduce")
        self.hca_name = names[0]  # one rail: the first device (the GB10 nodes have one cabled port)
        if len(names) > 1:
            logger.warning("GB10 two-shot uses one RDMA device; using %s of %s", self.hca_name, ",".join(names))

        self._lib = _load_library()
        self._layout = _Layout(self.world_size, self.max_bytes)
        lay = self._layout
        self._slot_bytes, self._cslot_bytes = lay.slot_bytes, lay.cslot_bytes
        with torch.cuda.device(self.device):
            # Pinned, zero-initialised: the flags and the control record start at 0 and the first sequence is 1.
            self._region = torch.zeros(lay.total_bytes, dtype=torch.uint8, pin_memory=True)
            # Epoch and poison are global.  Stage and tail arrivals are separate for every power-of-two grid size
            # because their free-running counters use the launch grid as their modulus.
            self._counters = torch.zeros(2 + 2 * self._counter_classes, dtype=torch.int32, device=self.device)
        host_ptr = self._region.data_ptr()
        if _device_pointer(host_ptr) != host_ptr:
            raise RuntimeError("the two-shot all-reduce needs host pointers that are directly device accessible "
                               "(integrated GPU with unified addressing)")
        self._send_base = host_ptr + lay.send_off
        self._recv2_base = host_ptr + lay.recv2_off
        self._flag2_base = host_ptr + lay.flag2_off
        self._ctrl_base = host_ptr + lay.ctrl_off
        self._ctrl_words = self._region[lay.ctrl_off: lay.ctrl_off + lay.ctrl_bytes].view(torch.int32)
        self._ctrl_np = self._ctrl_words.numpy()  # reading it costs nanoseconds: the health checks stay off the profile
        with torch.cuda.device(self.device):
            # GPU timestamp rows (profiling builds of the kernel only); a dummy word otherwise
            self._gpu_prof = torch.zeros(8192 * 8 if self._gpu_profile else 8, dtype=torch.int64, pin_memory=True)
        self._prof_address = self._gpu_prof.data_ptr()
        self._epoch_address = self._counters.data_ptr()
        self._poison_address = self._epoch_address + 4 * (1 + 2 * self._counter_classes)

        error: Optional[str] = None
        blob = b""
        endpoints: Any = ()
        traffic_class = None
        self._launcher = None
        try:
            self._launcher = _cute.get_launcher(self.world_size, self._threads, lay.slots, lay.flag_stride, lay.rails,
                                                self.device.index, self._gpu_profile)
            traffic_class = _roce_proxy._traffic_class()
            err = ctypes.create_string_buffer(512)
            ctx = self._lib.gb10ts_create(
                self.world_size, self.rank, self.hca_name.encode(), self.gid_index, traffic_class,
                ctypes.c_void_p(host_ptr), lay.total_bytes, self.max_bytes, int(self.proxy_timeout_s * 1000), err,
                len(err))
            if not ctx:
                raise RuntimeError(f"two-shot proxy setup failed: {err.value.decode(errors='replace')}")
            self._proxy_ctx = ctx
            n = int(self._lib.gb10ts_blob_bytes())
            buf = ctypes.create_string_buffer(n)
            if self._lib.gb10ts_local_blob(ctx, buf, n) != 0:
                raise RuntimeError("two-shot proxy blob size mismatch")
            blob = buf.raw
            endpoints = _roce_routes.local_endpoints((self.hca_name,), self.gid_index)
        except Exception as exc:  # noqa: BLE001 - reported collectively below
            error = f"{type(exc).__name__}: {exc}"
        # Every rank publishes the configuration the protocol depends on; the ranks must agree exactly.
        config = {
            "api_version": API_VERSION,
            "proxy_abi": self._lib.gb10ts_abi_version(),
            "proxy_digest": _LIB_DIGEST,
            "kernel_digest": _cute._SRC_HASH,
            "world_size": self.world_size,
            "traffic_class": traffic_class,
            "max_bytes": self.max_bytes,
            "slot_bytes": lay.slot_bytes,
            "cslot_bytes": lay.cslot_bytes,
            "slots": lay.slots,
            "rails": lay.rails,
            "flag_stride": lay.flag_stride,
            "spin_limit": self.spin_limit,
            "threads": self._threads,
            "blocks": self._blocks,
            "piece_bytes": self.piece_bytes,  # sender and receiver must cut a chunk into the same pieces
        }
        statuses = _roce._exchange((error, blob, config, endpoints), exchange_group)
        failures = [f"rank {i}: {s[0]}" for i, s in enumerate(statuses) if s[0] is not None]
        if not failures:
            reference = statuses[0][2]
            for i, s in enumerate(statuses):
                differing = {k: (reference[k], s[2].get(k)) for k in reference if s[2].get(k) != reference[k]}
                if differing:
                    failures.append(f"rank {i} configuration differs from rank 0: {differing}")
        if failures:
            self.close()
            raise RuntimeError("two-shot all-reduce setup failed: " + "; ".join(failures))
        try:
            # One rail: every rank pairs its only device with the peers' only device; plan_routes verifies the
            # subnets (or trusts the device order when a GID is not IPv4-mapped).
            _roce_routes.plan_routes([s[3] for s in statuses], self.rank, 1)
            joined = b"".join(s[1] for s in statuses)
            if self._lib.gb10ts_connect(self._proxy_ctx, ctypes.create_string_buffer(joined, len(joined)),
                                        len(joined)) != 0:
                raise RuntimeError(f"two-shot queue-pair connect failed: {self._error_text()}")
            if self._lib.gb10ts_start(self._proxy_ctx, self.proxy_cpu, 1 if self._profile else 0,
                                      self.prefetch_bytes, self.piece_bytes, self.idle_ms, self.options) != 0:
                raise RuntimeError(f"two-shot proxy thread failed to start: {self._error_text()}")
            error = None
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        verdicts = _roce._exchange(error, exchange_group)
        failures = [f"rank {i}: {v}" for i, v in enumerate(verdicts) if v is not None]
        if failures:
            self.close()
            raise RuntimeError("two-shot all-reduce connect failed: " + "; ".join(failures))
        if self.rank == 0:
            logger.info("GB10 two-shot ready: world=%d hca=%s gid_index=%d max_bytes=%d proxy_cpu=%d threads=%d blocks=%d",
                        self.world_size, self.hca_name, self.gid_index, self.max_bytes, self.proxy_cpu, self._threads,
                        self._blocks)
        # Allocate the alignment scratch before any capture.
        self._aligned_scratch(0, self._region[:PACK_BYTES])

    # -- helpers ----------------------------------------------------------------------------------

    def _error_text(self) -> str:
        raw = self._lib.gb10ts_error(self._proxy_ctx) if (self._lib is not None and self._proxy_ctx) else None
        return raw.decode(errors="replace") if raw else ""

    def _counter_addresses(self, blocks: int) -> tuple[int, int]:
        """Stage and tail counter addresses for one power-of-two grid size."""
        counter_class = int(blocks).bit_length() - 1
        stage = self._epoch_address + 4 * (1 + counter_class)
        tail = self._epoch_address + 4 * (1 + self._counter_classes + counter_class)
        return stage, tail

    def _blocks_for(self, packs: int) -> int:
        """Power-of-two grid with about ``packs_per_thread`` packs per thread, at most ``blocks``."""
        per_block = self._packs_per_thread * self._threads
        required = max(1, (int(packs) + per_block - 1) // per_block)
        return min(1 << (required - 1).bit_length(), self._blocks)

    def _aligned_scratch(self, which: int, like: torch.Tensor) -> torch.Tensor:
        """16-byte-aligned device scratch shaped like ``like`` (input 0, output 1); allocated before capture."""
        if self._align_buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("two-shot alignment scratch must be allocated before CUDA graph capture")
            with torch.cuda.device(self.device):
                self._align_buffers = (torch.empty(self.max_bytes, dtype=torch.uint8, device=self.device),
                                       torch.empty(self.max_bytes, dtype=torch.uint8, device=self.device))
        nbytes = like.numel() * like.element_size()
        return self._align_buffers[which][:nbytes].view(like.dtype).view(like.shape)

    # -- policy -----------------------------------------------------------------------------------

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        """Eligible: contiguous bf16/fp16/fp32 CUDA tensor on this device, 16-byte total size within ``max_bytes``.

        Depends only on dtype, shape, contiguity and size, which tensor-parallel ranks share, never on pointers or
        per-rank state; a closed runtime raises rather than declining, so no rank can silently fall back alone."""
        if self._closed or self._proxy_ctx is None:
            raise RuntimeError("two-shot runtime is closed")
        if inp.dtype not in SUPPORTED_DTYPES or not inp.is_cuda:
            return False
        if inp.device != self.device or not inp.is_contiguous():
            return False
        nbytes = inp.numel() * inp.element_size()
        return 0 < nbytes <= self.max_bytes and nbytes % PACK_BYTES == 0

    # -- execution --------------------------------------------------------------------------------

    def all_reduce(self, inp: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sum ``inp`` over the group (fp32 in rank order, one rounding to the element type); eager or under CUDA graph capture."""
        with self._lock:
            self.check_health()
            if not self.should_allreduce(inp):
                raise ValueError("input is not eligible for the GB10 two-shot all-reduce")
            if out is not None and (out.shape != inp.shape or out.dtype != inp.dtype or out.device != inp.device
                                    or not out.is_contiguous()):
                raise ValueError("out must be a contiguous tensor on the input's device matching the input")
            nbytes = inp.numel() * inp.element_size()
            with torch.cuda.device(self.device):
                # Capture state belongs to the current stream, so query it here.
                capturing = torch.cuda.is_current_stream_capturing()
                if out is None:
                    out = torch.empty_like(inp)
                src = inp
                if inp.data_ptr() % PACK_BYTES != 0:
                    src = self._aligned_scratch(0, inp)
                    src.copy_(inp)
                dst = out if out.data_ptr() % PACK_BYTES == 0 else self._aligned_scratch(1, out)
                blocks = self._blocks_for(nbytes // PACK_BYTES)
                stage_counter, tail_counter = self._counter_addresses(blocks)
                self._order_stream(capturing)
                self._launcher(
                    src.data_ptr(), dst.data_ptr(), nbytes // PACK_BYTES, nbytes, _DT_CODE[inp.dtype],
                    self._send_base, self._recv2_base, self._flag2_base, self._ctrl_base,
                    self._slot_bytes, self._cslot_bytes, self._epoch_address, stage_counter, tail_counter,
                    self._poison_address, self.spin_limit, self._prof_address, blocks)
                if dst is not out:
                    out.copy_(dst)
                self._mark_stream(capturing)
        if not capturing:
            self.check_health()
        return out

    def _order_stream(self, capturing: bool) -> None:
        """Make the current stream wait for the previous collective's stream.

        The runtime has one epoch, one doorbell and two transport slots, so collectives must execute in launch order
        even across streams.  Under capture no event is used: every collective must be on the capture stream, which
        keeps the graph's own ordering; captures are told apart by CUDA's capture id, so consecutive captures may use
        different streams."""
        current = torch.cuda.current_stream(self.device)
        if capturing:
            capture_id = _roce._capture_id(current)
            if capture_id != self._capture_id:
                self._capture_id = capture_id
                self._capture_stream = current
            elif current != self._capture_stream:
                raise RuntimeError("two-shot collectives must all be captured on one stream")
            return
        if self._last_stream is not None and current != self._last_stream:
            current.wait_event(self._stream_event)

    def _mark_stream(self, capturing: bool) -> None:
        """Record the ordering event on the current stream after a launch."""
        if capturing:
            return
        current = torch.cuda.current_stream(self.device)
        self._stream_event.record(current)
        self._last_stream = current

    @contextmanager
    def capture(self):
        """Context manager around CUDA graph capture (``torch.cuda.graph``); the runtime must be constructed.

        The cross-stream ordering event is dropped for the capture: an event recorded outside a capture must not be
        waited on inside one.  Inside, every collective must use one stream."""
        self._last_stream = None
        self._capture_stream = None
        self._capture_id = 0
        try:
            yield self
        finally:
            self._capture_stream = None
            self._capture_id = 0
            self._last_stream = None

    def check_health(self) -> None:
        """Raise if the proxy thread died or a kernel wait timed out / was aborted.  Cheap (host memory reads); call it
        after graph replays, which cannot check inline."""
        words = self._ctrl_np
        code = int(words[_CW_PROXY_ERR])
        if code != 0:
            raise RuntimeError(
                f"GB10 two-shot proxy failed on rank {self.rank}: {_PROXY_ERRORS.get(code, code)} "
                f"(peer {int(words[_CW_PROXY_PEER])}, seq {int(words[_CW_PROXY_SEQ])}): {self._error_text()}")
        failed_seq = int(words[_CW_GPU_ERR_SEQ])
        if failed_seq != 0:
            why = "the proxy had failed" if int(words[_CW_GPU_ERR_CODE]) == 2 else "the flag never arrived"
            raise RuntimeError(
                f"GB10 two-shot collective on rank {self.rank} gave up waiting for rank {int(words[_CW_GPU_ERR_PEER])}'s "
                f"reduced chunk at sequence {failed_seq} ({why}); the runtime is poisoned (its epoch stopped at "
                f"{failed_seq - 1}, later launches do nothing) and rank data is no longer trustworthy")

    @property
    def poisoned(self) -> bool:
        """True once a wait gave up or the proxy failed; the runtime cannot be reused."""
        return int(self._ctrl_np[_CW_PROXY_ERR]) != 0 or int(self._ctrl_np[_CW_GPU_ERR_SEQ]) != 0

    def warmup(self, sizes: Sequence[int] = (8, 6 * 5120, 12 * 5120),
               dtypes: Sequence[torch.dtype] = (torch.bfloat16,)) -> None:
        """Collective self-test: rank r contributes the constant r + 1; the exact sum is representable in every dtype."""
        want = float(self.world_size * (self.world_size + 1) // 2)
        for dtype in dtypes:
            esize = torch.empty(0, dtype=dtype).element_size()
            for n in sorted({s for s in sizes if 0 < s * esize <= self.max_bytes and (s * esize) % PACK_BYTES == 0}):
                x = torch.full((n,), float(self.rank + 1), dtype=dtype, device=self.device)
                y = self.all_reduce(x)
                torch.cuda.synchronize(self.device)
                if not bool(torch.all(y == want)):
                    raise RuntimeError(f"GB10 two-shot warm-up wrong on rank {self.rank} ({dtype}, n={n})")
        self.check_health()

    # -- diagnostics / lifecycle ------------------------------------------------------------------

    def epoch(self) -> int:
        """Operations completed on the device (synchronizes)."""
        return int(self._counters[0].item())

    def stats(self) -> dict[str, Any]:
        """Runtime and proxy counters for diagnostics."""
        info: dict[str, Any] = {
            "world_size": self.world_size, "rank": self.rank, "hca": self.hca_name, "max_bytes": self.max_bytes,
            "slot_bytes": self._slot_bytes, "cslot_bytes": self._cslot_bytes, "spin_limit": self.spin_limit,
            "proxy_cpu": self.proxy_cpu, "threads": self._threads, "blocks": self._blocks,
            "prefetch_bytes": self.prefetch_bytes, "piece_bytes": self.piece_bytes, "idle_ms": self.idle_ms, "options": self.options,
            "ctrl_bell": int(self._ctrl_np[0]), "proxy_done": int(self._ctrl_np[_CW_PROXY_DONE]),
        }
        if self._proxy_ctx:
            lib, ctx = self._lib, self._proxy_ctx
            info.update(ops_done=int(lib.gb10ts_stat(ctx, 0)), writes_completed=int(lib.gb10ts_stat(ctx, 1)),
                        last_seq=int(lib.gb10ts_stat(ctx, 2)), bytes_posted=int(lib.gb10ts_stat(ctx, 3)),
                        reduced_packs=int(lib.gb10ts_stat(ctx, 4)), pinned=bool(lib.gb10ts_stat(ctx, 5)))
        return info

    def proxy_arrivals(self) -> np.ndarray:
        """Per-op first-seen times (ns, proxy clock) of each peer's phase-1 flag (needs ``profile=True``); rows match
        ``proxy_profile()``, columns are ranks (the own rank's column is 0)."""
        lib, ctx = self._lib, self._proxy_ctx
        rows = int(lib.gb10ts_stat(ctx, 6))
        buf = (ctypes.c_uint64 * (rows * 16))()
        n = int(lib.gb10ts_arrive_read(ctx, buf, rows))
        arr = np.frombuffer(buf, dtype=np.uint64, count=n * 16).reshape(n, 16).astype(np.int64)
        return arr[:, : self.world_size]

    def proxy_profile(self) -> np.ndarray:
        """Per-op proxy timestamps (needs ``profile=True``): rows [seq, bell seen, phase 1 posted (all pieces), every
        peer's first piece in, first piece reduced, last piece reduced and posted, flag2 stored, nbytes], times in ns
        (oldest first)."""
        lib, ctx = self._lib, self._proxy_ctx
        rows = int(lib.gb10ts_stat(ctx, 6))
        buf = (ctypes.c_uint64 * (rows * 8))()
        n = int(lib.gb10ts_prof_read(ctx, buf, rows))
        arr = np.frombuffer(buf, dtype=np.uint64, count=n * 8).reshape(n, 8).astype(np.int64)
        hz = int(lib.gb10ts_tick_hz())
        arr[:, 1:7] = arr[:, 1:7] * (1_000_000_000 // hz) if hz <= 1_000_000_000 else arr[:, 1:7]
        return arr

    def gpu_profile(self) -> np.ndarray:
        """GPU timestamp rows (needs ``gpu_profile=True``): [seq, entry b0, staged b0, doorbell section done b0, doorbell
        rung (last block), spin done b0, copy done b0, epoch store (last block)], globaltimer ns; rows with seq 0 unused."""
        return self._gpu_prof.numpy().reshape(8192, 8).copy()

    def close(self) -> None:
        """Stop the proxy and release the transport.

        Waits for the device first so no in-flight kernel still reads the pinned region or rings the doorbell after
        the proxy is gone.  Peers that write to this rank afterwards see a remote-access completion error and raise on
        their side, which is the intended shutdown signal."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            with contextlib.suppress(Exception):
                torch.cuda.synchronize(self.device)
            if self._proxy_ctx:
                ctx, self._proxy_ctx = self._proxy_ctx, None
                self._lib.gb10ts_destroy(ctx)

    def __del__(self) -> None:  # pragma: no cover - defensive teardown (module globals may be gone at exit)
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["API_VERSION", "DEFAULT_MAX_BYTES", "GB10TwoShotAllReduce", "SUPPORTED_DTYPES", "SUPPORTED_WORLD_SIZES",
           "cpu_reduce", "fast_cpu_cores"]
