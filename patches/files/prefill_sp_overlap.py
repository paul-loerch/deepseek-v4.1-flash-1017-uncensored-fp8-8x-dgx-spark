# SPDX-License-Identifier: Apache-2.0
# [patch-028 prefill-sp-overlap] installed as vllm/_prefill_sp_overlap.py by 028-prefill-sp-overlap.sh (opt-in VLLM_GB10_PREFILL_SP_OVERLAP=1).
"""Two-half communication/compute overlap for the patch-022 sequence-parallel prefill region of DeepSeek-V4.1 on 8 x GB10.
Same math per row as the plain SP path; only the NCCL reduction order / GEMM tiling can differ (as in 022). Needs
VLLM_GB10_PREFILL_SP=1 (patch 022 and its vllm/_prefill_sp.py are imported, reused and left untouched).

WHY. The 022 region (layers 0-19) runs, per layer, all-gather -> attention -> reduce-scatter -> mHC -> all-gather -> MoE ->
reduce-scatter -> mHC with every collective EXPOSED: ~534 ms of the ~1.31 s kernel time of an 8,192-token chunk is NCCL, nothing
overlaps it. Splitting the forward's T rows into two halves H1 | H2 lets one half's RS -> mHC -> AG chain run on a
communication stream while the other half's attention / MoE runs on the compute stream.

SCHEDULE (A = attention block, M = MoE block, per layer L; one fixed program order, identical on every rank):
    compute stream : A1(L) A2(L) M1(L) M2(L) A1(L+1) ...
    comm stream    :        [RS mHC AG](A1) [RS mHC AG](A2) [RS mHC AG](M1) [RS mHC AG](M2) ...
Each RS -> mHC -> AG chain of a half runs behind exactly one compute block of the OTHER half. ALL NCCL collectives (and the
small mHC / Engram boundary kernels between them) are issued on ONE dedicated high-priority stream; the compute stream only
runs attention and MoE (which have their TP all-reduces suppressed, exactly as in 022). Every cross-stream dependency is a CUDA
event; every tensor that crosses streams gets record_stream() so the caching allocator cannot recycle it early.

THE SPLIT (plan_split). Rows [0, T) are cut at a request boundary or, inside a PREFILLING request, at a sequence position that
is a multiple of 128; both halves keep >= 1,024 rows; the cut closest to T/2 wins. Per-half SP ownership: rank r owns S_h =
8 * ceil(T_h / 64) rows of each half. The second half of a straddling request sees the first half's tokens as context, exactly
the shape the V2 runner's own DBO helpers build (ubatch_utils._slice_input_batch): query offsets rebased, seq_lens truncated
only past the half's end.

PER-HALF STATE (audited in tests/028-overlap/NOTES.md):
  * attention metadata: built per half with OWN metadata-builder copies (builders keep the metadata they built), through the
    base ModelState.prepare_attn (the DSV4.1 override would overwrite the shared replay-start buffer and the replay batch);
  * topk_indices_buffer (written by index-source layers 2/8/14, read by the layers after them): the second half gets its own
    buffer (attribute swap per layer call), otherwise H2's layer-2 indexer would overwrite H1's rows before H1 layer 3 reads them;
  * Engram: the staged rows of the whole forward are fetched ONCE per layer (NVMe fetch is once per publish) and sliced per half;
  * forward context: a copy per half (own attention metadata / slot mappings) overrides the full one only inside a compute block;
    the full context stays untouched for the bounded replay (layer 20 write_kv + layers 20-39 on the last 128 rows).

RUNTIME SWITCH. VLLM_GB10_PREFILL_SP_OVERLAP_CTL=<file>: rank 0 reads it at the region of every eligible forward and broadcasts the
text over the TP group (gloo); "on" / "off" (+ "log=<n>"). Counters and region timing are logged every N eligible
forwards. FAIL CLOSED: the first eligible overlapped forward is a self-test that runs the region both ways (plain SP, then overlap)
on the same inputs, compares the carried states and raises a [patch-028] error on every rank on a mismatch; all waits are bounded.

SELF-TEST DIAGNOSTICS (revision 2). Both paths are also run a second time on the same inputs (the run-to-run noise floor of each: the MoE
kernel is not bit-deterministic, and top-k routing turns tiny input differences into discrete expert flips, so a single layer's output
can differ by tens of percent between two runs of the SAME path). When the verdict is "fail", the report (rank-0 log + the error text)
adds: every state's RMS / per-half RMS / per-row error distribution / worst rows; the noise floors; the same pipeline run serially on
ONE stream (separates a scheduling problem from numerics); a per-layer trace (x and residual, H1 | H2 separately, all runs) and the
per-layer expert-routing flip rate with the share of the x error carried by the flipped tokens; and plain-language hints. The default
criterion is unchanged (absolute: RMS <= RTOL per state). Opt-in: VLLM_GB10_PREFILL_SP_OVERLAP_CRITERION=noise (bound = max(0.02,
K x run-to-run noise), per state and per half) and VLLM_GB10_PREFILL_SP_OVERLAP_SELFTEST=report (diagnosis only: a purely numeric failure
is logged with the full report and not enforced).

INDEXER ROW SHARDING (revision 3). vLLM can shard the sparse-attention indexer's prefill rows over the TP ranks
(attention_config.indexer_tp_shard_min_rows): every rank scores only its slice of the rows and the ranks exchange the final top-k
indices with the TP group's all_gatherv (PrefillRowShard.exchange_topk, issued on the CURRENT stream). Revision 2 refused that
configuration: the exchange runs on the compute stream while this module drives the TP communicator from the communication stream,
and one NCCL communicator driven from two streams can deadlock (the ranks may interleave the two streams' collectives differently).
VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER (default off = revision 2: sharding is refused at the preflight) selects:
  comm2      the exchange gets its OWN NCCL communicator (a second PyNcclCommunicator over the TP group's CPU group, same 8 ranks), created
             and warmed up collectively at the preflight, and PrefillRowShard.exchange_topk is routed through it for every prefill of the
             process (two-half and plain). Each communicator is then used from exactly one stream at a time: the TP communicator from
             the communication stream inside a region, the new one from the compute stream only; the per-rank program order is fixed, so
             no two ranks can disagree about the order of any communicator's collectives (section 8, tests/028-overlap/NOTES.md 11).
  exclusive  no new communicator: a forward whose indexer metadata is sharded runs the plain 022 region (counter skip_sharded), an
             unsharded one runs two-half; the halves' own metadata is never sharded. Together with VLLM_GB10_PREFILL_SP_OVERLAP_IDX_MIN_POS
             (a depth gate on vLLM's sharding decision: rows are sharded only when the mean number of keys per row reaches the threshold)
             this splits the work by depth: shallow chunks overlap, deep chunks shard.
Nothing in vLLM's files is edited: the exchange and the gate are installed as runtime wrappers (guarded by source hashes).

POOL DISCIPLINE (revision 4). PyTorch's caching allocator keeps one pool per stream: a block allocated while the communication stream is current can
only be reused by that stream, and a block that crossed streams through record_stream() is not reusable until the GPU has passed the free point AS THE
HOST SEES IT -- the host issues the whole region (~80 chains and ~80 compute blocks, no host synchronisation) long before the GPU has executed it, so
within the region that means "practically never". Revision 3 allocated every gather output, every
zero-padded scatter input and every mHC / Engram intermediate on the communication stream and protected each crossing tensor with record_stream(): the
region then held ~4-6 GiB more RESERVED memory than the plain path although its tensors peak lower (on GB10's unified memory reserved memory is gone from
MemAvailable). Revision 4:
  (1) the outputs of the pipeline's all-gathers (the gathered input of every block, the aux taps, the exit states) and the padded reduce-scatter inputs
      come from the COMPUTE stream's pool: a TorchFunctionMode around the collective serves the large torch.empty() of the collective from there, the
      kernel still runs on the communication stream (VLLM_GB10_PREFILL_SP_OVERLAP_POOL=comm switches this off: revision 3's placement, the gathered
      input stays protected by record_stream as there). The group's out-of-place all-gather is called directly inside the scope: vLLM's own
      GroupCoordinator.all_gather goes through torch.ops.vllm.all_gather, an op-level call that a TorchFunctionMode runs with the mode stashed;
  (2) nothing that crosses streams inside the loop relies on record_stream() deferral any more: the pipeline holds it in a per-half keep list and drops
      the list when the compute stream has waited for the chain that used it (event DAG, not host timing), so a block is recycled at the earliest provably
      safe point whatever the host-to-GPU lead is. Every redirected allocation is ordered after the compute-stream work queued before it (the previous
      owners of the block) by an event the communication stream waits for;
  (3) what the communication stream still allocates are the per-rank mHC / Engram intermediates (at most S rows each), which it recycles in stream order
      at once: a pool of ~0.2-0.3 GiB at production sizes instead of every chain's gather outputs at the same time.
VLLM_GB10_PREFILL_SP_OVERLAP_EMPTY_CACHE=1 additionally calls torch.cuda.empty_cache() at the end of every overlapped region (emergency fallback: it
synchronises the device and returns the compute pool's cached blocks too -- costs throughput). The stats line reports torch.cuda.memory_allocated /
memory_reserved (+ peaks), the lowest MemAvailable since the last line and, on the first lines, the reserved size of each stream's pool.

SELF-TEST MEMORY (revision 5). The boot self-test runs the region four times (plain, plain again, overlap, overlap again) and compares the carried
states. Its extra memory is now bounded without touching the verdict: (1) at most ONE region result (the reference, 400 MiB at T = 8,192 without aux
taps) is retained at any time -- each pair is compared as soon as both exist and the older one is dropped; (2) every temporary of the comparison, of the
per-layer trace and of the routing probe is bounded by SELFTEST_TEMP_BYTES (16 MiB) instead of a fixed row count (the old 512-row chunks of the
float64 residual comparison held ~300 MiB); (3) the noise-run aux dictionaries are cleared at once. The verdict, its thresholds and the ACTIVE line are
unchanged (the per-row sums are the same float64 sums, in smaller chunks). A new INFO line per rank, "[patch-028] self-test memory", reports MemAvailable
(start / lowest with the phase it happened in / end), the /proc/meminfo lines that move (MemFree, Buffers, Cached, Slab, AnonPages, Shmem, Mlocked,
Unevictable, SwapCached), the allocator's allocated / reserved bytes and the worker's child processes (JIT compilers), sampled at the phase boundaries
and every 100 ms (children every 500 ms), so any memory event around the self-test can be attributed.
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import hashlib
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm import _prefill_sp as sp

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001 - CPU tests outside a vLLM install
    logger = logging.getLogger(__name__)

TAG = "[patch-028]"
ENV = "VLLM_GB10_PREFILL_SP_OVERLAP"
ENV_MIN_T = "VLLM_GB10_PREFILL_SP_OVERLAP_MIN_T"
ENV_CTL = "VLLM_GB10_PREFILL_SP_OVERLAP_CTL"
ENV_LOG = "VLLM_GB10_PREFILL_SP_OVERLAP_LOG_EVERY"
ENV_PRIO = "VLLM_GB10_PREFILL_SP_OVERLAP_PRIORITY"
ENV_RTOL = "VLLM_GB10_PREFILL_SP_OVERLAP_RTOL"
ENV_REPEAT = "VLLM_GB10_PREFILL_SP_OVERLAP_SELFTEST_REPEAT"
ENV_MEM = "VLLM_GB10_PREFILL_SP_OVERLAP_MEM_GUARD_GIB"
ENV_SELFTEST = "VLLM_GB10_PREFILL_SP_OVERLAP_SELFTEST"          # enforce (default) | report: log a numeric failure with the full diagnostics, keep serving
ENV_CRITERION = "VLLM_GB10_PREFILL_SP_OVERLAP_CRITERION"        # absolute (default) | noise: judge against the run-to-run noise of the same path
ENV_NOISE_K = "VLLM_GB10_PREFILL_SP_OVERLAP_NOISE_K"            # criterion=noise: allowed multiple of the run-to-run noise (default 2)
ENV_TRACE = "VLLM_GB10_PREFILL_SP_OVERLAP_TRACE"                # 1 (default): per-layer divergence trace in the self-test (used by the failure report)
ENV_INDEXER = "VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER"            # off (default: sharding refused) | comm2 (own NCCL communicator) | exclusive (sharded forwards run plain)
ENV_IDX_MIN_POS = "VLLM_GB10_PREFILL_SP_OVERLAP_IDX_MIN_POS"    # depth gate on vLLM's row-sharding decision (mean keys per row, tokens); 0 = vLLM's own rule
ENV_POOL = "VLLM_GB10_PREFILL_SP_OVERLAP_POOL"                  # compute (default: gather outputs / padded scatter inputs from the compute stream's pool) | comm (revision 3's placement)
ENV_EMPTY_CACHE = "VLLM_GB10_PREFILL_SP_OVERLAP_EMPTY_CACHE"    # 1: torch.cuda.empty_cache() at the end of every overlapped region (emergency fallback; default 0)
REVISION = 5
POOL_MODES = ("compute", "comm")
SELFTEST_TEMP_BYTES = 16 << 20  # upper bound of the temporaries of ONE comparison / trace projection / routing-probe call of the self-test (revision 5)
STATE_METRIC_BYTES_PER_ELEM = 32    # state_metrics: float32 copies of both tensors, their difference, its float64 view and square, masks (~30 bytes per element)
SELFTEST_MEM_PERIOD_S = 0.1     # the self-test memory sampler's period
SELFTEST_SCAN_EVERY = 5          # ... and every how many samples it also looks for child processes of the worker (JIT compilers: nvcc / cicc / ptxas)
MEMINFO_FIELDS = ("MemAvailable", "MemFree", "Buffers", "Cached", "Slab", "AnonPages", "Shmem", "Mlocked", "Unevictable", "SwapCached")
REDIRECT_MIN_BYTES = 64 * 1024  # a torch.empty() of at least this size inside a collective of the communication stream is served from the compute stream's pool
POOL_SNAPSHOTS = 3              # stats lines that also report the reserved size of each stream's pool (torch.cuda.memory._snapshot() is not free)
IDX_MODES = ("off", "comm2", "exclusive")
DEFAULT_IDX_MIN_POS = 200000    # exclusive mode: ~ the depth where one chunk's sharding saving equals the overlap saving (NOTES 11.9)
IDX_MIN_POS_MAX = 1 << 24
DEFAULT_MIN_T = 2048
MIN_T_FLOOR = 2048              # two halves of >= MIN_HALF_ROWS
MIN_T_CEIL = 1 << 20
MIN_HALF_ROWS = 1024            # both halves keep at least this many rows
CUT_ALIGN = 128                 # a cut inside a request is at a sequence position that is a multiple of this
DEFAULT_LOG_EVERY = 20
DEFAULT_PRIORITY = -1           # CUDA stream priority of the communication stream (lower = higher priority)
DEFAULT_RTOL = 0.10             # self-test: ||plain - overlap|| / ||plain|| of every carried state (RMS-relative)
ROW_TOL = 0.5                   # self-test: a row whose own relative error exceeds this counts as "bad"
BAD_ROW_FRAC = 0.05             # self-test: at most this fraction of rows may be bad
DEFAULT_MEM_GUARD_GIB = 2.0     # free GPU memory required before the per-half builder copies are created
SYNC_TIMEOUT_S = 300.0          # bounded wait for the GPU work of the self-test
STAGE_LOG_MAX = 4096            # collective log entries kept per region (tests read them; production keeps the hash)
DEFAULT_NOISE_K = 2.0
NOISE_MIN_RMS = 0.02            # criterion=noise: the bound never goes below this RMS-relative error ...
NOISE_MIN_BAD = 0.02            # ... or this fraction of rows with > ROW_TOL error (a deterministic stack has a noise floor of 0)
TRACE_K = 16                    # per-layer trace: rows are projected on this many fixed random directions (JL sketch)
REPORT_TOP_ROWS = 5             # failure report: worst rows listed per state
SIZE_BUCKETS = ((4096, "2k-4k"), (8192, "4k-8k"), (1 << 62, "8k+"))     # stats line: region timing is kept per forward size class


def size_bucket(rows: int) -> str:
    for hi, name in SIZE_BUCKETS:
        if rows < hi:
            return name
    return SIZE_BUCKETS[-1][1]


# ---------------------------------------------------------------------------------------------------------------------
# env parsing (fail closed, at import under the gate)
# ---------------------------------------------------------------------------------------------------------------------
def parse_enabled(raw: str | None) -> bool:
    if raw is None:
        return False
    s = raw.strip().lower()
    if s in ("", "0", "off", "false", "no"):
        return False
    if s in ("1", "on", "true", "yes"):
        return True
    raise ValueError(f"{TAG} {ENV}={raw!r} must be unset/0 (off) or 1 (on)")


def _parse_int(name: str, raw: str | None, default: int, lo: int, hi: int) -> int:
    if raw is None or raw.strip() == "":
        return default
    s = raw.strip()
    if s.lstrip("-").isdigit() is False:
        raise ValueError(f"{TAG} {name}={raw!r} is not an integer")
    v = int(s)
    if not lo <= v <= hi:
        raise ValueError(f"{TAG} {name}={v} outside [{lo}, {hi}]")
    return v


def _parse_float(name: str, raw: str | None, default: float, lo: float, hi: float) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        v = float(raw.strip())
    except ValueError as e:
        raise ValueError(f"{TAG} {name}={raw!r} is not a number") from e
    if not lo <= v <= hi:
        raise ValueError(f"{TAG} {name}={v} outside [{lo}, {hi}]")
    return v


def parse_min_t(raw: str | None) -> int:
    return _parse_int(ENV_MIN_T, raw, DEFAULT_MIN_T, MIN_T_FLOOR, MIN_T_CEIL)


def parse_log_every(raw: str | None) -> int:
    return _parse_int(ENV_LOG, raw, DEFAULT_LOG_EVERY, 0, 1 << 30)


def parse_priority(raw: str | None) -> int:
    return _parse_int(ENV_PRIO, raw, DEFAULT_PRIORITY, -8, 0)


def parse_rtol(raw: str | None) -> float:
    return _parse_float(ENV_RTOL, raw, DEFAULT_RTOL, 1e-4, 1.0)


def parse_mem_guard(raw: str | None) -> float:
    return _parse_float(ENV_MEM, raw, DEFAULT_MEM_GUARD_GIB, 0.0, 512.0)


def parse_selftest(raw: str | None) -> str:
    s = (raw or "").strip().lower()
    if s in ("", "enforce"):
        return "enforce"
    if s == "report":
        return "report"
    raise ValueError(f"{TAG} {ENV_SELFTEST}={raw!r} must be 'enforce' (default) or 'report'")


def parse_criterion(raw: str | None) -> str:
    s = (raw or "").strip().lower()
    if s in ("", "absolute"):
        return "absolute"
    if s == "noise":
        return "noise"
    raise ValueError(f"{TAG} {ENV_CRITERION}={raw!r} must be 'absolute' (default) or 'noise'")


def parse_noise_k(raw: str | None) -> float:
    return _parse_float(ENV_NOISE_K, raw, DEFAULT_NOISE_K, 1.0, 100.0)


def parse_trace(raw: str | None = None) -> bool:
    return _parse_int(ENV_TRACE, raw if raw is not None else os.environ.get(ENV_TRACE), 1, 0, 1) == 1


def parse_indexer_mode(raw: str | None) -> str:
    s = (raw or "").strip().lower()
    if s in ("", "off", "0", "false", "no"):
        return "off"
    if s in ("comm2", "exclusive"):
        return s
    raise ValueError(f"{TAG} {ENV_INDEXER}={raw!r} must be 'off' (default), 'comm2' or 'exclusive'")


def parse_idx_min_pos(raw: str | None, mode: str) -> int:
    """The depth gate (mean keys per row, in tokens). off: must be unset/0 (the gate only exists for comm2 / exclusive); exclusive:
    DEFAULT_IDX_MIN_POS when unset (0 is allowed: no depth gate, only the exclusion)."""
    unset = raw is None or raw.strip() == ""
    if mode == "off":
        if not unset and _parse_int(ENV_IDX_MIN_POS, raw, 0, 0, IDX_MIN_POS_MAX) != 0:
            raise ValueError(f"{TAG} {ENV_IDX_MIN_POS}={raw!r} needs {ENV_INDEXER}=comm2 or exclusive")
        return 0
    return _parse_int(ENV_IDX_MIN_POS, raw, DEFAULT_IDX_MIN_POS if mode == "exclusive" else 0, 0, IDX_MIN_POS_MAX)


def indexer_settings() -> tuple[str, int]:
    """(mode, depth gate) from the environment; ValueError on a bad value."""
    mode = parse_indexer_mode(os.environ.get(ENV_INDEXER))
    return mode, parse_idx_min_pos(os.environ.get(ENV_IDX_MIN_POS), mode)


def parse_pool(raw: str | None) -> str:
    s = (raw or "").strip().lower()
    if s in ("", "compute"):
        return "compute"
    if s == "comm":
        return "comm"
    raise ValueError(f"{TAG} {ENV_POOL}={raw!r} must be 'compute' (default) or 'comm'")


def parse_empty_cache(raw: str | None = None) -> bool:
    return _parse_int(ENV_EMPTY_CACHE, raw if raw is not None else os.environ.get(ENV_EMPTY_CACHE), 0, 0, 1) == 1


def _own_sha16() -> str:
    try:
        with open(__file__, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


MODULE_SHA16 = _own_sha16()
if parse_enabled(os.environ.get(ENV)):                  # model.py imports us only under the gate: validate everything at import
    parse_min_t(os.environ.get(ENV_MIN_T))
    parse_log_every(os.environ.get(ENV_LOG))
    parse_priority(os.environ.get(ENV_PRIO))
    parse_rtol(os.environ.get(ENV_RTOL))
    parse_mem_guard(os.environ.get(ENV_MEM))
    _parse_int(ENV_REPEAT, os.environ.get(ENV_REPEAT), 1, 0, 1)
    parse_selftest(os.environ.get(ENV_SELFTEST))
    parse_noise_k(os.environ.get(ENV_NOISE_K))
    parse_trace(os.environ.get(ENV_TRACE))
    if parse_criterion(os.environ.get(ENV_CRITERION)) == "noise" and _parse_int(ENV_REPEAT, os.environ.get(ENV_REPEAT), 1, 0, 1) != 1:
        raise ValueError(f"{TAG} {ENV_CRITERION}=noise needs {ENV_REPEAT}=1 (the noise floor comes from the repeated runs)")
    parse_pool(os.environ.get(ENV_POOL))
    parse_empty_cache(os.environ.get(ENV_EMPTY_CACHE))
    indexer_settings()                                  # the installer of the depth gate runs at the end of the module


# ---------------------------------------------------------------------------------------------------------------------
# 1. The split (pure host-side arithmetic; identical on every rank because it reads scheduler-derived CPU data only)
# ---------------------------------------------------------------------------------------------------------------------
class Split(NamedTuple):
    cut: int        # first row of the second half
    kind: str       # "boundary": the cut is a request boundary; "interior": it falls inside a prefilling request
    req: int        # boundary: first request of H2; interior: the straddling request (in both halves)


class HalfSpec(NamedTuple):
    index: int      # 0 = H1, 1 = H2
    tok_lo: int
    tok_hi: int
    req_lo: int
    req_hi: int     # exclusive; a straddling request is in both halves

    @property
    def rows(self) -> int:
        return self.tok_hi - self.tok_lo

    @property
    def reqs(self) -> int:
        return self.req_hi - self.req_lo


def plan_split(qsl: np.ndarray, start_pos: np.ndarray, prefilling: np.ndarray, *, min_half: int = MIN_HALF_ROWS,
               align: int = CUT_ALIGN) -> tuple[Split | None, str]:
    """Pick the row cut of a forward. qsl: [n+1] query offsets (qsl[n] = T); start_pos: [n] absolute position of each request's
    first query row; prefilling: [n] bool. Candidates: every request boundary, and every row inside a PREFILLING request whose
    absolute position is a multiple of `align` (decode / verification rows are never cut). Both halves must have >= min_half
    rows. The candidate closest to T/2 wins (ties: the smaller cut, boundary before interior). Returns (split, reason): reason
    is "" or why no cut exists."""
    qsl = np.asarray(qsl, dtype=np.int64)
    n = len(qsl) - 1
    if n < 1:
        return None, "no requests"
    T = int(qsl[n])
    if int(qsl[0]) != 0 or np.any(np.diff(qsl) <= 0):
        return None, "malformed query offsets (empty request or not increasing)"
    if len(start_pos) < n or len(prefilling) < n:
        return None, "request arrays shorter than the batch"
    lo_c, hi_c = min_half, T - min_half
    if lo_c > hi_c:
        return None, f"T={T} < 2 x {min_half}"
    best = None                                   # (|2c - T|, c, kind rank, kind, request)

    def consider(c: int, kind: str, req: int) -> None:
        nonlocal best
        if lo_c <= c <= hi_c:
            key = (abs(2 * c - T), c, 0 if kind == "boundary" else 1, kind, req)
            if best is None or key < best:
                best = key

    for i in range(1, n):                         # the boundary between request i-1 and request i
        consider(int(qsl[i]), "boundary", i)
    for i in range(n):
        a, b = int(qsl[i]), int(qsl[i + 1])
        if b - a < 2 or not bool(prefilling[i]):
            continue
        d = (-int(start_pos[i])) % align          # rows from the request start to its first aligned position
        first = a + d if d > 0 else a + align     # row a itself is the boundary candidate, not an interior cut
        kmin = max(0, -((first - lo_c) // align))                 # ceil((lo_c - first) / align), >= 0
        kmax = min((b - 1 - first) // align, (hi_c - first) // align)
        if kmin > kmax:
            continue
        k0 = min(max((T // 2 - first) // align, kmin), kmax)
        for k in (k0 - 1, k0, k0 + 1, k0 + 2):
            if kmin <= k <= kmax:
                consider(first + k * align, "interior", i)
    if best is None:
        return None, f"no cut with both halves >= {min_half} rows at a request boundary or a {align}-aligned prefill position"
    _, c, _, kind, req = best
    return Split(int(c), kind, int(req)), ""


def half_specs(split: Split, n_reqs: int, T: int) -> tuple[HalfSpec, HalfSpec]:
    """The two halves of a forward for `split`. A straddling request belongs to both."""
    if split.kind == "boundary":
        h1 = HalfSpec(0, 0, split.cut, 0, split.req)
        h2 = HalfSpec(1, split.cut, T, split.req, n_reqs)
    else:
        h1 = HalfSpec(0, 0, split.cut, 0, split.req + 1)
        h2 = HalfSpec(1, split.cut, T, split.req, n_reqs)
    return h1, h2


def check_split_invariants(specs: tuple[HalfSpec, HalfSpec], qsl: np.ndarray, T: int) -> None:
    """Cheap structural checks of a split (used by the runner hook and the tests)."""
    h1, h2 = specs
    assert h1.tok_lo == 0 and h2.tok_hi == T and h1.tok_hi == h2.tok_lo, specs
    assert h1.rows >= MIN_HALF_ROWS and h2.rows >= MIN_HALF_ROWS, specs
    assert 0 < h1.reqs and 0 < h2.reqs, specs
    assert h1.req_hi - h2.req_lo in (0, 1), specs
    for h in specs:
        assert qsl[h.req_lo] <= h.tok_lo < h.tok_hi <= qsl[h.req_hi], (h, qsl)


# ---------------------------------------------------------------------------------------------------------------------
# 2. Runtime: streams, events, record_stream (production = CUDA; the CPU suite substitutes a tracing fake with this interface)
# ---------------------------------------------------------------------------------------------------------------------
class CudaRuntime:
    """One high-priority communication stream next to the engine's current (compute) stream. Interface used by Pipeline:
         refresh()                    the engine's current stream becomes `compute`
         timing_event(stream)         an event with timing enabled (region timing)
         begin()                      comm waits for everything already queued on compute; resets the collective log
         run(name, stream, fn, reads) fn() with `stream` current (kernels are queued on it)
         record(stream) -> event ; wait(stream, event)
         produced(t, stream)          t was written by work queued on `stream` (tracing only)
         handover(t, consumer)        t crosses to `consumer`: record_stream (keeps the allocation alive for the consumer)
         finish()                     compute waits for everything queued on comm
         bounded_sync(stream, s, what)  host wait with a deadline
         note_x(entry), check_compute(what)   (revision 3, optional) log of the indexer exchanges on the second communicator;
                                      stream guard of that communicator
       revision 4 (pool discipline, section 2b):
         keep_into(sink)              context: the tensors the redirect hands out go to `sink` (the keep list of the half whose chain runs)
         hold(t, sink, consumer)      t is read by `consumer`'s queued work: keep it alive in `sink` INSTEAD of record_stream (the caller drops the
                                      sink once the producer stream has waited for the consumer's completion event)
         pool_active() / pool_scope() is the redirect in force / context for ONE collective on the comm stream: its large output buffer comes
                                      from the compute pool
         redirect_ok(device) / redirect_alloc(func, args, kwargs, nbytes)   used by the pool_scope() mode
         alloc_compute(shape, dtype, device)   a tensor from the compute pool for the comm stream to use (ordered after the compute work queued so far)
                                      (counters of the last pipeline run: redirect_n / redirect_bytes count the collective outputs served by the
                                      redirect, pad_n / pad_bytes the tensors alloc_compute handed out)
         empty_cache(), mem_basic(), mem_stats(), pool_snapshot(), quarantine(objs)   diagnostics / the emergency fallback / error path
    """

    kind = "cuda"

    def __init__(self, device: torch.device, priority: int = DEFAULT_PRIORITY, serial: bool = False) -> None:
        self.device = device
        self.priority = priority
        self.serial = serial                      # diagnostics: the SAME pipeline on ONE stream (no concurrency, no cross-stream hazard)
        self.compute = torch.cuda.current_stream(device)
        self.comm = self.compute if serial else torch.cuda.Stream(device=device, priority=priority)
        self.log: list = []
        self.log_hash = hashlib.sha256()
        self.log_n = 0
        self.xlog_hash = hashlib.sha256()                 # the indexer exchanges (second communicator) of this pipeline run
        self.xlog_n = 0
        self.in_pipeline = False                          # between begin() and finish(): the TP communicator belongs to the comm stream
        self.sink: list | None = None                     # revision 4: keep list of the half whose chain is being issued (None outside a chain)
        self.pool = True                                  # serve the collectives' large outputs from the compute pool (ENV_POOL=compute)
        self.redirect_n = 0                               # collective outputs served from the compute pool during this pipeline run (begin() resets)
        self.redirect_bytes = 0
        self.pad_n = 0                                    # padded scatter inputs served from the compute pool during this pipeline run
        self.pad_bytes = 0

    def serial_twin(self) -> "CudaRuntime":
        return CudaRuntime(self.device, self.priority, serial=True)

    def refresh(self) -> None:
        """The engine's current stream at region entry (the stream the model forward runs on)."""
        self.compute = torch.cuda.current_stream(self.device)
        if self.serial:
            self.comm = self.compute

    def begin(self) -> None:
        self.refresh()
        if not self.serial:
            self.comm.wait_stream(self.compute)
        self.log, self.log_hash, self.log_n = [], hashlib.sha256(), 0
        self.xlog_hash, self.xlog_n = hashlib.sha256(), 0
        self.redirect_n, self.redirect_bytes, self.pad_n, self.pad_bytes, self.sink = 0, 0, 0, 0, None
        self.in_pipeline = True

    def timing_event(self, stream):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(stream)
        return ev

    def note(self, entry: tuple) -> None:
        self.log_n += 1
        self.log_hash.update(repr(entry).encode())
        if len(self.log) < STAGE_LOG_MAX:
            self.log.append(entry)

    def note_x(self, entry: tuple) -> None:
        """An indexer exchange on the second communicator (kept apart from the TP communicator's log: its count is part of the ACTIVE line)."""
        self.xlog_n += 1
        self.xlog_hash.update(repr(entry).encode())

    def check_compute(self, what: str) -> None:
        """Inside a two-half region the second communicator may only be used from the compute stream (the TP communicator from the
        communication stream): a violation is the precondition of the deadlock argument, so it stops the forward on every rank."""
        if self.in_pipeline and not self.serial:
            cur = int(torch.cuda.current_stream(self.device).cuda_stream)
            if cur != int(self.compute.cuda_stream):
                raise RuntimeError(f"{TAG} {what} was issued on a stream other than the compute stream inside the two-half region: the "
                                   "dedicated indexer communicator must only be used from the compute stream")

    def run(self, name: str, stream, fn: Callable[[], Any], reads: tuple = ()) -> Any:
        del name, reads
        with torch.cuda.stream(stream):
            return fn()

    def record(self, stream):
        ev = torch.cuda.Event()
        ev.record(stream)
        return ev

    def wait(self, stream, event) -> None:
        stream.wait_event(event)

    def produced(self, t, stream) -> None:
        pass

    def handover(self, t, consumer) -> None:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            t.record_stream(consumer)

    def finish(self) -> None:
        if not self.serial:
            self.compute.wait_event(self.record(self.comm))
        self.in_pipeline = False

    def current_matches(self, stream) -> bool:
        """vLLM's thread-local current_stream() (what PyNccl launches on) follows torch.cuda.stream(): checked by the self-test."""
        from vllm.utils.torch_utils import current_stream

        with torch.cuda.stream(stream):
            return int(current_stream().cuda_stream) == int(stream.cuda_stream)

    def bounded_sync(self, stream, timeout_s: float, what: str) -> None:
        ev = self.record(stream)
        t0 = time.monotonic()
        while not ev.query():
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"{TAG} {what}: GPU work did not finish within {timeout_s:.0f} s (a rank is stuck or the "
                                   "collective schedules diverged)")
            time.sleep(0.0005)

    # -- revision 4: pool discipline (section 2b) ------------------------------------------------------------------------
    @contextlib.contextmanager
    def keep_into(self, sink: list):
        prev, self.sink = self.sink, sink
        try:
            yield
        finally:
            self.sink = prev

    def hold(self, t, sink: list, consumer=None) -> None:
        """Keep `t` alive in `sink` for the work `consumer` has queued on it. No record_stream(): the block is recycled when the caller drops the
        sink, which it does only after the producing stream has waited for an event recorded behind that work."""
        del consumer
        sink.append(t)

    def redirect_ok(self, device) -> bool:
        """May an allocation on `device` made now (inside a collective of the communication stream) come from the compute pool?"""
        return (self.pool and self.sink is not None and self.in_pipeline and not self.serial and device is not None
                and device.type == "cuda" and (device.index is None or self.device.index is None or device.index == self.device.index))

    def redirect_alloc(self, func, args, kwargs, nbytes: int):
        """func(*args, **kwargs) -- a torch.empty() family call -- with the COMPUTE stream current: the block comes from its pool. The
        communication stream (current again afterwards) waits for an event recorded behind everything the compute stream has queued so far, which
        includes every earlier user of the block this allocation may have recycled; the tensor is parked in the chain's keep list."""
        with torch.cuda.stream(self.compute):
            out = func(*args, **kwargs)
        self.comm.wait_event(self.record(self.compute))
        self.sink.append(out)
        self.redirect_n += 1
        self.redirect_bytes += int(nbytes)
        return out

    def pool_active(self) -> bool:
        """Is the pool redirect in force right now? Inside a chain of a pipeline run with the compute pool selected; not in the serial twin (one
        stream: nothing to redirect)."""
        return not (self.serial or not self.pool or self.sink is None or not self.in_pipeline)

    def pool_scope(self):
        """Context around ONE collective issued on the communication stream (see _PoolRedirect); inert when pool_active() is False."""
        if not self.pool_active():
            return contextlib.nullcontext()
        return _PoolRedirect(self, REDIRECT_MIN_BYTES)

    def alloc_compute(self, shape, dtype, device):
        """An uninitialised tensor from the compute stream's pool for the communication stream to fill / read (the caller parks it in a keep list)."""
        if self.serial or not self.pool:
            return torch.empty(shape, dtype=dtype, device=device)
        with torch.cuda.stream(self.compute):
            t = torch.empty(shape, dtype=dtype, device=device)
        self.comm.wait_event(self.record(self.compute))
        self.pad_n += 1
        self.pad_bytes += t.numel() * t.element_size()
        return t

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def mem_basic(self):
        """(allocated, reserved) bytes of this device's caching allocator: two lock-and-read calls, cheap enough for every region."""
        return int(torch.cuda.memory_allocated(self.device)), int(torch.cuda.memory_reserved(self.device))

    def mem_stats(self) -> dict:
        s = torch.cuda.memory_stats(self.device)
        g = lambda k: int(s.get(k, 0))  # noqa: E731
        return {"allocated": g("allocated_bytes.all.current"), "allocated_peak": g("allocated_bytes.all.peak"),
                "active": g("active_bytes.all.current"), "reserved": g("reserved_bytes.all.current"),
                "reserved_peak": g("reserved_bytes.all.peak"), "inactive_split": g("inactive_split_bytes.all.current"),
                "retries": g("num_alloc_retries")}

    def pool_snapshot(self):
        """Reserved / allocated bytes of the segments of the communication stream's pool, of the compute stream's pool and of all others (a
        snapshot walks every block: used for the first stats lines only). None when the allocator cannot tell."""
        dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
        comm, compute = int(self.comm.cuda_stream), int(self.compute.cuda_stream)
        out = {"comm": [0, 0], "compute": [0, 0], "other": [0, 0]}
        for seg in torch.cuda.memory._snapshot()["segments"]:
            if seg.get("device") != dev:
                continue
            s = seg.get("stream")
            key = "comm" if s == comm else "compute" if s == compute else "other"
            out[key][0] += int(seg.get("total_size", 0))
            out[key][1] += int(seg.get("allocated_size", 0))
        return out

    def quarantine(self, objs) -> None:
        """An error inside the region may leave GPU work of the other stream reading / writing tensors the pool discipline would recycle once their
        keep lists are dropped: park them for the life of the process (the engine is going down)."""
        _QUARANTINE.append(objs)


# ---------------------------------------------------------------------------------------------------------------------
# 2b. Pool discipline: the redirect of a collective's output buffer
# ---------------------------------------------------------------------------------------------------------------------
_QUARANTINE: list = []              # objects parked for good after an error inside a region (Pipeline._quarantine)


def _itemsize(dtype) -> int:
    n = getattr(dtype, "itemsize", None)
    return int(n) if n else int(torch.empty(0, dtype=dtype).element_size())


def _numel_of(args, kwargs) -> int:
    if "size" in kwargs:
        dims = kwargs["size"]
    elif len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
        dims = args[0]
    else:
        dims = args
    n = 1
    for d in dims:
        n *= int(d)
    return n


def _alloc_info(func, args, kwargs):
    """(bytes, device) of one call of a redirectable allocation function, or None when the call is not a plain allocation (`out=`, pinned
    memory, a layout, ...): such calls are never redirected."""
    if kwargs.get("out") is not None or kwargs.get("pin_memory") or kwargs.get("layout") not in (None, torch.strided):
        return None
    if func is torch.empty:
        n, dtype, device = _numel_of(args, kwargs), kwargs.get("dtype"), kwargs.get("device")
    elif func is torch.empty_like:
        t = args[0] if args else kwargs["input"]
        n, dtype, device = int(t.numel()), kwargs.get("dtype") or t.dtype, kwargs.get("device") or t.device
    else:                                               # Tensor.new_empty(self, size, *, dtype=None, device=None)
        t = args[0]
        n, dtype, device = _numel_of(args[1:], kwargs), kwargs.get("dtype") or t.dtype, kwargs.get("device") or t.device
    if dtype is None:
        dtype = torch.get_default_dtype()
    if isinstance(device, (str, int)):
        device = torch.device(device)
    return n * _itemsize(dtype), device


class _PoolRedirect(torch.overrides.TorchFunctionMode):
    """Active around ONE collective issued on the communication stream (runtime.pool_scope()): the collective's output buffer -- a torch.empty()
    of >= min_bytes inside vLLM's all_gather -- is allocated from the COMPUTE stream's pool by the runtime (which orders it and keeps it alive);
    the collective itself still runs on the communication stream. Every other torch call passes straight through. Thread-local, so no other
    thread of the worker is affected."""

    def __init__(self, rt, min_bytes: int = REDIRECT_MIN_BYTES) -> None:
        super().__init__()
        self.rt, self.min_bytes = rt, min_bytes

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.empty or func is torch.empty_like or func is torch.Tensor.new_empty:
            try:
                info = _alloc_info(func, args, kwargs)
            except Exception:  # noqa: BLE001 - an allocation that cannot be sized is simply not redirected
                info = None
            if info is not None and info[0] >= self.min_bytes and self.rt.redirect_ok(info[1]):
                return self.rt.redirect_alloc(func, args, kwargs, info[0])
        return func(*args, **kwargs)


_RUNTIME_FACTORY: Callable[[Any], Any] | None = None     # tests only: device -> runtime with the CudaRuntime interface
_HALVES_FACTORY: Callable[[Any, Any], Any] | None = None  # tests only: (pending, model) -> (halves, env)


def _make_runtime(device):
    if _RUNTIME_FACTORY is not None:
        return _RUNTIME_FACTORY(device)
    return CudaRuntime(device, parse_priority(os.environ.get(ENV_PRIO)))


class _LoggedWorld:
    """The 022 world interface with a per-rank collective log (the schedule must be identical on every rank)."""

    kind = "p028-logged"

    def __init__(self, world, rt) -> None:
        self.w, self.rt = world, rt

    @property
    def rank(self) -> int:
        return self.w.rank

    @property
    def size(self) -> int:
        return self.w.size

    @property
    def pp_size(self) -> int:
        return getattr(self.w, "pp_size", 1)

    def all_gather_rows(self, x):
        self.rt.note(("ag", tuple(x.shape), str(x.dtype)))
        if not self.rt.pool_active():
            return self.w.all_gather_rows(x)
        with self.rt.pool_scope():                       # revision 4: the gathered buffer (padded rows x width) comes from the compute stream's pool
            return self._gather_in_scope(x)

    def _gather_in_scope(self, x):
        """The 022 world's all-gather with the group's OUT-OF-PLACE implementation called directly. On CUDA vLLM's GroupCoordinator.all_gather
        reaches that implementation through torch.ops.vllm.all_gather, and a TorchFunctionMode that sees an op-level call runs the op with the
        mode stashed (probed on the image's torch 2.14, see tests/028-overlap/NOTES.md 12.3): the torch.empty() inside vLLM's all-gather would
        never reach the redirect. The op's body is exactly group._all_gather_out_place(tensor, dim). A world without such a group (or a group of
        one rank) keeps its own path: correct, just without the redirect (the counters show it)."""
        grp = getattr(self.w, "group", None)
        fn = getattr(grp, "_all_gather_out_place", None)
        if fn is None or int(getattr(grp, "world_size", 1) or 1) <= 1:
            return self.w.all_gather_rows(x)
        return fn(x, 0)

    def reduce_scatter_rows(self, x):
        self.rt.note(("rs", tuple(x.shape), str(x.dtype)))
        return self.w.reduce_scatter_rows(x)

    def all_reduce_rows(self, x):
        self.rt.note(("ar", tuple(x.shape), str(x.dtype)))
        return self.w.all_reduce_rows(x)

    def all_gather_object(self, obj):
        return self.w.all_gather_object(obj)

    def barrier_bounded(self, timeout_s: float) -> None:
        self.w.barrier_bounded(timeout_s)


# ---------------------------------------------------------------------------------------------------------------------
# 3. Per-half forward contexts (production). The tests inject a stand-in with the HalfEnv interface instead.
# ---------------------------------------------------------------------------------------------------------------------
class HalfContext(NamedTuple):
    spec: HalfSpec
    forward_context: Any


@dataclasses.dataclass
class Pending:
    """What the runner hands over for ONE eager step: everything the region needs to build the per-half contexts lazily."""

    runner: Any
    input_batch: Any
    block_tables: Any
    slot_mappings: Any
    replay_start: Any
    num_tokens: int
    positions: Any
    split: Split
    specs: tuple[HalfSpec, HalfSpec]
    qsl: np.ndarray


class _Scratch:
    """Per-process objects created at the first overlapped forward: two copies of the region's attention groups (own metadata
    builders), the per-half query/seq-len buffers and the second half's top-k buffer."""

    def __init__(self) -> None:
        self.groups: list | None = None
        self.qsl: list | None = None
        self.seq: list | None = None
        self.topk2: Any = None
        self.bytes: int = 0


class _State:
    def __init__(self) -> None:
        self.enabled: bool | None = None
        self.min_t = DEFAULT_MIN_T
        self.pending: Pending | None = None
        self.preflight_done = False
        self.verified = False
        self.world = None
        self.scratch = _Scratch()
        self.rt: Any = None
        self.ctl_text: str | None = None
        self.ctl_on = True
        self.ctl_log: int | None = None
        self.forwards = 0                 # regions opened for an eligible (halves available) forward
        self.counters: collections.Counter = collections.Counter()
        self.timing: collections.deque = collections.deque(maxlen=256)
        self.sums: dict[tuple[str, str], list[float]] = {}     # (mode, size class) -> [n, ms, rows]
        self.last_log_forward = 0
        self.report: dict | None = None
        self.last_selftest: dict | None = None    # what the self-test's tracer / routing probe recorded (tests assert on it)
        self.degraded = False             # SELFTEST=report: the numeric self-test failure was logged, not enforced
        self.fingerprint: str | None = None
        self.keep_debug = False           # tests: keep the last Pipeline object
        self.last_pipe: Any = None
        self.idx_mode = "off"             # revision 3: set by the preflight (identical on every rank: part of the fingerprint)
        self.idx_min_pos = 0
        self.idx_comm: Any = None         # IndexerComm once PrefillRowShard.exchange_topk is routed through the second communicator
        self.idx_report: dict | None = None
        self.idx_first_logged = False     # the first model exchange on the second communicator is logged once (the setup's synthetic check is not)
        self.pool_mode = "compute"        # revision 4: set by the preflight (identical on every rank: part of the fingerprint)
        self.empty_cache = False          # revision 4: torch.cuda.empty_cache() at the end of every overlapped region (set by the preflight)
        self.min_avail_mib: float | None = None    # lowest MemAvailable (MiB) sampled at region exits since the last stats line
        self.reserved_prev: int | None = None      # torch.cuda.memory_reserved at the last stats line
        self.mem_basic: tuple[int, int] | None = None   # (allocated, reserved) at the last region exit
        self.pool_snaps_left = POOL_SNAPSHOTS
        self.pool_first_logged = False
        self.selftest_mem: Any = None     # revision 5: the last self-test's memory record (_SelfTestMem)


_STATE = _State()


def configured() -> bool:
    st = _STATE
    if st.enabled is None:
        st.enabled = parse_enabled(os.environ.get(ENV))
        st.min_t = parse_min_t(os.environ.get(ENV_MIN_T))
    return bool(st.enabled)


# --- runner hook ------------------------------------------------------------------------------------------------------
def runner_hook(runner, input_batch, batch_desc, block_tables, slot_mappings, ubatch_state, dummy_run: bool) -> None:
    """Called by the V2 model runner right before the eager model call (after the full attention metadata is built). Decides,
    from host data only, whether this step can run two-half; if so stores a Pending for the region of this forward. Never
    raises on an ineligible step. Identical decision on every rank."""
    st = _STATE
    st.pending = None
    if not configured():
        return
    try:
        if dummy_run or ubatch_state is not None or block_tables is None or slot_mappings is None:
            return
        mode = getattr(getattr(batch_desc, "cg_mode", None), "name", str(getattr(batch_desc, "cg_mode", None)))
        if mode != "NONE":
            st.counters["skip_graph"] += 1
            return
        if getattr(runner, "pcp_manager", None) is not None or getattr(runner, "dcp_size", 1) > 1:
            st.counters["skip_cp"] += 1
            return
        T = int(input_batch.num_tokens)
        if T < st.min_t or int(input_batch.num_tokens_after_padding) != T:
            st.counters["skip_small"] += 1
            return
        n = int(input_batch.num_reqs)
        qsl = np.asarray(input_batch.query_start_loc_np[: n + 1], dtype=np.int64)
        if int(qsl[n]) != T:
            st.counters["skip_shape"] += 1
            return
        qlens = np.diff(qsl)
        ub = input_batch.seq_lens_cpu_upper_bound
        ub = ub.numpy() if hasattr(ub, "numpy") else np.asarray(ub)
        start_pos = ub[:n].astype(np.int64) - qlens                  # precise for prefill rows (the only rows cut inside)
        computed = np.asarray(input_batch.num_computed_tokens_np[:n], dtype=np.int64)
        prefilling = np.asarray(input_batch.is_prefilling_np[:n], dtype=bool) & (computed == start_pos)
        split, why = plan_split(qsl, start_pos, prefilling)
        if split is None:
            st.counters["skip_nosplit"] += 1
            return
        specs = half_specs(split, n, T)
        check_split_invariants(specs, qsl, T)
        ms = runner.model_state
        replay = getattr(ms, "_replay_start", None)
        replay_start = replay[:n] if replay is not None else None
        st.pending = Pending(runner, input_batch, block_tables, slot_mappings, replay_start, T, input_batch.positions, split,
                             specs, qsl)
    except Exception as e:  # noqa: BLE001 - an ineligible step must never break the forward
        st.pending = None
        st.counters["skip_error"] += 1
        logger.warning("%s runner hook: step not eligible (%s: %s)", TAG, type(e).__name__, e)


def _consume_pending(positions) -> Pending | None:
    st = _STATE
    pend, st.pending = st.pending, None
    if pend is None:
        return None
    try:
        same = int(positions.shape[0]) == pend.num_tokens and positions.data_ptr() == pend.positions.data_ptr()
    except Exception:  # noqa: BLE001
        same = False
    return pend if same else None


# --- region layers / prefixes ------------------------------------------------------------------------------------------
def region_prefixes(model) -> set[str]:
    """Names of the attention-metadata entries the region layers read (the groups holding them get builder copies)."""
    out: set[str] = set()
    for ly in sp.region_layers(model):
        a = ly.attn
        out.add(a.swa_cache_layer.prefix)
        if getattr(a, "compressed_cache_prefix", None) is not None:
            out.add(a.compressed_cache_prefix)
        ix = getattr(a, "indexer", None)
        if ix is not None:
            out.add(ix.k_cache.prefix)
        comp = getattr(a, "compressor", None)
        sc = getattr(comp, "state_cache", None) if comp is not None else None
        if sc is not None:
            out.add(sc.prefix)
    return out


def _topk_targets(layer) -> list[tuple[Any, str]]:
    a = layer.attn
    out = [(a, "topk_indices_buffer")]
    ix = getattr(a, "indexer", None)
    if ix is not None:
        out.append((ix, "topk_indices_buffer"))
        out.append((ix.indexer_op, "topk_indices_buffer"))
    return out


@contextlib.contextmanager
def topk_swap(layer, buf):
    """Point every user of the shared top-k buffer of this layer's attention at `buf` (restored afterwards)."""
    if buf is None:
        yield
        return
    saved = [(o, a, getattr(o, a)) for o, a in _topk_targets(layer)]
    for o, a, _ in saved:
        setattr(o, a, buf)
    try:
        yield
    finally:
        for o, a, v in saved:
            setattr(o, a, v)


def _mem_free() -> int:
    try:
        return int(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001
        return -1


def build_half_contexts(pend: Pending, model, st: _State) -> tuple[list[HalfContext], list]:
    """Per-half attention metadata + forward contexts (production). Mirrors UBatchRunner.prepare for two microbatches but goes
    through ModelState._build_attn_metadata (DefaultModelState.prepare_attn): the DSV4.1 prepare_attn would re-upload the shared
    replay-start buffer (the full batch's metadata views it) and re-prepare the replay batch. Each half owns copies of the
    attention groups (builders keep the metadata they built) and a query_start_loc / seq_lens buffer."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, get_forward_context
    from vllm.models.deepseek_v41.nvidia.model_state import ReplayAttnMetadata
    from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
    from vllm.v1.worker.gpu.ubatch_utils import _slice_input_batch
    from vllm.v1.worker.ubatch_utils import UBatchSlice

    runner, ib = pend.runner, pend.input_batch
    ms = runner.model_state
    sc = st.scratch
    if sc.groups is None:
        free0 = _mem_free()
        sc.groups = [ms._copy_attn_groups(runner.attn_groups, region_prefixes(model)) for _ in range(2)]
        dev = runner.device
        sc.qsl = [torch.zeros(runner.max_num_reqs + 1, dtype=torch.int32, device=dev) for _ in range(2)]
        sc.seq = [torch.zeros(runner.max_num_reqs, dtype=torch.int32, device=dev) for _ in range(2)]
        sc.topk2 = torch.empty_like(model.topk_indices_buffer)
        free1 = _mem_free()
        sc.bytes = (free0 - free1) if free0 >= 0 and free1 >= 0 else -1
        logger.info("%s created the per-half attention-metadata builders (2 sets of %d groups) and the second top-k buffer: "
                    "%.1f MiB of GPU memory", TAG, sum(len(g) for g in sc.groups[0]), max(sc.bytes, 0) / 2**20)
    base = get_forward_context()
    halves: list[HalfContext] = []
    for spec in pend.specs:
        sl = UBatchSlice(slice(spec.req_lo, spec.req_hi), slice(spec.tok_lo, spec.tok_hi))
        sub = _slice_input_batch(ib, sl, sc.qsl[spec.index], sc.seq[spec.index])
        bts = tuple(b[sl.request_slice] for b in pend.block_tables)
        sm = pend.slot_mappings[:, sl.token_slice]
        extra = ReplayAttnMetadata(pend.replay_start[sl.request_slice]) if pend.replay_start is not None else None
        # exclusive mode: the halves' indexer metadata is never sharded (a sharded half would exchange on the compute stream)
        with (no_row_shard() if st.idx_mode == "exclusive" else contextlib.nullcontext()):
            am = ms._build_attn_metadata(sub, CUDAGraphMode.NONE, bts, sm, sc.groups[spec.index], runner.kv_cache_config,
                                         for_capture=False, ubatch_idx=0, model_specific_attn_metadata=extra)
        ctx = dataclasses.replace(
            base, attn_metadata=am, slot_mapping=build_slot_mappings_by_layer(sm, runner.kv_cache_config),
            dp_metadata=None, cudagraph_runtime_mode=CUDAGraphMode.NONE, ubatch_slices=None, is_padding=sub.is_padding,
            batch_descriptor=BatchDescriptor(num_tokens=spec.rows), moe_layer_index=0)
        missing = sorted(region_prefixes(model) - set(am))
        if missing:
            raise RuntimeError(f"{TAG} the half-{spec.index + 1} attention metadata lacks the entries {missing[:4]} (attention "
                               "group copies do not cover the region layers)")
        halves.append(HalfContext(spec, ctx))
    return halves, [None, sc.topk2]


class ProdEnv:
    """HalfEnv for the engine: a compute block of half h runs under the half's forward context and top-k buffer."""

    def __init__(self, halves: list[HalfContext], topk_bufs: list) -> None:
        self.halves, self.topk_bufs = halves, topk_bufs

    @contextlib.contextmanager
    def compute_scope(self, h: int, layer):
        from vllm.forward_context import override_forward_context

        with override_forward_context(self.halves[h].forward_context), topk_swap(layer, self.topk_bufs[h]):
            yield


# ---------------------------------------------------------------------------------------------------------------------
# 4. The pipeline
# ---------------------------------------------------------------------------------------------------------------------
class _HalfState:
    def __init__(self, spec: HalfSpec, plan, handle) -> None:
        self.spec, self.plan, self.handle = spec, plan, handle
        self.x = None                  # comm side [S_h, H]: embedding rows first, then the MoE reduce-scatter output
        self.residual = self.post_mix = self.res_mix = None
        self.pre_mix = None            # FFN pre-mix carried into the next attention boundary
        self.attn_pre = None
        self.x_full = None             # [T_h, H] gathered input of the next compute block
        self.ev_ready = None           # comm event: x_full complete
        self.ev_part = None            # compute event: the block's partial output complete
        self.exit = None               # gathered carried states (comm) after the last layer
        self.aux: dict[int, Any] = {}  # layer index -> gathered previous_aux [T_h, H]
        self.keep: list = []           # compute-pool tensors the chain in flight still reads / writes on the comm stream (revision 4): dropped when
                                       # the compute stream has waited for the chain's ev_ready (or for the end of the region)


class Pipeline:
    """Runs layers [start, replay_start) of one forward over two halves. See the module docstring for the schedule."""

    def __init__(self, rt, world, model, layers: list, layer_ids: list[int], env, halves: list[HalfContext], plans: list,
                 ops, hidden_states, positions, input_ids, engram_hashes, engram_mask, aux_layers, tracer: "Tracer | None" = None) -> None:
        self.rt, self.model, self.layers, self.layer_ids = rt, model, layers, layer_ids
        self.tracer = tracer                      # self-test only: per-layer divergence trace (no collective)
        self.env, self.halves, self.ops = env, halves, ops
        self.world = _LoggedWorld(world, rt)
        self.rank = world.rank
        self.hidden_states, self.positions, self.input_ids = hidden_states, positions, input_ids
        self.engram_hashes, self.engram_mask = engram_hashes, engram_mask
        self.aux_layers = set(aux_layers)
        self.T = int(positions.shape[0])
        self.hs = [_HalfState(h.spec, plans[i], sp._Handle(plans[i], self.world, self.rank)) for i, h in enumerate(halves)]
        self._rows: dict[int, Any] = {}

    # -- small helpers ----------------------------------------------------------------------------------------------
    def _slice(self, t, spec: HalfSpec):
        return None if t is None else t[spec.tok_lo:spec.tok_hi]

    def _common(self, layer):
        return (layer.rms_norm_eps, layer.hc_eps, layer.hc_eps, layer.hc_post_alpha, layer.hc_sinkhorn_iters)

    # -- comm blocks --------------------------------------------------------------------------------------------------
    def _k(self, name: str, st: _HalfState, fn, reads: tuple = ()):
        """A block of half `st`'s chain on the communication stream. The compute-pool tensors it takes (redirected collective outputs, the padded
        scatter input) are parked in st.keep, which the compute stream's next block of this half drops after waiting for the chain."""
        rt = self.rt
        with rt.keep_into(st.keep):
            return rt.run(name, rt.comm, fn, reads=reads)

    def _reduce_scatter(self, partial, st: _HalfState):
        """sp.reduce_scatter_rows with the zero-padded input taken from the compute stream's pool (revision 4): the collective receives the same
        bytes as from 022's zero-padded copy (padding rows are zero), but no [padded, width] block is allocated on the communication stream."""
        plan = st.plan
        if partial.shape[0] != plan.T:
            raise RuntimeError(f"{TAG} reduce-scatter got {partial.shape[0]} rows, expected T={plan.T}")
        if partial.shape[0] == plan.padded:
            src = partial.contiguous()
        else:
            src = self.rt.alloc_compute((plan.padded, *partial.shape[1:]), partial.dtype, partial.device)
            self.rt.hold(src, st.keep, self.rt.comm)
            src[:plan.T].copy_(partial)
            src[plan.T:].zero_()
        out = self.world.reduce_scatter_rows(src)
        if out.shape[0] != plan.S:
            raise RuntimeError(f"{TAG} reduce-scatter returned {out.shape[0]} rows, expected S={plan.S}")
        return out

    def _exit_states(self, st: _HalfState):
        """sp.exit_region (same checks, same gathers in the same order) without its three small full-T copies on the communication stream: the
        fp32 mix states come back as views of the one packed gather, the compute stream's torch.cat makes the contiguous copies."""
        plan = st.plan
        for nm, t in (("hidden_states", st.x), ("residual", st.residual), ("post_mix", st.post_mix), ("res_mix", st.res_mix),
                      ("pre_mix", st.pre_mix)):
            if not isinstance(t, torch.Tensor) or t.shape[0] != plan.S:
                raise RuntimeError(f"{TAG} exit_region: {nm} {getattr(t, 'shape', None)} does not have S={plan.S} rows")
        S, T = plan.S, plan.T
        x = sp.gather_rows(st.x, self.world, plan)
        res = sp.gather_rows(st.residual, self.world, plan)
        packed = torch.cat([st.post_mix.reshape(S, -1), st.res_mix.reshape(S, -1), st.pre_mix.reshape(S, -1)], dim=1).contiguous()
        g = sp.gather_rows(packed, self.world, plan)
        n_post, n_res = st.post_mix[0].numel(), st.res_mix[0].numel()
        assert st.residual.shape[1] == st.post_mix.shape[1], (st.residual.shape, st.post_mix.shape)
        return (x, res, g[:, :n_post].reshape(T, *st.post_mix.shape[1:]), g[:, n_post:n_post + n_res].reshape(T, *st.res_mix.shape[1:]),
                g[:, n_post + n_res:].reshape(T, *st.pre_mix.shape[1:]))

    def _quarantine(self) -> None:
        """An error inside the region: GPU work queued on the other stream may still touch what the keep lists hold (and what the states
        reference). Park all of it for the life of the process instead of letting the allocator recycle it."""
        objs = []
        for st in self.hs:
            objs.extend([st.keep, st.x_full, st.x, st.residual, st.post_mix, st.res_mix, st.pre_mix, st.attn_pre, st.aux, st.exit, st.ev_ready,
                         st.ev_part])
        objs.append(self._rows)
        self.rt.quarantine(objs)

    def _embed_shard(self, st: _HalfState):
        return sp.shard_rows(self._slice(self.hidden_states, st.spec), st.plan, self.rank)

    def _engram_rows(self, eng):
        r = self._rows.get(id(eng))
        if r is None:
            r = eng._ready_rows(self.T)                       # ONCE per forward and layer (NVMe fetch is once per publish)
            self._rows[id(eng)] = r
        return r

    def _boundary_attn_math(self, L: int, st: _HalfState):
        """The attention-side mHC boundary of layer L on this half's rows (comm stream), mirroring sp.sp_layer_forward."""
        layer, ops, plan = self.layers[L], self.ops, st.plan
        idx = self.layer_ids[L]
        common = self._common(layer)
        capture = idx in self.aux_layers
        previous_aux = None
        x, residual, post_mix, res_mix, pre_mix = st.x, st.residual, st.post_mix, st.res_mix, st.pre_mix
        if residual is None:
            assert x.dim() == 2 and layer.hc_attn_fn_broadcast is not None, "SP expects the embedding rows at the first layer"
            residual = x.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
            post_mix, res_mix, x, attn_pre = ops.pre_delayed(
                residual, layer.hc_attn_fn_broadcast, layer.hc_attn_scale, layer.hc_attn_base, *common,
                x=x, norm_weight=layer.attn_norm.weight, norm_eps=layer.attn_norm.variance_epsilon)
        elif layer.engram is not None and self.engram_hashes is not None:
            previous_post = ops.post(x, residual, post_mix, res_mix)
            if capture:
                previous_aux = previous_post.mean(dim=1)
            eng = layer.engram
            rows_h = self._engram_rows(eng)[st.spec.tok_lo:st.spec.tok_hi]
            with _ready_rows_view(eng, rows_h):
                residual = sp._engram_sp(eng, previous_post, self.engram_hashes[st.spec.tok_lo:st.spec.tok_hi,
                                                                               eng.layer_hash_index],
                                         self._slice(self.engram_mask, st.spec), st.handle)
            post_mix, res_mix, x, attn_pre = ops.pre_delayed(
                residual, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, *common, pre_mix=pre_mix,
                norm_weight=layer.attn_norm.weight, norm_eps=layer.attn_norm.variance_epsilon)
        else:
            residual, post_mix, res_mix, x, attn_pre, aux = ops.shifted_post_pre(
                x, residual, post_mix, res_mix, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, *common,
                pre_mix=pre_mix, norm_weight=layer.attn_norm.weight, norm_eps=layer.attn_norm.variance_epsilon,
                capture_aux=capture, stream=None, reduce_results=False)
            if capture:
                previous_aux = aux
        if previous_aux is not None:
            st.aux[idx] = sp.gather_rows(previous_aux, self.world, plan)       # the drafter consumes full [T_h, H] rows
            self.rt.produced(st.aux[idx], self.rt.comm)
        x_full = sp.gather_rows(x, self.world, plan)
        return residual, post_mix, res_mix, attn_pre, x_full

    def _comm_attn_boundary(self, L: int, st: _HalfState) -> None:
        rt = self.rt
        reads = (st.x, st.residual, st.post_mix, st.res_mix, st.pre_mix)
        residual, post_mix, res_mix, attn_pre, x_full = self._k(
            f"BA[{st.spec.index},{L}]", st, lambda: self._boundary_attn_math(L, st), reads=reads)
        st.residual, st.post_mix, st.res_mix, st.attn_pre, st.x_full = residual, post_mix, res_mix, attn_pre, x_full
        st.ev_ready = rt.record(rt.comm)

    def _comm_after_attn(self, L: int, st: _HalfState, partial) -> None:
        rt, layer, ops = self.rt, self.layers[L], self.ops
        rt.wait(rt.comm, st.ev_part)
        rt.hold(partial, st.keep, rt.comm)           # the scatter reads it on the comm stream: kept (not record_stream'ed) until the compute stream has waited for this chain

        def body():
            x = self._reduce_scatter(partial, st)
            residual, post_mix, res_mix, x, ffn_pre, _ = ops.shifted_post_pre(
                x, st.residual, st.post_mix, st.res_mix, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                *self._common(layer), pre_mix=st.attn_pre, norm_weight=layer.ffn_norm.weight,
                norm_eps=layer.ffn_norm.variance_epsilon, stream=None, reduce_results=False)
            x_full = sp.gather_rows(x, self.world, st.plan)
            return residual, post_mix, res_mix, ffn_pre, x_full

        residual, post_mix, res_mix, ffn_pre, x_full = self._k(
            f"BF[{st.spec.index},{L}]", st, body, reads=(partial, st.residual, st.post_mix, st.res_mix, st.attn_pre))
        st.residual, st.post_mix, st.res_mix, st.pre_mix, st.x_full = residual, post_mix, res_mix, ffn_pre, x_full
        st.ev_ready = rt.record(rt.comm)

    def _comm_after_moe(self, L: int, st: _HalfState, partial) -> None:
        rt = self.rt
        rt.wait(rt.comm, st.ev_part)
        rt.hold(partial, st.keep, rt.comm)
        last = L + 1 == len(self.layers)

        def body():
            return self._reduce_scatter(partial, st)

        st.x = self._k(f"RSM[{st.spec.index},{L}]", st, body, reads=(partial,))
        if self.tracer is not None:
            rt.run(f"TR[{st.spec.index},{L}]", rt.comm, lambda: self.tracer.capture(L, st.spec.index, st.x, st.residual),
                   reads=(st.x, st.residual))
        if not last:
            self._comm_attn_boundary(L + 1, st)
            return

        st.exit = self._k(f"EXIT[{st.spec.index}]", st, lambda: self._exit_states(st),
                          reads=(st.x, st.residual, st.post_mix, st.res_mix, st.pre_mix))
        st.ev_ready = rt.record(rt.comm)

    # -- compute blocks -----------------------------------------------------------------------------------------------
    def _compute_block(self, L: int, st: _HalfState, kind: str):
        rt, layer = self.rt, self.layers[L]
        h = st.spec.index
        rt.wait(rt.compute, st.ev_ready)
        st.keep.clear()          # the compute stream is now ordered behind the chain that produced x_full: what it read / wrote can be recycled
        rt.handover(st.x_full, rt.compute)
        pos_h, ids_h = self._slice(self.positions, st.spec), self._slice(self.input_ids, st.spec)

        def body():
            with self.env.compute_scope(h, layer):
                if kind == "A":
                    with sp.attn_reduce_handle(layer):
                        return layer.attn(pos_h, st.x_full, None)
                with sp.ffn_reduce_handle(layer):
                    return layer.ffn(st.x_full, ids_h, None)

        partial = rt.run(f"{kind}[{h},{L}]", rt.compute, body, reads=(st.x_full,))
        st.ev_part = rt.record(rt.compute)
        return partial

    # -- driver -------------------------------------------------------------------------------------------------------
    def run(self):
        try:
            return self._run()
        except BaseException:
            self._quarantine()
            raise

    def _run(self):
        rt = self.rt
        rt.produced(self.hidden_states, rt.compute)        # written by the embedding / hashing before the region (compute stream)
        if self.engram_mask is not None:
            rt.produced(self.engram_mask, rt.compute)
        rt.begin()                                           # the comm stream waits for everything queued so far
        rt.handover(self.hidden_states, rt.comm)
        if self.engram_mask is not None:
            rt.handover(self.engram_mask, rt.comm)
        n = len(self.layers)
        for st in self.hs:
            st.x = rt.run(f"EMB[{st.spec.index}]", rt.comm, lambda st=st: self._embed_shard(st), reads=(self.hidden_states,))
        for st in self.hs:
            self._comm_attn_boundary(0, st)
        for L in range(n):
            for st in self.hs:
                self._comm_after_attn(L, st, self._compute_block(L, st, "A"))
            for st in self.hs:
                self._comm_after_moe(L, st, self._compute_block(L, st, "M"))
        return self._finish()

    def _finish(self):
        rt = self.rt
        rt.finish()
        for st in self.hs:
            st.keep.clear()                                 # the compute stream has waited for the whole communication stream
        parts = [st.exit for st in self.hs]
        for p in parts:
            for t in p:
                rt.handover(t, rt.compute)
        for st in self.hs:
            for t in st.aux.values():
                rt.handover(t, rt.compute)

        def cat():
            return tuple(torch.cat([p[i] for p in parts], dim=0) for i in range(5))

        out = rt.run("CAT", rt.compute, cat, reads=tuple(t for p in parts for t in p))
        aux = {}
        for idx in sorted({i for st in self.hs for i in st.aux}):
            aux[idx] = rt.run(f"CATAUX[{idx}]", rt.compute, lambda idx=idx: torch.cat([st.aux[idx] for st in self.hs], dim=0),
                              reads=tuple(st.aux[idx] for st in self.hs))
        self.aux_out = aux
        return out


@contextlib.contextmanager
def _ready_rows_view(engram, rows):
    """Within the block engram._ready_rows(n) returns `rows` (this half's staged rows; n must equal their count)."""
    prev = engram.__dict__.get("_ready_rows", None)

    def view(num_tokens: int):
        if num_tokens != rows.shape[0]:
            raise RuntimeError(f"{TAG} Engram rows requested for {num_tokens} tokens inside a half of {rows.shape[0]}")
        return rows

    engram._ready_rows = view
    try:
        yield
    finally:
        if prev is None:
            engram.__dict__.pop("_ready_rows", None)
        else:
            engram._ready_rows = prev


@contextlib.contextmanager
def _engram_rows_once(model):
    """Self-test only: the plain run and the overlap run of the SAME forward share one fetch of the staged rows per Engram layer
    (the NVMe channel poisons the engine when a publish is fetched twice)."""
    cache: dict[int, Any] = {}
    patched = []
    for ly in sp.region_layers(model):
        eng = getattr(ly, "engram", None)
        if eng is None:
            continue
        orig = eng._ready_rows

        def once(num_tokens: int, eng=eng, orig=orig):
            r = cache.get(id(eng))
            if r is None:
                r = cache[id(eng)] = orig(num_tokens)
            return r[:num_tokens]

        had = "_ready_rows" in eng.__dict__
        prev = eng.__dict__.get("_ready_rows")
        eng._ready_rows = once
        patched.append((eng, had, prev))
    try:
        yield
    finally:
        for eng, had, prev in patched:
            if had:
                eng._ready_rows = prev
            else:
                eng.__dict__.pop("_ready_rows", None)


# ---------------------------------------------------------------------------------------------------------------------
# 5. Rank-synchronised control, envelope, snapshots
# ---------------------------------------------------------------------------------------------------------------------
def parse_ctl(text: str | None) -> tuple[bool | None, dict]:
    """"on|off|1|0" plus an optional "log=<n>" (whitespace, ',' or ';' separated). Unknown tokens are ignored by the caller with a
    warning; the same text yields the same result on every rank. (MIN_T is fixed at boot: the control file is only read at forwards
    that already have a cut, so a raised threshold could never be lowered again.)"""
    on: bool | None = None
    kv: dict = {}
    bad: list[str] = []
    for tok in (text or "").replace(",", " ").replace(";", " ").split():
        t = tok.strip().lower()
        if t in ("on", "1", "true", "yes", "overlap"):
            on = True
        elif t in ("off", "0", "false", "no", "plain"):
            on = False
        elif t.startswith("log=") and t[4:].isdigit():
            kv["log"] = int(t[4:])
        else:
            bad.append(tok)
    if bad:
        kv["_bad"] = bad
    return on, kv


def _sync_ctl(world) -> tuple[bool, str | None]:
    """Rank 0 reads the control file, everyone adopts rank 0's text (a bounded gloo exchange). Returns (overlap on, text)."""
    path = os.environ.get(ENV_CTL)
    st = _STATE
    text = None
    if world.rank == 0:
        try:
            with open(path) as f:
                text = f.read().strip() or None
        except OSError:
            text = None
    try:
        world.barrier_bounded(60.0)
        text = world.all_gather_object(text)[0]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{TAG} control-file exchange failed on this rank ({type(e).__name__}: {str(e)[:200]}); a TP rank "
                           "is missing or stuck") from e
    if text != st.ctl_text:
        st.ctl_text = text
        on, kv = parse_ctl(text)
        if kv.get("_bad"):
            logger.warning("%s control text %r: ignored tokens %s", TAG, text, kv["_bad"])
        if on is not None:
            st.ctl_on = on
        if "log" in kv:
            st.ctl_log = kv["log"]
        logger.info("%s control %r -> overlap %s, log every %s", TAG, text, "ON" if st.ctl_on else "OFF",
                    st.ctl_log if st.ctl_log is not None else parse_log_every(os.environ.get(ENV_LOG)))
    return st.ctl_on, text


def overlap_envelope_errors(model, world) -> list[str]:
    """What this patch needs on top of 022's envelope (structure only; the runner-side needs are checked at the first
    overlapped forward)."""
    errs: list[str] = []
    vc = getattr(model, "vllm_config", None)
    ac = getattr(vc, "aux_output_config", None)
    if ac is not None and getattr(ac, "enable_return_routed_experts", False):
        errs.append("enable_return_routed_experts is on (its capture buffer is indexed by the full batch rows)")
    sched = getattr(vc, "scheduler_config", None)
    max_tokens = int(getattr(sched, "max_num_batched_tokens", 0) or 0)
    attn_cfg = getattr(vc, "attention_config", None)
    shard_min = getattr(attn_cfg, "indexer_tp_shard_min_rows", None)
    mode, _ = indexer_settings()
    sharding = shard_min is not None and bool(max_tokens) and int(shard_min) <= max_tokens
    errs += indexer_envelope_errors(model, world, mode, sharding, shard_min, max_tokens)
    topk = getattr(model, "topk_indices_buffer", None)
    if topk is None:
        errs.append("the model has no topk_indices_buffer")
    for ly in sp.region_layers(model):
        a = ly.attn
        i = getattr(a, "layer_id", "?")
        for obj, name in _topk_targets(ly):
            try:
                if getattr(obj, name) is not topk:
                    errs.append(f"layer {i}: {type(obj).__name__}.{name} is not the model's shared top-k buffer")
            except AttributeError:
                errs.append(f"layer {i}: {type(obj).__name__} has no {name}")
        ix = getattr(a, "indexer", None)
        if ix is not None and getattr(ix.indexer_op, "candidate_blocks", None) is not None:
            errs.append(f"layer {i}: the indexer uses the candidate-block buffer (candidate source before the replay start?)")
        if not hasattr(a, "swa_cache_layer"):
            errs.append(f"layer {i}: attention has no swa_cache_layer")
    try:
        from vllm.model_executor.layers.fused_moe.runner import moe_runner

        if not getattr(moe_runner, "_USE_LAYERNAME", True):
            errs.append("torch without opaque LayerName: the MoE layer would be looked up through the shared forward-context counter")
    except Exception:  # noqa: BLE001
        pass
    return errs


def _ring_blocks(model, pend: Pending) -> list[tuple[Any, Any]]:
    """[(state cache tensor, ring block ids of this batch's requests)] of the region's compressors (ratio > 1)."""
    out = []
    groups = pend.runner.kv_cache_config.kv_cache_groups
    n = int(pend.input_batch.num_reqs)
    for ly in sp.region_layers(model):
        comp = getattr(ly.attn, "compressor", None)
        sc = getattr(comp, "state_cache", None) if comp is not None else None
        if sc is None:
            continue
        gi = next((i for i, g in enumerate(groups) if sc.prefix in g.layer_names), None)
        if gi is None:
            raise RuntimeError(f"{TAG} compressor state cache {sc.prefix} is in no KV-cache group")
        ids = pend.block_tables[gi][:n, 0].to(torch.int64)
        out.append((sc.kv_cache, ids))
    return out


class _RingSnapshot:
    """The compressor rings are the only state a region reads that it also writes before the next forward (the open group of a
    chunk that starts mid-group): the self-test runs the region twice, so it restores them between the runs."""

    def __init__(self, items: list[tuple[Any, Any]]) -> None:
        self.items = []
        for cache, ids in items:
            keep = ids[ids != 0]                                   # block 0 is the null block
            self.items.append((cache, keep, cache[keep].clone()))

    def restore(self) -> None:
        for cache, keep, saved in self.items:
            cache[keep] = saved


# ---------------------------------------------------------------------------------------------------------------------
# 6. Self-test comparison (pure function of the exchanged per-rank metrics, identical on every rank)
# ---------------------------------------------------------------------------------------------------------------------
def state_metrics(a: torch.Tensor, b: torch.Tensor, chunk_rows: int | None = None, cut: int | None = None, top: int = REPORT_TOP_ROWS,
                  budget: int | None = None) -> dict:
    """a (the path under test) against the reference b: ||a-b||/||b|| over all rows ("rms"), amax-relative, the fraction of rows whose
    own relative error exceeds ROW_TOL ("bad_rows") and the shape of the per-row error distribution (fraction above 10 %, p50 / p90 / p99 /
    max, how much of the squared error the worst row and the worst ten rows carry, the largest-to-median reference row norm, the worst
    `top` rows as [row, relative error, reference norm]); with `cut` also the same for rows [0,cut) and [cut,n) ("h1", "h2"). Row-chunked: the
    chunk is as many rows as fit `budget` bytes of temporaries (default SELFTEST_TEMP_BYTES; ~30 bytes per element), unless `chunk_rows` is given
    (the residual stream is 335 MB per tensor; revision 4's fixed 512-row chunks held ~300 MB of float32 / float64 temporaries). Every per-row sum is
    the same float64 sum whatever the chunking. Pure function of the two tensors."""
    n = int(a.shape[0])
    dev = a.device
    if n == 0:
        return {"rms": 0.0, "amax": 0.0, "bad_rows": 0.0, "finite": True}
    if chunk_rows is None:
        chunk_rows = max(1, min(n, (SELFTEST_TEMP_BYTES if budget is None else int(budget)) // (max(1, int(a[0].numel())) * STATE_METRIC_BYTES_PER_ELEM)))
    row_d2 = torch.zeros(n, dtype=torch.float64, device=dev)
    row_b2 = torch.zeros(n, dtype=torch.float64, device=dev)
    mx_d = torch.zeros((), dtype=torch.float32, device=dev)
    mx_b = torch.zeros((), dtype=torch.float32, device=dev)
    fin = torch.ones((), dtype=torch.float32, device=dev)
    for s0 in range(0, n, chunk_rows):
        e = min(s0 + chunk_rows, n)
        a32 = a[s0:e].to(torch.float32)
        b32 = b[s0:e].to(torch.float32)
        d = a32 - b32
        row_d2[s0:e] = (d.double() ** 2).reshape(e - s0, -1).sum(dim=1)
        row_b2[s0:e] = (b32.double() ** 2).reshape(e - s0, -1).sum(dim=1)
        mx_d = torch.maximum(mx_d, d.abs().amax())
        mx_b = torch.maximum(mx_b, b32.abs().amax())
        fin = fin * torch.isfinite(a32).all().float() * torch.isfinite(b32).all().float()
    rel = (row_d2 / row_b2.clamp_min(1e-60)).sqrt()
    refn = row_b2.sqrt()
    tot_d2, tot_b2 = row_d2.sum(), row_b2.sum()
    srt = torch.sort(rel).values

    def q(p: float) -> torch.Tensor:
        return srt[min(n - 1, int(p * (n - 1) + 0.5))]

    kk = min(max(top, 10), n)
    tv, ti = torch.topk(row_d2, kk)
    scal = [tot_d2.sqrt() / tot_b2.sqrt().clamp_min(1e-30), (mx_d / mx_b.clamp_min(1e-30)).double(), (rel > ROW_TOL).double().mean(),
            fin.double(), (rel > 0.1).double().mean(), q(0.5), q(0.9), q(0.99), srt[-1],
            tv[0] / tot_d2.clamp_min(1e-300), tv.sum() / tot_d2.clamp_min(1e-300), refn.max() / refn.median().clamp_min(1e-30)]
    if cut is not None and 0 < cut < n:
        for lo, hi in ((0, cut), (cut, n)):
            scal += [row_d2[lo:hi].sum().sqrt() / row_b2[lo:hi].sum().sqrt().clamp_min(1e-30), (rel[lo:hi] > ROW_TOL).double().mean()]
    sel = ti[:top]
    v = torch.stack([x.double().reshape(()) for x in scal]).cpu().tolist()
    rows = torch.stack([sel.double(), rel[sel], refn[sel]], dim=1).cpu().tolist()
    out = {"rms": v[0], "amax": v[1], "bad_rows": v[2], "finite": bool(v[3] > 0.5), "frac10": v[4], "p50": v[5], "p90": v[6], "p99": v[7],
           "max_rel": v[8], "top1_share": v[9], "top10_share": v[10], "norm_ratio": v[11],
           "top": [[int(r), float(e), float(nr)] for r, e, nr in rows]}
    if len(v) > 12:
        out["h1"] = {"rms": v[12], "bad_rows": v[13]}
        out["h2"] = {"rms": v[14], "bad_rows": v[15]}
    return out


def checksum(t: torch.Tensor) -> int:
    """A cheap deterministic checksum of the bits of t (position weighted; int64 wrap-around is deterministic), equal on two
    ranks iff the data is (w.h.p.). One host sync."""
    v = t.contiguous().reshape(-1)
    if v.element_size() == 2:
        v = v.view(torch.int16)
    elif v.element_size() == 4:
        v = v.view(torch.int32)
    else:
        v = v.view(torch.uint8)
    acc = torch.zeros((), dtype=torch.int64, device=v.device)
    step = 1 << 18                                   # 2 MiB of int64 per temporary (the value does not depend on the chunking)
    for s0 in range(0, v.numel(), step):
        c = v[s0:s0 + step].to(torch.int64)
        w = (torch.arange(s0, s0 + c.numel(), device=c.device, dtype=torch.int64) % 65521) + 1
        acc += (c * w).sum()
    return int(acc.item()) % ((1 << 61) - 1)


# --- per-layer divergence trace (self-test diagnostics) -----------------------------------------------------------------
class Tracer:
    """Per-layer divergence trace of the self-test runs. After every region layer the rank-local output rows -- x (the reduce-scattered MoE
    output) and the residual stream after the layer's attention-side update -- are projected onto TRACE_K fixed random directions (the
    same on every rank; float64 products, so no TF32). The shards of all runs are exchanged over gloo ONCE, only when the self-test fails,
    reassembled into full rows and compared layer by layer and half by half (assemble_trace / compare_traces). No collective, no host
    synchronisation while a run is in flight, a few tens of KB per layer on the device."""

    CHUNK = 256

    def __init__(self, device, hidden: int, hc: int, k: int = TRACE_K, seed: int = 28, layers: list | None = None) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.k = k
        self.rx = torch.randn(hidden, k, generator=gen, dtype=torch.float64).to(device)
        self.rr = torch.randn(hc * hidden, k, generator=gen, dtype=torch.float64).to(device)
        self.cur: dict | None = None
        self.runs: dict[str, dict] = {}
        self.errors: list[str] = []
        self.probe = RoutingProbe(layers) if layers else None

    def begin(self, name: str, parts: list) -> None:
        """parts: [(first row, rows, rows per rank S)] of the run (plain: one part; overlap: one per half)."""
        self.cur = {"name": name, "parts": [tuple(int(v) for v in p) for p in parts], "layers": {}}
        if self.probe is not None:
            self.probe.begin(name)

    def _project(self, t: torch.Tensor, r64: torch.Tensor) -> torch.Tensor:
        t2 = t.reshape(t.shape[0], -1)
        step = max(8, min(self.CHUNK, SELFTEST_TEMP_BYTES // max(1, int(t2.shape[1]) * 16)))       # a float64 copy of the chunk (+ the product): bounded bytes
        outs = [(t2[s0:s0 + step].to(torch.float64) @ r64).to(torch.float32) for s0 in range(0, max(t2.shape[0], 1), step)]
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

    def capture(self, layer: int, part: int, x: torch.Tensor, residual: torch.Tensor) -> None:
        if self.cur is None:
            return
        try:
            self.cur["layers"].setdefault(int(layer), {})[int(part)] = (self._project(x, self.rx), self._project(residual, self.rr))
        except Exception as e:  # noqa: BLE001 - diagnostics never break a run
            if len(self.errors) < 8:
                self.errors.append(f"capture layer {layer} part {part}: {type(e).__name__}: {str(e)[:120]}")

    def end(self) -> None:
        """After the run's GPU work has been awaited: the projections move to the host."""
        cur, self.cur = self.cur, None
        if self.probe is not None:
            self.probe.end()
        if cur is None:
            return
        try:
            layers = {L: {p: (px.cpu(), pr.cpu()) for p, (px, pr) in d.items()} for L, d in cur["layers"].items()}
            self.runs[cur["name"]] = {"parts": cur["parts"], "layers": layers}
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"end {cur['name']}: {type(e).__name__}: {str(e)[:120]}")

    def abort(self) -> None:
        self.cur = None
        if self.probe is not None:
            self.probe.abort()

    def close(self) -> None:
        if self.probe is not None:
            self.probe.close()

    def export(self, cut: int = 0) -> dict:
        routing = None
        if self.probe is not None:
            try:
                routing = self.probe.summary(cut)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"routing summary: {type(e).__name__}: {str(e)[:120]}")
        return {"k": self.k, "runs": self.runs, "errors": list(self.errors), "routing": routing}


class RoutingProbe:
    """Self-test only: a forward hook on the gate of every region MoE layer records, for every call, the experts the DSV4 router picks
    from the fp32 router logits -- the router's own rule, sqrt(softplus(logit)) + correction bias, top-k -- and the score gap between the
    last selected and the first rejected expert. Compared between runs this gives the share of tokens whose expert set changed (the
    routing-flip rate, per layer) and how many tokens sit within a given margin of the decision boundary. The probe changes nothing
    (the hooks return None) and fails soft: no hook fired = no routing lines in the report."""

    def __init__(self, layers: list) -> None:
        self.handles: list = []
        self.cur: dict | None = None
        self.name = ""
        self.runs: dict[str, dict] = {}
        self.errors: list[str] = []
        self.topk = 6
        for i, ly in enumerate(layers):
            try:
                ffn = getattr(ly, "ffn", None)
                gate = getattr(ffn, "gate", None)
                bias = getattr(gate, "e_score_correction_bias", None) if gate is not None else None
                if gate is None or bias is None or getattr(gate, "tid2eid", None) is not None or not hasattr(gate, "register_forward_hook"):
                    continue
                self.topk = int(getattr(ffn, "n_activated_experts", self.topk) or self.topk)
                self.handles.append(gate.register_forward_hook(self._hook(i, gate)))
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"hook layer {i}: {type(e).__name__}: {str(e)[:100]}")
        self.n_hooked = len(self.handles)

    def _hook(self, layer: int, gate):
        def hook(module, inputs, output):
            if self.cur is None:
                return None
            try:
                logits = output[0] if isinstance(output, (tuple, list)) else output
                bias = gate.e_score_correction_bias
                bias = getattr(bias, "data", bias)
                if not isinstance(logits, torch.Tensor) or logits.dim() != 2 or logits.shape[1] != bias.shape[0]:
                    return None
                rows, E = int(logits.shape[0]), int(logits.shape[1])
                if rows == 0:
                    return None
                step = max(1, min(rows, SELFTEST_TEMP_BYTES // (E * 16)))       # rows per chunk: ~4 float32 temporaries of E experts per row
                ids_p, gap_p = [], []
                for s0 in range(0, rows, step):
                    score = torch.sqrt(torch.nn.functional.softplus(logits[s0:s0 + step].float())) + bias.float()
                    v, ids = torch.topk(score, self.topk + 1, dim=1)
                    ids_p.append(ids[:, :self.topk].sort(dim=1).values.to(torch.int16))
                    gap_p.append(v[:, self.topk - 1] - v[:, self.topk])
                self.cur.setdefault(layer, []).append((ids_p[0] if len(ids_p) == 1 else torch.cat(ids_p),
                                                       gap_p[0] if len(gap_p) == 1 else torch.cat(gap_p)))
            except Exception as e:  # noqa: BLE001
                if len(self.errors) < 8:
                    self.errors.append(f"hook call layer {layer}: {type(e).__name__}: {str(e)[:100]}")
            return None
        return hook

    def begin(self, name: str) -> None:
        self.cur = {}
        self.name = name

    def end(self) -> None:
        cur, self.cur = self.cur, None
        if cur is None:
            return
        try:
            self.runs[self.name] = {L: (torch.cat([a for a, _ in lst]).cpu(), torch.cat([g for _, g in lst]).float().cpu()) for L, lst in cur.items()}
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"end {self.name}: {type(e).__name__}: {str(e)[:100]}")

    def abort(self) -> None:
        self.cur = None

    def close(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self.handles = []

    def summary(self, cut: int) -> dict | None:
        """{"hooked": n layers, "pairs": {name: {layer: (flip share all, H1, H2)}}, "margin": {layer: (share of tokens whose gap < 1e-2, < 1e-3)}}
        of the recorded runs (the reference run for the margins is 'plain'), or None when nothing was recorded."""
        runs = self.runs
        if "plain" not in runs or not runs["plain"]:
            return None

        masks: dict = {}

        def flips(name: str, a: str, b: str) -> dict:
            out = {}
            masks[name] = {}
            for L in sorted(set(runs.get(a, {})) & set(runs.get(b, {}))):
                (ids_a, _), (ids_b, _) = runs[a][L], runs[b][L]
                if ids_a.shape != ids_b.shape:
                    continue
                flipped = (ids_a != ids_b).any(dim=1)
                d = flipped.double()
                masks[name][L] = flipped.to(torch.uint8)
                out[L] = (float(d.mean()), float(d[:cut].mean()) if cut > 0 else float("nan"), float(d[cut:].mean()) if cut < len(d) else float("nan"))
            return out

        pairs = {"overlap-v-plain": flips("overlap-v-plain", "overlap", "plain"), "plain2-v-plain": flips("plain2-v-plain", "plain2", "plain"),
                 "overlap2-v-overlap": flips("overlap2-v-overlap", "overlap2", "overlap"), "serial-v-plain": flips("serial-v-plain", "serial", "plain"),
                 "overlap-v-serial": flips("overlap-v-serial", "overlap", "serial")}
        margin = {L: (float((g < 1e-2).double().mean()), float((g < 1e-3).double().mean())) for L, (_, g) in runs["plain"].items()}
        return {"hooked": len(runs["plain"]), "topk": self.topk, "pairs": pairs, "margin": margin, "masks": masks, "errors": list(self.errors)}


@contextlib.contextmanager
def _plain_layer_trace(model, tracer: Tracer | None):
    """While active, every 022 layer forward of the region reports its (x, residual) output rows to the tracer (plain runs only; the
    decoder layer looks sp_layer_forward up on the module at call time)."""
    if tracer is None:
        yield
        return
    index = {id(ly): i for i, ly in enumerate(sp.region_layers(model))}
    orig = sp.sp_layer_forward

    def traced(layer, *a, **k):
        out = orig(layer, *a, **k)
        i = index.get(id(layer))
        if i is not None:
            tracer.capture(i, 0, out[0], out[1])
        return out

    sp.sp_layer_forward = traced
    try:
        yield
    finally:
        sp.sp_layer_forward = orig


def assemble_trace(per_rank: list, run: str):
    """(T, {layer: (x [T,k], residual [T,k])}) of one run from every rank's exported shards, or None when something is missing."""
    runs = [(pr or {}).get("runs", {}).get(run) for pr in per_rank]
    if not runs or any(r is None for r in runs):
        return None
    parts = runs[0]["parts"]
    if any(r["parts"] != parts for r in runs):
        return None
    T = max(lo + rows for lo, rows, _ in parts)
    out: dict = {}
    for L in sorted(runs[0]["layers"]):
        shard0 = next(iter(runs[0]["layers"][L].values()))
        k = shard0[0].shape[1]
        X = torch.zeros(T, k)
        R = torch.zeros(T, k)
        for p, (lo, rows, S) in enumerate(parts):
            for r, run_r in enumerate(runs):
                a = r * S
                real = max(0, min(S, rows - a))
                if real == 0:
                    continue
                shard = run_r["layers"].get(L, {}).get(p)
                if shard is None:
                    return None
                X[lo + a:lo + a + real] = shard[0][:real]
                R[lo + a:lo + a + real] = shard[1][:real]
        out[L] = (X, R)
    return T, out


def _rms_rel(pa: torch.Tensor, pb: torch.Tensor, lo: int, hi: int) -> float:
    if hi <= lo:
        return float("nan")
    num = (pa[lo:hi] - pb[lo:hi]).double().pow(2).sum()
    den = pb[lo:hi].double().pow(2).sum().clamp_min(1e-300)
    return float((num / den).sqrt())


def compare_traces(a: dict, b: dict, T: int, cut: int) -> dict:
    """{layer: {"x": (rms H1, rms H2), "res": (rms H1, rms H2), "ratio": |x| / |residual| of b}}: the RMS-relative error of a
    against b of the random projections of the layer outputs (a Johnson-Lindenstrauss sketch: about 1.5 % accurate per half at
    >= 700 rows), rows [0,cut) and [cut,T) separately."""
    out = {}
    for L in sorted(set(a) & set(b)):
        row = {nm: (_rms_rel(a[L][i], b[L][i], 0, cut), _rms_rel(a[L][i], b[L][i], cut, T)) for nm, i in (("x", 0), ("res", 1))}
        row["ratio"] = float(b[L][0].double().pow(2).sum().sqrt() / b[L][1].double().pow(2).sum().sqrt().clamp_min(1e-300))
        out[L] = row
    return out


# --- the verdict --------------------------------------------------------------------------------------------------------------
def _noise_bounds(r: dict, nm: str, k: float):
    """criterion=noise: (rms bound, bad-row bound, per-half rms bounds, text) for state nm from this rank's two run-to-run records."""
    n1 = (r.get("noise_plain") or {}).get(nm)
    n2 = (r.get("noise_overlap") or {}).get(nm)
    if n1 is None or n2 is None:
        return None
    nr = max(n1["rms"], n2["rms"])
    nb = max(n1["bad_rows"], n2["bad_rows"])
    halves = {}
    for h in ("h1", "h2"):
        if h in n1 and h in n2:
            halves[h] = max(NOISE_MIN_RMS, k * max(n1[h]["rms"], n2[h]["rms"]))
    return (max(NOISE_MIN_RMS, k * nr), max(NOISE_MIN_BAD, k * nb), halves,
            f" (noise bound = max({NOISE_MIN_RMS:g}, {k:g} x run-to-run {nr:.2e}))")


def evaluate_kinds(allr: list[dict], rtol: float, criterion: str = "absolute", noise_k: float = DEFAULT_NOISE_K) -> list[tuple[str, str]]:
    """[(kind, text)] of the problems found in the exchanged self-test records (same decision on every rank). kind: "numeric" (a state
    deviates more than allowed) or "structural" (a rank failed, a state is not finite, ranks disagree). criterion "absolute": every
    state within rtol (RMS-relative) and at most BAD_ROW_FRAC of its rows with > ROW_TOL error. criterion "noise": the same, but the
    bounds are max(NOISE_MIN_*, noise_k x the run-to-run noise of the plain and of the overlap path on these very inputs), also for each
    half separately."""
    problems: list[tuple[str, str]] = []
    for r in allr:
        if r.get("err"):
            problems.append(("structural", f"rank {r['rank']}: {r['err']}"))
            continue
        for nm, m in r["metrics"].items():
            if not m["finite"]:
                problems.append(("structural", f"rank {r['rank']}: {nm} is not finite"))
                continue
            bound_rms, bound_bad, halves, why = rtol, BAD_ROW_FRAC, {}, ""
            if criterion == "noise":
                nb = _noise_bounds(r, nm, noise_k)
                if nb is not None:
                    bound_rms, bound_bad, halves, why = nb
            if not m["rms"] <= bound_rms:
                problems.append(("numeric", f"rank {r['rank']}: {nm} RMS-relative error {m['rms']:.3e} > {bound_rms:.1e}{why}"))
            elif m["bad_rows"] > bound_bad:
                problems.append(("numeric", f"rank {r['rank']}: {nm}: {100 * m['bad_rows']:.2f} % of the rows differ by > {ROW_TOL:.0%}{why}"))
            else:
                for h, hb in halves.items():
                    if h in m and not m[h]["rms"] <= hb:
                        problems.append(("numeric", f"rank {r['rank']}: {nm} {h.upper()} RMS-relative error {m[h]['rms']:.3e} > {hb:.1e}{why}"))
    ref = next((r for r in allr if not r.get("err")), None)
    if ref is not None:
        for key in ("sums", "log_hash", "log_n", "geo", "xlog_hash", "xlog_n"):
            vals = [json_key(r.get(key)) for r in allr if not r.get("err")]
            if len(set(vals)) > 1:
                what = {"sums": "the overlap result is not bit-identical across ranks",
                        "log_hash": "the per-rank collective schedules differ", "log_n": "collective counts differ",
                        "geo": "split geometry differs",
                        "xlog_hash": "the per-rank indexer-exchange schedules differ (row_shard_sizes differ between ranks?)",
                        "xlog_n": "indexer-exchange counts differ"}[key]
                problems.append(("structural", what))
    return problems


def evaluate(allr: list[dict], rtol: float, criterion: str = "absolute", noise_k: float = DEFAULT_NOISE_K) -> list[str]:
    """The texts of evaluate_kinds()."""
    return [t for _, t in evaluate_kinds(allr, rtol, criterion, noise_k)]


# --- the failure report ---------------------------------------------------------------------------------------------------
def _fmt(v, f: str = "{:.2e}") -> str:
    return "n/a" if v is None or v != v else f.format(v)


def _pc(v) -> str:
    return "n/a" if v is None or v != v else f"{100 * v:.1f}%"


def _ratio(num: float, den: float | None) -> str:
    if den is None:
        return "n/a"
    r = num / den if den > 1e-12 else float("inf")
    return ">1e4" if r >= 1e4 else f"{r:.2f}"


def format_state_table(metrics: dict) -> list[str]:
    """One line per carried state: global and per-half RMS-relative error, amax-relative error, the share of rows above 10 % / 50 %
    error, the per-row error percentiles, and how much of the squared error the worst row / the worst ten rows carry."""
    out = [f"  {'state':<11}{'RMS':>10}{'H1 RMS':>10}{'H2 RMS':>10}{'amax':>10}{'>10%rows':>9}{'>50%rows':>9}{'p50':>9}{'p90':>9}{'p99':>9}"
           f"{'worst':>9}{'top1':>6}{'top10':>6}{'|ref| max/med':>14}"]
    for nm, m in metrics.items():
        h1, h2 = (m.get("h1") or {}).get("rms"), (m.get("h2") or {}).get("rms")
        out.append(f"  {nm:<11}{_fmt(m['rms']):>10}{_fmt(h1):>10}{_fmt(h2):>10}{_fmt(m['amax']):>10}{_pc(m.get('frac10')):>9}{_pc(m['bad_rows']):>9}"
                   f"{_fmt(m.get('p50'), '{:.1e}'):>9}{_fmt(m.get('p90'), '{:.1e}'):>9}{_fmt(m.get('p99'), '{:.1e}'):>9}{_fmt(m.get('max_rel'), '{:.1e}'):>9}"
                   f"{_pc(m.get('top1_share')):>6}{_pc(m.get('top10_share')):>6}{_fmt(m.get('norm_ratio'), '{:.1f}'):>14}")
    return out


def format_trace_table(title: str, columns: list, T: int, cut: int, key: str, ratio_from: dict | None) -> list[str]:
    """columns: [(heading, {layer: row})] from compare_traces; key "x" or "res"."""
    out = [title, "  " + f"{'L':>3} " + "".join(f"{h[:21]:>22}" for h, _ in columns) + (f"{'|x|/|res|':>11}" if ratio_from else "")]
    layers = sorted(set().union(*[set(c) for _, c in columns])) if columns else []
    for L in layers:
        cells = []
        for _, c in columns:
            v = c.get(L)
            cells.append(f"{_fmt(v[key][0], '{:.1e}'):>11}{_fmt(v[key][1], '{:.1e}'):>11}" if v else f"{'n/a':>11}{'n/a':>11}")
        ratio = f"{_fmt((ratio_from.get(L) or {}).get('ratio'), '{:.2f}'):>11}" if ratio_from else ""
        out.append(f"  {L:>3} " + "".join(cells) + ratio)
    return out


def routing_attribution(pa: torch.Tensor, pb: torch.Tensor, mask: torch.Tensor):
    """The error of the projections pa against pb split by a per-row flip mask: (RMS-relative error over the rows whose expert set did NOT change,
    over the rows whose set changed, share of the squared error carried by the changed rows). NaN where a group is empty."""
    err = (pa - pb).double().pow(2).sum(dim=1)
    ref = pb.double().pow(2).sum(dim=1)
    m = mask.bool()
    if m.shape[0] != err.shape[0]:
        return None

    def rms(sel):
        return float((err[sel].sum() / ref[sel].sum().clamp_min(1e-300)).sqrt()) if bool(sel.any()) else float("nan")
    return rms(~m), rms(m), float(err[m].sum() / err.sum().clamp_min(1e-300))


def format_attribution_table(routing: dict, runs: dict) -> tuple[list[str], dict]:
    """Per layer: how much of the x error (random-projection estimate) sits on the tokens whose expert set changed -- for overlap-vs-plain and for
    the plain path against itself. Returns the lines and {pair: {layer: (flipped share, rms stable, rms flipped, error share of flipped)}}."""
    pairs = (("overlap-v-plain", "overlap", "plain"), ("plain2-v-plain", "plain2", "plain"))
    out = ["x error by routing (rows whose expert set changed vs unchanged in the SAME pair of runs; random-projection estimate)",
           "  " + f"{'L':>3} " + "".join(f"{n:>13}{'':>1}{'RMS stable':>11}{'RMS flipped':>12}{'err share':>10}" for n, _, _ in pairs)]
    data: dict = {n: {} for n, _, _ in pairs}
    for L in sorted(routing["margin"]):
        cells = ""
        for n, a, b in pairs:
            mask = (routing.get("masks") or {}).get(n, {}).get(L)
            res = None
            if mask is not None and a in runs and b in runs and L in runs[a] and L in runs[b]:
                att = routing_attribution(runs[a][L][0], runs[b][L][0], mask)
                if att is not None:
                    res = (float(mask.double().mean()),) + att
            if res is None:
                cells += f"{'n/a':>14}{'n/a':>11}{'n/a':>12}{'n/a':>10}"
            else:
                data[n][L] = res
                cells += f"{_pc(res[0]):>13} {_fmt(res[1], '{:.1e}'):>11}{_fmt(res[2], '{:.1e}'):>12}{_pc(res[3]):>10}"
        out.append(f"  {L:>3} {cells}")
    return out, data


def format_routing_table(routing: dict) -> list[str]:
    """Per layer: the share of tokens whose expert set differs between two runs (overlap-v-plain, the two noise pairs, serial-v-plain,
    overlap-v-serial), rows of H1 | H2, and the share of tokens within 1e-2 / 1e-3 of the selection boundary in the plain run."""
    names = ["overlap-v-plain", "plain2-v-plain", "overlap2-v-overlap", "serial-v-plain", "overlap-v-serial"]
    out = [f"routing flips: share of tokens whose top-{routing['topk']} expert set differs between two runs (emulated from the gate logits with the "
           f"router's rule; {routing['hooked']} layers hooked), rows of H1 | H2",
           "  " + f"{'L':>3} " + "".join(f"{n[:21]:>22}" for n in names) + f"{'gap<1e-2':>10}{'gap<1e-3':>10}"]
    layers = sorted(routing["margin"])
    for L in layers:
        cells = ""
        for n in names:
            v = routing["pairs"].get(n, {}).get(L)
            cells += f"{_pc(v[1]):>11}{_pc(v[2]):>11}" if v else f"{'n/a':>11}{'n/a':>11}"
        m = routing["margin"][L]
        out.append(f"  {L:>3} {cells}{_pc(m[0]):>10}{_pc(m[1]):>10}")
    return out


def selftest_hints(main: dict, noise_p: dict | None, noise_o: dict | None, machinery: dict | None, cmp: dict, cut: int, T: int,
                   routing: dict | None = None) -> list[str]:
    """Plain-language reading of the numbers (heuristics, in this order of strength)."""
    hints: list[str] = []
    if not main:
        return hints
    worst = max(main, key=lambda nm: main[nm]["rms"] if main[nm]["rms"] == main[nm]["rms"] else 0.0)
    m = main[worst]
    dev = m["rms"]
    np_, no_ = (noise_p or {}).get(worst), (noise_o or {}).get(worst)
    nz = max([v["rms"] for v in (np_, no_) if v is not None], default=None)
    mach = (machinery or {}).get(worst)
    if nz is not None:
        if dev <= 2.0 * nz:
            hints.append(f"hint: NOISE: {worst}: the same path run twice already differs by {nz:.2e} (plain {_fmt(np_ and np_['rms'])}, overlap {_fmt(no_ and no_['rms'])}); "
                         f"overlap-vs-plain {dev:.2e} is {_ratio(dev, nz)}x that: consistent with a nondeterministic kernel / expert-routing flips, "
                         "not with a scheduling or data error; an absolute tolerance below this run-to-run noise cannot be met by this stack for this state")
        else:
            hints.append(f"hint: {worst}: overlap-vs-plain {dev:.2e} is {_ratio(dev, nz)}x the run-to-run noise {nz:.2e} of the same path "
                         f"(plain {_fmt(np_ and np_['rms'])}, overlap {_fmt(no_ and no_['rms'])}): NOT explained by kernel nondeterminism alone")
    if mach is not None:
        if mach["rms"] >= 0.5 * dev and (nz is None or mach["rms"] > 2.0 * nz):
            hints.append(f"hint: SCHEDULING: overlap differs from the same pipeline run serially on one stream by {mach['rms']:.2e} (>= half of the deviation, "
                         f"{_fmt(nz)} noise): suspect a missing event / record_stream or a race; see the overlap-vs-serial columns")
        elif nz is None or dev > 2.0 * nz:
            hints.append(f"hint: HALVING / MATH: the serial run on one stream matches the overlap (overlap-vs-serial {mach['rms']:.2e}), so streams and events are not "
                         "the cause; the difference comes from splitting the batch in two halves (M-dependent kernels, reduction order) or from per-half state / metadata")
    if m.get("h1") and m.get("h2"):
        a, b = m["h1"]["rms"], m["h2"]["rms"]
        if max(a, b) > 3.0 * max(min(a, b), 1e-30):
            hints.append(f"hint: ONE-SIDED: {worst} H1 {a:.2e} vs H2 {b:.2e}: the {'second' if b > a else 'first'} half carries the error")
    if m.get("top1_share", 0.0) > 0.5:
        r = m["top"][0] if m.get("top") else None
        hints.append(f"hint: ONE ROW DOMINATES {worst}: row {r[0] if r else '?'} carries {_pc(m['top1_share'])} of the squared error (its reference norm {_fmt(r and r[2], '{:.3g}')}; "
                     f"largest/median reference row norm {_fmt(m.get('norm_ratio'), '{:.1f}')}): a massive-activation row, not a broad error")
    elif m["bad_rows"] >= 0.25:
        hints.append(f"hint: BROAD: {worst}: {_pc(m['bad_rows'])} of the rows differ by > {ROW_TOL:.0%} and {_pc(m.get('frac10'))} by > 10 % (median row error {_fmt(m.get('p50'), '{:.1e}')}): "
                     "most rows are off, not a few outliers: a systematic error or a noise level that every row sees")
    elif m["bad_rows"] > 0.0:
        hints.append(f"hint: FEW ROWS: {worst}: {_pc(m['bad_rows'])} of the rows differ by > {ROW_TOL:.0%} and {_pc(m.get('frac10'))} by > 10 % on a quiet background "
                     f"(median row error {_fmt(m.get('p50'), '{:.1e}')}): the signature of expert-routing flips (a marginal expert swapped changes a token's MoE output by tens of percent)")
    if routing and routing.get("pairs"):
        last = max(routing["margin"]) if routing.get("margin") else None
        ov = (routing["pairs"].get("overlap-v-plain") or {}).get(last)
        nz_r = [v for v in ((routing["pairs"].get("plain2-v-plain") or {}).get(last), (routing["pairs"].get("overlap2-v-overlap") or {}).get(last)) if v]
        if ov and nz_r:
            nzr = max(v[0] for v in nz_r)
            hints.append(f"hint: ROUTING at layer {last}: {_pc(ov[0])} of the tokens pick a different expert set in overlap vs plain, {_pc(nzr)} between two runs of the same path; "
                         + ("the flip rate is the same between identical runs: routing flips are noise of this stack, not caused by the overlap" if ov[0] <= 2.0 * max(nzr, 1e-9)
                            else "the flip rate is higher than between identical runs: the overlap perturbs the router inputs more than run-to-run noise does"))
    att = (routing or {}).get("attribution", {}).get("overlap-v-plain") if routing else None
    if att:
        last = max(att)
        fl, r_stable, r_flip, share = att[last]
        d = ((cmp.get("ov_p") or {}).get(last) or {}).get("x")
        if fl == fl and r_stable == r_stable:
            hints.append(f"hint: ATTRIBUTION at layer {last}: {_pc(fl)} of the tokens changed their expert set and carry {_pc(share)} of the squared x error; the tokens with unchanged "
                         f"routing agree to {_fmt(r_stable)} (flipped tokens {_fmt(r_flip)})"
                         + ("; the x deviation IS the routing flips" if share >= 0.7 and d and r_stable <= 0.35 * max(d) else ""))
    if cmp.get("ov_p") and cmp.get("o2_o") and cmp.get("p2_p"):
        for hi, h in enumerate(("H1", "H2")):
            for key in ("x", "res"):
                first = None
                for L, row in sorted(cmp["ov_p"].items()):
                    nzl = max(((cmp["o2_o"].get(L) or {}).get(key, (0.0, 0.0))[hi] or 0.0), ((cmp["p2_p"].get(L) or {}).get(key, (0.0, 0.0))[hi] or 0.0))
                    if row[key][hi] == row[key][hi] and row[key][hi] > max(3.0 * nzl, 1e-3):
                        first = L
                        break
                hints.append(f"hint: first layer where the {key} trace of {h} exceeds 3x the run-to-run noise: {first if first is not None else 'none'}")
    return hints


def build_selftest_report(allr: list[dict], diag: list[dict] | None, problems: list[tuple[str, str]], rtol: float, criterion: str,
                          noise_k: float) -> tuple[list[str], list[str]]:
    """(summary lines, per-layer lines) of the failure report -- a pure function of the exchanged records, so every rank builds the
    same text. diag: the phase-2 records (serial run, traces) or None."""
    ref = next((r for r in allr if not r.get("err")), None)
    lines: list[str] = []
    trace_lines: list[str] = []
    if ref is None:
        return ["no rank produced metrics: " + "; ".join(str(r.get("err"))[:160] for r in allr[:3])], []
    geo = ref.get("geo")
    cut = T = None
    try:
        T, specs, split = geo
        cut = split.cut
        lines.append(f"geometry: T={T} rows on {len(allr)} ranks; cut at row {cut} ({split.kind} cut, request {split.req}): "
                     f"H1 = {specs[0].rows} rows, H2 = {specs[1].rows} rows")
    except Exception:  # noqa: BLE001
        lines.append(f"geometry: {geo}")
    terr = [f"rank {r['rank']}: {r['trace_err']}" for r in allr if r.get("trace_err")]
    if terr:
        lines.append("trace/probe setup failed: " + "; ".join(terr[:2]))
    if criterion == "noise":
        lines.append(f"criterion: noise -- per state RMS <= max({NOISE_MIN_RMS:g}, {noise_k:g} x run-to-run noise), rows with > {ROW_TOL:.0%} error <= max({NOISE_MIN_BAD:g}, "
                     f"{noise_k:g} x noise), each half likewise")
    else:
        lines.append(f"criterion: absolute -- per state RMS <= {rtol:.1e} and <= {100 * BAD_ROW_FRAC:.0f} % of the rows with > {ROW_TOL:.0%} error")
    same = len({json_key({k: v for k, v in r["metrics"].items()}) for r in allr if not r.get("err")}) == 1
    lines.append(f"final carried states, overlap vs plain (rank {ref['rank']}; metrics identical on every rank: {'yes' if same else 'NO'}):")
    lines += format_state_table(ref["metrics"])
    mach = None
    machs = [d.get("machinery") for d in (diag or []) if d and not d.get("err") and d.get("machinery")]
    if machs:
        mach = machs[0]
    np_, no_ = ref.get("noise_plain"), ref.get("noise_overlap")
    if np_ or no_ or mach:
        lines.append(f"run-to-run noise and decomposition (RMS-relative error; same inputs): 'plain2' = the plain path run again, 'overlap2' = the overlap path run again, "
                     "'serial' = the same pipeline on ONE stream (no concurrency):")
        lines.append(f"  {'state':<11}{'plain2-v-plain':>16}{'overlap2-v-ov':>15}{'ov-v-serial':>13}{'ov-v-plain':>12}{'ov-v-plain / noise':>20}")
        for nm, m in ref["metrics"].items():
            a, b = (np_ or {}).get(nm), (no_ or {}).get(nm)
            c = (mach or {}).get(nm)
            nz = max([v["rms"] for v in (a, b) if v is not None], default=None)
            lines.append(f"  {nm:<11}{_fmt(a and a['rms']):>16}{_fmt(b and b['rms']):>15}{_fmt(c and c['rms']):>13}{_fmt(m['rms']):>12}{_ratio(m['rms'], nz):>20}")
    for nm, m in ref["metrics"].items():
        if m.get("top") and (nm == "x" or any(nm in t for _, t in problems)):
            tops = ", ".join(f"row {r}{'' if cut is None else (' [H1]' if r < cut else ' [H2]')} err {e:.2g} (ref norm {nr:.3g})" for r, e, nr in m["top"])
            lines.append(f"worst rows of {nm}: {tops}")
    # per-layer traces
    cmp: dict = {}
    traces = [d.get("trace") for d in (diag or [])] if diag else []
    if traces and all(t for t in traces) and cut is not None:
        asm = {run: assemble_trace(traces, run) for run in ("plain", "plain2", "overlap", "overlap2", "serial")}
        runs = {k: v[1] for k, v in asm.items() if v is not None}
        Tt = next((v[0] for v in asm.values() if v is not None), T)

        def cmp_pair(a, b):
            return compare_traces(runs[a], runs[b], Tt, cut) if a in runs and b in runs else {}

        cmp = {"ov_p": cmp_pair("overlap", "plain"), "p2_p": cmp_pair("plain2", "plain"), "o2_o": cmp_pair("overlap2", "overlap"),
               "s_p": cmp_pair("serial", "plain"), "o_s": cmp_pair("overlap", "serial")}
        cols = [("overlap-v-plain", cmp["ov_p"]), ("plain2-v-plain", cmp["p2_p"]), ("overlap2-v-overlap", cmp["o2_o"]), ("serial-v-plain", cmp["s_p"]),
                ("overlap-v-serial", cmp["o_s"])]
        for key, title in (("x", "x (the layer's reduce-scattered MoE output)"), ("res", "residual stream after the layer's attention-side update")):
            trace_lines += format_trace_table(f"per-layer trace of {title}: RMS-relative error of random projections (k={traces[0]['k']}), rows of H1 | H2"
                                              + ("; '|x|/|res|' = size of x relative to the residual (reference run)" if key == "x" else ""),
                                              cols, Tt, cut, key, cmp["ov_p"] if key == "x" else None)
        routing = (traces[0] or {}).get("routing")
        if routing:
            trace_lines += format_routing_table(routing)
            try:
                att_lines, att = format_attribution_table(routing, {run: {L: v for L, v in layers.items()} for run, layers in runs.items()})
                trace_lines += att_lines
                routing["attribution"] = att
            except Exception as e:  # noqa: BLE001
                trace_lines.append(f"x error by routing: unavailable ({type(e).__name__}: {str(e)[:80]})")
        else:
            trace_lines.append("routing flips: no router-gate call was recorded (no hook fired or the gate is not the DSV4 noaux gate): not available")
        errs = [e for t in traces for e in t.get("errors", [])] + [e for t in traces for e in ((t or {}).get("routing") or {}).get("errors", [])]
        if errs:
            trace_lines.append("trace errors: " + "; ".join(errs[:3]))
    elif diag is not None:
        trace_lines.append("per-layer trace unavailable: " + "; ".join(str(d.get("err") or "no trace") for d in diag[:2]))
    lines += selftest_hints(ref["metrics"], np_, no_, mach, cmp, cut or 0, T or 0, ((traces[0] or {}).get("routing") if traces and traces[0] else None))
    return lines, trace_lines


def dedupe_problems(texts: list[str], nranks: int) -> list[str]:
    """The same problem on every rank is reported once ("all 8 ranks: ...")."""
    groups: dict[str, list] = {}
    for t in texts:
        mt = re.match(r"rank (\d+): (.*)", t, re.S)
        key, rk = (mt.group(2), int(mt.group(1))) if mt else (t, None)
        groups.setdefault(key, []).append(rk)
    out = []
    for key, ranks in groups.items():
        if nranks > 1 and None not in ranks and len(set(ranks)) == nranks:
            out.append(f"all {nranks} ranks: {key}")
        else:
            out += [f"rank {r}: {key}" if r is not None else key for r in ranks]
    return out


def json_key(v) -> str:
    return repr(v)


# ---------------------------------------------------------------------------------------------------------------------
# 7. The region
# ---------------------------------------------------------------------------------------------------------------------
def _fingerprint(model, world_size: int) -> str:
    parts = ["p028", f"rev={REVISION}", f"module={MODULE_SHA16}", f"min_t={_STATE.min_t}", f"tp={world_size}",
             f"layers={model.start_layer}-{model.decoder_replay_start}", f"ctl={'1' if os.environ.get(ENV_CTL) else '0'}",
             f"prio={os.environ.get(ENV_PRIO, '')}", f"align={CUT_ALIGN}", f"minhalf={MIN_HALF_ROWS}",
             f"repeat={int(parse_repeat())}", f"trace={int(parse_trace())}", f"rtol={parse_rtol(os.environ.get(ENV_RTOL))}",
             f"selftest={parse_selftest(os.environ.get(ENV_SELFTEST))}", f"criterion={parse_criterion(os.environ.get(ENV_CRITERION))}",
             f"noise_k={parse_noise_k(os.environ.get(ENV_NOISE_K))}"]
    mode, min_pos = indexer_settings()
    if mode != "off":                                       # off: the fingerprint keeps its revision-2 shape (plus the revision number)
        parts.append(f"indexer={mode}:{min_pos}:{EXCHANGE_SRC_SHA[:12]}")
    if parse_pool(os.environ.get(ENV_POOL)) != "compute":   # revision 4: only the non-default settings extend the fingerprint
        parts.append(f"pool={parse_pool(os.environ.get(ENV_POOL))}")
    if parse_empty_cache():
        parts.append("emptycache=1")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _exchange(world, obj, what: str):
    """gloo all-gather of a per-rank object behind a bounded monitored barrier (a missing rank raises on every present one)."""
    try:
        world.barrier_bounded(sp.SELFTEST_BARRIER_S)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{TAG} {ENV}: {what}: not every TP rank reached the exchange within {sp.SELFTEST_BARRIER_S:.0f} s "
                           f"({type(e).__name__}: {str(e)[:300]}); refusing - a rank failed or is stuck before it (see that "
                           "rank's log)") from e
    return world.all_gather_object(obj)


class Region:
    """What replaces `_run_layers(...)` (+ the 022 shard_input / exit_region around it) for ONE eligible forward."""

    def __init__(self, model, handle, pend: Pending | None, mode: str, hidden_states, positions, input_ids) -> None:
        self.model, self.handle, self.pend, self.mode = model, handle, pend, mode
        self.positions, self.input_ids = positions, input_ids
        self.closed = False

    # -- plain SP (exactly what model.py does without this patch) ---------------------------------------------------------
    def _run_plain(self, layer_ids, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual, aux, eh, em):
        m, h = self.model, self.handle
        hs = sp.shard_input(hidden_states, h)
        hs, residual, post_mix, res_mix, pre_mix = m._run_layers(layer_ids, hs, positions, input_ids, pre_mix, post_mix, res_mix,
                                                                  residual, aux, eh, em)
        return sp.exit_region(h, hs, residual, post_mix, res_mix, pre_mix)

    # -- overlap ----------------------------------------------------------------------------------------------------------
    def _halves(self):
        """(halves, env): production = per-half metadata + forward contexts; the CPU suite injects its own."""
        if _HALVES_FACTORY is not None:
            return _HALVES_FACTORY(self.pend, self.model)
        halves, topk = build_half_contexts(self.pend, self.model, _STATE)
        return halves, ProdEnv(halves, topk)

    def _run_pipeline(self, layer_ids, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual, aux_out, eh, em,
                      halves=None, env=None, rt=None, tracer=None):
        st = _STATE
        m = self.model
        if not (pre_mix is None and post_mix is None and res_mix is None and residual is None):
            raise RuntimeError(f"{TAG} the region must start from the embedding (pipeline-parallel input is not supported)")
        if list(layer_ids) != list(range(m.start_layer, m.decoder_replay_start)):
            raise RuntimeError(f"{TAG} layer range {layer_ids} is not the SP region")
        if halves is None:
            halves, env = self._halves()
        plans = [sp.plan_rows(h.spec.rows, self.handle.world.size) for h in halves]
        layers = [m.layers[i] for i in layer_ids]
        pipe = Pipeline(rt or st.rt, self.handle.world, m, layers, list(layer_ids), env, halves, plans, sp._ops(), hidden_states,
                        positions, input_ids, eh, em, getattr(m, "aux_hidden_state_layers", ()), tracer=tracer)
        out = pipe.run()
        aux_out.update(pipe.aux_out)
        if st.keep_debug:
            st.last_pipe = pipe
        return out

    def run(self, layer_ids, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual, aux_hidden_by_layer,
            engram_hashes=None, engram_mask=None):
        st = _STATE
        rt = st.rt
        t0 = time.perf_counter()
        rt.refresh()
        ev0 = rt.timing_event(rt.compute)
        args = (layer_ids, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)
        if self.mode == "plain":
            kind = "plain"
            out = self._run_plain(*args, aux_hidden_by_layer, engram_hashes, engram_mask)
        elif not st.verified:
            kind = "selftest"                                   # three region runs (plain, overlap, overlap): kept out of the averages
            out = self._run_selftest(*args, aux_hidden_by_layer, engram_hashes, engram_mask)
        else:
            kind = "overlap"
            out = self._run_pipeline(*args, aux_hidden_by_layer, engram_hashes, engram_mask)
        ev1 = rt.timing_event(rt.compute)
        st.counters[f"region_{self.mode}"] += 1
        st.timing.append((kind, int(positions.shape[0]), ev0, ev1, time.perf_counter() - t0))
        if kind != "plain" and st.empty_cache:                 # emergency fallback (default off): hand the cached blocks of BOTH pools back
            rt.empty_cache()
            st.counters["empty_cache"] += 1
        _sample_mem(rt, kind)
        _maybe_log()
        return out

    # -- self-test --------------------------------------------------------------------------------------------------------
    def _metrics(self, out, ref, aux, aux_ref, cut) -> dict:
        """State metrics of the five carried states (+ aux taps) of a run `out` against the reference run `ref`."""
        names = ("x", "residual", "post_mix", "res_mix", "pre_mix")
        met = {nm: state_metrics(b, a, cut=cut) for nm, a, b in zip(names, ref, out)}
        for k in sorted(aux_ref):
            met[f"aux[{k}]"] = state_metrics(aux[k], aux_ref[k], cut=cut)
        return met

    def _selftest_run(self, kind, name, tracer, parts, args, aux, eh, em, rt=None):
        """One awaited run of the region for the self-test: kind "plain" (the 022 region) or "pipe" (the overlap pipeline; on `rt` it
        is the same pipeline serially on one stream). The tracer records the per-layer rows of the run."""
        st = _STATE
        use = rt or st.rt
        if tracer is not None:
            tracer.begin(name, parts)
        try:
            if kind == "plain":
                with _plain_layer_trace(self.model, tracer):
                    out = self._run_plain(*args, aux, eh, em)
            else:
                out = self._run_pipeline(*args, aux, eh, em, rt=rt, tracer=tracer)
            use.bounded_sync(use.compute, SYNC_TIMEOUT_S, f"self-test run '{name}'")
        except BaseException:
            if tracer is not None:
                tracer.abort()
            raise
        if tracer is not None:
            tracer.end()
        return out

    def _run_selftest(self, layer_ids, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual, aux_out, eh, em):
        """The boot self-test (see _run_selftest_body) with its memory record: MemAvailable / /proc/meminfo / allocator sampled at the phase
        boundaries and every 100 ms, one INFO line per rank at the end (also when the test raises)."""
        st = _STATE
        smem = st.selftest_mem = _SelfTestMem(st.rt, self.handle.world.rank)
        smem.start()
        try:
            return self._run_selftest_body(smem, layer_ids, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual, aux_out, eh, em)
        finally:
            smem.stop()

    def _run_selftest_body(self, smem, layer_ids, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual, aux_out, eh, em):
        st = _STATE
        world, rt, m = self.handle.world, st.rt, self.model
        t0 = time.time()
        rec: dict = {"rank": world.rank, "err": None, "geo": None}
        aux_a: dict = {}
        aux_b: dict = {}
        stream_ok = True
        try:
            rec["geo"] = (self.pend.num_tokens, tuple(self.pend.specs), self.pend.split)
            if hasattr(rt, "current_matches"):
                stream_ok = rt.current_matches(rt.comm)
            snap = _RingSnapshot(_ring_blocks(m, self.pend)) if self.pend.runner is not None else None
        except Exception as e:  # noqa: BLE001 - every rank must still reach the exchange
            rec["err"] = f"setup: {type(e).__name__}: {str(e)[:300]}"
            snap = None
        votes = _exchange(world, {"rank": world.rank, "ok": rec["err"] is None and stream_ok, "err": rec["err"],
                                  "stream": stream_ok, "geo": repr(rec["geo"])}, "self-test setup")
        bad = [v for v in votes if not v["ok"]]
        if bad or len({v["geo"] for v in votes}) != 1:
            raise RuntimeError(f"{TAG} {ENV}: self-test setup refused: "
                               + "; ".join(f"rank {v['rank']}: {v['err'] or ('vLLM current_stream() does not follow torch.cuda.stream()' if not v['stream'] else 'geometry differs')}"
                                           for v in (bad or votes))[:1500] + f"; unset {ENV} to serve without it")
        args = (layer_ids, hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual)
        repeat = parse_repeat()
        rtol, criterion, noise_k, mode = (parse_rtol(os.environ.get(ENV_RTOL)), parse_criterion(os.environ.get(ENV_CRITERION)),
                                          parse_noise_k(os.environ.get(ENV_NOISE_K)), parse_selftest(os.environ.get(ENV_SELFTEST)))
        cut = int(self.pend.split.cut)
        T, W = int(positions.shape[0]), int(world.size)
        parts_plain = [(0, T, self.handle.plan.S)]
        parts_ov = [(sp_.tok_lo, sp_.rows, sp.plan_rows(sp_.rows, W).S) for sp_ in self.pend.specs]
        tracer = None
        if parse_trace():
            try:
                tracer = Tracer(hidden_states.device, int(hidden_states.shape[-1]), int(m.layers[layer_ids[0]].hc_mult),
                                layers=[m.layers[i] for i in layer_ids])
            except Exception as e:  # noqa: BLE001 - diagnostics only
                rec["trace_err"] = f"{type(e).__name__}: {str(e)[:200]}"
        b_out = None
        # one fetch of the Engram rows for ALL runs (also the diagnostic ones): the NVMe channel hands a publish over once
        with _engram_rows_once(m), contextlib.ExitStack() as _closer:
            if tracer is not None:
                _closer.callback(tracer.close)
            try:
                # ONE reference is retained at any time (the previous result of the chain plain -> plain2 -> overlap -> overlap2): each pair is
                # compared as soon as both results exist, the older one is dropped at once; the new result of a run is the forward's own output
                smem.phase("plain")
                a_out = self._selftest_run("plain", "plain", tracer, parts_plain, args, aux_a, eh, em)
                smem.retain(_nbytes(a_out) + _nbytes(aux_a))
                if repeat:                                           # the plain path against itself: the noise floor of the reference
                    if snap is not None:
                        snap.restore()
                    smem.phase("plain2")
                    aux_a2: dict = {}
                    a2_out = self._selftest_run("plain", "plain2", tracer, parts_plain, args, aux_a2, eh, em)
                    smem.phase("compare plain2")
                    rec["noise_plain"] = self._metrics(a2_out, a_out, aux_a2, aux_a, cut)
                    del a2_out
                    aux_a2.clear()
                if snap is not None:
                    snap.restore()
                smem.phase("overlap")
                b_out = self._selftest_run("pipe", "overlap", tracer, parts_ov, args, aux_b, eh, em)
                if set(aux_a) != set(aux_b):
                    raise RuntimeError(f"aux taps differ: plain {sorted(aux_a)}, overlap {sorted(aux_b)}")
                smem.phase("compare overlap")
                rec["metrics"] = self._metrics(b_out, a_out, aux_b, aux_a, cut)
                rec["sums"] = tuple(checksum(t) for t in b_out) + tuple(checksum(aux_b[k]) for k in sorted(aux_b))
                del a_out
                aux_a.clear()                                        # the plain reference is not needed any more: the overlap result is the next one
                smem.retain(_nbytes(b_out) + _nbytes(aux_b))
                if repeat:                                           # the overlap path against itself
                    if snap is not None:
                        snap.restore()
                    smem.phase("overlap2")
                    aux_b2: dict = {}
                    b2_out = self._selftest_run("pipe", "overlap2", tracer, parts_ov, args, aux_b2, eh, em)
                    smem.phase("compare overlap2")
                    rec["noise_overlap"] = self._metrics(b2_out, b_out, aux_b2, aux_b, cut)
                    del b2_out
                    aux_b2.clear()
                rec["log_hash"] = rt.log_hash.hexdigest()[:16]
                rec["log_n"] = rt.log_n
                rec["redirect"] = (int(getattr(rt, "redirect_n", 0)), int(getattr(rt, "redirect_bytes", 0)),
                                   int(getattr(rt, "pad_n", 0)), int(getattr(rt, "pad_bytes", 0)))     # the last pipeline run
                if st.idx_comm is not None:                          # the indexer exchanges of the last pipeline run: same schedule on every rank
                    xh = getattr(rt, "xlog_hash", None)
                    rec["xlog_hash"] = xh.hexdigest()[:16] if xh is not None else None
                    rec["xlog_n"] = getattr(rt, "xlog_n", None)
            except Exception as e:  # noqa: BLE001
                rec["err"] = f"{type(e).__name__}: {str(e)[:400]}"
            smem.phase("verdict")
            allr = _exchange(world, rec, "self-test verdict")
            pool_problem = None if any(r.get("err") for r in allr) else _pool_problem(allr)
            if pool_problem is not None:                             # identical on every rank (computed from the exchanged records): a clean raise everywhere
                raise RuntimeError(f"{TAG} {ENV}=1: {pool_problem}")
            problems = evaluate_kinds(allr, rtol, criterion, noise_k)
            diag = self._diagnose(allr, b_out, args, aux_b, eh, em, snap, tracer, parts_ov, cut,
                                  serial=all(k == "numeric" for k, _ in problems)) if problems else None
        if tracer is not None:
            pr_ = tracer.probe
            st.last_selftest = {"tracer_errors": list(tracer.errors) + (list(pr_.errors) if pr_ is not None else []),
                                "runs": {n: len(r["layers"]) for n, r in tracer.runs.items()},
                                "routing_runs": {n: len(r) for n, r in (pr_.runs.items() if pr_ is not None else [])},
                                "hooked": pr_.n_hooked if pr_ is not None else 0}
        if problems:
            summary, trace_lines = build_selftest_report(allr, diag, problems, rtol, criterion, noise_k)
            texts = dedupe_problems([t for _, t in problems], len(allr))
            if world.rank == 0:
                logger.error("%s self-test FAILED -- diagnostic report:\n  %s\n%s", TAG, "\n  ".join(summary), "\n".join(trace_lines))
            else:
                logger.error("%s self-test FAILED (the diagnostic report is in the rank-0 log): %s", TAG, "; ".join(texts)[:600])
            if mode == "report" and all(k == "numeric" for k, _ in problems):
                logger.error("%s %s=report: the numeric self-test failure above is NOT enforced; the overlap path stays ON, UNVERIFIED -- diagnosis only, "
                             "do not serve production traffic like this", TAG, ENV_SELFTEST)
                st.verified = True
                st.degraded = True
                st.report = {"seconds": round(time.time() - t0, 2), "degraded": True, "problems": texts[:4], "ranks": len(allr)}
                aux_out.update(aux_b)
                return b_out
            raise RuntimeError(f"{TAG} {ENV}=1: BOOT SELF-TEST FAILED (overlap vs plain sequence-parallel region on the same "
                               f"inputs): {'; '.join(texts)[:3000]}\n" + "\n".join(summary)[:6000] + f"\nrefusing to serve - unset {ENV}")
        st.verified = True
        worst = {nm: max(r["metrics"][nm]["rms"] for r in allr) for nm in allr[0]["metrics"]}
        rep = {nm: max((r["noise_overlap"][nm]["rms"] for r in allr if r.get("noise_overlap") and nm in r["noise_overlap"]), default=None)
               for nm in worst}
        rep_p = {nm: max((r["noise_plain"][nm]["rms"] for r in allr if r.get("noise_plain") and nm in r["noise_plain"]), default=None)
                 for nm in worst}
        st.report = {"seconds": round(time.time() - t0, 2), "rms": worst, "repeat_rms": rep, "plain_repeat_rms": rep_p, "ranks": len(allr),
                     "collectives_per_region": allr[0]["log_n"], "criterion": criterion, "indexer_exchanges_per_region": allr[0].get("xlog_n")}
        f = lambda d: {k: (f"{v:.2e}" if v is not None else "n/a") for k, v in d.items()}  # noqa: E731
        logger.info("%s prefill two-half overlap ACTIVE (rank %d/%d): self-test OK on all %d ranks in %.1f s - overlap vs plain SP "
                    "RMS-relative error %s (criterion %s, tolerance %.1e); run-to-run noise of the plain path %s, of the overlap path %s; "
                    "%d logged collectives per region; NUMERICS: reduction order / GEMM tiling only.%s%s", TAG, world.rank, world.size, len(allr),
                    st.report["seconds"], f(worst), criterion, rtol, f(rep_p), f(rep), allr[0]["log_n"], _indexer_suffix(allr[0]),
                    _pool_suffix(allr))
        aux_out.update(aux_b)
        return b_out

    def _diagnose(self, allr, b_out, args, aux_b, eh, em, snap, tracer, parts_ov, cut, serial: bool = True):
        """Phase 2 of a failed self-test (collective: every rank calls it, the verdict is identical on every rank): the same pipeline serially on
        one stream, compared with the overlap result (a scheduling problem shows there, a numerical one does not), then the per-layer traces of
        all runs are exchanged. The extra run is made only for NUMERIC failures (a structural one -- a failed rank, ranks that disagree -- is
        not run again). Never raises; returns the per-rank records."""
        st = _STATE
        world, rt = self.handle.world, st.rt
        drec: dict = {"rank": world.rank, "err": None, "machinery": None, "trace": None}
        try:
            if serial and b_out is not None and all(not r.get("err") for r in allr) and hasattr(rt, "serial_twin"):
                srt = rt.serial_twin()
                if snap is not None:
                    snap.restore()
                aux_s: dict = {}
                s_out = self._selftest_run("pipe", "serial", tracer, parts_ov, args, aux_s, eh, em, rt=srt)
                drec["machinery"] = self._metrics(b_out, s_out, aux_b, aux_s, cut)
                del s_out
            if tracer is not None:
                drec["trace"] = tracer.export(cut)
        except Exception as e:  # noqa: BLE001
            drec["err"] = f"{type(e).__name__}: {str(e)[:300]}"
            drec["trace"] = tracer.export(cut) if tracer is not None else None
        return _exchange(world, drec, "self-test diagnostics")

    def close(self) -> None:
        self.closed = True
        _STATE.pending = None


def parse_repeat() -> bool:
    return _parse_int(ENV_REPEAT, os.environ.get(ENV_REPEAT), 1, 0, 1) == 1


# --- stats ------------------------------------------------------------------------------------------------------------
def _maybe_log() -> None:
    st = _STATE
    every = st.ctl_log if st.ctl_log is not None else parse_log_every(os.environ.get(ENV_LOG))
    if every <= 0:
        return
    n = st.counters["region_overlap"] + st.counters["region_plain"]
    if n - st.last_log_forward < every:
        return
    st.last_log_forward = n
    done: dict[tuple[str, str], list[float]] = {}
    keep = collections.deque(maxlen=st.timing.maxlen)
    for mode, rows, ev0, ev1, host in st.timing:
        if ev0 is None or ev1 is None:
            continue
        try:
            if not ev1.query():
                keep.append((mode, rows, ev0, ev1, host))
                continue
            ms = ev0.elapsed_time(ev1)
        except Exception:  # noqa: BLE001
            continue
        d = done.setdefault((mode, size_bucket(rows)), [0, 0.0, 0])
        d[0] += 1
        d[1] += ms
        d[2] += rows
    st.timing = keep
    for key, d in done.items():
        s = st.sums.setdefault(key, [0, 0.0, 0])
        s[0] += d[0]
        s[1] += d[1]
        s[2] += d[2]
    parts = []
    for _hi, bucket in SIZE_BUCKETS:
        for mode in ("overlap", "plain", "selftest"):
            c, ms, rows = st.sums.get((mode, bucket), (0, 0.0, 0))
            if c:
                parts.append(f"{mode}[{bucket}]: {c} regions, {rows} rows, avg {ms / c:.1f} ms ({ms / max(rows, 1) * 1000:.2f} ms per 1k rows)")
    logger.info("%s stats: eligible forwards %d (overlap %d, plain %d), skipped: %s | %s%s%s", TAG, st.forwards,
                st.counters["region_overlap"], st.counters["region_plain"],
                {k: v for k, v in sorted(st.counters.items()) if k.startswith("skip")}, "; ".join(parts) or "no completed timings yet",
                f" | indexer exchanges on the second communicator: {st.counters['idx_exchange']}" if st.idx_comm is not None else "",
                _mem_suffix(st.rt))


def _gib(b) -> str:
    return f"{b / 2**30:.2f}"


def _sample_mem(rt, kind: str) -> None:
    """At every region exit: the lowest MemAvailable seen (what a node-level OOM guard watches; GB10 has one memory for CPU and GPU) and the allocator's
    allocated / reserved bytes. One /proc read and two allocator reads; diagnostics must never take the worker down."""
    st = _STATE
    try:
        avail = _mem_available_mib()
        if avail >= 0 and (st.min_avail_mib is None or avail < st.min_avail_mib):
            st.min_avail_mib = avail
        fn = getattr(rt, "mem_basic", None)
        st.mem_basic = fn() if fn is not None else None
        if kind == "overlap" and not st.pool_first_logged:
            st.pool_first_logged = True
            mb = st.mem_basic
            logger.info("%s first overlapped region done (pool=%s): %d collective outputs / %.0f MiB and %d padded scatter inputs / %.0f MiB served "
                        "from the compute stream's pool%s; MemAvailable %.2f GiB", TAG, st.pool_mode, getattr(rt, "redirect_n", 0),
                        getattr(rt, "redirect_bytes", 0) / 2**20, getattr(rt, "pad_n", 0), getattr(rt, "pad_bytes", 0) / 2**20,
                        f"; torch.cuda allocated {_gib(mb[0])} GiB, reserved {_gib(mb[1])} GiB" if mb else "", max(avail, 0.0) / 1024)
    except Exception as e:  # noqa: BLE001
        logger.debug("%s memory sample failed: %s", TAG, e)


def _mem_suffix(rt) -> str:
    """The memory part of the stats line: allocator allocated / active / reserved (+ peaks, growth since the last line), the lowest MemAvailable
    since the last line, what the pool redirect served in the last region and, on the first lines, the reserved size of each stream's pool."""
    st = _STATE
    bits: list[str] = []
    if rt is None:
        return ""
    try:
        ms = rt.mem_stats() if hasattr(rt, "mem_stats") else None
        if ms:
            grow = "" if st.reserved_prev is None else f", {(ms['reserved'] - st.reserved_prev) / 2**30:+.2f} since the last line"
            st.reserved_prev = ms["reserved"]
            bits.append(f"torch.cuda GiB: allocated {_gib(ms['allocated'])} (peak {_gib(ms['allocated_peak'])}), active {_gib(ms['active'])}, "
                        f"reserved {_gib(ms['reserved'])} (peak {_gib(ms['reserved_peak'])}{grow}), inactive split {_gib(ms['inactive_split'])}, "
                        f"alloc retries {ms['retries']}")
        if st.min_avail_mib is not None:
            bits.append(f"MemAvailable min {st.min_avail_mib / 1024:.2f} GiB since the last line")
            st.min_avail_mib = None
        if getattr(rt, "redirect_n", 0) or getattr(rt, "pad_n", 0) or st.pool_mode != "compute":
            bits.append(f"pool={st.pool_mode}: last region {rt.redirect_n} collective outputs / {rt.redirect_bytes / 2**20:.0f} MiB + {rt.pad_n} padded "
                        f"scatter inputs / {rt.pad_bytes / 2**20:.0f} MiB from the compute pool")
        if st.empty_cache:
            bits.append(f"empty_cache after {st.counters['empty_cache']} regions")
        if st.pool_snaps_left > 0 and hasattr(rt, "pool_snapshot"):
            st.pool_snaps_left -= 1
            t0 = time.perf_counter()
            ps = rt.pool_snapshot()
            if time.perf_counter() - t0 > 1.0:
                st.pool_snaps_left = 0                       # a snapshot walks every block: stop if it is expensive on this process
            if ps:
                bits.append("allocator pools reserved/allocated GiB: " + ", ".join(f"{k} stream {_gib(v[0])}/{_gib(v[1])}" for k, v in ps.items()))
    except Exception as e:  # noqa: BLE001
        bits.append(f"memory stats unavailable ({type(e).__name__})")
    return (" | " + "; ".join(bits)) if bits else ""


def _pool_problem(allr: list) -> str | None:
    """POOL=compute on CUDA, but on some rank the last self-test region served no collective output from the compute stream's pool: the
    TorchFunctionMode never saw vLLM's all-gather allocate (NOTES 12.3), so the memory saving of this revision is not delivered. Refused
    (fail closed, on every rank) instead of serving into the out-of-memory guard; POOL=comm accepts revision 3's placement explicitly."""
    st = _STATE
    if st.pool_mode != "compute" or getattr(st.rt, "kind", "") != "cuda":
        return None
    none = [r["rank"] for r in allr if not (r.get("redirect") and r["redirect"][0] > 0)]
    if not none:
        return None
    return (f"{ENV_POOL}=compute, but the last self-test region served no collective output from the compute stream's pool on rank(s) {none}: "
            "the gathers still allocate on the communication stream, which is the memory problem this setting exists to remove (results would be "
            f"right). Set {ENV_POOL}=comm to accept that placement, or unset {ENV}")


def _pool_suffix(allr: list) -> str:
    """The pool-discipline part of the ACTIVE line."""
    st = _STATE
    red = [r.get("redirect") for r in allr if r.get("redirect")]
    if not red:
        return ""
    n, b, pn, pb = red[0]
    txt = f" Pool discipline: {st.pool_mode}"
    if st.pool_mode == "compute":
        txt += (f" ({n} collective outputs / {b / 2**20:.0f} MiB and {pn} padded scatter inputs / {pb / 2**20:.0f} MiB of the last region from the "
                "compute stream's pool)")
    if st.empty_cache:
        txt += "; empty_cache at the end of every overlapped region"
    return txt + "."


def _nbytes(x) -> int:
    """Bytes of the tensors in x (a tensor, or a list / tuple / dict of them)."""
    if isinstance(x, torch.Tensor):
        return int(x.numel() * x.element_size())
    if isinstance(x, dict):
        return sum(_nbytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_nbytes(v) for v in x)
    return 0


def _meminfo() -> dict:
    """{field: kB} of the /proc/meminfo lines that say where host memory went (MEMINFO_FIELDS); {} when unreadable."""
    out: dict = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                if k in MEMINFO_FIELDS:
                    out[k] = int(rest.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return out


def _descendants() -> tuple[int, int, str]:
    """(number, summed RSS in kB, name of the largest) of the descendant processes of this process -- JIT compilers (nvcc, cicc, ptxas, ...) and
    other helpers a first-time kernel launch starts count against the node's MemAvailable but not against the allocator; (0, 0, "") when there
    are none or /proc is unreadable. One pass over /proc/*/stat."""
    try:
        me = os.getpid()
        page_kb = max(1, os.sysconf("SC_PAGE_SIZE") // 1024)
        info: dict = {}
        for ent in os.listdir("/proc"):
            if not ent.isdigit():
                continue
            try:
                with open(f"/proc/{ent}/stat") as fh:
                    raw = fh.read()
                i, j = raw.find("("), raw.rfind(")")
                rest = raw[j + 2:].split()
                info[int(ent)] = (int(rest[1]), int(rest[21]) * page_kb, raw[i + 1:j])         # ppid, RSS (kB), name
            except (OSError, ValueError, IndexError):
                continue
        mine, grew = {me}, True
        while grew:
            grew = False
            for pid, (ppid, _, _) in info.items():
                if pid not in mine and ppid in mine:
                    mine.add(pid)
                    grew = True
        kids = [info[p] for p in mine if p != me and p in info]
        if not kids:
            return (0, 0, "")
        return (len(kids), sum(k[1] for k in kids), max(kids, key=lambda k: k[1])[2])
    except Exception:  # noqa: BLE001 - a diagnostic
        return (0, 0, "")


class _SelfTestMem:
    """Self-test only: where did memory go while the boot self-test ran? MemAvailable and the /proc/meminfo lines that move (MEMINFO_FIELDS) plus
    the allocator's allocated / reserved bytes are sampled at the phase boundaries and every SELFTEST_MEM_PERIOD_S by a daemon thread, the worker's
    descendant processes every SELFTEST_SCAN_EVERY samples; stop() logs ONE line (INFO, every rank) with the start / lowest / end values and the
    phase the lowest MemAvailable fell in. Diagnostics: a failing read is ignored, nothing here touches a tensor."""

    def __init__(self, rt, rank: int, period: float = SELFTEST_MEM_PERIOD_S) -> None:
        self.rt, self.rank, self.period = rt, rank, period
        self.phase_name = "start"
        self.t0 = time.perf_counter()
        self.start_info: dict = {}
        self.low_info: dict = {}
        self.end_info: dict = {}
        self.low_phase = "start"
        self.low_t = 0.0                                    # seconds from the start / wall clock of the lowest MemAvailable sample (to line it up with an external memwatch)
        self.low_clock = time.strftime("%H:%M:%S")
        self.phase_t: list[tuple[str, float]] = []
        self.alloc0: tuple | None = None
        self.alloc_end: tuple | None = None
        self.alloc_max = 0
        self.alloc_max_phase = "start"
        self.retained = 0
        self.retained_max = 0
        self.samples = 0
        self.scans = 0
        self.kids: tuple = (0, 0, "")                       # the sample with the most child-process RSS: (processes, kB, name of the largest)
        self.kids_phase = "start"
        self.phases: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.text: str | None = None

    def _basic(self):
        fn = getattr(self.rt, "mem_basic", None)
        try:
            return fn() if fn is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _sample(self) -> tuple:
        mi, mb = _meminfo(), self._basic()
        with self._lock:
            self.samples += 1
            a = mi.get("MemAvailable")
            if a is not None and ("MemAvailable" not in self.low_info or a < self.low_info["MemAvailable"]):
                self.low_info, self.low_phase = mi, self.phase_name
                self.low_t, self.low_clock = time.perf_counter() - self.t0, time.strftime("%H:%M:%S")
            if mb is not None and mb[0] > self.alloc_max:
                self.alloc_max, self.alloc_max_phase = mb[0], self.phase_name
        return mi, mb

    def phase(self, name: str) -> None:
        self.phase_name = name
        self.phases.append(name)
        self.phase_t.append((name, time.perf_counter() - self.t0))
        try:
            self._sample()
        except Exception:  # noqa: BLE001 - a diagnostic never breaks the test
            pass

    def retain(self, nbytes: int) -> None:
        """The test now holds `nbytes` of reference (the one region result it keeps for the next comparison)."""
        self.retained = int(nbytes)
        self.retained_max = max(self.retained_max, self.retained)

    def start(self) -> None:
        try:
            mi, mb = self._sample()
            self.start_info, self.alloc0 = mi, mb
            self.low_info, self.low_phase = dict(mi), "start"
            self.low_t, self.low_clock = 0.0, time.strftime("%H:%M:%S")
            self._thread = threading.Thread(target=self._loop, daemon=True, name="p028-selftest-mem")
            self._thread.start()
        except Exception as e:  # noqa: BLE001 - e.g. no thread available: the record then has the phase boundaries only
            self._thread = None
            logger.debug("%s self-test memory sampler not started: %s", TAG, e)

    def _scan(self) -> None:
        n, kb, name = _descendants()
        with self._lock:
            self.scans += 1
            if kb > self.kids[1] or (kb == self.kids[1] and n > self.kids[0]):
                self.kids, self.kids_phase = (n, kb, name), self.phase_name

    def _loop(self) -> None:
        k = 0
        while not self._stop.wait(self.period):
            try:
                self._sample()
                k += 1
                if k % SELFTEST_SCAN_EVERY == 0:
                    self._scan()
            except Exception:  # noqa: BLE001 - a diagnostic never takes the worker down
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
        try:
            self.phase_name = "end"
            self.end_info, self.alloc_end = self._sample()
            self.text = self.describe()
            logger.info("%s self-test memory (rank %d, %.1f s): %s", TAG, self.rank, time.perf_counter() - self.t0, self.text)
        except Exception as e:  # noqa: BLE001
            logger.debug("%s self-test memory record failed: %s", TAG, e)

    def describe(self) -> str:
        g = lambda kb: kb / 2**20  # noqa: E731
        bits = []
        s0, lo, e1 = self.start_info.get("MemAvailable"), self.low_info.get("MemAvailable"), self.end_info.get("MemAvailable")
        if s0 is not None and lo is not None and e1 is not None:
            bits.append(f"MemAvailable start {g(s0):.2f} GiB, lowest {g(lo):.2f} GiB ({g(lo - s0):+.2f}, during '{self.low_phase}'), end {g(e1):.2f} GiB")
            bits.append(f"lowest at t={self.low_t:.1f} s ({self.low_clock}); phases (s from the start): "
                        + ", ".join(f"{n} {t:.1f}" for n, t in self.phase_t))
            moved = ", ".join(f"{k} {g(self.low_info.get(k, 0) - self.start_info.get(k, 0)):+.2f}" for k in MEMINFO_FIELDS if k != "MemAvailable"
                              and k in self.low_info and k in self.start_info)
            if moved:
                bits.append(f"/proc/meminfo start -> lowest (GiB): {moved}")
        if self.alloc0 and self.alloc_end:
            a0, r0 = self.alloc0
            bits.append(f"allocator allocated start {a0 / 2**30:.2f} GiB, highest {self.alloc_max / 2**30:.2f} GiB ({(self.alloc_max - a0) / 2**30:+.2f}, "
                        f"during '{self.alloc_max_phase}'), end {self.alloc_end[0] / 2**30:.2f}; reserved start {r0 / 2**30:.2f}, end {self.alloc_end[1] / 2**30:.2f} "
                        f"({(self.alloc_end[1] - r0) / 2**30:+.2f})")
        if self.scans:
            n, kb, name = self.kids
            bits.append(f"child processes of the worker: peak {n} / {g(kb):.2f} GiB RSS (largest '{name}', during '{self.kids_phase}'; {self.scans} scans)"
                        if n else f"no child processes seen ({self.scans} scans)")
        bits.append(f"held by the test: ONE region result at a time (largest {self.retained_max / 2**20:.0f} MiB), comparison temporaries <= "
                    f"{SELFTEST_TEMP_BYTES >> 20} MiB per call; {self.samples} samples")
        return "; ".join(bits)


# --- public hook (called from the patched DeepseekV4Model.forward) -----------------------------------------------------------
def open_region(model, handle, hidden_states, positions, input_ids) -> Region | None:
    """Called after sp.begin_forward returned an SP handle. Returns the Region that replaces shard_input + _run_layers +
    exit_region for this forward, or None (the 022 path runs unchanged: no halves for this step)."""
    st = _STATE
    world = handle.world
    if not st.preflight_done:
        _preflight(model, world, positions.device)
    pend = _consume_pending(positions)
    if pend is None:
        st.counters["skip_no_pending"] += 1
        return None
    if st.idx_mode == "exclusive":
        # the full batch's indexer metadata is sharded (vLLM's rule, or the depth gate): the plain 022 region exchanges on the one
        # stream it uses; the two-half region never runs next to a sharded indexer. Undecidable (no indexer metadata) -> plain as well.
        sharded = forward_is_sharded(model)
        if sharded is not False:
            st.counters["skip_sharded" if sharded else "skip_sharded_unknown"] += 1
            return None
    st.forwards += 1
    if st.rt is None:
        st.rt = _make_runtime(positions.device)
    st.rt.pool = st.pool_mode == "compute"
    on = True
    if os.environ.get(ENV_CTL):
        on, _ = _sync_ctl(world)
    if on and not st.verified:
        _first_use_checks(model, pend, world)
    return Region(model, handle, pend, "overlap" if on else "plain", hidden_states, positions, input_ids)


def _preflight(model, world, device=None) -> None:
    """Structural checks + fingerprint exchanged over the TP group before anything else (fail closed on every rank, bounded). With
    VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER=comm2 the dedicated indexer communicator is created, warmed up and installed here, after the
    vote and before the first region runs (collective: every rank reaches it at the same forward)."""
    st = _STATE
    try:
        errs = overlap_envelope_errors(model, world)
        env_err = "; ".join(errs) if errs else None
        fp = _fingerprint(model, world.size)
    except Exception as e:  # noqa: BLE001
        env_err, fp = f"{type(e).__name__}: {e}", "error"
    allp = _exchange(world, {"rank": world.rank, "fp": fp, "err": env_err}, "preflight")
    errs = [(p["rank"], p["err"]) for p in allp if p["err"]]
    if errs:
        raise RuntimeError(f"{TAG} {ENV}=1 refused on rank(s) {[r for r, _ in errs]}: {errs[0][1]}; unset {ENV} to serve without it")
    if len({p["fp"] for p in allp}) != 1:
        raise RuntimeError(f"{TAG} rank fingerprint mismatch across the TP group: {[p['fp'] for p in allp]}: a rank has a different "
                           f"patch-028 module / settings; apply the same recipe on every node")
    st.fingerprint = fp
    st.idx_mode, st.idx_min_pos = indexer_settings()
    st.pool_mode, st.empty_cache = parse_pool(os.environ.get(ENV_POOL)), parse_empty_cache()
    if st.idx_mode == "comm2":
        setup_indexer_exchange(model, world, device)
    st.preflight_done = True
    logger.info("%s preflight OK on all %d ranks (fingerprint %s, min_t %d, halves >= %d rows, cut alignment %d, control file %s, "
                "indexer row sharding: %s)", TAG, len(allp), fp, st.min_t, MIN_HALF_ROWS, CUT_ALIGN, os.environ.get(ENV_CTL) or "none",
                {"off": "refused (INDEXER unset)", "comm2": "own NCCL communicator for the top-k exchange",
                 "exclusive": f"sharded forwards run plain (depth gate {st.idx_min_pos} keys per row)"}[st.idx_mode])


def _first_use_checks(model, pend: Pending, world) -> None:
    """Runner-side requirements, voted before the first overlapped forward (no rank starts the pipeline unless all can)."""
    err = None
    try:
        ms = pend.runner.model_state
        for need in ("_build_attn_metadata", "_copy_attn_groups", "_replay_start"):
            if not hasattr(ms, need):
                raise RuntimeError(f"model state {type(ms).__name__} has no {need} (not DeepseekV41ModelState?)")
        free = _mem_free()
        guard = parse_mem_guard(os.environ.get(ENV_MEM)) * 2**30
        if free >= 0 and _STATE.scratch.groups is None and free < guard:
            raise RuntimeError(f"only {free / 2**30:.2f} GiB of GPU memory free, {guard / 2**30:.1f} GiB required for the per-half "
                               "attention-metadata builders")
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:300]}"
    votes = _exchange(world, {"rank": world.rank, "err": err}, "first-use checks")
    bad = [v for v in votes if v["err"]]
    if bad:
        raise RuntimeError(f"{TAG} {ENV}=1: first-use checks failed on rank(s) {[v['rank'] for v in bad]}: {bad[0]['err']}; "
                           f"unset {ENV} to serve without it")


def stats() -> dict:
    st = _STATE
    return {"enabled": st.enabled, "verified": st.verified, "forwards": st.forwards, "counters": dict(st.counters),
            "report": st.report, "min_t": st.min_t, "ctl_on": st.ctl_on,
            "indexer": {"mode": st.idx_mode, "min_pos": st.idx_min_pos, "comm": st.idx_comm is not None, "report": st.idx_report}}


# [rev3-idx-begin] -------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# 8. Indexer row sharding (revision 3): VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER = comm2 | exclusive
#    The only code of this module that creates a communicator or replaces a vLLM function. Everything is installed at run time
#    (no file of vLLM is edited), guarded by source hashes of the replaced / mirrored functions, voted on over gloo.
# ---------------------------------------------------------------------------------------------------------------------
_SAI_MODULE = "vllm.model_executor.layers.sparse_attn_indexer"       # PrefillRowShard (exchange_topk)
_IDX_MODULE = "vllm.v1.attention.backends.mla.indexer"               # balanced_prefill_row_shard (the sharding decision)
# sha256 of dedent(source).strip() of the vLLM functions this section replaces / mirrors (the installer pins the whole files; these
# stop a tree whose function differs from the audited one even if a pin was bypassed)
EXCHANGE_SRC_SHA = "13ed5796417a247a6caf8fde5fca3f0f52982b497e2f819b0fe3e03126d50308"        # PrefillRowShard.exchange_topk
BALANCED_SRC_SHA = "a79a36eb2aebe976a53dd9eb0e9e7895eacd09261413a95e72db8124288452e4"        # balanced_prefill_row_shard

_COMM_FACTORY: Callable[[Any], Any] | None = None     # tests only: world -> an object with PyNcclCommunicator's all_gather / all_gatherv


def _src_sha(fn) -> str:
    import inspect
    import textwrap

    return hashlib.sha256(textwrap.dedent(inspect.getsource(fn)).strip().encode()).hexdigest()


def _sai_class():
    import importlib

    return importlib.import_module(_SAI_MODULE).PrefillRowShard


def _idx_module():
    import importlib

    return importlib.import_module(_IDX_MODULE)


def _mem_available_mib() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return -1.0


# --- the depth gate on vLLM's sharding decision, and the exclusion of sharded halves -------------------------------------------
def _mean_context_tokens(seq_lens_cpu, query_lens_cpu, compress_ratio: int) -> float:
    """Mean number of keys (in tokens) a prefill row of the step attends to: what balanced_prefill_row_shard's cost sums (compressed keys
    per row), times the compression ratio, divided by the rows. The sharding saving per row grows with it, the exchange cost per row does
    not."""
    ql = query_lens_cpu.to(torch.int64)
    rows = int(ql.sum())
    if rows <= 0:
        return 0.0
    first_key = torch.repeat_interleave(seq_lens_cpu.to(torch.int64) - ql + 1, ql)
    row_in_request = torch.arange(rows) - torch.repeat_interleave(torch.cumsum(ql, 0) - ql, ql)
    total = int(((first_key + row_in_request) // compress_ratio).sum())
    return total * compress_ratio / rows


def _make_gated(orig, mod, min_pos: int):
    def balanced_prefill_row_shard(seq_lens_cpu, query_lens_cpu, compress_ratio, tp_size, min_rows):
        if getattr(mod, "_p028_suppress", 0) > 0:
            return None                                   # inside a half's metadata build (exclusive mode): halves are never sharded
        sizes = orig(seq_lens_cpu, query_lens_cpu, compress_ratio, tp_size, min_rows)
        if sizes is None or min_pos <= 0:
            return sizes
        if _mean_context_tokens(seq_lens_cpu, query_lens_cpu, compress_ratio) < min_pos:
            return None                                   # shallow step: the replicated indexer is cheaper than the exchange
        return sizes

    balanced_prefill_row_shard._p028_gate = min_pos
    return balanced_prefill_row_shard


def install_shard_gate(min_pos: int, mod=None) -> None:
    """Replace vllm...mla.indexer.balanced_prefill_row_shard by the gated wrapper (idempotent; the builder looks the function up in the
    module namespace at every build)."""
    mod = mod if mod is not None else _idx_module()
    orig = getattr(mod, "_p028_orig_balanced", None)
    if orig is None:
        orig = mod.balanced_prefill_row_shard
        got = _src_sha(orig)
        if got != BALANCED_SRC_SHA:
            raise RuntimeError(f"{TAG} balanced_prefill_row_shard source sha256 {got[:16]}.. is not the audited {BALANCED_SRC_SHA[:16]}..: "
                               "the sharding gate was written against another vLLM build; refusing")
        mod._p028_orig_balanced = orig
    if not hasattr(mod, "_p028_suppress"):
        mod._p028_suppress = 0
    mod.balanced_prefill_row_shard = _make_gated(orig, mod, int(min_pos))


def uninstall_shard_gate(mod=None) -> None:
    mod = mod if mod is not None else sys.modules.get(_IDX_MODULE)
    orig = getattr(mod, "_p028_orig_balanced", None) if mod is not None else None
    if orig is not None:
        mod.balanced_prefill_row_shard = orig
        del mod._p028_orig_balanced
        mod._p028_suppress = 0


def gate_installed() -> bool:
    try:
        return getattr(_idx_module().balanced_prefill_row_shard, "_p028_gate", None) is not None
    except Exception:  # noqa: BLE001
        return False


@contextlib.contextmanager
def no_row_shard():
    """While active, the gated balanced_prefill_row_shard returns None (the metadata built inside is never sharded)."""
    mod = _idx_module()
    mod._p028_suppress = getattr(mod, "_p028_suppress", 0) + 1
    try:
        yield
    finally:
        mod._p028_suppress -= 1


def _install_gate_from_env() -> None:
    """Import time (model.py imports this module under the gate): the gate must be in place before the first step builds its metadata."""
    mode, min_pos = indexer_settings()
    if mode == "off" or (mode == "comm2" and min_pos <= 0):
        return
    install_shard_gate(min_pos)


def forward_is_sharded(model) -> bool | None:
    """Whether the FULL batch of the current forward has sharded indexer metadata (any region indexer): True / False, None when it
    cannot be decided (no forward context, no indexer entry). Reads CPU-built metadata only: the same answer on every rank."""
    try:
        from vllm.forward_context import get_forward_context

        am = get_forward_context().attn_metadata
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(am, dict):
        return None
    seen = False
    for ly in sp.region_layers(model):
        ix = getattr(ly.attn, "indexer", None)
        if ix is None:
            continue
        md = am.get(ix.k_cache.prefix)
        if md is None:
            return None
        seen = True
        pre = getattr(md, "prefill", None)
        if pre is not None and getattr(pre, "row_shard_sizes", None) is not None:
            return True
    return False if seen else None


# --- the dedicated communicator of the top-k exchange -------------------------------------------------------------------------
class IndexerComm:
    """A second NCCL communicator over the TP group's ranks, used ONLY for PrefillRowShard.exchange_topk and ONLY from the compute
    stream. all_gatherv is GroupCoordinator.all_gatherv(dim=0) / CudaCommunicator.all_gatherv: all sizes equal -> all_gather, else
    the grouped broadcast of PyNcclCommunicator.all_gatherv; the output is allocated here, on the current stream."""

    kind = "p028-idx"

    def __init__(self, comm, rank: int, world_size: int) -> None:
        self.comm, self.rank, self.world_size = comm, int(rank), int(world_size)
        self.calls = 0

    def all_gatherv(self, local: torch.Tensor, sizes) -> torch.Tensor:
        sizes = [int(s) for s in sizes]
        if len(sizes) != self.world_size or int(local.shape[0]) != sizes[self.rank]:
            raise RuntimeError(f"{TAG} indexer exchange: this rank's block has {int(local.shape[0])} rows, sizes {sizes} "
                               f"(rank {self.rank} of {self.world_size})")
        out = torch.empty((sum(sizes),) + tuple(local.shape[1:]), dtype=local.dtype, device=local.device)
        if all(s == sizes[0] for s in sizes):
            self.comm.all_gather(out, local)
        else:
            self.comm.all_gatherv(out, local, sizes)
        self.calls += 1
        return out


def _p028_exchange_topk(self, topk_buffer, width):
    """PrefillRowShard.exchange_topk on the dedicated communicator: the same slice, the same destination rows, the same bytes as
    vLLM's version (which gathers with the TP group's all_gatherv, i.e. on the TP communicator)."""
    st = _STATE
    ic = st.idx_comm
    if ic is None:                                            # not active: vLLM's own function
        return type(self)._p028_orig_exchange_topk(self, topk_buffer, width)
    rt = st.rt
    if rt is not None and hasattr(rt, "check_compute"):
        rt.check_compute("PrefillRowShard.exchange_topk")
    # The local input must outlive the async PyNCCL gather's enqueued copy: it is a stream-ordered allocation on the stream the
    # gather runs on (the current one), alive until this function returns.
    local = topk_buffer[self.start:self.stop, :width].contiguous()
    end = self.num_decode_tokens + sum(self.sizes)
    topk_buffer[self.num_decode_tokens:end, :width] = ic.all_gatherv(local, self.sizes)
    st.counters["idx_exchange"] += 1
    if not st.idx_first_logged:                               # once per process: proof in the log that the route is the one in use
        st.idx_first_logged = True
        logger.info("%s first indexer exchange on the second communicator: per-rank rows %s (this rank %d), width %d, %s", TAG,
                    [int(x) for x in self.sizes], int(local.shape[0]), int(width), str(local.dtype))
    if rt is not None and hasattr(rt, "note_x"):
        rt.note_x(("ixg", tuple(int(s) for s in self.sizes), int(width), str(local.dtype)))


def install_exchange(ic) -> None:
    cls = _sai_class()
    orig = cls.__dict__.get("_p028_orig_exchange_topk")
    if orig is None:
        orig = cls.__dict__["exchange_topk"]
        got = _src_sha(orig)
        if got != EXCHANGE_SRC_SHA:
            raise RuntimeError(f"{TAG} PrefillRowShard.exchange_topk source sha256 {got[:16]}.. is not the audited {EXCHANGE_SRC_SHA[:16]}..: "
                               "the dedicated-communicator route was written against another vLLM build; refusing")
        cls._p028_orig_exchange_topk = orig
    cls.exchange_topk = _p028_exchange_topk
    _STATE.idx_comm = ic


def uninstall_exchange() -> None:
    mod = sys.modules.get(_SAI_MODULE)
    cls = getattr(mod, "PrefillRowShard", None) if mod is not None else None
    orig = cls.__dict__.get("_p028_orig_exchange_topk") if cls is not None else None
    if orig is not None:
        cls.exchange_topk = orig
        delattr(cls, "_p028_orig_exchange_topk")
    _STATE.idx_comm = None


def _make_indexer_comm(world):
    """A second NCCL communicator over the TP group (collective, like the one CudaCommunicator builds for the group itself)."""
    if _COMM_FACTORY is not None:
        return _COMM_FACTORY(world)
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    grp = world.group
    comm = PyNcclCommunicator(group=grp.cpu_group, device=grp.device)
    if getattr(comm, "disabled", True) or not getattr(comm, "available", False):
        raise RuntimeError("the new PyNcclCommunicator is disabled (no NCCL library, VLLM_DISABLE_PYNCCL or a world of one)")
    return comm


def indexer_envelope_errors(model, world, mode: str, sharding: bool, shard_min, max_tokens) -> list[str]:
    """This rank's objections to the indexer setting (the preflight votes on them: a refusal on one rank refuses on every rank)."""
    errs: list[str] = []
    if mode == "off":
        if sharding:
            errs.append(f"indexer_tp_shard_min_rows={shard_min} <= max_num_batched_tokens={max_tokens}: the indexer would issue an "
                        f"all-gather (exchange_topk) on the compute stream; set {ENV_INDEXER}=comm2 (its own communicator) or =exclusive "
                        "(sharded forwards run without the overlap), or raise indexer_tp_shard_min_rows")
        return errs
    if not sharding:
        errs.append(f"{ENV_INDEXER}={mode} but indexer row sharding is not configured (indexer_tp_shard_min_rows={shard_min}, "
                    f"max_num_batched_tokens={max_tokens}): there is no exchange to route; unset {ENV_INDEXER} or set "
                    "attention_config.indexer_tp_shard_min_rows <= max_num_batched_tokens")
    try:
        cls = _sai_class()
        got = _src_sha(cls.__dict__.get("_p028_orig_exchange_topk") or cls.__dict__["exchange_topk"])
        if got != EXCHANGE_SRC_SHA:
            errs.append(f"PrefillRowShard.exchange_topk source sha256 {got[:16]}.. is not the audited {EXCHANGE_SRC_SHA[:16]}..")
    except Exception as e:  # noqa: BLE001
        errs.append(f"cannot audit PrefillRowShard.exchange_topk ({type(e).__name__}: {str(e)[:120]})")
    if mode == "comm2":
        grp = getattr(world, "group", None)
        pn = getattr(getattr(grp, "device_communicator", None), "pynccl_comm", None)
        if pn is None or getattr(pn, "disabled", True) or not getattr(pn, "available", True):
            errs.append("the TP group has no usable PyNccl communicator to clone (VLLM_DISABLE_PYNCCL? a single rank?)")
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "").strip() == "1":
            errs.append("CUDA_DEVICE_MAX_CONNECTIONS=1 runs the kernels of all streams in launch order: two communicators on two streams "
                        "need concurrent hardware queues")
    elif not gate_installed():
        errs.append("the sharding gate (balanced_prefill_row_shard wrapper) is not installed (import-time install failed?)")
    return errs


def _pattern(rank: int, rows: int, width: int, dtype, device) -> torch.Tensor:
    """Distinct, rank- and position-dependent top-k-shaped data (fits int32)."""
    i = torch.arange(rows, dtype=torch.int64, device=device).unsqueeze(1)
    j = torch.arange(width, dtype=torch.int64, device=device).unsqueeze(0)
    return ((rank + 1) * 1000003 + i * 131 + j).to(dtype)


def _warmup_sizes(W: int, max_rows: int) -> list[list[int]]:
    per = max(max_rows // W, 1)
    big = max(max_rows - 8 * (W - 1), 1)
    return [[1] * W, [64] * W, [per] * W, [1 + (r * 37) % 61 for r in range(W)], [big] + [8] * (W - 1), [8] * (W - 1) + [big],
            [2 * r + 1 for r in range(W)]]


def _warm_and_check(ic: IndexerComm, rt, world, topk) -> tuple[int, list]:
    """Explicit gathers over the new communicator before the schedule ever uses it. NCCL connects the algorithm a collective needs at
    its first use (host handshake + device allocations); that must not happen in the middle of the two-stream schedule. The sizes cover
    tiny / medium / maximal messages, equal sizes (all_gather) and uneven ones (grouped broadcast). Every gather is issued on every
    rank whatever the verdict (a rank that found a mismatch must not skip the collectives its peers still run); the bytes are checked
    after one bounded wait. Returns (gathers, the size lists whose bytes were wrong)."""
    W, rank = world.size, world.rank
    width, dtype, dev = int(topk.shape[1]), topk.dtype, topk.device
    max_rows = int(topk.shape[0])
    plan = _warmup_sizes(W, max_rows)
    outs = []
    for sizes in plan:
        outs.append(ic.all_gatherv(_pattern(rank, sizes[rank], width, dtype, dev), sizes))
    rt.bounded_sync(rt.compute, SYNC_TIMEOUT_S, "indexer communicator warm-up")
    bad = []
    for sizes, out in zip(plan, outs):
        want = torch.cat([_pattern(r, sizes[r], width, dtype, dev) for r in range(W)], dim=0)
        if out.shape != want.shape or not torch.equal(out, want):
            bad.append(sizes)
    return len(plan), bad


def _check_installed(rt, world, topk) -> str | None:
    """The installed PrefillRowShard.exchange_topk (vLLM's real class, our function) on a synthetic shard: windows, offsets, widths.
    One gather on every rank; returns the problem text or None."""
    W, rank = world.size, world.rank
    width, dtype, dev = int(topk.shape[1]), topk.dtype, topk.device
    sizes = [3 + (r * 5) % 7 for r in range(W)]
    ndec = 2
    rs = _sai_class()(ndec, sizes, rank)
    total = ndec + sum(sizes)
    want = _pattern(0, total, width, dtype, dev) + 77
    buf = torch.full((total + 3, width), -1, dtype=dtype, device=dev)
    buf[rs.start:rs.stop] = want[rs.start:rs.stop]
    rs.exchange_topk(buf, width)
    rt.bounded_sync(rt.compute, SYNC_TIMEOUT_S, "indexer communicator installed-method check")
    if not torch.equal(buf[ndec:total], want[ndec:total]) or int(buf[total:].min()) != -1:
        return "the installed PrefillRowShard.exchange_topk did not reconstruct the global rows"
    return None


def setup_indexer_exchange(model, world, device=None) -> None:
    """INDEXER=comm2, collective (every rank calls it at the same forward): create the communicator, warm it up, verify the bytes, route
    PrefillRowShard.exchange_topk through it. Raises a [patch-028] error on EVERY rank on any failure."""
    st = _STATE
    t0 = time.time()
    if st.rt is None:
        st.rt = _make_runtime(device if device is not None else world.group.device)
    rt = st.rt
    rt.refresh()
    free0, avail0 = _mem_free(), _mem_available_mib()
    rec: dict = {"rank": world.rank, "err": None}
    ic = None
    try:
        ic = IndexerComm(_make_indexer_comm(world), world.rank, world.size)
    except Exception as e:  # noqa: BLE001 - every rank must still reach the vote
        rec["err"] = f"create: {type(e).__name__}: {str(e)[:300]}"
    votes = _exchange(world, rec, "indexer communicator creation")
    bad = [v for v in votes if v["err"]]
    if bad:
        raise RuntimeError(f"{TAG} {ENV_INDEXER}=comm2: the dedicated NCCL communicator of the top-k exchange could not be created on "
                           f"rank(s) {[v['rank'] for v in bad]}: {bad[0]['err']}; refusing - unset {ENV_INDEXER} (and "
                           "attention_config.indexer_tp_shard_min_rows) to serve without it")
    rec2: dict = {"rank": world.rank, "err": None, "n": 0}
    st.idx_first_logged = True                                # the synthetic exchange of the installed-method check below is not the first model exchange
    try:
        rec2["n"], wrong = _warm_and_check(ic, rt, world, model.topk_indices_buffer)
        install_exchange(ic)                                  # also on a rank that found wrong bytes: every rank runs the same collectives
        problems = ([f"gathered bytes differ from the expected rank-major layout for sizes {wrong[:2]}"] if wrong else [])
        chk = _check_installed(rt, world, model.topk_indices_buffer)
        rec2["err"] = "; ".join(problems + ([chk] if chk else [])) or None
    except Exception as e:  # noqa: BLE001
        rec2["err"] = f"{type(e).__name__}: {str(e)[:300]}"
    try:
        votes2 = _exchange(world, rec2, "indexer communicator verification")
    except BaseException:                                    # a missing rank: nothing stays installed behind the raise
        uninstall_exchange()
        raise
    bad = [v for v in votes2 if v["err"]]
    if bad:
        uninstall_exchange()
        raise RuntimeError(f"{TAG} {ENV_INDEXER}=comm2: the dedicated communicator failed its verification on rank(s) "
                           f"{[v['rank'] for v in bad]}: {bad[0]['err']}; refusing - unset {ENV_INDEXER} to serve without it")
    free1, avail1 = _mem_free(), _mem_available_mib()
    st.counters["idx_exchange"] = 0                          # the installed-method check above is not a model exchange
    st.idx_first_logged = False
    st.idx_report = {"seconds": round(time.time() - t0, 2), "gathers": max(v["n"] for v in votes2) + 1,
                     "gpu_free_delta_mib": round((free1 - free0) / 2**20, 1) if free0 >= 0 and free1 >= 0 else None,
                     "mem_available_delta_mib": round(avail1 - avail0, 1) if avail0 >= 0 and avail1 >= 0 else None}
    logger.info("%s indexer exchange communicator ready (rank %d/%d): a second NCCL communicator over the TP group's CPU group, %d "
                "verification gathers (warm-up plan + installed-method check) OK on all %d ranks in %.1f s; GPU free %s MiB, MemAvailable %s MiB "
                "(this rank); "
                "PrefillRowShard.exchange_topk now runs on it, from the compute stream only (the TP communicator stays on the "
                "communication stream inside a region)", TAG, world.rank, world.size, st.idx_report["gathers"], len(votes2),
                st.idx_report["seconds"], st.idx_report["gpu_free_delta_mib"], st.idx_report["mem_available_delta_mib"])


def _indexer_suffix(rec0: dict) -> str:
    st = _STATE
    if st.idx_mode == "comm2":
        return (f" Indexer row sharding: comm2 (PrefillRowShard.exchange_topk on its own NCCL communicator, compute stream only; "
                f"{rec0.get('xlog_n')} exchanges in the last region).")
    if st.idx_mode == "exclusive":
        return f" Indexer row sharding: exclusive (sharded forwards run plain 022; depth gate {st.idx_min_pos} keys per row)."
    return ""


# [rev3-idx-end] ---------------------------------------------------------------------------------------------------------------


# --- test hooks -------------------------------------------------------------------------------------------------------
def _set_runtime_factory_for_test(factory: Callable[[Any], Any] | None) -> None:
    global _RUNTIME_FACTORY
    _RUNTIME_FACTORY = factory


def _set_halves_factory_for_test(factory: Callable[[Any, Any], Any] | None) -> None:
    global _HALVES_FACTORY
    _HALVES_FACTORY = factory


def _set_comm_factory_for_test(factory: Callable[[Any], Any] | None) -> None:
    global _COMM_FACTORY
    _COMM_FACTORY = factory


def _reset_for_test(enabled: bool | None = True, min_t: int = DEFAULT_MIN_T) -> None:
    global _STATE
    uninstall_exchange()                                   # a wrapper left by an earlier module instance of the same process
    uninstall_shard_gate()
    _STATE = _State()
    _STATE.enabled, _STATE.min_t = enabled, min_t
    if parse_enabled(os.environ.get(ENV)):                  # a fresh process has the gate installed by the import
        _install_gate_from_env()


if parse_enabled(os.environ.get(ENV)):                      # model.py imports this module under the gate
    _install_gate_from_env()
