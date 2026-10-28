# SPDX-License-Identifier: Apache-2.0
"""GB10 RoCE collectives for small TP messages (patch 018; opt-in VLLM_GB10_ROCE=1): b12x RoCEnante one-shot all-reduce /
all-gather, and (VLLM_GB10_ROCE_TWOSHOT_MAX_BYTES > 0) our two-shot all-reduce (vllm/_gb10_twoshot) in front of it.

WHY: on 8 DGX Sparks (one 100 GbE RoCE port each, no GPUDirect RDMA) NCCL costs ~130 us per decode all-reduce in a CUDA
graph (patch 012's one-shot over NCCL send/recv: ~88 us) and stalls ~0.5 ms at the first NCCL operation of every graph
launch. RoCEnante (flashinfer.experimental.b12x.comm.roce, shipped in the image) is a one-shot all-reduce / all-gather
over its own RC queue pairs: one fused kernel stages the input in pinned host memory, rings a C proxy thread that posts
the RDMA writes, waits for every peer's payload and reduces in rank order. Measured 2026-10-26 on this cluster (8 ranks,
CUDA graphs): all-reduce 23 / 61 / 100 us at 1 / 6 / 12 tokens (NCCL 81 / 130 / 131, patch 012 47 / 88 / -), slower
than NCCL from ~24 tokens; all-gather [6, 768] bf16 26 us (NCCL 72), [6, 3200] 51 us (107).

NUMERICS: all-reduce = fp32 sum in rank order 0..W-1, one rounding to the input dtype -- the same bits as patch 012 (test:
bit-exact vs patch 012 and the fp32 reference, identical on all 8 ranks). All-gather: plain copies.

SCOPE: the TP group only; all-reduce of bf16/fp16/fp32 contiguous tensors up to VLLM_GB10_ROCE_AR_MAX_BYTES (default
131072 = 12 tokens of DeepSeek-V4.1); all-gather of contiguous tensors whose per-rank shard is at most
VLLM_GB10_ROCE_AG_MAX_BYTES (default 524288), concatenated along dim 0, the last dim, or (by flattening the trailing
dims) any dim of a contiguous tensor. A limit of 0 leaves that collective on NCCL. Everything else stays on NCCL /
patch 012.

TWO-SHOT (VLLM_GB10_ROCE_TWOSHOT_MAX_BYTES, default 0 = off): one-shot sends a rank's whole message to all 7 peers and
is bandwidth-bound from ~2 tokens; the two-shot runtime sends each peer only its eighth, the Grace CPU proxy reduces
its own eighth (NEON, fp32 in rank order, one round-to-nearest-even: the same bits as RoCEnante / patch 012) and sends
it to everyone. Measured 2026-10-26 (8 ranks, CUDA graphs, next to the idle engine): 19 / 32 / 49 / 69 / 117 / 355 us
at 1 / 6 / 12 / 24 / 48 / 192 tokens (one-shot 23 / 61 / 99 / 175 / 329 / 1249, NCCL 73 / 125 / 124 / 148 / 184 / 359).
When on, every eligible all-reduce up to the limit takes the two-shot; all-gathers stay on RoCEnante.

FAIL-CLOSED: the constructor is collective and runs a warm-up that raises on any wrong element; every eager call and
every step (AsyncOutput.get_output, after the step's copy event) checks the runtime's health and raises if a wait timed
out or the proxy died (RoCEnante poisons itself instead of returning data).
"""

from __future__ import annotations

import contextlib
import os

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENV = "VLLM_GB10_ROCE"
_ENV_AR = "VLLM_GB10_ROCE_AR_MAX_BYTES"
_ENV_AG = "VLLM_GB10_ROCE_AG_MAX_BYTES"
_ENV_TS = "VLLM_GB10_ROCE_TWOSHOT_MAX_BYTES"
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_ACTIVE: "GB10Roce | None" = None


class GB10Roce:
    def __init__(self, cpu_group, device: torch.device, ar_max: int, ag_max: int, ts_max: int = 0):
        from flashinfer.experimental.b12x.comm import roce
        from flashinfer.experimental.b12x.comm.roce import _allgather_cute
        from flashinfer.experimental.b12x.comm.roce import _preparation as prep_mod
        from flashinfer.experimental.b12x.comm.roce._oneshot_cute import get_launcher

        if roce.API_VERSION != 1:
            raise RuntimeError(f"[patch-018] RoCEnante API_VERSION {roce.API_VERSION} != 1")
        self.device = device
        self.ar_max = ar_max
        self.ag_max = ag_max
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)
        self.rt = roce.AllReduce(exchange_group=cpu_group, device=device, max_size=max(ar_max, 16),
                                 max_gather_bytes=max(ag_max, 16))
        self.rt._prepare_resources(_DTYPES, padded_gather=True)
        self.prepared = prep_mod._PreparedRoce(
            self.rt,
            {d: get_launcher(*self.rt._launcher_key(d)) for d in _DTYPES},
            _allgather_cute.get_launcher(*self.rt._gather_launcher_key()),
        )
        self.ts_max = ts_max
        self.twoshot = None
        if ts_max > 0:
            from vllm._gb10_twoshot.gb10_twoshot import GB10TwoShotAllReduce

            self.twoshot = GB10TwoShotAllReduce(exchange_group=cpu_group, device=device, max_bytes=ts_max)
        self._warmup()

    # -- policy (depends only on dtype, shape and size, which all TP ranks share) --
    def should_ar(self, x: torch.Tensor) -> bool:
        if self.twoshot is not None and self.twoshot.should_allreduce(x):
            return True
        return self._oneshot_ar(x)

    def _oneshot_ar(self, x: torch.Tensor) -> bool:
        return (x.dtype in _DTYPES and x.is_contiguous() and 0 < x.numel() * x.element_size() <= self.ar_max
                and self.rt.should_allreduce(x))

    def should_ag(self, x: torch.Tensor, dim: int) -> bool:
        if not x.is_contiguous() or x.dim() == 0 or x.numel() == 0:
            return False
        return x.numel() * x.element_size() <= self.ag_max and self.rt.should_all_gather(self._ag_view(x, dim)[0], -1)

    # -- collectives --
    # The runtime checks its health on every eager call (before and after the launch); graph replays are covered by
    # check_health() once per step.
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.twoshot is not None and self.twoshot.should_allreduce(x):
            return self.twoshot.all_reduce(x)
        return self.rt._run_prepared_all_reduce(x, prepared=self.prepared)

    @staticmethod
    def _ag_view(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, int]:
        """2-D view [prod(shape[:dim]), prod(shape[dim:])] whose last-dim concatenation equals the dim-`dim` one."""
        if dim < 0:
            dim += x.dim()
        lead = 1
        for s in x.shape[:dim]:
            lead *= s
        return x.reshape(lead, -1), dim

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        v, d = self._ag_view(x, dim)
        out = self.rt._run_prepared_all_gather(v, prepared=self.prepared, dim=-1)
        shape = list(x.shape)
        shape[d] *= self.world_size
        return out.reshape(shape)

    @contextlib.contextmanager
    def capture(self):
        with self.rt.capture():
            if self.twoshot is None:
                yield
            else:
                with self.twoshot.capture():
                    yield

    def check_health(self) -> None:
        self.rt.check_health()
        if self.twoshot is not None:
            self.twoshot.check_health()

    def _warmup(self) -> None:
        # Collective: every rank contributes (rank + 1); the exact sum ws * (ws + 1) / 2 is representable in bf16.
        # Sizes stay within the runtime's capacity (at least 16 bytes even when a limit is 0).
        want = float(self.world_size * (self.world_size + 1) // 2)
        for n in sorted({8, min(6 * 5120, self.ar_max // 2), self.ar_max // 2} - {0}):
            x = torch.full((n,), float(self.rank + 1), dtype=torch.bfloat16, device=self.device)
            out = self.rt._run_prepared_all_reduce(x, prepared=self.prepared)
            torch.cuda.synchronize(self.device)
            if not bool(torch.all(out == want)):
                raise RuntimeError(f"[patch-018] RoCE all-reduce warm-up wrong on rank {self.rank} (n={n})")
        if self.twoshot is not None:
            self.twoshot.warmup()
            for n in sorted({8, 6 * 5120, 48 * 5120, self.ts_max // 2} - {0}):
                n = min(n, self.ts_max // 2)
                x = torch.full((n,), float(self.rank + 1), dtype=torch.bfloat16, device=self.device)
                out = self.twoshot.all_reduce(x)
                torch.cuda.synchronize(self.device)
                if not bool(torch.all(out == want)):
                    raise RuntimeError(f"[patch-018] two-shot all-reduce warm-up wrong on rank {self.rank} (n={n})")
        rows, inner = (6, 256) if self.ag_max >= 6 * 3 * 256 * 2 else (1, 2)  # the Engram shape, or 12 bytes
        x = torch.full((rows, 3, inner), float(self.rank), dtype=torch.bfloat16, device=self.device)
        out = self.all_gather(x, 1)
        torch.cuda.synchronize(self.device)
        ref = torch.arange(self.world_size, dtype=torch.bfloat16, device=self.device).repeat_interleave(3)
        if out.shape != (rows, 3 * self.world_size, inner) or not bool(torch.all(out[:, :, 0] == ref)):
            raise RuntimeError(f"[patch-018] RoCE all-gather warm-up wrong on rank {self.rank}")
        self.check_health()


def maybe_create(cpu_group, device: torch.device, unique_name: str | None):
    global _ACTIVE
    if os.environ.get(_ENV, "0") != "1":
        return None
    if cpu_group is None or dist.get_world_size(cpu_group) < 2:
        return None
    if not unique_name or unique_name.split(":")[0] != "tp":
        return None
    ar_max = int(os.environ.get(_ENV_AR, "131072"))
    ag_max = int(os.environ.get(_ENV_AG, "524288"))
    ts_max = int(os.environ.get(_ENV_TS, "0"))
    if min(ar_max, ag_max, ts_max) < 0 or ar_max % 16 or ag_max % 16 or ts_max % 16:
        raise ValueError(f"[patch-018] {_ENV_AR} / {_ENV_AG} / {_ENV_TS} must be non-negative multiples of 16 bytes")
    op = GB10Roce(cpu_group, device, ar_max, ag_max, ts_max)
    _ACTIVE = op
    logger.info(
        "[patch-018] RoCEnante ACTIVE for group %s (world %d, hcas %s, gid %d; all-reduce <= %d B, all-gather shard <= %d B)",
        unique_name, op.world_size, ",".join(op.rt.hca_names), op.rt.gid_index, ar_max, ag_max,
    )
    if op.twoshot is not None:
        st = op.twoshot.stats()
        logger.info(
            "[patch-018] two-shot all-reduce ACTIVE for group %s (all-reduce <= %d B; proxy cpu %s, piece %s B, prefetch %s B)",
            unique_name, ts_max, st.get("proxy_cpu"), st.get("piece_bytes"), st.get("prefetch_bytes"),
        )
    return op


def check_health() -> None:
    """Called once per step after the step's outputs are on the host; raises if a RoCE wait timed out."""
    if _ACTIVE is not None:
        _ACTIVE.check_health()
