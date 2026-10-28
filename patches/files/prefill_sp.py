# SPDX-License-Identifier: Apache-2.0
# [patch-022 prefill-sp] installed as vllm/_prefill_sp.py by 022-prefill-sp.sh (opt-in VLLM_GB10_PREFILL_SP=1).
"""Sequence-parallel mHC for eager prefill of DeepSeek-V4.1 on 8 x GB10 (TP=8, split experts, no EP). NUMERICS CHANGE
in the prefill reduction order only; every row is computed on exactly one rank and gathered by copy.

Ported from the GLM-5.3-Flash patch 025 (gb10_prefill_sp.py); written against models/deepseek_v41/nvidia/model.py as the
running image has it after patches 013-021 (sha256 238b8a74...). Used only by the patch-022 edits of that file.

WHY. At TP=8 every rank runs the whole mHC (mhc_post, the TF32 pre-norm GEMM, mhc_pre_big_fuse_with_norm) on all T rows
of a prefill chunk: ~18 % of the GPU time of a 4,096-token chunk, replicated eight times. The two row-parallel outputs of
each layer (attention wo_b, MoE) end in an all-reduce (~34 % of the chunk). Sequence-parallel (SP) turns each all-reduce
into reduce-scatter (every rank keeps the reduced rows it owns) + all-gather (the next block needs all rows): the same
bytes on the wire, and the mHC between them runs on T/8 rows.

WHAT (an eligible forward: eager, no CUDA graph, T >= VLLM_GB10_PREFILL_SP_MIN_T, default 1,024).
Region = the first _run_layers call of DeepseekV4Model.forward, layers start .. decoder_replay_start (0-19 with the
bounded decoder replay of the running build, 0-39 without it). Rank r owns rows [r*S, (r+1)*S), S = 8*ceil(T/64); the rows
past T are zero padding that no collective result ever reads. Per layer (sp_layer_forward replaces
DeepseekV4DecoderLayer.forward inside the region, mirroring it line by line for this config):

   attention mHC boundary (post of the previous FFN + pre of this attention)   local rows   (as upstream, on S rows)
   -> all-gather x [S,H] -> [T,H]
   -> attention (wo_b all-reduce suppressed)                                    -> TP partial [T,H]
   -> reduce-scatter                                                            -> [S,H]  (this rank's reduced rows)
   FFN mHC boundary                                                             local rows
   -> all-gather -> MoE (final all-reduce suppressed: moe_config.skip_final_all_reduce) -> partial [T,H]
   -> reduce-scatter                                                            -> [S,H]

The embedding (replicated by patch 015) is sliced locally at entry. At the end of the region (exit_region) the five carried
states are all-gathered to full T (x, the 4-stream residual, post/comb mixes, pre-mix) and everything after runs exactly as
before: the bounded replay (layer 20 write_kv for every row, layers 20-39 on each request's last 128 rows), the collapse,
the MTP/HC buffer, the final norm and the logits. Decode, short prefills, CUDA-graph captures/replays and micro-batched
steps never open a region.

DSV4.1 SPECIFICS
* DSpark aux hidden states (target layers 37/38/39): in the running build they come out of the bounded replay (layers
  >= 21), i.e. outside the region, as full [T,H] tensors. A tap inside the region (capture_previous_aux, e.g. an engine
  without replay) is all-gathered to [T,H] inside sp_layer_forward, so the drafter always sees full rows.
* Engram layers (1, 14): the n-gram kv = wkv(embed(hash_ids)) needs the same token set on every rank (the embedding table
  is head-sharded over TP and wkv is column-parallel with a gathered output), so it is computed for ALL T rows exactly as
  Engram.forward does (layer 14's rows come from the NVMe reader, publish()/fetch() unchanged, full T). Only the gate/add
  kernel (_fused_engram_post_wkv_kernel: one program per (token, hc copy), no cross-token term) runs on this rank's rows,
  with kv/mask windows [r*S, r*S + real_rows). Bitwise equal to the full-T kernel on those rows.
* Indexer / compressor / KV writes: the attention runs on the gathered full x with the full positions, exactly as before.
* final norm / logits rows: after exit_region everything is full T and identical on all ranks.
* Padding (T not a multiple of 64): zero rows; the all-gather result is sliced [:T] before attention/MoE/exit, the
  reduce-scatter input is zero-padded to W*S rows. No padded row is ever read by a consumer.
* Replay path: DeepseekV4DecoderLayer.write_kv asserts "no sequence parallel"; the region is closed (exit_region +
  end_forward) before DecoderReplayLayers runs, and write_kv additionally asserts that no region is active.
* Mixed prefill + decode batches: eligibility depends on the total number of rows T only. Decode (and spec-verify) rows
  are ordinary rows of the same eager forward: their mHC/attention/MoE math is the same per-row math, and attention/MoE
  see the full batch in the original order. FULL_DECODE_ONLY graph replays never enter Python, captures are refused.

NUMERICS. Reduce-scatter changes the order of the 8-way sum of the TP partials (NCCL ring RS vs ring AR); the pre-norm
GEMM runs with the split-K factor of S rows instead of T rows. Both are valid computations; neither is bitwise equal to
the replicated path. What is guaranteed: row t is computed on exactly one rank (t // S), all consumers read the all-gathered
copy, so all ranks hold identical tensors at every point (the boot self-test and the CPU suite check this).

BOOT SELF-TEST (first eligible forward = the profile run; collective; fail closed; cannot hang): envelope + rank
fingerprint + geometry exchanged over gloo behind a bounded monitored barrier; every NCCL collective of the test goes
through a schedule guard (same order and planned shapes on every rank, a local failure becomes a failing row and a zero
input is issued instead); synthetic RS+AG cases; the live layers' mHC ops on a zero-padded shard vs the full call (tolerance:
the pre-norm GEMM split changes); the Engram kernel on a shard vs full (bitwise); the suppression handles (wo_b,
MoE) and, after a bounded go/no-go vote, the live wo_b and live MoE: suppressed partial + SP reduction vs the unsuppressed
module output and vs the decode all-reduce. A failure raises a [patch-022] error on every rank.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import logging
import os
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001 - CPU tests outside a vLLM install
    logger = logging.getLogger(__name__)

TAG = "[patch-022]"
ENV = "VLLM_GB10_PREFILL_SP"
ENV_MIN_T = "VLLM_GB10_PREFILL_SP_MIN_T"
REVISION = 1
DEFAULT_MIN_T = 1024
MIN_T_FLOOR = 256
MIN_T_CEIL = 1 << 20
REQUIRED_TP = 8
ROW_ALIGN = 8                       # rows per rank are a multiple of this (S = ROW_ALIGN * ceil(T / (ROW_ALIGN * W)))
SELFTEST_BARRIER_S = 600.0          # bounded gloo barrier before every boot exchange (tests lower it)
CHECK2_T = 64                       # rows of the live wo_b / MoE checks: S = 8 per rank, no padding
LIVE_T_EXTRA = 3                    # live mHC checks at T = MIN_T and MIN_T + 3 (the last rank gets a zero-padded tail)
SELFTEST_MAX_CHUNK = 8192           # synthetic reduction cases never exceed this many rows
RTOL_RED = 3.0e-2                   # reduction check: amax-relative error of the NCCL reduce-scatter (bf16 hops) vs the fp32 rank-order sum
RTOL_ACT = 2.0e-2                   # bf16 activations shard vs full (GEMM split-K order may move the coefficients)
RTOL_COEF = 1.0e-2                  # fp32 mixes / coefficients shard vs full
RTOL_CHECK2 = 3.0e-2                # live module: sp-reduced suppressed partial vs unsuppressed output / decode all-reduce


# ---------------------------------------------------------------------------------------------------------------------
# Env parsing (fail closed, at import under the gate)
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


def parse_min_t(raw: str | None) -> int:
    if raw is None or raw.strip() == "":
        return DEFAULT_MIN_T
    s = raw.strip()
    if not s.isdigit():
        raise ValueError(f"{TAG} {ENV_MIN_T}={raw!r} is not a non-negative integer")
    v = int(s)
    if not MIN_T_FLOOR <= v <= MIN_T_CEIL:
        raise ValueError(f"{TAG} {ENV_MIN_T}={v} outside [{MIN_T_FLOOR}, {MIN_T_CEIL}] (default {DEFAULT_MIN_T})")
    return v


def _own_sha16() -> str:
    try:
        with open(__file__, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


MODULE_SHA16 = _own_sha16()
if parse_enabled(os.environ.get(ENV)):
    parse_min_t(os.environ.get(ENV_MIN_T))     # fail closed at import (model.py imports us only under the gate)


# ---------------------------------------------------------------------------------------------------------------------
# Row plan: uniform shards (what NCCL all-gather / reduce-scatter need); rows [T, W*S) of the last shards are padding.
# ---------------------------------------------------------------------------------------------------------------------
class Plan(NamedTuple):
    T: int
    W: int
    S: int                # rows per rank (a multiple of ROW_ALIGN); the padded length is W*S >= T
    pad: int              # W*S - T

    @property
    def padded(self) -> int:
        return self.W * self.S

    def start(self, r: int) -> int:
        return r * self.S

    def real_rows(self, r: int) -> int:
        """Rows of rank r's shard that are < T (the rest is padding)."""
        return max(0, min((r + 1) * self.S, self.T) - r * self.S)


@functools.lru_cache(maxsize=512)
def plan_rows(T: int, W: int, align: int = ROW_ALIGN) -> Plan:
    if T < 1 or W < 1 or align < 1:
        raise ValueError(f"{TAG} bad plan request T={T} W={W} align={align}")
    unit = align * W
    padded = -(-T // unit) * unit
    return Plan(T, W, padded // W, padded - T)


def check_plan(p: Plan, align: int = ROW_ALIGN) -> None:
    assert p.S % align == 0 and p.S > 0, p
    assert p.padded >= p.T and p.padded - p.T == p.pad and p.pad < p.W * align, p
    assert sum(p.real_rows(r) for r in range(p.W)) == p.T, p


def pad_rows(x: torch.Tensor, n_rows: int) -> torch.Tensor:
    """Zero-pad a [rows, ...] tensor along dim 0 up to n_rows (contiguous; no copy when already that size)."""
    if x.shape[0] == n_rows:
        return x.contiguous()
    return torch.nn.functional.pad(x.contiguous(), (0, 0) * (x.dim() - 1) + (0, n_rows - x.shape[0]))


def shard_rows(x_full: torch.Tensor, plan: Plan, rank: int) -> torch.Tensor:
    """This rank's S rows of a full-T tensor (zero-padded tail). No collective."""
    lo, real = plan.start(rank), plan.real_rows(rank)
    if real == plan.S:
        return x_full[lo:lo + plan.S].contiguous()
    out = x_full.new_zeros((plan.S, *x_full.shape[1:]))
    if real > 0:
        out[:real].copy_(x_full[lo:lo + real])
    return out


def gather_rows(x_local: torch.Tensor, world, plan: Plan) -> torch.Tensor:
    """All-gather this rank's S rows -> the first T rows of the gathered tensor (identical on every rank)."""
    if x_local.shape[0] != plan.S:
        raise RuntimeError(f"{TAG} gather got {x_local.shape[0]} rows, expected S={plan.S}")
    g = world.all_gather_rows(x_local.contiguous())
    if g.shape[0] != plan.padded:
        raise RuntimeError(f"{TAG} all-gather returned {g.shape[0]} rows, expected {plan.padded}")
    return g[: plan.T]


def reduce_scatter_rows(partial: torch.Tensor, world, plan: Plan) -> torch.Tensor:
    """Sum the per-rank partials [T, ...] across the TP group and keep this rank's rows -> [S, ...]."""
    if partial.shape[0] != plan.T:
        raise RuntimeError(f"{TAG} reduce-scatter got {partial.shape[0]} rows, expected T={plan.T}")
    out = world.reduce_scatter_rows(pad_rows(partial, plan.padded))
    if out.shape[0] != plan.S:
        raise RuntimeError(f"{TAG} reduce-scatter returned {out.shape[0]} rows, expected S={plan.S}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Worlds (production: the vLLM TP group; tests inject gloo / emulated worlds with the same interface)
# ---------------------------------------------------------------------------------------------------------------------
class _TPWorld:
    """The vLLM TP group: dim-0 all-gather / reduce-scatter / all-reduce on the current stream (NCCL for prefill-size
    messages; patch 018 only takes small ones), gloo object exchange and a bounded barrier on the group's CPU group."""

    kind = "tp"

    def __init__(self) -> None:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        self.group = get_tp_group()
        self.rank = int(self.group.rank_in_group)
        self.size = int(self.group.world_size)
        self.pp_size = int(get_pp_group().world_size)

    def all_gather_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.group.all_gather(x, dim=0)

    def reduce_scatter_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.group.reduce_scatter(x, dim=0)

    def all_reduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.group.all_reduce(x)

    def all_gather_object(self, obj):
        import torch.distributed as dist

        out = [None] * self.size
        dist.all_gather_object(out, obj, group=self.group.cpu_group)
        return out

    def barrier_bounded(self, timeout_s: float) -> None:
        """gloo monitored barrier: returns when every rank arrived; raises on every present rank after timeout_s if one
        did not (instead of blocking in the following all-gather for the 30 min gloo default)."""
        import torch.distributed as dist

        dist.monitored_barrier(group=self.group.cpu_group, timeout=datetime.timedelta(seconds=timeout_s),
                               wait_all_ranks=True)


_WORLD_FACTORY: Callable[[], object] | None = None   # tests only
_OPS_OVERRIDE: "Ops | None" = None                     # tests only


def _make_world():
    return _WORLD_FACTORY() if _WORLD_FACTORY is not None else _TPWorld()


# ---------------------------------------------------------------------------------------------------------------------
# The mHC / Engram callables the layer forward uses (resolved lazily from vLLM; tests inject torch references)
# ---------------------------------------------------------------------------------------------------------------------
class Ops(NamedTuple):
    post: Callable              # mhc_post_tilelang(x, residual, post_mix, res_mix) -> residual'
    pre_delayed: Callable       # mhc_pre_delayed_tilelang(residual, fn, scale, base, ...) -> (post, comb, layer_input, next_pre)
    shifted_post_pre: Callable  # mhc_shifted_post_pre(x, residual, post, comb, fn, ...) -> 6-tuple
    engram_post: Callable       # (engram, hidden_local, kv, token_mask, plan, rank) -> output rows


def _engram_post_real(engram, hidden: torch.Tensor, kv: torch.Tensor, token_mask: torch.Tensor | None, plan: Plan,
                      rank: int, kv_is_local: bool = False) -> torch.Tensor:
    """Engram.forward's gate/add kernel (the last third of Engram.forward) on this rank's rows. `kv` holds ALL T rows
    (computed exactly as Engram.forward computes it); rows [lo, lo + real) belong to this rank. The kernel has one program
    per (token, hc copy) and no cross-token term, so the rows it writes are bitwise those of the full-T call."""
    import triton

    from vllm.models.deepseek_v41.common.engram import _fused_engram_post_wkv_kernel

    num_tokens, hc_mult, dim = hidden.shape
    assert num_tokens == plan.S and hc_mult == engram.hc_mult and dim == engram.dim, (hidden.shape, plan)
    assert kv.ndim == 2 and kv.shape[1] == (hc_mult + 1) * dim, kv.shape
    lo, real = plan.start(rank), plan.real_rows(rank)
    output = torch.empty_like(hidden)
    if kv_is_local:  # VLLM_GB10_PREFILL_SP_ENGRAM=a2a: kv holds this rank's S rows (vllm/_engram_sp_a2a.py)
        assert kv.shape[0] == plan.S, (kv.shape, plan)
        kv_local = kv[:real]
    else:
        kv_local = kv[lo:lo + real] if real > 0 else kv[:0]
    mask_local = None
    if token_mask is not None:
        mask_local = torch.zeros(plan.S, dtype=token_mask.dtype, device=token_mask.device)
        if real > 0:
            mask_local[:real].copy_(token_mask[lo:lo + real])
    block_size = triton.next_power_of_2(dim)
    num_warps = 8 if block_size >= 2048 else 4
    mask = mask_local if mask_local is not None else hidden
    _fused_engram_post_wkv_kernel[(num_tokens * hc_mult,)](
        hidden,
        kv_local,
        engram.q_weight,
        engram.k_weight,
        mask,
        output,
        real,
        hidden.stride(0),
        hidden.stride(1),
        hidden.stride(2),
        kv_local.stride(0),
        kv_local.stride(1),
        engram.q_weight.stride(0),
        engram.q_weight.stride(1),
        engram.k_weight.stride(0),
        engram.k_weight.stride(1),
        mask_local.stride(0) if mask_local is not None else 0,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        engram.eps,
        engram.clamp_value,
        DIM=dim,
        HC_MULT=hc_mult,
        BLOCK_SIZE=block_size,
        HAS_MASK=mask_local is not None,
        num_warps=num_warps,
    )
    return output


def _ops() -> Ops:
    if _OPS_OVERRIDE is not None:
        return _OPS_OVERRIDE
    st = _STATE
    if st.ops is None:
        from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_delayed_tilelang
        from vllm.models.deepseek_v41.nvidia.ops.mhc import mhc_shifted_post_pre

        st.ops = Ops(mhc_post_tilelang, mhc_pre_delayed_tilelang, mhc_shifted_post_pre, _engram_post_real)
    return st.ops


# ---------------------------------------------------------------------------------------------------------------------
# Row-parallel all-reduce suppression (eager, runtime attributes; restored in __exit__). Fail closed on an unknown layout.
# ---------------------------------------------------------------------------------------------------------------------
class _SuppressReduce:
    """Context manager: turn the module's final all-reduce flag off, restore it afterwards (even on exception). Refuses to
    nest: the flag must be at its default on entry."""

    def __init__(self, owner, name: str, off_value: bool):
        self.owner, self.name, self.off = owner, name, off_value
        self.saved = getattr(owner, name)

    def __enter__(self):
        if self.saved == self.off:
            raise RuntimeError(f"{TAG} {type(self.owner).__name__}.{self.name} is already {self.off!r} on entry "
                               "(a leaked or nested suppression)")
        setattr(self.owner, self.name, self.off)
        return self

    def __exit__(self, *exc):
        setattr(self.owner, self.name, self.saved)
        return False


def attn_reduce_handle(layer) -> _SuppressReduce:
    """attn.wo_b is a RowParallelLinear whose forward all-reduces iff `reduce_results` (read at call time); with it off the
    attention returns the per-rank TP partial [T, H] (the contract upstream's own sequence-parallel path relies on)."""
    wo_b = getattr(getattr(layer, "attn", None), "wo_b", None)
    if wo_b is None or not hasattr(wo_b, "reduce_results"):
        raise RuntimeError(f"{TAG} layer.attn.wo_b has no reduce_results flag (unexpected attention layout)")
    return _SuppressReduce(wo_b, "reduce_results", False)


def ffn_reduce_handle(layer) -> _SuppressReduce:
    """The MoE runner's `moe_config.skip_final_all_reduce` (read in MoERunner._maybe_reduce_final_output at call time,
    honoured on the late-all-reduce TP path): the FFN returns shared + routed partials, unreduced."""
    experts = getattr(getattr(layer, "ffn", None), "experts", None)
    cfg = getattr(experts, "moe_config", None)
    if cfg is None or not hasattr(cfg, "skip_final_all_reduce"):
        raise RuntimeError(f"{TAG} layer.ffn.experts.moe_config has no skip_final_all_reduce (unexpected MoE layout)")
    return _SuppressReduce(cfg, "skip_final_all_reduce", True)


# ---------------------------------------------------------------------------------------------------------------------
# Per-process state
# ---------------------------------------------------------------------------------------------------------------------
class _Handle(NamedTuple):
    plan: Plan
    world: object
    rank: int


class _State:
    def __init__(self) -> None:
        self.enabled: bool | None = None
        self.min_t: int | None = None
        self.world = None
        self.report: dict | None = None
        self.active_handle: _Handle | None = None
        self.ops: Ops | None = None
        self.forwards = 0
        self.layers = 0
        self.small_skips = 0
        self.capture_skips = 0
        self.graph_skips = 0
        self.booted = False
        self.first_logged = False
        self.required_tp = REQUIRED_TP


_STATE = _State()


def configured() -> tuple[bool, int]:
    st = _STATE
    if st.enabled is None:
        st.enabled = parse_enabled(os.environ.get(ENV))
        st.min_t = parse_min_t(os.environ.get(ENV_MIN_T))
    return bool(st.enabled), int(st.min_t or DEFAULT_MIN_T)


def sp_active() -> bool:
    """True while a decoder layer executes inside an eligible SP region (set by begin_forward, cleared by end_forward)."""
    return _STATE.active_handle is not None


def active_handle() -> _Handle | None:
    return _STATE.active_handle


def _capturing(t: torch.Tensor | None) -> bool:
    return bool(t is not None and t.is_cuda and torch.cuda.is_current_stream_capturing())


def _eager_forward_context() -> bool:
    """False inside a CUDA-graph runtime mode or a micro-batched forward. Read from the forward context (identical on every
    rank); without a context (CPU tests) True."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return True
        ctx = get_forward_context()
        mode = getattr(ctx, "cudagraph_runtime_mode", None)
        if mode is not None and getattr(mode, "name", str(mode)) != "NONE":
            return False
        return getattr(ctx, "ubatch_slices", None) is None
    except Exception:  # noqa: BLE001 - no vLLM
        return True


# ---------------------------------------------------------------------------------------------------------------------
# Envelope (what this patch is validated for) -- every violation is reported, all ranks raise together
# ---------------------------------------------------------------------------------------------------------------------
def region_layers(model) -> list:
    return [model.layers[i] for i in range(model.start_layer, model.decoder_replay_start)]


def envelope_errors(model, world) -> list[str]:
    errs: list[str] = []
    st = _STATE
    if world.size != st.required_tp:
        errs.append(f"{ENV} is validated for TP={st.required_tp} only (TP group has {world.size} ranks)")
    if getattr(world, "pp_size", 1) != 1:
        errs.append(f"pipeline parallel size {world.pp_size} != 1")
    pc = getattr(model, "parallel_config", None)
    if pc is not None:
        for name, bad in (("pipeline_parallel_size", lambda v: v not in (None, 1)),
                          ("data_parallel_size", lambda v: v not in (None, 1)),
                          ("prefill_context_parallel_size", lambda v: v not in (None, 1)),
                          ("decode_context_parallel_size", lambda v: v not in (None, 1)),
                          ("enable_expert_parallel", bool), ("use_ubatching", bool), ("enable_dbo", bool)):
            v = getattr(pc, name, None)
            if v is not None and bad(v):
                errs.append(f"parallel_config.{name}={v!r} (SP is validated for PP=DP=PCP=DCP=1, no EP, no micro-batching)")
    if getattr(model, "use_sequence_parallel", False):
        errs.append("upstream sequence parallel (use_sequence_parallel) is on: the two shard the residual differently")
    if getattr(model, "fuse_mhc_all_reduce", False):
        errs.append("fuse_mhc_all_reduce is on (the layer's x can be an unfinalized MoEOutput; not supported)")
    if getattr(model, "use_native_mega_moe", False):
        errs.append("native MegaMoE is on (requires EP)")
    if getattr(model, "engram_batch", None) is not None:
        errs.append("EngramBatch (DP-sharded Engram tables) is on")
    vc = getattr(model, "vllm_config", None)
    if vc is not None and getattr(vc, "lora_config", None) is not None:
        errs.append("LoRA is on (it wraps wo_b / the MoE)")
    try:
        import vllm.envs as envs

        if getattr(envs, "VLLM_BATCH_INVARIANT", False):
            errs.append("VLLM_BATCH_INVARIANT=1 (SP changes the reduction order)")
    except Exception:  # noqa: BLE001
        pass
    layers = region_layers(model)
    if not layers:
        errs.append("the SP region has no layers (start_layer == decoder_replay_start)")
    for ly in layers:
        i = getattr(getattr(ly, "attn", None), "layer_id", "?")
        if not hasattr(ly, "hc_attn_fn") or not hasattr(ly, "ffn") or not hasattr(ly, "attn"):
            errs.append(f"layer {i}: not a DeepseekV4DecoderLayer")
            continue
        if getattr(ly, "use_sequence_parallel", False) or getattr(ly, "fuse_mhc_all_reduce", False) \
                or getattr(ly, "mhc_stream", None) is not None:
            errs.append(f"layer {i}: use_sequence_parallel / fuse_mhc_all_reduce / mhc_stream is set")
        wo_b = getattr(ly.attn, "wo_b", None)
        if wo_b is None or getattr(wo_b, "reduce_results", None) is not True:
            errs.append(f"layer {i}: attn.wo_b.reduce_results is {getattr(wo_b, 'reduce_results', None)!r}, expected True")
        elif getattr(wo_b, "tp_size", world.size) != world.size:
            errs.append(f"layer {i}: attn.wo_b.tp_size {wo_b.tp_size} != {world.size}")
        if getattr(ly.attn, "gemm_rs", None) is not None:
            errs.append(f"layer {i}: attn.gemm_rs is bound")
        experts = getattr(ly.ffn, "experts", None)
        cfg = getattr(experts, "moe_config", None)
        if cfg is None or getattr(cfg, "skip_final_all_reduce", None) is not False:
            errs.append(f"layer {i}: ffn.experts.moe_config.skip_final_all_reduce is "
                        f"{getattr(cfg, 'skip_final_all_reduce', None)!r}, expected False")
        else:
            if getattr(cfg, "is_sequence_parallel", False):
                errs.append(f"layer {i}: moe_config.is_sequence_parallel")
            for attr, want in (("ep_size", 1), ("dp_size", 1), ("pcp_size", 1)):
                if getattr(cfg, attr, want) != want:
                    errs.append(f"layer {i}: moe_config.{attr}={getattr(cfg, attr)} != {want}")
            if getattr(cfg, "tp_size", world.size) != world.size:
                errs.append(f"layer {i}: moe_config.tp_size={cfg.tp_size} != {world.size}")
            if getattr(cfg, "use_deferred_moe_finalize", False):
                errs.append(f"layer {i}: the MoE defers its finalize")
            if getattr(experts, "_fused_output_is_reduced", False) is True:
                errs.append(f"layer {i}: the MoE kernel reduces its own output (skip_final_all_reduce needs the late-AR path)")
        eng = getattr(ly, "engram", None)
        if eng is not None and getattr(eng, "_batch", None) is not None:
            errs.append(f"layer {i}: Engram uses an EngramBatch")
        if eng is not None and getattr(eng, "use_sequence_parallel", False):
            errs.append(f"layer {i}: Engram was built sequence-parallel")
    first = layers[0] if layers else None
    if first is not None and getattr(first, "hc_attn_fn_broadcast", None) is None:
        errs.append("the first region layer has no hc_attn_fn_broadcast (finalize_mhc_broadcast_weights not run)")
    return errs


def fingerprint(model, min_t: int, world_size: int) -> str:
    cfg = getattr(model, "config", None)
    parts = ["p022", f"rev={REVISION}", f"module={MODULE_SHA16}", f"min_t={min_t}", f"tp={world_size}",
             f"layers={model.start_layer}-{model.decoder_replay_start}", f"hidden={getattr(cfg, 'hidden_size', '?')}",
             f"hc={getattr(cfg, 'hc_mult', '?')}",
             f"engram={os.environ.get('VLLM_GB10_PREFILL_SP_ENGRAM', 'gather').strip().lower()}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------
# Self-test helpers
# ---------------------------------------------------------------------------------------------------------------------
def _hash_u32(idx: torch.Tensor, seed: int) -> torch.Tensor:
    m = 0xFFFFFFFF
    h = (idx * 0x9E3779B1 + (seed * 0x85EBCA6B & m)) & m
    h = h ^ (h >> 16)
    h = (h * 0x045D9F3B) & m
    h = h ^ (h >> 16)
    h = (h * 0x045D9F3B) & m
    return h ^ (h >> 16)


def synthetic_rows(T: int, width: int, seed: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    """[T, width] with values in +-64 over 8 binary exponents, from the hash of the flat index (identical on all ranks).
    Row t depends only on t, so slicing a shard gives that shard's rows exactly."""
    out = torch.empty(T, width, dtype=dtype, device=device)
    for r0 in range(0, T, 64):
        r1 = min(T, r0 + 64)
        h = _hash_u32(torch.arange(r0 * width, r1 * width, dtype=torch.int64, device=device), seed)
        v = (h & 0xFFFF).to(torch.float32) * (1.0 / 4096.0) - 8.0
        e = ((h >> 16) & 7).to(torch.float32) - 4.0
        out[r0:r1].copy_((v * torch.exp2(e)).view(r1 - r0, width).to(dtype))
    return out


def mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), b.numel(), 1)
    if a.dtype in (torch.bfloat16, torch.float16):
        iv = torch.int16
    elif a.dtype == torch.float32:
        iv = torch.int32
    elif a.dtype == torch.float64:
        iv = torch.int64
    else:
        return int((a != b).sum())
    return int((a.contiguous().view(iv) != b.contiguous().view(iv)).sum())


def amax_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    if not bool(torch.isfinite(a32).all() and torch.isfinite(b32).all()):
        return float("inf")
    return float((a32 - b32).abs().amax() / b32.abs().amax().clamp_min(1e-12))


def _tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.to(torch.float32).contiguous().cpu().numpy().tobytes()).hexdigest()


def _dev(device) -> torch.device:
    return torch.empty(0, device=device).device


def _case_ts(min_t: int, chunk: int) -> list[int]:
    """Synthetic reduction sizes: the threshold, threshold + 3 (a ragged tail) and the served chunk."""
    ts = {min_t, min_t + LIVE_T_EXTRA}
    if min_t <= chunk <= SELFTEST_MAX_CHUNK:
        ts.add(chunk)
    return sorted(ts)


def _live_ts(min_t: int) -> tuple[int, int]:
    return (min_t, min_t + LIVE_T_EXTRA)


class _LocalFail(RuntimeError):
    """A rank-local self-test failure: recorded as a failing row, never allowed to skip a collective."""


def _exchange(world, obj, what: str):
    """gloo all-gather of a per-rank object behind a bounded monitored barrier."""
    try:
        world.barrier_bounded(SELFTEST_BARRIER_S)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{TAG} {ENV}: boot {what}: not every TP rank reached the exchange within "
                           f"{SELFTEST_BARRIER_S:.0f} s ({type(e).__name__}: {str(e)[:300]}); refusing - a rank failed "
                           "or is stuck before it (see that rank's log)") from e
    return world.all_gather_object(obj)


class _Sched:
    """The boot self-test's collective schedule guard. Every NCCL collective of the self-test goes through _issue(): it is
    issued on every rank in the same order with the planned shape, whatever the local code did. A step declares its planned
    collectives [(op, rows)]; a call that does not match the plan (another op, an extra call, an input of another shape /
    dtype / device) raises locally and the PLANNED collective is issued with a zero input; collectives a step did not reach
    (a local exception) are issued with zero inputs when the step ends. So a rank-local failure becomes a failing row on
    that rank and garbage (-> failing rows) for its peers, never a missing or mismatched collective."""

    kind = "selftest-guard"

    def __init__(self, world, width: int, dtype, device) -> None:
        self.w = world
        self.width = int(width)         # every guarded collective moves [rows, hidden] tensors
        self.dtype = dtype
        self.device = device
        self.ops: list = []
        self.i = 0
        self.issued = 0

    @property
    def rank(self) -> int:
        return self.w.rank

    @property
    def size(self) -> int:
        return self.w.size

    def all_gather_rows(self, x):
        return self._issue("ag", x)

    def reduce_scatter_rows(self, x):
        return self._issue("rs", x)

    def all_reduce_rows(self, x):
        return self._issue("ar", x)

    def _problem(self, x, rows) -> str | None:
        if not isinstance(x, torch.Tensor):
            return f"input is {type(x).__name__}, not a tensor"
        if x.shape[0] != rows or x.dim() != 2 or x.shape[1] != self.width or x.dtype != self.dtype or x.device != self.device:
            return f"input {tuple(x.shape)} {x.dtype} {x.device} != planned ({rows}, {self.width}) {self.dtype} {self.device}"
        if not x.is_contiguous():
            return "input not contiguous"
        return None

    def _zeros(self, rows: int) -> torch.Tensor:
        return torch.zeros(rows, self.width, dtype=self.dtype, device=self.device)

    def _real(self, op, x):
        self.issued += 1
        if op == "ag":
            return self.w.all_gather_rows(x)
        if op == "rs":
            return self.w.reduce_scatter_rows(x)
        if op == "ar":
            return self.w.all_reduce_rows(x)
        raise AssertionError(op)

    def _issue(self, op, x):
        if self.i >= len(self.ops):
            raise _LocalFail(f"unplanned extra collective '{op}' (not issued)")
        pop, rows = self.ops[self.i]
        if op != pop:
            raise _LocalFail(f"collective '{op}' where '{pop}' was planned (the planned one is issued)")
        self.i += 1
        bad = self._problem(x, rows)
        if bad is not None:
            self._real(pop, self._zeros(rows))      # keep the group in step
            raise _LocalFail(f"'{op}' {bad} (a zero input of the planned shape was issued instead)")
        return self._real(op, x)

    def run(self, rows_out: list, name: str, ops, fn):
        """Run one step: fn() with `ops` [(op, rows)] as its planned collectives on [rows, hidden] tensors. NEVER raises;
        returns fn()'s result, or None after recording a failing row."""
        self.ops, self.i = list(ops), 0
        res, err = None, None
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001 - a rank-local failure becomes a failing row
            err = f"{type(e).__name__}: {e}"
        try:
            if err is None and self.i < len(self.ops):
                err = f"planned collective(s) {self.ops[self.i:]} not issued by the step"
            while self.i < len(self.ops):
                op, nrows = self.ops[self.i]
                self.i += 1
                self._real(op, self._zeros(nrows))
        except Exception as e:  # noqa: BLE001 - the transport itself failed locally; nothing more can be done here
            err = (err + "; " if err else "") + f"drain failed: {type(e).__name__}: {e}"
        self.ops, self.i = [], 0
        if err is not None:
            rows_out.append({"case": name, "ok": False, "err": err[:400]})
            return None
        return res


def _flag_cases(layer, tag: str) -> list[dict]:
    """The real suppression handles flip the live module's flag inside the context and restore it after."""
    rows: list[dict] = []

    def one(name, make_handle, read, off_value):
        before = read()
        inside = None
        try:
            with make_handle():
                inside = read()
        except Exception as e:  # noqa: BLE001
            rows.append({"case": f"suppress-{tag}-{name}", "ok": False, "err": repr(e)[:160]})
            return
        after = read()
        rows.append({"case": f"suppress-{tag}-{name}", "ok": bool(inside == off_value and after == before and before != off_value),
                     "before": before, "inside": inside, "after": after})

    one("wo_b", lambda: attn_reduce_handle(layer), lambda: layer.attn.wo_b.reduce_results, False)
    one("moe", lambda: ffn_reduce_handle(layer), lambda: layer.ffn.experts.moe_config.skip_final_all_reduce, True)
    return rows


class _SelfTest:
    """The collective boot self-test. Every method issues a FIXED sequence of guarded steps (only the agreed geometry
    selects steps), so all ranks issue the same collectives in the same order."""

    def __init__(self, world, model, geo, live, min_t, hidden, device, dtype, input_ids) -> None:
        self.w, self.W, self.r = world, world.size, world.rank
        self.model, self.geo, self.live = model, geo, live
        self.min_t, self.H, self.dev, self.dtype = min_t, int(hidden), device, dtype
        self.input_ids = input_ids
        self.sched = _Sched(world, hidden, dtype, device)
        self.rows: list[dict] = []
        self.shas: dict[str, str] = {}      # must be bit-identical across ranks
        self.psh: dict[str, str] = {}       # suppressed partials: must differ across ranks
        self.metrics: dict[str, float] = {}

    def step(self, name, ops, fn):
        return self.sched.run(self.rows, name, ops, fn)

    def note(self, key: str, val: float) -> None:
        self.metrics[key] = max(self.metrics.get(key, 0.0), float(val))

    # (1) synthetic reduction cases ---------------------------------------------------------------------------
    def reduction_case(self, T: int) -> None:
        W, r, H, dev, dt = self.W, self.r, self.H, self.dev, self.dtype
        plan = plan_rows(T, W)
        name = f"T={T}"
        c: dict = {}

        def prep():
            check_plan(plan)
            syn = synthetic_rows(T, H, 9001 + T, dev, torch.float32)
            # rank p's partial is bf16(syn * (1 + 0.1 p)): rank-dependent, so the cross-rank sum is non-trivial
            c["partial"] = (syn * (1.0 + 0.1 * r)).to(dt)
            lo, hi = plan.start(r), min(plan.start(r) + plan.S, T)
            # this rank's rows of the sum of ALL ranks' partials: rank order, fp32, one rounding (computed locally)
            acc = None
            for p in range(W):
                v = (syn[lo:hi] * (1.0 + 0.1 * p)).to(dt).to(torch.float32)
                acc = v if acc is None else acc + v
            c["ref"] = acc.to(dt)

        def production():
            c["got"] = reduce_scatter_rows(c["partial"], self.sched, plan)

        def sensitivity():
            # an unsuppressed all-reduce would feed the RS an already-summed partial -> a double reduction; it must be
            # detectably different from the single reduction (so the reconstruction check can fail)
            c["dbl"] = reduce_scatter_rows(gather_rows(c["got"], self.sched, plan), self.sched, plan)

        def final():
            c["full"] = gather_rows(c["got"], self.sched, plan)

        def verdict():
            real = plan.real_rows(r)
            got, ref = c["got"][:real], c["ref"][:real]
            rel = amax_rel(got, ref)
            sens_ok = amax_rel(c["dbl"][:real], ref) > RTOL_RED
            full = c["full"]
            probe = full.clone()
            iv = torch.int16 if full.dtype in (torch.bfloat16, torch.float16) else torch.int32
            probe.view(-1).view(iv)[0] ^= 1
            comparator_ok = mismatches(probe, full) == 1
            ok = rel <= RTOL_RED and sens_ok and comparator_ok and bool(torch.isfinite(full.to(torch.float32)).all())
            self.shas[name] = _tensor_sha(full)
            self.note("rs_rel", rel)
            self.rows.append({"case": name, "T": T, "S": plan.S, "rs_rel": round(rel, 6), "sensitivity_ok": sens_ok,
                              "comparator_ok": comparator_ok, "ok": bool(ok)})

        self.step(f"{name} prep", (), prep)
        self.step(f"{name} reduce-scatter", [("rs", plan.padded)], production)
        self.step(f"{name} double-reduction probe", [("ag", plan.S), ("rs", plan.padded)], sensitivity)
        self.step(f"{name} final all-gather", [("ag", plan.S)], final)
        self.step(f"{name} verdict", (), verdict)

    # (2) live layers: the mHC ops on a zero-padded shard vs the full call (no collective inside) -------------------
    # The shard call gets the SAME inputs as the full call, sliced to this rank's rows, so the comparison isolates the op.
    # The pre-norm GEMM picks its split-K factor from the row count (S vs T rows), so the post/comb/next_pre coefficients
    # (and what is derived from them) may differ in the last bits; the residual stream and the collapsed layer input of the
    # shifted boundary depend only on their inputs and are expected to match bitwise (reported, not required).
    def live_mhc_case(self, ly, T: int, first: bool, engram: bool, tag0: str) -> None:
        W, r, H, dev, dt = self.W, self.r, self.H, self.dev, self.dtype
        plan = plan_rows(T, W)
        lo, real = plan.start(r), plan.real_rows(r)
        hc = int(ly.hc_mult)
        ops = _ops()
        tag = f"live-L{tag0} T={T} S={plan.S}"
        c: dict = {}

        def sh(t):
            return shard_rows(t.contiguous(), plan, r)

        def cmp(nm, full_out, shard_out, tol):
            if full_out.numel() == 0 and shard_out.numel() == 0:
                return                                   # e.g. aux when it was not captured
            f2 = full_out.reshape(full_out.shape[0], -1)[lo:lo + real]
            s2 = shard_out.reshape(shard_out.shape[0], -1)[:real]
            rel = amax_rel(s2, f2)
            self.note("mhc_act_rel" if tol == RTOL_ACT else "mhc_coef_rel", rel)
            self.rows.append({"case": f"{tag} {nm}", "rel": round(rel, 6), "bitwise": mismatches(s2, f2) == 0,
                              "real_rows": real, "ok": bool(rel <= tol)})

        def cmp_all(prefix, names, kinds, full_t, shard_t):
            for nm, k, f, s_ in zip(names, kinds, full_t, shard_t):
                cmp(f"{prefix}.{nm}", f, s_, RTOL_COEF if k == "c" else RTOL_ACT)

        def inputs():
            def syn(seed):
                return (synthetic_rows(T, H, seed + T, dev, torch.float32) * (1.0 / 64)).to(dt)
            c["x0"], c["x1"], c["x2"] = syn(70251), syn(70252), syn(70253)
            c["res0"] = c["x0"].unsqueeze(1).expand(-1, hc, -1).contiguous()

        def calls():
            nw, ne = ly.attn_norm.weight, ly.attn_norm.variance_epsilon
            fw, fe = ly.ffn_norm.weight, ly.ffn_norm.variance_epsilon
            common = (ly.rms_norm_eps, ly.hc_eps, ly.hc_eps, ly.hc_post_alpha, ly.hc_sinkhorn_iters)
            res0, x0, x1, x2 = c["res0"], c["x0"], c["x1"], c["x2"]
            pre_names, pre_kinds = ("post", "comb", "layer_input", "next_pre"), "ccac"
            sp_names, sp_kinds = ("residual", "post", "comb", "layer_input", "next_pre", "aux"), "accaca"
            if first:
                # layer-0 entry: the embedding broadcast to hc copies, projected with the broadcast weight
                fn0, xa, xs = ly.hc_attn_fn_broadcast, x0, sh(x0)
            else:
                fn0, xa, xs = ly.hc_attn_fn, None, None
            full = ops.pre_delayed(res0, fn0, ly.hc_attn_scale, ly.hc_attn_base, *common, x=xa, norm_weight=nw, norm_eps=ne)
            shard = ops.pre_delayed(sh(res0), fn0, ly.hc_attn_scale, ly.hc_attn_base, *common, x=xs, norm_weight=nw,
                                    norm_eps=ne)
            cmp_all("pre_delayed(L0 entry)" if first else "pre_delayed(plain)", pre_names, pre_kinds, full, shard)
            post0, comb0, _, pre0 = (t.contiguous() for t in full)
            if engram:
                # Engram layer: mhc_post, (the Engram add is tested separately), mhc_pre with the carried pre-mix
                fp = ops.post(x1, res0, post0, comb0)
                spp = ops.post(sh(x1), sh(res0), sh(post0), sh(comb0))
                cmp("post(engram prelude)", fp, spp, RTOL_ACT)
                fe_ = ops.pre_delayed(fp, ly.hc_attn_fn, ly.hc_attn_scale, ly.hc_attn_base, *common, pre_mix=pre0,
                                      norm_weight=nw, norm_eps=ne)
                se_ = ops.pre_delayed(spp, ly.hc_attn_fn, ly.hc_attn_scale, ly.hc_attn_base, *common, pre_mix=sh(pre0),
                                      norm_weight=nw, norm_eps=ne)
                cmp_all("pre_delayed(engram)", pre_names, pre_kinds, fe_, se_)
            # attention boundary (post of the previous FFN + pre of this attention), aux captured
            fa = ops.shifted_post_pre(x1, res0, post0, comb0, ly.hc_attn_fn, ly.hc_attn_scale, ly.hc_attn_base, *common,
                                      pre_mix=pre0, norm_weight=nw, norm_eps=ne, capture_aux=True, stream=None,
                                      reduce_results=False)
            sa = ops.shifted_post_pre(sh(x1), sh(res0), sh(post0), sh(comb0), ly.hc_attn_fn, ly.hc_attn_scale,
                                      ly.hc_attn_base, *common, pre_mix=sh(pre0), norm_weight=nw, norm_eps=ne,
                                      capture_aux=True, stream=None, reduce_results=False)
            cmp_all("shifted(attn)", sp_names, sp_kinds, fa, sa)
            # FFN boundary, carrying the attention boundary's outputs (full-call outputs, sliced)
            ff = ops.shifted_post_pre(x2, fa[0].contiguous(), fa[1].contiguous(), fa[2].contiguous(), ly.hc_ffn_fn,
                                      ly.hc_ffn_scale, ly.hc_ffn_base, *common, pre_mix=fa[4].contiguous(),
                                      norm_weight=fw, norm_eps=fe, stream=None, reduce_results=False)
            sf = ops.shifted_post_pre(sh(x2), sh(fa[0]), sh(fa[1]), sh(fa[2]), ly.hc_ffn_fn, ly.hc_ffn_scale,
                                      ly.hc_ffn_base, *common, pre_mix=sh(fa[4]), norm_weight=fw, norm_eps=fe,
                                      stream=None, reduce_results=False)
            cmp_all("shifted(ffn)", sp_names, sp_kinds, ff, sf)

        self.step(f"{tag} inputs", (), inputs)
        self.step(f"{tag} mhc ops shard vs full", (), calls)

    def engram_kernel_case(self, ly, T: int, tag0: str) -> None:
        W, r, H, dev, dt = self.W, self.r, self.H, self.dev, self.dtype
        plan = plan_rows(T, W)
        plan1 = plan_rows(T, 1)
        lo, real = plan.start(r), plan.real_rows(r)
        eng = ly.engram
        hc = int(ly.hc_mult)
        ops = _ops()
        tag = f"engram-L{tag0} T={T}"

        def go():
            hid = (synthetic_rows(T, hc * H, 70261 + T, dev, torch.float32) * (1.0 / 64)).to(dt).view(T, hc, H)
            kv = (synthetic_rows(T, (hc + 1) * H, 70262 + T, dev, torch.float32) * (1.0 / 64)).to(dt)
            mask = (torch.arange(T, device=dev) % 5) != 0
            full = ops.engram_post(eng, shard_rows(hid, plan1, 0), kv, mask, plan1, 0)
            shard = ops.engram_post(eng, shard_rows(hid, plan, r), kv, mask, plan, r)
            mm = mismatches(shard[:real], full[lo:lo + real])
            self.rows.append({"case": tag, "mismatch": mm, "real_rows": real,
                              "ok": mm == 0 and bool(torch.isfinite(shard[:real].to(torch.float32)).all())})

        self.step(f"{tag} kernel shard vs full", (), go)

    # (3) live wo_b and live MoE: suppressed partial + SP reduction vs the unsuppressed output ----------------------
    def check2_case(self, ly, which: str, tag0: str) -> None:
        W, r, H, dev, dt = self.W, self.r, self.H, self.dev, self.dtype
        T2 = CHECK2_T
        plan2 = plan_rows(T2, W)
        tag = f"check2-L{tag0}.{which} T={T2}"
        c: dict = {}

        def module_and_input():
            if which == "wo_b":
                mod = ly.attn.wo_b
                k = int(getattr(mod, "input_size_per_partition", 0) or 0)
                if k <= 0:
                    raise _LocalFail("wo_b has no input_size_per_partition (not a RowParallelLinear?)")
                z = (synthetic_rows(T2, k, 7025 + T2, dev, torch.float32) * (1.0 / 64)).to(dt)
                return (lambda: mod(z)), attn_reduce_handle(ly)
            x2 = (synthetic_rows(T2, H, 7026 + T2, dev, torch.float32) * (1.0 / 64)).to(dt)
            ids = self.input_ids[:T2].clone() if isinstance(self.input_ids, torch.Tensor) and self.input_ids.shape[0] >= T2 \
                else None
            return (lambda: ly.ffn(x2, ids, None)), ffn_reduce_handle(ly)

        def vote_local() -> bool:
            try:
                call, handle = module_and_input()
                c["call"], c["handle"] = call, handle
                return True
            except Exception as e:  # noqa: BLE001
                c["vote_err"] = f"{type(e).__name__}: {e}"
                return False

        # go / no-go vote over gloo (bounded): no rank enters the module call (whose unsuppressed variant contains a
        # collective) unless every rank prepared it
        ok_local = vote_local()
        votes = _exchange(self.w, {"rank": r, "go": bool(ok_local), "err": c.get("vote_err")}, f"{tag} go/no-go")
        if not all(v["go"] for v in votes):
            bad = [(v["rank"], v["err"]) for v in votes if not v["go"]]
            self.rows.append({"case": tag, "ok": False, "err": f"no-go on rank(s) {bad}"[:400]})
            return

        def partial():
            try:
                with c["handle"]:
                    out = c["call"]()
                part = out[0] if isinstance(out, tuple) else out
                if not (isinstance(part, torch.Tensor) and tuple(part.shape) == (T2, H) and part.dtype == dt):
                    raise _LocalFail(f"{which} partial {getattr(part, 'shape', None)} {getattr(part, 'dtype', None)} != ({T2}, {H}) {dt}")
                c["part"] = part.contiguous()
                self.psh[tag] = _tensor_sha(c["part"])
            except Exception:
                c["partial_failed"] = True
                raise

        def sp_reduce():
            c["red"] = gather_rows(reduce_scatter_rows(c["part"], self.sched, plan2), self.sched, plan2)

        def decode_ar():
            c["ar"] = self.sched.all_reduce_rows(c["part"])

        def unsuppressed():
            # the module's own reduction (contains a collective on every rank): entered only if the suppressed call
            # succeeded on EVERY rank (second bounded vote), so a rank-local failure can never leave peers inside it
            votes2 = _exchange(self.w, {"rank": r, "go": not c.get("partial_failed", False)}, f"{tag} go/no-go 2")
            if not all(v["go"] for v in votes2):
                raise _LocalFail("the suppressed call failed on rank(s) "
                                 f"{[v['rank'] for v in votes2 if not v['go']]}; the unsuppressed call was skipped on all ranks")
            out = c["call"]()
            ref = out[0] if isinstance(out, tuple) else out
            if not (isinstance(ref, torch.Tensor) and tuple(ref.shape) == (T2, H)):
                raise _LocalFail(f"{which} unsuppressed output {getattr(ref, 'shape', None)}")
            c["ref"] = ref.contiguous()

        def verdict():
            red, ar, ref = c["red"], c["ar"], c["ref"]
            rel_ar, rel_ref = amax_rel(red, ar), amax_rel(red, ref)
            finite = bool(torch.isfinite(red.to(torch.float32)).all() and torch.isfinite(ref.to(torch.float32)).all())
            ok = finite and rel_ar <= RTOL_CHECK2 and rel_ref <= RTOL_CHECK2
            self.shas[f"{tag} sp-reduced"] = _tensor_sha(red)
            self.shas[f"{tag} unsuppressed"] = _tensor_sha(ref)
            self.note("check2_rel", max(rel_ar, rel_ref))
            self.rows.append({"case": tag, "rel_vs_all_reduce": round(rel_ar, 6), "rel_vs_module": round(rel_ref, 6), "ok": bool(ok)})

        self.step(f"{tag} suppressed partial", (), partial)
        self.step(f"{tag} sp reduce-scatter + all-gather", [("rs", plan2.padded), ("ag", plan2.S)], sp_reduce)
        self.step(f"{tag} decode all-reduce", [("ar", T2)], decode_ar)
        self.step(f"{tag} unsuppressed module output", (), unsuppressed)
        self.step(f"{tag} verdict", (), verdict)

    def run(self) -> None:
        for T in self.geo["case_ts"]:
            self.reduction_case(T)
        for desc, ly in zip(self.geo["live"], self.live):
            tag0 = str(desc["idx"])
            self.step(f"L{tag0} suppression handles", (), lambda ly=ly, tag0=tag0: self.rows.extend(_flag_cases(ly, tag0)))
            for T in self.geo["live_ts"]:
                self.live_mhc_case(ly, T, first=desc["first"], engram=desc["engram"], tag0=tag0)
                if desc["engram"]:
                    self.engram_kernel_case(ly, T, tag0)
        for desc, ly in zip(self.geo["live"], self.live):
            if desc["check2"]:
                for which in ("wo_b", "ffn"):
                    self.check2_case(ly, which, str(desc["idx"]))


def _live_layers(model) -> tuple[list, list[dict]]:
    """(layers, descriptors): the region's first layer (layer-0 entry path), its first Engram layer (Engram path) and its
    first plain layer after the first (plain boundary path; also the live wo_b / MoE check target)."""
    layers = region_layers(model)
    first = layers[0]
    engram = next((ly for ly in layers[1:] if getattr(ly, "engram", None) is not None), None)
    plain = next((ly for ly in layers[1:] if getattr(ly, "engram", None) is None), None)
    picked: list = []
    descs: list[dict] = []
    for ly, is_first, is_eng, c2 in ((first, True, False, plain is None and engram is None),
                                     (engram, False, True, False), (plain, False, False, True)):
        if ly is None:
            continue
        picked.append(ly)
        descs.append({"idx": int(getattr(ly.attn, "layer_id", -1)), "first": is_first, "engram": is_eng, "check2": c2,
                      "hc": int(ly.hc_mult)})
    return picked, descs


def _geometry(model, world, min_t, hidden, chunk) -> tuple[dict, list]:
    picked, descs = _live_layers(model)
    geo = {"W": world.size, "H": int(hidden), "min_t": int(min_t), "chunk": int(chunk),
           "case_ts": _case_ts(min_t, chunk), "live_ts": list(_live_ts(min_t)), "live": descs,
           "layers": [model.start_layer, model.decoder_replay_start]}
    return geo, picked


def _preflight(model, world, min_t, hidden, chunk) -> tuple[str, dict, list]:
    """Envelope + fingerprint + geometry of EVERY rank, exchanged before the first NCCL collective; every rank takes the
    same decision from the same exchanged data (a refusal on one rank raises on all, never a hang)."""
    try:
        fp = fingerprint(model, min_t, world.size)
    except Exception as e:  # noqa: BLE001 - a rank-local failure must still reach the exchange
        fp = f"fingerprint-error:{type(e).__name__}"
    env_err = geo = geo_err = None
    live: list = []
    try:
        errs = envelope_errors(model, world)
        env_err = "; ".join(errs) if errs else None
    except Exception as e:  # noqa: BLE001 - reported to every rank below
        env_err = f"{type(e).__name__}: {e}"
    if env_err is None:
        try:
            geo, live = _geometry(model, world, min_t, hidden, chunk)
        except Exception as e:  # noqa: BLE001
            geo_err = f"{type(e).__name__}: {e}"
    allp = _exchange(world, {"rank": world.rank, "fp": fp, "env_err": env_err, "geo": geo, "geo_err": geo_err}, "preflight")
    errs = [(p["rank"], p["env_err"]) for p in allp if p["env_err"]]
    if errs:
        raise RuntimeError(f"{TAG} boot refused on rank(s) {[r for r, _ in errs]}: {errs[0][1]}"
                           + ("" if len({e for _, e in errs}) == 1 else f" (and: {sorted({e for _, e in errs[1:]})})")
                           + f"; unset {ENV} to serve without it")
    fps = [p["fp"] for p in allp]
    if len(set(fps)) != 1:
        raise RuntimeError(f"{TAG} rank fingerprint mismatch across the TP group: {fps}. A rank has a different patch-022 "
                           f"module, {ENV_MIN_T}, layer range or TP size: the SP collectives would desync. Apply the same "
                           "recipe on every node.")
    gerrs = [(p["rank"], p["geo_err"]) for p in allp if p["geo_err"]]
    if gerrs:
        raise RuntimeError(f"{TAG} boot self-test refused on rank(s) {[r for r, _ in gerrs]}: {gerrs[0][1]}")
    if any(p["geo"] != allp[0]["geo"] for p in allp):
        diff = sorted({k for p in allp for k in (p["geo"] or {}) if p["geo"].get(k) != allp[0]["geo"].get(k)})
        raise RuntimeError(f"{TAG} boot self-test geometry differs across the TP group ({diff}): refusing")
    return fp, geo, live


def _evaluate(allr: list) -> list[str]:
    """Pure function of the exchanged verdicts (identical on every rank): the list of problems."""
    problems = []
    for r in allr:
        if not r["ok"]:
            problems.append(f"rank {r['rank']}: " + ", ".join(f"{c['case']} {c}" for c in r["cases"] if not c["ok"]))
    keys = sorted(set().union(*(set(r["shas"]) for r in allr)))
    for k in keys:
        if len({r["shas"].get(k) for r in allr}) != 1:
            problems.append(f"{k}: the result is not bit-identical across ranks")
    pkeys = sorted(set().union(*(set(r["psh"]) for r in allr)))
    for k in pkeys:
        vals = [r["psh"].get(k) for r in allr]
        if None in vals or len(set(vals)) < 2:
            problems.append(f"{k}: the suppressed partial is identical on every rank or missing (the module output "
                            "was already reduced?)")
    return problems


def selftest(model, world, min_t, hidden, device, dtype, input_ids, chunk, geo=None, live=None) -> dict:
    """Collective boot self-test; raises on EVERY rank unless every rank passed every case. Hang-free: every NCCL
    collective is issued on every rank through the schedule guard, the verdicts are exchanged over gloo behind a bounded
    barrier, and the decision is a pure function of them."""
    device = _dev(device)
    if geo is None:
        geo, live = _geometry(model, world, min_t, hidden, chunk)
    st = _SelfTest(world, model, geo, live or [], min_t, hidden, device, dtype, input_ids)
    t0 = time.time()
    st.run()
    try:
        mine = {"rank": world.rank, "ok": all(r["ok"] for r in st.rows), "cases": st.rows, "shas": st.shas, "psh": st.psh}
    except Exception as e:  # noqa: BLE001 - still take part in the exchange
        mine = {"rank": world.rank, "ok": False, "shas": {}, "psh": {},
                "cases": [{"case": "verdict build", "ok": False, "err": f"{type(e).__name__}: {e}"[:300]}]}
    allr = _exchange(world, mine, "self-test verdict exchange")
    problems = _evaluate(allr)
    if problems:
        raise RuntimeError(f"{TAG} {ENV}=1: BOOT SELF-TEST FAILED (the SP data flow does not reconstruct the reduction "
                           f"reference, ranks disagree, a live mHC / Engram op is not shard-consistent, a suppression "
                           f"handle did not fire, or a step failed on a rank): {'; '.join(problems)[:6000]}; refusing to "
                           f"serve - unset {ENV}")
    return {"cases": len(st.rows), "ranks": len(allr), "collectives": st.sched.issued, "metrics": st.metrics,
            "seconds": round(time.time() - t0, 2), "live_layers": [d["idx"] for d in geo["live"]],
            "case_ts": geo["case_ts"]}


# ---------------------------------------------------------------------------------------------------------------------
# The hooks (called from the patch-022 edits of model.py)
# ---------------------------------------------------------------------------------------------------------------------
def begin_forward(model, hidden_states: torch.Tensor, positions: torch.Tensor, input_ids) -> _Handle | None:
    """Called in DeepseekV4Model.forward right before the first _run_layers (after the specialized-decode early return).
    Returns a handle if this forward is eligible (opening the SP region), else None. Boots on the first eligible forward.
    The decision depends only on T, the capture state and the forward context: identical on every rank."""
    st = _STATE
    enabled, min_t = configured()
    if not enabled:
        return None
    T = int(positions.shape[0])
    if T < min_t:
        st.small_skips += 1
        return None
    if _capturing(hidden_states):
        st.capture_skips += 1
        return None
    if not _eager_forward_context():
        st.graph_skips += 1
        return None
    if not (isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 2 and hidden_states.shape[0] == T):
        raise RuntimeError(f"{TAG} hidden_states {getattr(hidden_states, 'shape', None)} is not [T={T}, H] "
                           "(a MoEOutput or a PP intermediate?): refusing")
    if st.world is None:
        try:
            world = _make_world()
        except Exception as e:  # noqa: BLE001 - this rank cannot even join the preflight
            raise RuntimeError(f"{TAG} {ENV}: could not set up the TP world on this rank ({type(e).__name__}: {e}); the "
                               f"other ranks refuse at the preflight barrier after {SELFTEST_BARRIER_S:.0f} s") from e
        dev = _dev(hidden_states.device)
        chunk = int(getattr(getattr(getattr(model, "vllm_config", None), "scheduler_config", None),
                            "max_num_batched_tokens", T) or T)
        hidden = int(hidden_states.shape[1])
        fp, geo, live = _preflight(model, world, min_t, hidden, chunk)
        st.report = selftest(model, world, min_t, hidden, dev, hidden_states.dtype, input_ids, chunk, geo=geo, live=live)
        st.world = world
        st.booted = True
        rep = st.report
        logger.info(
            "%s prefill sequence-parallel ACTIVE (rank %d/%d, T >= %d, eager prefill only, layers %d-%d, fingerprint %s): "
            "boot self-test OK on all %d ranks (%d cases, %d guarded collectives, %.1f s; live layers %s; synthetic T %s; "
            "max err: reduce-scatter %.2e, mHC act %.2e, coef %.2e, live module vs all-reduce %.2e). NUMERICS CHANGE vs the "
            "all-reduce path (prefill reduction order, pre-norm GEMM split).",
            TAG, world.rank, world.size, min_t, model.start_layer, model.decoder_replay_start - 1, fp, rep["ranks"],
            rep["cases"], rep["collectives"], rep["seconds"], rep["live_layers"], rep["case_ts"],
            rep["metrics"].get("rs_rel", 0.0), rep["metrics"].get("mhc_act_rel", 0.0), rep["metrics"].get("mhc_coef_rel", 0.0),
            rep["metrics"].get("check2_rel", 0.0))
    world = st.world
    a2a = _engram_a2a()
    if a2a is not None:
        a2a.register_model(model)
    plan = plan_rows(T, world.size)
    handle = _Handle(plan, world, world.rank)
    st.active_handle = handle
    st.forwards += 1
    if not st.first_logged:
        st.first_logged = True
        logger.info("%s first SP prefill forward: T=%d rows -> S=%d per rank (pad %d), layers %d-%d, then full-T exit "
                    "gather for the replay/collapse", TAG, T, plan.S, plan.pad, model.start_layer, model.decoder_replay_start - 1)
    return handle


def end_forward(handle: _Handle | None) -> None:
    """Called in a finally around the region. Clears sp_active; idempotent."""
    _STATE.active_handle = None


def shard_input(hidden_states: torch.Tensor, handle: _Handle) -> torch.Tensor:
    """Region entry: the (replicated) embedding [T, H] -> this rank's [S, H] rows. A local slice, no collective."""
    return shard_rows(hidden_states, handle.plan, handle.rank)


def exit_region(handle: _Handle, hidden_states, residual, post_mix, res_mix, pre_mix):
    """Region exit: the five carried states, this rank's rows -> full T, identical on every rank. The bounded replay
    (write_kv of the last KV source for every row, the trimmed replay layers), the collapse and the MTP/HC buffer all take
    full-T tensors, so everything after the region runs as before."""
    plan, world = handle.plan, handle.world
    for nm, t in (("hidden_states", hidden_states), ("residual", residual), ("post_mix", post_mix),
                  ("res_mix", res_mix), ("pre_mix", pre_mix)):
        if not isinstance(t, torch.Tensor) or t.shape[0] != plan.S:
            raise RuntimeError(f"{TAG} exit_region: {nm} {getattr(t, 'shape', None)} does not have S={plan.S} rows")
    T, S = plan.T, plan.S
    x = gather_rows(hidden_states, world, plan)
    res = gather_rows(residual, world, plan)
    # the three small fp32 states travel in one gather
    hc = residual.shape[1]
    packed = torch.cat([post_mix.reshape(S, -1), res_mix.reshape(S, -1), pre_mix.reshape(S, -1)], dim=1).contiguous()
    g = gather_rows(packed, world, plan)
    n_post, n_res = post_mix[0].numel(), res_mix[0].numel()
    post = g[:, :n_post].reshape(T, *post_mix.shape[1:]).contiguous()
    resm = g[:, n_post:n_post + n_res].reshape(T, *res_mix.shape[1:]).contiguous()
    pre = g[:, n_post + n_res:].reshape(T, *pre_mix.shape[1:]).contiguous()
    assert hc == post_mix.shape[1], (hc, post_mix.shape)
    return x, res, post, resm, pre


def _engram_sp(engram, hidden_local: torch.Tensor, hash_ids: torch.Tensor, token_mask, handle: _Handle) -> torch.Tensor:
    """Engram.forward for this rank's rows (see the module docstring)."""
    plan = handle.plan
    if hash_ids.shape[0] != plan.T:
        raise RuntimeError(f"{TAG} engram hash_ids has {hash_ids.shape[0]} rows, expected T={plan.T}")
    assert token_mask is None or token_mask.shape == (plan.T,)
    a2a = _engram_a2a()
    if a2a is not None:  # VLLM_GB10_PREFILL_SP_ENGRAM=a2a: this rank's rows only, no all-gather (first use self-verified)
        kv_local = a2a.verified_local_kv(engram, hash_ids, plan, handle.rank, handle.world)
        if kv_local is not None:
            return _ops().engram_post(engram, hidden_local, kv_local, token_mask, plan, handle.rank, kv_is_local=True)
    kv = engram.wkv(engram.embed(hash_ids).flatten(-2))      # ALL T rows, TP collectives inside, as Engram.forward
    return _ops().engram_post(engram, hidden_local, kv, token_mask, plan, handle.rank)


_A2A_MOD: Any = None


def _engram_a2a():
    """vllm/_engram_sp_a2a.py when VLLM_GB10_PREFILL_SP_ENGRAM=a2a (and not disabled by its self-test), else None."""
    global _A2A_MOD
    if os.environ.get("VLLM_GB10_PREFILL_SP_ENGRAM", "gather").strip().lower() != "a2a":
        return None
    if _A2A_MOD is None:
        from vllm import _engram_sp_a2a as m

        _A2A_MOD = m
    return _A2A_MOD if _A2A_MOD.active() else None


def sp_layer_forward(layer, x, positions, input_ids, pre_mix, post_mix, res_mix, residual, engram_hashes, engram_mask,
                     *, capture_previous_aux: bool = False, mega_gate_metadata=None):
    """Replacement of DeepseekV4DecoderLayer.forward inside an SP region, for this config (no mHC overlap stream, no fused
    mHC all-reduce, no upstream sequence parallel, PP=1). Mirrors the upstream forward statement by statement; the residual
    stream stays on this rank's S rows and the two row-parallel all-reduces become reduce-scatters."""
    handle = _STATE.active_handle
    assert handle is not None, f"{TAG} sp_layer_forward called outside an SP region"
    ops = _ops()
    plan, world = handle.plan, handle.world
    S = plan.S
    previous_aux = None
    common = (layer.rms_norm_eps, layer.hc_eps, layer.hc_eps, layer.hc_post_alpha, layer.hc_sinkhorn_iters)
    if not (isinstance(x, torch.Tensor) and x.shape[0] == S):
        raise RuntimeError(f"{TAG} layer input {getattr(x, 'shape', type(x).__name__)} is not this rank's {S} rows "
                           "(a MoEOutput, or a layer called outside the SP region's shard?)")
    if residual is None:
        assert x.dim() == 2 and layer.hc_attn_fn_broadcast is not None, "SP expects the embedding rows at the first layer"
        residual = x.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
        post_mix, res_mix, x, attn_pre = ops.pre_delayed(
            residual, layer.hc_attn_fn_broadcast, layer.hc_attn_scale, layer.hc_attn_base, *common,
            x=x, norm_weight=layer.attn_norm.weight, norm_eps=layer.attn_norm.variance_epsilon)
    elif layer.engram is not None and engram_hashes is not None:
        previous_post = ops.post(x, residual, post_mix, res_mix)
        if capture_previous_aux:
            previous_aux = previous_post.mean(dim=1)
        residual = _engram_sp(layer.engram, previous_post, engram_hashes[:, layer.engram.layer_hash_index], engram_mask, handle)
        post_mix, res_mix, x, attn_pre = ops.pre_delayed(
            residual, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, *common, pre_mix=pre_mix,
            norm_weight=layer.attn_norm.weight, norm_eps=layer.attn_norm.variance_epsilon)
    else:
        residual, post_mix, res_mix, x, attn_pre, aux = ops.shifted_post_pre(
            x, residual, post_mix, res_mix, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, *common,
            pre_mix=pre_mix, norm_weight=layer.attn_norm.weight, norm_eps=layer.attn_norm.variance_epsilon,
            capture_aux=capture_previous_aux, stream=None, reduce_results=False)
        if capture_previous_aux:
            previous_aux = aux
    if previous_aux is not None:
        previous_aux = gather_rows(previous_aux, world, plan)       # the drafter consumes full [T, H] rows

    # attention on the whole sequence; wo_b's all-reduce suppressed -> reduce-scatter
    x_full = gather_rows(x, world, plan)
    with attn_reduce_handle(layer):
        x_partial = layer.attn(positions, x_full, None)
    x = reduce_scatter_rows(x_partial, world, plan)

    residual, post_mix, res_mix, x, ffn_pre, _ = ops.shifted_post_pre(
        x, residual, post_mix, res_mix, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, *common,
        pre_mix=attn_pre, norm_weight=layer.ffn_norm.weight, norm_eps=layer.ffn_norm.variance_epsilon,
        stream=None, reduce_results=False)

    # MoE on the whole sequence; the final all-reduce suppressed -> reduce-scatter
    x_full = gather_rows(x, world, plan)
    with ffn_reduce_handle(layer):
        x_partial = layer.ffn(x_full, input_ids, mega_gate_metadata)
    x = reduce_scatter_rows(x_partial, world, plan)
    _STATE.layers += 1
    return x, residual, post_mix, res_mix, ffn_pre, previous_aux


def stats() -> dict:
    st = _STATE
    return {"booted": st.booted, "enabled": st.enabled, "min_t": st.min_t, "world": getattr(st.world, "kind", None),
            "forwards": st.forwards, "layers": st.layers, "small_skips": st.small_skips,
            "capture_skips": st.capture_skips, "graph_skips": st.graph_skips, "active": sp_active(), "report": st.report}


def _set_world_factory_for_test(factory: Callable[[], object] | None) -> None:
    global _WORLD_FACTORY
    _WORLD_FACTORY = factory


def _set_ops_for_test(ops: Ops | None) -> None:
    global _OPS_OVERRIDE
    _OPS_OVERRIDE = ops


def _reset_for_test(enabled: bool | None = None, min_t: int | None = None, required_tp: int = REQUIRED_TP) -> None:
    global _STATE
    _STATE = _State()
    _STATE.enabled, _STATE.min_t, _STATE.required_tp = enabled, min_t, required_tp
