# SPDX-License-Identifier: Apache-2.0
# [patch-038 dspark-topk-gather] installed as vllm/_dspark_topk_gather.py by 038-dspark-topk-gather.sh (opt-in VLLM_DSPARK_TOPK_GATHER).
"""DSpark drafter: all-gather the per-rank top-k CANDIDATES of the draft logits instead of the full vocabulary (patch 038).

WHY. With dspark_draft_topk (patch 013) the drafter needs only the top-k (2,048) base logits of each of its num_reqs x 5 draft
positions: stock vLLM computes the vocab-parallel head on every rank, ALL-GATHERS the full vocabulary (129,280 bf16 per row,
8 ranks x 16,160), copies it into rank-major order and takes torch.topk over it. On the 8 x 100 GbE fabric that all-gather runs at
line rate (torch profile b056, 64 users, 65 drafter-graph replays: NCCL AllGather RING_LL 6.10 ms for 82.8 MB, + ~0.8 ms rank-major layout
copy, + 1.46 ms of top-k kernels).

WHAT. Every rank takes the top-k of its OWN shard, the ranks all-gather only those candidates (one int32 per candidate: the bf16
bits of the logit in the high half, the shard-local index in the low half; k x 4 B per row and rank instead of 16,160 x 2 B: 4x
less traffic), and every rank selects the global top-k from the 8 x k candidates. The result (values, ids, order) is the one
torch.topk gives over the gathered full vocabulary: the global top-k can only contain elements that are in their own shard's
top-k (an element outside its shard's top-k has >= k elements ahead of it, in the total order value-desc / index-asc, so it cannot
be in the global top-k either). The dense [num_reqs, n_spec, V] buffer the sampler sees (-inf filled by the stock code that
follows) and the candidate lists are therefore the stock ones; the drafter's sampling, the rejection sampler's cached draft
logits and the output distribution are untouched. Only the draft candidates' PRODUCTION changed -> lossless by construction.

TIES. bf16 logits tie often (8 mantissa bits; synthetic random logits have a tie group at the 2,048th value in most rows), so the tie
order matters for bit-identity with the stock path (never for the output distribution). PyTorch's CUDA topk (TensorTopK.cu) takes the elements
strictly above the k-th value and fills the rest with the lowest-index elements equal to it, then sorts with a stable radix sort: the
canonical order, value descending / index ascending. The selection above is exactly that total order, so the result equals the stock
one IF the two torch.topk calls used here resolve ties canonically too -- verified on the target build by tests/038/single_gpu_test.py
and tp8_test.py, and at every boot by a one-time diagnostic on the first eager call (a log line, no behaviour change; the shadow check
compares real data). VLLM_DSPARK_TOPK_GATHER=keys makes the order canonical by construction (unique int32 keys carry value order and
index in one word; ~4x the top-k traffic of the default mode).

ENV (all read at import; nothing changes unless VLLM_DSPARK_TOPK_GATHER is set):
  VLLM_DSPARK_TOPK_GATHER=1|keys           1 = torch.topk on the bf16 values (default mode); keys = exact keys
  VLLM_DSPARK_TOPK_GATHER_MIN_ROWS=16      use the new path only when num_reqs x n_spec >= this (the stock RoCE all-gather
                                           is already cheap for a few users); the choice is static per CUDA graph
  VLLM_DSPARK_TOPK_GATHER_CHECK=N          shadow check: also run the stock path (full all-gather + torch.topk) on the first two capture warm-ups
                                           (ONE repeated dummy row: plumbing and shapes only) and on the first N eager calls AFTER the CUDA graphs are
                                           captured (real traffic: mixed prefill / decode steps with >= MIN_ROWS rows), and compare values / ids /
                                           order bit for bit (a log line per check; costs one extra full gather each)

Set the SAME VLLM_DSPARK_TOPK_GATHER* values on EVERY rank (one recipe, M_DOCKER_EXTRA): a rank without the hook issues a different collective
(bf16 [R, 16160] instead of int32 [R, 2048]) and the group hangs; nothing inside a disabled rank can detect that.

SCOPE / FALLBACK. Active only for the DeepSeek-V4.1 DSpark draft model with a vocab-parallel head, tp > 1, no added (LoRA)
vocabulary, no draft->target map, a bf16 head, k <= shard size, vocab >= k and shard-local indices that fit 15 bits (native) /
16 bits (keys); native mode also needs an unpadded vocabulary (DeepSeek-V4.1: 8 x 16,160 = 129,280 exactly; keys mode masks the padding
rows and orders -inf ties by index). Anything else runs the stock path unchanged (one log line says why). No host synchronisation, static shapes: the
drafter's FULL CUDA graph captures it. The collective is the TP group's own all_gather (NCCL, or patch 018's RoCE path for small
payloads), issued by every rank in the same order as the stock all-gather it replaces.
"""
from __future__ import annotations

import inspect
import logging
import os
import sys

import torch

try:  # inside vLLM; standalone (unit tests) falls back to the stdlib logger
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)

try:
    from vllm.triton_utils import tl, triton

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

TAG = "[patch-038]"
ENV = "VLLM_DSPARK_TOPK_GATHER"
ENV_MIN_ROWS = "VLLM_DSPARK_TOPK_GATHER_MIN_ROWS"
ENV_CHECK = "VLLM_DSPARK_TOPK_GATHER_CHECK"
_MODEL_CLASS = "DSparkDeepseekV4ForCausalLM"
_MAX_P = {"native": 1 << 15, "keys": 1 << 16}  # shard-local indices must fit the low half of the word
_MAX_WK = 1 << 16  # keys mode: candidate positions (ws * k) must fit 16 bits
_WARM_CHECKS = 2  # shadow checks on the capture warm-ups (dummy rows) when VLLM_DSPARK_TOPK_GATHER_CHECK > 0


def _parse_mode(raw: str | None) -> str | None:
    s = (raw or "").strip().lower()
    if s in ("", "0", "off", "false", "no"):
        return None
    if s in ("1", "on", "true", "yes", "native"):
        return "native"
    if s == "keys":
        return "keys"
    raise ValueError(f"{TAG} {ENV}={raw!r} must be unset/0 (off), 1 (torch.topk tie order) or keys (exact keys)")


def _parse_int(name: str, default: int, lo: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{TAG} {name}={raw!r} is not an integer") from None
    if v < lo:
        raise ValueError(f"{TAG} {name}={v} must be >= {lo}")
    return v


MODE = _parse_mode(os.environ.get(ENV))
ENABLED = MODE is not None
MIN_ROWS = _parse_int(ENV_MIN_ROWS, 16)
CHECK = _parse_int(ENV_CHECK, 0)


# ----------------------------------------------------------------------------------------------------------------------
# pure selection logic (no vLLM, no collectives): importable and testable on CPU
# ----------------------------------------------------------------------------------------------------------------------
def torch_topk(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The stock call: values (sorted descending) and int64 indices of the k largest along the last dim."""
    return torch.topk(x, k, dim=-1)


# -- native mode: payload word = (bf16 bits << 16) | local index (index in the low half; little-endian int16 planes) --------
def native_pack(values: torch.Tensor, local_idx: torch.Tensor) -> torch.Tensor:
    """values bf16 [R, k], local_idx int [R, k] (0 <= idx < 2**15) -> int32 [R, k] words."""
    out = torch.empty(values.shape, dtype=torch.int32, device=values.device)
    planes = out.view(torch.int16).view(*values.shape, 2)
    planes[..., 0].copy_(local_idx)
    planes[..., 1].copy_(values.view(torch.int16))
    return out


def native_candidate_values(gathered: torch.Tensor, ws: int, rows: int, k: int) -> torch.Tensor:
    """gathered int32 [ws * rows, k] (rank-major, as dim-0 all_gather returns it) -> bf16 [rows, ws * k], candidates of a row
    in rank order, each rank's block in its own sorted order."""
    hi = gathered.view(torch.int16).view(ws, rows, k, 2)[..., 1]
    return hi.view(torch.bfloat16).permute(1, 0, 2).reshape(rows, ws * k)


def _finalize_torch(gathered: torch.Tensor, pos: torch.Tensor, ws: int, rows: int, k: int, shard: int) -> torch.Tensor:
    r = torch.div(pos, k, rounding_mode="floor")
    p = pos - r * k
    row = torch.arange(rows, device=pos.device, dtype=torch.int64).unsqueeze(1)
    word = (r * rows + row) * k + p
    lo = gathered.view(torch.int16).view(-1, 2)[:, 0]
    return (lo[word].to(torch.int64) & 0xFFFF) + r * shard


def _jit(fn):
    """triton.jit with the int specialisation (divisible by 16 / equal to 1) of R and P switched off: ONE compiled variant, built by the first eager
    call, so no variant is ever compiled inside a CUDA graph capture. A Triton without the option gets the plain decorator."""
    try:
        return triton.jit(do_not_specialize=["R", "P"])(fn)
    except Exception:  # noqa: BLE001
        return triton.jit(fn)


if _HAS_TRITON:

    @_jit
    def _finalize_kernel(
        pos_ptr,  # [R, K] int64: chosen candidate positions in [0, WS * K)
        g_ptr,  # [WS * R, K] int32: gathered payload words (low 16 bits: shard-local index)
        out_ptr,  # [R, K] int64: global vocabulary ids
        R,
        P,
        K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = offs < K
        pos = tl.load(pos_ptr + row * K + offs, mask=m, other=0)
        r = pos // K
        p = pos - r * K
        word = tl.load(g_ptr + (r * R + row) * K + p, mask=m, other=0)
        local = (word & 0xFFFF).to(tl.int64)
        tl.store(out_ptr + row * K + offs, local + r * P, mask=m)


_TRITON_OK: bool | None = None  # None = not tried yet; set by the first eager call


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # pragma: no cover
        return False


def native_finalize(gathered: torch.Tensor, pos: torch.Tensor, ws: int, rows: int, k: int, shard: int) -> torch.Tensor:
    """Global vocabulary ids (int64 [rows, k]) of the chosen candidate positions."""
    global _TRITON_OK
    if _HAS_TRITON and pos.is_cuda and _TRITON_OK is not False:
        if _TRITON_OK is None and _capturing():
            pass  # never compile under capture: this graph takes the torch ops, eager calls still try Triton
        else:
            try:
                out = torch.empty((rows, k), dtype=torch.int64, device=pos.device)
                _finalize_kernel[(rows, triton.cdiv(k, 1024))](pos, gathered, out, rows, shard, K=k, BLOCK=1024, num_warps=4)
                _TRITON_OK = True
                return out
            except Exception as e:  # noqa: BLE001 -- identical result from the torch ops, so a fallback is safe on any rank
                _TRITON_OK = False
                logger.warning("%s Triton finalize kernel unavailable (%s: %s); using torch ops", TAG, type(e).__name__, e)
    return _finalize_torch(gathered, pos, ws, rows, k, shard)


def native_select(local_logits: torch.Tensor, k: int, topk_fn=torch_topk) -> torch.Tensor:
    """Per-rank stage: top-k of the shard's logits [R, P] -> payload int32 [R, k]."""
    values, idx = topk_fn(local_logits, k)
    return native_pack(values, idx)


def native_finish(gathered: torch.Tensor, ws: int, rows: int, k: int, shard: int, topk_fn=torch_topk):
    """Global stage: payloads of all ranks (int32 [ws * rows, k], rank-major) -> (values bf16 [rows, k], ids int64 [rows, k])."""
    cand = native_candidate_values(gathered, ws, rows, k)
    values, pos = topk_fn(cand, k)
    return values, native_finalize(gathered, pos, ws, rows, k, shard)


# -- keys mode: payload word = (ordered bf16 key - 0x8000) * 65536 + (0xFFFF - local index): unique, torch.topk's order --------
def ordered_center(x: torch.Tensor) -> torch.Tensor:
    """bf16 [..] -> int32 in [-32768, 32767] whose order is torch.topk's value order (-0.0 < +0.0, NaN largest)."""
    b = x.view(torch.int16).to(torch.int32) & 0xFFFF
    o = torch.where(b >= 0x8000, 0xFFFF - b, b + 0x8000)
    o = o.masked_fill(x != x, 0xFFFF)
    return o - 0x8000


def keys_select(local_logits: torch.Tensor, k: int, topk_fn=torch_topk) -> torch.Tensor:
    rows, shard = local_logits.shape
    inv = 0xFFFF - torch.arange(shard, device=local_logits.device, dtype=torch.int32)
    key = ordered_center(local_logits) * 65536 + inv
    top, _ = topk_fn(key, k)
    return top


def keys_finish(gathered: torch.Tensor, ws: int, rows: int, k: int, shard: int, topk_fn=torch_topk):
    n = ws * k
    posinv = (0xFFFF - torch.arange(n, device=gathered.device, dtype=torch.int32)).view(ws, 1, k)
    key2 = ((gathered.view(ws, rows, k) >> 16) * 65536 + posinv).permute(1, 0, 2).reshape(rows, n)
    top, _ = topk_fn(key2, k)
    pos = (0xFFFF - (top & 0xFFFF)).to(torch.int64)
    o = (top >> 16) + 0x8000
    bits = torch.where(o >= 0x8000, o - 0x8000, 0xFFFF - o)
    values = bits.to(torch.int16).view(torch.bfloat16)
    r = torch.div(pos, k, rounding_mode="floor")
    p = pos - r * k
    row = torch.arange(rows, device=gathered.device, dtype=torch.int64).unsqueeze(1)
    word = gathered.view(-1)[(r * rows + row) * k + p]
    local = 0xFFFF - (word & 0xFFFF).to(torch.int64)
    return values, local + r * shard


def select(mode: str, local_logits: torch.Tensor, k: int, topk_fn=torch_topk) -> torch.Tensor:
    if k > local_logits.shape[-1]:
        raise ValueError(f"{TAG} k={k} exceeds the shard size {local_logits.shape[-1]}")
    return (native_select if mode == "native" else keys_select)(local_logits, k, topk_fn)


def finish(mode: str, gathered: torch.Tensor, ws: int, rows: int, k: int, shard: int, topk_fn=torch_topk):
    return (native_finish if mode == "native" else keys_finish)(gathered, ws, rows, k, shard, topk_fn)


def mask_padding(local_logits: torch.Tensor, n_valid: int) -> torch.Tensor:
    """-inf into the shard's vocabulary-padding rows (the stock path slices them off the gathered vector); in place."""
    if n_valid < local_logits.shape[-1]:
        local_logits[..., n_valid:] = float("-inf")
    return local_logits


# ----------------------------------------------------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------------------------------------------------
def tie_order_report(device, k: int, shard: int, ws: int, vocab: int, topk_fn=torch_topk, seed: int = 38) -> dict[str, bool]:
    """Is topk_fn's result on tie-heavy bf16 data the canonical one (value desc, index asc = a stable descending sort)?
    Checked at the shapes the patch (and the stock path) use: small rows (single-block top-k) and many rows (multi-block).
    Returns {"rows x length": canonical?}. One-time, a few small kernels, eager only."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out: dict[str, bool] = {}
    for rows in (4, 24):
        for n, kk in ((shard, k), (ws * k, k), (vocab, k)):
            kk = min(kk, n)
            x = ((torch.randint(0, 24, (rows, n), generator=g) - 12).to(torch.bfloat16) * 0.25).to(device)
            ref_i = torch.sort(x.float(), dim=-1, descending=True, stable=True).indices[:, :kk]
            v, i = topk_fn(x, kk)
            ok = bool(torch.equal(i, ref_i)) and bool(torch.equal(v, x.gather(1, ref_i)))
            out[f"{rows}x{n}"] = ok
    return out


# ----------------------------------------------------------------------------------------------------------------------
# the drafter hook
# ----------------------------------------------------------------------------------------------------------------------
_get_tp_group = None  # tests install a fake; None = vllm.distributed.get_tp_group


def _tp_group():
    if _get_tp_group is not None:
        return _get_tp_group()
    from vllm.distributed import get_tp_group

    return get_tp_group()


class _Plan:
    """Static facts about one drafter: computed once from attributes (no device work)."""

    def __init__(self, spec) -> None:
        self.ok = False
        self.reason = ""
        self.mode = MODE
        self.checks_left = CHECK  # shadow checks on real traffic (eager calls after the CUDA graphs are captured)
        self.warm_checks_left = min(_WARM_CHECKS, CHECK)  # shadow checks on the capture warm-ups (one dummy row)
        self.diag_done = False
        try:
            self._build(spec)
        except Exception as e:  # noqa: BLE001 -- a surprise in the model layout disables the patch rather than the engine
            self.ok = False
            self.reason = f"plan failed: {type(e).__name__}: {e}"

    def _off(self, why: str) -> None:
        self.ok = False
        self.reason = why

    def _build(self, spec) -> None:
        if sys.byteorder != "little":
            return self._off("big-endian host (the payload planes and the Triton kernel assume little-endian)")
        m = spec.model
        if not any(c.__name__ == _MODEL_CLASS for c in type(m).__mro__):
            return self._off(f"draft model is {type(m).__name__}, not {_MODEL_CLASS}")
        lm, lp = m.lm_head, m.logits_processor
        if getattr(m, "draft_id_to_target_id", None) is not None:
            return self._off("reduced draft vocabulary (draft->target map)")
        ws = int(lm.tp_size)
        if ws < 2:
            return self._off("tensor-parallel size 1 (nothing is gathered)")
        if getattr(lm, "num_added_embeddings", 0) != 0:
            return self._off("added (LoRA) vocabulary")
        if "skip_gather" not in inspect.signature(type(lp).forward).parameters:
            return self._off("LogitsProcessor.forward has no skip_gather")
        if getattr(spec, "dtype", torch.bfloat16) != torch.bfloat16 or getattr(lp, "head_dtype", None) not in (None, torch.bfloat16):
            return self._off("the draft head is not bf16")
        shard = int(lm.num_embeddings_per_partition)
        vocab = int(lp.org_vocab_size)
        k = int(spec._draft_topk)
        rank = int(lm.tp_rank)
        n_valid = max(0, min(shard, vocab - rank * shard))
        if n_valid != int(lm.shard_indices.num_org_elements):
            return self._off(f"shard layout differs from the contiguous-vocabulary assumption ({n_valid} vs "
                             f"{lm.shard_indices.num_org_elements} valid rows)")
        if not 1 <= k <= shard:
            return self._off(f"draft top-k {k} outside [1, shard size {shard}]")
        if vocab < k or ws * shard < vocab:
            return self._off(f"vocabulary {vocab} < k {k} or smaller than the shards ({ws} x {shard})")
        if shard > _MAX_P[self.mode]:
            return self._off(f"shard size {shard} does not fit the payload index field ({self.mode} mode)")
        if self.mode == "keys" and ws * k > _MAX_WK:
            return self._off(f"{ws} x {k} candidates do not fit 16 position bits (keys mode)")
        if self.mode == "native" and ws * shard != vocab:
            # masked padding rows (-inf) could tie with real -inf logits in torch.topk's own tie order; keys mode orders those ties by index
            return self._off(f"the vocabulary is padded ({ws} x {shard} rows for {vocab} tokens): use VLLM_DSPARK_TOPK_GATHER=keys")
        if _tp_group().world_size != ws:
            return self._off("the TP group size differs from the head's")
        self.ws, self.shard, self.vocab, self.k, self.rank, self.n_valid = ws, shard, vocab, k, rank, n_valid
        self.ok = True


def _head(lm_head):
    """The head the draft logits use: patch 020's FP8 copy when VLLM_DSPARK_FP8_HEAD=1, else the target's own."""
    try:
        from vllm._dspark_lean import draft_head
    except ImportError:
        return lm_head
    return draft_head(lm_head)


def _local_logits(spec, plan: _Plan, sample_hidden: torch.Tensor) -> torch.Tensor:
    m = spec.model
    hidden = m.model.norm(sample_hidden)
    # the stock compute_logits minus the all-gather: same head GEMM, same soft-cap / scale
    return m.logits_processor(_head(m.lm_head), hidden, skip_gather=True)


def _log_plan(plan: _Plan) -> None:
    if plan.ok:
        logger.info(
            "%s DSpark top-k gather ACTIVE (mode %s): shard %d rows (%d valid), vocab %d, tp %d, k %d, for >= %d draft rows; "
            "all-gather payload int32 [rows, %d] = %.1f KiB/row instead of bf16 [rows, %d] = %.1f KiB/row%s",
            TAG, plan.mode, plan.shard, plan.n_valid, plan.vocab, plan.ws, plan.k, MIN_ROWS, plan.k, plan.k * 4 / 1024,
            plan.shard, plan.shard * 2 / 1024, (f"; shadow check: {plan.warm_checks_left} capture warm-up calls (dummy rows) + the first {plan.checks_left} eager calls on real traffic after the CUDA graphs are captured"
             if plan.checks_left else ""),
        )
    else:
        logger.warning("%s DSpark top-k gather requested but NOT active: %s (stock path)", TAG, plan.reason)


def _post_capture(spec) -> bool:
    """True once the draft model's CUDA graphs are captured: an eager call after that runs on real traffic (a mixed prefill / decode step); the capture
    warm-ups before it all use one dummy hidden state (sample_indices zeroed, sample_idx_mapping -1)."""
    return bool(getattr(getattr(spec, "query_cudagraph_manager", None), "_graphs_captured", False))


def _take_check(plan: _Plan, post_capture: bool) -> str | None:
    """Which shadow check this eager call runs: 'real' (after the graphs are captured; the first CHECK of them), 'warm' (the first two capture warm-ups), or None."""
    if post_capture:
        if plan.checks_left > 0:
            plan.checks_left -= 1
            return "real"
        return None
    if plan.warm_checks_left > 0:
        plan.warm_checks_left -= 1
        return "warm"
    return None


def _shadow_check(spec, plan: _Plan, sample_hidden, num_reqs: int, n_spec: int, values, ids, kind: str = "real") -> None:
    """Eager only: the stock path (full all-gather + torch.topk) against the result just computed; logs, never raises.
    kind 'warm' = a capture warm-up (dummy rows: checks the plumbing and shapes only), 'real' = real traffic."""
    what = "(capture warm-up, dummy rows)" if kind == "warm" else "(real traffic)"
    full = spec.model.compute_draft_logits(sample_hidden).view(num_reqs, n_spec, -1)
    sv, si = full.topk(plan.k, dim=-1)
    same_v = bool(torch.equal(sv.view(torch.int16), values.view(torch.int16)))
    same_i = bool(torch.equal(si, ids))
    if same_v and same_i:
        logger.info("%s shadow check %s %d rows: values, ids and order IDENTICAL to the stock path", TAG, what, num_reqs * n_spec)
        return
    a, _ = torch.sort(si, dim=-1)
    b, _ = torch.sort(ids, dim=-1)
    logger.warning(
        "%s shadow check %s %d rows: DIFFERENT from the stock path (values %s, ids %s, same id SET %s, rows with a different set %d); "
        "expected only at boundary ties of torch.topk (or NaN payloads in keys mode) -- the drafts stay valid, see tests/038/NOTES.md",
        TAG, what, num_reqs * n_spec, "same" if same_v else "differ", "same" if same_i else "differ",
        bool(torch.equal(a, b)), int((a != b).any(dim=-1).sum().item()),
    )


def draft_topk(spec, sample_hidden: torch.Tensor, num_reqs: int, n_spec: int):
    """Replacement for  compute_draft_logits -> view -> topk  in DSparkSpeculator._sample_sequential_topk.

    Returns (dense, values, ids) -- dense: UNINITIALISED bf16 [num_reqs, n_spec, V] (the caller fills it with -inf), values: bf16
    [num_reqs, n_spec, k], ids: int64 [num_reqs, n_spec, k], sorted by value descending -- or None to run the stock path."""
    plan = getattr(spec, "_p038_plan", None)
    if plan is None:
        plan = spec._p038_plan = _Plan(spec)
        _log_plan(plan)
    rows = num_reqs * n_spec
    if not plan.ok or rows < max(1, MIN_ROWS):
        return None
    eager = not _capturing()
    local = _local_logits(spec, plan, sample_hidden)
    if local.shape != (rows, plan.shard) or local.dtype != torch.bfloat16:
        raise RuntimeError(f"{TAG} local logits {tuple(local.shape)} {local.dtype}, expected ({rows}, {plan.shard}) bfloat16")
    if eager and not plan.diag_done:
        plan.diag_done = True
        try:
            rep = tie_order_report(local.device, plan.k, plan.shard, plan.ws, plan.vocab)
            if all(rep.values()):
                logger.info("%s torch.topk tie order is canonical (value desc, index asc) at %d shapes (%s): the results are bit-identical "
                            "to the stock path", TAG, len(rep), ", ".join(sorted(rep)))
            else:
                logger.warning("%s torch.topk tie order is NOT canonical at %s: %s", TAG, [s for s, ok in rep.items() if not ok],
                               "keys mode orders ties canonically, so it differs from the STOCK path at boundary ties (the stock call itself is not canonical); "
                               "drafts stay valid (lossless)" if plan.mode == "keys" else
                               "boundary ties may resolve differently from the stock path (drafts stay valid, lossless); "
                               "VLLM_DSPARK_TOPK_GATHER=keys makes the order canonical")
        except Exception as e:  # noqa: BLE001 -- diagnostic only
            logger.warning("%s tie-order diagnostic failed: %s: %s", TAG, type(e).__name__, e)
    mask_padding(local, plan.n_valid)
    payload = select(plan.mode, local, plan.k)
    gathered = _tp_group().all_gather(payload, dim=0).contiguous()  # [ws * rows, k] int32, rank-major
    values, ids = finish(plan.mode, gathered, plan.ws, rows, plan.k, plan.shard)
    dense = torch.empty((rows, plan.vocab), dtype=local.dtype, device=local.device)
    out = (dense.view(num_reqs, n_spec, plan.vocab), values.view(num_reqs, n_spec, plan.k), ids.view(num_reqs, n_spec, plan.k))
    if eager and (plan.checks_left > 0 or plan.warm_checks_left > 0):
        kind = _take_check(plan, _post_capture(spec))
        if kind:
            _shadow_check(spec, plan, sample_hidden, num_reqs, n_spec, out[1], out[2], kind)
    return out
