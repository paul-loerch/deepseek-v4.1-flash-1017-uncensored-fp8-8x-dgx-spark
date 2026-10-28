# SPDX-License-Identifier: Apache-2.0
"""Masked verification for DSpark on GB10 (patch 026; installed as vllm/_mv.py). Opt-in: VLLM_GB10_MV=auto | fixed:<c> | off (auto = exact threshold policy on conf_1).

WHY. A single-user decode step verifies 1 + 5 tokens. Every target MoE layer streams the experts of all 6 tokens (~35 of
384 experts, ~6 more per token), ~44 % of the step (14 ms of 32 on GB10). A draft that will be rejected costs the same
expert reads as one that is accepted: at an acceptance length of ~4.2, ~1.8 of the 6 tokens are wasted per step.

WHAT. After DSpark drafts, each request gets a cap c in [0, 5], the number of drafts to verify, from the confidence of its
FIRST draft (mtp.2.confidence_head at the anchor position): cap = #{k : thresholds[k] <= conf_1} (default 0, .3, .6, .7, .8
-> caps 1..5, fitted on b037 per-step logs against the measured per-cap step costs). In the next target forward the tokens
after the cap keep their place in the batch (CUDA graph shapes do not change) but take their request anchor's (first
token's) experts in every target MoE layer, so they add no expert reads. Verification treats drafts after the cap as absent
(vLLM's -1 placeholder): block verification of a c-token block, exactly as with c speculative tokens.

EXACTNESS. Tokens 0..c compute what they compute without the cap: attention is causal, every other op is per token, and
the MoE kernel computes each token's rows independently (checked once on the live layer at the first eligible step, every
rank voting; the b12x kernel itself is not bit-deterministic run to run, so the check accepts its own run-to-run noise; on
failure MV switches off for the process). Verification reads only logits rows 0..c. The cap must not depend on this step's
draft tokens: under block verification the acceptance of draft j depends on the block length, so a length chosen from
the drafted tokens (e.g. from conf_2.., which the confidence head computes from the previous draft token) would bias the
output. conf_1 depends only on the anchor and the drafter backbone (no sampled draft), the verification draws are not
involved, so the block length is fixed before the drafts are drawn and the output distribution is the target model's. Every TP rank computes the same cap from the
same replicated confidences; mode changes at runtime (VLLM_GB10_MV_CTL) are broadcast from rank 0 at a fixed step count so
all ranks switch at the same step.

CHAIN POLICY (mode "chain"). The confidence head also runs along the drafter's argmax chain: the same backbone states with
the Markov embedding of the most likely previous draft (computed from the base top-k logits and the Markov bias, no sampled
token involved). Those five confidences are fixed before the drafts are drawn too, so a cap chosen from their survival with
the measured per-cap step costs (argmax of expected tokens / cost) is exact as well; ~0.1 ms of extra drafter work per step.

ELIGIBLE STEPS. Pure decode, every request with exactly K drafts, uniform 1 + K tokens per request, at most
VLLM_GB10_MV_MAX_REQS requests (default 1). Other steps run unchanged (identity remap, no cap).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

TAG = "[patch-026 mv]"
ENV = "VLLM_GB10_MV"                       # off | auto | fixed:<c>
ENV_MAX_REQS = "VLLM_GB10_MV_MAX_REQS"
ENV_CTL = "VLLM_GB10_MV_CTL"               # runtime control file (rank 0 reads it; "auto [thr=..]" / "fixed:c" / "off")
ENV_SYNC = "VLLM_GB10_MV_SYNC_STEPS"       # broadcast period of the control state, in drafter steps
ENV_LOG = "VLLM_GB10_MV_LOG"               # path prefix: rank 0 writes <prefix>-rank0.npy (per-step records)

_OFF, _FIXED, _AUTO, _CHAIN = 0, 1, 3, 4
_DEFAULT_COSTS = (22.51, 25.14, 27.45, 29.50, 31.23, 32.91)   # b037: step ms with 0..5 drafts verified (fixed:c medians)
_DEFAULT_THR = (0.0, 0.3, 0.6, 0.7, 0.8)    # b037 fit: conf_1 <.3 -> 1 draft, <.6 -> 2, <.7 -> 3, <.8 -> 4, else 5
_LOG_COLS = 16   # conf_1..5, cap, num_sampled, step_on, t_sample (s), mode, argmax-chain conf_1..5, draft sharpen s


def _parse(text: str) -> tuple[int, int, dict]:
    """'auto [thr=0,.3,.6,.7,.8]' | 'chain [costs=c0,..,c5]' | 'fixed:3' | 'off', each with an optional sharpen=s (patch 024
    draft temperature factor, default 1) -> (kind, fixed_c, params)."""
    parts = text.strip().split()
    if not parts:
        return _OFF, 0, {}
    head, params = parts[0].lower(), {}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        k = k.strip().lower()
        params[k] = [float(t) for t in v.split(",")] if k in ("thr", "costs") else float(v)
    if head == "off":
        return _OFF, 0, params
    if head == "auto":
        return _AUTO, 0, params
    if head == "chain":
        return _CHAIN, 0, params
    if head.startswith("fixed:"):
        return _FIXED, int(head.split(":", 1)[1]), params
    raise ValueError(f"{TAG} cannot parse mode {text!r} (off | auto | fixed:<c>)")


@triton.jit
def _policy_kernel(conf_ptr, conf_stride, chain_ptr, chain_stride, idx_ptr, cap_ptr, conf_state_ptr, cs_stride,
                   chain_state_ptr, thr_ptr, cost_ptr, FIXED_C, KIND: tl.constexpr, K: tl.constexpr):
    """One program per request of the drafter batch: cap[req_state], conf_state[req_state], chain_state[req_state].
    KIND 0: cap K; 1: cap FIXED_C; 3: cap = #{k < K : thr[k] <= conf_1}; 4: argmax_c (1 + sum_{i<=c} S_i) / cost[c] with
    S_i the survival of the argmax-chain confidences. conf_1 and the chain confidences do not depend on this step's sampled
    drafts (see EXACTNESS); conf_2.. do and are only recorded."""
    b = tl.program_id(0)
    r = tl.load(idx_ptr + b).to(tl.int64)
    best_c = tl.zeros((), tl.int32) + K
    if KIND == 1:
        best_c = tl.zeros((), tl.int32) + FIXED_C
    c1 = tl.load(conf_ptr + b * conf_stride).to(tl.float32)
    if KIND == 3:
        best_c = tl.zeros((), tl.int32)
        for k in tl.static_range(K):
            best_c += (tl.load(thr_ptr + k) <= c1).to(tl.int32)
    s = tl.full((), 1.0, tl.float32)
    e = tl.full((), 1.0, tl.float32)
    best = 1.0 / tl.load(cost_ptr)
    if KIND == 4:
        best_c = tl.zeros((), tl.int32)
    for i in tl.static_range(K):
        tl.store(conf_state_ptr + r * cs_stride + i, tl.load(conf_ptr + b * conf_stride + i).to(tl.float32))
        ch = tl.load(chain_ptr + b * chain_stride + i).to(tl.float32)
        tl.store(chain_state_ptr + r * cs_stride + i, ch)
        if KIND == 4:
            s = s * tl.minimum(tl.maximum(ch, 0.0), 1.0)
            e = e + s
            sc = e / tl.load(cost_ptr + i + 1)
            better = sc > best
            best = tl.where(better, sc, best)
            best_c = tl.where(better, i + 1, best_c)
    tl.store(cap_ptr + r, best_c)


@triton.jit
def _src_kernel(src_ptr, idx_ptr, cap_ptr, n_tok, n_pad, Q: tl.constexpr, ON: tl.constexpr, BLOCK: tl.constexpr):
    """src[t] = the token whose experts token t uses: its request's first token if t is past the request's cap."""
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = t < n_pad
    out = t
    if ON:
        valid = m & (t < n_tok)
        b = t // Q
        j = t - b * Q
        r = tl.load(idx_ptr + b, mask=valid, other=0).to(tl.int64)
        c = tl.load(cap_ptr + r, mask=valid, other=Q)
        out = tl.where(valid & (j > c), b * Q, t)
    tl.store(src_ptr + t, out.to(tl.int32), mask=m)


@triton.jit
def _remap_kernel(ids_ptr, ids_stride, w_ptr, w_stride, src_ptr, n, TOPK: tl.constexpr, TOPK_P: tl.constexpr):
    """In place: rows whose src differs take the src row's expert ids and weights (src rows are never remapped)."""
    t = tl.program_id(0)
    s = tl.load(src_ptr + t)
    if s != t:
        k = tl.arange(0, TOPK_P)
        mk = k < TOPK
        ids = tl.load(ids_ptr + s * ids_stride + k, mask=mk)
        w = tl.load(w_ptr + s * w_stride + k, mask=mk)
        tl.store(ids_ptr + t * ids_stride + k, ids, mask=mk)
        tl.store(w_ptr + t * w_stride + k, w, mask=mk)


class _State:
    def __init__(self) -> None:
        self.enabled = False          # buffers exist, hooks active
        self.remap = False            # target routers marked (mode may become on)
        self.kind, self.fixed_c, self.params = _OFF, 0, {}
        self.max_reqs = 1
        self.K = 0
        self.cap = self.conf_state = self.chain_state = self.src = self.thr = self.costs = None
        self.chain = None              # [max_num_reqs, K] argmax-chain confidences, written inside the drafter graph
        self.thr_list: list[float] | None = None
        self.costs_list: list[float] | None = None
        self.step_on = False
        self.dirty = 0                # src entries [0, dirty) may be non-identity
        self.steps = 0
        self.sync = 0
        self.ctl = None
        self.ctl_text = None
        self.verified = False
        self.self_test_note = ""
        self.disabled_reason: str | None = None
        self.runner = None
        self.log_prefix = None
        self.log_rows = None
        self.log_n = 0
        self.log_stage = None
        self.t0 = time.perf_counter()
        self.sharpen = 1.0


_ST = _State()


def _apply_mode(text: str, source: str) -> None:
    kind, fixed_c, params = _parse(text)
    if kind == _FIXED and not 0 <= fixed_c <= _ST.K:
        raise ValueError(f"{TAG} fixed:{fixed_c} outside [0, {_ST.K}]")
    if kind != _OFF and not _ST.remap:
        logger.warning("%s mode %r ignored: the routers were not marked at load (start with %s set)", TAG, text, ENV)
        kind = _OFF
    _ST.kind, _ST.fixed_c, _ST.params = kind, fixed_c, params
    thr = params.get("thr", list(_DEFAULT_THR))
    if len(thr) != _ST.K or any(b < a for a, b in zip(thr, thr[1:])):
        raise ValueError(f"{TAG} thr needs {_ST.K} ascending values, got {thr}")
    _ST.thr_list = [float(t) for t in thr]
    costs = params.get("costs", list(_DEFAULT_COSTS))
    if len(costs) != _ST.K + 1 or min(costs) <= 0:
        raise ValueError(f"{TAG} costs needs {_ST.K + 1} positive values, got {costs}")
    _ST.costs_list = [float(c) for c in costs]
    if _ST.thr is not None:
        _ST.thr.copy_(torch.tensor(_ST.thr_list, dtype=torch.float32))
        _ST.costs.copy_(torch.tensor(_ST.costs_list, dtype=torch.float32))
    sharpen = float(params.get("sharpen", 1.0))       # draft temperature factor (patch 024), the same step on every rank
    _ST.sharpen = sharpen
    try:
        import vllm._draft_top_p as _dtp

        if hasattr(_dtp, "set_s") and not _dtp.set_s(sharpen, source) and sharpen != 1.0:
            logger.warning("%s sharpen=%.3f ignored: start with VLLM_DSPARK_DRAFT_SHARPEN_RUNTIME=1", TAG, sharpen)
    except ImportError:
        pass
    if kind == _CHAIN and _ST.chain is None:
        logger.warning("%s mode chain needs the argmax-chain confidences (start with %s or %s set): using auto", TAG, ENV_CTL,
                       ENV_LOG)
        kind = _ST.kind = _AUTO
    logger.info("%s mode %s (from %s): kind=%d fixed_c=%d thr=%s costs=%s max_reqs=%d", TAG, text.strip(), source, kind,
                fixed_c, ",".join(f"{t:.3f}" for t in _ST.thr_list), ",".join(f"{c:.2f}" for c in _ST.costs_list),
                _ST.max_reqs)


def register(runner) -> None:
    """Called once after the target model and the drafter are loaded."""
    mode = os.environ.get(ENV, "off")
    ctl = os.environ.get(ENV_CTL) or None
    log = os.environ.get(ENV_LOG) or None
    if _parse(mode)[0] == _OFF and ctl is None and log is None:
        return
    spec = getattr(runner, "speculator", None)
    K = int(getattr(runner, "num_speculative_steps", 0) or 0)
    if spec is None or K <= 0:
        logger.warning("%s no speculator: masked verification stays off", TAG)
        return
    dev = runner.device
    _ST.K = K
    _ST.max_reqs = int(os.environ.get(ENV_MAX_REQS, "1"))
    _ST.sync = int(os.environ.get(ENV_SYNC, "128" if ctl else "0"))
    _ST.ctl = ctl
    _ST.cap = torch.full((runner.max_num_reqs,), K, dtype=torch.int32, device=dev)
    _ST.conf_state = torch.ones((runner.max_num_reqs, K), dtype=torch.float32, device=dev)
    _ST.src = torch.arange(runner.max_num_tokens, dtype=torch.int32, device=dev)
    _ST.thr = torch.zeros(K, dtype=torch.float32, device=dev)
    _ST.costs = torch.ones(K + 1, dtype=torch.float32, device=dev)
    _ST.chain_state = torch.ones((runner.max_num_reqs, K), dtype=torch.float32, device=dev)
    if getattr(spec, "use_confidence_head", False) and getattr(spec, "_draft_topk", None):
        # the drafter graph (captured after this) computes the argmax-chain confidences into this buffer
        _ST.chain = torch.ones((runner.max_num_reqs, K), dtype=torch.float32, device=dev)
        spec._p026_chain = True
    _ST.runner = runner
    _ST.enabled = True
    if not getattr(spec, "use_confidence_head", False):
        logger.warning("%s the drafter computes no confidences (no confidence head?): auto mode unavailable", TAG)
    if _parse(mode)[0] != _OFF or ctl is not None:
        from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter

        import re

        n_layers = runner.model_config.hf_config.get_text_config().num_hidden_layers
        n, seen = 0, set()
        for name, m in runner.model.named_modules():
            hit = re.search(r"(?:^|\.)layers\.(\d+)\.", name + ".")
            if hit is None or int(hit.group(1)) >= n_layers:   # target decoder layers only, never a drafter layer
                continue
            for r in (getattr(getattr(m, "runner", None), "router", None), getattr(m, "router", None)):
                if isinstance(r, FusedMoERouter) and id(r) not in seen:
                    seen.add(id(r))
                    r._p026_mv = True
                    n += 1
        _ST.remap = n > 0
        logger.info("%s %d target MoE routers remap capped draft tokens (target layers < %d; drafter untouched)", TAG,
                    n, n_layers)
    if log:
        from vllm.distributed.parallel_state import get_tp_group

        if get_tp_group().rank_in_group == 0:
            _ST.log_prefix = log
            _ST.log_rows = torch.zeros((1 << 18, _LOG_COLS), dtype=torch.float64, pin_memory=True)
            _ST.log_stage = torch.zeros(_LOG_COLS, dtype=torch.float64, device=dev)
    _apply_mode(mode, ENV)


def remap_topk(topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inside every target MoE layer (CUDA-graph safe: reads the persistent src buffer)."""
    if not _ST.enabled:
        return topk_weights, topk_ids
    n, topk = topk_ids.shape
    _remap_kernel[(n,)](topk_ids, topk_ids.stride(0), topk_weights, topk_weights.stride(0), _ST.src, n,
                        TOPK=topk, TOPK_P=triton.next_power_of_2(topk), num_warps=1)
    return topk_weights, topk_ids


def _eligible(ib) -> bool:
    K = _ST.K
    nd = ib.num_draft_tokens_per_req
    return (0 < ib.num_reqs <= _ST.max_reqs and not ib.has_prefill and nd is not None and len(nd) == ib.num_reqs
            and bool((nd == K).all()) and ib.num_tokens == ib.num_reqs * (K + 1) and ib.num_tokens < 1024)


def _self_test(runner) -> str:
    """Bitwise: a live target MoE layer on 1 + K tokens, with and without tokens 3.. remapped to token 0."""
    from vllm.forward_context import set_forward_context

    layers = [m for m in runner.model.modules() if type(m).__name__ == "DeepseekV4MoE"]
    if not layers:
        return "no DeepseekV4MoE layer found"
    ffn = layers[min(3, len(layers) - 1)]
    Q, keep = _ST.K + 1, 3
    hidden = ffn.hidden_size
    g = torch.Generator(device=_ST.src.device).manual_seed(26)
    x = (torch.randn(Q, hidden, generator=g, device=_ST.src.device) * 0.5).to(torch.bfloat16)
    ids = torch.zeros(Q, dtype=torch.int32, device=_ST.src.device)
    src = _ST.src
    ident = torch.arange(Q, dtype=torch.int32, device=src.device)
    with set_forward_context(None, runner.vllm_config, num_tokens=Q):
        src[:Q].copy_(ident)
        out_full = ffn(x.clone(), ids).clone()
        out_full2 = ffn(x.clone(), ids).clone()
        src[keep:Q].fill_(0)
        out_cap = ffn(x.clone(), ids).clone()
        src[:Q].copy_(ident)
    torch.cuda.synchronize()
    if not torch.isfinite(out_full).all():
        return "non-finite MoE output in the self-test"
    if torch.equal(out_full[keep:], out_cap[keep:]):
        return "remap had no effect on the capped rows"
    if torch.equal(out_full, out_full2):          # deterministic kernel: kept rows must be bitwise equal
        if not torch.equal(out_full[:keep], out_cap[:keep]):
            d = (out_full[:keep].float() - out_cap[:keep].float()).abs().max().item()
            return f"kept rows differ (max abs {d:.3e}) on a deterministic kernel"
        _ST.self_test_note = "bitwise"
        return "ok"
    noise = (out_full.float() - out_full2.float()).abs().max().item()
    d = (out_full[:keep].float() - out_cap[:keep].float()).abs().max().item()
    if d > 2.0 * noise:
        return f"kept rows differ by {d:.3e}, more than twice the kernel's own run-to-run noise {noise:.3e}"
    _ST.self_test_note = f"kernel not bitwise-deterministic (run-to-run {noise:.3e}); kept rows within it ({d:.3e})"
    return "ok"


def _verify_once(runner) -> None:
    from vllm.distributed.parallel_state import get_tp_group

    try:
        verdict = _self_test(runner)
    except Exception as e:  # noqa: BLE001 - every rank must still vote
        verdict = f"{type(e).__name__}: {e}"
    import torch.distributed as dist

    tp = get_tp_group()
    votes = [None] * tp.world_size
    dist.all_gather_object(votes, verdict, group=tp.cpu_group)
    bad = [(r, v) for r, v in enumerate(votes) if v != "ok"]
    _ST.verified = True
    if bad:
        _ST.disabled_reason = "; ".join(f"rank {r}: {v}" for r, v in bad)[:500]
        _ST.kind = _OFF
        _ST.remap = False
        logger.error("%s self-test FAILED -> masked verification off for this process: %s", TAG, _ST.disabled_reason)
    else:
        logger.info("%s ACTIVE: self-test passed on all %d ranks (live MoE layer, capped rows remapped; kept rows: %s)",
                    TAG, len(votes), _ST.self_test_note)


def before_forward(ib, dummy_run: bool) -> None:
    """Before the target forward (eager): the per-token expert source for this step."""
    if not _ST.enabled:
        return
    on = (not dummy_run) and _ST.kind != _OFF and _ST.remap and _eligible(ib)
    if on and not _ST.verified:
        _verify_once(_ST.runner)
        on = _ST.kind != _OFF
    n_pad = ib.num_tokens_after_padding
    if on:
        _src_kernel[(triton.cdiv(n_pad, 256),)](_ST.src, ib.idx_mapping, _ST.cap, ib.num_tokens, n_pad, Q=_ST.K + 1,
                                               ON=True, BLOCK=256)
        _ST.dirty = max(_ST.dirty, n_pad)
    elif _ST.dirty:
        _src_kernel[(triton.cdiv(_ST.dirty, 256),)](_ST.src, ib.idx_mapping, _ST.cap, 0, _ST.dirty, Q=_ST.K + 1,
                                                   ON=False, BLOCK=256)
        _ST.dirty = 0
    _ST.step_on = on


def invalid_drafts(ib, invalid: torch.Tensor | None) -> torch.Tensor | None:
    """Drafts past the request's cap are absent for verification (they were computed with borrowed experts)."""
    if not (_ST.enabled and _ST.step_on):
        return invalid
    mv = ib.expanded_local_pos > _ST.cap[ib.expanded_idx_mapping]
    return mv if invalid is None else (invalid | mv)


def _sync_ctl() -> None:
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    text = None
    if tp.rank_in_group == 0:
        try:
            with open(_ST.ctl) as f:
                text = f.read().strip() or None
        except OSError:
            text = None
    text = tp.broadcast_object(text, src=0)
    if text is not None and text != _ST.ctl_text:
        _ST.ctl_text = text
        try:
            _apply_mode(text, f"{_ST.ctl} at drafter step {_ST.steps}")
        except Exception as e:  # noqa: BLE001 - same text on every rank: every rank refuses it alike
            logger.error("%s control text %r refused: %s", TAG, text, e)


def after_propose(speculator, ib) -> None:
    """After DSpark drafted for the next step: caps from the drafts' confidences (eager, on the GPU)."""
    if not _ST.enabled:
        return
    _ST.steps += 1
    if _ST.sync and _ST.ctl and _ST.steps % _ST.sync == 0:
        _sync_ctl()
    n = ib.num_reqs
    if n == 0:
        return
    kind = _ST.kind
    if kind in (_AUTO, _CHAIN) and not getattr(speculator, "use_confidence_head", False):
        kind = _OFF
    conf = speculator.draft_token_confidence_probs
    chain = _ST.chain if _ST.chain is not None else conf
    _policy_kernel[(n,)](conf, conf.stride(0), chain, chain.stride(0), ib.idx_mapping, _ST.cap, _ST.conf_state,
                         _ST.conf_state.stride(0), _ST.chain_state, _ST.thr, _ST.costs, _ST.fixed_c, KIND=kind, K=_ST.K,
                         num_warps=1)


def log_step(ib, num_sampled: torch.Tensor | None) -> None:
    """Rank 0, VLLM_GB10_MV_LOG: one record per step for the batch's first request (async copy, no sync)."""
    if _ST.log_rows is None or num_sampled is None or ib.num_reqs == 0:
        return
    i = _ST.log_n
    if i >= _ST.log_rows.shape[0]:
        return
    K = _ST.K
    st = _ST.log_stage
    r = ib.idx_mapping[:1]
    st[:K].copy_(_ST.conf_state[r][0])
    st[K].copy_(_ST.cap[r][0])
    st[K + 1].copy_(num_sampled[0])
    st[K + 2].fill_(1.0 if _ST.step_on else 0.0)
    st[K + 3].fill_(time.perf_counter() - _ST.t0)
    st[K + 4].fill_(float(_ST.kind * 10 + (_ST.fixed_c if _ST.kind == _FIXED else 0)))
    st[K + 5:K + 5 + K].copy_(_ST.chain_state[r][0])
    st[2 * K + 5].fill_(_ST.sharpen)
    _ST.log_rows[i].copy_(st, non_blocking=True)
    _ST.log_n = i + 1
    if _ST.log_n % 2048 == 0:
        torch.cuda.current_stream().synchronize()
        np.save(f"{_ST.log_prefix}-rank0.npy", _ST.log_rows[: _ST.log_n].numpy())


def chain_confidence(spec, num_reqs: int, sample_hidden: torch.Tensor, base_values: torch.Tensor,
                     draft_indices: torch.Tensor) -> None:
    """Inside the drafter graph, before the drafts are sampled: the confidence head along the argmax chain (anchor, then
    at each position the candidate with the highest Markov-corrected base logit). Uses no sampled draft, so it is fixed
    before the drafts are drawn. Writes _ST.chain[:num_reqs]."""
    if _ST.chain is None:
        return
    m = spec.model
    n_spec = spec.num_speculative_steps
    scale = m.logits_processor.scale
    w2 = m.model.markov_head.markov_w2.weight
    prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    embeds = []
    for i in range(n_spec):
        me = m.markov_embed(prev)
        embeds.append(me)
        idx = draft_indices[:, i]
        bias = torch.bmm(w2[idx].float(), me.float().unsqueeze(-1)).squeeze(-1)
        corrected = base_values[:, i].float() + scale * bias
        prev = idx.gather(1, corrected.argmax(dim=1, keepdim=True)).squeeze(1)
    conf = m.compute_confidence(sample_hidden, torch.stack(embeds, dim=1).flatten(0, 1))
    _ST.chain[:num_reqs].copy_(conf.view(num_reqs, n_spec))
