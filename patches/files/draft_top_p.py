# SPDX-License-Identifier: Apache-2.0
"""The request's top-p, and optionally a sharper temperature, applied to the DSpark draft distribution (patch 024; installed as
vllm/_draft_top_p.py).

WHY: vLLM samples draft tokens with the request temperature only ("we only consider the temperature and ignore ... top_k and
top_p, for simplicity and performance. While this may slightly degrade the acceptance rate, it does not affect the output
distribution after rejection sampling"). Our requests run top_p 0.95: every draft token sampled outside the target's nucleus is
rejected for sure, so the probability mass the draft puts there is wasted. Truncating the draft distribution to its own top-p
nucleus (at the request temperature) moves that mass onto tokens that can be accepted.
A small drafter is usually less confident than its target; drafting at a lower temperature (request temperature x s, s < 1)
puts more of the draft mass on the tokens the target favours. Whether that raises acceptance is measured, not assumed.

LOSSLESS: speculative verification keeps the output distribution exactly the target's for ANY draft distribution, provided it
is the distribution the drafts were sampled from; the truncated (and sharpened) logits are what gumbel_sample samples AND caches
for the rejection sampler (q). Only acceptance changes. Requests with top_p 1 and s = 1 are unchanged, and greedy requests
(temperature 0) keep their argmax.

WHERE: the DSpark top-k draft path (patch 013, dspark_draft_topk): each sequential draft step has its candidates' corrected
logits scattered into a -inf row. This scales the candidates' logits by 1/s, keeps the smallest set of them whose probability
(at the request temperature) reaches top_p, and sets the rest to -inf.
  VLLM_DSPARK_DRAFT_TOP_P=1              the top-p truncation
  VLLM_DSPARK_DRAFT_SHARPEN=s            draft temperature = request temperature x s, 0.2 <= s <= 2 (default 1: off;
                                         s > 1 flattens the draft distribution)
  VLLM_DSPARK_DRAFT_SHARPEN_RUNTIME=1    s may change at runtime through set_s() (patch 026's control broadcast calls it at
                                         the same drafter step on every TP rank)
  VLLM_DSPARK_DRAFT_SHARPEN_FILE=path    A/B sweeps only, NOT rank-synchronised: s is re-read from this file (one number) when
                                         its mtime changes, so ranks may switch a step apart (their drafts then differ for that
                                         step). Prefer VLLM_DSPARK_DRAFT_SHARPEN_RUNTIME.
                                         Every TP rank samples the drafts itself, so all ranks must use the same s at every
                                         step: change the file on every node only while the engine is idle.
The model runner registers its sampling states before each propose(); that eager call is where the file is checked.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_ENV = "VLLM_DSPARK_DRAFT_TOP_P"
_ON = os.environ.get(_ENV, "0") == "1"
_ITERS = int(os.environ.get("VLLM_DSPARK_DRAFT_TOP_P_ITERS", "16"))
_WARPS = int(os.environ.get("VLLM_DSPARK_DRAFT_TOP_P_WARPS", "4"))
_SHARPEN_FILE = os.environ.get("VLLM_DSPARK_DRAFT_SHARPEN_FILE", "")
_STATES = None
_INV_S: torch.Tensor | None = None  # [1] fp32 on the GPU: 1/s, read by the kernel (graph-safe: updated in place)
_FILE_MTIME: int | None = None


def _check_s(s: float) -> float:
    if not 0.2 <= s <= 2.0:
        raise ValueError(f"[patch-024] draft sharpening s={s} outside [0.2, 2]")
    return s


_S_CUR = _check_s(float(os.environ.get("VLLM_DSPARK_DRAFT_SHARPEN", "1")))  # the s in effect
_RUNTIME = os.environ.get("VLLM_DSPARK_DRAFT_SHARPEN_RUNTIME", "0") == "1"
_SHARPEN = _S_CUR != 1.0 or bool(_SHARPEN_FILE) or _RUNTIME


def _inv_s(device: torch.device) -> torch.Tensor:
    global _INV_S
    if _INV_S is None:
        _INV_S = torch.full((1,), 1.0 / _S_CUR, dtype=torch.float32, device=device)
        logger.info("[patch-024] DSpark draft sharpening ACTIVE: s=%.3f%s", _S_CUR,
                    f" (runtime file {_SHARPEN_FILE})" if _SHARPEN_FILE else "")
    return _INV_S


def _reload_s() -> None:
    """A/B control: when the file's mtime changes, set 1/s from its contents (eager, before the drafter graph replays)."""
    global _FILE_MTIME, _S_CUR
    try:
        st = os.stat(_SHARPEN_FILE)
    except OSError:
        return
    if st.st_mtime_ns == _FILE_MTIME:
        return
    _FILE_MTIME = st.st_mtime_ns
    try:
        with open(_SHARPEN_FILE) as f:
            s = _check_s(float(f.read().strip()))
    except (OSError, ValueError) as e:
        logger.warning("[patch-024] ignoring %s: %s", _SHARPEN_FILE, e)
        return
    _S_CUR = s
    if _INV_S is not None:
        _INV_S.fill_(1.0 / s)
    logger.info("[patch-024] DSpark draft sharpening s=%.3f (from %s)", s, _SHARPEN_FILE)


def set_s(s: float, source: str) -> bool:
    """Runtime s (VLLM_DSPARK_DRAFT_SHARPEN_RUNTIME=1). The caller must call it at the same drafter step on every TP rank:
    every rank drafts with the same s or the ranks' drafts differ. Returns False when sharpening was not built in."""
    global _S_CUR
    if not _SHARPEN:
        return False
    s = _check_s(float(s))
    if s == _S_CUR:
        return True
    _S_CUR = s
    if _INV_S is not None:
        _INV_S.fill_(1.0 / s)
    logger.info("[patch-024] DSpark draft sharpening s=%.3f (from %s)", s, source)
    return True


def register_states(states) -> None:
    """Called by the model runner before propose(): the sampler's per-request states (top_p lives in states.top_p.gpu)."""
    global _STATES
    _STATES = states
    if _SHARPEN_FILE:
        _reload_s()


@triton.jit
def _draft_top_p_kernel(logits_ptr, stride_l, cand_ptr, stride_c, idx_ptr, temp_ptr, top_p_ptr, inv_s_ptr,
                        K: tl.constexpr, BLOCK: tl.constexpr, ITERS: tl.constexpr, HAS_TOP_P: tl.constexpr,
                        SHARPEN: tl.constexpr):
    """One row per program: scale the candidates' logits by 1/s (SHARPEN); then keep the candidates whose probability (at the
    request temperature) is >= tau, with tau the largest threshold whose kept mass still reaches top_p (= the smallest
    top-probability prefix reaching top_p; ties at the boundary kept), found by bisection instead of a sort; write -inf to
    the others."""
    row = tl.program_id(0)
    req = tl.load(idx_ptr + row)
    if req < 0:
        return
    p = 1.0
    if HAS_TOP_P:
        p = tl.load(top_p_ptr + req)
    inv_s = 1.0
    if SHARPEN:
        inv_s = tl.load(inv_s_ptr)
    if (p >= 1.0) & (inv_s == 1.0):
        return
    t = tl.maximum(tl.load(temp_ptr + req).to(tl.float32), 1e-4)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    c = tl.load(cand_ptr + row * stride_c + offs, mask=mask, other=0)
    v = tl.load(logits_ptr + row * stride_l + c, mask=mask, other=float("-inf")).to(tl.float32) * inv_s
    z = v / t
    m = tl.max(z, 0)
    e = tl.where(mask, tl.exp(z - m), 0.0)
    prob = e / tl.sum(e, 0)
    lo = 0.0
    if p < 1.0:
        hi = tl.max(prob, 0)
        for _ in range(ITERS):
            mid = (lo + hi) * 0.5
            mass = tl.sum(tl.where(prob >= mid, prob, 0.0), 0)
            keep_mid = mass >= p
            lo = tl.where(keep_mid, mid, lo)
            hi = tl.where(keep_mid, hi, mid)
    drop = mask & (prob < lo)
    out = tl.where(drop, float("-inf"), v)
    if SHARPEN:
        tl.store(logits_ptr + row * stride_l + c, out.to(logits_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(logits_ptr + row * stride_l + c, out.to(logits_ptr.dtype.element_ty), mask=drop)


def apply_draft_top_p(logits: torch.Tensor, cand: torch.Tensor, idx: torch.Tensor, temperature: torch.Tensor) -> None:
    """In place: per row, scale the `cand` logits by 1/s (when sharpening), keep the smallest set of `cand` (by probability at
    the request temperature) whose mass reaches the request's top_p, and set the other candidates' logits to -inf.
    logits [B, V] (row stride free), cand [B, k], idx [B] request slots (-1 = padded row, left alone), temperature
    [max_num_reqs]. One Triton launch (no sort)."""
    if not (_ON or _SHARPEN) or _STATES is None:
        return
    top_p = getattr(_STATES, "top_p", None) if _ON else None
    if top_p is None and not _SHARPEN:
        return
    rows, k = cand.shape
    if rows == 0:
        return
    assert logits.stride(1) == 1 and cand.stride(1) == 1
    inv_s = _inv_s(logits.device) if _SHARPEN else temperature
    _draft_top_p_kernel[(rows,)](
        logits, logits.stride(0), cand, cand.stride(0), idx, temperature,
        top_p.gpu if top_p is not None else temperature, inv_s,
        K=k, BLOCK=triton.next_power_of_2(k), ITERS=_ITERS, HAS_TOP_P=top_p is not None, SHARPEN=_SHARPEN,
        num_warps=_WARPS,
    )
