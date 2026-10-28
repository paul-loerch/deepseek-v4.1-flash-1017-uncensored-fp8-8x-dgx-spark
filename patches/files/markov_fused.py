# SPDX-License-Identifier: Apache-2.0
# [patch-029 markov-fused] installed as vllm/_markov_fused.py by 029-markov-fused.sh (opt-in VLLM_DSPARK_MARKOV_FUSED=1).
"""DSpark top-k Markov bias in one Triton launch (patch 013's apply_markov_bias_gathered without temporaries).

WHY. With dspark_draft_topk (patch 013) every one of the 5 sequential draft steps corrects the 2,048 candidate logits of each
request by the Markov bias markov_w2[cand] . markov_embed(prev). Patch 013 does it with torch ops:
    weight = markov_w2[index]            # gather -> [B, k, r] bf16      (B x 2048 x 256 x 2 B)
    weight.float()                       # copy   -> [B, k, r] fp32      (x 2)
    torch.bmm(weight, embed[..., None])  # cuBLAS fp32 GEMV over [B, k, r]
    values.float() + scale * bias, scatter_ into the -inf filled logits row
At 64 requests that is ~0.2 GB of temporaries per draft step, ~1 GB per drafter run: the 64-user profile of b042 shows the
drafter graph at ~29 ms of kernels, ~10 ms of them in these gathers / copies / GEMVs (1 user: negligible).

WHAT. One launch per draft step, grid (B, cdiv(k, BLOCK_K)): each program loads BLOCK_K candidate ids, reads their markov_w2
rows (256 bf16 each, contiguous) once, forms the fp32 dot product with the request's embedding, adds scale * bias to the fp32
candidate value and stores the result (in the logits' dtype) at logits[b, id]. Reads ~k x r x 2 B per request, writes k values.

NUMERICS. The same fp32 operations per candidate (bf16 -> fp32 products, fp32 sum, value + scale * bias, cast to the logits'
dtype) except the order of the 256-term sum (cuBLAS GEMV vs a Triton tree reduction): a corrected logit can differ in its last
fp32 bit. That moves only the DRAFT distribution; the drafts are still sampled from, and verified against, the logits this
function writes (the rejection sampler reads the same cached draft logits), so the target's output distribution is unchanged
(lossless, like patch 013 itself). The GPU unit test (tests/029/markov_fused_test.py) bounds the difference.
"""
from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

TAG = "[patch-029]"
ENV = "VLLM_DSPARK_MARKOV_FUSED"


def _parse(raw: str | None) -> bool:
    s = (raw or "").strip().lower()
    if s in ("", "0", "off", "false", "no"):
        return False
    if s in ("1", "on", "true", "yes"):
        return True
    raise ValueError(f"{TAG} {ENV}={raw!r} must be unset/0 (off) or 1 (on)")


ENABLED = _parse(os.environ.get(ENV))
BLOCK_K = 16
NUM_WARPS = 4


@triton.jit
def _markov_gathered_kernel(
    w_ptr,  # [V, R] markov_w2 (contiguous)
    e_ptr,  # [B, R] markov embedding of the previous token, row stride e_s0
    e_s0,
    v_ptr,  # [B, k] candidate base logits, row stride v_s0
    v_s0,
    i_ptr,  # [B, k] candidate ids (int64), row stride i_s0
    i_s0,
    o_ptr,  # [B, V] logits row of this draft step, row stride o_s0, column stride 1
    o_s0,
    scale,
    K,
    R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    m = offs < K
    idx = tl.load(i_ptr + b * i_s0 + offs, mask=m, other=0).to(tl.int64)
    r = tl.arange(0, R)
    e = tl.load(e_ptr + b * e_s0 + r).to(tl.float32)
    w = tl.load(w_ptr + idx[:, None] * R + r[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    bias = tl.sum(w * e[None, :], axis=1)
    v = tl.load(v_ptr + b * v_s0 + offs, mask=m, other=0.0).to(tl.float32)
    out = v + scale * bias
    tl.store(o_ptr + b * o_s0 + idx, out.to(o_ptr.dtype.element_ty), mask=m)


def markov_bias_gathered(
    weight: torch.Tensor,
    markov_embed: torch.Tensor,
    logits: torch.Tensor,
    values: torch.Tensor,
    index: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """logits[b, index[b, j]] = values[b, j] + scale * <weight[index[b, j]], markov_embed[b]> (fp32), in place; returns logits.

    logits: [B, V], -inf filled by the caller, unit column stride; values / index: [B, k]; markov_embed: [B, R]; weight [V, R].
    """
    B, k = index.shape
    R = weight.shape[1]
    assert weight.stride() == (R, 1), f"{TAG} markov_w2 must be contiguous, got strides {weight.stride()}"
    assert logits.stride(1) == 1 and values.stride(1) == 1 and index.stride(1) == 1 and markov_embed.stride(1) == 1
    assert markov_embed.shape == (B, R) and values.shape == (B, k) and logits.shape[0] == B
    if B == 0:
        return logits
    _markov_gathered_kernel[(B, triton.cdiv(k, BLOCK_K))](
        weight, markov_embed, markov_embed.stride(0), values, values.stride(0), index, index.stride(0),
        logits, logits.stride(0), float(scale), k, R=R, BLOCK_K=BLOCK_K, num_warps=NUM_WARPS,
    )
    return logits


def reference(weight, markov_embed, logits, values, index, scale):
    """Patch 013's torch implementation (for tests)."""
    w = weight[index]
    bias = torch.bmm(w.float(), markov_embed.float().unsqueeze(-1)).squeeze(-1)
    corrected = values.float() + scale * bias
    return logits.scatter_(1, index, corrected.to(logits.dtype))
