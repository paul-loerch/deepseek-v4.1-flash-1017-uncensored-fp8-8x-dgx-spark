# SPDX-License-Identifier: Apache-2.0
"""DSpark drafter cost cuts (patch 020; installed as vllm/_dspark_lean.py). Drafter only: the draft tokens and their
probabilities change slightly, the target's verification does not, so outputs stay exactly the target's distribution
(block verification / rejection sampling is lossless for any draft distribution); only acceptance can move.

WHY (b023 profile, one user, 33.9 ms per step): the DSpark drafter graph takes 3.05 ms per step, of which
  - the draft logits read the target's BF16 lm_head (16,192 x 5,120 per rank, 166 MB): 736 us per step;
  - main_proj (15,360 -> 5,120, MXFP8, 78.6 MB) is a ReplicatedLinear, so every rank streams all of it: ~320 us.

  VLLM_DSPARK_FP8_HEAD=1      the draft logits use an MXFP8 copy of the lm_head shard (E8M0 scales per 32 inputs, the
                              checkpoint's own format family), through the same Marlin MXFP8 kernel as the dense layers
                              (~2x fewer bytes). Built lazily on the first eager call, after the proposer has attached
                              the target's lm_head; the target's BF16 lm_head is unchanged.
  VLLM_DSPARK_MAIN_PROJ_TP=1  main_proj becomes column-parallel (each rank reads 1/8 of the weights) with an all-gather
                              of the [T, 5,120] output (RoCE all-gather via patch 018). Each output column is the same dot
                              product as before.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENV_HEAD = "VLLM_DSPARK_FP8_HEAD"
_ENV_TP = "VLLM_DSPARK_MAIN_PROJ_TP"
_HEADS: dict[int, "_Fp8DraftHead"] = {}
_BLOCK = 32
_E4M3_MAX = 448.0


def make_main_proj(replicated_cls, input_size: int, output_size: int, *, quant_config, prefix: str):
    """main_proj as before (ReplicatedLinear), or column-parallel with a gathered output when VLLM_DSPARK_MAIN_PROJ_TP=1."""
    if os.environ.get(_ENV_TP, "0") != "1":
        return replicated_cls(input_size, output_size, bias=False, return_bias=False, quant_config=quant_config,
                              prefix=prefix)
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    logger.info("[patch-020] DSpark main_proj column-parallel (%d -> %d, output all-gathered)", input_size, output_size)
    return ColumnParallelLinear(input_size, output_size, bias=False, gather_output=True, return_bias=False,
                                quant_config=quant_config, prefix=prefix)


def _to_mxfp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [N, K] -> (FP8-E4M3 [N, K], E8M0 uint8 [N, K/32]); the block exponent is chosen so nothing saturates."""
    n, k = w.shape
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty((n, k // _BLOCK), dtype=torch.uint8, device=w.device)
    for i in range(0, n, 2048):
        blk = w[i:i + 2048].float().view(-1, k // _BLOCK, _BLOCK)
        amax = blk.abs().amax(dim=-1, keepdim=True).clamp(min=2.0 ** -120)
        e = torch.ceil(torch.log2(amax / _E4M3_MAX)).clamp(-127, 127)
        q[i:i + 2048] = (blk / torch.exp2(e)).view(-1, k).to(torch.float8_e4m3fn)
        s[i:i + 2048] = (e + 127).view(-1, k // _BLOCK).to(torch.uint8)
    return q, s


class _Apply:
    """The quant_method LogitsProcessor calls: lm_head.quant_method.apply(lm_head, x, bias=...)."""

    def __init__(self, kernel) -> None:
        self.kernel = kernel

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class _Fp8DraftHead(torch.nn.Module):
    """Stand-in for the vocab-parallel lm_head shard: same padded rows, same tp_size, MXFP8 weights through Marlin."""

    def __init__(self, src) -> None:
        super().__init__()
        from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
        from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel

        w = src.weight.data
        n, k = w.shape
        if k % _BLOCK:
            raise ValueError(f"[patch-020] lm_head input size {k} is not a multiple of {_BLOCK}")
        q, s = _to_mxfp8(w)
        self.weight = torch.nn.Parameter(q, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(s, requires_grad=False)
        self.output_size_per_partition = n
        self.input_size_per_partition = k
        self.output_size = n
        self.input_size = k
        self.params_dtype = w.dtype
        self.orig_dtype = w.dtype
        self.tp_size = src.tp_size
        kernel = MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
        kernel.process_weights_after_loading(self)
        self.quant_method = _Apply(kernel)
        # Self-check against the BF16 head on a fixed random input: report, and refuse a broken conversion.
        g = torch.Generator(device=w.device).manual_seed(20)
        x = torch.randn(6, k, generator=g, device=w.device, dtype=w.dtype)
        ref = torch.nn.functional.linear(x, w).float()
        got = self.quant_method.apply(self, x).float()
        cos = torch.nn.functional.cosine_similarity(ref, got, dim=-1).min().item()
        top1 = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
        if not cos > 0.99:
            raise RuntimeError(f"[patch-020] MXFP8 draft head self-check failed: min cosine {cos:.5f}")
        logger.info("[patch-020] DSpark draft lm_head: MXFP8 copy %s (BF16 target lm_head unchanged); self-check min "
                    "cosine %.5f, top-1 %.2f", tuple(self.weight.shape), cos, top1)


def draft_head(lm_head):
    """The lm_head the draft logits use: the target's own (default) or its MXFP8 copy (VLLM_DSPARK_FP8_HEAD=1)."""
    if os.environ.get(_ENV_HEAD, "0") != "1":
        return lm_head
    head = _HEADS.get(id(lm_head))
    if head is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[patch-020] the MXFP8 draft head must be built before CUDA graph capture")
        head = _Fp8DraftHead(lm_head)
        _HEADS[id(lm_head)] = head
    return head
