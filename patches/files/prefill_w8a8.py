# SPDX-License-Identifier: Apache-2.0
"""MXFP8 W8A8 for prefill-sized GEMMs, W8A16 Marlin for everything else (patch 021; installed as vllm/_prefill_w8a8.py).

WHY: --linear-backend marlin (b015 on) keeps activations in BF16 (W8A16): the most exact arithmetic and the fastest decode
GEMMs on GB10, but prefill pays ~4-5 % against FlashInfer's CUTLASS W8A8 MXFP8 GEMM (b013: 4,363 tok/s at 32k, Marlin 4,139).
Decision of 2026-10-26: W8A8 for prefill only, gated by the quality check (KL vs official <= 0.0185, same task scores).

WHAT: when VLLM_MXFP8_PREFILL_W8A8=1 and the selected MXFP8 kernel is Marlin, every MXFP8 linear gets a hybrid kernel:
  rows <  VLLM_MXFP8_PREFILL_W8A8_MIN_TOKENS (default 512)  -> Marlin W8A16, unchanged (every CUDA-graph decode batch: at most
                                                              max_num_seqs x (1 + k) = 64 x 6 = 384 rows)
  rows >= the threshold                                     -> FlashInfer CUTLASS MXFP8 W8A8 (activations quantized to MXFP8
                                                              by the kernel, the model's own FP8 recipe)
The W8A8 path keeps its own copy of the checkpoint's FP8 weights and E8M0 scales (Marlin repacks the layer's in place):
about +1.2 GB per rank. A step that mixes decode rows into a prefill chunk of >= threshold rows runs W8A8 for the whole batch.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENV = "VLLM_MXFP8_PREFILL_W8A8"
_ENV_MIN = "VLLM_MXFP8_PREFILL_W8A8_MIN_TOKENS"
_LOGGED = False


def _base():
    from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearKernel

    return Mxfp8LinearKernel


class _W8A8Weights(torch.nn.Module):
    """The FlashInfer kernel's own copy of the layer's FP8 weight and E8M0 scales."""


def _make_hybrid_cls():
    base = _base()

    class HybridMxfp8LinearKernel(base):
        """Marlin W8A16 below the row threshold, FlashInfer CUTLASS W8A8 from it on."""

        supports_pre_processed_weights = False

        def __init__(self, small, large, config, min_tokens: int) -> None:  # no base __init__: both kernels checked
            self.small = small
            self.large = large
            self.config = config
            self.min_tokens = min_tokens
            self._w8a8: _W8A8Weights | None = None

        @classmethod
        def is_supported(cls, compute_capability=None):
            return True, None

        @classmethod
        def can_implement(cls, c):
            return True, None

        def input_quant_key(self):
            return None  # BF16 activations in: Marlin needs them, the W8A8 kernel quantizes itself

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            w8 = _W8A8Weights()
            w8.weight = torch.nn.Parameter(layer.weight.data.clone(), requires_grad=False)
            w8.weight_scale = torch.nn.Parameter(layer.weight_scale.data.clone(), requires_grad=False)
            self.large.process_weights_after_loading(w8)
            self._w8a8 = w8  # held by the kernel object, not registered on the layer (no state-dict change)
            self.small.process_weights_after_loading(layer)

        def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None):
            rows = x.numel() // x.shape[-1]
            if rows >= self.min_tokens:
                return self.large.apply_weights(self._w8a8, x, bias)
            return self.small.apply_weights(layer, x, bias)

    return HybridMxfp8LinearKernel


_HYBRID_CLS = None


def maybe_hybrid(kernel, config):
    """Wrap the selected MXFP8 kernel when the gate is on and it is Marlin; otherwise return it unchanged."""
    global _HYBRID_CLS, _LOGGED
    if os.environ.get(_ENV, "0") != "1":
        return kernel
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel

    if not isinstance(kernel, MarlinMxfp8LinearKernel) or getattr(config, "bmm_batch_size", None) is not None:
        return kernel
    ok, why = FlashInferCutlassMxfp8LinearKernel.is_supported()
    if ok:
        ok, why = FlashInferCutlassMxfp8LinearKernel.can_implement(config)
    if not ok:
        logger.info("[patch-021] %s stays on Marlin for all sizes: %s", tuple(config.weight_shape), why)
        return kernel
    min_tokens = int(os.environ.get(_ENV_MIN, "512"))
    if min_tokens <= 384:
        raise ValueError(f"[patch-021] {_ENV_MIN}={min_tokens} would put CUDA-graph decode batches (up to 384 rows) on W8A8")
    if _HYBRID_CLS is None:
        _HYBRID_CLS = _make_hybrid_cls()
    if not _LOGGED:
        _LOGGED = True
        logger.info("[patch-021] MXFP8 prefill W8A8 ACTIVE: Marlin W8A16 below %d rows, FlashInfer CUTLASS W8A8 from %d rows",
                    min_tokens, min_tokens)
    return _HYBRID_CLS(kernel, FlashInferCutlassMxfp8LinearKernel(config), config, min_tokens)
