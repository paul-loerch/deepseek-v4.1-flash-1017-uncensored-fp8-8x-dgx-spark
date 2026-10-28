# SPDX-License-Identifier: Apache-2.0
"""GB10 router GEMM through vLLM's CuTe dot-product kernel (patch 023; installed as vllm/_router_ll.py).

WHY: GateLinear's fast router tier (cuteDSL ll_bf16_gemm: bf16 in, fp32 out) is enabled only on SM90 / SM100, so on GB10
(sm_121) the MoE router runs cuBLAS bf16 -> fp32 (an sm_80 WMMA kernel + a split-K reduce): 44 x ~36 us + 44 x 2.9 us per
decode step in b024 (it shares DRAM with the shared expert on the second stream). ll_bf16_gemm compiles and runs on sm_121a;
vLLM routes only M <= 4 to its dot-product kernel (the split-K variant is slower here); this uses the dot-product kernel for
M <= 8 (single-user decode: 5 / 6 rows). Measured on GB10 (cold weights, CUDA graphs): N=384 M=6 16.2 us vs cuBLAS
25.8; M=12 19.0 vs 20.6 (no gain); the DSpark router N=128: M=6 5.0 vs 7.1, M=12 10.2 vs 7.5 (worse) -> limit 8.

NUMERICS: fp32 accumulation either way (both against fp64: errors ~1e-6 relative, the dot-product kernel's slightly lower);
the summation order differs from cuBLAS, so router logits can differ in the last bits.
"""

from __future__ import annotations

import os

import torch

_ON = os.environ.get("VLLM_GB10_ROUTER_LL", "0") == "1"
_READY = None


def _ready() -> bool:
    global _READY
    if _READY is None:
        ok = False
        if _ON:
            from vllm.platforms import current_platform

            if current_platform.is_cuda() and current_platform.is_device_capability_family(120):
                import vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 as ll

                ll._DEFAULT_DOTPROD_MAX_M = max(ll._DEFAULT_DOTPROD_MAX_M, 8)
                ok = ll.is_available()
        _READY = ok
    return _READY


def eligible(layer, x: torch.Tensor) -> bool:
    return (_ON and x.dtype == torch.bfloat16 and x.dim() == 2 and 1 <= x.shape[0] <= 8
            and layer.weight.dtype == torch.bfloat16 and getattr(layer, "out_dtype", None) == torch.float32
            and getattr(layer, "_router_gemm_no_bias", False) and _ready())


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

    return ll_bf16_gemm(x, weight)
