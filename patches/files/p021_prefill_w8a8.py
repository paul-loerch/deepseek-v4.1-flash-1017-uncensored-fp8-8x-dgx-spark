# SPDX-License-Identifier: Apache-2.0
"""Patch 021 -- MXFP8 W8A8 for prefill-sized GEMMs, Marlin W8A16 below (applied by xpatch.py after patch 014; why and what:
files/prefill_w8a8.py, installed as vllm/_prefill_w8a8.py by the wrapper). Opt-in: VLLM_MXFP8_PREFILL_W8A8=1.

Edit (model_executor/kernels/linear/__init__.py as patch 014 leaves it): init_mxfp8_linear_kernel returns
maybe_hybrid(kernel, config) instead of the kernel.
"""
ID = "021"
TAG = "prefill-w8a8"
M = f"# [patch-{ID} {TAG}]"

LINEAR_INIT_P014_SHA = "18a52e31fcd5702ae01dc0ef46d7bba1a521db6f78298e025ddfe5a7d1d05402"

OLD = """        logger.info_once("Using %s for MXFP8 GEMM", kernel_cls.__name__)
        return kernel_cls(config)
"""
NEW = f"""        logger.info_once("Using %s for MXFP8 GEMM", kernel_cls.__name__)
        {M} VLLM_MXFP8_PREFILL_W8A8=1: Marlin below the row threshold, FlashInfer W8A8 from it on
        from vllm._prefill_w8a8 import maybe_hybrid

        return maybe_hybrid(kernel_cls(config), config)
"""

EDITS = [("model_executor/kernels/linear/__init__.py", LINEAR_INIT_P014_SHA, OLD, NEW)]
