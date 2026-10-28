# SPDX-License-Identifier: Apache-2.0
"""Patch 033 (PROPOSAL, not applied anywhere) -- zero-pad the TP expert slice of b12x W4A8-MX experts to a multiple of 64.

DSV4.1-Flash at TP=8 holds all 384 routed experts at intermediate width 2304/8 = 288 = 2 x 128 + 32.  b12x stores such
a width "ceil-tiled" (288 -> 384, +33 %: 45.9 GiB instead of 34.5 GiB) and launches its dynamic kernel at 384.  A width
with n % 128 == 64 (320) selects b12x's compact N64 path instead (exact storage, bounded reads, micro/M16 split kernels).
Padding the slice 288 -> 320 is exact: the extra gate/up rows and w2 columns are FP4 zeros with zero E8M0 scales, so
silu(0) * 0 = 0 and the zero MXFP8 block of the intermediate contributes nothing to FC2.

Edit: model_executor/layers/fused_moe/oracle/mxfp4.py, mxfp4_round_up_hidden_size_and_intermediate_size: for the b12x
W4A8 backend round the per-rank intermediate size up to a multiple of 64.  vLLM then allocates the padded weights and scales
zero-initialised (Mxfp4MoEMethod.create_weights) and its generic loader narrows the checkpoint shard into the first 288
rows/columns (RoutedExperts._load_w13/_load_w2 -> _narrow_expert_data_for_padding); B12xExperts reads intermediate_size
from the padded w2 (b12x.py:344) and plans for 320.

Opt-in: VLLM_GB10_B12X_PAD64=1 (so one patch dir serves both sides of an A/B).  Applied by xpatch.py like the other
experimental patches:  python3 xpatch.py apply p033_b12x_pad.py [VLLM_ROOT]
"""
ID = "033"
TAG = "b12x-pad"
M = f"# [patch-{ID} {TAG}]"

# sha256 of the unpatched oracle/mxfp4.py of image dsv41-flash-sm121:n1 (vLLM 0.32.1rc1.dev466+ge4a5b5537)
ORACLE_SHA = "3623558965ae75bafa53ff76b8a6bdd22976176d760ff799c2d454ddf85eb751"

OLD = """    if backend in B12X_BACKENDS:
        # b12x plans for the exact model dimensions. B12xExperts validates the
        # required MXFP4 block alignment before selecting the backend.
        return hidden_size, intermediate_size
"""
NEW = f"""    if backend in B12X_BACKENDS:
        # b12x plans for the exact model dimensions. B12xExperts validates the
        # required MXFP4 block alignment before selecting the backend.
        {M} GB10: W4A8-MX widths that are not a multiple of 64 (TP8 of 2304 = 288) are
        # stored ceil-tiled to N128 (+33 %) and run at the padded width; a zero-padded multiple of 64
        # (320 = 2 x 128 + 64) takes b12x's exact compact N64 path.  Zero rows/columns are exact.
        import os as _p033_os

        if (
            backend == Mxfp4MoeBackend.B12X_MXFP4_MXFP8
            and _p033_os.environ.get("VLLM_GB10_B12X_PAD64", "0") == "1"
            and intermediate_size % 64 != 0
        ):
            intermediate_size = round_up(intermediate_size, 64)
        return hidden_size, intermediate_size
"""

EDITS = [("model_executor/layers/fused_moe/oracle/mxfp4.py", ORACLE_SHA, OLD, NEW)]
