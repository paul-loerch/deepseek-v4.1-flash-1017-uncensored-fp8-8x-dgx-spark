# SPDX-License-Identifier: Apache-2.0
"""Patch 025 -- GB10 split configs for the fused mHC post + pre GEMM at decode token counts (applied by xpatch.py; why and what:
files/mhc_split.py, installed as vllm/_mhc_split.py). Opt-in: VLLM_GB10_MHC_SPLIT (unset -> upstream configs).

Edit: model_executor/kernels/mhc/tilelang_kernels.py, mhc_fused_post_pre_split_config: consult the override first.
"""
ID = "025"
TAG = "mhc-split"
M = f"# [patch-{ID} {TAG}]"

TK_SHA = "1cf58ea6d51dd56f886b0a9571dc0ad9d75a0a908fd68b1c2d6d28234bd13d55"

OLD = """    fused kernel's block. One source of truth for the dispatch decision, the
    compile key and the launch, which must agree.
    \"\"\"
    if num_tokens > _FUSED_POST_PRE_MAX_TOKENS:
        return None
"""
NEW = f"""    fused kernel's block. One source of truth for the dispatch decision, the
    compile key and the launch, which must agree.
    \"\"\"
    {M} GB10: per-token-count configs from VLLM_GB10_MHC_SPLIT (vllm/_mhc_split.py), else upstream
    from vllm._mhc_split import override as _p025_override

    _p025_cfg = _p025_override(num_tokens, hidden_size, hc_mult)
    if _p025_cfg is not None:
        return _p025_cfg
    if num_tokens > _FUSED_POST_PRE_MAX_TOKENS:
        return None
"""

EDITS = [("model_executor/kernels/mhc/tilelang_kernels.py", TK_SHA, OLD, NEW)]
