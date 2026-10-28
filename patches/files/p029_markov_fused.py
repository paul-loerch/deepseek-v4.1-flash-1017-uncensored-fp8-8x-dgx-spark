# SPDX-License-Identifier: Apache-2.0
"""Patch 029 -- DSpark top-k Markov bias in one Triton launch (applied by xpatch.py after 013/020; why and what:
files/markov_fused.py, installed as vllm/_markov_fused.py). Opt-in: VLLM_DSPARK_MARKOV_FUSED=1 (unset -> patch 013's torch ops).

Edit: models/deepseek_v41/nvidia/dspark.py (as patches 013 + 020 leave it), apply_markov_bias_gathered: the fused kernel first.
"""
ID = "029"
TAG = "markov-fused"
M = f"# [patch-{ID} {TAG}]"

DSPARK_SHA = "46caeb4889fa0e4e4c55eec797cdbe3e6b2711447239cef484ba385d4c968885"

OLD = (
    "        # logits: [B, V] filled with -inf by the caller; values/index: [B, k] top-k base logits and their ids.\n"
    "        weight = self.model.markov_head.markov_w2.weight[index]  # [B, k, r]\n"
)
NEW = (
    "        # logits: [B, V] filled with -inf by the caller; values/index: [B, k] top-k base logits and their ids.\n"
    f"        {M} VLLM_DSPARK_MARKOV_FUSED=1: gather + fp32 dot + add + scatter in one launch (vllm/_markov_fused.py)\n"
    "        from vllm import _markov_fused as _p029\n"
    "\n"
    "        if _p029.ENABLED:\n"
    "            return _p029.markov_bias_gathered(\n"
    "                self.model.markov_head.markov_w2.weight,\n"
    "                markov_embed,\n"
    "                logits,\n"
    "                values,\n"
    "                index,\n"
    "                self.logits_processor.scale,\n"
    "            )\n"
    "        weight = self.model.markov_head.markov_w2.weight[index]  # [B, k, r]\n"
)

EDITS = [("models/deepseek_v41/nvidia/dspark.py", DSPARK_SHA, OLD, NEW)]
