# SPDX-License-Identifier: Apache-2.0
"""Patch 045 -- switch patch 015's replicated token embedding off again at run time (applied by xpatch.py).

Patch 015 builds DeepseekV4Model.embed_tokens with disable_tp=True: the full 129,280 x 5,120 BF16 table on every rank (+1.16 GiB)
so the decode graphs do not start with the embedding's all-reduce. With VLLM_GB10_EMBED_TP=1 the table is vocab-parallel again
(stock vLLM: 1/8 per rank, one all-reduce per forward), which gives the 1.16 GiB back to the KV cache. Numerics are identical
either way: every row has exactly one owner, the all-reduce only adds zeros.

Edit: models/deepseek_v41/nvidia/model.py as left by the whole patch set up to 028 (the last patch that edits it), replacing 015's
literal `disable_tp=True,` by an expression of the environment variable (read once, when the model is built). Unset or 0 = 015's
behaviour, byte-for-byte the same module object graph.
"""
ID = "045"
TAG = "embed-tp"
M = f"# [patch-{ID} {TAG}]"

# sha256 of models/deepseek_v41/nvidia/model.py after patches-i063 (013 .. 042; 028 is the last edit)
MODEL_SHA = "0670a5f828115ced91974f6c9d06663e1ecfa0d25866ce3b9ae4ce4c8422a920"

OLD = (
    "                # [patch-015 replicated-embedding] full table per rank: no all-reduce\n"
    "                disable_tp=True,\n"
)
NEW = (
    "                # [patch-015 replicated-embedding] full table per rank: no all-reduce\n"
    f"                {M} VLLM_GB10_EMBED_TP=1: vocab-parallel again (stock: 1/8 of the table per rank + one all-reduce)\n"
    "                disable_tp=__import__(\"os\").environ.get(\"VLLM_GB10_EMBED_TP\", \"0\").strip().lower()\n"
    "                not in (\"1\", \"true\", \"yes\", \"on\"),\n"
)

EDITS = [("models/deepseek_v41/nvidia/model.py", MODEL_SHA, OLD, NEW)]
