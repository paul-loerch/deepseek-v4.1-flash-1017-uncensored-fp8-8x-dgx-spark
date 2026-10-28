# SPDX-License-Identifier: Apache-2.0
"""Patch 015 -- DeepSeek-V4.1: replicated token embedding, no all-reduce at the start of each graph (applied by xpatch.py).

WHY: the vocab-parallel embedding ends in an all-reduce, which is the first collective of both decode CUDA graphs (target
and DSpark drafter; the drafter shares the target's embedding module). NCCL's host-callback nodes in the graphs pay the
CPU wake-up once per graph at the first collective: in the 2026-10-26 profiles the first all-reduce of each graph ended
~0.75-1.0 ms (target) and ~1.2-1.5 ms (drafter) after the graph started, against ~65 us for every later one, on all
ranks. A replicated table (full 129,280 x 5,120 BF16 per rank, +1.16 GB) needs no all-reduce: the graphs' first
collective moves to layer 0's attention output, ~1 ms into the graph, which gives the callbacks time to run ahead.
Numerics: identical (each row has one owner today; the all-reduce only adds zeros).

WHAT: models/deepseek_v41/nvidia/model.py, DeepseekV4Model.__init__: VocabParallelEmbedding(..., disable_tp=True).
"""
ID = "015"
TAG = "replicated-embedding"
M = f"# [patch-{ID} {TAG}]"

MODEL_SHA = "a7e237a578cb4e2ebaa7060efbfbaf2ae57da2c316d9146dc6b4d09581a552a1"

MODEL_OLD = (
    "        if get_pp_group().is_first_rank:\n"
    "            self.embed_tokens = VocabParallelEmbedding(\n"
    "                config.vocab_size,\n"
    "                config.hidden_size,\n"
    "                quant_config=quant_config,\n"
    "                prefix=f\"{prefix}.embed_tokens\",\n"
    "            )\n"
)
MODEL_NEW = (
    "        if get_pp_group().is_first_rank:\n"
    "            self.embed_tokens = VocabParallelEmbedding(\n"
    "                config.vocab_size,\n"
    "                config.hidden_size,\n"
    "                quant_config=quant_config,\n"
    "                prefix=f\"{prefix}.embed_tokens\",\n"
    f"                {M} full table per rank: no all-reduce\n"
    "                disable_tp=True,\n"
    "            )\n"
)

EDITS = [
    ("models/deepseek_v41/nvidia/model.py", MODEL_SHA, MODEL_OLD, MODEL_NEW),
]
