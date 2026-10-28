# SPDX-License-Identifier: Apache-2.0
"""Patch 040 -- skip patch 026's argmax-chain confidences in drafter graphs that can never use them.

Patch 026 (masked verification, VLLM_GB10_MV=chain) applies its per-request caps only to steps that pass _eligible():
0 < num_reqs <= VLLM_GB10_MV_MAX_REQS (default 1), pure decode, uniform drafts.  Its chain_confidence() runs inside the DSpark
drafter graph, which vLLM captures once per batch size, so every graph -- also the ones for 2..256 requests, whose steps are
never eligible -- carries the chain: per draft position a gather of w2 rows [B, 2048, 256], an fp32 copy, a batched GEMV and an
argmax, then the confidence head.  The 10-27 profiles put that at ~9.6 ms of a 33.7 ms drafter graph at 64 users (it grows
with the batch) and ~0 at one user.

Edit (vllm/_mv.py, installed by 026): chain_confidence() returns before the chain when num_reqs > max_reqs, so the graphs for
larger batches are captured without it.  Exact: the confidences only choose which drafts are verified (the cap is fixed before
the drafts are drawn) and are never read for a step that _eligible() rejects; after_propose() then reads the previous values of
_ST.chain for those requests, which can only make the first cap of a request that later runs alone less tuned, never inexact.
Opt-in VLLM_GB10_MV_SKIP_INELIGIBLE=1 (read once at import); without it the module is byte-for-byte 026's behaviour.
"""
ID = "040"
TAG = "mv-skip"
M = f"# [patch-{ID} {TAG}]"

# sha256 of files/mv.py as installed by patch 026 (patches-i056 .. i060)
MV_SHA = "90e6d3e6b7526e016a5fa908de44120e5d4edc00ff4a60839ceaeff2455686ab"
_P = "_mv.py"

OLD_STATE = "_ST = _State()\n"
NEW_STATE = (
    "_ST = _State()\n"
    f"{M} VLLM_GB10_MV_SKIP_INELIGIBLE=1: graphs for more than VLLM_GB10_MV_MAX_REQS requests skip the chain (never used there)\n"
    "_P040_SKIP = os.environ.get(\"VLLM_GB10_MV_SKIP_INELIGIBLE\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\", \"on\")\n"
)

OLD_CHAIN = (
    "    before the drafts are drawn. Writes _ST.chain[:num_reqs].\"\"\"\n"
    "    if _ST.chain is None:\n"
    "        return\n"
)
NEW_CHAIN = (
    "    before the drafts are drawn. Writes _ST.chain[:num_reqs].\"\"\"\n"
    "    if _ST.chain is None:\n"
    "        return\n"
    f"    {M} masks apply only to steps with <= max_reqs requests (_eligible); a graph for more requests skips the chain\n"
    "    if _P040_SKIP and num_reqs > _ST.max_reqs:\n"
    "        return\n"
)

EDITS = [
    (_P, MV_SHA, OLD_STATE, NEW_STATE),
    (_P, MV_SHA, OLD_CHAIN, NEW_CHAIN),
]
