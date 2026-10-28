# SPDX-License-Identifier: Apache-2.0
"""Patch 038 -- DSpark drafter: all-gather the per-rank top-k candidates, not the full vocabulary (applied by xpatch.py after
patches 013/020/024/026/029; why and what: files/dspark_topk_gather.py, installed as vllm/_dspark_topk_gather.py).
Opt-in: VLLM_DSPARK_TOPK_GATHER=1 | keys (unset -> the edit below runs the stock lines, byte for byte the same computation).

Edit: v1/worker/gpu/spec_decode/dspark/speculator.py (as patches 024 + 026 leave it), _sample_sequential_topk: the three lines
    base_logits = self.model.compute_draft_logits(sample_hidden)
    base_logits = base_logits.view(num_reqs, n_spec, -1)
    base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
become the hook's result when it is active (dense uninitialised buffer + the same top-k values / ids), else themselves. Everything
that follows (the -inf fill, patch 026's argmax-chain confidences, the sequential Markov loop, patch 024's top-p, the sampler and
the rejection sampler's cached draft logits) is untouched and sees the same tensors.
"""
ID = "038"
TAG = "dspark-topk-gather"
M = f"# [patch-{ID} {TAG}]"

SPEC_SHA = "1ce38cd00e5b675fefff3a23619f9c90d7c350a7263b2534bd5f705c4535d92a"  # dspark/speculator.py after patches 024 + 026

OLD = (
    "        base_logits = self.model.compute_draft_logits(sample_hidden)\n"
    "        base_logits = base_logits.view(num_reqs, n_spec, -1)\n"
    "        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)\n"
)
NEW = (
    f"        {M} VLLM_DSPARK_TOPK_GATHER: each rank takes the top-k of its own vocabulary shard and only\n"
    "        # the candidates are all-gathered (4x less traffic than the full vocabulary); same values, ids and order\n"
    "        from vllm import _dspark_topk_gather as _p038\n"
    "\n"
    "        _p038_out = _p038.draft_topk(self, sample_hidden, num_reqs, n_spec) if _p038.ENABLED else None\n"
    "        if _p038_out is not None:\n"
    "            base_logits, base_values, draft_indices = _p038_out\n"
    "        else:\n"
    "            base_logits = self.model.compute_draft_logits(sample_hidden)\n"
    "            base_logits = base_logits.view(num_reqs, n_spec, -1)\n"
    "            base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)\n"
)

EDITS = [("v1/worker/gpu/spec_decode/dspark/speculator.py", SPEC_SHA, OLD, NEW)]
