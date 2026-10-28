# SPDX-License-Identifier: Apache-2.0
"""Patch 024 -- the request's top-p on the DSpark draft distribution (applied by xpatch.py; why and what: files/draft_top_p.py,
installed as vllm/_draft_top_p.py). Opt-in: VLLM_DSPARK_DRAFT_TOP_P=1.

Edits:
  v1/worker/gpu/model_runner.py               register the sampler's states before both speculator.propose() calls
  v1/worker/gpu/spec_decode/dspark/speculator.py  _sample_sequential_topk: truncate each step's candidates to the nucleus
"""
ID = "024"
TAG = "draft-top-p"
M = f"# [patch-{ID} {TAG}]"

MR_SHA = "49a6f9c289c91de31837bfd1fa03d7f82c341b18dc30bd9b693c963c5805e074"
SP_SHA = "3c22193d18f0d2f5f5dcd4ec907f52e10710a5b39b084dfa03eef254d3c174eb"

REG = (f"            {M} the drafter reads the requests' top_p from the sampler's states\n"
       "            from vllm._draft_top_p import register_states as _p024_register\n"
       "            _p024_register(self.sampler.sampling_states)\n")
A1 = "            with use_workspace_lane(self._draft_workspace_lane):\n                self.speculator.propose(\n"
A2 = "            with use_workspace_lane(self._draft_workspace_lane):\n                draft_tokens = self.speculator.propose(\n"
A3 = """            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
"""
A3_NEW = A3 + (f"            {M} the request's top-p on the draft distribution (drafts only; verification stays exact)\n"
               "            from vllm._draft_top_p import apply_draft_top_p as _p024_top_p\n"
               "            _p024_top_p(logits_i, draft_indices[:, i], idx_map[:, i], self.temperature)\n")

EDITS = [
    ("v1/worker/gpu/model_runner.py", MR_SHA, A1, REG + A1),
    ("v1/worker/gpu/model_runner.py", MR_SHA, A2, REG + A2),
    ("v1/worker/gpu/spec_decode/dspark/speculator.py", SP_SHA, A3, A3_NEW),
]
