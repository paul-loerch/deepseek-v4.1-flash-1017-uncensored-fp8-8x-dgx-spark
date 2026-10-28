# SPDX-License-Identifier: Apache-2.0
"""Patch 026 -- masked verification (applied by xpatch.py; why and what: files/mv.py, installed as vllm/_mv.py).
Opt-in: VLLM_GB10_MV=auto | fixed:<c> (off by default: every hook returns at once).

Edits:
  v1/worker/gpu/model_runner.py                      register after load; src remap before the target forward; capped
                                                     drafts invalid for verification; caps after each proposal; log
  v1/worker/gpu/spec_decode/dspark/speculator.py     compute the confidence head's outputs when MV wants them, and the
                                                     argmax-chain confidences (mode chain) before the drafts are sampled
  model_executor/layers/fused_moe/router/fused_moe_router.py   marked (target) routers remap capped tokens' experts
"""
ID = "026"
TAG = "mv"
M = f"# [patch-{ID} {TAG}]"

MR_SHA = "5cf64d4bb034dc31d7fd4d2ceac5ec55029b9e8106487c1005ee789d077828e7"   # model_runner.py after patch 024
SP_SHA = "76cd85a3b74885c1eea063991807ef1018774cd3d95324f088eedb32ae855522"   # dspark/speculator.py after patch 024
RT_SHA = "73361b728826c25065e827a6d27cbdcdf2fa23665e382ac4d01cb6f7962c7298"   # fused_moe_router.py (stock)

R1 = "        self.model_memory_usage = m.consumed_memory\n"
R1_NEW = R1 + (f"        {M} masked verification: buffers, target routers, mode\n"
               "        from vllm._mv import register as _p026_register\n"
               "        _p026_register(self)\n")

R2 = "        self.step_timing.forward_start()\n\n        # Run model.\n"
R2_NEW = ("        self.step_timing.forward_start()\n\n"
          f"        {M} which tokens borrow their request anchor's experts in this forward\n"
          "        from vllm._mv import before_forward as _p026_before_forward\n"
          "        _p026_before_forward(input_batch, dummy_run)\n\n"
          "        # Run model.\n")

R3 = "        sampler_output: SamplerOutput | None\n        if input_batch.num_reqs == 0:\n"
R3_NEW = (f"        {M} drafts past the request's cap are absent for verification\n"
          "        from vllm._mv import invalid_drafts as _p026_invalid_drafts\n"
          "        invalid_drafts = _p026_invalid_drafts(input_batch, invalid_drafts)\n"
          + R3)

R4 = """            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
                invalid_drafts,
            )
"""
R4_NEW = R4 + (f"            {M} per-step record (VLLM_GB10_MV_LOG, rank 0 only)\n"
               "            from vllm._mv import log_step as _p026_log_step\n"
               "            _p026_log_step(input_batch, sampler_output.num_sampled)\n")

R5 = "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"
R5_NEW = R5 + (f"            {M} caps for the next step from these drafts' confidences\n"
               "            from vllm._mv import after_propose as _p026_after_propose\n"
               "            _p026_after_propose(self.speculator, input_batch)\n")

S1 = """        self.use_confidence_head = (
            self.enable_adaptive_verification
            and model.model.confidence_head is not None
        )
"""
S1_NEW = (f"        {M} masked verification (VLLM_GB10_MV / _CTL / _LOG) uses the confidences too\n"
          "        import os as _p026_os\n"
          "        _p026_wants = any(_p026_os.environ.get(k) for k in (\"VLLM_GB10_MV_CTL\", \"VLLM_GB10_MV_LOG\")) or (\n"
          "            _p026_os.environ.get(\"VLLM_GB10_MV\", \"off\").strip().lower() not in (\"\", \"off\"))\n"
          "        self.use_confidence_head = (\n"
          "            (self.enable_adaptive_verification or _p026_wants)\n"
          "            and model.model.confidence_head is not None\n"
          "        )\n")

T1 = """            self._routing_replay_out[: topk_ids.shape[0]].copy_(
                topk_ids.to(torch.int16)
            )

        return topk_weights, topk_ids
"""
T1_NEW = ("""            self._routing_replay_out[: topk_ids.shape[0]].copy_(
                topk_ids.to(torch.int16)
            )

"""
          f"        {M} GB10: draft tokens past their request's cap take the request anchor's experts\n"
          "        if getattr(self, \"_p026_mv\", False):\n"
          "            from vllm._mv import remap_topk as _p026_remap\n"
          "            topk_weights, topk_ids = _p026_remap(topk_weights, topk_ids)\n"
          "\n"
          "        return topk_weights, topk_ids\n")

S2 = """        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
"""
S2_NEW = S2 + (f"        {M} confidences along the argmax chain (no sampled draft): input of the exact chain policy\n"
               "        if getattr(self, \"_p026_chain\", False):\n"
               "            from vllm._mv import chain_confidence as _p026_chain_confidence\n"
               "            _p026_chain_confidence(self, num_reqs, sample_hidden, base_values, draft_indices)\n")

EDITS = [
    ("v1/worker/gpu/model_runner.py", MR_SHA, R1, R1_NEW),
    ("v1/worker/gpu/model_runner.py", MR_SHA, R2, R2_NEW),
    ("v1/worker/gpu/model_runner.py", MR_SHA, R3, R3_NEW),
    ("v1/worker/gpu/model_runner.py", MR_SHA, R4, R4_NEW),
    ("v1/worker/gpu/model_runner.py", MR_SHA, R5, R5_NEW),
    ("v1/worker/gpu/spec_decode/dspark/speculator.py", SP_SHA, S1, S1_NEW),
    ("v1/worker/gpu/spec_decode/dspark/speculator.py", SP_SHA, S2, S2_NEW),
    ("model_executor/layers/fused_moe/router/fused_moe_router.py", RT_SHA, T1, T1_NEW),
]
