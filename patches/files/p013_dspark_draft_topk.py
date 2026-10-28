# SPDX-License-Identifier: Apache-2.0
"""Patch 013 -- DSpark draft top-k for DeepSeek-V4.1 (applied by xpatch.py).

WHY: the DSpark drafter samples its 5 draft tokens left to right; each step adds a Markov bias = markov_w2 @ embed(prev
token) over the full 129,280-token vocabulary. markov_w2 is a replicated BF16 [129280 x 256] matrix (66 MB), so every rank
reads it 5 times per decode step: ~300 us each on GB10, ~1.5 ms of a ~54 ms step (profile 2026-10-26, rank 0 and 4).
vLLM already has the cheaper form for Qwen3 DSpark (SpeculativeConfig.dspark_draft_topk, DSparkSpeculator.
_sample_sequential_topk): the base draft logits pick their top-k candidates once, and the Markov bias is applied only to
those (a gathered [k x 256] product per step). The rejection sampler then uses the same truncated draft distribution the
draft token was sampled from, so the served distribution stays exactly the target model's; only the acceptance rate can
change. vLLM gates the option to the Qwen3 architectures and the DeepSeek-V4.1 draft model lacks the gathered-bias hook.

WHAT (two anchored edits):
  config/speculative.py                   accept dspark_draft_topk for the DeepSeek-V4.1 draft (DSparkV41DraftModel)
  models/deepseek_v41/nvidia/dspark.py    DSparkDeepseekV4ForCausalLM.apply_markov_bias_gathered (fp32 product, result in
                                          the logits' dtype; same scale as markov_bias, the logits processor's)
Inert unless the speculative config sets "dspark_draft_topk".
"""
ID = "013"
TAG = "dspark-draft-topk"
M = f"# [patch-{ID} {TAG}]"

SPEC_SHA = "1f3a4e17b7235917203e4836895d740f73cac093253f35d03fc9f3dbe8eac8a2"
DSPARK_SHA = "5934ce7fb1df4d308c0e11f8c4235f07f083e6dc7dc4edea83bd885b30f05d02"

SPEC_OLD = (
    "                        if (\n"
    "                            \"Qwen3DSparkModel\"\n"
    "                            not in self.draft_model_config.architectures\n"
    "                            and _QWEN3_OMNI_DSPARK_ARCHITECTURE\n"
    "                            not in self.draft_model_config.architectures\n"
    "                        ):\n"
)
SPEC_NEW = (
    "                        if (\n"
    "                            \"Qwen3DSparkModel\"\n"
    "                            not in self.draft_model_config.architectures\n"
    "                            and _QWEN3_OMNI_DSPARK_ARCHITECTURE\n"
    "                            not in self.draft_model_config.architectures\n"
    f"                            {M} DeepSeek-V4.1 DSpark has the gathered-bias hook too\n"
    "                            and \"DSparkV41DraftModel\"\n"
    "                            not in self.draft_model_config.architectures\n"
    "                        ):\n"
)

DSPARK_OLD = (
    "    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:\n"
    "        return self.model.markov_head.bias(markov_embed, self.logits_processor)\n"
)
DSPARK_NEW = DSPARK_OLD + (
    "\n"
    f"    {M} Markov bias for the top-k base candidates only (dspark_draft_topk)\n"
    "    def apply_markov_bias_gathered(\n"
    "        self,\n"
    "        markov_embed: torch.Tensor,\n"
    "        logits: torch.Tensor,\n"
    "        values: torch.Tensor,\n"
    "        index: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        # logits: [B, V] filled with -inf by the caller; values/index: [B, k] top-k base logits and their ids.\n"
    "        weight = self.model.markov_head.markov_w2.weight[index]  # [B, k, r]\n"
    "        bias = torch.bmm(weight.float(), markov_embed.float().unsqueeze(-1)).squeeze(-1)\n"
    "        corrected = values.float() + self.logits_processor.scale * bias\n"
    "        return logits.scatter_(1, index, corrected.to(logits.dtype))\n"
)

EDITS = [
    ("config/speculative.py", SPEC_SHA, SPEC_OLD, SPEC_NEW),
    ("models/deepseek_v41/nvidia/dspark.py", DSPARK_SHA, DSPARK_OLD, DSPARK_NEW),
]
