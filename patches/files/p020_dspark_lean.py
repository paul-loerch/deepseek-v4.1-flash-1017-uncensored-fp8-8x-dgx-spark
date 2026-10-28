# SPDX-License-Identifier: Apache-2.0
"""Patch 020 -- DSpark drafter cost cuts (applied by xpatch.py after patch 013; why and what: files/dspark_lean.py,
installed as vllm/_dspark_lean.py by the wrapper). Opt-in per part: VLLM_DSPARK_FP8_HEAD=1, VLLM_DSPARK_MAIN_PROJ_TP=1.

Edits (models/deepseek_v41/nvidia/dspark.py as patch 013 leaves it):
  DSparkDeepseekV4Model.__init__          main_proj built by make_main_proj (ReplicatedLinear unless the env asks)
  DSparkDeepseekV4ForCausalLM.compute_logits   draft logits through draft_head(self.lm_head)
"""
ID = "020"
TAG = "dspark-lean"
M = f"# [patch-{ID} {TAG}]"

DSPARK_P013_SHA = "f0736931c72f38a6c8c00bfd808aba8ca70e6c31e3413daa538a122a5d8e4451"

MP_OLD = """        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
"""
MP_NEW = f"""        {M} VLLM_DSPARK_MAIN_PROJ_TP=1: column-parallel, output all-gathered
        from vllm._dspark_lean import make_main_proj

        self.main_proj = make_main_proj(
            ReplicatedLinear,
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
"""
CL_OLD = """    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        \"\"\"Base logits U_k = lm_head(norm(head_hidden)).\"\"\"
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))
"""
CL_NEW = f"""    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        \"\"\"Base logits U_k = lm_head(norm(head_hidden)).\"\"\"
        {M} VLLM_DSPARK_FP8_HEAD=1: drafts use an MXFP8 copy of the lm_head; verification stays exact
        from vllm._dspark_lean import draft_head

        return self.logits_processor(draft_head(self.lm_head), self.model.norm(hidden_states))
"""

EDITS = [
    ("models/deepseek_v41/nvidia/dspark.py", DSPARK_P013_SHA, MP_OLD, MP_NEW),
    ("models/deepseek_v41/nvidia/dspark.py", DSPARK_P013_SHA, CL_OLD, CL_NEW),
]
