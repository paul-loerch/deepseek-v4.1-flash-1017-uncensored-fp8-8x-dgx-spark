# SPDX-License-Identifier: Apache-2.0
"""Patch 023 -- GB10 router GEMM through the CuTe dot-product kernel (applied by xpatch.py; why and what: files/router_ll.py,
installed as vllm/_router_ll.py). Opt-in: VLLM_GB10_ROUTER_LL=1.

Edit: model_executor/layers/fused_moe/router/gate_linear.py, GateLinear.forward: a guarded fast path before tier 1.
"""
ID = "023"
TAG = "router-ll"
M = f"# [patch-{ID} {TAG}]"

GATE_SHA = "dac35035853e6d737983b5c28c4b20615b36cf0690611d4ae034668e1704c3b5"

OLD = """    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # Tier 1: cuteDSL ll_bf16_gemm (SM90+, bf16 in, fp32 out,
"""
NEW = f"""    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        {M} GB10 single-user decode (M <= 8): the CuTe dot-product router GEMM instead of cuBLAS + split-K
        from vllm._router_ll import eligible as _p023_eligible, router_gemm as _p023_gemm

        if _p023_eligible(self, x):
            return self._return(_p023_gemm(x, self.weight))
        # Tier 1: cuteDSL ll_bf16_gemm (SM90+, bf16 in, fp32 out,
"""

EDITS = [("model_executor/layers/fused_moe/router/gate_linear.py", GATE_SHA, OLD, NEW)]
