# SPDX-License-Identifier: Apache-2.0
"""Patch 014 -- DeepSeek-V4.1 wo_a as a plain MXFP8 GEMM when a rank holds one output group (applied by xpatch.py).

WHY: the attention output projection wo_a is a grouped GEMM (8 groups of [1024 x 4096]); vLLM runs it through an MXFP8
BMM kernel. On GB10 (SM12x) the only BMM kernel DeepGEMM offers needs Blackwell datacenter (family 100), so vLLM falls back
to EmulationMxfp8LinearKernel: it dequantizes the FP8 weight to BF16 once and runs a BF16 torch.bmm per layer (~50 us per
layer and twice the weight bytes; profile 2026-10-26: 45 launches, 3.9 ms of a 54 ms decode step). At TP=8 every rank
holds exactly one group (n_local_groups == 1), so the BMM is a plain [T, 4096] x [4096, 1024] GEMM, which the FlashInfer
CUTLASS MXFP8 kernel (already used by every other linear of the model) runs from the FP8 weight. The activation is then
quantized to MXFP8 (per-32 UE8M0 scales) like every other linear input -- the precision the reference model uses for
wo_a on Blackwell (fp8_einsum); today's SM12x fallback keeps it in BF16.

WHAT (three anchored edits):
  model_executor/kernels/linear/__init__.py     init_mxfp8_linear_kernel: bmm_batch_size == 1 on CUDA and no DeepGEMM
                                                BMM support -> select among the plain MXFP8 kernels
  model_executor/layers/quantization/modelopt.py  ModelOptLinearMethod.process_weights_after_loading: mark such a layer
                                                  (layer._p014_plain_linear = True) when the re-selected kernel is plain
  models/deepseek_v4/nvidia/ops/o_proj.py       deep_gemm_fp8_o_proj: for a marked wo_a and a BF16 attention output,
                                                inverse RoPE without quantization, then wo_a as a linear, then wo_b
Inert elsewhere: more than one local group (TP < 8), DeepGEMM BMM available (SM100/103), or an attention kernel that
already produced a QuantizedActivation keep the stock path.
"""
ID = "014"
TAG = "wo-a-plain-mxfp8"
M = f"# [patch-{ID} {TAG}]"

KERNELS_SHA = "0613c234bd04a76b857b9593ebc2d2b14398bd635154922d5f119013ff8fc3e0"
MODELOPT_SHA = "4de086461994c7d5f5e59dadb9356ee8de4e4c2b3509100569b0f0d0fcacc7f2"
OPROJ_SHA = "f2893ee6458eef50a371aae835b03fe1296f60f604236ab56235235637836e91"

KERNELS_OLD = (
    "    \"\"\"Select and instantiate the best MXFP8 linear kernel for the\n"
    "    current platform and `(N, K)` weight shape.\"\"\"\n"
    "    config = Mxfp8LinearLayerConfig(\n"
)
KERNELS_NEW = (
    "    \"\"\"Select and instantiate the best MXFP8 linear kernel for the\n"
    "    current platform and `(N, K)` weight shape.\"\"\"\n"
    f"    {M} one local group is a plain [N, K] GEMM: without DeepGEMM's\n"
    "    # BMM kernel (SM12x) use the plain MXFP8 kernels instead of the BF16 emulation.\n"
    "    if (\n"
    "        bmm_batch_size == 1\n"
    "        and current_platform.is_cuda()\n"
    "        and not DeepGemmMxfp8BmmLinearKernel.is_supported()[0]\n"
    "    ):\n"
    "        bmm_batch_size = None\n"
    "    config = Mxfp8LinearLayerConfig(\n"
)

MODELOPT_OLD = (
    "        if self.spec.weight == kMxfp8Static and getattr(layer, \"is_bmm\", False):\n"
    "            self.kernel = init_mxfp8_linear_kernel(\n"
    "                weight_shape=(layer.weight.shape[-2], layer.weight.shape[-1]),\n"
    "                bmm_batch_size=layer.bmm_batch_size,\n"
    "            )\n"
)
MODELOPT_NEW = MODELOPT_OLD + (
    f"            {M} a single-group BMM that got a plain kernel runs as a linear\n"
    "            layer._p014_plain_linear = self.kernel.config.bmm_batch_size is None\n"
)

OPROJ_OLD = (
    "    \"\"\"\n"
    "    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn\n"
    "    if isinstance(o, QuantizedActivation):\n"
)
OPROJ_NEW = (
    "    \"\"\"\n"
    f"    {M} single local group on a plain MXFP8 kernel: inverse RoPE\n"
    "    # in BF16, then wo_a as a linear (the kernel quantizes its input to MXFP8), then wo_b.\n"
    "    if getattr(wo_a, \"_p014_plain_linear\", False) and not isinstance(\n"
    "        o, QuantizedActivation\n"
    "    ):\n"
    "        assert n_groups == 1, n_groups\n"
    "        rotated, _ = fused_inv_rope_fp8_quant(\n"
    "            o,\n"
    "            positions,\n"
    "            cos_sin_cache,\n"
    "            n_groups=n_groups,\n"
    "            heads_per_group=heads_per_group,\n"
    "            nope_dim=nope_dim,\n"
    "            rope_dim=rope_dim,\n"
    "            quant_group_size=einsum_recipe[2],\n"
    "            tma_aligned_scales=tma_aligned_scales,\n"
    "            quantize=False,\n"
    "        )\n"
    "        return wo_b(wo_a(rotated.reshape(rotated.shape[0], -1)))\n"
    "    use_fp8 = wo_a.weight.dtype == torch.float8_e4m3fn\n"
    "    if isinstance(o, QuantizedActivation):\n"
)

EDITS = [
    ("model_executor/kernels/linear/__init__.py", KERNELS_SHA, KERNELS_OLD, KERNELS_NEW),
    ("model_executor/layers/quantization/modelopt.py", MODELOPT_SHA, MODELOPT_OLD, MODELOPT_NEW),
    ("models/deepseek_v4/nvidia/ops/o_proj.py", OPROJ_SHA, OPROJ_OLD, OPROJ_NEW),
]
