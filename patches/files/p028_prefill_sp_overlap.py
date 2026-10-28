# SPDX-License-Identifier: Apache-2.0
"""Patch 028 -- two-half communication/compute overlap for the patch-022 sequence-parallel prefill region (applied by xpatch.py
after patch 022; why and what: files/prefill_sp_overlap.py, installed as vllm/_prefill_sp_overlap.py by the wrapper).
Opt-in: VLLM_GB10_PREFILL_SP_OVERLAP=1 together with VLLM_GB10_PREFILL_SP=1 (the wrapper applies nothing when it is unset).

Edits to models/deepseek_v41/nvidia/model.py as the running image has it after patches 001-026 (sha256 36ccef3d...):
  1. module gate after the patch-022 gate: _P028_ON (env read once, at import; needs _P022_ON) and, only when set,
     `from vllm import _prefill_sp_overlap as _p028` (gate set + module missing -> ImportError at import: fail closed).
  2. DeepseekV4Model.forward, the 022 region: after begin_forward, `_p028.open_region(...)` returns the Region of an eligible
     forward (or None); with a Region, shard_input / _run_layers / exit_region are replaced by Region.run (same signature as
     _run_layers, returns the five carried states already gathered to full T).
  3. the 022 exit_region call is skipped when a Region ran (its states are already full-T).
  4. the finally clause also closes the Region.
Nothing else in model.py changes: the decoder layer forward, the bounded replay, the collapse, the logits.
The model_runner.py edit (the hook that hands the region the per-step data) is files/p028_runner.py.
"""
ID = "028"
TAG = "prefill-sp-overlap"
M = f"# [patch-{ID} {TAG}]"

# models/deepseek_v41/nvidia/model.py of the running image (patches 001-026 applied)
MODEL_I041_SHA = "36ccef3daa12ea1cee1214d73ac7ea315dba59cbdf970588810ba6f39f202fc8"

ENV = "VLLM_GB10_PREFILL_SP_OVERLAP"

# ---- 1. module gate ----
GATE_OLD = (
    "if _P022_ON:\n"
    "    from vllm import _prefill_sp as _p022\n"
    "\n\nclass DeepseekV4DecoderLayer(nn.Module):\n"
)
GATE_NEW = (
    "if _P022_ON:\n"
    "    from vllm import _prefill_sp as _p022\n"
    "\n\n"
    f"{M} opt-in {ENV}=1 (read once, at import; needs VLLM_GB10_PREFILL_SP=1): the sequence-parallel\n"
    "# prefill region runs two row halves so one half's reduce-scatter / mHC / all-gather chain overlaps the other half's\n"
    "# attention or MoE (vllm/_prefill_sp_overlap.py). Unset/0 -> unchanged. Gate set without the 022 gate / module missing /\n"
    "# invalid value -> ValueError / ImportError at import: fail closed.\n"
    f"_P028_ON = __import__(\"os\").environ.get(\"{ENV}\", \"0\").strip().lower() not in "
    "(\"\", \"0\", \"off\", \"false\", \"no\")\n"
    "if _P028_ON:\n"
    "    if not _P022_ON:\n"
    f"        raise ValueError(\"[patch-028] {ENV}=1 requires VLLM_GB10_PREFILL_SP=1\")\n"
    "    from vllm import _prefill_sp_overlap as _p028\n"
    "\n\nclass DeepseekV4DecoderLayer(nn.Module):\n"
)

# ---- 2. open the region; Region.run replaces shard_input + _run_layers (+ exit_region) ----
OPEN_OLD = (
    "        try:\n"
    "            if _p022_h is not None:\n"
    "                hidden_states = _p022.shard_input(hidden_states, _p022_h)\n"
    "            hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(\n"
    "                range(self.start_layer, self.decoder_replay_start),\n"
)
OPEN_NEW = (
    "        _p028_ov = None\n"
    "        try:\n"
    f"            {M} an eligible forward with a usable row cut runs the region two-half\n"
    "            # (Region.run has _run_layers' signature and returns the carried states already gathered to full T).\n"
    "            if _P028_ON and _p022_h is not None:\n"
    "                _p028_ov = _p028.open_region(self, _p022_h, hidden_states, positions, input_ids)\n"
    "            if _p022_h is not None and _p028_ov is None:\n"
    "                hidden_states = _p022.shard_input(hidden_states, _p022_h)\n"
    "            hidden_states, residual, post_mix, res_mix, pre_mix = (\n"
    "                self._run_layers if _p028_ov is None else _p028_ov.run\n"
    "            )(\n"
    "                range(self.start_layer, self.decoder_replay_start),\n"
)

# ---- 3. exit_region only for the plain 022 path ----
EXIT_OLD = (
    "            if _p022_h is not None:\n"
    "                hidden_states, residual, post_mix, res_mix, pre_mix = _p022.exit_region(\n"
)
EXIT_NEW = (
    f"            if _p022_h is not None and _p028_ov is None:  {M} Region.run already gathered\n"
    "                hidden_states, residual, post_mix, res_mix, pre_mix = _p022.exit_region(\n"
)

# ---- 4. close the region ----
CLOSE_OLD = (
    "        finally:\n"
    "            if _p022_h is not None:\n"
    "                _p022.end_forward(_p022_h)\n"
)
CLOSE_NEW = (
    "        finally:\n"
    f"            if _p028_ov is not None:  {M}\n"
    "                _p028_ov.close()\n"
    "            if _p022_h is not None:\n"
    "                _p022.end_forward(_p022_h)\n"
)

N_MARKERS = 4
EDITS = [
    ("models/deepseek_v41/nvidia/model.py", MODEL_I041_SHA, GATE_OLD, GATE_NEW),
    ("models/deepseek_v41/nvidia/model.py", MODEL_I041_SHA, OPEN_OLD, OPEN_NEW),
    ("models/deepseek_v41/nvidia/model.py", MODEL_I041_SHA, EXIT_OLD, EXIT_NEW),
    ("models/deepseek_v41/nvidia/model.py", MODEL_I041_SHA, CLOSE_OLD, CLOSE_NEW),
]
