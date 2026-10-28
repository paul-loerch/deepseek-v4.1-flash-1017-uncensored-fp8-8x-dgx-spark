# SPDX-License-Identifier: Apache-2.0
"""Patch 022 -- sequence-parallel mHC in prefill for DeepSeek-V4.1 (applied by xpatch.py after patch 015; why and what:
files/prefill_sp.py, installed as vllm/_prefill_sp.py by the wrapper). Opt-in: VLLM_GB10_PREFILL_SP=1 (the wrapper applies
nothing when it is unset).

Edits (models/deepseek_v41/nvidia/model.py as patch 015 leaves it, sha256 238b8a74...):
  1. module gate before class DeepseekV4DecoderLayer: _P022_ON (env read once, at import) and, only when set,
     `from vllm import _prefill_sp as _p022` (gate set + module missing -> ImportError at import: fail closed).
  2. DeepseekV4DecoderLayer.forward, first statements: inside an SP region the layer runs _p022.sp_layer_forward
     (a line-by-line mirror of the upstream forward for this config, residual stream on this rank's rows, the two
     row-parallel all-reduces replaced by reduce-scatter + all-gather).
  3. DeepseekV4DecoderLayer.write_kv: assert that no region is active (the replay batch is full-T; it only runs after the
     region closed).
  4. DeepseekV4Model.forward, the first _run_layers call (layers start .. decoder_replay_start): open the region
     (begin_forward: eligibility + boot self-test), slice the embedding to this rank's rows, run the layers, gather the five
     carried states back to full T (exit_region) and close the region in a finally. Everything after (the bounded decoder
     replay with its write_kv, the collapse, the MTP/HC buffer, the final norm, the logits) runs as before on full T.
Inert unless the gate is on AND the forward is an eligible, eager, >= MIN_T-row prefill/mixed batch.
"""
ID = "022"
TAG = "prefill-sp"
M = f"# [patch-{ID} {TAG}]"

# models/deepseek_v41/nvidia/model.py after patch 015 (the running image's file)
MODEL_P015_SHA = "238b8a7461db576cf68536e10154a8a1ad95d5f0aafe502fb5021adbecbd0125"

ENV = "VLLM_GB10_PREFILL_SP"

# ---- 1. module gate ----
GATE_OLD = "\n\nclass DeepseekV4DecoderLayer(nn.Module):\n"
GATE_NEW = (
    "\n\n"
    f"{M} opt-in {ENV}=1 (read once, at import): during eligible eager prefills (T >= {ENV}_MIN_T, default 1,024)\n"
    "# layers start..decoder_replay_start run sequence-parallel (the residual stream sharded over the 8 TP ranks, the\n"
    "# attention / MoE all-reduces replaced by reduce-scatter + all-gather: vllm/_prefill_sp.py) -- a NUMERICS CHANGE in\n"
    "# the prefill reduction order. Unset/0 -> unchanged. Gate set + module missing / invalid value -> ImportError /\n"
    "# ValueError at import: fail closed.\n"
    f"_P022_ON = __import__(\"os\").environ.get(\"{ENV}\", \"0\").strip().lower() not in "
    "(\"\", \"0\", \"off\", \"false\", \"no\")\n"
    "if _P022_ON:\n"
    "    from vllm import _prefill_sp as _p022\n"
    "\n\nclass DeepseekV4DecoderLayer(nn.Module):\n"
)

# ---- 2. decoder-layer forward hook ----
LAYER_OLD = (
    "        previous_aux: torch.Tensor | None = None\n"
    "        mhc_stream = self.mhc_stream\n"
)
LAYER_NEW = (
    f"        {M} inside an SP region (opened by DeepseekV4Model.forward for an eligible prefill) the layer runs\n"
    "        # on this rank's rows: all-gather -> attention -> reduce-scatter, mHC on S rows. See vllm/_prefill_sp.py.\n"
    "        if _P022_ON and _p022.sp_active():\n"
    "            return _p022.sp_layer_forward(\n"
    "                self,\n"
    "                x,\n"
    "                positions,\n"
    "                input_ids,\n"
    "                pre_mix,\n"
    "                post_mix,\n"
    "                res_mix,\n"
    "                residual,\n"
    "                engram_hashes,\n"
    "                engram_mask,\n"
    "                capture_previous_aux=capture_previous_aux,\n"
    "                mega_gate_metadata=mega_gate_metadata,\n"
    "            )\n"
    "        previous_aux: torch.Tensor | None = None\n"
    "        mhc_stream = self.mhc_stream\n"
)

# ---- 3. write_kv guard ----
WKV_OLD = "        assert self.engram is None and not self.use_sequence_parallel\n"
WKV_NEW = (
    WKV_OLD
    + f"        {M} the replay batch is full-T: it runs after the SP region closed\n"
    "        assert not (_P022_ON and _p022.sp_active()), \"write_kv inside an SP region\"\n"
)

# ---- 4. model forward: the SP region around the first _run_layers ----
RUN_OLD = (
    "        aux_hidden_by_layer: dict[int, torch.Tensor] = {}\n"
    "        hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(\n"
    "            range(self.start_layer, self.decoder_replay_start),\n"
    "            hidden_states,\n"
    "            positions,\n"
    "            input_ids,\n"
    "            pre_mix,\n"
    "            post_mix,\n"
    "            res_mix,\n"
    "            residual,\n"
    "            aux_hidden_by_layer,\n"
    "            engram_hashes,\n"
    "            engram_mask,\n"
    "        )\n"
    "        late_aux: list[torch.Tensor] = []\n"
)
RUN_NEW = (
    "        aux_hidden_by_layer: dict[int, torch.Tensor] = {}\n"
    f"        {M} VLLM_GB10_PREFILL_SP=1: an eligible eager prefill (T >= MIN_T, no CUDA graph) runs the layers\n"
    "        # before the bounded replay on this rank's rows (T/8 each) and gathers the carried states back to full T\n"
    "        # afterwards; the replay, the collapse, the HC buffer, the norm and the logits are unchanged. begin_forward runs\n"
    "        # the fail-closed boot self-test on the first eligible forward (the profile run).\n"
    "        _p022_h = (\n"
    "            _p022.begin_forward(self, hidden_states, positions, input_ids) if _P022_ON else None\n"
    "        )\n"
    "        try:\n"
    "            if _p022_h is not None:\n"
    "                hidden_states = _p022.shard_input(hidden_states, _p022_h)\n"
    "            hidden_states, residual, post_mix, res_mix, pre_mix = self._run_layers(\n"
    "                range(self.start_layer, self.decoder_replay_start),\n"
    "                hidden_states,\n"
    "                positions,\n"
    "                input_ids,\n"
    "                pre_mix,\n"
    "                post_mix,\n"
    "                res_mix,\n"
    "                residual,\n"
    "                aux_hidden_by_layer,\n"
    "                engram_hashes,\n"
    "                engram_mask,\n"
    "            )\n"
    "            if _p022_h is not None:\n"
    "                hidden_states, residual, post_mix, res_mix, pre_mix = _p022.exit_region(\n"
    "                    _p022_h, hidden_states, residual, post_mix, res_mix, pre_mix\n"
    "                )\n"
    "        finally:\n"
    "            if _p022_h is not None:\n"
    "                _p022.end_forward(_p022_h)\n"
    "        late_aux: list[torch.Tensor] = []\n"
)

N_MARKERS = 4
EDITS = [
    ("models/deepseek_v41/nvidia/model.py", MODEL_P015_SHA, GATE_OLD, GATE_NEW),
    ("models/deepseek_v41/nvidia/model.py", MODEL_P015_SHA, LAYER_OLD, LAYER_NEW),
    ("models/deepseek_v41/nvidia/model.py", MODEL_P015_SHA, WKV_OLD, WKV_NEW),
    ("models/deepseek_v41/nvidia/model.py", MODEL_P015_SHA, RUN_OLD, RUN_NEW),
]
