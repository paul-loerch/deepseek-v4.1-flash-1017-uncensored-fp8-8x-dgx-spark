# SPDX-License-Identifier: Apache-2.0
"""Patch 028 (runner half) -- the hook that hands the sequence-parallel region the per-step data it needs to build two half
contexts (applied by xpatch.py; why and what: files/prefill_sp_overlap.py, installed as vllm/_prefill_sp_overlap.py).
Opt-in: VLLM_GB10_PREFILL_SP_OVERLAP=1 (the wrapper applies nothing when it is unset; with the edit present and the variable
unset the hook is one environment lookup).

Edit to v1/worker/gpu/model_runner.py, in execute_model after the full attention metadata is built and before the model runs:
the hook decides from host data only whether this eager step has a usable row cut (halves of >= 1,024 rows, at a request
boundary or a 128-aligned prefill position) and stores the batch, block tables and slot mappings for the region. It never raises
for an ineligible step and changes nothing about the step itself.

The file is pinned to the sha256 it has after patch 026 (cd6f9a44...) or after patch 027 (212fc051...): P028_RUNNER_PIN=026|027
(the wrapper picks it from the file; default 026).
"""
import os

ID = "028"
TAG = "prefill-sp-overlap"
M = f"# [patch-{ID} {TAG}]"

PINS = {
    "026": "cd6f9a44a53758aa763f40c2abe6b4021bff83757bcc3bf4d12dab756d9512d9",   # after patch 026 (the running image)
    "027": "212fc051c70f72521a04720fe2557d15f37769659ae6fd693288c516aae7c4f3",   # after patch 027 (copy-aware drafting)
}
PIN = os.environ.get("P028_RUNNER_PIN", "026")
if PIN not in PINS:
    raise SystemExit(f"[FAIL] [028] P028_RUNNER_PIN={PIN!r} must be one of {sorted(PINS)}")
MR_SHA = PINS[PIN]

ENV = "VLLM_GB10_PREFILL_SP_OVERLAP"

R1 = (
    "        input_ids = input_batch.input_ids\n"
    "        inputs_embeds = None\n"
    "        ec_connector_output = None\n"
)
R1_NEW = (
    f"        {M} an eligible eager prefill step hands the SP region what it needs to run two row halves\n"
    f"        if __import__(\"os\").environ.get(\"{ENV}\", \"0\").strip().lower() not in (\"\", \"0\", \"off\", \"false\", \"no\"):\n"
    "            from vllm._prefill_sp_overlap import runner_hook as _p028_runner_hook\n"
    "\n"
    "            _p028_runner_hook(self, input_batch, batch_desc, block_tables, slot_mappings, ubatch_state, dummy_run)\n"
    "\n"
    + R1
)

N_MARKERS = 1
EDITS = [
    ("v1/worker/gpu/model_runner.py", MR_SHA, R1, R1_NEW),
]
