# SPDX-License-Identifier: Apache-2.0
"""Patch 031 -- runtime scheduler knobs from a control file (applied by xpatch.py; why and what: files/sched_ctl.py, installed as
vllm/_sched_ctl.py). Opt-in: VLLM_GB10_SCHED_CTL=<file> (unset -> poll() returns at once, nothing changes).

Edit: v1/core/sched/scheduler.py (the running image's file), Scheduler.schedule: poll the control file before the step's budget
decisions read the knobs.
"""
ID = "031"
TAG = "sched-ctl"
M = f"# [patch-{ID} {TAG}]"

SCHED_SHA = "2aab08cb511d5905c280697fcc47fffa60804759d6fba22496779892c290b41b"

OLD = (
    "        # For logging.\n"
    "        scheduled_timestamp = time.monotonic()\n"
    "\n"
    "        self.kv_cache_manager.new_step_starts()\n"
)
NEW = (
    f"        {M} VLLM_GB10_SCHED_CTL: prefill_schedule_interval / long_prefill_token_threshold from a control file\n"
    "        from vllm import _sched_ctl as _p031\n"
    "\n"
    "        _p031.poll(self)\n"
    + OLD
)

EDITS = [("v1/core/sched/scheduler.py", SCHED_SHA, OLD, NEW)]
