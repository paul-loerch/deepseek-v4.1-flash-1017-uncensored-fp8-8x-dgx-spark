# SPDX-License-Identifier: Apache-2.0
"""Patch 032 -- periodic caching-allocator statistics per worker (applied by xpatch.py; files/memstats.py -> vllm/_memstats.py).
Opt-in VLLM_GB10_MEMSTATS=<seconds> (unset -> start() returns at once); VLLM_GB10_MEMSTATS_HISTORY=1 also records allocation\nstacks from load_model to warm-up and logs the top allocation sites once (v2). Logging only.

Edit: v1/worker/gpu_worker.py (the running image's file), Worker.compile_or_warm_up_model: start the logger thread first.
"""
ID = "032"
TAG = "memstats"
M = f"# [patch-{ID} {TAG}]"

WORKER_SHA = "ccba545c1883c723836fbfeac3babee57e502a46cd3dabfe94a46852500e6140"

OLD = (
    "    def compile_or_warm_up_model(self) -> CompilationTimes:\n"
    "        if not self.use_v2_model_runner:\n"
)
NEW = (
    "    def compile_or_warm_up_model(self) -> CompilationTimes:\n"
    f"        {M} VLLM_GB10_MEMSTATS=<s>: log allocator statistics every <s> seconds (vllm/_memstats.py)\n"
    "        from vllm import _memstats as _p032\n"
    "\n"
    "        _p032.start(self.device)\n"
    "        if not self.use_v2_model_runner:\n"
)

OLD2 = (
    "    def load_model(self, *, load_dummy_weights: bool = False) -> None:\n"
    "        with (\n"
)
NEW2 = (
    "    def load_model(self, *, load_dummy_weights: bool = False) -> None:\n"
    f"        {M} VLLM_GB10_MEMSTATS_HISTORY=1: record allocation stacks until warm-up (vllm/_memstats.py)\n"
    "        from vllm import _memstats as _p032\n"
    "\n"
    "        _p032.start_history()\n"
    "        with (\n"
)

EDITS = [("v1/worker/gpu_worker.py", WORKER_SHA, OLD, NEW), ("v1/worker/gpu_worker.py", WORKER_SHA, OLD2, NEW2)]
