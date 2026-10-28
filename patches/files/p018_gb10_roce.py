# SPDX-License-Identifier: Apache-2.0
"""Patch 018 -- GB10 RoCE one-shot all-reduce / all-gather for small TP messages via b12x RoCEnante (applied by xpatch.py
after patch 012; why and what: files/gb10_roce.py, installed as vllm/_gb10_roce.py by the wrapper).

Edits:
  distributed/device_communicators/cuda_communicator.py   (patch 012's result)
     __init__     self._gb10_roce = maybe_create(self.cpu_group, self.device, self.unique_name)   (None unless VLLM_GB10_ROCE=1)
     all_reduce   eligible messages -> RoCEnante, before patch 012's one-shot and NCCL
     all_gather   eligible shards -> RoCEnante (any dim of a contiguous tensor, via a 2-D view)
  distributed/parallel_state.py   GroupCoordinator.graph_capture: enter the runtime's capture() context
  v1/worker/gpu/async_utils.py    AsyncOutput.get_output: health check once per step (fail closed on a timed-out wait)
"""
ID = "018"
TAG = "gb10-roce"
M = f"# [patch-{ID} {TAG}]"

COMM_P012_SHA = "375950851b7d772782db591036841e4c33964cda321275e9babdf055fb59f11f"
PSTATE_SHA = "862b722012fc8a32a75028657146599cd06b341ecafb203e77e2138dd4e66b4a"
ASYNC_SHA = "af54328bad11b63b09ea8fff7756354c59c58a6713499e7e94b566309331896c"

INIT_OLD = "        self._gb10_oneshot = _gb10_os_create(self.pynccl_comm, self.device, self.unique_name)\n"
INIT_NEW = INIT_OLD + (
    f"        {M} opt-in: VLLM_GB10_ROCE=1 -> RoCEnante one-shot all-reduce/all-gather for small TP messages\n"
    "        from vllm._gb10_roce import maybe_create as _gb10_roce_create\n"
    "        self._gb10_roce = _gb10_roce_create(self.cpu_group, self.device, self.unique_name)\n"
)
AR_OLD = (
    "        # [patch-012 gb10-oneshot-ar]\n"
    "        _gb10_os = getattr(self, '_gb10_oneshot', None)\n"
)
AR_NEW = (
    f"        {M}\n"
    "        _gb10_roce = getattr(self, '_gb10_roce', None)\n"
    "        if _gb10_roce is not None and _gb10_roce.should_ar(input_):\n"
    "            return _gb10_roce.all_reduce(input_)\n"
) + AR_OLD
AG_OLD = "    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:\n"
AG_NEW = AG_OLD + (
    f"        {M}\n"
    "        _gb10_roce = getattr(self, '_gb10_roce', None)\n"
    "        if _gb10_roce is not None and _gb10_roce.should_ag(input_, dim):\n"
    "            return _gb10_roce.all_gather(input_, dim)\n"
)

PS1_OLD = "        maybe_aiter_ar_context = nullcontext()\n"
PS1_NEW = PS1_OLD + f"        maybe_gb10_roce_context = nullcontext()  {M}\n"
PS2_OLD = (
    "            if aiter_ar_comm is not None:\n"
    "                maybe_aiter_ar_context = aiter_ar_comm.capture()  # type: ignore\n"
)
PS2_NEW = PS2_OLD + (
    f"            {M} the RoCE runtime keeps its own capture state\n"
    "            _gb10_roce = getattr(self.device_communicator, \"_gb10_roce\", None)\n"
    "            if _gb10_roce is not None:\n"
    "                maybe_gb10_roce_context = _gb10_roce.capture()\n"
)
PS3_OLD = "            maybe_aiter_ar_context,\n        ):\n"
PS3_NEW = "            maybe_aiter_ar_context,\n" f"            maybe_gb10_roce_context,  {M}\n" "        ):\n"

AS_OLD = (
    "    def get_output(self) -> ModelRunnerOutput:\n"
    "        self.copy_event.synchronize()\n"
    "\n"
    "        # NOTE(woosuk): The following code is to ensure compatibility with\n"
)
AS_NEW = (
    "    def get_output(self) -> ModelRunnerOutput:\n"
    "        self.copy_event.synchronize()\n"
    f"        {M} once per step: fail closed if a RoCE wait timed out\n"
    "        from vllm._gb10_roce import check_health as _gb10_roce_health\n"
    "        _gb10_roce_health()\n"
    "\n"
    "        # NOTE(woosuk): The following code is to ensure compatibility with\n"
)

EDITS = [
    ("distributed/device_communicators/cuda_communicator.py", COMM_P012_SHA, INIT_OLD, INIT_NEW),
    ("distributed/device_communicators/cuda_communicator.py", COMM_P012_SHA, AR_OLD, AR_NEW),
    ("distributed/device_communicators/cuda_communicator.py", COMM_P012_SHA, AG_OLD, AG_NEW),
    ("distributed/parallel_state.py", PSTATE_SHA, PS1_OLD, PS1_NEW),
    ("distributed/parallel_state.py", PSTATE_SHA, PS2_OLD, PS2_NEW),
    ("distributed/parallel_state.py", PSTATE_SHA, PS3_OLD, PS3_NEW),
    ("v1/worker/gpu/async_utils.py", ASYNC_SHA, AS_OLD, AS_NEW),
]
