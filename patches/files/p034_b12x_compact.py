# SPDX-License-Identifier: Apache-2.0
"""Patch 034 -- attach b12x's compact W4A8 micro launchers to vLLM's scratch-plan bindings.

With patch 033 (VLLM_GB10_B12X_PAD64=1) the TP8 expert slice is 320 = 2 x 128 + 64 wide, which b12x runs on its exact compact
N64 path.  For M <= 8 tokens (single-user decode: 5-6 rows per step) the planner picks implementation "micro", and b12x's
compact micro dispatch (_impl.py b12x_moe_fp4: "compact W4A8 execution requires its prepared native launchers") reads
binding.compact_launches.  Only b12x's own prepared-state bind (_preparation._FusedMoeState.bind) sets that field; vLLM's
b12x.py plans with fused_moe.plan(Caps) -> TPMoEScratchPlan and binds with fused_moe.bind -> TPMoEScratchPlan.bind, which
leaves it None, so the padded build failed at its first M <= 8 launch (b055 boot, 2026-10-27).

Edit: model_executor/layers/fused_moe/b12x.py
  * _p034_compact_launches(plan): b12x's own builder _preparation._compact_launches(plan.launch_plan, plan.caps), once per
    plan (cached by plan identity; it returns None for every plan that is not compact micro, so 288-wide builds are untouched).
  * B12xExperts._plan builds them when the plan is created (plans are never created during CUDA capture, and the builder
    JIT-compiles), _run_b12x_moe_plan attaches them with dataclasses.replace -- what _FusedMoeState.bind does.
  * v2: opt-in VLLM_GB10_B12X_COMPACT_MICRO=0 re-plans a compact micro plan (M <= 8) with an explicit decode config for
    b12x's dyn-compact M16 kernels (the M >= 10 path); default 1 keeps b12x's choice.
  * v3: the compact-width test of that re-plan also accepts n % 128 == 32 when b12x patch 036 (native compact path for
    n % 128 == 32, B12X_COMPACT_N32=1) is installed and enabled -- so the 288-wide TP8 slice works with
    VLLM_GB10_B12X_COMPACT_MICRO=0 too.  Nothing else changes: _p034_compact_launches already defers to b12x's own builder
    (_preparation._compact_launches), which makes the same decision.
"""
ID = "034"
TAG = "b12x-compact"
M = f"# [patch-{ID} {TAG}]"

# sha256 of the unpatched model_executor/layers/fused_moe/b12x.py of image dsv41-flash-sm121:n1
B12X_SHA = "ad14c7eb3a9c83161de6b4e568085e6e7acc1a5dcc9b01e502bd66a1f8bd2a40"

OLD_HELPER = """def _run_b12x_moe_plan(
"""
NEW_HELPER = f"""{M} plan id -> (plan, compact launchers or None); the plan reference keeps the id unique.
import dataclasses as _p034_dataclasses

_P034_COMPACT: dict[int, tuple[Any, Any]] = {{}}
_P034_LOGGED: list[int] = []
import os as _p034_os

# VLLM_GB10_B12X_COMPACT_MICRO=0: plan b12x's dyn-compact M16 kernels (what M >= 10 uses) instead of the
# micro-compact path for M <= 8 (micro re-reads an expert per route: slower when the rows share experts).
_P034_DYN_SMALL = _p034_os.environ.get("VLLM_GB10_B12X_COMPACT_MICRO", "1") == "0"


def _p034_compact_n(n: Any) -> bool:
    # Whether b12x runs the exact compact W4A8 path at intermediate width n (mirrors b12x's own predicate).
    n = int(n)
    if n % 128 == 64:
        return True
    if n % 128 == 32:
        try:
            from b12x._lib.env import w4a8_compact_tail_rows  # b12x patch 036
        except ImportError:
            return False
        return bool(w4a8_compact_tail_rows(n))
    return False


def _p034_dynamic_small(fused_moe: Any, plan: Any) -> Any:
    launch_plan = getattr(plan, "launch_plan", None)
    caps = getattr(plan, "caps", None)
    if (
        not _P034_DYN_SMALL
        or launch_plan is None
        or caps is None
        or getattr(launch_plan, "implementation", None) != "micro"
        or getattr(launch_plan, "quant_mode", None) != "w4a8_mx"
        or not _p034_compact_n(getattr(launch_plan, "n", 0))
    ):
        return plan
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    config = MoeDecodeConfig(
        backend="dynamic",
        route_planner="internal",
        max_active_clusters=None,
        dynamic_tile_m=16,
        dynamic_route_mode="grouped",
    )
    return fused_moe.plan(_p034_dataclasses.replace(caps, decode_config=config))


def _p034_compact_launches(plan: Any) -> Any:
    hit = _P034_COMPACT.get(id(plan))
    if hit is not None and hit[0] is plan:
        return hit[1]
    launch_plan = getattr(plan, "launch_plan", None)
    caps = getattr(plan, "caps", None)
    compact = None
    if (
        launch_plan is not None
        and caps is not None
        and getattr(launch_plan, "implementation", None) == "micro"
    ):
        if _is_current_stream_capturing():
            raise RuntimeError(
                "b12x compact MoE launchers must be built before CUDA capture"
            )
        from b12x.moe.fused_moe._preparation import _compact_launches

        compact = _compact_launches(launch_plan, caps)
        if compact is not None and len(_P034_LOGGED) < 4:
            _P034_LOGGED.append(int(launch_plan.max_rows))
            logger.info(
                "[patch-034] b12x compact W4A8 micro launchers built "
                "(n=%d, rows=%d, topk=%d)",
                int(launch_plan.n),
                int(launch_plan.max_rows),
                int(launch_plan.num_topk),
            )
    _P034_COMPACT[id(plan)] = (plan, compact)
    return compact


def _run_b12x_moe_plan(
"""

OLD_RUN = """        input_scales_static=True,
        unit_scale_contract=unit_scale_contract,
    )
    fused_moe.run(binding=binding)
"""
NEW_RUN = f"""        input_scales_static=True,
        unit_scale_contract=unit_scale_contract,
    )
    {M} the scratch-plan bind leaves compact_launches None (only b12x's
    # prepared-state bind sets it); the compact micro path refuses to run without them.
    compact = _p034_compact_launches(plan)
    if compact is not None:
        binding = _p034_dataclasses.replace(binding, compact_launches=compact)
    fused_moe.run(binding=binding)
"""

OLD_PLAN = """        self._plans[key] = plan
        return plan
"""
NEW_PLAN = f"""        {M} optional dyn-compact M16 instead of micro-compact (M <= 8); then
        # build (JIT-compile) compact micro launchers outside CUDA capture.
        plan = _p034_dynamic_small(fused_moe, plan)
        self._plans[key] = plan
        _p034_compact_launches(plan)
        return plan
"""

_P = "model_executor/layers/fused_moe/b12x.py"
EDITS = [
    (_P, B12X_SHA, OLD_HELPER, NEW_HELPER),
    (_P, B12X_SHA, OLD_RUN, NEW_RUN),
    (_P, B12X_SHA, OLD_PLAN, NEW_PLAN),
]
