#!/usr/bin/env bash
# 040-mv-skip.sh -- patch 026's argmax-chain confidences only in drafter graphs that can use them (opt-in
# VLLM_GB10_MV_SKIP_INELIGIBLE=1): graphs captured for more than VLLM_GB10_MV_MAX_REQS requests skip the chain (~9.6 ms of the
# drafter at 64 users), exact -- masks are never applied to those steps. Edits vllm/_mv.py (installed by 026, so this runs after
# it). Applies the anchored edits files/p040_mv_skip.py. Fail-closed, idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
echo "=== Patch 040: skip the MV chain in graphs for more than VLLM_GB10_MV_MAX_REQS requests (now: ${VLLM_GB10_MV_SKIP_INELIGIBLE:-unset}) ==="
for f in p040_mv_skip.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [040] $D/$f not found" >&2; exit 1; }
done
python3 "$D/xpatch.py" apply "$D/p040_mv_skip.py"
echo " [OK] [040] mv-skip edit in place"
