#!/usr/bin/env bash
# 038-dspark-topk-gather.sh -- DSpark drafter: all-gather the per-rank top-k candidates instead of the full vocabulary (opt-in
# VLLM_DSPARK_TOPK_GATHER=1|keys; drafter-only, lossless by construction: same top-k values, ids and order as the stock path).
# Needs patches 013 + 020 + 024 + 026 first (pins speculator.py as they leave it). Installs vllm/_dspark_topk_gather.py and applies the
# anchored edit (files/p038_dspark_topk_gather.py). Fail-closed, idempotent. With the flag unset the edit runs the stock lines.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
echo "=== Patch 038: DSpark top-k gather (VLLM_DSPARK_TOPK_GATHER=${VLLM_DSPARK_TOPK_GATHER:-unset}, min rows ${VLLM_DSPARK_TOPK_GATHER_MIN_ROWS:-16}, check ${VLLM_DSPARK_TOPK_GATHER_CHECK:-0}) ==="
for f in dspark_topk_gather.py p038_dspark_topk_gather.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [038] $D/$f not found" >&2; exit 1; }
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/dspark_topk_gather.py"
install -m 0644 "$D/dspark_topk_gather.py" "$VLLM_ROOT/_dspark_topk_gather.py"
echo " [OK] [038] installed _dspark_topk_gather.py"
python3 "$D/xpatch.py" apply "$D/p038_dspark_topk_gather.py" "$VLLM_ROOT"
