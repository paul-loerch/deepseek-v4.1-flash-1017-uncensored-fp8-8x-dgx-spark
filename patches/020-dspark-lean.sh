#!/usr/bin/env bash
# 020-dspark-lean.sh -- DSpark drafter cost cuts (opt-in VLLM_DSPARK_FP8_HEAD=1, VLLM_DSPARK_MAIN_PROJ_TP=1); needs patch 013
# first. Installs vllm/_dspark_lean.py and applies the anchored edits (files/p020_dspark_lean.py). Fail-closed, idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
echo "=== Patch 020: DSpark drafter cuts (FP8 head: ${VLLM_DSPARK_FP8_HEAD:-unset}, main_proj TP: ${VLLM_DSPARK_MAIN_PROJ_TP:-unset}) ==="
for f in dspark_lean.py p020_dspark_lean.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [020] $D/$f not found" >&2; exit 1; }
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/dspark_lean.py"
install -m 0644 "$D/dspark_lean.py" "$VLLM_ROOT/_dspark_lean.py"
echo " [OK] [020] installed _dspark_lean.py"
python3 "$D/xpatch.py" apply "$D/p020_dspark_lean.py"
