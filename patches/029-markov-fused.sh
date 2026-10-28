#!/usr/bin/env bash
# 029-markov-fused.sh -- DSpark top-k Markov bias in one Triton launch (opt-in VLLM_DSPARK_MARKOV_FUSED=1; drafter-only, lossless).
# Needs patches 013 + 020 first (pins dspark.py as they leave it). Installs vllm/_markov_fused.py and applies the anchored edit
# (files/p029_markov_fused.py). Fail-closed, idempotent. With the flag unset the edit is inert (patch 013's torch ops run).
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
echo "=== Patch 029: DSpark fused top-k Markov bias (VLLM_DSPARK_MARKOV_FUSED=${VLLM_DSPARK_MARKOV_FUSED:-unset}) ==="
for f in markov_fused.py p029_markov_fused.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [029] $D/$f not found" >&2; exit 1; }
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/markov_fused.py"
install -m 0644 "$D/markov_fused.py" "$VLLM_ROOT/_markov_fused.py"
echo " [OK] [029] installed _markov_fused.py"
python3 "$D/xpatch.py" apply "$D/p029_markov_fused.py"
