#!/usr/bin/env bash
# 025-mhc-split.sh -- GB10 split configs for the fused mHC post + pre GEMM at decode token counts (opt-in VLLM_GB10_MHC_SPLIT).
# Installs vllm/_mhc_split.py and applies the anchored edit (files/p025_mhc_split.py). Fail-closed, idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
echo "=== Patch 025: fused mHC split configs (VLLM_GB10_MHC_SPLIT=${VLLM_GB10_MHC_SPLIT:-unset}) ==="
for f in mhc_split.py p025_mhc_split.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [025] $D/$f not found" >&2; exit 1; }
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/mhc_split.py"
install -m 0644 "$D/mhc_split.py" "$VLLM_ROOT/_mhc_split.py"
echo " [OK] [025] installed _mhc_split.py"
python3 "$D/xpatch.py" apply "$D/p025_mhc_split.py"
