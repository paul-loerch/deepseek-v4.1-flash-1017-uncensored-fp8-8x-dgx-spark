#!/usr/bin/env bash
# 021-prefill-w8a8.sh -- MXFP8 W8A8 for prefill-sized GEMMs, Marlin W8A16 below (opt-in VLLM_MXFP8_PREFILL_W8A8=1); needs patch
# 014 first. Installs vllm/_prefill_w8a8.py and applies the anchored edit (files/p021_prefill_w8a8.py). Fail-closed, idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
echo "=== Patch 021: MXFP8 prefill W8A8 (now: ${VLLM_MXFP8_PREFILL_W8A8:-unset}, from ${VLLM_MXFP8_PREFILL_W8A8_MIN_TOKENS:-512} rows) ==="
for f in prefill_w8a8.py p021_prefill_w8a8.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [021] $D/$f not found" >&2; exit 1; }
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/prefill_w8a8.py"
install -m 0644 "$D/prefill_w8a8.py" "$VLLM_ROOT/_prefill_w8a8.py"
echo " [OK] [021] installed _prefill_w8a8.py"
python3 "$D/xpatch.py" apply "$D/p021_prefill_w8a8.py"
