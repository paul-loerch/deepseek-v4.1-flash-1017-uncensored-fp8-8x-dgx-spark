#!/usr/bin/env bash
# 033-b12x-pad.sh -- zero-pad the TP expert slice of b12x W4A8-MX experts to a multiple of 64 (288 -> 320 at TP8): b12x then
# takes its exact compact N64 path instead of storing and running the slice ceil-tiled to 384 (opt-in VLLM_GB10_B12X_PAD64=1;
# exact math: the pad is FP4 zeros). Applies the anchored edit files/p033_b12x_pad.py (oracle/mxfp4.py). Fail-closed, idempotent;
# with the flag unset the edit is inert.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
echo "=== Patch 033: b12x expert slice padded to a multiple of 64 (VLLM_GB10_B12X_PAD64=${VLLM_GB10_B12X_PAD64:-unset}) ==="
for f in p033_b12x_pad.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [033] $D/$f not found" >&2; exit 1; }
done
python3 "$D/xpatch.py" apply "$D/p033_b12x_pad.py"
echo " [OK] [033] b12x pad64 edit in place"
