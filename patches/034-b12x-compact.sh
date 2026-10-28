#!/usr/bin/env bash
# 034-b12x-compact.sh -- build b12x's compact W4A8 micro launchers per MoE plan and attach them to vLLM's scratch-plan bindings
# (needed by patch 033's 320-wide slice at M <= 8; inert for every plan that is not compact micro). Applies the anchored edits
# files/p034_b12x_compact.py (fused_moe/b12x.py). Fail-closed, idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
echo "=== Patch 034: b12x compact micro launchers on vLLM bindings ==="
for f in p034_b12x_compact.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [034] $D/$f not found" >&2; exit 1; }
done
python3 "$D/xpatch.py" apply "$D/p034_b12x_compact.py"
echo " [OK] [034] b12x compact launcher edit in place"
