#!/usr/bin/env bash
# 026-mv.sh -- masked verification for DSpark (opt-in VLLM_GB10_MV=auto|fixed:<c>; lossless: drafts past a per-request cap
# chosen from DSpark's confidence head are not verified and their tokens borrow the request anchor's experts, so they add
# no expert reads). Installs vllm/_mv.py and applies the anchored edits (files/p026_mv.py). Fail-closed.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
echo "=== Patch 026: masked verification (now: ${VLLM_GB10_MV:-unset}, ctl ${VLLM_GB10_MV_CTL:-unset}, log ${VLLM_GB10_MV_LOG:-unset}) ==="
for f in mv.py p026_mv.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [026] $D/$f not found" >&2; exit 1; }
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/mv.py"
install -m 0644 "$D/mv.py" "$VLLM_ROOT/_mv.py"
echo " [OK] [026] installed _mv.py"
python3 "$D/xpatch.py" apply "$D/p026_mv.py"
