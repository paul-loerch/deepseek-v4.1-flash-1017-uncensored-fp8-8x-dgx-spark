#!/usr/bin/env bash
# 024-draft-top-p.sh -- the request's top-p on the DSpark draft distribution (opt-in VLLM_DSPARK_DRAFT_TOP_P=1; lossless: only
# acceptance changes). Installs vllm/_draft_top_p.py and applies the anchored edits (files/p024_draft_top_p.py). Fail-closed.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
echo "=== Patch 024: draft top-p (now: ${VLLM_DSPARK_DRAFT_TOP_P:-unset}) ==="
for f in draft_top_p.py p024_draft_top_p.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [024] $D/$f not found" >&2; exit 1; }
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/draft_top_p.py"
install -m 0644 "$D/draft_top_p.py" "$VLLM_ROOT/_draft_top_p.py"
echo " [OK] [024] installed _draft_top_p.py"
python3 "$D/xpatch.py" apply "$D/p024_draft_top_p.py"
