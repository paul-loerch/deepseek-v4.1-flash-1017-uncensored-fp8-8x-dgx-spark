#!/usr/bin/env bash
# 032-memstats.sh -- periodic CUDA allocator statistics per worker (opt-in VLLM_GB10_MEMSTATS=<seconds>; logging only).
# Installs vllm/_memstats.py and applies the anchored edit (files/p032_memstats.py). Fail-closed, idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
echo "=== Patch 032: allocator statistics (VLLM_GB10_MEMSTATS=${VLLM_GB10_MEMSTATS:-unset}) ==="
for f in memstats.py p032_memstats.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [032] $D/$f not found" >&2; exit 1; }
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/memstats.py"
install -m 0644 "$D/memstats.py" "$VLLM_ROOT/_memstats.py"
echo " [OK] [032] installed _memstats.py"
python3 "$D/xpatch.py" apply "$D/p032_memstats.py"
