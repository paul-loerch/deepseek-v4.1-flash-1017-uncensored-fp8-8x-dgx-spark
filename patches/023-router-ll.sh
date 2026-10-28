#!/usr/bin/env bash
# 023-router-ll.sh -- GB10 router GEMM through vLLM's CuTe dot-product kernel for decode batches (opt-in VLLM_GB10_ROUTER_LL=1).
# Installs vllm/_router_ll.py and applies the anchored edit (files/p023_router_ll.py). Fail-closed, idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
echo "=== Patch 023: router GEMM via the CuTe dot-product kernel (now: ${VLLM_GB10_ROUTER_LL:-unset}) ==="
for f in router_ll.py p023_router_ll.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [023] $D/$f not found" >&2; exit 1; }
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/router_ll.py"
install -m 0644 "$D/router_ll.py" "$VLLM_ROOT/_router_ll.py"
echo " [OK] [023] installed _router_ll.py"
python3 "$D/xpatch.py" apply "$D/p023_router_ll.py"
