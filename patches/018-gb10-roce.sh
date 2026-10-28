#!/usr/bin/env bash
# 018-gb10-roce.sh -- GB10 RoCE all-reduce / all-gather for small TP messages: b12x RoCEnante one-shot + our two-shot
# all-reduce (opt-in VLLM_GB10_ROCE=1, VLLM_GB10_ROCE_TWOSHOT_MAX_BYTES); needs patch 012 applied first. See files/p018_gb10_roce.py and files/gb10_roce.py for why and what.
# Installs vllm/_gb10_roce.py and applies the anchored edits. Fail-closed, idempotent (files/xpatch.py).
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
echo "=== Patch 018: GB10 RoCEnante collectives (opt-in VLLM_GB10_ROCE=1; now: ${VLLM_GB10_ROCE:-unset}) ==="
for f in gb10_roce.py p018_gb10_roce.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [018] $D/$f not found" >&2; exit 1; }
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/gb10_roce.py"
python3 -c "from flashinfer.experimental.b12x.comm import roce; assert roce.API_VERSION == 1" \
  || { echo "[FAIL] [018] flashinfer.experimental.b12x.comm.roce (API 1) not importable" >&2; exit 1; }
install -m 0644 "$D/gb10_roce.py" "$VLLM_ROOT/_gb10_roce.py"
echo " [OK] [018] installed _gb10_roce.py"
# Two-shot all-reduce runtime (used when VLLM_GB10_ROCE_TWOSHOT_MAX_BYTES > 0): a small package so its relative import
# and the C proxy source next to it resolve; the proxy library is built now so a toolchain problem fails the boot here.
if [ -f "$D/gb10_twoshot.py" ]; then
  for f in gb10_twoshot.py gb10_twoshot_cute.py gb10_twoshot_proxy.c; do
    [ -f "$D/$f" ] || { echo "[FAIL] [018] $D/$f not found" >&2; exit 1; }
  done
  python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$D/gb10_twoshot.py" "$D/gb10_twoshot_cute.py"
  install -d "$VLLM_ROOT/_gb10_twoshot"
  install -m 0644 /dev/null "$VLLM_ROOT/_gb10_twoshot/__init__.py"
  install -m 0644 "$D/gb10_twoshot.py" "$D/gb10_twoshot_cute.py" "$D/gb10_twoshot_proxy.c" "$VLLM_ROOT/_gb10_twoshot/"
  python3 -c "from vllm._gb10_twoshot import gb10_twoshot as t; t._load_library(); print(' [OK] [018] installed _gb10_twoshot, proxy library', t._LIB_DIGEST)" \
    || { echo "[FAIL] [018] two-shot proxy library did not build" >&2; exit 1; }
fi
python3 "$D/xpatch.py" apply "$D/p018_gb10_roce.py"
