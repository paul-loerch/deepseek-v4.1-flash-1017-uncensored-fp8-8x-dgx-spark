#!/usr/bin/env bash
# 031-sched-ctl.sh -- runtime scheduler knobs from a control file (opt-in VLLM_GB10_SCHED_CTL=<file>; scheduling only, lossless).
# Installs vllm/_sched_ctl.py and applies the anchored edit (files/p031_sched_ctl.py). Fail-closed, idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
echo "=== Patch 031: scheduler control file (VLLM_GB10_SCHED_CTL=${VLLM_GB10_SCHED_CTL:-unset}) ==="
for f in sched_ctl.py p031_sched_ctl.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [031] $D/$f not found" >&2; exit 1; }
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/sched_ctl.py"
install -m 0644 "$D/sched_ctl.py" "$VLLM_ROOT/_sched_ctl.py"
echo " [OK] [031] installed _sched_ctl.py"
python3 "$D/xpatch.py" apply "$D/p031_sched_ctl.py"
