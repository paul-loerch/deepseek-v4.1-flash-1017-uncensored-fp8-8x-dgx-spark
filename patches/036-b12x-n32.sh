#!/usr/bin/env bash
# 036-b12x-n32.sh -- b12x NATIVE compact W4A8-MX path for intermediate widths with n % 128 == 32 (DSV4.1-Flash TP8: 2304/8 = 288):
# the expert slice is stored and read exactly (2.35 MB per expert and rank) instead of ceil-tiled to 384 (3.13 MB) or zero-padded
# to 320 (2.61 MB, patch 033).  micro-compact kernels for M <= 8, split M16 kernels above, K tail of one K32 block in FC2.
# Opt-in at run time with B12X_COMPACT_N32=1 -- without it every b12x decision is the stock one.  Do not combine with 033
# (VLLM_GB10_B12X_PAD64=1 makes the slice 320 wide and the native path never applies).
# Applies the anchored edits files/p036_b12x_n32.py to the b12x package (flashinfer/experimental/b12x, override with B12X_ROOT).
# NOTE: this changes the b12x package fingerprint, which keys b12x's CuTe object cache -> every b12x kernel JIT-compiles once more
# (first boot / first call after installing).  Patch 034 (v3) attaches the compact micro launchers on the vLLM side.
# Fail-closed (pinned sha256 of every original file), idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
B12X_ROOT="${B12X_ROOT:-/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x}"
echo "=== Patch 036: b12x native compact W4A8-MX path for n % 128 == 32 (B12X_COMPACT_N32=${B12X_COMPACT_N32:-unset}) ==="
for f in p036_b12x_n32.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [036] $D/$f not found" >&2; exit 1; }
done
[ -f "$B12X_ROOT/moe/fused_moe/_impl.py" ] || { echo "[FAIL] [036] b12x package not found at $B12X_ROOT" >&2; exit 1; }
python3 "$D/xpatch.py" apply "$D/p036_b12x_n32.py" "$B12X_ROOT"
# predicate smoke test (loads the one small module directly: no torch / no b12x package import)
python3 - "$B12X_ROOT/_lib/env.py" <<'PY' || { echo "[FAIL] [036] width predicate self-test failed" >&2; exit 1; }
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("b12x_env_036", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
os.environ.pop("B12X_COMPACT_N32", None)
assert [m.w4a8_compact_tail_rows(n) for n in (256, 288, 320, 384)] == [0, 0, 64, 0]
os.environ["B12X_COMPACT_N32"] = "1"
assert [m.w4a8_compact_tail_rows(n) for n in (256, 288, 320, 384, 416)] == [0, 32, 64, 0, 32]
PY
echo " [OK] [036] b12x native compact n32 edit in place (inert until B12X_COMPACT_N32=1)"
