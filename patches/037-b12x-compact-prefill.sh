#!/usr/bin/env bash
# 037-b12x-compact-prefill.sh -- b12x: large launches on the compact W4A8-MX layouts (n % 128 == 64: patch 033's 320; n % 128 == 32: patch 036's
# native 288) use the M64 split kernels instead of the M16 ones (DSV4.1-Flash TP8 prefill: 8192-token chunks = 128 routed rows per expert).
# Opt-in at run time with B12X_COMPACT_PREFILL_TILE_M=64 (size gate: B12X_COMPACT_PREFILL_MIN_ROWS_PER_EXPERT, default 36 = 2304 tokens for the
# 384-expert target layers, 1536 for the 128-expert drafter) -- without it every b12x decision is the old one.
# Applies the anchored edits files/p037_b12x_compact_prefill.py (4 files, 9 edits) to the b12x package (flashinfer/experimental/b12x, override
# with B12X_ROOT) ON TOP OF patch 036: the module pins the sha256 of the 036-patched files, so it refuses an unpatched or foreign tree.
# NOTE: changes the b12x package fingerprint (keys b12x's CuTe object cache) -> every b12x kernel JIT-compiles once more.
# Fail-closed, idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
B12X_ROOT="${B12X_ROOT:-/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x}"
echo "=== Patch 037: b12x compact W4A8-MX large launches on the M64 split kernels (B12X_COMPACT_PREFILL_TILE_M=${B12X_COMPACT_PREFILL_TILE_M:-unset}) ==="
for f in p037_b12x_compact_prefill.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [037] $D/$f not found" >&2; exit 1; }
done
[ -f "$B12X_ROOT/moe/fused_moe/_impl.py" ] || { echo "[FAIL] [037] b12x package not found at $B12X_ROOT" >&2; exit 1; }
grep -q "patch-036 b12x-n32" "$B12X_ROOT/_lib/env.py" || { echo "[FAIL] [037] patch 036 (036-b12x-n32.sh) must be applied first" >&2; exit 1; }
python3 "$D/xpatch.py" apply "$D/p037_b12x_compact_prefill.py" "$B12X_ROOT"
# policy smoke test (loads the one small module directly: no torch / no b12x package import)
python3 - "$B12X_ROOT/_lib/env.py" <<'PY' || { echo "[FAIL] [037] tile policy self-test failed" >&2; exit 1; }
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("b12x_env_037", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for k in ("B12X_COMPACT_PREFILL_TILE_M", "B12X_COMPACT_PREFILL_MIN_ROWS_PER_EXPERT"):
    os.environ.pop(k, None)
assert m.compact_prefill_tile_m() == 0 and m.w4a8_compact_dynamic_tile_m(10**6, 384) == 16          # off: always the stock M16
os.environ["B12X_COMPACT_PREFILL_TILE_M"] = "64"
assert m.compact_prefill_min_rows_per_expert() == 36
assert m.w4a8_compact_dynamic_tile_m(2303 * 6, 384) == 16 and m.w4a8_compact_dynamic_tile_m(2304 * 6, 384) == 64
os.environ["B12X_COMPACT_PREFILL_MIN_ROWS_PER_EXPERT"] = "0"
assert m.w4a8_compact_dynamic_tile_m(1, 384) == 64
os.environ["B12X_COMPACT_PREFILL_TILE_M"] = "32"
try:
    m.compact_prefill_tile_m()
except ValueError:
    pass
else:
    raise AssertionError("an unsupported tile must be refused")
PY
echo " [OK] [037] b12x compact prefill tile edit in place (inert until B12X_COMPACT_PREFILL_TILE_M=64)"
