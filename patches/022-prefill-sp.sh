#!/usr/bin/env bash
# 022-prefill-sp.sh -- sequence-parallel mHC in prefill for DeepSeek-V4.1 (opt-in VLLM_GB10_PREFILL_SP=1; NUMERICS CHANGE in the
# prefill reduction order only); needs patch 015 first (it pins model.py as 015 leaves it). Installs vllm/_prefill_sp.py and applies
# the four anchored edits (files/p022_prefill_sp.py through files/xpatch.py). Fail-closed, idempotent. Why and what: files/prefill_sp.py.
#
# With VLLM_GB10_PREFILL_SP unset/0 NOTHING is touched (model.py stays exactly as patch 015 leaves it). If a reused container already
# carries the edits, their import-time gate keeps upstream's replicated mHC + all-reduce path.
#
# Set the SAME flags on EVERY rank (M_DOCKER_EXTRA): among ranks that run the gate the boot fingerprint (module sha, MIN_T, TP, layer
# range) must match or every participating rank raises at the first eligible forward; a rank without the gate would never join
# the SP collectives and the others would block (up to the gloo / execute timeout).
#
#   VLLM_GB10_PREFILL_SP=1            on (unset/0/off: inert)
#   VLLM_GB10_PREFILL_SP_MIN_T=1024   smallest forward (rows) that runs sequence-parallel; 256 .. 1048576
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
MODEL="$VLLM_ROOT/models/deepseek_v41/nvidia/model.py"
MOD_SRC="$D/prefill_sp.py"
MOD_DST="$VLLM_ROOT/_prefill_sp.py"
MARKER="# [patch-022 prefill-sp]"
MARKERS=4
fail() { echo "[022] FATAL: $*" >&2; exit 1; }
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# Read-only dependencies: the SP forward mirrors / relies on these exact files of the running image (after patches 013-021). A
# different build is refused instead of being served with an unverified data flow. relative path (to the vllm package) sha256
DEPS="
models/deepseek_v41/common/engram.py d9af3da522740e81df9907b298567ec8fc6024cd851470c40e01e30907d9bbf6
models/deepseek_v41/attention.py 6f288f1a1393378eafaef16483e0d6ee13c5ae1ff63d9552453a34237919c371
models/deepseek_v41/decoder_replay_layers.py 0038dd2ee11cfd3234c89226b772db958388216d5f3a7bd74b33c8d22dd03880
models/deepseek_v41/nvidia/ops/mhc.py c8233a70004719017ff6a5826ca538c92ea6319a9c7c0090dc8fc2e603db25e7
models/deepseek_v41/nvidia/ops/o_proj.py b8a2c9358c6cc70b279f4d41647956a32c2cc83b9d53545086d0724cd30b8403
models/deepseek_v41/nvidia/flashinfer_sparse.py 4a4f0b3994883edff668023dc137298816bfadcd22be4831a93f2f3fbc8c716a
models/deepseek_v4/nvidia/ops/o_proj.py 05be23af9251f6c572da741a1949bfcf8ea012e62d90e326aab1fc91e1a0abee
models/deepseek_v4/nvidia/model.py 7f27d2167dc225d2f7d773cb5256e4a1736fdee14c09d2c204f5ecdbabd60c7f
model_executor/layers/linear.py d4b97d850eb8bdee5fd2459e7ac688d00623b54f5ec0ea590c08da0173f8b4d4
model_executor/layers/fused_moe/runner/moe_runner.py 64eee9565bdb2c599c8f3341c24e50062b80b9e625e544db44250372311289ef
model_executor/layers/fused_moe/config.py 9da7b7b7f825e466792fc47efad92d47c46716a2e5774ec112d945c5ada5d2ad
model_executor/kernels/mhc/tilelang.py f9b13bf93ce42426d2137e1065b2d0bee36c17d3b6d7657e1a959e9e4365137d
"

echo "=== Patch 022: DeepSeek-V4.1 prefill sequence-parallel (VLLM_GB10_PREFILL_SP=${VLLM_GB10_PREFILL_SP:-unset}, MIN_T=${VLLM_GB10_PREFILL_SP_MIN_T:-unset -> 1024}) ==="
mode_raw="$(printf '%s' "${VLLM_GB10_PREFILL_SP:-0}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$mode_raw" in
  ''|0|off|false|no)
    if [ -f "$MODEL" ] && grep -qF "$MARKER" "$MODEL" 2>/dev/null; then
      echo " [note] VLLM_GB10_PREFILL_SP unset but model.py already carries patch 022 (reused container?);"
      echo "        its import-time gate (_P022_ON) keeps upstream's replicated mHC + all-reduce path."
    fi
    echo "[OK] patch 022 skipped (VLLM_GB10_PREFILL_SP unset); files untouched"
    exit 0 ;;
  1|on|true|yes) ;;
  *) fail "VLLM_GB10_PREFILL_SP='${VLLM_GB10_PREFILL_SP}' must be unset/0 (inert) or 1 (on)" ;;
esac

MIN_T="${VLLM_GB10_PREFILL_SP_MIN_T:-1024}"
case "$MIN_T" in ''|*[!0-9]*) fail "VLLM_GB10_PREFILL_SP_MIN_T='$MIN_T' must be an integer";; esac
{ [ "${#MIN_T}" -le 8 ] && [ "$MIN_T" -ge 256 ] && [ "$MIN_T" -le 1048576 ]; } \
  || fail "VLLM_GB10_PREFILL_SP_MIN_T=$MIN_T outside [256, 1048576] (default 1024)"

for f in "$MODEL" "$MOD_SRC" "$D/p022_prefill_sp.py" "$D/xpatch.py"; do [ -f "$f" ] || fail "$f not found"; done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MOD_SRC" || fail "$MOD_SRC does not parse"

while read -r rel want; do
  [ -n "$rel" ] || continue
  [ -f "$VLLM_ROOT/$rel" ] || fail "read-only dependency $rel not found"
  got="$(sha "$VLLM_ROOT/$rel")"
  [ "$got" = "$want" ] || fail "$rel sha256 $got != pinned $want (a different image build, or a patch edited it after the SP forward was written against it); refusing"
done <<<"$DEPS"
echo " [OK] [022] read-only dependencies match the pinned image files"

had_mod=0; [ -e "$MOD_DST" ] && had_mod=1
install -m 0644 "$MOD_SRC" "$MOD_DST"
echo " [OK] [022] installed _prefill_sp.py"
# Engram under SP without all-gathers (VLLM_GB10_PREFILL_SP_ENGRAM=a2a; default gather = the upstream path): the module is
# installed whenever it ships with the patch, and used only in a2a mode.
ENGRAM_MODE="$(printf '%s' "${VLLM_GB10_PREFILL_SP_ENGRAM:-gather}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$ENGRAM_MODE" in gather|a2a) ;; *) fail "VLLM_GB10_PREFILL_SP_ENGRAM='${VLLM_GB10_PREFILL_SP_ENGRAM}' must be gather or a2a";; esac
if [ -f "$D/engram_sp_a2a.py" ]; then
  python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/engram_sp_a2a.py" || fail "engram_sp_a2a.py does not parse"
  install -m 0644 "$D/engram_sp_a2a.py" "$VLLM_ROOT/_engram_sp_a2a.py"
  echo " [OK] [022] installed _engram_sp_a2a.py (Engram mode: $ENGRAM_MODE)"
elif [ "$ENGRAM_MODE" = a2a ]; then
  fail "VLLM_GB10_PREFILL_SP_ENGRAM=a2a but files/engram_sp_a2a.py is missing"
fi
if ! python3 "$D/xpatch.py" apply "$D/p022_prefill_sp.py" "$VLLM_ROOT"; then
  [ "$had_mod" = 1 ] || rm -f "$MOD_DST"
  fail "xpatch refused model.py (not the post-015 file, or an anchor moved)"
fi

[ "$(grep -cF "$MARKER" "$MODEL")" = "$MARKERS" ] || fail "expected $MARKERS markers in $MODEL, found $(grep -cF "$MARKER" "$MODEL")"
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$MODEL" "$MOD_DST" || fail "patched/installed files do not parse"
echo "[OK] patch 022 in place (VLLM_GB10_PREFILL_SP=1, MIN_T=$MIN_T): the boot self-test runs on the first eligible forward and logs '[patch-022] prefill sequence-parallel ACTIVE'. NUMERICS CHANGE (prefill reduction order): quality-gate before serving."
