#!/usr/bin/env bash
# 001-gb10-spin-wait.sh — Reduce vLLM IPC spin-wait CPU load / SoC heat on GB10
#
# ROOT CAUSE: vLLM's cross-process IPC wait path (shm_broadcast.SpinCondition)
# spins on performance cores for `busy_loop_s` (default 1s) before sleeping.
# Under decode the message gap is always < 1s, so the sleep path never engages
# and 3–4 P-cores spin at max clock (issue #79).
#
# FIX (one value):
#   vllm/distributed/device_communicators/shm_broadcast.py
#   SpinCondition.__init__  busy_loop_s: float = 1 → 0.002  (2ms grace, then sleep via notify socket)
#
# TP>=2 only (this recipe). TP=1 uses the uni executor and never forms this path. Wait policy only — model
# outputs unchanged. In a multi-node mp run (one GPU per node) the shared-memory queue, and so this wait, exists only
# between the engine core and the worker that lives on the head node; the other nodes' workers read the broadcast
# over a zmq socket (a blocking recv, no spin). The patch is applied on every node anyway (identical containers).
#
# vLLM nightly e4a5b55 PORT (vllm/vllm-openai nightly, 0.32.1rc1.dev466): the edited line is byte-identical to v0.32.0's
# (`busy_loop_s: float = 1,` in SpinCondition.__init__, nightly shm_broadcast.py line 141); upstream still has no knob for
# it. The nightly changed the neighbours only: the reader's wait moved into MessageQueue._wait_until_readable (it still
# ends in SpinCondition.wait(timeout_ms=...), capped at SHM_READER_RECHECK_INTERVAL_MS = 5 s, which bounds the cost of a lost
# notification) and SpinCondition.record_read() drains the notify socket every 2 s (SHM_NOTIFY_POLL_INTERVAL_S). The optional C
# spin-loop extension (VLLM_USE_SPINLOOP_EXT, default 0) would spin up to 100 ms per wait before the SpinCondition is
# reached; this script warns when it is on. The edit is applied by files/p001_apply.py (one anchored edit, AST-checked:
# the anchor must be the default of SpinCondition.__init__'s busy_loop_s parameter, wait() must still switch on it).
# The file is pinned by sha256 (nightly original and patch-001 result; any other content is refused: fail closed, no half
# state). Messages and the `busy_loop_s = 0.002` verification line (a launch/readiness.txt `require` rule) are unchanged.
#
# Applied from the recipe entrypoint before exec vllm. Idempotent.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
SPIN_WAIT_FILE="$VLLM_ROOT/distributed/device_communicators/shm_broadcast.py"
APPLIER="$SRC_DIR/p001_apply.py"
READY_NEW="busy_loop_s: float = 0.002"

# pins (sha256): the file in the nightly image (vllm/vllm-openai nightly-e4a5b5537344, arm64), its patch-001 result,
# and the vendored applier
SHM_ORIG=7f253343529f09de0a6b2365d7719ac03a532756fd2439cef7983fbb3e834563
SHM_P001=9f738c7ec38112568198d54bb38a896989782cc3a27b1ad4bce0a864487a62ab
APPLIER_SHA=a28dfceb687458486a80957b74e37f5533b8106a212c35ec49833b8c035e28c3

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[001] FATAL: $*" >&2; exit 1; }

if [ ! -f "$SPIN_WAIT_FILE" ]; then
  echo "ERROR: vLLM shm_broadcast.py not found at $SPIN_WAIT_FILE" >&2
  exit 1
fi

echo "=== Hotfix: GB10 vLLM IPC spin-wait (CPU load / SoC heat, issue #79) ==="
echo "vLLM file: $SPIN_WAIT_FILE"

[ -f "$APPLIER" ] || fail "$APPLIER not found"
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p001_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
now="$(sha "$SPIN_WAIT_FILE")"
if [ "$now" = "$SHM_P001" ]; then
  echo " [skip] busy_loop_s already set to 0.002"
elif [ "$now" = "$SHM_ORIG" ]; then
  python3 "$APPLIER" apply "$SPIN_WAIT_FILE" || fail "p001_apply.py refused shm_broadcast.py"
  got="$(sha "$SPIN_WAIT_FILE")"
  [ "$got" = "$SHM_P001" ] || fail "patched shm_broadcast.py sha256 $got != pinned $SHM_P001"
else
  fail "shm_broadcast.py sha256 $now is neither the nightly e4a5b55 original ($SHM_ORIG) nor patch 001's result ($SHM_P001); refusing (a different vLLM build? re-pin after reading the diff)"
fi

case "${VLLM_USE_SPINLOOP_EXT:-0}" in
  ''|0) ;;
  *) echo " [WARN] VLLM_USE_SPINLOOP_EXT='${VLLM_USE_SPINLOOP_EXT}': vLLM's C spin-loop extension polls the shared-memory flags for up to 100 ms per wait before the SpinCondition (the 2 ms value patched here) is consulted; the CPU saving of this patch is reduced" >&2 ;;
esac

echo "=== Verification ==="
if grep -qF "$READY_NEW" "$SPIN_WAIT_FILE"; then
  echo "[OK] busy_loop_s = 0.002 — IPC wait sleeps after 2ms instead of spinning 1s."
else
  echo "[FAIL] busy_loop_s not set to 0.002" >&2
  exit 1
fi
