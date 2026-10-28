#!/usr/bin/env bash
# 012-gb10-oneshot-allreduce.sh — opt-in one-shot all-reduce for small TP messages (VLLM_GB10_ONESHOT_AR=1).
#
# WHY: decode all-reduces (~90 per TP=8 step: the embedding, then the attention output projection and the MoE combine of
# every layer) are latency-bound: NCCL's tree makes ~6 sequential host-staged hops over 100 GbE (no GPUDirect RDMA on
# GB10). Grouped send/recv to all peers + a local fp32 sum is one network step. Measured in CUDA-graph replay (nccl-tests -G,
# 8 nodes): tree 79 / 98 / 121 / 124 us vs one-shot 52 / 49 / 64 / 86 us at 8 / 16 / 32 / 64 KiB. For DeepSeek-V4.1
# (hidden size 5120, one token = 10 KiB of bf16) the default limit of 64 KiB covers decode steps of up to 6 tokens, i.e.
# one request with DSpark's 5 draft tokens (60 KiB); larger decode batches and prefill chunks stay on NCCL.
#
# WHAT: installs files/gb10_oneshot_ar.py as vllm/_gb10_oneshot_ar.py and wires two anchored edits (files/p012_apply.py) into
# vllm/distributed/device_communicators/cuda_communicator.py:
#   __init__    after the PyNcclCommunicator block: self._gb10_oneshot = maybe_create(...) (None unless the env var is 1
#               and the group is the TP group; the constructor runs a collective, self-checking warm-up)
#   all_reduce  first statement: bf16 contiguous tensors <= VLLM_GB10_ONESHOT_AR_MAX_BYTES (default 65536) take the one-shot
# Inert without VLLM_GB10_ONESHOT_AR=1 (maybe_create returns None, all_reduce unchanged). Numerics: canonical rank-order fp32
# sum, one bf16 rounding, bit-identical on all ranks — different from NCCL's per-hop bf16 adds, so greedy outputs are not
# bit-equal to earlier runs (nor to a stock pynccl all-reduce); gate with logprob/KL + quality checks.
# Fail-closed (pinned applier / module / target shas, anchors must match exactly once, AST placement checked, patched files
# must parse), idempotent (the patched file's sha).
#
# vLLM nightly e4a5b55 PORT (vllm/vllm-openai nightly, 0.32.1rc1.dev466). Both anchors are unchanged and unique:
#   __init__    `if is_symmetric_memory_enabled(): register_nccl_symmetric_ops(self.pynccl_comm)` (lines 112-113)
#   all_reduce  `def all_reduce(self, input_):` (line 362)
# CudaCommunicator.all_reduce is the dispatch chain introduced in v0.32.0 (NCCL symmetric memory, quick reduce, FlashInfer
# PCIe-IPC, FlashInfer all-reduce [default ON: VLLM_ALLREDUCE_USE_FLASHINFER=1], AITER, custom all-reduce, torch symm mem,
# pynccl); the nightly only changed the symmetric-memory scratch handling of the file. The hook is the FIRST statement, so an
# eligible message (TP group, bf16, contiguous, <= 64 KiB) never reaches any of them. Everything else (prefill-sized bf16
# messages, other dtypes) does: with VLLM_ALLREDUCE_USE_FLASHINFER=0 (set by the recipe) CudaCommunicator.__init__ builds no
# FlashInferAllReduce (use_flashinfer_allreduce False -> fi_ar_comm None -> use_fi_ar False) and with
# --disable-custom-all-reduce no ca_comm, so the chain ends in pynccl.all_reduce; with the default (1) a multi-node TP
# group would get FlashInfer's MNNVL workspace attempt (needs NVSwitch multicast, absent on GB10): this script prints a
# [WARN] line then (the boot is not refused). Every TP all-reduce of the DeepSeek-V4.1 stock path reaches the hook:
# RowParallelLinear and the MoE runner call tensor_model_parallel_all_reduce -> GroupCoordinator.all_reduce ->
# CudaCommunicator.all_reduce (the model's fused mHC all-reduce needs TP=4 and NVLink multicast, so it is off for TP=8 on GB10).
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
COMM="$VLLM_ROOT/distributed/device_communicators/cuda_communicator.py"
APPLIER="$SRC_DIR/p012_apply.py"
MODULE_SRC="$SRC_DIR/gb10_oneshot_ar.py"
MODULE_DST="$VLLM_ROOT/_gb10_oneshot_ar.py"
MARKER="# [patch-012 gb10-oneshot-ar]"

# pins (sha256): cuda_communicator.py of the nightly image (vllm/vllm-openai nightly-e4a5b5537344, arm64) and its patch-012
# result; the vendored applier and module
COMM_ORIG=c4661aa7e3a393e9ea707f6bae016ed5483536cce9ed1a2b38b884c5f865ecc8
COMM_P012=375950851b7d772782db591036841e4c33964cda321275e9babdf055fb59f11f
APPLIER_SHA=9848009271df1f191ee1d10033b9d84079cd89a24ddcd7fddb5a3059fd6c3721
MOD_SHA=7aca8e954e3866b414ba10923d7c5c4fcd00e11667ed55450a21da4f97cdd047

sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fail() { echo "[FAIL] [012] $*" >&2; exit 1; }

echo "=== Patch 012: GB10 one-shot all-reduce for small TP messages (opt-in VLLM_GB10_ONESHOT_AR=1; now: ${VLLM_GB10_ONESHOT_AR:-unset}) ==="
[ -f "$COMM" ] || { echo "[FAIL] $COMM not found" >&2; exit 1; }
[ -f "$MODULE_SRC" ] || { echo "[FAIL] $MODULE_SRC not found" >&2; exit 1; }
[ -f "$APPLIER" ] || { echo "[FAIL] $APPLIER not found" >&2; exit 1; }
[ "$(sha "$APPLIER")" = "$APPLIER_SHA" ] || fail "files/p012_apply.py sha256 $(sha "$APPLIER") != pinned $APPLIER_SHA"
[ "$(sha "$MODULE_SRC")" = "$MOD_SHA" ] || fail "files/gb10_oneshot_ar.py sha256 $(sha "$MODULE_SRC") != pinned $MOD_SHA"
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MODULE_SRC" || fail "$MODULE_SRC does not parse"

now="$(sha "$COMM")"
mode=""
if [ "$now" = "$COMM_P012" ]; then
  mode=installed
elif [ "$now" = "$COMM_ORIG" ]; then
  mode=apply
else
  if grep -qF "$MARKER" "$COMM"; then
    fail "cuda_communicator.py carries a patch-012 edit that is not this revision's pinned result (sha ${now:0:16}...); recreate the container from the image"
  fi
  fail "cuda_communicator.py sha256 $now is neither the nightly e4a5b55 original ($COMM_ORIG) nor patch 012's result ($COMM_P012); refusing (a different vLLM build? re-pin after reading the diff)"
fi

if [ "${VLLM_GB10_ONESHOT_AR:-0}" = 1 ] && [ "${VLLM_ALLREDUCE_USE_FLASHINFER:-1}" != 0 ]; then
  echo " [WARN] VLLM_GB10_ONESHOT_AR=1 but VLLM_ALLREDUCE_USE_FLASHINFER='${VLLM_ALLREDUCE_USE_FLASHINFER:-unset (default 1)}':" \
       "messages the one-shot does not take (prefill-sized, non-bf16) may be routed to FlashInfer's all-reduce, which a multi-node TP group on GB10 cannot use; set -e VLLM_ALLREDUCE_USE_FLASHINFER=0" >&2
fi

install -m 0644 "$MODULE_SRC" "$MODULE_DST"
[ "$(sha "$MODULE_DST")" = "$MOD_SHA" ] || fail "installed $MODULE_DST sha256 mismatch"
echo " [OK] installed $(basename "$MODULE_DST") (sha256 $(sha "$MODULE_DST" | cut -c1-12))"

if [ "$mode" = installed ]; then
  echo " [skip] cuda_communicator.py already wired"
else
  python3 "$APPLIER" apply "$COMM" || fail "p012_apply.py refused cuda_communicator.py"
  got="$(sha "$COMM")"
  [ "$got" = "$COMM_P012" ] || fail "patched cuda_communicator.py sha256 $got != pinned $COMM_P012"
fi

echo "=== Verification ==="
[ "$(grep -cF "$MARKER" "$COMM")" = 2 ] || { echo "[FAIL] expected 2 markers in $COMM" >&2; exit 1; }
python3 -c "import ast; ast.parse(open('$COMM').read()); ast.parse(open('$MODULE_DST').read())" \
  || { echo "[FAIL] patched files do not parse" >&2; exit 1; }
echo "[OK] patch 012 in place (active only when VLLM_GB10_ONESHOT_AR=1)"
