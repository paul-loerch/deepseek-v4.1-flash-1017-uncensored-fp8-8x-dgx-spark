#!/usr/bin/env bash
# 028-prefill-sp-overlap.sh (revision 5) -- two-half communication/compute overlap for the patch-022 sequence-parallel prefill region of
# DeepSeek-V4.1 (opt-in VLLM_GB10_PREFILL_SP_OVERLAP=1; needs VLLM_GB10_PREFILL_SP=1 and patch 022; NUMERICS: the same math per row,
# only the NCCL reduction order / GEMM tiling can differ, as in 022). Installs vllm/_prefill_sp_overlap.py and applies the anchored
# edits of models/deepseek_v41/nvidia/model.py (files/p028_prefill_sp_overlap.py) and v1/worker/gpu/model_runner.py
# (files/p028_runner.py) through files/xpatch.py. Fail-closed, idempotent. Why and what: files/prefill_sp_overlap.py.
#
# Revision 5 (no new setting): the boot self-test holds ONE region result as its reference at any time (as revision 4 did) and now compares in
# budgeted chunks (<= 16 MiB of temporaries per call instead of ~300 MiB), so its memory above a normal forward is about one region result
# (~0.45 GiB at 8192 rows, +80 MiB per aux tap; revision 4: ~0.77 GiB). Every rank logs one line "[patch-028] self-test memory (rank N, X s):
# MemAvailable start / lowest (phase) / end, /proc/meminfo deltas, allocator allocated / reserved, child processes" next to the ACTIVE line.
#
# With VLLM_GB10_PREFILL_SP_OVERLAP unset/0 NOTHING is touched (model.py / model_runner.py stay exactly as the earlier patches leave
# them). If a reused container already carries the edits, their import-time / per-step gates keep the 022 path.
#
# Set the SAME flags on EVERY rank (M_DOCKER_EXTRA): a rank without the gate would never join the pipeline's collectives; the
# fingerprint exchanged at the first eligible forward refuses mismatching ranks.
#
#   VLLM_GB10_PREFILL_SP_OVERLAP=1               on (unset/0/off: inert)
#   VLLM_GB10_PREFILL_SP_OVERLAP_MIN_T=2048      smallest forward (rows) that may run two-half; 2048 .. 1048576
#   VLLM_GB10_PREFILL_SP_OVERLAP_CTL=<file>      runtime switch: rank 0 reads it at every eligible forward (on|off [log=N])
#   VLLM_GB10_PREFILL_SP_OVERLAP_LOG_EVERY=20    INFO stats every N regions (0 = never)
#   VLLM_GB10_PREFILL_SP_OVERLAP_PRIORITY=-1     CUDA priority of the communication stream (-8 .. 0, lower = higher)
#   VLLM_GB10_PREFILL_SP_OVERLAP_RTOL=0.10       self-test tolerance (RMS-relative error of every carried state)
#   VLLM_GB10_PREFILL_SP_OVERLAP_SELFTEST_REPEAT=1   run BOTH paths twice (the run-to-run noise floor of the plain and of the overlap path is
#                                                reported next to the overlap-vs-plain error; needed by CRITERION=noise)
#   VLLM_GB10_PREFILL_SP_OVERLAP_CRITERION=absolute  self-test verdict: absolute (RMS <= RTOL, default) | noise (RMS <= max(0.02, NOISE_K x the
#                                                run-to-run noise of the same path on the same inputs), per state and per half)
#   VLLM_GB10_PREFILL_SP_OVERLAP_NOISE_K=2       criterion=noise: allowed multiple of the run-to-run noise (1 .. 100)
#   VLLM_GB10_PREFILL_SP_OVERLAP_SELFTEST=enforce  enforce (default: a failed self-test refuses to serve) | report (DIAGNOSIS ONLY: a purely
#                                                numeric failure is logged with the full report and the overlap stays on, unverified)
#   VLLM_GB10_PREFILL_SP_OVERLAP_TRACE=1         per-layer divergence trace of the self-test runs (used by the failure report)
#   VLLM_GB10_PREFILL_SP_OVERLAP_MEM_GUARD_GIB=2 free GPU memory required before the per-half builder copies are created
#   VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER=off     (revision 3) indexer row sharding (attention_config.indexer_tp_shard_min_rows <= max_num_batched_tokens):
#                                                off (default) = refused at the preflight, as in revision 2 | comm2 = PrefillRowShard.exchange_topk runs on
#                                                its OWN NCCL communicator (second PyNcclCommunicator over the TP group, compute stream only) | exclusive =
#                                                sharded forwards run the plain 022 region, unsharded ones two-half (no new communicator)
#   VLLM_GB10_PREFILL_SP_OVERLAP_IDX_MIN_POS=0   (revision 3, comm2 | exclusive) depth gate on vLLM's sharding decision: rows are sharded only when
#                                                the step's mean number of keys per row (tokens) reaches this value; 0 = vLLM's own rule; exclusive
#                                                without a value uses 200000. Needs INDEXER=comm2 or exclusive. Same value on every rank.
#   VLLM_GB10_PREFILL_SP_OVERLAP_POOL=compute    (revision 4) allocator placement of the pipeline's large buffers: compute (default) = the gathered
#                                                inputs / aux taps / exit states and the padded scatter inputs come from the COMPUTE stream's pool (the
#                                                collective still runs on the communication stream) | comm = revision 3's placement of those buffers (the
#                                                communication stream's own pool, the gathered input protected by record_stream as in revision 3); the
#                                                attention / MoE outputs are held in the keep lists (no record_stream) in both settings
#   VLLM_GB10_PREFILL_SP_OVERLAP_EMPTY_CACHE=0   (revision 4) 1 = torch.cuda.empty_cache() at the end of every overlapped region (emergency fallback
#                                                for MemAvailable pressure; synchronises the device and drops the compute pool's cached blocks too)
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
MODEL="$VLLM_ROOT/models/deepseek_v41/nvidia/model.py"
RUNNER="$VLLM_ROOT/v1/worker/gpu/model_runner.py"
MOD_SRC="$D/prefill_sp_overlap.py"
MOD_DST="$VLLM_ROOT/_prefill_sp_overlap.py"
MARKER="# [patch-028 prefill-sp-overlap]"
MODEL_MARKERS=4
RUNNER_MARKERS=1
fail() { echo "[028] FATAL: $*" >&2; exit 1; }
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# Read-only dependencies: the pipeline mirrors / relies on these exact files of the running image (after patches 001-026) and on the
# patch-022 module. A different build is refused instead of being served with an unverified data flow. relative path sha256
DEPS="
_prefill_sp.py a47cc1adf642b370782169703fb4b3ad4e19835f96f733c90a825023d8795753
models/deepseek_v41/common/engram.py d9af3da522740e81df9907b298567ec8fc6024cd851470c40e01e30907d9bbf6
models/deepseek_v41/common/engram_nvme.py af31cdf72f392f76221fbd812e68217023011aa1d25ea63d0b0bfc2ce6f84153
models/deepseek_v41/common/ops/fused_compress_quant_cache.py fd50982063f3d1e5f05f08fd627c9a347d75718ac9057e359d7eff4cdc091e82
models/deepseek_v41/attention.py 6f288f1a1393378eafaef16483e0d6ee13c5ae1ff63d9552453a34237919c371
models/deepseek_v41/compressor.py 9ff13496756830cf0ca037833fabd4c875c986fce78dbebdedb50f756c11905b
models/deepseek_v41/sparse_mla.py 6e74963e73ca09181a4ae79ee7e7df749aaa244583d1e7ab7edcca00c5df6cc7
models/deepseek_v41/decoder_replay_layers.py 0038dd2ee11cfd3234c89226b772db958388216d5f3a7bd74b33c8d22dd03880
models/deepseek_v41/nvidia/model_state.py 91bc578f5a856a956c6b5b224641b7809bf6282891be224e6d668d00c0f2f5cb
models/deepseek_v41/nvidia/flashinfer_sparse.py 4a4f0b3994883edff668023dc137298816bfadcd22be4831a93f2f3fbc8c716a
models/deepseek_v41/nvidia/ops/mhc.py c8233a70004719017ff6a5826ca538c92ea6319a9c7c0090dc8fc2e603db25e7
models/deepseek_v41/nvidia/ops/o_proj.py b8a2c9358c6cc70b279f4d41647956a32c2cc83b9d53545086d0724cd30b8403
models/deepseek_v4/nvidia/ops/o_proj.py 05be23af9251f6c572da741a1949bfcf8ea012e62d90e326aab1fc91e1a0abee
models/deepseek_v4/nvidia/model.py 7f27d2167dc225d2f7d773cb5256e4a1736fdee14c09d2c204f5ecdbabd60c7f
model_executor/layers/linear.py d4b97d850eb8bdee5fd2459e7ac688d00623b54f5ec0ea590c08da0173f8b4d4
model_executor/layers/sparse_attn_indexer.py ed9cfdd83544d15c088b052e287971610d0de96062493b6c757d16ea2248c6aa
model_executor/layers/fused_moe/runner/moe_runner.py 64eee9565bdb2c599c8f3341c24e50062b80b9e625e544db44250372311289ef
model_executor/layers/fused_moe/config.py 9da7b7b7f825e466792fc47efad92d47c46716a2e5774ec112d945c5ada5d2ad
model_executor/kernels/mhc/tilelang.py f9b13bf93ce42426d2137e1065b2d0bee36c17d3b6d7657e1a959e9e4365137d
v1/worker/gpu/ubatch_utils.py b6de38677e73d79929761f479b54777ce93724f53117aea32bbd9e19f176d0a7
v1/worker/gpu/attn_utils.py 705a1eb16d26dc0ec658af9e10e00831f16d1925c14361228853807e456f6553
v1/worker/gpu/model_states/default.py 453eb32c5efef9f4064c333c3e5e714aef3adc16645db1f4a311f8a397860f02
v1/worker/gpu/input_batch.py eb269588b889c8fa9f76b4bbc03432d4fcbb8920a40e7b62268e476a2dc50c52
v1/worker/ubatch_utils.py 8aef5a51c45d603ac1408d1224301f0cd536d59698975d704f53af84410453dd
v1/worker/utils.py 85b23497017e5db07ea8f3fa56807c44dd7dd269f1977031bd86acdadfb0b5ec
v1/attention/backend.py 1583b551b8996887d7ad372763f2de1bbb2143e1153329b261128646ff24b3af
v1/attention/backends/mla/sparse_swa.py 56594499e686d287e4fbfd2586591f91cfbee9ad62976040e166d022373a3167
v1/attention/backends/mla/indexer.py 43dbebe3ec26e04a005472a89fa4d73914e285809ca79ea874b007dcadd8f004
v1/attention/backends/mla/compressor_utils.py 94966398f822512ac35968ca5d1634e707deaa198c68f1e145befc12bdde379e
forward_context.py b0329d052897f747bf0c1d9c7c55a9cff678e18cbaa2c0ce1aa4203dc333d9bc
utils/torch_utils.py 019077728fcf8ab898bd665c2260ffda6a7c8882319ebe0568ed05304cb14a7b
distributed/device_communicators/cuda_communicator.py 4c8688519c281bd88e6971027284ca15cce10538dc9a31bf6e0b0be555867cce
"
# the Engram all-to-all module is read through _prefill_sp only in VLLM_GB10_PREFILL_SP_ENGRAM=a2a mode
ENGRAM_A2A_SHA="710361d171a24442b8dca7b42014b8fa2f0b2c267079515d368c96dde62e838f"
# revision 3, INDEXER=comm2|exclusive only: the NCCL wrapper the second communicator is an instance of (the image's file; no patch edits it)
IDX_DEPS="
distributed/device_communicators/pynccl.py 35ffa32408c675248e07016ca936eb73e2423777a67572d145b8fc1176f173f8
"
RUNNER_PIN_026="cd6f9a44a53758aa763f40c2abe6b4021bff83757bcc3bf4d12dab756d9512d9"
RUNNER_PIN_027="212fc051c70f72521a04720fe2557d15f37769659ae6fd693288c516aae7c4f3"

echo "=== Patch 028: DeepSeek-V4.1 prefill two-half comm/compute overlap (VLLM_GB10_PREFILL_SP_OVERLAP=${VLLM_GB10_PREFILL_SP_OVERLAP:-unset}, SP=${VLLM_GB10_PREFILL_SP:-unset}) ==="
mode_raw="$(printf '%s' "${VLLM_GB10_PREFILL_SP_OVERLAP:-0}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$mode_raw" in
  ''|0|off|false|no)
    if { [ -f "$MODEL" ] && grep -qF "$MARKER" "$MODEL" 2>/dev/null; } || { [ -f "$RUNNER" ] && grep -qF "$MARKER" "$RUNNER" 2>/dev/null; }; then
      echo " [note] VLLM_GB10_PREFILL_SP_OVERLAP unset but the files already carry patch 028 (reused container?);"
      echo "        its import-time / per-step gates keep the patch-022 path."
    fi
    echo "[OK] patch 028 skipped (VLLM_GB10_PREFILL_SP_OVERLAP unset); files untouched"
    exit 0 ;;
  1|on|true|yes) ;;
  *) fail "VLLM_GB10_PREFILL_SP_OVERLAP='${VLLM_GB10_PREFILL_SP_OVERLAP}' must be unset/0 (inert) or 1 (on)" ;;
esac

sp_raw="$(printf '%s' "${VLLM_GB10_PREFILL_SP:-0}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$sp_raw" in 1|on|true|yes) ;; *) fail "VLLM_GB10_PREFILL_SP_OVERLAP=1 needs VLLM_GB10_PREFILL_SP=1 (the overlap runs inside the patch-022 region)";; esac

intvar() {  # name default lo hi
  local v="${!1:-$2}"
  [[ "$v" =~ ^-?[0-9]{1,12}$ ]] || fail "$1='$v' must be an integer"
  { [ "$v" -ge "$3" ] && [ "$v" -le "$4" ]; } || fail "$1=$v outside [$3, $4] (default $2)"
}
intvar VLLM_GB10_PREFILL_SP_OVERLAP_MIN_T 2048 2048 1048576
intvar VLLM_GB10_PREFILL_SP_OVERLAP_LOG_EVERY 20 0 1073741824
intvar VLLM_GB10_PREFILL_SP_OVERLAP_PRIORITY -1 -8 0
intvar VLLM_GB10_PREFILL_SP_OVERLAP_SELFTEST_REPEAT 1 0 1
intvar VLLM_GB10_PREFILL_SP_OVERLAP_TRACE 1 0 1
intvar VLLM_GB10_PREFILL_SP_OVERLAP_IDX_MIN_POS 0 0 16777216
intvar VLLM_GB10_PREFILL_SP_OVERLAP_EMPTY_CACHE 0 0 1
pool_raw="$(printf '%s' "${VLLM_GB10_PREFILL_SP_OVERLAP_POOL:-compute}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$pool_raw" in ''|compute) pool_raw=compute ;; comm) ;;
  *) fail "VLLM_GB10_PREFILL_SP_OVERLAP_POOL='${VLLM_GB10_PREFILL_SP_OVERLAP_POOL}' must be compute (default) or comm" ;; esac
idx_raw="$(printf '%s' "${VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER:-off}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$idx_raw" in ''|off|0|false|no) idx_raw=off ;; comm2|exclusive) ;;
  *) fail "VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER='${VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER}' must be off (default), comm2 or exclusive" ;; esac
if [ "$idx_raw" = off ] && [ -n "${VLLM_GB10_PREFILL_SP_OVERLAP_IDX_MIN_POS:-}" ] && [ "${VLLM_GB10_PREFILL_SP_OVERLAP_IDX_MIN_POS}" != 0 ]; then
  fail "VLLM_GB10_PREFILL_SP_OVERLAP_IDX_MIN_POS=${VLLM_GB10_PREFILL_SP_OVERLAP_IDX_MIN_POS} needs VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER=comm2 or exclusive"
fi
st_raw="$(printf '%s' "${VLLM_GB10_PREFILL_SP_OVERLAP_SELFTEST:-enforce}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$st_raw" in ''|enforce|report) ;; *) fail "VLLM_GB10_PREFILL_SP_OVERLAP_SELFTEST='${VLLM_GB10_PREFILL_SP_OVERLAP_SELFTEST}' must be enforce or report" ;; esac
cr_raw="$(printf '%s' "${VLLM_GB10_PREFILL_SP_OVERLAP_CRITERION:-absolute}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$cr_raw" in ''|absolute|noise) ;; *) fail "VLLM_GB10_PREFILL_SP_OVERLAP_CRITERION='${VLLM_GB10_PREFILL_SP_OVERLAP_CRITERION}' must be absolute or noise" ;; esac
[[ "${VLLM_GB10_PREFILL_SP_OVERLAP_NOISE_K:-2}" =~ ^[0-9]*\.?[0-9]+$ ]] || fail "VLLM_GB10_PREFILL_SP_OVERLAP_NOISE_K must be a number"
if [ "$cr_raw" = noise ] && [ "${VLLM_GB10_PREFILL_SP_OVERLAP_SELFTEST_REPEAT:-1}" != 1 ]; then
  fail "VLLM_GB10_PREFILL_SP_OVERLAP_CRITERION=noise needs VLLM_GB10_PREFILL_SP_OVERLAP_SELFTEST_REPEAT=1 (the noise floor comes from the repeated runs)"
fi
[[ "${VLLM_GB10_PREFILL_SP_OVERLAP_RTOL:-0.10}" =~ ^[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$ ]] || fail "VLLM_GB10_PREFILL_SP_OVERLAP_RTOL must be a number"
[[ "${VLLM_GB10_PREFILL_SP_OVERLAP_MEM_GUARD_GIB:-2}" =~ ^[0-9]*\.?[0-9]+$ ]] || fail "VLLM_GB10_PREFILL_SP_OVERLAP_MEM_GUARD_GIB must be a number"

for f in "$MODEL" "$RUNNER" "$MOD_SRC" "$D/p028_prefill_sp_overlap.py" "$D/p028_runner.py" "$D/xpatch.py"; do [ -f "$f" ] || fail "$f not found"; done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$MOD_SRC" || fail "$MOD_SRC does not parse"

# patch 022 must be in place (its markers in model.py, its module installed)
n22="$(grep -cF '# [patch-022 prefill-sp]' "$MODEL" || true)"
[ "$n22" = 4 ] || fail "model.py carries $n22 patch-022 markers, expected 4: patch 022 (VLLM_GB10_PREFILL_SP=1) must be applied first"

while read -r rel want; do
  [ -n "$rel" ] || continue
  [ -f "$VLLM_ROOT/$rel" ] || fail "read-only dependency $rel not found"
  got="$(sha "$VLLM_ROOT/$rel")"
  [ "$got" = "$want" ] || fail "$rel sha256 $got != pinned $want (a different image build, or a patch edited it after the overlap was written against it); refusing"
done <<<"$DEPS"
em="$(printf '%s' "${VLLM_GB10_PREFILL_SP_ENGRAM:-gather}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
if [ "$em" = a2a ]; then
  [ -f "$VLLM_ROOT/_engram_sp_a2a.py" ] || fail "VLLM_GB10_PREFILL_SP_ENGRAM=a2a but vllm/_engram_sp_a2a.py is not installed"
  got="$(sha "$VLLM_ROOT/_engram_sp_a2a.py")"
  [ "$got" = "$ENGRAM_A2A_SHA" ] || fail "_engram_sp_a2a.py sha256 $got != pinned $ENGRAM_A2A_SHA"
fi
if [ "$idx_raw" != off ]; then
  while read -r rel want; do
    [ -n "$rel" ] || continue
    [ -f "$VLLM_ROOT/$rel" ] || fail "read-only dependency $rel not found (VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER=$idx_raw)"
    got="$(sha "$VLLM_ROOT/$rel")"
    [ "$got" = "$want" ] || fail "$rel sha256 $got != pinned $want (VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER=$idx_raw clones this NCCL wrapper); refusing"
  done <<<"$IDX_DEPS"
fi
echo " [OK] [028] read-only dependencies match the pinned image files"

# the model_runner.py pin depends on whether patch 027 ran before this one
if grep -qF "$MARKER" "$RUNNER"; then
  PIN=026                                    # already patched by this module: xpatch recognises the marker
else
  rsha="$(sha "$RUNNER")"
  case "$rsha" in
    "$RUNNER_PIN_026") PIN=026 ;;
    "$RUNNER_PIN_027") PIN=027 ;;
    *) fail "model_runner.py sha256 $rsha is neither the post-026 ($RUNNER_PIN_026) nor the post-027 ($RUNNER_PIN_027) file; refusing" ;;
  esac
fi

# dry runs first: both files must accept their edits before either is touched
python3 "$D/xpatch.py" check "$D/p028_prefill_sp_overlap.py" "$VLLM_ROOT" || fail "xpatch refused model.py (not the post-026 file, or an anchor moved)"
P028_RUNNER_PIN="$PIN" python3 "$D/xpatch.py" check "$D/p028_runner.py" "$VLLM_ROOT" || fail "xpatch refused model_runner.py (pin $PIN)"
install -m 0644 "$MOD_SRC" "$MOD_DST"
echo " [OK] [028] installed _prefill_sp_overlap.py"
python3 "$D/xpatch.py" apply "$D/p028_prefill_sp_overlap.py" "$VLLM_ROOT" || fail "xpatch failed on model.py after a successful dry run"
P028_RUNNER_PIN="$PIN" python3 "$D/xpatch.py" apply "$D/p028_runner.py" "$VLLM_ROOT" || fail "xpatch failed on model_runner.py after a successful dry run"

[ "$(grep -cF "$MARKER" "$MODEL")" = "$MODEL_MARKERS" ] || fail "expected $MODEL_MARKERS markers in $MODEL, found $(grep -cF "$MARKER" "$MODEL")"
[ "$(grep -cF "$MARKER" "$RUNNER")" = "$RUNNER_MARKERS" ] || fail "expected $RUNNER_MARKERS marker in $RUNNER, found $(grep -cF "$MARKER" "$RUNNER")"
python3 -c "import ast, sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" "$MODEL" "$RUNNER" "$MOD_DST" || fail "patched/installed files do not parse"
echo "[OK] patch 028 in place (runner pin $PIN, MIN_T=${VLLM_GB10_PREFILL_SP_OVERLAP_MIN_T:-2048}, control file ${VLLM_GB10_PREFILL_SP_OVERLAP_CTL:-none}, indexer row sharding: $idx_raw, pool: $pool_raw, empty_cache: ${VLLM_GB10_PREFILL_SP_OVERLAP_EMPTY_CACHE:-0}): the first eligible forward runs the self-test and logs '[patch-028] prefill two-half overlap ACTIVE'. NUMERICS: reduction order / GEMM tiling only; quality-gate before serving."
