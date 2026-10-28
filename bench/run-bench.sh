#!/usr/bin/env bash
# run-bench.sh -- the release benchmark sequence for DeepSeek-V4.1-Flash-1017 on vLLM (tensor parallel over eight DGX
# Spark nodes, OpenAI-compatible API). Run it on any machine that can reach the API; the machine only sends requests.
#
#   bench/run-bench.sh [--out DIR] [--quick] [more options, see --help]
#
# Steps, in this order (the names are for --skip / --only):
#   -         health check; record the engine version and the model
#   drift     cold-vs-warm prefix-cache drift probe                          drift-probe.py
#   -         cool-down hint; --cooldown SECONDS also waits
#   ctx       context curve: a discarded warm-up cell, then one user at 8k, 32k, 128k, 256k, 512k and 1,040,000 tokens,
#             then 32k with 2, 4 and 8 users                                 ctx-curve.py
#   decode    decode speed on recorded agent turns and on chat prompts, then the chat concurrency sweep with
#             1, 2, 3, 4, 6 and 8 users x 120 s                              arm-measure.sh
#   serve     vllm bench serve, random 8192 in / 1024 out, --ignore-eos, seed 1234 + concurrency, concurrency 1, 2, 4, 8,
#             num_prompts = max(8, 4 x concurrency), percentiles 50/90/99 of ttft, tpot, itl, e2el
#   canary    cross-request leakage / corruption canary, 8 sessions, 80 minutes   canary-leak-test.py
# Then summary.md. Use the individual tools in bench/tools directly to run one measurement; see bench/README.md.
#
# Output (DIR, default bench/results/run-<UTC time>): summary.md, run-info.json, steps.tsv, the JSON files of the tools
# (drift.json, ctx-curve.json, ctx-conc.json, decode-agent.json, decode-chat.json, decode-conc.json, vllm-bench-c*.json),
# canary/ and raw-local/ (the console log; git-ignored, may contain host names). DIR must not exist or must be empty.
#
# Prerequisites: python3 and curl on this machine, the API up and quiet (no other clients), the prepared datasets (bench/data),
# and for the serve step either a local vllm installation or a vLLM container image (both described in bench/README.md).
# Exit status: 0 all steps ran, 1 a step failed or the engine became unhealthy, 2 usage or setup error.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)       # <repo>/bench
REPO=$(dirname "$HERE")
TOOLS=$HERE/tools

BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
DATA_DIR=${BENCH_DATA_DIR:-$HERE/data/prepared}
CLUSTER_ENV=${CLUSTER_ENV:-$REPO/cluster.env}
OUT=''
QUICK=0
GUARD=0
COOLDOWN_S=${COOLDOWN_S:-0}
SKIP=''
ONLY=''
VLLM_BENCH=${VLLM_BENCH:-auto}          # auto | local | docker | skip
VLLM_BIN=${VLLM_BIN:-vllm}
BENCH_IMAGE=${BENCH_IMAGE:-}
TOK_DIR=${BENCH_TOKENIZER_DIR:-}
CLOCK_CAP=${CLOCK_CAP:-}
CANARY_BUDGET=''
CANARY_ARGS=${CANARY_ARGS:-}            # extra arguments for canary-leak-test.py, e.g. "--ca-cert ca.pem" (no spaces inside words)
ZSTOP=${ZSTOP:-88}                      # --guard: hottest thermal zone that stops ctx-curve, degrees C
MEMSTOP=${MEMSTOP:-7}                   # --guard: MemAvailable in GiB below which ctx-curve stops (this recipe's lowest while serving: 8.08; its rule: keep >= 8)

usage() {
  cat <<'EOF'
usage: run-bench.sh [--out DIR] [--quick] [options]

  --out DIR            output directory (default bench/results/run-<UTC time>); must not exist or must be empty
  --quick              smoke test: fewer and smaller cells, about 25 minutes; numbers are NOT comparable with a full run
  --base-url URL       API root without /v1 (default $BASE_URL, else http://127.0.0.1:8000)
  --data-dir DIR       prepared datasets (default $BENCH_DATA_DIR, else bench/data/prepared)
  --skip LIST          comma-separated steps to leave out: drift, ctx, decode, serve, canary
  --only LIST          run only these steps
  --clock-cap TEXT     GPU clock cap to record in the summary, e.g. "2200 MHz" (it is not detected; default $CLOCK_CAP)
  --cooldown SECONDS   wait this long at the cool-down hint (default 0: print the hint only)
  --guard              ctx-curve watches node temperature and memory over ssh and stops the run; arm-measure.sh snapshots
                       the RoCE / link counters (needs cluster.env with the node list, see bench/README.md)
  --cluster-env FILE   cluster description for --guard and for the tokenizer default (default $CLUSTER_ENV, else <repo>/cluster.env)
  --vllm-bench MODE    how to run vllm bench serve: auto (default), local, docker or skip
  --image IMAGE        vLLM container image for --vllm-bench docker (default $BENCH_IMAGE, else M_IMAGE of config/recipe.env)
  --tokenizer DIR      tokenizer directory for vllm bench serve (default $BENCH_TOKENIZER_DIR, else MODEL_DIR of cluster.env)
  --canary-budget S    canary run time in seconds (default 4800, quick 300); $CANARY_ARGS adds canary options
  -h, --help           this text

Environment: BASE_URL, BENCH_DATA_DIR, CLUSTER_ENV, CLOCK_CAP, VLLM_BENCH, VLLM_BIN, BENCH_IMAGE, BENCH_TOKENIZER_DIR, CANARY_ARGS,
ZSTOP and MEMSTOP (guard limits, default 88 C and 7 GiB).
EOF
}

need() { [ $# -ge 2 ] || { echo "run-bench.sh: $1 needs a value" >&2; usage >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --out) need "$@"; OUT=$2; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --base-url) need "$@"; BASE_URL=$2; shift 2 ;;
    --data-dir) need "$@"; DATA_DIR=$2; shift 2 ;;
    --skip) need "$@"; SKIP=$2; shift 2 ;;
    --only) need "$@"; ONLY=$2; shift 2 ;;
    --clock-cap) need "$@"; CLOCK_CAP=$2; shift 2 ;;
    --cooldown) need "$@"; COOLDOWN_S=$2; shift 2 ;;
    --guard) GUARD=1; shift ;;
    --cluster-env) need "$@"; CLUSTER_ENV=$2; shift 2 ;;
    --vllm-bench) need "$@"; VLLM_BENCH=$2; shift 2 ;;
    --image) need "$@"; BENCH_IMAGE=$2; shift 2 ;;
    --tokenizer) need "$@"; TOK_DIR=$2; shift 2 ;;
    --canary-budget) need "$@"; CANARY_BUDGET=$2; shift 2 ;;
    *) echo "run-bench.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
BASE_URL=${BASE_URL%/}
export BASE_URL BENCH_DATA_DIR="$DATA_DIR" CLUSTER_ENV
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

for s in $(printf '%s,%s' "$SKIP" "$ONLY" | tr ',' ' '); do
  case $s in drift | ctx | decode | serve | canary) ;; *) echo "run-bench.sh: unknown step '$s' (steps: drift, ctx, decode, serve, canary)" >&2; exit 2 ;; esac
done
case $COOLDOWN_S in '' | *[!0-9]*) echo "run-bench.sh: --cooldown must be a whole number of seconds" >&2; exit 2 ;; esac
case ${CANARY_BUDGET:-0} in *[!0-9]*) echo "run-bench.sh: --canary-budget must be a whole number of seconds" >&2; exit 2 ;; esac
case $VLLM_BENCH in auto | local | docker | skip) ;; *) echo "run-bench.sh: --vllm-bench must be auto, local, docker or skip" >&2; exit 2 ;; esac
command -v python3 >/dev/null 2>&1 || { echo "run-bench.sh: python3 is needed on this machine" >&2; exit 2; }
command -v curl >/dev/null 2>&1 || { echo "run-bench.sh: curl is needed on this machine" >&2; exit 2; }

# ---- the two profiles ----------------------------------------------------------------------------------------------
if [ "$QUICK" = 1 ]; then
  PROFILE=quick
  DRIFT_DOCS=1
  CTX_CELLS=8192x1,32768x1
  CONC_CELLS=32768x2
  DEC_REPS=1
  SWEEP_LEVELS=1,2,4
  SWEEP_SECS=30
  SERVE_LEVELS='1 4'
  CANARY_SESSIONS=4
  CANARY_BUDGET=${CANARY_BUDGET:-300}
  CANARY_GRACE=120
else
  PROFILE=full
  DRIFT_DOCS=4
  CTX_CELLS=8192x1,32768x1,131072x1,262144x1,524288x1,1040000x1
  CONC_CELLS=32768x2,32768x4,32768x8
  DEC_REPS=2
  SWEEP_LEVELS=1,2,3,4,6,8
  SWEEP_SECS=120
  SERVE_LEVELS='1 2 4 8'
  CANARY_SESSIONS=8
  CANARY_BUDGET=${CANARY_BUDGET:-4800}
  CANARY_GRACE=900
fi
CTX_MAX_TOKENS=512

# ---- output directory and logging ------------------------------------------------------------------------------------
[ -n "$OUT" ] || OUT=$HERE/results/run-$(date -u +%Y%m%d-%H%M%S)
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "run-bench.sh: $OUT exists and is not empty; choose another --out" >&2; exit 2
fi
mkdir -p "$OUT/raw-local" || { echo "run-bench.sh: cannot create $OUT" >&2; exit 2; }
OUT=$(cd "$OUT" && pwd)
LOG=$OUT/raw-local/run.log
: > "$OUT/steps.tsv"
say()  { printf '%s %s\n' "$(date +%T)" "$*" | tee -a "$LOG"; }
warn() { printf '%s WARN  %s\n' "$(date +%T)" "$*" | tee -a "$LOG" >&2; }
abort_setup() {   # abort_setup <message>: nothing has been measured yet, so leave no output directory behind
  echo "run-bench.sh: $*" >&2
  rm -rf "$OUT/raw-local" "$OUT/steps.tsv" "$OUT/run-info.json"
  rmdir "$OUT" 2>/dev/null
  exit "${ABORT_RC:-2}"
}
want() {   # want <step>: is this step selected?
  case ",$SKIP," in *",$1,"*) return 1 ;; esac
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0 ;; esac
  return 1
}
health() { curl -sf -m 5 "$BASE_URL/health" >/dev/null 2>&1; }

STATUS=0
FINISHED=0
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$OUT/steps.tsv"; }   # record <name> <result> <rc> <seconds>
finish() {   # finish <exit status>: always leave a summary behind
  [ "$FINISHED" = 1 ] && exit "$1"
  FINISHED=1
  python3 "$TOOLS/summarize.py" "$OUT" 2>&1 | tee -a "$LOG"
  say "done: $OUT/summary.md (exit status $1)"
  exit "$1"
}
trap 'say "interrupted"; record interrupted "stopped by a signal" 130 0; finish 130' INT TERM

describe() {   # describe <kind> <rc>: the result text of a step
  case "$1:$2" in
    *:0) echo ok ;;
    ctx:2) echo "stopped by the guard or by the cell time limit (exit 2)" ;;
    ctx:3) echo "stalled: possible wedge, check the engine (exit 3)" ;;
    canary:1) echo "FAIL: leak, degenerate output or warm/cold divergence (exit 1)" ;;
    canary:2) echo "WARN: findings, or too few requests to certify anything; see canary/summary.txt (exit 2)" ;;
    canary:3) echo "ERROR: harness or server error (exit 3)" ;;
    *) echo "failed (exit $2)" ;;
  esac
}

run_step() {   # run_step <name> <kind> <command...>: run, log, record
  local name=$1 kind=$2 t0 rc res
  shift 2
  say "== $name"
  t0=$(date +%s)
  "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  res=$(describe "$kind" "$rc")
  case "$kind:$rc" in *:0 | canary:2) ;; *) STATUS=1 ;; esac
  record "$name" "$res" "$rc" "$(( $(date +%s) - t0 ))"
  say "-- $name: $res"
  return 0
}

gate() {   # gate <where>: stop everything when the engine is gone
  health && return 0
  say "ENGINE NOT HEALTHY after $1 -- stopping. Look at launch/status.sh and the rank logs before restarting."
  record "health after $1" "FAILED" 1 0
  finish 1
}

get_facts() {   # one JSON line: engine version, served model, max_model_len
  python3 - "$BASE_URL" <<'PY'
import json, sys, urllib.request
base = sys.argv[1].rstrip("/")
def get(path):
    try:
        return json.loads(urllib.request.urlopen(base + path, timeout=15).read())
    except Exception:
        return {}
m = (get("/v1/models").get("data") or [{}])[0]
print(json.dumps({"engine_version": get("/version").get("version", ""), "model": m.get("id", ""),
                  "max_model_len": m.get("max_model_len")}))
PY
}

filter_cells() {   # filter_cells <cells> <max_tokens>: line 1 = the cells that fit max_model_len, line 2 = the ones that do not
  python3 - "$1" "$MAX_MODEL_LEN" "$2" <<'PY'
import sys
cells, maxlen, mt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
keep, drop = [], []
for c in cells.split(","):
    n = int(c.split("x")[0])
    (keep if (maxlen <= 0 or n + mt + 64 <= maxlen) else drop).append(c)
print(",".join(keep))
print(",".join(drop))
PY
}

needs_w2() {   # needs_w2 <cells>: does a prompt run past the W3 text into the W2 agent traces (ctx-curve.py: > 900,000 tokens)?
  local c n u
  for c in $(printf '%s' "$1" | tr ',' ' '); do
    n=${c%x*}
    u=${c#*x}
    if [ $(( (u - 1) * (n + 4096) + n + 2048 )) -gt 900000 ]; then return 0; fi
  done
  return 1
}

write_run_info() {
  python3 - "$OUT" <<'PY'
import datetime, json, os, sys
facts = json.loads(os.environ["RB_FACTS"])
ver = facts.get("engine_version") or ""
info = {"profile": os.environ["RB_PROFILE"],
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "engine": ("vLLM " + ver) if ver else "NOT RECORDED",
        "engine_version": ver, "model": facts.get("model", ""), "max_model_len": facts.get("max_model_len"),
        "clock_cap": os.environ.get("RB_CLOCK_CAP", ""), "guard": os.environ["RB_GUARD"] == "1",
        "fabric": os.environ["RB_GUARD"] == "1", "cells_skipped": [c for c in os.environ.get("RB_SKIPPED", "").split(",") if c]}
with open(os.path.join(sys.argv[1], "run-info.json"), "w") as fh:
    json.dump(info, fh, indent=1)
PY
}

# ---- step 0: health and run facts --------------------------------------------------------------------------------
say "run-bench.sh: profile $PROFILE, output $OUT"
if ! health; then
  hint=''
  if [ -r "$CLUSTER_ENV" ]; then
    p=$( . "$CLUSTER_ENV" >/dev/null 2>&1; printf '%s' "${API_PORT:-}" )
    [ -n "$p" ] && hint=" (cluster.env says API_PORT=$p)"
  fi
  ABORT_RC=1 abort_setup "no healthy server at $BASE_URL (GET /health failed)$hint. Start it with launch/up.sh --wait, or set BASE_URL / --base-url."
fi
FACTS=$(get_facts)
MODEL=$(printf '%s' "$FACTS" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("model",""))')
MAX_MODEL_LEN=$(printf '%s' "$FACTS" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("max_model_len") or 0)')
ENGINE_VERSION=$(printf '%s' "$FACTS" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("engine_version") or "unknown")')
[ -n "$MODEL" ] || ABORT_RC=1 abort_setup "GET /v1/models returned no model at $BASE_URL"

SKIPPED_CELLS=''
r=$(filter_cells "$CTX_CELLS" "$CTX_MAX_TOKENS")
CTX_CELLS=$(printf '%s\n' "$r" | sed -n 1p)
SKIPPED_CELLS=$(printf '%s\n' "$r" | sed -n 2p)
r=$(filter_cells "$CONC_CELLS" "$CTX_MAX_TOKENS")
CONC_CELLS=$(printf '%s\n' "$r" | sed -n 1p)
d=$(printf '%s\n' "$r" | sed -n 2p)
if [ -n "$d" ]; then SKIPPED_CELLS=${SKIPPED_CELLS:+$SKIPPED_CELLS,}$d; fi
if [ -n "$SKIPPED_CELLS" ] && want ctx; then warn "cells left out, the served max_model_len is $MAX_MODEL_LEN: $SKIPPED_CELLS"; fi

# ---- preflight: data, guard, serve mode ----------------------------------------------------------------------------
missing=''
if want drift || want ctx; then
  [ -f "$DATA_DIR/w3_docqa-v1.jsonl" ] || missing="$missing w3_docqa-v1.jsonl (bench/data/prep_w3_docqa.py)"
fi
if want decode; then
  [ -f "$DATA_DIR/w2_agent-v1.jsonl" ] || missing="$missing w2_agent-v1.jsonl (bench/data/prep_w2_agent.py)"
fi
if want ctx && { needs_w2 "$CTX_CELLS" || needs_w2 "$CONC_CELLS"; } && [ ! -f "$DATA_DIR/w2_agent-v1.jsonl" ]; then
  missing="$missing w2_agent-v1.jsonl (the longest prompts continue into it; bench/data/prep_w2_agent.py)"
fi
[ -z "$missing" ] || abort_setup "prepared data missing in $DATA_DIR:$missing. See bench/data/README.md, or leave the step out with --skip."

GUARD_ARGS=()
if [ "$GUARD" = 1 ]; then
  if want ctx; then
    say "guard check: reading every node once over ssh (about 15 s)"
    python3 "$TOOLS/ctx-curve.py" --guard --dry-run --cluster-env "$CLUSTER_ENV" > "$OUT/raw-local/guard-check.txt" 2>&1
    rc=$?
    tee -a "$LOG" < "$OUT/raw-local/guard-check.txt"
    [ "$rc" -eq 0 ] || abort_setup "--guard was asked for but the nodes cannot be read (see above). Fix ssh access and cluster.env, or run without --guard."
  fi
  GUARD_ARGS=(--guard --cluster-env "$CLUSTER_ENV" --zstop "$ZSTOP" --memstop "$MEMSTOP")
else
  say "node guard: OFF (no thermal / memory watchdog, no automatic cool-down wait; --guard turns it on, bench/README.md says what it needs)"
fi

SERVE_MODE=skip
SERVE_WHY=''
TOK=''
if want serve; then
  if [ -z "$TOK_DIR" ] && [ -r "$CLUSTER_ENV" ]; then
    TOK_DIR=$( . "$CLUSTER_ENV" >/dev/null 2>&1; printf '%s' "${MODEL_DIR:-}" )
  fi
  case $TOK_DIR in *CHANGE_ME*) TOK_DIR='' ;; esac
  if [ "$VLLM_BENCH" = skip ]; then
    SERVE_WHY="skipped on request (--vllm-bench skip)"
  elif [ -z "$TOK_DIR" ] || [ ! -d "$TOK_DIR" ]; then
    SERVE_WHY="no tokenizer directory on this machine: pass --tokenizer DIR (the checkpoint, or any directory with its tokenizer files) or set MODEL_DIR in cluster.env"
  else
    TOK=$(cd "$TOK_DIR" && pwd)
    if [ -z "$BENCH_IMAGE" ] && [ -r "$REPO/config/recipe.env" ]; then
      BENCH_IMAGE=$(sed -n 's/^M_IMAGE=//p' "$REPO/config/recipe.env" | head -1 | tr -d "'\"")
      case $BENCH_IMAGE in *@* | *' '*) BENCH_IMAGE='' ;; esac
    fi
    case $VLLM_BENCH in
      local | auto)
        if command -v "$VLLM_BIN" >/dev/null 2>&1 && "$VLLM_BIN" bench serve --help >/dev/null 2>&1; then
          SERVE_MODE=local
        elif [ "$VLLM_BENCH" = local ]; then
          abort_setup "--vllm-bench local, but '$VLLM_BIN bench serve' does not run here (set VLLM_BIN, or use docker)"
        fi ;;
    esac
    if [ "$SERVE_MODE" = skip ] && { [ "$VLLM_BENCH" = docker ] || [ "$VLLM_BENCH" = auto ]; }; then
      if command -v docker >/dev/null 2>&1 && [ -n "$BENCH_IMAGE" ]; then
        SERVE_MODE=docker
      elif [ "$VLLM_BENCH" = docker ]; then
        abort_setup "--vllm-bench docker needs docker and an image (--image IMAGE, \$BENCH_IMAGE, or M_IMAGE in config/recipe.env)"
      fi
    fi
    [ "$SERVE_MODE" != skip ] || SERVE_WHY="no usable 'vllm bench serve' on this machine: install vllm here, or pass --image for the container mode"
  fi
  if [ "$SERVE_MODE" = skip ]; then
    warn "the serve step will be SKIPPED: $SERVE_WHY"
  else
    say "serve step: $SERVE_MODE mode"
  fi
fi

RB_FACTS=$FACTS RB_PROFILE=$PROFILE RB_CLOCK_CAP=$CLOCK_CAP RB_GUARD=$GUARD RB_SKIPPED=$SKIPPED_CELLS write_run_info
say "server: model $MODEL, max_model_len $MAX_MODEL_LEN, engine vLLM $ENGINE_VERSION"
[ -n "$CLOCK_CAP" ] || warn "no --clock-cap given: the summary will say NOT RECORDED for the GPU clock cap"
record "health" ok 0 0

# ---- drift probe -------------------------------------------------------------------------------------------------------
if want drift; then
  run_step "drift probe" drift python3 "$TOOLS/drift-probe.py" "$OUT/drift.json" --docs "$DRIFT_DOCS" \
    --base-url "$BASE_URL" --data "$DATA_DIR/w3_docqa-v1.jsonl"
  gate "the drift probe"
fi

# ---- cool-down hint ------------------------------------------------------------------------------------------------------
if want ctx; then
  say "cool-down: the long-context cells heat the nodes. Before they start every node should be back at idle (in our measurements the"
  say "  hottest board sensor at 44-47 C and the GPU under 10 W; launch/status.sh shows GPU W and C per node). Back-to-back runs on warm"
  say "  nodes measure the heat, not the recipe. --guard makes ctx-curve wait for it and stop on overheating; --cooldown SECONDS pauses here."
  if [ "$COOLDOWN_S" -gt 0 ]; then say "cool-down: waiting $COOLDOWN_S s"; sleep "$COOLDOWN_S"; fi
fi

# ---- context curve --------------------------------------------------------------------------------------------------------
if want ctx; then
  if [ -z "$CTX_CELLS" ]; then
    warn "ctx step skipped: no cell fits the served max_model_len ($MAX_MODEL_LEN)"
    record "ctx" "skipped (max_model_len)" 0 0
  else
    CTX_COMMON=(--base-url "$BASE_URL" --data-dir "$DATA_DIR" ${GUARD_ARGS[@]+"${GUARD_ARGS[@]}"})
    run_step "ctx warm-up (discarded)" ctx python3 "$TOOLS/ctx-curve.py" "$OUT/raw-local/ctx-warmup.json" \
      --cells 32768x1 --reps 1 --max-tokens 64 "${CTX_COMMON[@]}"
    run_step "ctx curve, one user, cold" ctx python3 "$TOOLS/ctx-curve.py" "$OUT/ctx-curve.json" \
      --cells "$CTX_CELLS" --reps 1 --max-tokens "$CTX_MAX_TOKENS" --max-cell-s 1500 "${CTX_COMMON[@]}"
    if [ -n "$CONC_CELLS" ]; then
      run_step "ctx several users at 32k" ctx python3 "$TOOLS/ctx-curve.py" "$OUT/ctx-conc.json" \
        --cells "$CONC_CELLS" --reps 1 --max-tokens "$CTX_MAX_TOKENS" "${CTX_COMMON[@]}"
    fi
    gate "the context curve"
  fi
fi

# ---- decode arm and concurrency sweep -----------------------------------------------------------------------------------------
if want decode; then
  ARM_EXTRA=()
  if [ "$GUARD" = 1 ]; then ARM_EXTRA=(--fabric --cluster-env "$CLUSTER_ENV"); fi
  run_step "decode arm and chat concurrency sweep" decode bash "$TOOLS/arm-measure.sh" --out "$OUT" --label decode \
    --reps "$DEC_REPS" --levels "$SWEEP_LEVELS" --secs "$SWEEP_SECS" --base-url "$BASE_URL" --data-dir "$DATA_DIR" \
    ${ARM_EXTRA[@]+"${ARM_EXTRA[@]}"}
  # arm-measure.sh writes decode-agent.json, decode-chat.json, decode-conc.json and decode.log (the console log goes to raw-local)
  if [ -f "$OUT/decode.log" ]; then mv "$OUT/decode.log" "$OUT/raw-local/decode.log"; fi
  gate "the decode arm"
fi

# ---- vllm bench serve ---------------------------------------------------------------------------------------------------------------
if want serve; then
  if [ "$SERVE_MODE" = skip ]; then
    if [ "$VLLM_BENCH" = skip ]; then
      record "vllm bench serve" "SKIPPED on request" 0 0
    else
      record "vllm bench serve" "SKIPPED: $SERVE_WHY" 1 0
      STATUS=1
    fi
  else
    for c in $SERVE_LEVELS; do
      n=$((c * 4))
      if [ "$n" -lt 8 ]; then n=8; fi
      if [ "$SERVE_MODE" = local ]; then
        run_step "vllm bench serve c=$c n=$n" serve env VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 "$VLLM_BIN" bench serve \
          --backend vllm --base-url "$BASE_URL" --endpoint /v1/completions --model "$MODEL" --tokenizer "$TOK" \
          --trust-remote-code --dataset-name random --random-input-len 8192 --random-output-len 1024 --ignore-eos \
          --num-prompts "$n" --max-concurrency "$c" --seed "$((1234 + c))" --percentile-metrics ttft,tpot,itl,e2el \
          --metric-percentiles 50,90,99 --save-result --result-dir "$OUT" --result-filename "vllm-bench-c$c.json"
      else
        run_step "vllm bench serve c=$c n=$n" serve docker run --rm --name "bench-vllm-c$c" --network host \
          -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 -e HF_HUB_OFFLINE=1 -v "$TOK:$TOK:ro" -v "$OUT:/out" --entrypoint vllm "$BENCH_IMAGE" \
          bench serve --backend vllm --base-url "$BASE_URL" --endpoint /v1/completions --model "$MODEL" --tokenizer "$TOK" \
          --trust-remote-code --dataset-name random --random-input-len 8192 --random-output-len 1024 --ignore-eos \
          --num-prompts "$n" --max-concurrency "$c" --seed "$((1234 + c))" --percentile-metrics ttft,tpot,itl,e2el \
          --metric-percentiles 50,90,99 --save-result --result-dir /out --result-filename "vllm-bench-c$c.json"
      fi
    done
    gate "vllm bench serve"
  fi
fi

# ---- canary --------------------------------------------------------------------------------------------------------------------------
if want canary; then
  mkdir -p "$OUT/canary"
  # $CANARY_ARGS is split into words on purpose
  # shellcheck disable=SC2086
  run_step "canary ($CANARY_BUDGET s)" canary python3 "$TOOLS/canary-leak-test.py" "$BASE_URL" --sessions "$CANARY_SESSIONS" \
    --time-budget "$CANARY_BUDGET" --grace "$CANARY_GRACE" --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
    --out-dir "$OUT/canary" $CANARY_ARGS
fi

finish "$STATUS"
