#!/usr/bin/env bash
# arm-measure.sh -- one controlled decode "arm": decode speed on recorded agent turns, decode speed on short chat prompts,
# then the chat concurrency sweep. Optionally brackets the decode runs with RoCE / link counter snapshots of every node.
#
#   arm-measure.sh [--out DIR] [--label NAME] [--reps N] [--levels L] [--secs S] [--no-sweep]
#                  [--base-url URL] [--data-dir DIR] [--fabric [--cluster-env FILE]]
#
# What runs, in this order (the calls of the published sequence):
#   1. decode-bench-w2.py  DIR/NAME-agent.json  REPS            agent turns, one user    (needs the prepared W2 data)
#   2. decode-bench.py     DIR/NAME-chat.json   REPS            short chat prompts, one user
#   3. conc-sweep.py       DIR/NAME-conc.json   LEVELS SECS     chat, closed-loop users  (skipped with --no-sweep)
# Defaults: DIR = ., NAME = arm, REPS = 2, LEVELS = 1,2,3,4,6,8, SECS = 120. Everything is also written to DIR/NAME.log; the
# last line of the log is ARM-DONE. Exit status 0 when all three tools succeeded, 1 otherwise, 2 on a usage error.
#
# --fabric (off by default): before and after step 1 and 2, read the RoCE error counters and the PHY link-down counter of
# every node over ssh (packet_seq_err, out_of_sequence, local_ack_timeout_err of ROCE_HCA, link_down_events_phy of
# FABRIC_IFACE), write DIR/NAME.fab-before.txt and DIR/NAME.fab-after.txt, and print the change. A run that adds packet
# loss or a link flap is not a clean run: any local_ack_timeout_err or link-down growth, or more than about 100 new
# out_of_sequence per run, is reported as a warning. The node list
# (NODES, else NODE1..NODE8), SSH_OPTS, SSH_USER, ROCE_HCA and FABRIC_IFACE come from cluster.env ($CLUSTER_ENV or
# --cluster-env, default: cluster.env in the repository root). The files name nodes by position (node1 = first entry), not
# by host name. Without --fabric, or if cluster.env cannot be used, the snapshot is skipped and the log says why.
#
# The server root is --base-url, else $BASE_URL, else http://127.0.0.1:8000. The prepared data is read from --data-dir, else
# $BENCH_DATA_DIR, else bench/data/prepared. Needs python3 and curl on this machine.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)       # <repo>/bench/tools
BENCH=$(dirname "$HERE")
REPO=$(dirname "$BENCH")
OUT=. LABEL=arm REPS=2 LEVELS=1,2,3,4,6,8 SECS=120 SWEEP=1 FABRIC=0
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
DATA_DIR=${BENCH_DATA_DIR:-$BENCH/data/prepared}
CLUSTER_ENV=${CLUSTER_ENV:-$REPO/cluster.env}

usage() {
  cat <<'EOF'
usage: arm-measure.sh [--out DIR] [--label NAME] [--reps N] [--levels L] [--secs S] [--no-sweep]
                      [--base-url URL] [--data-dir DIR] [--fabric [--cluster-env FILE]]

One controlled decode arm: agent-turn decode (decode-bench-w2.py), chat decode (decode-bench.py), then the chat
concurrency sweep (conc-sweep.py). Writes DIR/NAME-agent.json, NAME-chat.json, NAME-conc.json and NAME.log.

  --out DIR         output directory (default .)
  --label NAME      file name prefix (default arm)
  --reps N          repetitions for the two decode tools (default 2)
  --levels L        sweep levels, comma separated (default 1,2,3,4,6,8)
  --secs S          measured seconds per sweep level (default 120)
  --no-sweep        skip the concurrency sweep
  --base-url URL    server root without /v1 (default $BASE_URL, else http://127.0.0.1:8000)
  --data-dir DIR    prepared data (default $BENCH_DATA_DIR, else bench/data/prepared)
  --fabric          snapshot RoCE / link counters of every node before and after the decode runs (needs ssh, see cluster.env)
  --cluster-env F   cluster description for --fabric (default $CLUSTER_ENV, else <repo>/cluster.env)
  -h, --help        this text
EOF
}

need() { [ $# -ge 2 ] || { echo "arm-measure.sh: $1 needs a value" >&2; usage >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --out) need "$@"; OUT=$2; shift 2 ;;
    --label) need "$@"; LABEL=$2; shift 2 ;;
    --reps) need "$@"; REPS=$2; shift 2 ;;
    --levels) need "$@"; LEVELS=$2; shift 2 ;;
    --secs) need "$@"; SECS=$2; shift 2 ;;
    --no-sweep) SWEEP=0; shift ;;
    --base-url) need "$@"; BASE_URL=$2; shift 2 ;;
    --data-dir) need "$@"; DATA_DIR=$2; shift 2 ;;
    --fabric) FABRIC=1; shift ;;
    --cluster-env) need "$@"; CLUSTER_ENV=$2; shift 2 ;;
    *) echo "arm-measure.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
BASE_URL=${BASE_URL%/}
export PYTHONUNBUFFERED=1
case $REPS in '' | *[!0-9]*) echo "arm-measure.sh: --reps must be a whole number" >&2; exit 2 ;; esac
case $SECS in '' | *[!0-9]*) echo "arm-measure.sh: --secs must be a whole number" >&2; exit 2 ;; esac

mkdir -p "$OUT" || exit 2
LOG=$OUT/$LABEL.log
log() { printf '%s\n' "$*" | tee -a "$LOG"; }
: > "$LOG"

# ---- optional fabric snapshots -----------------------------------------------------------------------------------
FAB_NODES=()
SSH_OPT_ARR=()
LOCAL_NAMES=" localhost 127.0.0.1 ::1 $(hostname 2>/dev/null || true) $(hostname -s 2>/dev/null || true) "
read -r -d '' FAB_SCRIPT <<'EOS' || true
c=/sys/class/infiniband/$1/ports/1/hw_counters
printf '%s %s\n' "$(cat "$c/packet_seq_err" "$c/out_of_sequence" "$c/local_ack_timeout_err" 2>/dev/null | tr '\n' ' ')" \
  "$(ethtool -S "$2" 2>/dev/null | awk '/link_down_events_phy/ { print $2; exit }')"
EOS

fabric_setup() {   # fills FAB_NODES and SSH_OPT_ARR; returns 1 (after saying why) when the snapshot cannot be taken
  local i h re='^[A-Za-z0-9][A-Za-z0-9_.:-]*$' list
  if [ ! -r "$CLUSTER_ENV" ]; then
    log "fabric counters: SKIPPED ($CLUSTER_ENV not found; copy cluster.env.example to cluster.env and fill it in)"
    return 1
  fi
  # shellcheck disable=SC1090
  . "$CLUSTER_ENV" >/dev/null 2>&1 || true
  list=${NODES:-}
  if [ -n "$list" ]; then
    for h in $(printf '%s' "$list" | tr ',' ' '); do FAB_NODES[${#FAB_NODES[@]}]=$h; done
  else
    for i in 1 2 3 4 5 6 7 8; do
      eval "h=\${NODE$i:-}"
      if [ -n "$h" ]; then FAB_NODES[${#FAB_NODES[@]}]=$h; fi
    done
  fi
  if [ "${#FAB_NODES[@]}" -eq 0 ]; then log "fabric counters: SKIPPED (no NODES or NODE1..NODE8 in $CLUSTER_ENV)"; return 1; fi
  for h in "${FAB_NODES[@]}"; do
    case $h in *CHANGE_ME* | *'<'*) log "fabric counters: SKIPPED ($CLUSTER_ENV still has the example node names)"; return 1 ;; esac
  done
  if ! [[ ${ROCE_HCA:-} =~ $re ]] || ! [[ ${FABRIC_IFACE:-} =~ $re ]]; then
    log "fabric counters: SKIPPED (set ROCE_HCA and FABRIC_IFACE in $CLUSTER_ENV; see ibdev2netdev and ip -br a)"
    return 1
  fi
  read -r -a SSH_OPT_ARR <<<"${SSH_OPTS-"-o BatchMode=yes -o ConnectTimeout=5"}"
  return 0
}

fabric_read() {   # fabric_read <host>: "packet_seq_err out_of_sequence local_ack_timeout_err link_down_events_phy"
  local host=$1
  case $LOCAL_NAMES in
    *" $host "*) bash -s -- "$ROCE_HCA" "$FABRIC_IFACE" <<<"$FAB_SCRIPT" ;;
    *) ssh ${SSH_OPT_ARR[@]+"${SSH_OPT_ARR[@]}"} "${SSH_USER:+$SSH_USER@}$host" bash -s -- "$ROCE_HCA" "$FABRIC_IFACE" <<<"$FAB_SCRIPT" ;;
  esac
}

fabric_snapshot() {   # fabric_snapshot <file>: one line per node
  local f=$1 i=0 h line
  : > "$f"
  for h in "${FAB_NODES[@]}"; do
    i=$((i + 1))
    line=$(fabric_read "$h" 2>/dev/null) || line=''
    printf 'node%s %s\n' "$i" "${line:-unreadable}" >> "$f"
  done
}

fabric_delta() {   # fabric_delta <before> <after>: print the change per node and warn about growth
  awk '
    NR == FNR { for (i = 2; i <= NF; i++) b[$1 SUBSEP i] = $i; next }
    FNR == 1 { printf "%-7s %16s %16s %22s %22s\n", "node", "packet_seq_err", "out_of_sequence", "local_ack_timeout_err", "link_down_events_phy" }
    {
      row = sprintf("%-7s", $1); n = 0
      for (i = 2; i <= 5; i++) {
        k = $1 SUBSEP i
        if (($i ~ /^[0-9]+$/) && ((k in b) && b[k] ~ /^[0-9]+$/)) {
          d = $i - b[k]; row = row sprintf((i < 4 ? " %16.0f" : " %22.0f"), d)
          if (i == 3 && d > 100) { warn = warn sprintf("  %s out_of_sequence +%.0f (more than about 100 per run)\n", $1, d) }
          if (i == 4 && d > 0) { warn = warn sprintf("  %s local_ack_timeout_err +%.0f\n", $1, d) }
          if (i == 5 && d > 0) { warn = warn sprintf("  %s link_down_events_phy +%.0f\n", $1, d) }
        } else {
          row = row sprintf((i < 4 ? " %16s" : " %22s"), "n/a"); unread = 1
        }
      }
      print row
    }
    END {
      if (warn != "") { printf "fabric: WARN, this run is not clean:\n%s", warn }
      else if (unread) { print "fabric: some counters could not be read (n/a); the rest show no growth that matters" }
      else { print "fabric: OK, no ack timeouts, no link-down events, out_of_sequence within the usual range" }
    }' "$1" "$2"
}

# ---- run ---------------------------------------------------------------------------------------------------------
if ! curl -sf -m 5 "$BASE_URL/health" >/dev/null 2>&1; then
  log "arm-measure.sh: no healthy server at $BASE_URL (GET /health failed). Set BASE_URL or --base-url."
  exit 1
fi
MODEL=$(curl -sf -m 10 "$BASE_URL/v1/models" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
log "start $(date +%T) model=${MODEL:-unknown}"

FAB_OK=0
if [ "$FABRIC" = 1 ] && fabric_setup; then
  FAB_OK=1
  fabric_snapshot "$OUT/$LABEL.fab-before.txt"
  log "fabric counters: before-snapshot of ${#FAB_NODES[@]} node(s) -> $LABEL.fab-before.txt"
elif [ "$FABRIC" != 1 ]; then
  log "fabric counters: off (pass --fabric to snapshot RoCE / link counters before and after the decode runs)"
fi

rc=0
run() {   # run <command...>: log everything, remember a failure
  local st
  "$@" 2>&1 | tee -a "$LOG"
  st=${PIPESTATUS[0]}
  if [ "$st" -ne 0 ]; then rc=1; log "FAILED (exit $st): $*"; fi
}
run python3 "$HERE/decode-bench-w2.py" "$OUT/$LABEL-agent.json" "$REPS" --base-url "$BASE_URL" --data "$DATA_DIR/w2_agent-v1.jsonl"
run python3 "$HERE/decode-bench.py" "$OUT/$LABEL-chat.json" "$REPS" --base-url "$BASE_URL"
log "decode end $(date +%T)"

if [ "$FAB_OK" = 1 ]; then
  fabric_snapshot "$OUT/$LABEL.fab-after.txt"
  log "fabric counters: change during the decode runs (after - before)"
  fabric_delta "$OUT/$LABEL.fab-before.txt" "$OUT/$LABEL.fab-after.txt" 2>&1 | tee -a "$LOG"
fi

if [ "$SWEEP" = 1 ]; then
  run python3 "$HERE/conc-sweep.py" "$OUT/$LABEL-conc.json" "$LEVELS" "$SECS" --base-url "$BASE_URL"
  log "conc end $(date +%T)"
else
  log "concurrency sweep: skipped (--no-sweep)"
fi
log ARM-DONE
exit "$rc"
