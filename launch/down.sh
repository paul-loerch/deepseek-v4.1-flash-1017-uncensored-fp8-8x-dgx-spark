#!/usr/bin/env bash
# launch/down.sh -- remove the rank containers on all eight nodes, then check that every node is idle.
#
#   launch/down.sh [--force] [--verify]
#
#   (no flags)  refuse while rank 0 reports running or waiting requests, or while its /metrics cannot be read;
#               otherwise remove the containers and wait until every node is idle
#   --force     remove them anyway (after an engine death, or when /metrics is gone and you have checked)
#   --verify    only check that every node is idle and has no rank container; changes nothing
#
# Why it refuses: removing ranks while a request runs strands the other ranks inside an NCCL collective, and on
# GB10 a GPU context left spinning with no process to own it is only cleared by rebooting the node. An unreadable
# /metrics is "unknown", never "idle".
#
# Order: the workers (ranks 1..7) are removed first, in parallel, and rank 0 last, so that rank 0's API and log stay
# available until the workers are gone: if a worker cannot be reached, rank 0's log is still there to read.
#
# A node counts as idle when it has no rank container, nvidia-smi works, the GPU is at 0 % utilization and draws
# less than IDLE_GPU_POWER_MAX_W. A GPU that stays busy with no container is an orphaned context: leave it alone
# and reboot that node.
# Stale shared-memory rings: vLLM's message rings between the API server, the engine core and rank 0 live in /dev/shm of
# the head node (NODE1) as psm_* files and are not unlinked when the container is removed. They are root-owned, so after
# the rank containers are gone this script deletes them with a throwaway container of the recipe image (only when no rank
# container exists on that node; disable with CLEAN_SHM=0 in cluster.env). Without it, 136 leaked segments (243 MB of RAM)
# piled up on the head node over two days of restarts.
# Exit status: 0 all nodes idle, 1 refused or not idle, 2 usage or configuration error.
set -euo pipefail
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<'EOF'
usage: launch/down.sh [--force] [--verify]
  --force    remove the containers even if requests are running or /metrics cannot be read
  --verify   only check that every node is idle and has no rank container; changes nothing
exit status: 0 all nodes idle, 1 refused or not idle, 2 usage or configuration error
EOF
}

FORCE=0 VERIFY=0
while [ $# -gt 0 ]; do
  case $1 in
    --force) FORCE=1 ;;
    --verify) VERIFY=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" 2 ;;
  esac
  shift
done

load_config
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# node_table <dir>: one line per node; returns 1 when some node is not idle
node_table() {
  local d=$1 r st rc=0
  printf '  %-14s %-4s %-26s %8s %6s %11s\n' node rank state 'GPU W' 'util %' 'avail GiB'
  for ((r = 0; r < NNODES; r++)); do
    st=$(idle_reason "$d" "$r")
    printf '  %-14s %-4s %-26s %8s %6s %11s\n' "${NODE_NAME[$r]}" "$r" "$st" \
      "$(pget "$d" "$r" gpu_power_w)" "$(pget "$d" "$r" gpu_util)" "$(pget "$d" "$r" mem_avail_gib)"
    if [ "$st" != ok ]; then rc=1; fi
  done
  return "$rc"
}

if [ "$VERIFY" = 1 ]; then
  say "checking that all $NNODES nodes are idle"
  probe_all "$TMP/verify" quick
  if node_table "$TMP/verify"; then
    ok "all $NNODES nodes are idle and have no rank container"
    exit 0
  fi
  die "not every node is idle" 1
fi

# ---- is anything to remove, and is anything running? ------------------------------------------------------
probe_all "$TMP/before" quick
present=0
for ((r = 0; r < NNODES; r++)); do
  if [ -n "$(pget "$TMP/before" "$r" rank_containers)" ]; then present=1; fi
done

if [ "$present" = 0 ]; then
  say "no rank container found on any reachable node; nothing to remove"
elif [ "$FORCE" = 0 ]; then
  run_on "${NODE_NAME[0]}" 'curl -s -m 5 "$1"' "http://127.0.0.1:$API_PORT/metrics" >"$TMP/metrics" 2>/dev/null || true
  running=$(metric_sum num_requests_running "$TMP/metrics")
  waiting=$(metric_sum num_requests_waiting "$TMP/metrics")
  if [ "$running" = unknown ] || [ "$waiting" = unknown ]; then
    die "refusing: /metrics on ${NODE_NAME[0]}:$API_PORT is not readable (running=$running, waiting=$waiting). The engine may be busy,
        stuck or still booting, and removing ranks under a live request can leave a GPU spinning with no container
        (only a reboot clears that). If you are sure nothing is running, use: launch/down.sh --force"
  fi
  if [ "$running" != 0 ]; then
    die "refusing: $running request(s) are running on ${NODE_NAME[0]}:$API_PORT. Wait until there are none, or use --force if the engine is stuck"
  fi
  if [ "$waiting" != 0 ]; then
    die "refusing: $waiting request(s) are waiting on ${NODE_NAME[0]}:$API_PORT. Wait until there are none, or use --force if the engine is stuck"
  fi
else
  warn "--force: removing the containers without checking for running requests"
fi

# ---- remove: workers first (in parallel), rank 0 last -----------------------------------------------------
read -r -d '' REMOVE <<'EOS' || true
if docker inspect "$1" >/dev/null 2>&1; then
  if docker rm -f "$1" >/dev/null 2>&1; then echo removed; else echo "could not be removed"; exit 1; fi
else
  echo "was not present"
fi
EOS

remove_failed=0
if [ "$present" = 1 ]; then
  say "removing the worker containers on ${NODE_NAME[1]} .. ${NODE_NAME[$((NNODES - 1))]} (in parallel)"
  PIDS=()
  for ((r = 1; r < NNODES; r++)); do
    run_on "${NODE_NAME[$r]}" "$REMOVE" "$(rank_container "$r")" >"$TMP/rm.$r" 2>&1 &
    PIDS[$r]=$!
  done
  for ((r = 1; r < NNODES; r++)); do
    if wait "${PIDS[$r]}"; then
      printf '  %-14s %s: %s\n' "${NODE_NAME[$r]}" "$(rank_container "$r")" "$(tr '\n' ' ' <"$TMP/rm.$r")"
    else
      remove_failed=1
      printf '  %-14s %s: FAILED (%s)\n' "${NODE_NAME[$r]}" "$(rank_container "$r")" "$(tr '\n' ' ' <"$TMP/rm.$r" | head -c 200)"
    fi
  done
  say "removing rank 0 on ${NODE_NAME[0]} (last)"
  if run_on "${NODE_NAME[0]}" "$REMOVE" "$(rank_container 0)" >"$TMP/rm.0" 2>&1; then
    printf '  %-14s %s: %s\n' "${NODE_NAME[0]}" "$(rank_container 0)" "$(tr '\n' ' ' <"$TMP/rm.0")"
  else
    remove_failed=1
    printf '  %-14s %s: FAILED (%s)\n' "${NODE_NAME[0]}" "$(rank_container 0)" "$(tr '\n' ' ' <"$TMP/rm.0" | head -c 200)"
  fi
fi

# ---- stale /dev/shm/psm_* on the head node ----------------------------------------------------------------
# Runs on NODE1 only (the shared-memory rings exist only between the API server, the engine core and rank 0), and only
# when no rank container exists there: with an engine up, the segments are in use. The files belong to root, so a
# throwaway container of the recipe image (no GPU, no network) deletes them.
read -r -d '' CLEAN_SHM_SCRIPT <<'EOS' || true
set -u
PREFIX=$1 IMAGE=$2
SHM=${PROBE_ROOT:-}/dev/shm   # PROBE_ROOT is unset on real nodes (the offline test uses it)
names=$(docker ps -a --format '{{.Names}}' 2>/dev/null) || { echo "skipped: docker is not usable"; exit 0; }
if printf '%s\n' "$names" | awk -v p="$PREFIX" 'index($0, p) == 1 && substr($0, length(p) + 1) ~ /^[0-9]+$/ { f = 1 } END { exit !f }'; then
  echo "skipped: a rank container exists on this node"; exit 0
fi
before=$(ls "$SHM" 2>/dev/null | grep -c '^psm_' || true)
if [ "${before:-0}" = 0 ]; then echo "none left over"; exit 0; fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "skipped: image $IMAGE is not present on this node ($before segments left)"; exit 0; }
kib=$(du -ck "$SHM"/psm_* 2>/dev/null | tail -n 1 | cut -f1)
docker run --rm --network none -v /dev/shm:/dev/shm --entrypoint sh "$IMAGE" -c 'find /dev/shm -maxdepth 1 -name "psm_*" -delete' >/dev/null 2>&1 || true
after=$(ls "$SHM" 2>/dev/null | grep -c '^psm_' || true)
echo "removed $((before - ${after:-0})) of $before psm_* segments (${kib:-?} KiB); ${after:-0} left"
EOS

clean_shm() {
  local out
  [ "$CLEAN_SHM" = 1 ] || return 0
  if out=$(run_on "${NODE_NAME[0]}" "$CLEAN_SHM_SCRIPT" "$CONTAINER_PREFIX" "$IMAGE" 2>&1); then
    say "stale shared-memory rings on ${NODE_NAME[0]}: $(printf '%s' "$out" | tail -n 1)"
  else
    warn "could not clean /dev/shm/psm_* on ${NODE_NAME[0]}: $(printf '%s' "$out" | tail -n 1 | head -c 200)"
  fi
}
if [ "$remove_failed" = 0 ]; then clean_shm; fi

# ---- verify: no container, GPUs idle ----------------------------------------------------------------------
# After a removal the GPU counters need a moment to settle; when nothing was removed, one look is enough.
deadline=$((SECONDS + VERIFY_WITHIN_S))
if [ "$present" = 0 ]; then deadline=$SECONDS; fi
i=0
while :; do
  i=$((i + 1))
  probe_all "$TMP/after.$i" quick
  if node_table "$TMP/after.$i" >"$TMP/table.$i"; then
    cat "$TMP/table.$i"
    if [ "$remove_failed" = 1 ]; then
      die "a container could not be removed (see above), although every node now looks idle; check each node with: docker ps -a" 1
    fi
    ok "all $NNODES nodes are idle, no rank containers"
    exit 0
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    cat "$TMP/table.$i"
    if grep -q UNREACHABLE "$TMP/table.$i"; then
      die "a node cannot be reached, so its containers were not removed and its state is unknown. Fix the node (or power-cycle it), then run launch/down.sh --verify" 1
    elif grep -q 'GPU busy' "$TMP/table.$i"; then
      die "a GPU is still busy with no container: an orphaned context. Do not try to kill it; reboot that node" 1
    fi
    die "the nodes are not idle (see the table above)" 1
  fi
  say "waiting for the nodes to settle..."
  sleep "$VERIFY_POLL_S"
done
