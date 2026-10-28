#!/usr/bin/env bash
# launch/up.sh -- start DeepSeek-V4.1-Flash-1017 with tensor parallelism 8 on eight nodes, one rank per node.
#
#   launch/up.sh [--dry-run] [--wait] [--timeout MIN]
#
#   1. preflight: read-only checks on all eight nodes; one failure and nothing is started
#   2. optional page-cache release on every node (DROP_CACHES=1 in cluster.env)
#   3. start ranks 1..7 (headless) on the workers, then rank 0, which serves the API
#   4. check that every rank container is running
#   5. with --wait: follow rank 0's log against launch/readiness.txt until the API answers
#
#   --dry-run      run the read-only preflight, print every command that would change something, change nothing
#   --wait         do step 5 (a first boot takes 15-30 minutes)
#   --timeout MIN  with --wait, give up after MIN minutes (default 45; decimals allowed)
#
# Needs cluster.env and config/recipe.env (see launch/lib.sh), key-based ssh to every node and docker on every
# node. Run it on rank 0's node (NODE1). Nothing is stopped or removed here; launch/down.sh does that.
# Exit status: 0 ok, 1 a check failed, 2 usage or configuration error.
set -euo pipefail
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<'EOF'
usage: launch/up.sh [--dry-run] [--wait] [--timeout MIN]
  --dry-run      run the read-only preflight, print every command that would change something, change nothing
  --wait         after starting, follow rank 0's log against launch/readiness.txt until the API answers
  --timeout MIN  with --wait: give up after MIN minutes (default 45; decimals allowed)
exit status: 0 ok, 1 a check failed, 2 usage or configuration error
EOF
}

DRY=0 WAIT=0 TIMEOUT_MIN=45
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --wait) WAIT=1 ;;
    --timeout)
      [ $# -ge 2 ] || die "--timeout needs a number of minutes" 2
      TIMEOUT_MIN=$2
      shift
      ;;
    --timeout=*) TIMEOUT_MIN=${1#*=} ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" 2 ;;
  esac
  shift
done
if ! is_num "$TIMEOUT_MIN" || ! fge "$TIMEOUT_MIN" 0.02; then
  die "--timeout must be a number of minutes, at least 0.02 (got '$TIMEOUT_MIN')" 2
fi
TIMEOUT_S=$(awk -v m="$TIMEOUT_MIN" 'BEGIN { printf "%d", m * 60 }')

load_config
READINESS_FILE=${READINESS_FILE:-$LAUNCH_DIR/readiness.txt}

# ---- readiness rules: "<require|fail|info> <extended regex>" per line, '#' starts a comment ------------------
REQ_RE=() BAD_RE=() INFO_RE=() INFO_SEEN=()
load_readiness() {
  local line kind rest n=0 rc
  [ -r "$READINESS_FILE" ] || die "readiness rules not found: $READINESS_FILE" 2
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    line=$(trim "$line")
    case $line in '' | '#'*) continue ;; esac
    kind=${line%%[[:space:]]*}
    rest=${line#"$kind"}
    rest=${rest#"${rest%%[![:space:]]*}"}
    rest=${rest//@API_PORT@/$API_PORT}
    rest=${rest//@MASTER_PORT@/$MASTER_PORT}
    [ -n "$rest" ] || die "$READINESS_FILE:$n: '$kind' has no pattern" 2
    rc=0
    grep -E -- "$rest" /dev/null >/dev/null 2>&1 || rc=$?
    [ "$rc" -le 1 ] || die "$READINESS_FILE:$n: not a valid extended regular expression: $rest" 2
    case $kind in
      require) REQ_RE[${#REQ_RE[@]}]=$rest ;;
      fail) BAD_RE[${#BAD_RE[@]}]=$rest ;;
      info) INFO_RE[${#INFO_RE[@]}]=$rest; INFO_SEEN[${#INFO_SEEN[@]}]=0 ;;
      *) die "$READINESS_FILE:$n: unknown rule '$kind' (use require, fail or info)" 2 ;;
    esac
  done <"$READINESS_FILE"
}
if [ "$WAIT" = 1 ]; then load_readiness; fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if [ "$DRY" = 1 ]; then
  say "DRY RUN: the preflight only reads; every command that would change something is printed, none is run"
fi
say "DeepSeek-V4.1-Flash-1017, tensor parallel $NNODES: ${NODE_NAME[*]} | image $IMAGE | API on ${NODE_NAME[0]}:$API_PORT"

# ---- 1. preflight ----------------------------------------------------------------------------------------
PROBLEMS=()
problem() { PROBLEMS[${#PROBLEMS[@]}]="$*"; }
PATCH_HASH=''

check_node() {  # check_node <dir> <rank>: the per-node rules
  local d=$1 r=$2 n=${NODE_NAME[$2]} v free total need
  if [ "$(pget "$d" "$r" reachable)" != 1 ]; then
    problem "$n: cannot run commands over ssh ($(trim "$(head -c 200 "$d/$r.err" 2>/dev/null | tr '\n' ' ')")); check: ssh $(ssh_target "$n") true"
    return 0
  fi
  if [ "$(pget "$d" "$r" docker_ok)" != 1 ]; then
    problem "$n: docker is not usable by this user ($(pget "$d" "$r" docker_err)); check: ssh $(ssh_target "$n") docker ps"
  else
    v=$(pget "$d" "$r" rank_containers)
    if [ -n "$v" ]; then problem "$n: rank container(s) already exist: ${v}-- remove them with launch/down.sh, then retry"; fi
    v=$(pget "$d" "$r" other_running)
    if [ -n "$v" ]; then warn "$n: other container(s) are running: ${v}-- they share this node's GPU and memory"; fi
  fi

  if [ "$(pget "$d" "$r" python_ok)" != 1 ]; then
    problem "$n: python3 not found; the launcher needs it on every node (to measure the model directory)"
    return 0
  fi

  v=$(pget "$d" "$r" image_id)
  if [ -z "$v" ] || [ "$v" = missing ]; then
    problem "$n: image $IMAGE is not present; copy it with: docker save $IMAGE | ssh $(ssh_target "$n") docker load"
  fi

  v=$(pget "$d" "$r" model_files)
  if [ -z "$v" ] || [ "$v" = -1 ]; then
    problem "$n: model directory $MODEL_DIR does not exist"
  elif [ "$v" = 0 ]; then
    problem "$n: model directory $MODEL_DIR is empty"
  fi

  if [ "$(pget "$d" "$r" gpu_ok)" != 1 ]; then
    problem "$n: nvidia-smi failed: $(pget "$d" "$r" gpu_err)"
  else
    v=$(pget "$d" "$r" gpu_util)
    if is_num "$v" && fge "$v" 5; then warn "$n: the GPU is $v % utilized before launch; is another job using it?"; fi
  fi

  v=$(pget "$d" "$r" mem_avail_gib)
  if ! is_num "$v"; then
    problem "$n: could not read MemAvailable"
  elif flt "$v" "$MEM_AVAILABLE_MIN_GIB"; then
    problem "$n: MemAvailable is $v GiB, the minimum is $MEM_AVAILABLE_MIN_GIB GiB; stop other workloads on this node first"
  fi
  free=$(pget "$d" "$r" mem_free_gib)
  total=$(pget "$d" "$r" mem_total_gib)
  if is_num "$free" && is_num "$total" && [ "$DROP_CACHES" != 1 ]; then
    need=$(awk -v g="$GPU_MEMORY_UTILIZATION" -v t="$total" 'BEGIN { printf "%.0f", g * t }')
    if flt "$free" "$need"; then
      warn "$n: only $free GiB of memory is free, vLLM wants $need GiB free at start-up ($GPU_MEMORY_UTILIZATION x $total GiB);" \
        "the rest is page cache: set DROP_CACHES=1 in cluster.env or drop the caches by hand"
    fi
  fi

  case "$(pget "$d" "$r" iface_state)" in
    missing | '') problem "$n: fabric interface $FABRIC_IFACE does not exist; check: ip -br a" ;;
    *)
      if [ "$(pget "$d" "$r" iface_up)" != 1 ]; then
        problem "$n: fabric interface $FABRIC_IFACE is not up (state $(pget "$d" "$r" iface_state)); check cable, switch port and: ip -br a"
      fi
      v=$(pget "$d" "$r" iface_mtu)
      if [ "$v" != "$FABRIC_MTU" ]; then
        warn "$n: $FABRIC_IFACE has MTU $v, expected $FABRIC_MTU; set it with the node's network configuration (jumbo frames), or lower FABRIC_MTU if intended"
      fi
      v=$(pget "$d" "$r" iface_addrs)
      case " $v " in
        *" ${NODE_IP[$r]} "*) ;;
        *) problem "$n: NODE$((r + 1))_FABRIC_IP ${NODE_IP[$r]} is not an address of $FABRIC_IFACE (it has: ${v:-none}); check: ip -br a" ;;
      esac
      ;;
  esac
  if [ "$(pget "$d" "$r" hca_ok)" != 1 ]; then
    problem "$n: RoCE device $ROCE_HCA not found (found: $(pget "$d" "$r" hca_list)); check: ibdev2netdev or rdma link"
  fi

  if [ "$r" -eq 0 ]; then
    v=$(pget "$d" "$r" ports_busy)
    if [ -n "$v" ]; then problem "$n: port(s) already in use: ${v}(API_PORT=$API_PORT, MASTER_PORT=$MASTER_PORT); is an earlier launch still up?"; fi
  fi
  if [ -n "$PATCH_DIR" ]; then
    v=$(pget "$d" "$r" patch_hash)
    if [ "$v" != "$PATCH_HASH" ]; then
      if [ -z "$v" ] || [ "$v" = missing ]; then v="missing"; else v="different from this machine's copy"; fi
      problem "$n: the patch directory $PATCH_DIR is $v on this node (it must be identical on every node);" \
        "copy it from this machine: rsync -a --delete $PATCH_DIR/ $(ssh_target "$n"):$PATCH_DIR/"
    fi
  fi
  if [ -n "$ENGRAM_PACKED_DIR" ]; then
    # each node serves its own rank's slice of the Engram tables from node-local NVMe (patches 016 + 030)
    v=$(pget "$d" "$r" engram_packs)
    case $v in
      '' | missing)
        problem "$n: the packed Engram directory $ENGRAM_PACKED_DIR does not exist; create the rows of rank $r there with" \
          "tools/engram_pack.py (README, step 4)"
        ;;
      0)
        problem "$n: $ENGRAM_PACKED_DIR has no packed Engram rows for rank $r (engram_L*_tp8_r$r.pack.json);" \
          "build them on this node: python3 tools/engram_pack.py pack --model-dir $MODEL_DIR --out-dir $ENGRAM_PACKED_DIR --rank $r"
        ;;
      1)
        warn "$n: only one packed Engram layer found for rank $r in $ENGRAM_PACKED_DIR; the recipe serves layers 1 and 14 from NVMe"
        ;;
    esac
  fi
}

check_same() {  # check_same <dir>: image ID and model directory must be identical on every node
  local d=$1 r id files bytes seen_i='' seen_m='' list_i='' list_m=''
  for ((r = 0; r < NNODES; r++)); do
    id=$(pget "$d" "$r" image_id)
    case $id in '' | missing) ;; *)
      case " $seen_i " in *" $id "*) ;; *) seen_i="$seen_i $id" ;; esac
      list_i="$list_i ${NODE_NAME[$r]}=${id:0:19}"
      ;;
    esac
    files=$(pget "$d" "$r" model_files)
    bytes=$(pget "$d" "$r" model_bytes)
    case $files in '' | -1) ;; *)
      case " $seen_m " in *" $files/$bytes "*) ;; *) seen_m="$seen_m $files/$bytes" ;; esac
      list_m="$list_m ${NODE_NAME[$r]}=[$files files, $bytes bytes]"
      ;;
    esac
  done
  # shellcheck disable=SC2086
  set -- $seen_i
  if [ $# -gt 1 ]; then
    problem "the image IDs differ between nodes:$list_i -- every node must run the same image; build or pull it once and copy" \
      "it with docker save | ssh <node> docker load, which keeps the ID"
  fi
  # shellcheck disable=SC2086
  set -- $seen_m
  if [ $# -gt 1 ]; then
    problem "the model directory differs between nodes:$list_m -- copy it again where it is incomplete"
  fi
}

preflight() {
  local d=$TMP/pre r n link
  if [ -n "$PATCH_DIR" ]; then PATCH_HASH=$(tree_hash "$PATCH_DIR"); fi
  say "preflight: reading the state of $NNODES nodes (nothing is changed)"
  if [ "$DRY" = 1 ]; then
    say "per node, one read-only probe (PROBE_BODY in launch/lib.sh: docker ps/inspect, nvidia-smi, ip, ss, /proc/meminfo): $(ssh_cmdline "${NODE_NAME[1]}")"
  fi
  probe_all "$d" full
  printf '  %-14s %-4s %-15s %-5s %-5s %10s  %-16s %s\n' node rank fabric-ip link mtu 'avail GiB' gpu rank-container
  for ((r = 0; r < NNODES; r++)); do
    n=${NODE_NAME[$r]}
    if [ "$(pget "$d" "$r" reachable)" != 1 ]; then
      printf '  %-14s %-4s %-15s %s\n' "$n" "$r" "${NODE_IP[$r]}" "not reachable"
      continue
    fi
    link=DOWN
    if [ "$(pget "$d" "$r" iface_up)" = 1 ]; then link=up; fi
    printf '  %-14s %-4s %-15s %-5s %-5s %10s  %-16s %s\n' "$n" "$r" "${NODE_IP[$r]}" "$link" "$(pget "$d" "$r" iface_mtu)" \
      "$(pget "$d" "$r" mem_avail_gib)" "$(pget "$d" "$r" gpu_power_w) W $(pget "$d" "$r" gpu_util)%" \
      "$(pget "$d" "$r" container_state)"
  done
  for ((r = 0; r < NNODES; r++)); do check_node "$d" "$r"; done
  check_same "$d"
  if [ "${#PROBLEMS[@]}" -eq 0 ]; then
    say "image $IMAGE  $(pget "$d" 0 image_id | cut -c1-19)  identical on all $NNODES nodes"
    say "model $MODEL_DIR  $(pget "$d" 0 model_files) files, $(gib "$(pget "$d" 0 model_bytes)") GiB  identical on all $NNODES nodes"
  fi
}

PREFLIGHT_OK=1
preflight
if [ "${#PROBLEMS[@]}" -gt 0 ]; then
  for p in "${PROBLEMS[@]}"; do bad "$p"; done
  if [ "$DRY" = 1 ]; then
    warn "a real run would stop here (${#PROBLEMS[@]} problem(s)); the dry run continues to show the commands"
    PREFLIGHT_OK=0
  else
    die "preflight failed (${#PROBLEMS[@]} problem(s)); nothing was changed" 1
  fi
else
  ok "preflight passed on all $NNODES nodes"
fi

# ---- 2. page cache (optional) ----------------------------------------------------------------------------
# GB10 has one memory pool for the CPU, the page cache and the GPU. Weights read from disk leave page cache that
# CUDA cannot reclaim, and vLLM then stops at start-up with "Free memory on device ... less than desired GPU
# memory utilization" although `free` shows plenty. DROP_CACHES=1 releases it before the containers start.
read -r -d '' PREP <<'EOS' || true
set -u
python3 - "$1" <<'PY' || echo "evict_warn=posix_fadvise pass failed"
import os, sys
n = 0
fadvise = getattr(os, "posix_fadvise", None)
if fadvise:
    for d, _, files in os.walk(sys.argv[1]):
        for f in files:
            try:
                fd = os.open(os.path.join(d, f), os.O_RDONLY)
            except OSError:
                continue
            try:
                fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
print("evicted_files=%d" % n)
PY
sync
if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then echo drop_caches=ok; else echo drop_caches=denied; fi
EOS

release_page_cache() {
  local r n out free need
  [ "$DROP_CACHES" = 1 ] || return 0
  if [ "$DRY" = 1 ]; then
    for ((r = 0; r < NNODES; r++)); do show_cmd "${NODE_NAME[$r]}" "$PREP" "$MODEL_DIR"; done
    return 0
  fi
  say "releasing page cache on $NNODES nodes (DROP_CACHES=1)"
  for ((r = 0; r < NNODES; r++)); do
    run_on "${NODE_NAME[$r]}" "$PREP" "$MODEL_DIR" >"$TMP/prep.$r" 2>&1 &
  done
  wait || true
  for ((r = 0; r < NNODES; r++)); do
    n=${NODE_NAME[$r]}
    out=$(tr '\n' ' ' <"$TMP/prep.$r")
    case $out in
      *drop_caches=ok*) ;;
      *) warn "$n: could not drop the page cache (needs the sudoers rule in cluster.env.example): ${out:0:200}" ;;
    esac
  done
  probe_all "$TMP/post-prep" quick
  for ((r = 0; r < NNODES; r++)); do
    n=${NODE_NAME[$r]}
    free=$(pget "$TMP/post-prep" "$r" mem_free_gib)
    printf '  %-14s MemAvailable %6s GiB   MemFree %6s GiB\n' "$n" "$(pget "$TMP/post-prep" "$r" mem_avail_gib)" "$free"
    need=$(awk -v g="$GPU_MEMORY_UTILIZATION" -v t="$(pget "$TMP/post-prep" "$r" mem_total_gib)" 'BEGIN { printf "%.0f", g * t }')
    if is_num "$free" && flt "$free" "$need"; then
      warn "$n: only $free GiB is free after releasing the page cache, vLLM wants $need GiB; find what holds the memory (ps aux --sort=-rss | head)"
    fi
  done
}

# ---- 3. start the ranks ----------------------------------------------------------------------------------
start_rank() {  # start_rank <rank>  -> output in $TMP/start.<rank>; returns the node's exit status
  run_on "${NODE_NAME[$1]}" "$(rank_script "$1")" >"$TMP/start.$1" 2>&1
}

start_ranks() {
  local r failed='' pid
  if [ "$DRY" = 1 ]; then
    for ((r = 1; r < NNODES; r++)); do show_cmd "${NODE_NAME[$r]}" "$(rank_script "$r")"; done
    show_cmd "${NODE_NAME[0]}" "$(rank_script 0)"
    return 0
  fi
  say "starting ranks 1..$((NNODES - 1)) (headless) on ${NODE_NAME[1]} .. ${NODE_NAME[$((NNODES - 1))]}"
  PIDS=()
  for ((r = 1; r < NNODES; r++)); do
    start_rank "$r" &
    PIDS[$r]=$!
  done
  for ((r = 1; r < NNODES; r++)); do
    pid=${PIDS[$r]}
    if wait "$pid"; then cat "$TMP/start.$r"; else failed="$failed $r"; fi
  done
  if [ -n "$failed" ]; then
    # shellcheck disable=SC2086
    for r in $failed; do bad "rank $r on ${NODE_NAME[$r]} did not start: $(tr '\n' ' ' <"$TMP/start.$r")"; done
    die "not all workers started, so rank 0 was NOT started. Ranks that did start are still running; remove them with launch/down.sh" 1
  fi
  say "starting rank 0 (API server) on ${NODE_NAME[0]}"
  if start_rank 0; then cat "$TMP/start.0"; else
    bad "rank 0 on ${NODE_NAME[0]} did not start: $(tr '\n' ' ' <"$TMP/start.0")"
    die "rank 0 did not start; the workers are running. Remove everything with launch/down.sh" 1
  fi
}

# rank_states <dir>: the docker state of every rank container -> <dir>/<rank>.state
rank_states() {
  local dir=$1 r
  mkdir -p "$dir"
  for ((r = 0; r < NNODES; r++)); do
    ( run_on "${NODE_NAME[$r]}" 'docker inspect -f "{{.State.Status}}" "$1" 2>/dev/null || echo absent' "$(rank_container "$r")" 2>/dev/null \
        || echo unreachable ) >"$dir/$r.state" &
  done
  wait || true
}
# not_running <dir>: "node:container=state" for every rank whose container is not running
not_running() {
  local dir=$1 r st out=''
  for ((r = 0; r < NNODES; r++)); do
    st=$(tail -n 1 "$dir/$r.state")
    if [ "$st" != running ]; then out="$out ${NODE_NAME[$r]}:$(rank_container "$r")=${st:-unknown}"; fi
  done
  printf '%s' "$out"
}

logs_hint() {
  printf 'rank 0:  ssh %s docker logs %s   (or, on that node: docker logs -f %s)\n' "$(ssh_target "${NODE_NAME[0]}")" "$(rank_container 0)" "$(rank_container 0)"
  printf 'workers: ssh <node> docker logs %s<rank>\n' "$CONTAINER_PREFIX"
}

# ---- 4. every container must stay up ---------------------------------------------------------------------
check_running() {
  local deadline=$((SECONDS + UP_WITHIN_S)) settled=0 bad_list=''
  while :; do
    sleep "$UP_POLL_S"
    rank_states "$TMP/up"
    bad_list=$(not_running "$TMP/up")
    if [ -z "$bad_list" ]; then settled=$((settled + 1)); else settled=0; fi
    if [ "$settled" -ge 2 ]; then break; fi
    case $bad_list in *exited* | *dead* | *absent*) break ;; esac
    if [ "$SECONDS" -ge "$deadline" ]; then break; fi
  done
  if [ -n "$bad_list" ]; then
    warn "not running:$bad_list"
    logs_hint >&2
    die "launch incomplete. Read the logs above, then remove all ranks with launch/down.sh" 1
  fi
  ok "all $NNODES rank containers are running"
}

# ---- 5. --wait: readiness rules, health, models ----------------------------------------------------------
boot_failed() {  # boot_failed <headline> [matched line]
  bad "$1"
  if [ -n "${2:-}" ]; then printf '          %s\n' "${2:0:300}" >&2; fi
  logs_hint >&2
  printf 'Nothing was stopped. Look at the logs, then remove all ranks with: launch/down.sh   (add --force if requests are stuck)\n' >&2
  exit 1
}

wait_ready() {
  local t0=$SECONDS last_lines=0 last_change=$SECONDS last_report=$SECONDS last_stall=0 lines i hit elapsed quiet bad_list missing=''
  local log=$TMP/rank0.log
  say "waiting for rank 0 (timeout $TIMEOUT_MIN min); rules: $READINESS_FILE"
  while :; do
    sleep "$READY_POLL_S"
    run_on "${NODE_NAME[0]}" 'docker logs "$1" 2>&1' "$(rank_container 0)" >"$log" 2>/dev/null || true
    lines=$(wc -l <"$log" | tr -d ' ')
    if [ "$lines" != "$last_lines" ]; then last_lines=$lines; last_change=$SECONDS; fi

    for ((i = 0; i < ${#BAD_RE[@]}; i++)); do
      if hit=$(grep -E -m1 -- "${BAD_RE[$i]}" "$log"); then
        boot_failed "boot failed: rank 0's log matches the fail rule /${BAD_RE[$i]}/" "$hit"
      fi
    done
    for ((i = 0; i < ${#INFO_RE[@]}; i++)); do
      if [ "${INFO_SEEN[$i]}" = 0 ] && hit=$(grep -E -m1 -- "${INFO_RE[$i]}" "$log"); then
        INFO_SEEN[$i]=1
        say "info (expected, not a failure): ${hit:0:160}"
      fi
    done
    if grep -q 'Application startup complete' "$log"; then break; fi

    rank_states "$TMP/boot"
    bad_list=$(not_running "$TMP/boot")
    if [ -n "$bad_list" ]; then
      boot_failed "a rank container stopped before the server was up:$bad_list"
    fi
    elapsed=$((SECONDS - t0))
    quiet=$((SECONDS - last_change))
    if [ "$elapsed" -ge "$TIMEOUT_S" ]; then
      boot_failed "not ready after $((elapsed / 60)) min (--timeout $TIMEOUT_MIN); rank 0's last log line:" "$(tail -n 1 "$log")"
    fi
    if [ "$quiet" -ge $((STALL_WARN_MIN * 60)) ] && [ $((SECONDS - last_stall)) -ge 300 ]; then
      warn "rank 0 has logged nothing for $((quiet / 60)) min; do not remove ranks while a collective may be running"
      last_stall=$SECONDS
    fi
    if [ $((SECONDS - last_report)) -ge 60 ]; then
      say "booting... $((elapsed / 60)) min, $lines log lines"
      last_report=$SECONDS
    fi
  done

  for ((i = 0; i < ${#REQ_RE[@]}; i++)); do
    if ! grep -q -E -- "${REQ_RE[$i]}" "$log"; then missing="$missing"$'\n''          '"${REQ_RE[$i]}"; fi
  done
  if [ -n "$missing" ]; then
    boot_failed "the server started, but rank 0's log lacks these required lines (a different configuration is running?):$missing"
  fi
  ok "readiness rules satisfied ($(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$log" || echo 'no KV cache line'))"

  sleep "$SETTLE_S"
  local code models
  code=$(api_code /health)
  [ "$code" = 200 ] || boot_failed "GET /health on ${NODE_NAME[0]}:$API_PORT answered $code, not 200"
  models=$(api_get /v1/models | models_summary)
  case $'\n'"$models" in
    *$'\n'"$SERVED_NAME "*) ;;
    *) boot_failed "GET /v1/models on ${NODE_NAME[0]}:$API_PORT does not list '$SERVED_NAME' (it lists: ${models:-nothing})" ;;
  esac
  probe_all "$TMP/ready" quick
  for ((i = 0; i < NNODES; i++)); do
    code=$(pget "$TMP/ready" "$i" mem_avail_gib)
    printf '  %-14s MemAvailable %6s GiB   GPU %s W %s%%\n' "${NODE_NAME[$i]}" "$code" \
      "$(pget "$TMP/ready" "$i" gpu_power_w)" "$(pget "$TMP/ready" "$i" gpu_util)"
    if is_num "$code" && flt "$code" "$MEM_SERVING_WARN_GIB"; then
      warn "${NODE_NAME[$i]}: MemAvailable $code GiB is below $MEM_SERVING_WARN_GIB GiB while serving"
    fi
  done
  ok "READY: $(printf '%s\n' "$models" | awk '{ printf "%s%s (max_model_len %s)", sep, $1, $2; sep = ", " }') on port $API_PORT of ${NODE_NAME[0]}, $(((SECONDS - t0) / 60)) min after launch"
}

# ---- run --------------------------------------------------------------------------------------------------
release_page_cache
start_ranks
if [ "$DRY" = 1 ]; then
  printf '\n'
  if [ "$WAIT" = 1 ]; then say "dry run: --wait ignored"; fi
  if [ "$PREFLIGHT_OK" = 1 ]; then
    ok "dry run complete: nothing was changed"
    exit 0
  fi
  warn "dry run complete: nothing was changed, but a real run would have been refused (see above)"
  exit 1
fi
check_running
if [ "$WAIT" = 1 ]; then
  wait_ready
else
  say "started; a first boot takes 15-30 minutes. Follow it with: ssh $(ssh_target "${NODE_NAME[0]}") docker logs -f $(rank_container 0)"
  say "launch/status.sh shows when /health answers; a start with --wait waits for you and checks the boot against launch/readiness.txt"
fi
