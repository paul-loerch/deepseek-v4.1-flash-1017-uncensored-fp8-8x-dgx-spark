#!/usr/bin/env bash
# tests/launcher_dryrun.sh -- offline test of launch/up.sh, launch/down.sh and launch/status.sh.
#
#   tests/launcher_dryrun.sh
#
# Builds a fake cluster in a temporary directory: its own cluster.env and recipe.env, a state directory per node,
# and stand-ins for ssh, docker, nvidia-smi, curl, ip, ss and sudo placed first on PATH. The ssh stand-in runs the
# "remote" command locally as the named node, so the scripts are exercised exactly as they run on a cluster, including
# the probe script and the quoting of the docker commands. No network, no docker, no cluster, nothing outside the
# temporary directory is touched. Needs bash, python3 and awk. Prints one line per failed check and a final count.
# Exit status: 0 if every check passed, 1 otherwise.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
UP=$REPO/launch/up.sh DOWN=$REPO/launch/down.sh STATUS=$REPO/launch/status.sh

T=$(mktemp -d "${TMPDIR:-/tmp}/launcher-test.XXXXXX")
trap 'rm -rf "$T"' EXIT
trap 'exit 130' INT TERM
BIN=$T/bin STUB_STATE=$T/state STUB_LOG=$T/log
mkdir -p "$BIN"

# ---- the fake site ---------------------------------------------------------------------------------------------
MODEL=/models/dsv41-flash-1017
ENGRAM=/engram-packed
HF=$T/dsv41-hf-cache   # the fake nodes share this machine's filesystem, so anything a rank script creates must live under $T
IFACE=enp1s0f1np1 HCA=rocep1s0f1 CIDR=198.51.100.0/24
IMG=dsv41-flash-sm121:test
IMGID=sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
mkdir -p "$T/patches/files"
printf '#!/usr/bin/env bash\nexec vllm serve "$@"\n' >"$T/patches/entrypoint.sh"
printf 'data\n' >"$T/patches/files/a.txt"
PATCHES=$(cd "$T/patches" && pwd -P)
GOOD_CLUSTER=$T/cluster.good.env
CLUSTER=$T/cluster.env RECIPE=$T/recipe.env READY=$T/readiness.txt

ip_of() { echo "198.51.100.$((10 + $1))"; }   # fabric address of node <n>

{
  for i in 1 2 3 4 5 6 7 8; do printf 'NODE%s=node%s\nNODE%s_FABRIC_IP=%s\n' "$i" "$i" "$i" "$(ip_of "$i")"; done
  cat <<EOF
FABRIC_IFACE=$IFACE
ROCE_HCA=$HCA
FABRIC_CIDR=$CIDR
MODEL_DIR=$MODEL
HF_CACHE_DIR=$HF
ENGRAM_PACKED_DIR=$ENGRAM
API_PORT=8000
MASTER_PORT=25100
CONTAINER_PREFIX=dsv41-rank
SSH_USER=
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=5"
GPU_MEMORY_UTILIZATION=0.70
EOF
} >"$GOOD_CLUSTER"

# A recipe written like the real one but with deliberately stale site values (--port 1, --nnodes 2, a different
# memory share): the launcher must force the cluster.env values into every rank's arguments.
cat >"$T/recipe.good.env" <<EOF
M_NAME=dsv41-test
M_BOOT_ENTRYPOINT=recipe
M_PATCH_DIR=$PATCHES
M_IMAGE=$IMG
M_VLLM_ARGS='@MODEL_DIR@ --served-model-name dsv41-test --host 0.0.0.0 --port 1 --tokenizer-mode deepseek_v41 --max-model-len 1048576 --gpu-memory-utilization 0.50 --tensor-parallel-size 2 --pipeline-parallel-size 1 --nnodes 2 --master-addr @MASTER_ADDR@ --master-port @MASTER_PORT@ --distributed-executor-backend mp --cudagraph-capture-sizes 1 2 4 --speculative-config '"'"'{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":false}'"'"' --attention-config '"'"'{"indexer_kv_dtype":"fp8"}'"'"''
M_DOCKER_EXTRA='-e VLLM_GB10_ONESHOT_AR=1 -e NCCL_MAX_NCHANNELS=4 -e TILELANG_CACHE_DIR=/cache/huggingface/tilelang-cache -e ENGRAM_NVME_PACKED_DIR=@ENGRAM_PACKED_DIR@ -v @ENGRAM_PACKED_DIR@:@ENGRAM_PACKED_DIR@:ro'
EOF

cat >"$T/readiness.good.txt" <<'EOF'
# test rules
require  \[recipe-entrypoint\] all patches applied
require  world_size=8 rank=0 .*backend=nccl
require  GPU KV cache size: [0-9,]+ tokens
require  Starting vLLM server on http://0\.0\.0\.0:@API_PORT@
require  Application startup complete
fail     OutOfMemoryError
fail     NCCL (WARN|error)
info     Custom collectives are disabled
EOF

cat >"$T/rank0.good.log" <<'EOF'
[recipe-entrypoint] applying 001-example.sh
[recipe-entrypoint] all patches applied -- starting vLLM
INFO Custom collectives are disabled
INFO world_size=8 rank=0 local_rank=0 distributed_init_method=tcp://198.51.100.11:25100 backend=nccl
INFO GPU KV cache size: 20,411,335 tokens
INFO Starting vLLM server on http://0.0.0.0:8000
INFO Application startup complete.
EOF

# ---- stand-ins ---------------------------------------------------------------------------------------------------
# Log lines are kept short (well under 1 KiB) so that parallel appends are written whole; the docker stand-in
# saves the full arguments of docker run in a file of its own.
cat >"$BIN/ssh" <<'EOF'
#!/usr/bin/env bash
# stand-in for ssh: run the remote command locally, as the named node
while [ $# -gt 0 ]; do
  case $1 in
    -o | -i | -p | -l | -F | -J | -L | -R | -D | -b | -c | -E | -e | -m | -O | -S | -W | -w) shift 2 ;;
    -*) shift ;;
    *) break ;;
  esac
done
target=$1
shift
node=${target#*@}
printf '%s %s\n' "$target" "$*" >>"$STUB_LOG/ssh.log"
if [ -e "$STUB_STATE/$node/unreachable" ]; then
  echo "ssh: connect to host $node port 22: Connection timed out" >&2
  exit 255
fi
if [ -e "$STUB_STATE/$node/no_python" ]; then PATH=$STUB_BIN:$STUB_BASIC; fi
STUB_NODE=$node PROBE_ROOT=$STUB_STATE/$node/root exec "${STUB_REMOTE_SHELL:-bash}" -c "$*"
EOF

cat >"$BIN/docker" <<'EOF'
#!/usr/bin/env bash
S=$STUB_STATE/${STUB_NODE:?docker stand-in called outside a node}
verb=$1
name=${!#}
[ "$1" = image ] && verb="image $2"
if [ "$1" = run ]; then
  name=''
  prev=''
  for a in "$@"; do
    [ "$prev" = --name ] && name=$a
    prev=$a
  done
fi
printf '%s docker %s %s\n' "$STUB_NODE" "$verb" "$name" | cut -c1-200 >>"$STUB_LOG/docker.log"
if [ -e "$S/no_docker" ]; then echo "bash: docker: command not found" >&2; exit 127; fi
case $1 in
  ps)
    all=0
    for a in "$@"; do [ "$a" = -a ] && all=1; done
    for f in "$S"/containers/*; do
      [ -e "$f" ] || continue
      if [ $all = 1 ] || [ "$(cat "$f")" = running ]; then basename "$f"; fi
    done
    ;;
  inspect)
    if [ -f "$S/containers/$name" ]; then cat "$S/containers/$name"; else echo "Error: No such object: $name" >&2; exit 1; fi
    ;;
  image)
    if [ -f "$S/image_id" ]; then cat "$S/image_id"; else echo "Error: No such image" >&2; exit 1; fi
    ;;
  rm)
    if [ -e "$S/rm_fails" ]; then echo "Error response from daemon: cannot remove container" >&2; exit 1; fi
    if [ -f "$S/containers/$name" ]; then rm -f "$S/containers/$name"; echo "$name"; else echo "Error: No such container: $name" >&2; exit 1; fi
    ;;
  run)
    shift
    mkdir -p "$S/containers"
    case " $* " in
      *" --rm "*)   # a throwaway container (down.sh's /dev/shm cleanup): delete the psm_* files of the fake node, start nothing
        printf '%s\n' "$@" >"$STUB_LOG/run.throwaway.args"
        case "$*" in *psm_*) rm -f "$S"/root/dev/shm/psm_* ;; esac
        exit 0
        ;;
    esac
    if [ -e "$S/die_after_run" ]; then echo exited >"$S/containers/$name"; else echo running >"$S/containers/$name"; fi
    printf '%s\n' "$@" >"$STUB_LOG/run.$name.args"
    echo 0123456789abcdef
    ;;
  logs)
    cat "$STUB_STATE/logs/$name" 2>/dev/null || true
    ;;
  *) echo "docker stand-in: unsupported command: $*" >&2; exit 2 ;;
esac
EOF

cat >"$BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
S=$STUB_STATE/${STUB_NODE:?}
if [ -e "$S/no_nvidia" ]; then echo "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver." >&2; exit 9; fi
cat "$S/gpu"
EOF

cat >"$BIN/curl" <<'EOF'
#!/usr/bin/env bash
A=$STUB_STATE/api
url=${!#}
code=0
for a in "$@"; do [ "$a" = -w ] && code=1; done
printf '%s curl %s\n' "${STUB_NODE:-?}" "$url" >>"$STUB_LOG/curl.log"
[ -e "$A/down" ] && { [ $code = 1 ] && printf '000\n'; exit 7; }
case $url in
  http://127.0.0.1:*/health) [ $code = 1 ] && printf '200\n' ;;
  http://127.0.0.1:*/v1/models) cat "$A/models.json" ;;
  http://127.0.0.1:*/metrics) [ -e "$A/metrics_broken" ] && exit 22; cat "$A/metrics.txt" ;;
  *) echo "curl stand-in: unexpected URL $url" >&2; exit 2 ;;
esac
EOF

cat >"$BIN/ip" <<'EOF'
#!/usr/bin/env bash
# "ip -o -4 addr show" without a device asks what addresses the machine running the launcher has
if [ "$*" = "-o -4 addr show" ]; then
  if [ -n "${STUB_LOCAL_ADDR:-}" ]; then printf '2: eth0    inet %s/24 scope global eth0\\       valid_lft forever\n' "$STUB_LOCAL_ADDR"; fi
  exit 0
fi
S=$STUB_STATE/${STUB_NODE:?}
case "$*" in
  "-o link show dev "*) [ -f "$S/link" ] && cat "$S/link" ;;
  "-o -4 addr show dev "*) [ -f "$S/addr" ] && cat "$S/addr" ;;
  *) echo "ip stand-in: unsupported: $*" >&2; exit 2 ;;
esac
exit 0
EOF

cat >"$BIN/ss" <<'EOF'
#!/usr/bin/env bash
S=$STUB_STATE/${STUB_NODE:?}
echo "State  Recv-Q Send-Q Local Address:Port Peer Address:Port Process"
[ -f "$S/listen" ] && cat "$S/listen"
exit 0
EOF

cat >"$BIN/sudo" <<'EOF'
#!/usr/bin/env bash
printf '%s sudo %s\n' "${STUB_NODE:-?}" "$*" >>"$STUB_LOG/sudo.log"
[ -e "$STUB_STATE/${STUB_NODE:-?}/sudo_denied" ] && { echo "sudo: a password is required" >&2; exit 1; }
exit 0
EOF

# a PATH without python3, for the node that has none: the stand-ins plus symlinks to the basic tools
mkdir -p "$T/basic"
for tool in bash awk sed grep tr head tail cat ls hostname sync mkdir rm wc cut sort date env sleep dirname basename; do
  p=$(command -v "$tool" 2>/dev/null) && [ -x "$p" ] && ln -sf "$p" "$T/basic/$tool"
done
for tool in scp rsync wget nc sftp; do
  printf '#!/usr/bin/env bash\nprintf "%%s %%s\\n" %s "$*" >>"$STUB_LOG/forbidden.log"\nexit 99\n' "$tool" >"$BIN/$tool"
done
chmod +x "$BIN"/*

# ---- state of the fake cluster ----------------------------------------------------------------------------------
build_state() {
  local i k n S
  rm -rf "$STUB_STATE" "$STUB_LOG" "$HF"
  mkdir -p "$STUB_STATE/api" "$STUB_STATE/logs" "$STUB_LOG"
  for i in 1 2 3 4 5 6 7 8; do
    n=node$i
    S=$STUB_STATE/$n
    mkdir -p "$S/containers" "$S/root/proc" "$S/root/sys/class/infiniband/$HCA" "$S/root$MODEL" "$S/root$PATCHES"
    echo "$IMGID" >"$S/image_id"
    echo "NVIDIA GB10, 4.21, 41, 0" >"$S/gpu"
    printf '3: %s: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 qdisc mq state UP mode DEFAULT group default qlen 1000\\    link/ether\n' "$IFACE" >"$S/link"
    printf '3: %s    inet %s/24 brd 198.51.100.255 scope global %s\\       valid_lft forever preferred_lft forever\n' "$IFACE" "$(ip_of "$i")" "$IFACE" >"$S/addr"
    set_meminfo "$n" 118000000 100000000
    printf 'aaaaaaaaaa\n' >"$S/root$MODEL/model-00001.safetensors"
    printf 'bbbbbbbbbbbbbbbbbbbb\n' >"$S/root$MODEL/model-00002.safetensors"
    printf '{}\n' >"$S/root$MODEL/config.json"
    # download bookkeeping that differs per node (count and size) and must not count as model files
    mkdir -p "$S/root$MODEL/.cache"
    head -c $((i * 10)) /dev/zero >"$S/root$MODEL/.cache/lock"
    for ((k = 0; k < i; k++)); do : >"$S/root$MODEL/.cache/extra-$k"; done
    cp -R "$T/patches/." "$S/root$PATCHES/"
    mkdir -p "$S/root$ENGRAM"
    : >"$S/root$ENGRAM/engram_L1_tp8_r$((i - 1)).pack.json"
    : >"$S/root$ENGRAM/engram_L14_tp8_r$((i - 1)).pack.json"
  done
  cat >"$STUB_STATE/api/models.json" <<'EOF'
{"object":"list","data":[{"id":"dsv41-test","object":"model","max_model_len":1048576}]}
EOF
  set_metrics 0 0
  cp "$T/rank0.good.log" "$STUB_STATE/logs/dsv41-rank0"
  cp "$GOOD_CLUSTER" "$CLUSTER"
  cp "$T/recipe.good.env" "$RECIPE"
  cp "$T/readiness.good.txt" "$READY"
}
set_meminfo() {  # set_meminfo <node> <MemAvailable kB> <MemFree kB>
  printf 'MemTotal:       127500000 kB\nMemFree:        %s kB\nMemAvailable:   %s kB\n' "$3" "$2" >"$STUB_STATE/$1/root/proc/meminfo"
}
set_metrics() {  # set_metrics <running> <waiting>
  cat >"$STUB_STATE/api/metrics.txt" <<EOF
# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="dsv41-test"} $1.0
vllm:num_requests_waiting{engine="0",model_name="dsv41-test"} $2.0
EOF
}

# run [VAR=value...] <command> [args]: run a launcher script against the fake cluster -> $OUT (stdout+stderr), $RC
run() {
  OUT=$(env PATH="$BIN:$PATH" CLUSTER_ENV="$CLUSTER" RECIPE_ENV="$RECIPE" READINESS_FILE="$READY" \
    STUB_STATE="$STUB_STATE" STUB_LOG="$STUB_LOG" STUB_BIN="$BIN" STUB_BASIC="$T/basic" UP_POLL_S=0 READY_POLL_S=0 SETTLE_S=0 VERIFY_POLL_S=0 VERIFY_WITHIN_S=0 \
    ALLOW_EXAMPLE_ADDRESSES=0 "$@" 2>&1)
  RC=$?
}

# ---- assertions ---------------------------------------------------------------------------------------------------
PASS=0 NFAIL=0
failed() {  # failed <description> [detail]
  NFAIL=$((NFAIL + 1))
  printf 'FAIL: %s\n' "$1" >&2
  if [ -n "${2:-}" ]; then printf '%s\n' "$2" | head -n 12 | sed 's/^/        /' >&2; fi
  if [ -n "${OUT:-}" ]; then printf '      last output:\n' >&2; printf '%s\n' "$OUT" | tail -n 12 | sed 's/^/        /' >&2; fi
}
good() { PASS=$((PASS + 1)); }
eq() { if [ "$2" = "$3" ]; then good; else failed "$1 (want '$2', got '$3')"; fi; }
has() { case $2 in *"$3"*) good ;; *) failed "$1 (missing: $3)" ;; esac; }
lacks() { case $2 in *"$3"*) failed "$1 (unexpected: $3)" ;; *) good ;; esac; }
rc_is() { eq "$1 (exit status)" "$2" "$RC"; }
# has_all <description> <text> <needle...>: one check; the failure names every missing needle
has_all() {
  local d=$1 text=$2 miss='' n
  shift 2
  for n in "$@"; do case $text in *"$n"*) ;; *) miss="$miss
$n" ;; esac; done
  if [ -z "$miss" ]; then good; else failed "$d (missing:$miss)"; fi
}
file_has() { if grep -qF -- "$3" "$2" 2>/dev/null; then good; else failed "$1 (missing in $(basename "$2"): $3)"; fi; }
count_is() { eq "$1" "$2" "$([ -f "$4" ] && { grep -c -E -- "$3" "$4" || true; } || echo 0)"; }
# flat <text>: join the backslash-continued lines of the docker commands
flat() { printf '%s\n' "$1" | python3 -c 'import re,sys; sys.stdout.write(re.sub(r" *\\\n *", " ", sys.stdin.read()))'; }

# =====================================================================================================================
echo "== static checks"
for f in "$REPO"/launch/lib.sh "$REPO"/launch/up.sh "$REPO"/launch/down.sh "$REPO"/launch/status.sh "$HERE/launcher_dryrun.sh"; do
  OUT=$(bash -n "$f" 2>&1) && good || failed "bash -n $(basename "$f")"
done
OUT=''
if command -v shellcheck >/dev/null 2>&1; then
  for f in "$REPO"/launch/*.sh "$HERE/launcher_dryrun.sh"; do
    OUT=$(shellcheck -x "$f" 2>&1) && good || failed "shellcheck $(basename "$f")"
  done
  OUT=''
else
  echo "   (shellcheck is not installed; skipped)"
fi
for f in up.sh down.sh status.sh; do [ -x "$REPO/launch/$f" ] && good || failed "launch/$f is not executable"; done
# nothing from a private site in the files this test covers: SITE_PATTERNS=<file kept OUTSIDE the repository> (one fixed
# string per line, '#' comments) enables the check; without it the check is skipped
if [ -n "${SITE_PATTERNS:-}" ] && [ -f "$SITE_PATTERNS" ]; then
  pf=$(mktemp); grep -vE '^(#|$)' "$SITE_PATTERNS" > "$pf"
  hits=$(grep -rnIiF -f "$pf" "$REPO/launch" "$REPO/cluster.env.example" "$HERE" 2>/dev/null || true); rm -f "$pf"
  if [ -z "$hits" ]; then good; else failed "site-specific text found" "$hits"; fi
else
  echo "   (SITE_PATTERNS not set: site-specific text check skipped)"
fi
hits=$(grep -rnIE '(^|[^0-9.])(10\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01])|192\.168|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7]))\.[0-9]+\.[0-9]+' \
  "$REPO/launch" "$REPO/cluster.env.example" "$HERE" 2>/dev/null || true)
if [ -z "$hits" ]; then good; else failed "private IPv4 address found" "$hits"; fi

# =====================================================================================================================
echo "== configuration: placeholders and bad values are rejected (exit 2, before any node is contacted)"
build_state
cp "$REPO/cluster.env.example" "$CLUSTER"
run "$UP" --dry-run
rc_is "unedited cluster.env.example" 2
has_all "unedited example names what to fix" "$OUT" "NODE1" "CHANGE_ME" "FABRIC_IFACE" "ROCE_HCA" "MODEL_DIR" "example address"
lacks "unedited example: no node was contacted" "$(cat "$STUB_LOG/ssh.log" 2>/dev/null)" "node"
run "$DOWN" --verify
rc_is "down.sh with the unedited example" 2
run "$STATUS"
rc_is "status.sh with the unedited example" 2

variant() {  # variant <description> <expected text> <sed expression>: edit the good cluster.env, run up.sh, expect exit 2
  sed -E "$3" "$GOOD_CLUSTER" >"$CLUSTER"
  run "$UP" --dry-run
  rc_is "$1" 2
  has "$1: message" "$OUT" "$2"
}
variant "one node left as CHANGE_ME" "NODE5" 's/^NODE5=.*/NODE5=CHANGE_ME/'
variant "angle-bracket placeholder" "NODE2" "s/^NODE2=.*/NODE2='<second-node>'/"
variant "example fabric address left in" "example address" 's/^NODE3_FABRIC_IP=.*/NODE3_FABRIC_IP=192.0.2.13/'
variant "malformed address" "NODE4_FABRIC_IP" 's/^NODE4_FABRIC_IP=.*/NODE4_FABRIC_IP=198.51.100.999/'
variant "duplicate fabric address" "both 198.51.100.11" 's/^NODE2_FABRIC_IP=.*/NODE2_FABRIC_IP=198.51.100.11/'
variant "duplicate node name" "both 'node1'" 's/^NODE2=.*/NODE2=node1/'
variant "address outside FABRIC_CIDR" "outside FABRIC_CIDR" 's/^NODE6_FABRIC_IP=.*/NODE6_FABRIC_IP=203.0.113.16/'
variant "example CIDR" "example subnet" 's|^FABRIC_CIDR=.*|FABRIC_CIDR=192.0.2.0/24|'
variant "relative model path" "MODEL_DIR" 's|^MODEL_DIR=.*|MODEL_DIR=models/x|'
variant "model path with a space" "MODEL_DIR" 's|^MODEL_DIR=.*|MODEL_DIR="/models/a b"|'
variant "interface left as CHANGE_ME" "FABRIC_IFACE" 's/^FABRIC_IFACE=.*/FABRIC_IFACE=CHANGE_ME/'
variant "same port twice" "must differ" 's/^MASTER_PORT=.*/MASTER_PORT=8000/'
variant "memory share above 1" "GPU_MEMORY_UTILIZATION" 's/^GPU_MEMORY_UTILIZATION=.*/GPU_MEMORY_UTILIZATION=1.5/'
variant "container prefix with a space" "CONTAINER_PREFIX" 's/^CONTAINER_PREFIX=.*/CONTAINER_PREFIX="a b"/'
# several mistakes are all reported together
sed -E 's/^NODE1=.*/NODE1=CHANGE_ME/; s/^ROCE_HCA=.*/ROCE_HCA=CHANGE_ME/; s|^MODEL_DIR=.*|MODEL_DIR=CHANGE_ME|' "$GOOD_CLUSTER" >"$CLUSTER"
run "$UP" --dry-run
has_all "all problems are listed in one go" "$OUT" "3 problem(s)" "NODE1" "ROCE_HCA" "MODEL_DIR"
rm -f "$CLUSTER"
run "$UP" --dry-run
rc_is "missing cluster.env" 2
has "missing cluster.env: tells how to create it" "$OUT" "cluster.env.example"
cp "$GOOD_CLUSTER" "$CLUSTER"

variant "packed Engram directory left as CHANGE_ME" "ENGRAM_PACKED_DIR" 's/^ENGRAM_PACKED_DIR=.*/ENGRAM_PACKED_DIR=CHANGE_ME/'
variant "packed Engram directory not set" "ENGRAM_PACKED_DIR" '/^ENGRAM_PACKED_DIR=/d'
variant "relative packed Engram directory" "ENGRAM_PACKED_DIR" 's|^ENGRAM_PACKED_DIR=.*|ENGRAM_PACKED_DIR=packs|'
cp "$GOOD_CLUSTER" "$CLUSTER"

echo "== configuration: recipe.env"
sed 's/@MASTER_ADDR@/@MASTER_ADRESS@/' "$T/recipe.good.env" >"$RECIPE"
run "$UP" --dry-run
rc_is "unknown placeholder in recipe.env" 2
has "unknown placeholder is named" "$OUT" "@MASTER_ADRESS@"
sed 's/^M_IMAGE=.*//' "$T/recipe.good.env" >"$RECIPE"
run "$UP" --dry-run
rc_is "recipe.env without M_IMAGE" 2
sed "s/^M_PATCH_DIR=.*/M_PATCH_DIR=$(printf '%s' "$T" | sed 's/[\/&]/\\&/g')\/nope/" "$T/recipe.good.env" >"$RECIPE"
run "$UP" --dry-run
rc_is "missing patch directory" 2
has "missing patch directory is named" "$OUT" "entrypoint.sh"
sed "s/--master-port @MASTER_PORT@/--node-rank 3/" "$T/recipe.good.env" >"$RECIPE"
run "$UP" --dry-run
rc_is "recipe.env that sets --node-rank" 2
cp "$T/recipe.good.env" "$RECIPE"
run "$UP" --bogus
rc_is "unknown option" 2
run "$UP" --timeout abc --wait
rc_is "bad --timeout" 2
printf 'bogus  something\n' >"$READY"
run "$UP" --dry-run --wait
rc_is "readiness rule of an unknown kind" 2
printf 'require  (unclosed\n' >"$READY"
run "$UP" --dry-run --wait
rc_is "readiness rule with an invalid regex" 2
cp "$T/readiness.good.txt" "$READY"
if [ -f "$REPO/launch/readiness.txt" ]; then
  run READINESS_FILE="$REPO/launch/readiness.txt" "$UP" --dry-run --wait
  eq "the shipped launch/readiness.txt parses" 0 "$RC"
else
  echo "   (launch/readiness.txt not present; skipped)"
fi

# =====================================================================================================================
echo "== up.sh --dry-run: the eight docker run commands"
build_state
run "$UP" --dry-run
rc_is "dry run on a healthy cluster" 0
has_all "dry run says what it did" "$OUT" "DRY RUN" "read-only probe" "preflight passed on all 8 nodes" "dry run complete: nothing was changed"
DRY=$(flat "$OUT")
CMDS=$(printf '%s\n' "$DRY" | grep -E '^ *docker run ' || true)
eq "eight docker run commands" 8 "$(printf '%s\n' "$CMDS" | grep -c 'docker run')"

COMMON=(
  "--gpus all" "--network host" "--ipc=host" "--ulimit memlock=-1" "--ulimit stack=67108864" "--ulimit nofile=524288:524288"
  "--shm-size 32g" "--device=/dev/infiniband"
  "-v $MODEL:$MODEL" "-v $ENGRAM:$ENGRAM:ro" "-e ENGRAM_NVME_PACKED_DIR=$ENGRAM" "-v $HF:/cache/huggingface" "-e HF_HOME=/cache/huggingface" "-e HF_HUB_OFFLINE=1"
  "-e NCCL_NET=IB" "-e NCCL_IB_HCA==$HCA" "-e NCCL_SOCKET_IFNAME==$IFACE" "-e GLOO_SOCKET_IFNAME=$IFACE"
  "-e TP_SOCKET_IFNAME=$IFACE" "-e NCCL_IB_ADDR_RANGE=$CIDR" "-e NCCL_NVLS_ENABLE=0" "-e NCCL_CUMEM_ENABLE=0"
  "-e NCCL_IGNORE_CPU_AFFINITY=1" "-e NCCL_DEBUG=WARN" "-e VLLM_ENGINE_READY_TIMEOUT_S=3600"
  "-e UCX_MEM_MMAP_HOOK_MODE=none" "-e UCX_RCACHE_MAX_UNRELEASED=1024" "-e OMP_NUM_THREADS=8"
  "-e VLLM_GB10_ONESHOT_AR=1" "-e NCCL_MAX_NCHANNELS=4" "-e TILELANG_CACHE_DIR=/cache/huggingface/tilelang-cache"
  "-v $PATCHES:/opt/recipe-patches:ro" "--entrypoint bash" "$IMG /opt/recipe-patches/entrypoint.sh $MODEL --served-model-name dsv41-test"
  "--tensor-parallel-size 8" "--nnodes 8" "--master-addr $(ip_of 1)" "--master-port 25100" "--port 8000"
  "--gpu-memory-utilization 0.70" "--distributed-executor-backend mp" "--cudagraph-capture-sizes 1 2 4"
  "--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"enable_adaptive_verification\":false}'"
  "--attention-config '{\"indexer_kv_dtype\":\"fp8\"}'"
)
for r in 0 1 2 3 4 5 6 7; do
  c=$(printf '%s\n' "$CMDS" | grep -F -- "--name dsv41-rank$r " || true)
  has_all "rank $r: every common flag, environment variable and vLLM argument" "$c" "${COMMON[@]}"
  has_all "rank $r: its own name, address and rank" "$c" "docker run -d --name dsv41-rank$r " "-e VLLM_HOST_IP=$(ip_of $((r + 1))) " "--node-rank $r"
  lacks "rank $r: no stale template values" "$c" "--nnodes 2"
  lacks "rank $r: no stale port" "$c" "--port 1 "
  lacks "rank $r: no stale memory share" "$c" "--gpu-memory-utilization 0.50"
  if [ "$r" -eq 0 ]; then
    lacks "rank 0 serves the API: not headless" "$c" "--headless"
    has "rank 0 is started on node1" "$DRY" "# [dry-run] node1: "
  else
    has "rank $r is headless" "$c" "--node-rank $r --headless >/dev/null"
    has "rank $r is started on node$((r + 1))" "$DRY" "# [dry-run] node$((r + 1)): "
  fi
done
# workers first, rank 0 last
pos0=$(printf '%s\n' "$DRY" | grep -n -- '--name dsv41-rank0 ' | head -1 | cut -d: -f1)
pos7=$(printf '%s\n' "$DRY" | grep -n -- '--name dsv41-rank7 ' | head -1 | cut -d: -f1)
if [ -n "$pos0" ] && [ -n "$pos7" ] && [ "$pos7" -lt "$pos0" ]; then good; else failed "ranks 1..7 are listed before rank 0"; fi
has "the rank commands create the cache directory" "$DRY" "mkdir -p $HF"
# a stranger could paste the printed command: it must still parse as shell
printf '%s\n' "$CMDS" | head -1 >"$T/one.cmd"
OUT=$(bash -n "$T/one.cmd" 2>&1) && good || failed "a printed docker run command is valid shell"
OUT=''

echo "== up.sh --dry-run changed nothing"
dlog=$(cat "$STUB_LOG/docker.log")
lacks "no docker run/rm/stop/kill/load/pull in the dry run" "$dlog" " docker run"
lacks "no docker rm in the dry run" "$dlog" " docker rm"
lacks "no docker stop/kill/start in the dry run" "$dlog" " docker st"
lacks "no docker kill in the dry run" "$dlog" " docker kill"
lacks "no docker pull in the dry run" "$dlog" " docker pull"
eq "no container was created" 0 "$(ls "$STUB_STATE"/node*/containers 2>/dev/null | grep -c dsv41 || true)"
eq "no sudo call" 0 "$([ -f "$STUB_LOG/sudo.log" ] && wc -l <"$STUB_LOG/sudo.log" | tr -d ' ' || echo 0)"
eq "no scp/rsync/wget/nc call" 0 "$([ -f "$STUB_LOG/forbidden.log" ] && wc -l <"$STUB_LOG/forbidden.log" | tr -d ' ' || echo 0)"
eq "one read-only ssh probe per node" 8 "$(wc -l <"$STUB_LOG/ssh.log" | tr -d ' ')"

echo "== up.sh --dry-run with DROP_CACHES=1 prints the cache release and does not run it"
echo "DROP_CACHES=1" >>"$CLUSTER"
run "$UP" --dry-run
DRY1=$(flat "$OUT")
rc_is "dry run with DROP_CACHES=1" 0
has_all "cache release is printed" "$DRY1" "posix_fadvise" "drop_caches" "node8"
eq "nothing ran sudo" 0 "$([ -f "$STUB_LOG/sudo.log" ] && wc -l <"$STUB_LOG/sudo.log" | tr -d ' ' || echo 0)"
cp "$GOOD_CLUSTER" "$CLUSTER"

echo "== up.sh: a dry run on a cluster that would be refused still prints the commands, and exits 1"
build_state
set_meminfo node4 8388608 7340032
run "$UP" --dry-run
rc_is "dry run that would be refused" 1
has_all "dry run reports the problem and still shows the commands" "$OUT" "node4: MemAvailable is 8.0 GiB" "a real run would stop here" "docker run -d --name dsv41-rank0"

# =====================================================================================================================
echo "== the shipped config/recipe.env renders through the launcher (its own flags and patch switches reach every rank)"
build_state
run env RECIPE_ENV="$REPO/config/recipe.env" "$UP" --dry-run
SHIPPED=$(flat "$OUT")
eq "shipped recipe: eight docker run commands" 8 "$(printf '%s\n' "$SHIPPED" | grep -c -E '^ *docker run ')"
has_all "shipped recipe: key flags and patch switches are in the commands" "$SHIPPED" \
  "--max-num-seqs 128" "--moe-backend b12x" "--kv-cache-memory-bytes " "-e VLLM_GB10_ONESHOT_AR=1" "-e VLLM_GB10_EMBED_TP=1" \
  "-e VLLM_GB10_INDEXER_DECODE_CHUNK_MB=512" "-e VLLM_GB10_SHARED_CAPTURE_STREAM=1" "-e ENGRAM_NVME_PACKED_DIR=$ENGRAM" \
  "-v $ENGRAM:$ENGRAM:ro" "--entrypoint bash"
lacks "shipped recipe: no profiler flag (it would expose /start_profile)" "$SHIPPED" "--profiler-config"
lacks "shipped recipe: no unresolved placeholder" "$SHIPPED" "@ENGRAM_PACKED_DIR@"

echo "== up.sh preflight refuses (exit 1, nothing started)"
refuse() {  # refuse <description> <expected text> <mutation command...>
  local d=$1 want=$2
  shift 2
  build_state
  "$@"
  run "$UP"
  rc_is "$d" 1
  has "$d: message" "$OUT" "$want"
  has "$d: says nothing was changed" "$OUT" "nothing was changed"
  lacks "$d: no container started" "$(cat "$STUB_LOG/docker.log")" " docker run"
}
refuse "unreachable node" "node3: cannot run commands over ssh" touch "$STUB_STATE/node3/unreachable"
refuse "docker missing" "node5: docker is not usable" touch "$STUB_STATE/node5/no_docker"
refuse "image missing" "node6: image $IMG is not present" rm -f "$STUB_STATE/node6/image_id"
refuse "image ID differs" "the image IDs differ between nodes" sh -c "echo sha256:ffff >'$STUB_STATE/node7/image_id'"
refuse "model directory missing" "node8: model directory $MODEL does not exist" rm -rf "$STUB_STATE/node8/root$MODEL"
refuse "model file count differs" "the model directory differs between nodes" rm -f "$STUB_STATE/node2/root$MODEL/config.json"
refuse "model size differs" "the model directory differs between nodes" sh -c "echo more >>'$STUB_STATE/node4/root$MODEL/model-00001.safetensors'"
refuse "MemAvailable too low" "node4: MemAvailable is 8.0 GiB, the minimum is 40 GiB" set_meminfo node4 8388608 7340032
refuse "leftover rank container" "node2: rank container(s) already exist: dsv41-rank1" sh -c "echo exited >'$STUB_STATE/node2/containers/dsv41-rank1'"
refuse "fabric interface down" "node3: fabric interface $IFACE is not up" sh -c "sed -i.bak 's/,UP,LOWER_UP/,UP/; s/state UP/state DOWN/' '$STUB_STATE/node3/link'"
refuse "fabric interface missing" "node3: fabric interface $IFACE does not exist" rm -f "$STUB_STATE/node3/link"
refuse "fabric address not on the interface" "node6: NODE6_FABRIC_IP $(ip_of 6) is not an address of $IFACE" sh -c "sed -i.bak 's/$(ip_of 6)/203.0.113.9/' '$STUB_STATE/node6/addr'"
refuse "RoCE device missing" "node7: RoCE device $HCA not found" rm -rf "$STUB_STATE/node7/root/sys/class/infiniband/$HCA"
refuse "nvidia-smi fails" "node8: nvidia-smi failed" touch "$STUB_STATE/node8/no_nvidia"
refuse "API port already in use" "node1: port(s) already in use" sh -c "echo 'LISTEN 0 4096 0.0.0.0:8000 0.0.0.0:*' >'$STUB_STATE/node1/listen'"
refuse "master port already in use" "node1: port(s) already in use" sh -c "echo 'LISTEN 0 4096 *:25100 *:*' >'$STUB_STATE/node1/listen'"
refuse "patch directory differs" "node4: the patch directory $PATCHES is" sh -c "echo extra >'$STUB_STATE/node4/root$PATCHES/extra.sh'"
refuse "patch directory missing" "rsync -a --delete $PATCHES/ node5:$PATCHES/" rm -rf "$STUB_STATE/node5/root$PATCHES"
refuse "python3 missing on a node" "node6: python3 not found" touch "$STUB_STATE/node6/no_python"
refuse "packed Engram directory missing on a node" "node5: the packed Engram directory $ENGRAM does not exist" rm -rf "$STUB_STATE/node5/root$ENGRAM"
refuse "no packed Engram rows for a node's rank" "node4: $ENGRAM has no packed Engram rows for rank 3" sh -c "rm -f '$STUB_STATE/node4/root$ENGRAM'/engram_L*_tp8_r3.pack.json"
refuse "packed Engram rows of another rank only" "node2: $ENGRAM has no packed Engram rows for rank 1" sh -c "cd '$STUB_STATE/node2/root$ENGRAM' && mv engram_L1_tp8_r1.pack.json engram_L1_tp8_r5.pack.json && mv engram_L14_tp8_r1.pack.json engram_L14_tp8_r5.pack.json"

echo "== up.sh preflight warns but goes on"
build_state
rm -f "$STUB_STATE/node3/root$ENGRAM/engram_L14_tp8_r2.pack.json"
run "$UP" --dry-run
rc_is "one packed Engram layer missing only warns" 0
has "one-layer warning names the node and the rank" "$OUT" "node3: only one packed Engram layer found for rank 2"
build_state
sed -i.bak 's/mtu 9000/mtu 1500/' "$STUB_STATE/node5/link"
run "$UP" --dry-run
rc_is "MTU 1500 only warns" 0
has "MTU warning names the node and values" "$OUT" "node5: $IFACE has MTU 1500, expected 9000"
build_state
set_meminfo node3 118000000 30000000
run "$UP" --dry-run
rc_is "low MemFree only warns" 0
has_all "MemFree warning explains the page cache" "$OUT" "node3: only 28.6 GiB of memory is free" "DROP_CACHES=1"
build_state
echo running >"$STUB_STATE/node2/containers/other-service"
run "$UP" --dry-run
rc_is "foreign container only warns" 0
has "foreign container is named" "$OUT" "node2: other container(s) are running: other-service"

# =====================================================================================================================
echo "== up.sh: the real start against the stand-ins"
build_state
run "$UP"
rc_is "up.sh on a healthy cluster" 0
has_all "up.sh reports the start" "$OUT" "preflight passed" "starting ranks 1..7" "starting rank 0" "all 8 rank containers are running"
eq "eight docker run calls" 8 "$(grep -c ' docker run ' "$STUB_LOG/docker.log")"
order=$(grep ' docker run ' "$STUB_LOG/docker.log" | awk '{ print $1 }' | tr '\n' ' ')
eq "rank 0 (node1) is started last" "node1" "$(printf '%s\n' "$order" | awk '{ print $NF }')"
eq "seven different workers were started before it" "node2 node3 node4 node5 node6 node7 node8" \
  "$(printf '%s\n' "$order" | awk '{ for (i = 1; i < NF; i++) print $i }' | sort | tr '\n' ' ' | sed 's/ $//')"
for r in 0 1 2 3 4 5 6 7; do
  f=$STUB_LOG/run.dsv41-rank$r.args
  [ -f "$f" ] && good || { failed "rank $r: docker run was called"; continue; }
  file_has "rank $r: --node-rank value" "$f" "$r"
  eq "rank $r: argument after --node-rank" "$r" "$(awk '$0 == "--node-rank" { getline; print; exit }' "$f")"
  if [ "$r" -eq 0 ]; then
    eq "rank 0: no --headless" 0 "$(grep -c -x -- '--headless' "$f")"
  else
    eq "rank $r: --headless is the last argument" "--headless" "$(tail -n 1 "$f")"
  fi
  file_has "rank $r: VLLM_HOST_IP" "$f" "VLLM_HOST_IP=$(ip_of $((r + 1)))"
  file_has "rank $r: exact-match HCA" "$f" "NCCL_IB_HCA==$HCA"
  file_has "rank $r: JSON argument stays one argument" "$f" '{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":false}'
  eq "rank $r: served on its own node" 1 "$(grep -c "^node$((r + 1)) docker run dsv41-rank$r\$" "$STUB_LOG/docker.log")"
done
eq "rank 3: model path is the first vLLM argument" "$MODEL" "$(awk 'prev == "/opt/recipe-patches/entrypoint.sh" { print; exit } { prev = $0 }' "$STUB_LOG/run.dsv41-rank3.args")"
if [ -d "$HF" ]; then good; else failed "the start script creates the cache directory"; fi
count_is "the launcher never removed anything" 0 ' docker rm' "$STUB_LOG/docker.log"
count_is "sudo was not used (DROP_CACHES=0)" 0 '.' "$STUB_LOG/sudo.log"
run "$UP"
rc_is "a second up.sh is refused (containers exist)" 1
has "second up.sh names the leftovers" "$OUT" "rank container(s) already exist"
eq "no second round of docker run" 8 "$(grep -c ' docker run ' "$STUB_LOG/docker.log")"

echo "== up.sh when the remote login shell is plain sh"
build_state
run STUB_REMOTE_SHELL=sh "$UP"
rc_is "up.sh with sh as the remote login shell" 0
eq "eight docker run calls through sh" 8 "$(grep -c ' docker run ' "$STUB_LOG/docker.log")"
file_has "JSON argument survives sh" "$STUB_LOG/run.dsv41-rank6.args" '{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":false}'
file_has "exact-match HCA survives sh" "$STUB_LOG/run.dsv41-rank6.args" "NCCL_IB_HCA==$HCA"
run STUB_REMOTE_SHELL=sh "$DOWN" --force
rc_is "down.sh with sh as the remote login shell" 0

echo "== up.sh as a different ssh user, and with DROP_CACHES=1"
build_state
echo "SSH_USER=ops" >>"$CLUSTER"
echo "DROP_CACHES=1" >>"$CLUSTER"
run "$UP"
rc_is "up.sh with SSH_USER and DROP_CACHES=1" 0
has "ssh targets carry the user" "$(cat "$STUB_LOG/ssh.log")" "ops@node2 "
eq "every ssh target carries the user" 0 "$(awk '$1 !~ /^ops@/' "$STUB_LOG/ssh.log" | wc -l | tr -d ' ')"
eq "page cache released on all eight nodes" 8 "$(grep -c "sudo -n sh -c echo 3 > /proc/sys/vm/drop_caches" "$STUB_LOG/sudo.log")"
has "post-release memory is reported" "$OUT" "MemFree"
build_state
echo "DROP_CACHES=1" >>"$CLUSTER"
touch "$STUB_STATE/node3/sudo_denied"
set_meminfo node3 118000000 30000000
run "$UP"
rc_is "up.sh when sudo is denied on one node" 0
has "denied sudo warns and names the sudoers rule" "$OUT" "node3: could not drop the page cache"
has_all "memory still low after the release is reported" "$OUT" "node3: only 28.6 GiB is free after releasing the page cache" "vLLM wants 85 GiB"

echo "== up.sh when a rank container dies right after the start"
build_state
touch "$STUB_STATE/node5/die_after_run"
run "$UP"
rc_is "a rank container that exits right after start" 1
has_all "dead rank is reported with the next step" "$OUT" "not running:" "node5:dsv41-rank4=exited" "launch/down.sh"

# =====================================================================================================================
echo "== up.sh --wait"
build_state
run "$UP" --wait
rc_is "--wait on a good boot" 0
has_all "--wait: readiness satisfied, API checked, ready" "$OUT" "readiness rules satisfied" "GPU KV cache size: 20,411,335 tokens" "READY: dsv41-test (max_model_len 1048576) on port 8000 of node1" "MemAvailable"
has "--wait: info rule is reported once and is not a failure" "$OUT" "info (expected, not a failure): INFO Custom collectives are disabled"
eq "--wait: info rule reported once" 1 "$(printf '%s\n' "$OUT" | grep -c 'info (expected')"
eq "--wait asked only the API on rank 0's node, on loopback" 0 "$(grep -vc '^node1 curl http://127.0.0.1:8000/' "$STUB_LOG/curl.log")"

build_state
printf 'INFO loading\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 GiB\n' >"$STUB_STATE/logs/dsv41-rank0"
run "$UP" --wait
rc_is "--wait on a log with a fail rule" 1
has_all "fail rule: the matched line and the way out" "$OUT" "fail rule /OutOfMemoryError/" "torch.OutOfMemoryError: CUDA out of memory" "launch/down.sh" "Nothing was stopped"
eq "fail rule: nothing was removed" 0 "$(grep -c ' docker rm' "$STUB_LOG/docker.log")"

build_state
grep -v 'GPU KV cache' "$T/rank0.good.log" >"$STUB_STATE/logs/dsv41-rank0"
run "$UP" --wait
rc_is "--wait when a required line is missing" 1
has_all "missing rule is listed" "$OUT" "lacks these required lines" "GPU KV cache size"

build_state
printf 'INFO still loading weights\n' >"$STUB_STATE/logs/dsv41-rank0"
run READY_POLL_S=1 "$UP" --wait --timeout 0.03
rc_is "--wait that times out" 1
has_all "timeout names the last log line and the next step" "$OUT" "not ready after" "INFO still loading weights" "launch/down.sh"

build_state
touch "$STUB_STATE/api/down"
run "$UP" --wait
rc_is "--wait when /health does not answer" 1
has "unhealthy API is reported" "$OUT" "/health"

build_state
echo '{"object":"list","data":[{"id":"other-model","max_model_len":8192}]}' >"$STUB_STATE/api/models.json"
run "$UP" --wait
rc_is "--wait when /v1/models lacks the served name" 1
has "wrong model is reported" "$OUT" "does not list 'dsv41-test'"

build_state
printf 'INFO still loading\n' >"$STUB_STATE/logs/dsv41-rank0"
touch "$STUB_STATE/node4/die_after_run"
run READY_POLL_S=1 "$UP" --wait --timeout 0.2
rc_is "--wait when a worker container is gone" 1
has "dead worker is named" "$OUT" "node4:dsv41-rank3=exited"

# =====================================================================================================================
echo "== status.sh"
build_state
run "$UP"
run "$STATUS"
rc_is "status.sh on a running cluster" 0
has_all "status.sh: container states, GPU power/temperature/util, memory" "$OUT" "dsv41-rank0" "dsv41-rank7" "running" "4.21" "41" "112.5"
has_all "status.sh: health, model, max_model_len, requests" "$OUT" "/health" "200" "dsv41-test (max_model_len 1048576)" "running 0, waiting 0"
eq "status.sh lists eight nodes" 8 "$(printf '%s\n' "$OUT" | grep -c -E '^ +node[1-8] ')"
set_metrics 3 2
run "$STATUS"
has "status.sh shows busy requests" "$OUT" "running 3, waiting 2"
rm -f "$STUB_STATE/node6/containers/dsv41-rank5"
run "$STATUS"
rc_is "status.sh with a missing rank container" 1
has_all "status.sh names the absent rank" "$OUT" "dsv41-rank5" "absent" "not healthy"
touch "$STUB_STATE/node2/unreachable"
run "$STATUS"
has "status.sh shows an unreachable node" "$OUT" "UNREACHABLE"
touch "$STUB_STATE/api/down"
rm -f "$STUB_STATE/node2/unreachable"
run "$STATUS"
rc_is "status.sh with the API down" 1
has "status.sh shows no answer from the API" "$OUT" "no answer"
run "$STATUS" --nope
rc_is "status.sh with an unknown option" 2

# =====================================================================================================================
echo "== down.sh"
build_state
run "$UP"
set_metrics 2 0
run "$DOWN"
rc_is "down.sh with running requests" 1
has_all "refusal names the requests and the override" "$OUT" "refusing: 2 request(s) are running" "--force"
eq "refusal removed nothing" 0 "$(grep -c ' docker rm' "$STUB_LOG/docker.log")"
set_metrics 0 4
run "$DOWN"
rc_is "down.sh with waiting requests" 1
has "refusal names the waiting requests" "$OUT" "refusing: 4 request(s) are waiting"
touch "$STUB_STATE/api/metrics_broken"
run "$DOWN"
rc_is "down.sh with unreadable /metrics" 1
has_all "unknown is not idle" "$OUT" "not readable" "launch/down.sh --force"
eq "unreadable /metrics removed nothing" 0 "$(grep -c ' docker rm' "$STUB_LOG/docker.log")"
run "$DOWN" --verify
rc_is "--verify while containers exist" 1
has_all "--verify says which node is not idle, with its GPU and memory figures" "$OUT" "rank container present" "4.21" "112.5" "node1"
lacks "--verify removes nothing" "$(cat "$STUB_LOG/docker.log")" " docker rm"
run "$DOWN" --force
rc_is "down.sh --force" 0
has_all "--force warns, removes, verifies" "$OUT" "--force" "removing rank 0 on node1 (last)" "all 8 nodes are idle"
rms=$(grep ' docker rm ' "$STUB_LOG/docker.log" | awk '{ print $1 }' | tr '\n' ' ')
eq "rank 0 (node1) is removed last" "node1" "$(printf '%s\n' "$rms" | awk '{ print $NF }')"
eq "the seven workers were removed before it" "node2 node3 node4 node5 node6 node7 node8" \
  "$(printf '%s\n' "$rms" | awk '{ for (i = 1; i < NF; i++) print $i }' | sort | tr '\n' ' ' | sed 's/ $//')"
eq "no container left" 0 "$(ls "$STUB_STATE"/node*/containers | grep -c dsv41 || true)"
run "$DOWN" --verify
rc_is "--verify on an idle cluster" 0
has_all "--verify prints the table" "$OUT" "avail GiB" "GPU W" "ok" "all 8 nodes are idle"
eq "--verify prints a row per node" 8 "$(printf '%s\n' "$OUT" | grep -c -E '^ +node[1-8] ')"

build_state
run "$UP"
set_metrics 0 0
rm -f "$STUB_LOG/curl.log"
run "$DOWN"
rc_is "down.sh with an idle API" 0
has "down.sh reports removal and idleness" "$OUT" "all 8 nodes are idle, no rank containers"
eq "the idle check read /metrics on node1" 1 "$(grep -c '^node1 curl http://127.0.0.1:8000/metrics' "$STUB_LOG/curl.log")"
rm -f "$STUB_LOG/curl.log"
run "$DOWN"
rc_is "down.sh when nothing is running" 0
has "down.sh knows there is nothing to remove" "$OUT" "nothing to remove"
eq "no /metrics read when there is nothing to remove" 0 "$([ -f "$STUB_LOG/curl.log" ] && wc -l <"$STUB_LOG/curl.log" | tr -d ' ' || echo 0)"

echo "== down.sh removes the stale /dev/shm/psm_* rings on the head node (and only then)"
build_state
run "$UP"
mkdir -p "$STUB_STATE/node1/root/dev/shm" "$STUB_STATE/node2/root/dev/shm"
: >"$STUB_STATE/node1/root/dev/shm/psm_aaaa"; : >"$STUB_STATE/node1/root/dev/shm/psm_bbbb"; : >"$STUB_STATE/node1/root/dev/shm/other_file"
: >"$STUB_STATE/node2/root/dev/shm/psm_cccc"
run "$DOWN" --force
rc_is "down.sh with stale shared-memory rings" 0
has "down.sh reports the cleanup" "$OUT" "stale shared-memory rings on node1: removed 2 of 2 psm_* segments"
eq "the head node's psm_* files are gone" 0 "$(ls "$STUB_STATE/node1/root/dev/shm" | grep -c '^psm_' || true)"
eq "other files in /dev/shm are left alone" 1 "$(ls "$STUB_STATE/node1/root/dev/shm" | grep -c '^other_file$' || true)"
eq "a worker's /dev/shm is not touched" 1 "$(ls "$STUB_STATE/node2/root/dev/shm" | grep -c '^psm_cccc$' || true)"
has_all "the cleanup is one throwaway container of the recipe image" "$(cat "$STUB_LOG/run.throwaway.args")" "--rm" "--network" "none" "/dev/shm:/dev/shm" "--entrypoint" "$IMG" 'find /dev/shm -maxdepth 1 -name "psm_*" -delete'
build_state
run "$UP"
mkdir -p "$STUB_STATE/node1/root/dev/shm"; : >"$STUB_STATE/node1/root/dev/shm/psm_aaaa"
echo "CLEAN_SHM=0" >>"$CLUSTER"
run "$DOWN" --force
rc_is "down.sh with CLEAN_SHM=0" 0
lacks "CLEAN_SHM=0 does not report a cleanup" "$OUT" "shared-memory rings"
eq "CLEAN_SHM=0 leaves the files" 1 "$(ls "$STUB_STATE/node1/root/dev/shm" | grep -c '^psm_aaaa$' || true)"
cp "$GOOD_CLUSTER" "$CLUSTER"
build_state
run "$UP"
mkdir -p "$STUB_STATE/node1/root/dev/shm"; : >"$STUB_STATE/node1/root/dev/shm/psm_aaaa"
run "$DOWN" --verify
eq "--verify never touches /dev/shm" 1 "$(ls "$STUB_STATE/node1/root/dev/shm" | grep -c '^psm_aaaa$' || true)"
build_state
touch "$STUB_STATE/node1/no_docker"
run "$DOWN" --force
rc_is "down.sh with docker unusable on the head node" 1

build_state
run "$UP"
echo "NVIDIA GB10, 31.0, 62, 97" >"$STUB_STATE/node4/gpu"
run "$DOWN" --force
rc_is "down.sh when a GPU stays busy after the removal" 1
has_all "busy GPU is named" "$OUT" "node4" "GPU busy (util 97 %)" "orphaned context"
build_state
run "$UP"
echo "NVIDIA GB10, [N/A], 41, 0" >"$STUB_STATE/node2/gpu"
run "$DOWN" --force
rc_is "down.sh when the driver reports no power figure" 0
has "a missing power figure does not make a GPU busy" "$OUT" "all 8 nodes are idle"
build_state
echo "NVIDIA GB10, 18.5, 50, 0" >"$STUB_STATE/node5/gpu"
run "$DOWN" --verify
rc_is "--verify with a GPU drawing power" 1
has "a GPU drawing power is not idle" "$OUT" "GPU busy (18.5 W)"
build_state
run "$UP"
touch "$STUB_STATE/node7/rm_fails"
run "$DOWN" --force
rc_is "down.sh when a container cannot be removed" 1
has "failed removal is reported" "$OUT" "FAILED"
build_state
run "$DOWN" --verify
echo "NVIDIA GB10, 4.0, 40, 0" >"$STUB_STATE/node1/gpu"
touch "$STUB_STATE/node6/unreachable"
run "$DOWN" --verify
rc_is "--verify with an unreachable node" 1
has "unreachable node is named" "$OUT" "UNREACHABLE"
build_state
run "$UP"
touch "$STUB_STATE/node6/unreachable"
run "$DOWN" --force
rc_is "down.sh --force with an unreachable node" 1
has_all "unreachable node is reported as such, not as an orphaned GPU" "$OUT" "FAILED" "UNREACHABLE" "cannot be reached"
lacks "unreachable node is not blamed on an orphaned context" "$OUT" "orphaned"

# =====================================================================================================================
echo "== the machine running the launcher is used directly, without ssh"
build_state
me=$(hostname -s 2>/dev/null || hostname)
if [[ $me =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  sed -E "s/^NODE1=.*/NODE1=$me/" "$GOOD_CLUSTER" >"$CLUSTER"
  OUT=$(env PATH="$BIN:$PATH" CLUSTER_ENV="$CLUSTER" RECIPE_ENV="$RECIPE" STUB_STATE="$STUB_STATE" STUB_LOG="$STUB_LOG" \
    STUB_NODE=node1 PROBE_ROOT="$STUB_STATE/node1/root" "$UP" --dry-run 2>&1)
  RC=$?
  rc_is "dry run with NODE1 = this machine" 0
  has "the local node is marked" "$OUT" "# [dry-run] $me (this machine): bash -s"
  eq "seven ssh calls, none to this machine" 7 "$(wc -l <"$STUB_LOG/ssh.log" | tr -d ' ')"
  eq "no ssh call to this machine" 0 "$(grep -c "^$me " "$STUB_LOG/ssh.log" || true)"
else
  echo "   (hostname '$me' is not a valid node name; skipped)"
fi
# NODE1 is an alias that is not this machine's hostname, but its fabric address is configured here
build_state
sed -E "s/^NODE1=.*/NODE1=head-alias/" "$GOOD_CLUSTER" >"$CLUSTER"
OUT=$(env PATH="$BIN:$PATH" CLUSTER_ENV="$CLUSTER" RECIPE_ENV="$RECIPE" STUB_STATE="$STUB_STATE" STUB_LOG="$STUB_LOG" \
  STUB_NODE=node1 STUB_LOCAL_ADDR="$(ip_of 1)" PROBE_ROOT="$STUB_STATE/node1/root" "$UP" --dry-run 2>&1)
RC=$?
rc_is "dry run with NODE1 an alias for this machine" 0
has "a node is recognised by its fabric address" "$OUT" "# [dry-run] head-alias (this machine): bash -s"
eq "no ssh call to the alias" 0 "$(grep -c '^head-alias ' "$STUB_LOG/ssh.log" || true)"
OUT=$(env PATH="$BIN:$PATH" CLUSTER_ENV="$CLUSTER" RECIPE_ENV="$RECIPE" STUB_STATE="$STUB_STATE" STUB_LOG="$STUB_LOG" \
  STUB_NODE=node1 PROBE_ROOT="$STUB_STATE/node1/root" "$UP" --dry-run 2>&1)
has "without a matching local address the alias goes over ssh" "$OUT" "# [dry-run] head-alias: ssh "

# =====================================================================================================================
echo
if [ "$NFAIL" -eq 0 ]; then
  echo "launcher_dryrun: $PASS passed, 0 failed"
  exit 0
fi
echo "launcher_dryrun: $PASS passed, $NFAIL FAILED"
exit 1
