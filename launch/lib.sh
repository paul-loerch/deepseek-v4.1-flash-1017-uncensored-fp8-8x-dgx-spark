# shellcheck shell=bash disable=SC2034
# launch/lib.sh -- helpers shared by up.sh, down.sh and status.sh (sourced, never executed).
#
#   load_config      reads and validates cluster.env and config/recipe.env, and sets NODE_NAME[], NODE_IP[],
#                    IMAGE, VLLM_ARGS[], DOCKER_EXTRA[], PATCH_DIR and the other variables the scripts use
#   run_on           runs a bash script on a node over ssh (directly, for the node this machine is)
#   probe_all        read-only facts about every node, gathered in parallel
#   docker_run_cmd   the docker run command of one rank, used for the real start and for --dry-run
#
# Files:
#   cluster.env        $CLUSTER_ENV, default <repo>/cluster.env         your nodes, addresses, paths, ports
#   config/recipe.env  $RECIPE_ENV,  default <repo>/config/recipe.env   what is served: M_IMAGE, M_VLLM_ARGS,
#                      M_DOCKER_EXTRA, M_BOOT_ENTRYPOINT, M_PATCH_DIR
#
# recipe.env may contain these placeholders; they are filled from cluster.env before the file is read:
#   @MASTER_ADDR@ (NODE1_FABRIC_IP)  @MASTER_PORT@  @API_PORT@  @MODEL_DIR@  @HF_CACHE_DIR@  @ENGRAM_PACKED_DIR@
#   @GPU_MEMORY_UTILIZATION@  @REPO_DIR@ (the directory that holds launch/ and config/)
# Whatever recipe.env says, these vLLM arguments are forced to the cluster.env values: the model path,
# --port, --master-addr, --master-port, --nnodes, --tensor-parallel-size and --gpu-memory-utilization.
# --node-rank and --headless are added per rank and must not appear in recipe.env.
#
# Needs bash 3.2 or newer, ssh, awk, python3 and curl on the machine that runs the launcher; docker,
# nvidia-smi, ip, awk and python3 on the nodes.

export LC_ALL=C   # one decimal point and one sort order, whatever the locale of the machine
LAUNCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(dirname "$LAUNCH_DIR")
NNODES=8

# Timing (seconds). The defaults suit a real cluster; the offline test shortens them through the environment.
UP_POLL_S=${UP_POLL_S:-15}               # between container checks right after the ranks were started
UP_WITHIN_S=${UP_WITHIN_S:-90}           # every rank container must be running this soon after the start
READY_POLL_S=${READY_POLL_S:-20}         # between reads of rank 0's log while waiting for readiness
SETTLE_S=${SETTLE_S:-30}                 # pause after "Application startup complete", before the final checks
STALL_WARN_MIN=${STALL_WARN_MIN:-15}     # warn when rank 0's log has been silent this long during boot
VERIFY_WITHIN_S=${VERIFY_WITHIN_S:-120}  # down.sh: how long the nodes get to go idle after the removal
VERIFY_POLL_S=${VERIFY_POLL_S:-10}

# ---- output ---------------------------------------------------------------------------------------------
if [ -t 1 ]; then
  C_RED=$'\033[31m' C_GRN=$'\033[32m' C_YEL=$'\033[33m' C_OFF=$'\033[0m'
else
  C_RED='' C_GRN='' C_YEL='' C_OFF=''
fi
say()  { printf '%s %s\n' "$(date +%T)" "$*"; }
ok()   { printf '%s %sok%s    %s\n' "$(date +%T)" "$C_GRN" "$C_OFF" "$*"; }
warn() { printf '%s %swarn%s  %s\n' "$(date +%T)" "$C_YEL" "$C_OFF" "$*" >&2; }
bad()  { printf '%s %sFAIL%s  %s\n' "$(date +%T)" "$C_RED" "$C_OFF" "$*" >&2; }
die()  { bad "$1"; exit "${2:-1}"; }   # die <message> [exit code]; 1 = a gate failed, 2 = usage or configuration

# ---- small helpers --------------------------------------------------------------------------------------
fge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 >= b + 0) }'; }   # fge A B: A >= B
flt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 < b + 0) }'; }    # flt A B: A <  B
is_num() { case ${1:-} in '' | . | *[!0-9.]* | *.*.*) return 1 ;; esac; return 0; }   # non-negative decimal
trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}; printf '%s' "$s"; }
gib() { awk -v b="${1:-0}" 'BEGIN { printf "%.1f", b / 1073741824 }'; }

# q <string>: shell-quote for display and for ssh command lines (plain words stay unquoted)
q() {
  case $1 in
    '') printf "''" ;;
    *[!A-Za-z0-9_@%+=:,./-]*) printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")" ;;
    *) printf '%s' "$1" ;;
  esac
}

# _timeout <seconds> <command...>: timeout(1) when the machine has it (macOS has not)
_timeout() {
  if command -v timeout >/dev/null 2>&1; then timeout "$@"
  elif command -v gtimeout >/dev/null 2>&1; then gtimeout "$@"
  else shift; "$@"; fi
}

# ---- configuration --------------------------------------------------------------------------------------
RE_HOST='^[A-Za-z0-9][A-Za-z0-9._-]*$'
RE_IFACE='^[A-Za-z0-9][A-Za-z0-9_.:-]*$'
RE_IPV4='^[0-9]{1,3}(\.[0-9]{1,3}){3}$'
RE_CIDR='^[0-9]{1,3}(\.[0-9]{1,3}){3}/[0-9]{1,2}$'
RE_PATH='^/[A-Za-z0-9_./+@%=,~-]+$'      # absolute; no spaces, no colons (docker -v syntax)
RE_NAME='^[A-Za-z][A-Za-z0-9_.-]*$'
RE_PORT='^[0-9]{1,5}$'
RE_INT='^[0-9]+$'

CFG_ERRORS=()
cfg_err() { CFG_ERRORS[${#CFG_ERRORS[@]}]=$1; }
cfg_fail_if_errors() {  # cfg_fail_if_errors <file>: print every recorded problem and stop with exit 2
  local e
  if [ "${#CFG_ERRORS[@]}" -gt 0 ]; then
    bad "$1 has ${#CFG_ERRORS[@]} problem(s):"
    for e in "${CFG_ERRORS[@]}"; do printf '          - %s\n' "$e" >&2; done
    exit 2
  fi
}

is_placeholder() { case $1 in *CHANGE_ME* | *'<'*'>'*) return 0 ;; esac; return 1; }
# 192.0.2.0/24 is the range cluster.env.example uses; ALLOW_EXAMPLE_ADDRESSES=1 lets a network that really uses it through
is_example_addr() {
  [ "${ALLOW_EXAMPLE_ADDRESSES:-0}" = 1 ] && return 1
  case $1 in 192.0.2.*) return 0 ;; esac
  return 1
}
valid_ipv4() {
  local ip=$1 o IFS=.
  [[ $ip =~ $RE_IPV4 ]] || return 1
  # shellcheck disable=SC2086
  for o in $ip; do [ "$((10#$o))" -le 255 ] || return 1; done
  return 0
}
valid_cidr() { [[ $1 =~ $RE_CIDR ]] && valid_ipv4 "${1%/*}" && [ "${1#*/}" -le 32 ]; }
ip_to_int() {  # ip_to_int <a.b.c.d>: the address as one number
  local IFS=.
  # shellcheck disable=SC2086
  set -- $1
  echo $(((10#$1 << 24) + (10#$2 << 16) + (10#$3 << 8) + 10#$4))
}
cidr_contains() {  # cidr_contains <a.b.c.d/n> <ip>
  local len=${1#*/} mask
  mask=$(((0xFFFFFFFF << (32 - 10#$len)) & 0xFFFFFFFF))
  [ $(($(ip_to_int "${1%/*}") & mask)) -eq $(($(ip_to_int "$2") & mask)) ]
}

# check_var <VAR> <ERE> <hint>: record an error unless $VAR is set, is not a placeholder and matches the pattern
check_var() {
  local name=$1 re=$2 hint=$3 val
  val=${!name:-}
  if [ -z "$val" ]; then
    cfg_err "$name is not set ($hint)"
  elif is_placeholder "$val"; then
    cfg_err "$name still has the example value '$val' ($hint)"
  elif ! [[ $val =~ $re ]]; then
    cfg_err "$name='$val' is not valid ($hint)"
  fi
  return 0
}

# render_file <in> <out>: copy <in> to <out>, replacing the @NAME@ placeholders with cluster.env values
render_file() {
  R_MASTER_ADDR=${NODE_IP[0]} R_MASTER_PORT=$MASTER_PORT R_API_PORT=$API_PORT R_MODEL_DIR=$MODEL_DIR \
    R_HF_CACHE_DIR=$HF_CACHE_DIR R_ENGRAM_PACKED_DIR=$ENGRAM_PACKED_DIR R_GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION \
    R_REPO_DIR=$REPO_DIR \
    awk '
      BEGIN { n = split("MASTER_ADDR MASTER_PORT API_PORT MODEL_DIR HF_CACHE_DIR ENGRAM_PACKED_DIR GPU_MEMORY_UTILIZATION REPO_DIR", names, " ") }
      {
        line = $0
        for (i = 1; i <= n; i++) {
          key = "@" names[i] "@"; val = ENVIRON["R_" names[i]]; out = ""
          while ((p = index(line, key)) > 0) { out = out substr(line, 1, p - 1) val; line = substr(line, p + length(key)) }
          line = out line
        }
        print line
      }' "$1" >"$2"
}

# vllm_set <--flag> <value>: replace the value of --flag in VLLM_ARGS, or append "--flag value"
vllm_set() {
  local flag=$1 val=$2 i n=${#VLLM_ARGS[@]}
  for ((i = 0; i < n; i++)); do
    case ${VLLM_ARGS[$i]} in
      "$flag") VLLM_ARGS[$((i + 1))]=$val; return 0 ;;
      "$flag="*) VLLM_ARGS[$i]="$flag=$val"; return 0 ;;
    esac
  done
  VLLM_ARGS[$n]=$flag
  VLLM_ARGS[$((n + 1))]=$val
}
vllm_set_model() {  # the model path: --model if the recipe uses that form, else the first positional argument
  local i
  for ((i = 0; i < ${#VLLM_ARGS[@]}; i++)); do
    case ${VLLM_ARGS[$i]} in
      --model) vllm_set --model "$1"; return 0 ;;
      --model=*) VLLM_ARGS[$i]="--model=$1"; return 0 ;;
    esac
  done
  case ${VLLM_ARGS[0]:-} in
    '' | -*) VLLM_ARGS=("$1" ${VLLM_ARGS[@]+"${VLLM_ARGS[@]}"}) ;;
    *) VLLM_ARGS[0]=$1 ;;
  esac
}

load_config() {
  local i name ipvar ipval j left a
  CLUSTER_FILE=${CLUSTER_ENV:-$REPO_DIR/cluster.env}
  RECIPE_FILE=${RECIPE_ENV:-$REPO_DIR/config/recipe.env}
  if [ ! -r "$CLUSTER_FILE" ]; then
    die "cluster description not found: $CLUSTER_FILE
        Copy cluster.env.example to cluster.env and fill in your nodes; the comments explain every value." 2
  fi
  [ -r "$RECIPE_FILE" ] || die "recipe configuration not found: $RECIPE_FILE" 2
  bash -n "$CLUSTER_FILE" 2>/dev/null || die "$CLUSTER_FILE has a shell syntax error (check: bash -n $CLUSTER_FILE)" 2
  # A command in the file that fails is reported by bash itself; the checks below then see the missing value.
  # shellcheck disable=SC1090
  . "$CLUSTER_FILE" || true

  API_PORT=${API_PORT:-8000}
  MASTER_PORT=${MASTER_PORT:-25100}
  CONTAINER_PREFIX=${CONTAINER_PREFIX:-dsv41-rank}
  SSH_USER=${SSH_USER:-}
  SSH_OPTS=${SSH_OPTS-"-o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=3"}
  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}
  DROP_CACHES=${DROP_CACHES:-0}
  MEM_AVAILABLE_MIN_GIB=${MEM_AVAILABLE_MIN_GIB:-40}
  MEM_SERVING_WARN_GIB=${MEM_SERVING_WARN_GIB:-8}
  CLEAN_SHM=${CLEAN_SHM:-1}
  ENGRAM_PACKED_DIR=${ENGRAM_PACKED_DIR:-}
  FABRIC_MTU=${FABRIC_MTU:-9000}
  IDLE_GPU_POWER_MAX_W=${IDLE_GPU_POWER_MAX_W:-10}
  NCCL_DEBUG_LEVEL=${NCCL_DEBUG_LEVEL:-WARN}

  # -- cluster.env ----------------------------------------------------------------------------------------
  CFG_ERRORS=()
  NODE_NAME=()
  NODE_IP=()
  for ((i = 1; i <= NNODES; i++)); do
    name=NODE$i
    ipvar=NODE${i}_FABRIC_IP
    check_var "$name" "$RE_HOST" "the ssh name of node $i"
    check_var "$ipvar" "$RE_IPV4" "the fabric address of node $i; see: ip -br a"
    NODE_NAME[$((i - 1))]=${!name:-}
    NODE_IP[$((i - 1))]=${!ipvar:-}
    ipval=${!ipvar:-}
    if [[ $ipval =~ $RE_IPV4 ]]; then
      if ! valid_ipv4 "$ipval"; then
        cfg_err "$ipvar=$ipval is not a valid address (a part is above 255)"
      elif is_example_addr "$ipval"; then
        cfg_err "$ipvar=$ipval is the example address from cluster.env.example; set the node's real fabric address"
      fi
    fi
  done
  for ((i = 0; i < NNODES; i++)); do
    for ((j = i + 1; j < NNODES; j++)); do
      if [ -n "${NODE_NAME[$i]}" ] && [ "${NODE_NAME[$i]}" = "${NODE_NAME[$j]}" ]; then
        cfg_err "NODE$((i + 1)) and NODE$((j + 1)) are both '${NODE_NAME[$i]}'; every rank needs its own node"
      fi
      if [ -n "${NODE_IP[$i]}" ] && [ "${NODE_IP[$i]}" = "${NODE_IP[$j]}" ]; then
        cfg_err "NODE$((i + 1))_FABRIC_IP and NODE$((j + 1))_FABRIC_IP are both ${NODE_IP[$i]}"
      fi
    done
  done
  check_var FABRIC_IFACE "$RE_IFACE" "the fabric interface name; see: ip -br a"
  check_var ROCE_HCA "$RE_IFACE" "the RoCE device of that port; see: ibdev2netdev or rdma link"
  check_var FABRIC_CIDR "$RE_CIDR" "the subnet of the fabric addresses, like 203.0.113.0/24"
  if [[ ${FABRIC_CIDR:-} =~ $RE_CIDR ]]; then
    if ! valid_cidr "$FABRIC_CIDR"; then
      cfg_err "FABRIC_CIDR=$FABRIC_CIDR is not a valid subnet"
    else
      if is_example_addr "$FABRIC_CIDR"; then
        cfg_err "FABRIC_CIDR=$FABRIC_CIDR is the example subnet from cluster.env.example; set your fabric subnet"
      fi
      for ((i = 0; i < NNODES; i++)); do
        if valid_ipv4 "${NODE_IP[$i]}" && ! cidr_contains "$FABRIC_CIDR" "${NODE_IP[$i]}"; then
          cfg_err "NODE$((i + 1))_FABRIC_IP=${NODE_IP[$i]} is outside FABRIC_CIDR=$FABRIC_CIDR"
        fi
      done
    fi
  fi
  check_var MODEL_DIR "$RE_PATH" "an absolute path without spaces or colons, the same on every node"
  check_var HF_CACHE_DIR "$RE_PATH" "an absolute path without spaces or colons"
  if grep -q '@ENGRAM_PACKED_DIR@' "$RECIPE_FILE" 2>/dev/null; then
    check_var ENGRAM_PACKED_DIR "$RE_PATH" "the directory with the packed Engram rows on every node, see tools/engram_pack.py"
  fi
  check_var API_PORT "$RE_PORT" "a TCP port"
  check_var MASTER_PORT "$RE_PORT" "a TCP port"
  if [[ $API_PORT =~ $RE_PORT && $MASTER_PORT =~ $RE_PORT ]]; then
    if [ "$API_PORT" -lt 1 ] || [ "$API_PORT" -gt 65535 ] || [ "$MASTER_PORT" -lt 1 ] || [ "$MASTER_PORT" -gt 65535 ]; then
      cfg_err "API_PORT and MASTER_PORT must be between 1 and 65535"
    elif [ "$API_PORT" = "$MASTER_PORT" ]; then
      cfg_err "API_PORT and MASTER_PORT are both $API_PORT; they must differ"
    fi
  fi
  check_var CONTAINER_PREFIX "$RE_NAME" "letters, digits, _ . - ; the first character a letter"
  if [ -n "$SSH_USER" ]; then check_var SSH_USER '^[A-Za-z_][A-Za-z0-9_.-]*$' "a login name, or empty"; fi
  if ! is_num "$GPU_MEMORY_UTILIZATION" || ! awk -v g="$GPU_MEMORY_UTILIZATION" 'BEGIN { exit !(g + 0 > 0 && g + 0 <= 1) }'; then
    cfg_err "GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' must be a number above 0 and at most 1"
  fi
  case $DROP_CACHES in 0 | 1) ;; *) cfg_err "DROP_CACHES must be 0 or 1 (got '$DROP_CACHES')" ;; esac
  case $CLEAN_SHM in 0 | 1) ;; *) cfg_err "CLEAN_SHM must be 0 or 1 (got '$CLEAN_SHM')" ;; esac
  is_num "$MEM_AVAILABLE_MIN_GIB" || cfg_err "MEM_AVAILABLE_MIN_GIB='$MEM_AVAILABLE_MIN_GIB' must be a number"
  is_num "$MEM_SERVING_WARN_GIB" || cfg_err "MEM_SERVING_WARN_GIB='$MEM_SERVING_WARN_GIB' must be a number"
  is_num "$IDLE_GPU_POWER_MAX_W" || cfg_err "IDLE_GPU_POWER_MAX_W='$IDLE_GPU_POWER_MAX_W' must be a number"
  [[ $FABRIC_MTU =~ $RE_INT ]] || cfg_err "FABRIC_MTU='$FABRIC_MTU' must be a whole number"
  cfg_fail_if_errors "$CLUSTER_FILE"

  # -- config/recipe.env ----------------------------------------------------------------------------------
  RENDERED=$(mktemp "${TMPDIR:-/tmp}/dsv41-recipe.XXXXXX") || die "cannot create a temporary file" 2
  render_file "$RECIPE_FILE" "$RENDERED"
  left=$(grep -oE '@[A-Z][A-Z0-9_]*@' "$RENDERED" | sort -u | tr '\n' ' ' || true)
  if [ -n "$left" ]; then
    rm -f "$RENDERED"
    cfg_err "unknown placeholder(s) $left; known: @MASTER_ADDR@ @MASTER_PORT@ @API_PORT@ @MODEL_DIR@ @HF_CACHE_DIR@ @ENGRAM_PACKED_DIR@ @GPU_MEMORY_UTILIZATION@ @REPO_DIR@"
    cfg_fail_if_errors "$RECIPE_FILE"
  fi
  if ! bash -n "$RENDERED" 2>/dev/null; then
    rm -f "$RENDERED"
    die "$RECIPE_FILE has a shell syntax error (check: bash -n $RECIPE_FILE)" 2
  fi
  unset M_IMAGE M_VLLM_ARGS M_DOCKER_EXTRA M_BOOT_ENTRYPOINT M_PATCH_DIR
  # shellcheck disable=SC1090
  . "$RENDERED" || true
  rm -f "$RENDERED"

  IMAGE=${M_IMAGE:-}
  [ -n "$IMAGE" ] || cfg_err "M_IMAGE is not set (the docker image to run)"
  VLLM_ARGS=()
  DOCKER_EXTRA=()
  if [ -z "${M_VLLM_ARGS:-}" ]; then
    cfg_err "M_VLLM_ARGS is not set (the arguments of vllm serve)"
  elif ! eval "VLLM_ARGS=( $M_VLLM_ARGS )" 2>/dev/null; then
    VLLM_ARGS=()
    cfg_err "M_VLLM_ARGS cannot be split into words (unbalanced quotes?)"
  fi
  if [ -n "${M_DOCKER_EXTRA:-}" ] && ! eval "DOCKER_EXTRA=( $M_DOCKER_EXTRA )" 2>/dev/null; then
    DOCKER_EXTRA=()
    cfg_err "M_DOCKER_EXTRA cannot be split into words (unbalanced quotes?)"
  fi
  if [ "${#VLLM_ARGS[@]}" -gt 0 ]; then
    for a in "${VLLM_ARGS[@]}"; do
      case $a in
        --node-rank | --node-rank=* | --headless) cfg_err "M_VLLM_ARGS contains $a; the launcher sets the rank itself" ;;
      esac
    done
    vllm_set_model "$MODEL_DIR"
    vllm_set --port "$API_PORT"
    vllm_set --master-addr "${NODE_IP[0]}"
    vllm_set --master-port "$MASTER_PORT"
    vllm_set --nnodes "$NNODES"
    vllm_set --tensor-parallel-size "$NNODES"
    vllm_set --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  fi
  SERVED_NAME=$MODEL_DIR
  for ((i = 0; i < ${#VLLM_ARGS[@]}; i++)); do
    case ${VLLM_ARGS[$i]} in
      --served-model-name) SERVED_NAME=${VLLM_ARGS[$((i + 1))]:-$MODEL_DIR}; break ;;
      --served-model-name=*) SERVED_NAME=${VLLM_ARGS[$i]#*=}; break ;;
    esac
  done

  PATCH_DIR=''
  case ${M_BOOT_ENTRYPOINT:-} in
    '') ;;
    recipe)
      PATCH_DIR=${M_PATCH_DIR:-}
      if [ -z "$PATCH_DIR" ]; then
        cfg_err "M_BOOT_ENTRYPOINT=recipe needs M_PATCH_DIR (the directory with entrypoint.sh)"
      else
        case $PATCH_DIR in /*) ;; *) PATCH_DIR=$REPO_DIR/$PATCH_DIR ;; esac
        if [ ! -f "$PATCH_DIR/entrypoint.sh" ]; then
          cfg_err "M_PATCH_DIR=$PATCH_DIR has no entrypoint.sh"
        else
          PATCH_DIR=$(cd "$PATCH_DIR" && pwd -P)
          [[ $PATCH_DIR =~ $RE_PATH ]] || cfg_err "the patch directory $PATCH_DIR must not contain spaces or colons"
        fi
      fi
      ;;
    *) cfg_err "M_BOOT_ENTRYPOINT must be 'recipe' or empty (got '${M_BOOT_ENTRYPOINT}')" ;;
  esac
  cfg_fail_if_errors "$RECIPE_FILE"

  # -- derived --------------------------------------------------------------------------------------------
  SSH_OPT_ARR=()
  if [ -n "$SSH_OPTS" ]; then read -r -a SSH_OPT_ARR <<<"$SSH_OPTS"; fi
  LOCAL_NAMES=" localhost 127.0.0.1 ::1 $(hostname 2>/dev/null || true) $(hostname -s 2>/dev/null || true) "
  LOCAL_ADDRS=" $(ip -o -4 addr show 2>/dev/null | awk '{ sub("/.*", "", $4); printf "%s ", $4 }') "
  PROBE="$TREE_HASH_FN"$'\n'"$PROBE_BODY"
  eval "$TREE_HASH_FN"
}

rank_container() { printf '%s%s' "$CONTAINER_PREFIX" "$1"; }   # rank_container <rank>

# ---- running things on nodes ----------------------------------------------------------------------------
# is_local_node <node>: true for the machine the launcher runs on: its name is this machine's hostname, or the
# node's fabric address is configured here. Such a node is run directly instead of over ssh.
is_local_node() {
  local i
  case $LOCAL_NAMES in *" $1 "*) return 0 ;; esac
  for ((i = 0; i < NNODES; i++)); do
    if [ "${NODE_NAME[$i]}" = "$1" ]; then
      case $LOCAL_ADDRS in *" ${NODE_IP[$i]} "*) return 0 ;; esac
    fi
  done
  return 1
}
ssh_target() { printf '%s%s' "${SSH_USER:+$SSH_USER@}" "$1"; }

# ssh_cmdline <node>: the command line run_on uses for <node>, for --dry-run output
ssh_cmdline() {
  if is_local_node "$1"; then
    printf 'bash -s'
  else
    printf 'ssh %s%s bash -s' "${SSH_OPTS:+$SSH_OPTS }" "$(ssh_target "$1")"
  fi
}

# run_on <node> <script> [arg...]: run a bash script on <node> with the given positional arguments. The script
# travels on stdin, so no quoting problem can reach the remote login shell. Bounded at 90 s.
run_on() {
  local node=$1 script=$2 a args=''
  shift 2
  for a in "$@"; do args="$args $(q "$a")"; done
  if is_local_node "$node"; then
    _timeout 90 bash -s -- "$@" <<<"$script"
  else
    _timeout 90 ssh ${SSH_OPT_ARR[@]+"${SSH_OPT_ARR[@]}"} "$(ssh_target "$node")" "bash -s --$args" <<<"$script"
  fi
}

# ---- node probe: read-only facts, one key=value per line --------------------------------------------------
# tree_hash <dir>: 16 hex digits over the names and contents of every file below <dir>
read -r -d '' TREE_HASH_FN <<'EOS' || true
tree_hash() {
  python3 - "$1" <<'PY'
import hashlib, os, sys
root = sys.argv[1]
h = hashlib.sha256()
for d, dirs, files in os.walk(root):
    dirs[:] = sorted(x for x in dirs if x != "__pycache__")
    for f in sorted(files):
        if f == ".DS_Store" or f.endswith(".pyc"):
            continue
        p = os.path.join(d, f)
        h.update(os.path.relpath(p, root).encode() + b"\0")
        with open(p, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
print(h.hexdigest()[:16])
PY
}
EOS

# Arguments: LEVEL IMAGE MODEL_DIR IFACE HCA PREFIX NAME API_PORT MASTER_PORT PATCH_DIR ENGRAM_DIR RANK
#   LEVEL is "full" (everything) or "quick" (containers, GPU and memory only).
#   ENGRAM_DIR (may be empty) is the directory with the packed Engram rows; RANK is this node's rank.
# PROBE_ROOT is a prefix for the filesystem paths read below; it is unset on real nodes (the offline test uses it).
read -r -d '' PROBE_BODY <<'EOS' || true
set -u
export LC_ALL=C
LEVEL=$1 IMAGE=$2 MODEL_DIR=$3 IFACE=$4 HCA=$5 PREFIX=$6 NAME=$7 API_PORT=$8 MASTER_PORT=$9 PATCH_DIR=${10:-} ENGRAM_DIR=${11:-} RANK=${12:-0}
ROOT=${PROBE_ROOT:-}
kv() { printf '%s=%s\n' "$1" "$2"; }
trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; s=${s%"${s##*[![:space:]]}"}; printf '%s' "$s"; }
kv reachable 1
kv host "$(hostname 2>/dev/null || echo unknown)"
if command -v python3 >/dev/null 2>&1; then kv python_ok 1; else kv python_ok 0; fi

# docker and the rank containers
if names=$(docker ps -a --format '{{.Names}}' 2>&1); then
  kv docker_ok 1
  kv rank_containers "$(printf '%s\n' "$names" | awk -v p="$PREFIX" 'index($0, p) == 1 && substr($0, length(p) + 1) ~ /^[0-9]+$/ { printf "%s ", $0 }')"
  kv container_state "$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo absent)"
  kv other_running "$(docker ps --format '{{.Names}}' 2>/dev/null | awk -v p="$PREFIX" 'NF && !(index($0, p) == 1 && substr($0, length(p) + 1) ~ /^[0-9]+$/) { printf "%s ", $0 }')"
else
  kv docker_ok 0
  kv docker_err "$(printf '%s' "$names" | head -c 200 | tr '\n' ' ')"
fi

# GPU (GB10 reports no memory figures; power, temperature and utilization are what matters)
if out=$(nvidia-smi --query-gpu=name,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>&1); then
  IFS=, read -r g_name g_pw g_tc g_ut <<<"$(printf '%s\n' "$out" | head -1)"
  kv gpu_ok 1
  kv gpu_name "$(trim "$g_name")"
  kv gpu_power_w "$(trim "$g_pw")"
  kv gpu_temp_c "$(trim "$g_tc")"
  kv gpu_util "$(trim "$g_ut")"
else
  kv gpu_ok 0
  kv gpu_err "$(printf '%s' "$out" | head -c 200 | tr '\n' ' ')"
fi

# memory
awk '/^MemAvailable:/ { a = $2 } /^MemFree:/ { f = $2 } /^MemTotal:/ { t = $2 }
     END { printf "mem_avail_gib=%.1f\nmem_free_gib=%.1f\nmem_total_gib=%.1f\n", a / 1048576, f / 1048576, t / 1048576 }' \
  "$ROOT/proc/meminfo" 2>/dev/null || true

if [ "$LEVEL" = full ] && command -v python3 >/dev/null 2>&1; then
  kv image_id "$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || echo missing)"

  # model directory: number of files and their total size; download bookkeeping under a .cache directory is ignored
  python3 - "$ROOT$MODEL_DIR" <<'PY'
import os, sys
root = sys.argv[1]
if not os.path.isdir(root):
    print("model_files=-1\nmodel_bytes=-1")
else:
    n = b = 0
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != ".cache"]
        for f in files:
            try:
                b += os.path.getsize(os.path.join(d, f))
                n += 1
            except OSError:
                pass
    print("model_files=%d\nmodel_bytes=%d" % (n, b))
PY

  # fabric interface and RoCE device
  link=$(ip -o link show dev "$IFACE" 2>/dev/null | head -1)
  if [ -z "$link" ]; then
    kv iface_state missing
    kv iface_up 0
    kv iface_mtu 0
  else
    kv iface_state "$(printf '%s\n' "$link" | awk '{ for (i = 1; i < NF; i++) if ($i == "state") { print $(i + 1); exit } }')"
    flags=$(printf '%s\n' "$link" | awk '{ print $3 }')
    flags=${flags#<}
    flags=${flags%>}
    case ",$flags," in
      *,UP,*) case ",$flags," in *,LOWER_UP,*) kv iface_up 1 ;; *) kv iface_up 0 ;; esac ;;
      *) kv iface_up 0 ;;
    esac
    kv iface_mtu "$(printf '%s\n' "$link" | awk '{ for (i = 1; i < NF; i++) if ($i == "mtu") { print $(i + 1); exit } }')"
  fi
  kv iface_addrs "$(ip -o -4 addr show dev "$IFACE" 2>/dev/null | awk '{ sub("/.*", "", $4); printf "%s ", $4 }')"
  if [ -e "$ROOT/sys/class/infiniband/$HCA" ]; then
    kv hca_ok 1
  else
    kv hca_ok 0
    kv hca_list "$(ls "$ROOT/sys/class/infiniband" 2>/dev/null | tr '\n' ' ')"
  fi

  # listening ports (only rank 0's node cares)
  kv ports_busy "$( (ss -ltn 2>/dev/null || true) | awk -v a=":$API_PORT" -v m=":$MASTER_PORT" \
    'NR > 1 { l = $4; if (substr(l, length(l) - length(a) + 1) == a || substr(l, length(l) - length(m) + 1) == m) printf "%s ", l }')"

  if [ -n "$PATCH_DIR" ]; then
    if [ -d "$ROOT$PATCH_DIR" ]; then kv patch_hash "$(tree_hash "$ROOT$PATCH_DIR")"; else kv patch_hash missing; fi
  fi

  # packed Engram rows (tools/engram_pack.py): one manifest per Engram layer served from NVMe for THIS node's rank
  if [ -n "$ENGRAM_DIR" ]; then
    if [ -d "$ROOT$ENGRAM_DIR" ]; then
      kv engram_packs "$(find "$ROOT$ENGRAM_DIR" -maxdepth 1 -name "engram_L*_tp8_r${RANK}.pack.json" 2>/dev/null | wc -l | tr -d ' ')"
    else
      kv engram_packs missing
    fi
  fi
fi
EOS

# probe_all <dir> [full|quick]: run the probe on every node in parallel -> <dir>/<rank>.kv and <rank>.err
probe_all() {
  local dir=$1 level=${2:-full} r
  mkdir -p "$dir"
  for ((r = 0; r < NNODES; r++)); do
    run_on "${NODE_NAME[$r]}" "$PROBE" "$level" "$IMAGE" "$MODEL_DIR" "$FABRIC_IFACE" "$ROCE_HCA" \
      "$CONTAINER_PREFIX" "$(rank_container "$r")" "$API_PORT" "$MASTER_PORT" "$PATCH_DIR" "$ENGRAM_PACKED_DIR" "$r" \
      >"$dir/$r.kv" 2>"$dir/$r.err" &
  done
  wait || true
}
# pget <dir> <rank> <key>: one value from a probe result ("" when absent)
pget() { awk -v k="$3" 'index($0, k "=") == 1 { print substr($0, length(k) + 2); exit }' "$1/$2.kv" 2>/dev/null || true; }

# idle_reason <dir> <rank>: "ok" when the node has no rank container and an idle GPU, otherwise what is wrong.
# A power reading the driver does not provide (not a number) is ignored; utilization decides then.
idle_reason() {
  local d=$1 r=$2 v
  if [ "$(pget "$d" "$r" reachable)" != 1 ]; then echo UNREACHABLE; return 0; fi
  if [ "$(pget "$d" "$r" docker_ok)" != 1 ]; then echo "docker not usable"; return 0; fi
  if [ -n "$(pget "$d" "$r" rank_containers)" ]; then echo "rank container present"; return 0; fi
  if [ "$(pget "$d" "$r" gpu_ok)" != 1 ]; then echo "nvidia-smi failed"; return 0; fi
  v=$(pget "$d" "$r" gpu_util)
  if ! is_num "$v" || fge "$v" 1; then echo "GPU busy (util ${v:-?} %)"; return 0; fi
  v=$(pget "$d" "$r" gpu_power_w)
  if is_num "$v" && fge "$v" "$IDLE_GPU_POWER_MAX_W"; then echo "GPU busy ($v W)"; return 0; fi
  echo ok
}

# ---- the docker run command of one rank ------------------------------------------------------------------
DR_LINES=()
dr() { DR_LINES[${#DR_LINES[@]}]=$1; }                      # dr <line>: add one option line
dr_env() { dr "-e $(q "$1=$2")"; }                           # dr_env NAME VALUE
# dr_group <ERE that starts a new line> <word...>: add the words, one line per option with its values
dr_group() {
  local start=$1 cur='' tok
  shift
  for tok in "$@"; do
    if [[ $tok =~ $start ]]; then
      if [ -n "$cur" ]; then dr "$cur"; fi
      cur=$(q "$tok")
    else
      cur="${cur:+$cur }$(q "$tok")"
    fi
  done
  if [ -n "$cur" ]; then dr "$cur"; fi
}

# docker_run_cmd <rank>: print the docker run command of that rank (no trailing newline), one option per line.
# Used for the real start and for --dry-run, so what is printed is what runs.
#   - the flags and environment below are what multi-node NCCL over RoCE needs on GB10; see cluster.env.example
#   - NCCL_IB_HCA and NCCL_SOCKET_IFNAME start with "==": the second "=" makes NCCL match the name exactly
#     instead of as a prefix
#   - rank 0 serves the API; ranks 1..7 get --headless and join rank 0 at --master-addr:--master-port
docker_run_cmd() {
  local r=$1 i n
  DR_LINES=()
  dr "docker run -d --name $(q "$(rank_container "$r")")"
  dr "--gpus all"
  dr "--network host"
  dr "--ipc=host"
  dr "--ulimit memlock=-1"
  dr "--ulimit stack=67108864"
  dr "--ulimit nofile=524288:524288"
  dr "--shm-size 32g"
  dr "--device=/dev/infiniband"
  dr "-v $(q "$MODEL_DIR:$MODEL_DIR")"
  dr "-v $(q "$HF_CACHE_DIR:/cache/huggingface")"
  dr_env HF_HOME /cache/huggingface
  dr_env HF_HUB_OFFLINE 1
  dr_env NCCL_NET IB
  dr_env NCCL_IB_HCA "=$ROCE_HCA"
  dr_env NCCL_SOCKET_IFNAME "=$FABRIC_IFACE"
  dr_env GLOO_SOCKET_IFNAME "$FABRIC_IFACE"
  dr_env TP_SOCKET_IFNAME "$FABRIC_IFACE"
  dr_env NCCL_IB_ADDR_RANGE "$FABRIC_CIDR"
  dr_env NCCL_NVLS_ENABLE 0
  dr_env NCCL_CUMEM_ENABLE 0
  dr_env NCCL_IGNORE_CPU_AFFINITY 1
  dr_env NCCL_DEBUG "$NCCL_DEBUG_LEVEL"
  dr_env VLLM_ENGINE_READY_TIMEOUT_S 3600
  dr_env UCX_MEM_MMAP_HOOK_MODE none
  dr_env UCX_RCACHE_MAX_UNRELEASED 1024
  dr_env OMP_NUM_THREADS 8
  if [ "${#DOCKER_EXTRA[@]}" -gt 0 ]; then dr_group '^-' "${DOCKER_EXTRA[@]}"; fi
  dr_env VLLM_HOST_IP "${NODE_IP[$r]}"
  if [ -n "$PATCH_DIR" ]; then
    dr "-v $(q "$PATCH_DIR:/opt/recipe-patches:ro")"
    dr "--entrypoint bash"
  fi
  dr "$(q "$IMAGE")"
  if [ -n "$PATCH_DIR" ]; then dr "/opt/recipe-patches/entrypoint.sh"; fi
  dr_group '^--' "${VLLM_ARGS[@]}"
  dr_group '^--' --node-rank "$r"
  if [ "$r" -gt 0 ]; then dr "--headless"; fi
  n=${#DR_LINES[@]}
  for ((i = 0; i < n; i++)); do
    if [ "$i" -eq 0 ]; then printf '%s' "${DR_LINES[$i]}"; else printf ' \\\n    %s' "${DR_LINES[$i]}"; fi
  done
}

# rank_script <rank>: the script that starts one rank on its node
rank_script() {
  printf 'set -e\nmkdir -p %s\n' "$(q "$HF_CACHE_DIR")"
  docker_run_cmd "$1"
  # shellcheck disable=SC2016
  printf ' >/dev/null\necho "rank %s started on $(hostname)"\n' "$1"
}

# show_cmd <node> <script> [arg...]: print what a dry run would execute on <node>
show_cmd() {
  local node=$1 script=$2 a args=''
  shift 2
  for a in "$@"; do args="$args $(q "$a")"; done
  printf '\n# [dry-run] %s%s: %s%s\n' "$node" "$(is_local_node "$node" && echo ' (this machine)')" "$(ssh_cmdline "$node")" "${args:+ --$args}"
  printf '%s\n' "$script" | sed 's/^/    /'
}

# ---- the API on rank 0 -------------------------------------------------------------------------------------
# api_get <path>: GET http://127.0.0.1:<API_PORT><path> on rank 0's node; prints the body (nothing on failure)
# shellcheck disable=SC2016
api_get() { run_on "${NODE_NAME[0]}" 'curl -s -m 10 "$1"' "http://127.0.0.1:$API_PORT$1" 2>/dev/null || true; }
# api_code <path>: the HTTP status code, 000 when nothing answers
api_code() {
  local c
  # shellcheck disable=SC2016
  c=$(run_on "${NODE_NAME[0]}" 'curl -s -m 5 -o /dev/null -w "%{http_code}\n" "$1"' "http://127.0.0.1:$API_PORT$1" 2>/dev/null | tail -n 1 || true)
  printf '%s' "${c:-000}"
}
# metric_sum <series> <file>: sum of a vllm gauge over its label sets, "unknown" when the series is absent
metric_sum() {
  awk -v re="^vllm:$1[{ ]" '$0 ~ re { s += $NF; seen = 1 } END { if (seen) printf "%d", s; else print "unknown" }' "$2"
}
# models_summary: one "<id> <max_model_len>" line per served model, from /v1/models on stdin
models_summary() {
  python3 -c '
import json, sys
try:
    for m in json.load(sys.stdin).get("data", []):
        print(m.get("id", "?"), m.get("max_model_len", "?"))
except Exception:
    pass
'
}
