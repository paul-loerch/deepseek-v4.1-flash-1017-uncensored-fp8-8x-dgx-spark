#!/usr/bin/env bash
# launch/status.sh -- show the state of the deployment. Read-only.
#
#   launch/status.sh
#
# Per node: the rank container's state, GPU power, temperature and utilization, and MemAvailable.
# Then, from rank 0's API: /health, the served model(s) with max_model_len, and the number of running and
# waiting requests (vllm:num_requests_running and vllm:num_requests_waiting from /metrics).
# Needs cluster.env and config/recipe.env (see launch/lib.sh).
# Exit status: 0 when all eight containers run and /health answers 200, 1 otherwise, 2 on a usage or configuration error.
set -euo pipefail
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

case ${1:-} in
  '') ;;
  -h | --help)
    printf 'usage: launch/status.sh\n  Shows containers, GPUs, memory and the API state. Read-only.\n'
    exit 0
    ;;
  *) die "unknown argument: $1 (usage: launch/status.sh)" 2 ;;
esac

load_config
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

probe_all "$TMP/n" quick
healthy=1
printf '  %-14s %-4s %-14s %-12s %8s %7s %7s %11s\n' node rank container state 'GPU W' 'GPU C' 'util %' 'avail GiB'
for ((r = 0; r < NNODES; r++)); do
  if [ "$(pget "$TMP/n" "$r" reachable)" != 1 ]; then
    st=UNREACHABLE
  elif [ "$(pget "$TMP/n" "$r" docker_ok)" != 1 ]; then
    st="docker unusable"
  else
    st=$(pget "$TMP/n" "$r" container_state)
  fi
  if [ "$st" != running ]; then healthy=0; fi
  printf '  %-14s %-4s %-14s %-12s %8s %7s %7s %11s\n' "${NODE_NAME[$r]}" "$r" "$(rank_container "$r")" "$st" \
    "$(pget "$TMP/n" "$r" gpu_power_w)" "$(pget "$TMP/n" "$r" gpu_temp_c)" "$(pget "$TMP/n" "$r" gpu_util)" \
    "$(pget "$TMP/n" "$r" mem_avail_gib)"
done

printf '\nAPI on %s, port %s\n' "${NODE_NAME[0]}" "$API_PORT"
code=$(api_code /health)
if [ "$code" != 200 ]; then healthy=0; fi
printf '  %-12s %s\n' /health "$code"
models=$(api_get /v1/models | models_summary)
if [ -n "$models" ]; then
  while IFS=' ' read -r id len; do printf '  %-12s %s (max_model_len %s)\n' /v1/models "$id" "$len"; done <<<"$models"
else
  printf '  %-12s %s\n' /v1/models "no answer"
fi
api_get /metrics >"$TMP/metrics"
printf '  %-12s running %s, waiting %s\n' requests "$(metric_sum num_requests_running "$TMP/metrics")" "$(metric_sum num_requests_waiting "$TMP/metrics")"

if [ "$healthy" = 1 ]; then ok "all $NNODES ranks are running and the API is healthy"; exit 0; fi
warn "not healthy: see the lines above"
exit 1
