#!/usr/bin/env bash
# recipes/<id>/patches/entrypoint.sh — container entrypoint for M_BOOT_ENTRYPOINT=recipe.
# Applies every numbered patch (NNN-*.sh) in order — each idempotent and fail-closed — then
# execs `vllm serve` with the recipe's arguments. Mounted read-only at /opt/recipe-patches.
set -euo pipefail
say() { printf '[recipe-entrypoint] %s\n' "$*" >&2; }
for p in /opt/recipe-patches/[0-9][0-9][0-9]-*.sh; do
  [ -e "$p" ] || continue
  say "applying $(basename "$p")"
  bash "$p" || { say "FATAL: $(basename "$p") failed"; exit 1; }
done
say "all patches applied — starting vLLM"
exec vllm serve "$@"
