#!/usr/bin/env bash
# 045-embed-tp.sh -- make patch 015's replicated token embedding switchable: VLLM_GB10_EMBED_TP=1 shards the table again
# (vocab-parallel + one all-reduce, -1.16 GiB per rank); unset = 015's behaviour. Edits models/deepseek_v41/nvidia/model.py as left
# by patches 013 .. 042 (028 is the last edit of that file), so this runs after them. Applies files/p045_embed_tp.py. Fail-closed,
# idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
echo "=== Patch 045: switchable token-embedding sharding (VLLM_GB10_EMBED_TP=${VLLM_GB10_EMBED_TP:-unset}) ==="
for f in p045_embed_tp.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [045] $D/$f not found" >&2; exit 1; }
done
python3 "$D/xpatch.py" apply "$D/p045_embed_tp.py"
echo " [OK] [045] embed-tp edit in place"
