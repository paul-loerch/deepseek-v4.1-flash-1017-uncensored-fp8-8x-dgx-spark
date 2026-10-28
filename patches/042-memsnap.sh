#!/usr/bin/env bash
# 042-memsnap.sh -- on-demand caching-allocator snapshot in patch 032's memstats thread: when the trigger file exists
# (VLLM_GB10_MEMSNAP_TRIGGER, default /cache/huggingface/dsv41-memsnap.trigger) it is deleted and the allocator's segments are
# logged grouped by (stream, pool). Logging only; needs VLLM_GB10_MEMSTATS=<s>. Edits vllm/_memstats.py (installed by 032, so this
# runs after it). Applies the anchored edits files/p042_memsnap.py. Fail-closed, idempotent.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
echo "=== Patch 042: allocator snapshot on demand (trigger ${VLLM_GB10_MEMSNAP_TRIGGER:-/cache/huggingface/dsv41-memsnap.trigger}) ==="
for f in p042_memsnap.py xpatch.py; do
  [ -f "$D/$f" ] || { echo "[FAIL] [042] $D/$f not found" >&2; exit 1; }
done
python3 "$D/xpatch.py" apply "$D/p042_memsnap.py"
echo " [OK] [042] memsnap edit in place"
