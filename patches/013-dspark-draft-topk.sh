#!/usr/bin/env bash
# 013-dspark-draft-topk.sh -- DSpark draft top-k for DeepSeek-V4.1 (see files/p013_dspark_draft_topk.py for why and what).
# Active only when the speculative config sets "dspark_draft_topk"; fail-closed, idempotent (files/xpatch.py).
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
echo "=== Patch 013: DSpark draft top-k for DeepSeek-V4.1 ==="
python3 "$D/xpatch.py" apply "$D/p013_dspark_draft_topk.py"
