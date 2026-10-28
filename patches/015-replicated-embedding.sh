#!/usr/bin/env bash
# 015-replicated-embedding.sh -- DeepSeek-V4.1 token embedding replicated per rank (no all-reduce at each graph's start);
# see files/p015_replicated_embedding.py for why and what. Fail-closed, idempotent (files/xpatch.py).
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
echo "=== Patch 015: replicated token embedding ==="
python3 "$D/xpatch.py" apply "$D/p015_replicated_embedding.py"
