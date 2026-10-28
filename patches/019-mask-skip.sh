#!/usr/bin/env bash
# 019-mask-skip.sh -- skip the indexer candidate-mask tiles past each row's end (exact; see files/p019_mask_skip.py).
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
echo "=== Patch 019: candidate-mask tiles past the row end ==="
python3 "$D/xpatch.py" apply "$D/p019_mask_skip.py"
