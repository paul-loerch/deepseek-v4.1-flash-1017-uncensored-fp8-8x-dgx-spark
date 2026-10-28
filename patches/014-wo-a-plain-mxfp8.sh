#!/usr/bin/env bash
# 014-wo-a-plain-mxfp8.sh -- DeepSeek-V4.1 wo_a as a plain MXFP8 GEMM when a rank holds one output group (TP=8 on SM12x);
# see files/p014_wo_a_plain_mxfp8.py for why and what. Fail-closed, idempotent (files/xpatch.py).
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
echo "=== Patch 014: wo_a as a plain MXFP8 GEMM (one local group, no DeepGEMM BMM) ==="
python3 "$D/xpatch.py" apply "$D/p014_wo_a_plain_mxfp8.py"
