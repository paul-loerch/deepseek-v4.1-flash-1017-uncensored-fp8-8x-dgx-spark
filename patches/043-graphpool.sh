#!/usr/bin/env bash
# 043-graphpool.sh -- shrink the CUDA-graph pool of the DeepSeek-V4.1 decode graphs (two independent opt-in levers; both off = stock behaviour):
#   VLLM_GB10_INDEXER_DECODE_CHUNK_MB=<cap MiB>  flattened decode rows of the sparse indexer in chunks (logits + candidate temporaries <= cap per chunk);
#                                                installs vllm/_indexer_decode_chunk.py + one import and one guarded early return in
#                                                model_executor/layers/sparse_attn_indexer.py
#   VLLM_GB10_SHARED_CAPTURE_STREAM=1            one capture stream per device in the module-level graph_capture(device) of distributed/parallel_state.py,
#                                                so the DSpark drafter's graphs reuse the main graphs' free pool blocks
#   VLLM_GB10_INDEXER_DECODE_CHUNK_SHARE_META=1  (default) chunk schedules computed once per forward context and shared by the layers of a KV-cache group
#   VLLM_GB10_INDEXER_DECODE_CHUNK_SCOPE=all|cand  (default all) cand = safe subset: stock logits + stock top-k, only the candidate kernels chunked (-0.4 GiB)
# Applies the anchored edits files/p043_graphpool.py through files/xpatch.py: sparse_attn_indexer.py is pinned to the image file (no earlier patch
# edits it), parallel_state.py to the file as patch 018 leaves it, so this runs after 018. Fail-closed (dry-run check of all edits first, setting
# values validated, helper module self-tested), idempotent. Why and what: files/p043_graphpool.py, files/indexer_decode_chunk.py.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
fail() { echo "[FAIL] [043] $*" >&2; exit 1; }
echo "=== Patch 043: graph-pool shrink (VLLM_GB10_INDEXER_DECODE_CHUNK_MB=${VLLM_GB10_INDEXER_DECODE_CHUNK_MB:-unset}, VLLM_GB10_SHARED_CAPTURE_STREAM=${VLLM_GB10_SHARED_CAPTURE_STREAM:-unset}) ==="
for f in indexer_decode_chunk.py p043_graphpool.py xpatch.py; do
  [ -f "$D/$f" ] || fail "$D/$f not found"
done

# settings: unset/empty = off; anything else must parse (the runtime would raise at import otherwise)
cap="${VLLM_GB10_INDEXER_DECODE_CHUNK_MB:-}"
case "$cap" in
  ''|0|off|false|no) ;;
  *[!0-9]*) fail "VLLM_GB10_INDEXER_DECODE_CHUNK_MB='$cap' must be unset/0 (off) or a whole number of MiB (1 .. 65536)" ;;
  *) { [ "${#cap}" -le 6 ] && [ "$((10#$cap))" -ge 1 ] && [ "$((10#$cap))" -le 65536 ]; } || fail "VLLM_GB10_INDEXER_DECODE_CHUNK_MB=$cap outside 1 .. 65536" ;;
esac
case "$(printf '%s' "${VLLM_GB10_INDEXER_DECODE_CHUNK_SCOPE:-}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')" in ''|all|cand) ;; *) fail "VLLM_GB10_INDEXER_DECODE_CHUNK_SCOPE='${VLLM_GB10_INDEXER_DECODE_CHUNK_SCOPE}' must be unset, all or cand" ;; esac
for v in VLLM_GB10_SHARED_CAPTURE_STREAM VLLM_GB10_INDEXER_DECODE_CHUNK_SHARE_META; do
  val="$(printf '%s' "${!v:-}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
  case "$val" in ''|0|1|true|false|yes|no|on|off) ;; *) fail "$v='${!v}' must be unset, 0 or 1" ;; esac
done

python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$D/indexer_decode_chunk.py"
# dry run of every edit before anything is written: a different file revision is refused here
python3 "$D/xpatch.py" check "$D/p043_graphpool.py" "$VLLM_ROOT"
install -m 0644 "$D/indexer_decode_chunk.py" "$VLLM_ROOT/_indexer_decode_chunk.py"
echo " [OK] [043] installed _indexer_decode_chunk.py"
python3 "$D/xpatch.py" apply "$D/p043_graphpool.py" "$VLLM_ROOT"

# self-test of the installed helper (pure Python): imports with the live settings, plans the 768-row case, leaves small graphs alone
python3 - "$VLLM_ROOT/_indexer_decode_chunk.py" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("p043_chunk_selftest", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
mib = 1 << 20
c = m.plan_chunks(768, 4 * mib + mib // 2, 512 * mib)          # ratio 1 + candidate scores, cap 512 MiB
assert c and c[0][0] == 0 and c[-1][1] == 768 and all(b > a for a, b in c) and all(c[i][1] == c[i + 1][0] for i in range(len(c) - 1)), c
assert all(b - a >= m.MIN_CHUNK_ROWS and (b - a) * (4 * mib + mib // 2) <= 512 * mib for a, b in c), c
assert m.plan_chunks(96, 4 * mib, 512 * mib) is None             # 384 MiB fit the cap: stock
assert m.plan_chunks(768, 4 * mib, 0) is None                    # no cap: stock
print(f" [OK] [043] helper self-test: cap {m.CAP_MB} MiB, 768 rows -> {len(c)} chunks of {min(b - a for a, b in c)}..{max(b - a for a, b in c)} rows (at 512 MiB), share_meta={m.SHARE_META}")
PY
echo " [OK] [043] graph-pool edits in place (chunking ${cap:-off}, shared capture stream ${VLLM_GB10_SHARED_CAPTURE_STREAM:-off})"
