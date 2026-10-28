#!/usr/bin/env bash
# 030-engram-packed.sh -- packed Engram rows for patch 016's NVMe reader: ONE 512-byte O_DIRECT read per unique row instead of two
# (opt-in: ENGRAM_NVME_PACKED_DIR=<dir with the engram_L*_tp*_r*.pack[.json] files of tools/engram_pack.py>; unset -> the reader does what
# patch 016's does, two reads per row, same protocol; lossless: the delivered bytes are identical either way).
#
# Needs patch 016 first. Replaces two files that 016 installed in models/deepseek_v41/common/ -- engram_nvme.c (+ rebuilds
# libengram_nvme.so, ABI 2) and engram_nvme_io.py -- by their patch-030 versions (files/engram_nvme/*.p030.*; 016's own files are not
# edited). Nothing in vLLM itself is edited and engram_nvme.py (the GPU side) stays EXACTLY as 016 installed it: patch 028 pins its
# sha256. Fail-closed: a file that is neither 016's nor already 030's is refused, as is a changed engram_nvme.py. Idempotent. Run it
# AFTER 016 (the recipe's numeric order does that; 016's installer would put its own files back if it ran later).
#
#   ENGRAM_NVME_PACKED_DIR=<dir>        where the packs are (path as seen inside the container); unset: packed rows off
#   ENGRAM_NVME_PACKED=auto|require|off auto (default when the dir is set): use a valid pack, otherwise fall back to the two-read layout
#                                       and log why; require: a missing/stale/invalid pack ends engine start-up; off: ignore the dir
#   ENGRAM_NVME_PACKED_SPOT=256         rows of each pack compared with the shard at engine start (0 = none)
#   ENGRAM_NVME_VERIFY_AFTER=1          (opt-in, independent of packing) nvme_verify_pct's re-reads of the original shard run after done_seq is
#                                       published instead of before: they cost ~80-100 ms per 8,192-token prefill request at 1 %, ~0 off the critical
#                                       path. A mismatch is still fatal (process exit 86), but the GPU may already have used that request's rows.
# Why and what: files/engram_nvme/engram_nvme.p030.c, engram_nvme_io.p030.py; format and tool: engram-nvme/tools/engram_pack.py.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
DEST="$VLLM_ROOT/models/deepseek_v41/common"
fail() { echo "[030] FATAL: $*" >&2; exit 1; }
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
# where a patch-030 file lives: files/engram_nvme/ (as shipped) or files/ (flat copy, e.g. through tools/mkpatchdir.sh)
src_of() { local d; for d in "$D/engram_nvme" "$D"; do [ -f "$d/$1" ] && { echo "$d/$1"; return 0; }; done; return 1; }

# sha256 of the files patch 016 installs: the bases the patch-030 versions are derived from, and the one that must not change
BASE_C=0c6cc94aaae49ce42af73dce8a1afe0c40e9cce2507099a1239bd66a09907333    # engram_nvme.c      (016)
BASE_IO=70bb526bc2346fae5951b7b1c4456a7d75b04c57bd140dd352d0cc932b2cad0a   # engram_nvme_io.py  (016)
KEEP_GPU=af31cdf72f392f76221fbd812e68217023011aa1d25ea63d0b0bfc2ce6f84153  # engram_nvme.py     (016; patch 028 pins it too)

mode_raw="$(printf '%s' "${ENGRAM_NVME_PACKED:-}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$mode_raw" in ''|auto|require|off) ;; *) fail "ENGRAM_NVME_PACKED='${ENGRAM_NVME_PACKED}' must be unset, auto, require or off" ;; esac
va_raw="$(printf '%s' "${ENGRAM_NVME_VERIFY_AFTER:-}" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$va_raw" in ''|0|1|on|off|true|false|yes|no) ;; *) fail "ENGRAM_NVME_VERIFY_AFTER='${ENGRAM_NVME_VERIFY_AFTER}' must be unset, 0 or 1" ;; esac
spot="${ENGRAM_NVME_PACKED_SPOT:-256}"
[[ "$spot" =~ ^[0-9]{1,6}$ ]] || fail "ENGRAM_NVME_PACKED_SPOT='$spot' must be an integer 0..999999"
dir="${ENGRAM_NVME_PACKED_DIR:-}"
echo "=== Patch 030: packed Engram rows (ENGRAM_NVME_PACKED_DIR=${dir:-unset}, ENGRAM_NVME_PACKED=${mode_raw:-auto}, spot ${spot}, verify-after ${va_raw:-off}) ==="
if [ "$mode_raw" = require ] && [ -z "$dir" ]; then fail "ENGRAM_NVME_PACKED=require needs ENGRAM_NVME_PACKED_DIR"; fi
if [ -n "$dir" ] && [ "$mode_raw" != off ]; then
  n_man="$(find "$dir" -maxdepth 1 -name '*.pack.json' 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$n_man" = 0 ]; then
    [ "$mode_raw" != require ] || fail "ENGRAM_NVME_PACKED=require but $dir holds no *.pack.json (path as seen inside the container)"
    echo " [warn] [030] $dir holds no *.pack.json: engines start with patch 016's two-read layout (run tools/engram_pack.py, mount the directory?)"
  else
    echo " [OK] [030] $dir: $n_man manifest(s); each engine rank picks the one whose layer and row slice match its own and validates it at start"
  fi
fi

[ -d "$DEST" ] || fail "$DEST not found (is VLLM_ROOT=$VLLM_ROOT a vLLM with DeepSeek-V4.1?)"
command -v gcc >/dev/null 2>&1 || fail "gcc not found (needed to build libengram_nvme.so)"
# patch 016 must be in place, and its GPU-side module untouched
[ -f "$DEST/engram_nvme.py" ] && [ -f "$DEST/engram_nvme_io.py" ] && [ -f "$DEST/engram_nvme.c" ] || fail "patch 016 is not applied (engram_nvme.{py,c} / engram_nvme_io.py missing in $DEST): run 016-engram-nvme.sh first"
grep -qF '# [patch-016 engram-nvme]' "$DEST/engram.py" || fail "$DEST/engram.py carries no patch-016 marker: run 016-engram-nvme.sh first"
got="$(sha "$DEST/engram_nvme.py")"
[ "$got" = "$KEEP_GPU" ] || fail "engram_nvme.py sha256 $got != the one patch 016 installs ($KEEP_GPU); patch 030 was written against that file (and patch 028 pins it): refusing"

# the two replacement files: install the patch-030 version over patch 016's (pinned) or leave an already installed one
for pair in "engram_nvme.c:engram_nvme.p030.c:$BASE_C" "engram_nvme_io.py:engram_nvme_io.p030.py:$BASE_IO"; do
  IFS=: read -r name p030 base <<<"$pair"
  src="$(src_of "$p030")" || fail "missing $p030 (looked in $D/engram_nvme and $D)"
  if cmp -s "$src" "$DEST/$name"; then
    echo " [skip] $name is already the patch-030 version"
    continue
  fi
  got="$(sha "$DEST/$name")"
  [ "$got" = "$base" ] || fail "$DEST/$name sha256 $got is neither patch 016's ($base) nor the patch-030 version: refusing to overwrite it"
done
python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$(src_of engram_nvme_io.p030.py)" || fail "engram_nvme_io.p030.py does not parse"
for pair in "engram_nvme.c:engram_nvme.p030.c" "engram_nvme_io.py:engram_nvme_io.p030.py"; do
  IFS=: read -r name p030 <<<"$pair"
  src="$(src_of "$p030")"
  if cmp -s "$src" "$DEST/$name"; then continue; fi
  install -m 0644 "$src" "$DEST/$name.tmp030" && mv -f "$DEST/$name.tmp030" "$DEST/$name"
  echo " [OK] installed the patch-030 $name"
done

src_sha="$(sha "$DEST/engram_nvme.c")"
if [ -f "$DEST/libengram_nvme.so" ] && [ "$(cat "$DEST/libengram_nvme.so.src" 2>/dev/null || true)" = "$src_sha" ]; then
  echo " [skip] libengram_nvme.so is up to date"
else
  rm -f "$DEST/libengram_nvme.so.tmp"
  gcc -O2 -shared -fPIC -pthread -Wall -o "$DEST/libengram_nvme.so.tmp" "$DEST/engram_nvme.c" || fail "building libengram_nvme.so failed"
  mv -f "$DEST/libengram_nvme.so.tmp" "$DEST/libengram_nvme.so"
  echo "$src_sha" > "$DEST/libengram_nvme.so.src"
  echo " [OK] built libengram_nvme.so"
fi

# the Python binding and the library must agree on every struct (ABI 2); report whether native AIO works under this container's seccomp
python3 -I - "$DEST" <<'PY' || fail "the reader library does not load or does not match its Python binding"
import importlib.util, os, sys
dest = sys.argv[1]
spec = importlib.util.spec_from_file_location("engram_nvme_io", os.path.join(dest, "engram_nvme_io.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["engram_nvme_io"] = mod
spec.loader.exec_module(mod)
assert mod.ABI_VERSION == 2 and hasattr(mod, "select_packed"), "engram_nvme_io.py is not the patch-030 version"
lib = mod.load_library(os.path.join(dest, "libengram_nvme.so"))
aio = bool(lib.en_probe_aio())
print(f" [OK] [030] libengram_nvme.so loads (ABI {mod.ABI_VERSION}, packed-row reader, pack layout {mod.PACK_LAYOUT_VERSION}); native AIO {'available' if aio else 'NOT available (the reader will use its pread thread pool)'}")
PY
