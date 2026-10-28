#!/usr/bin/env bash
# 016-engram-nvme.sh -- Engram tables on node-local NVMe for DeepSeek-V4.1; see files/p016_engram_nvme.py for why and what.
# Applies the anchored edits (files/xpatch.py), installs the new modules into the vllm package and builds the native reader
# (gcc, raw syscalls only: no libaio, no io_uring). Fail-closed, idempotent. Inert unless the engram config sets "nvme": true.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/files"
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
DEST="$VLLM_ROOT/models/deepseek_v41/common"
fail() { echo "[016] FATAL: $*" >&2; exit 1; }

echo "=== Patch 016: Engram tables on node-local NVMe ==="
[ -d "$DEST" ] || fail "$DEST not found (is VLLM_ROOT=$VLLM_ROOT a vLLM with DeepSeek-V4.1?)"
python3 "$D/xpatch.py" apply "$D/p016_engram_nvme.py" "$VLLM_ROOT"

for f in engram_nvme.py engram_nvme_io.py engram_nvme.c; do
  [ -f "$D/engram_nvme/$f" ] || fail "missing $D/engram_nvme/$f"
  if cmp -s "$D/engram_nvme/$f" "$DEST/$f"; then
    echo " [skip] $f already installed"
  else
    install -m 0644 "$D/engram_nvme/$f" "$DEST/$f"
    echo " [OK] installed $f"
  fi
done

command -v gcc >/dev/null 2>&1 || fail "gcc not found (needed to build libengram_nvme.so)"
src_sha="$(sha256sum "$DEST/engram_nvme.c" | cut -d' ' -f1)"
if [ -f "$DEST/libengram_nvme.so" ] && [ "$(cat "$DEST/libengram_nvme.so.src" 2>/dev/null || true)" = "$src_sha" ]; then
  echo " [skip] libengram_nvme.so is up to date"
else
  rm -f "$DEST/libengram_nvme.so.tmp"
  gcc -O2 -shared -fPIC -pthread -Wall -o "$DEST/libengram_nvme.so.tmp" "$DEST/engram_nvme.c" || fail "building libengram_nvme.so failed"
  mv -f "$DEST/libengram_nvme.so.tmp" "$DEST/libengram_nvme.so"
  echo "$src_sha" > "$DEST/libengram_nvme.so.src"
  echo " [OK] built libengram_nvme.so"
fi

# the Python binding and the library must agree on every struct; report whether native AIO works under this container's seccomp
python3 -I - "$DEST" <<'PY' || fail "the reader library does not load or does not match its Python binding"
import importlib.util, os, sys
dest = sys.argv[1]
spec = importlib.util.spec_from_file_location("engram_nvme_io", os.path.join(dest, "engram_nvme_io.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["engram_nvme_io"] = mod
spec.loader.exec_module(mod)
lib = mod.load_library(os.path.join(dest, "libengram_nvme.so"))
aio = bool(lib.en_probe_aio())
print(f" [OK] [016] libengram_nvme.so loads (ABI {mod.ABI_VERSION}); native AIO {'available' if aio else 'NOT available (the reader will use its pread thread pool)'}")
PY
