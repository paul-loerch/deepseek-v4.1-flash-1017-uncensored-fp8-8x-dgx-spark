#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""engram_pack.py -- repack one rank's slice of the DeepSeek-V4.1 Engram tables into 512-byte records (recipe patch 030).

WHY. The checkpoint keeps an Engram row in two tensors: 256 B of fp8 values (`layers.L.engram.embed.weight`) and 8 B of e8m0
scales (`layers.L.engram.embed.scale`), 98 GB apart, at unaligned offsets (data base 664 / 672). The NVMe reader of patch 016
therefore needs TWO O_DIRECT reads per row (a 512 or 1024 B window for the values, a 512 B window for the scales). This tool
writes a per-node copy of ONE rank's slice in which a row is a single 512 B sector:

    record i of the slice (global row id g = row_lo + i) at file offset  data_off + i * 512 :
        [ 0 .. 256)  the 256 fp8 value bytes, verbatim            (source: values_off + g * 256)
        [256 .. 264) the 8 e8m0 scale bytes, verbatim             (source: scales_off + g * 8)
        [264 .. 512) zero padding (248 B)

so patch 030's reader issues one 512 B read per unique row. `data_off` is 4096: the first 4 KiB of the file is a header.

FILES (all in --out-dir, names are fixed so the reader can find them from (layer, tp, rank) alone):
    engram_L<layer>_tp<tp>_r<rank>.pack           header (4 KiB) + data
    engram_L<layer>_tp<tp>_r<rank>.pack.json      manifest, written LAST and atomically: no manifest = not a valid pack
    engram_L<layer>_tp<tp>_r<rank>.pack.journal   progress journal while packing (removed on success)

HEADER (little endian, 4096 B, offset 0; the reader cross-checks it against the manifest):
    0   8  magic  b"ENGRPACK"          8  4  layout_version (1)     12 4  header_bytes (4096)   16 4  data_off (4096)
    20  4  row_stride (512)            24 4  value_bytes (256)      28 4  scale_bytes (8)       32 4  layer
    36  4  rank                        40 4  tp                     44 4  flags (bit 0 = complete)
    48  8  row_lo                      56 8  row_hi                 64 8  num_rows (whole table)
    72  8  values_off (source, absolute file offset of row 0)       80 8  scales_off (same)
    88  8  source_size                 96 8  source_mtime_ns (signed)
    104 32 sha256 of the source shard's header bytes (8 length bytes + JSON header)
    136 32 sha256 of the data region (all zero until complete)      168 8  data_bytes (= rows * 512)
    176 .. 4064 zero               4064 32 sha256 of bytes [0, 4064)

MANIFEST (JSON): the same facts plus: source path (informational only: identity is size + mtime + header sha256), the file's
size/mtime after packing, the layer/rank/head range, `prefix_sha256` (the running sha256 of the data region after every
`chunk_rows` rows: the last entry is `data_sha256`), what the end-of-run verification checked, tool, host, time.

ROW SLICES. Which rows are rank r's is NOT hard-coded: the 24 hash heads of a layer (3 n-gram orders x 8 heads) own consecutive
prime-sized bucket ranges (vLLM's EngramLayout: the primes are drawn in order over the layers and never reused, starting at
`engram_vocab_size`), rank r owns the heads [r * 24/tp, (r+1) * 24/tp) = ONE contiguous row range, in both tensors. The primes
are re-derived here with the same algorithm and must add up to the shard's row count (checked), the slice is checked again by
the reader against the live model at engine start, and `plan` prints the table of slices and file offsets.

USAGE (stdlib only, Python >= 3.9):
    engram_pack.py plan   --model-dir M [--tp 8] [--layers 1,14]
    engram_pack.py pack   --model-dir M --out-dir D --rank R [--tp 8] [--layers 1,14] [--chunk-rows 262144] [--io auto|direct|buffered]
                          [--max-mbps N] [--verify-rows 65536] [--force]
    engram_pack.py verify --model-dir M --out-dir D --rank R [--tp 8] [--layers 1,14] [--rows 65536] [--full-hash] [--full-compare]

`pack` is resumable (re-run the same command after an interruption: the finished prefix is re-hashed against the journal, the rest
is redone), idempotent (a pack whose manifest still matches the source is left alone, `--force` redoes it), and never grows the
page cache: the source is read and the pack written with O_DIRECT when the filesystem allows it (else buffered I/O followed by
fdatasync + POSIX_FADV_DONTNEED per chunk). It preallocates the output (fails early on ENOSPC), reads the source strictly
sequentially in `chunk_rows` blocks (64 MiB of values + 2 MiB of scales per 262,144 rows) and holds ~200 MiB of buffers.
`--max-mbps` caps the write rate so a pack can run next to a serving engine. At the end a random sample (plus the first/last
row of the slice and of every chunk) is compared with the ORIGINAL shard through independent buffered reads; the manifest is
written only if everything matched. Exit status 0 = ok, 1 = failure/mismatch, 2 = usage.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import mmap
import os
import random
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

TOOL = "engram_pack.py/1"
FORMAT = "engram-packed-rows"
LAYOUT_VERSION = 1
MAGIC = b"ENGRPACK"
HEADER_BYTES = 4096
DATA_OFF = 4096
STRIDE = 512
ROW_V = 256
ROW_S = 8
USED = ROW_V + ROW_S
PAD = STRIDE - USED
ALIGN = 4096
FLAG_COMPLETE = 1
DEFAULT_CHUNK_ROWS = 1 << 18
ZERO_PAD = bytes(PAD)

_HDR = struct.Struct("<8s10I6Qq32s32sQ")
assert _HDR.size == 176


class PackError(Exception):
    """A condition that makes the pack (or the check) fail; the message is shown to the user."""


def log(msg: str) -> None:
    print(f"[engram_pack {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# row slices (vLLM EngramLayout, models/deepseek_v41/common/engram.py)
# ---------------------------------------------------------------------------------------------------------------------
def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32 (vLLM's _is_prime, verbatim)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set) -> int:
    """The smallest prime above `start` that has not been handed out yet (vLLM's find_next_prime, verbatim)."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def engram_head_sizes(layer_ids, vocab_size: int, n_heads: int, max_ngram_size: int) -> dict:
    """{layer_id: [bucket count of hash column c]}; column c = n_heads * (order - 2) + head. EngramLayout.__init__ flattened."""
    seen: set = set()
    out = {}
    for layer in layer_ids:
        flat = []
        for _ in range(max_ngram_size - 1):
            current = vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                flat.append(current)
        out[int(layer)] = flat
    return out


def rank_slice(sizes, tp: int, rank: int):
    """(head_start, head_end, row_lo, row_hi) of TP rank `rank`: ParallelEngramEmbedding takes cdiv(n_hash_cols, tp) consecutive heads."""
    n = len(sizes)
    if tp < 1 or n % tp:
        raise PackError(f"{n} hash heads do not divide evenly over tp={tp} (patch 016/030 need that)")
    if not 0 <= rank < tp:
        raise PackError(f"rank {rank} is outside 0..{tp - 1}")
    part = n // tp
    h0, h1 = rank * part, (rank + 1) * part
    return h0, h1, sum(sizes[:h0]), sum(sizes[:h1])


# ---------------------------------------------------------------------------------------------------------------------
# the source checkpoint
# ---------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class EngramCfg:
    layer_ids: tuple
    num_embeddings: tuple
    max_ngram_size: int
    n_heads: int
    head_dim: int
    vocab_size: int


def load_engram_cfg(model_dir: str, **override) -> EngramCfg:
    """The engram_* keys of <model_dir>/config.json (top level or under text_config); `override` (non-None) wins."""
    cfg = {}
    path = os.path.join(model_dir, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        cfg = raw.get("text_config", raw) if isinstance(raw, dict) else {}
    pick = lambda key, name: override.get(name) if override.get(name) is not None else cfg.get(key)  # noqa: E731
    vals = dict(
        layer_ids=pick("engram_layer_ids", "layer_ids"),
        num_embeddings=pick("engram_num_embeddings", "num_embeddings"),
        max_ngram_size=pick("engram_max_ngram_size", "max_ngram_size"),
        n_heads=pick("engram_n_heads", "n_heads"),
        head_dim=pick("engram_head_dim", "head_dim"),
        vocab_size=pick("engram_vocab_size", "vocab_size"),
    )
    missing = [k for k, v in vals.items() if v is None]
    if missing:
        raise PackError(f"engram settings {missing} are neither in {path} nor given on the command line")
    ec = EngramCfg(tuple(int(i) for i in vals["layer_ids"]), tuple(int(i) for i in vals["num_embeddings"]),
                   int(vals["max_ngram_size"]), int(vals["n_heads"]), int(vals["head_dim"]), int(vals["vocab_size"]))
    if len(ec.layer_ids) != len(ec.num_embeddings):
        raise PackError("engram_layer_ids and engram_num_embeddings differ in length")
    if ec.head_dim != ROW_V:
        raise PackError(f"engram_head_dim={ec.head_dim}: the packed format holds {ROW_V}-byte value rows")
    return ec


@dataclasses.dataclass
class Table:
    layer: int
    path: str
    values_off: int
    scales_off: int
    num_rows: int
    file_size: int
    mtime_ns: int
    header_len: int
    header_sha256: str
    shard: str


def read_shard_header(path: str):
    """(header_len, header_bytes_including_the_8_length_bytes, parsed JSON header)."""
    with open(path, "rb") as f:
        raw8 = f.read(8)
        if len(raw8) != 8:
            raise PackError(f"{path}: too short for a safetensors header")
        (hlen,) = struct.unpack("<Q", raw8)
        if hlen > 64 * 1024 * 1024:
            raise PackError(f"{path}: implausible safetensors header length {hlen}")
        body = f.read(hlen)
    if len(body) != hlen:
        raise PackError(f"{path}: truncated safetensors header")
    return hlen, raw8 + body, json.loads(body)


def locate_table(model_dir: str, layer: int, num_rows: int) -> Table:
    """Find layers.<layer>.engram.embed.{weight,scale} and validate everything a raw-offset read depends on."""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx) as f:
        weight_map = json.load(f)["weight_map"]
    wname, sname = f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale"
    if wname not in weight_map or sname not in weight_map:
        raise PackError(f"{idx}: {wname} / {sname} not found")
    if weight_map[wname] != weight_map[sname]:
        raise PackError(f"layer {layer}: value and scale tensors are in different shards")
    path = os.path.join(model_dir, weight_map[wname])
    hlen, hbytes, header = read_shard_header(path)
    base = 8 + hlen
    w, s = header.get(wname), header.get(sname)
    if w is None or s is None:
        raise PackError(f"{path}: header lacks {wname} / {sname}")
    if w["dtype"] != "F8_E4M3" or s["dtype"] != "F8_E8M0":
        raise PackError(f"{path}: dtypes {w['dtype']} / {s['dtype']}, want F8_E4M3 / F8_E8M0")
    if list(w["shape"]) != [num_rows, ROW_V] or list(s["shape"]) != [num_rows, ROW_S]:
        raise PackError(f"{path}: shapes {w['shape']} / {s['shape']} != [{num_rows}, {ROW_V}] / [{num_rows}, {ROW_S}]")
    (w0, w1), (s0, s1) = w["data_offsets"], s["data_offsets"]
    if w1 - w0 != num_rows * ROW_V or s1 - s0 != num_rows * ROW_S:
        raise PackError(f"{path}: tensor byte lengths do not match the shapes")
    st = os.stat(path)
    if base + max(w1, s1) > st.st_size:
        raise PackError(f"{path}: truncated ({st.st_size} B, the tensors end at {base + max(w1, s1)})")
    return Table(layer, path, base + w0, base + s0, num_rows, st.st_size, st.st_mtime_ns, hlen,
                 hashlib.sha256(hbytes).hexdigest(), weight_map[wname])


def derive_slices(model_dir: str, tp: int, layers=None, **override):
    """For every requested layer: (Table, head sizes). Also checks that the re-derived primes add up to the shard's row count."""
    ec = load_engram_cfg(model_dir, **override)
    sizes = engram_head_sizes(ec.layer_ids, ec.vocab_size, ec.n_heads, ec.max_ngram_size)
    want = ec.layer_ids if not layers else tuple(layers)
    out = []
    for layer in want:
        if layer not in ec.layer_ids:
            raise PackError(f"layer {layer} is not an Engram layer of this model ({list(ec.layer_ids)})")
        num_rows = ec.num_embeddings[ec.layer_ids.index(layer)]
        if sum(sizes[layer]) != num_rows:
            raise PackError(
                f"layer {layer}: the re-derived hash-head sizes add up to {sum(sizes[layer])}, the model says {num_rows} rows "
                "(unknown layout: refusing to guess row slices)")
        out.append((locate_table(model_dir, layer, num_rows), sizes[layer]))
    return ec, out


# ---------------------------------------------------------------------------------------------------------------------
# file names, header, manifest
# ---------------------------------------------------------------------------------------------------------------------
def pack_names(layer: int, tp: int, rank: int):
    stem = f"engram_L{layer}_tp{tp}_r{rank}.pack"
    return stem, stem + ".json", stem + ".journal"


def encode_header(*, layer, rank, tp, flags, row_lo, row_hi, num_rows, values_off, scales_off, source_size, source_mtime_ns,
                  source_header_sha256: str, data_sha256: str = "", data_bytes) -> bytes:
    body = _HDR.pack(MAGIC, LAYOUT_VERSION, HEADER_BYTES, DATA_OFF, STRIDE, ROW_V, ROW_S, layer, rank, tp, flags,
                     row_lo, row_hi, num_rows, values_off, scales_off, source_size, source_mtime_ns,
                     bytes.fromhex(source_header_sha256), bytes.fromhex(data_sha256) if data_sha256 else bytes(32), data_bytes)
    body = body + bytes(HEADER_BYTES - 32 - len(body))
    return body + hashlib.sha256(body).digest()


def decode_header(raw: bytes) -> dict:
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"header block is {len(raw)} B, want {HEADER_BYTES}")
    raw = bytes(raw[:HEADER_BYTES])
    if raw[:8] != MAGIC:
        raise ValueError("bad magic (not an engram packed-rows file)")
    if hashlib.sha256(raw[:HEADER_BYTES - 32]).digest() != raw[HEADER_BYTES - 32:]:
        raise ValueError("header checksum mismatch")
    (magic, ver, hbytes, data_off, stride, vb, sb, layer, rank, tp, flags, row_lo, row_hi, num_rows, values_off, scales_off,
     source_size, source_mtime_ns, hsha, dsha, data_bytes) = _HDR.unpack_from(raw)
    return dict(layout_version=ver, header_bytes=hbytes, data_off=data_off, row_stride=stride, value_bytes=vb, scale_bytes=sb,
                layer=layer, rank=rank, tp=tp, flags=flags, complete=bool(flags & FLAG_COMPLETE), row_lo=row_lo, row_hi=row_hi,
                num_rows=num_rows, values_off=values_off, scales_off=scales_off, source_size=source_size,
                source_mtime_ns=source_mtime_ns, source_header_sha256=hsha.hex(), data_sha256=dsha.hex(), data_bytes=data_bytes)


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------------------------------------------------
# I/O helpers: page-aligned buffers, O_DIRECT with a buffered+DONTNEED fallback, rate limit
# ---------------------------------------------------------------------------------------------------------------------
class Buf:
    """Page-aligned scratch memory (anonymous mmap: zero-filled, valid for O_DIRECT)."""

    def __init__(self, nbytes: int):
        self.size = (nbytes + ALIGN - 1) & ~(ALIGN - 1)
        self.m = mmap.mmap(-1, self.size)

    def view(self, start: int = 0, length: int | None = None) -> memoryview:
        end = self.size if length is None else start + length
        return memoryview(self.m)[start:end]

    def close(self) -> None:
        try:
            self.m.close()
        except BufferError:  # a view is still alive somewhere: the GC frees it
            pass


def fadvise_dontneed(fd: int, off: int = 0, length: int = 0) -> None:
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, off, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def fdatasync(fd: int) -> None:
    (getattr(os, "fdatasync", None) or os.fsync)(fd)


def open_file(path: str, flags: int, io: str, mode: int = 0o644):
    """(fd, direct): O_DIRECT when io allows it and the filesystem accepts it, else a buffered descriptor."""
    flags |= getattr(os, "O_CLOEXEC", 0)
    want_direct = io in ("auto", "direct") and hasattr(os, "O_DIRECT")
    if io == "direct" and not want_direct:
        raise PackError("--io direct: O_DIRECT is not available on this platform")
    if want_direct:
        try:
            return os.open(path, flags | os.O_DIRECT, mode), True
        except OSError as e:
            if io == "direct":
                raise PackError(f"--io direct: cannot open {path} with O_DIRECT: {e}")
    return os.open(path, flags, mode), False


class Source:
    """Sequential big-block reader of the checkpoint shard."""

    def __init__(self, path: str, io: str):
        self.path = path
        self.fd, self.direct = open_file(path, os.O_RDONLY, io)
        if self.direct:  # one aligned probe read: some filesystems accept the flag at open() and fail the first read
            probe = Buf(ALIGN)
            try:
                os.preadv(self.fd, [probe.view()], 0)
            except OSError:
                os.close(self.fd)
                if io == "direct":
                    raise PackError(f"--io direct: O_DIRECT reads fail on {path}")
                self.fd, self.direct = open_file(path, os.O_RDONLY, "buffered")
            finally:
                probe.close()

    def read_into(self, buf: Buf, off: int, n: int) -> memoryview:
        """The n bytes at file offset `off`, as a view into `buf` (O_DIRECT: through the aligned window around them)."""
        a0 = off & ~(ALIGN - 1) if self.direct else off
        skew = off - a0
        want = ((skew + n + ALIGN - 1) & ~(ALIGN - 1)) if self.direct else n
        if want > buf.size:
            raise PackError("internal: scratch buffer too small")
        mv = buf.view(0, want)
        got = 0
        while got < want:
            r = os.preadv(self.fd, [mv[got:]], a0 + got)
            if r <= 0:
                break
            got += r
        if got < skew + n:
            raise PackError(f"{self.path}: short read at offset {a0 + got} (wanted {n} B at {off}); the shard is truncated?")
        if not self.direct:
            fadvise_dontneed(self.fd, off, n)
        return mv[skew:skew + n]

    def close(self) -> None:
        os.close(self.fd)


class Throttle:
    """Average-rate limiter: sleep so that bytes/elapsed stays under max_mbps (0 = off)."""

    def __init__(self, max_mbps: float):
        self.rate = max_mbps * 1e6
        self.t0 = time.monotonic()
        self.bytes = 0

    def account(self, n: int) -> None:
        if self.rate <= 0:
            return
        self.bytes += n
        ahead = self.bytes / self.rate - (time.monotonic() - self.t0)
        if ahead > 0:
            time.sleep(ahead)


def transform_chunk(v: memoryview, s: memoryview, n: int, out: Buf) -> None:
    """Records for n rows: values and scales in, 512-byte records out. Word-wise (8-byte) strided copies; the padding stays zero."""
    ov = out.view(0, n * STRIDE).cast("Q")
    vv = v.cast("Q")
    sv = s.cast("Q")
    wv, wr = ROW_V // 8, STRIDE // 8
    for w in range(wv):
        ov[w::wr] = vv[w::wv]
    ov[wv::wr] = sv  # the 8 scale bytes of a row are exactly one word, right after the 32 value words


# ---------------------------------------------------------------------------------------------------------------------
# journal
# ---------------------------------------------------------------------------------------------------------------------
def journal_params(tbl: Table, layer, tp, rank, row_lo, row_hi, chunk_rows) -> dict:
    return dict(version=1, layer=layer, tp=tp, rank=rank, row_lo=row_lo, row_hi=row_hi, chunk_rows=chunk_rows,
                source_size=tbl.file_size, source_mtime_ns=tbl.mtime_ns, source_header_sha256=tbl.header_sha256,
                values_off=tbl.values_off, scales_off=tbl.scales_off)


def load_journal(path: str):
    try:
        with open(path) as f:
            j = json.load(f)
        return j if isinstance(j, dict) else None
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------------------------------------------------
class _TestStop(Exception):
    pass


def manifest_matches(m: dict, tbl: Table, layer, tp, rank, row_lo, row_hi, packed_path: str) -> str:
    """'' if an existing manifest still describes a complete pack of exactly this slice of this source, else why not."""
    try:
        if m.get("format") != FORMAT or m.get("layout_version") != LAYOUT_VERSION or not m.get("complete"):
            return "manifest is not a complete layout-1 manifest"
        src = m["source"]
        for key, want in (("layer", layer), ("tp", tp), ("rank", rank), ("row_lo", row_lo), ("row_hi", row_hi)):
            if m[key] != want:
                return f"manifest {key} {m[key]} != {want}"
        if (src["size"], src["mtime_ns"], src["header_sha256"], src["values_off"], src["scales_off"]) != (
                tbl.file_size, tbl.mtime_ns, tbl.header_sha256, tbl.values_off, tbl.scales_off):
            return "the source shard changed since the pack was made"
        if os.stat(packed_path).st_size != m["file_size"]:
            return "the packed file's size differs from the manifest"
    except (KeyError, OSError, TypeError) as e:
        return f"manifest unusable ({type(e).__name__}: {e})"
    return ""


def pack_slice(tbl: Table, layer: int, tp: int, rank: int, row_lo: int, row_hi: int, out_dir: str, *, chunk_rows: int = DEFAULT_CHUNK_ROWS,
               io: str = "auto", max_mbps: float = 0.0, verify_rows: int = 65536, verify_threads: int = 4, seed: int | None = None,
               force: bool = False, name_suffix: str = "", head_range=(None, None), reserve_bytes: int = 1 << 30,
               _stop_after_chunks: int | None = None) -> dict:
    if chunk_rows < 8 or chunk_rows % 8:
        raise PackError("--chunk-rows must be a multiple of 8 (chunk boundaries stay 4 KiB aligned)")
    rows = row_hi - row_lo
    if rows <= 0 or row_lo < 0 or row_hi > tbl.num_rows:
        raise PackError(f"row range [{row_lo}, {row_hi}) is empty or outside the table (0..{tbl.num_rows})")
    os.makedirs(out_dir, exist_ok=True)
    stem, mname, jname = pack_names(layer, tp, rank)
    stem, mname, jname = stem + name_suffix, mname.replace(".pack.json", f".pack{name_suffix}.json"), jname.replace(".pack.journal", f".pack{name_suffix}.journal")
    packed_path, manifest_path, journal_path = (os.path.join(out_dir, x) for x in (stem, mname, jname))
    data_bytes = rows * STRIDE
    total = DATA_OFF + data_bytes
    nchunks = (rows + chunk_rows - 1) // chunk_rows
    tag = f"L{layer} r{rank}/{tp}"

    if os.path.exists(manifest_path):
        m = None
        try:
            with open(manifest_path) as f:
                m = json.load(f)
        except (OSError, ValueError):
            pass
        why = manifest_matches(m, tbl, layer, tp, rank, row_lo, row_hi, packed_path) if isinstance(m, dict) else "manifest unreadable"
        if not why and not force:
            log(f"{tag}: already packed ({packed_path}, manifest matches the source); --force repacks, `verify` checks it")
            return m
        log(f"{tag}: removing the existing manifest ({'--force' if not why else why}) before rewriting the pack")
        os.unlink(manifest_path)
        fsync_dir(out_dir)

    st = os.statvfs(out_dir)
    jr = load_journal(journal_path)
    params = journal_params(tbl, layer, tp, rank, row_lo, row_hi, chunk_rows)
    resume = bool(jr) and not force and all(jr.get(k) == v for k, v in params.items()) and os.path.exists(packed_path)
    have = os.path.getsize(packed_path) if os.path.exists(packed_path) else 0
    need_free = max(0, total - (have if resume else 0)) + reserve_bytes
    if st.f_bavail * st.f_frsize < need_free:
        raise PackError(f"{out_dir}: only {st.f_bavail * st.f_frsize / 2**30:.1f} GiB free, {need_free / 2**30:.1f} GiB needed "
                        f"(the pack + a {reserve_bytes / 2**30:.1f} GiB reserve; --reserve-mib)")

    src = Source(tbl.path, io)
    dst_fd, dst_direct = open_file(packed_path, os.O_RDWR | os.O_CREAT, io, 0o644)
    log(f"{tag}: rows [{row_lo}, {row_hi}) = {rows} -> {packed_path} ({total / 2**30:.2f} GiB), {nchunks} chunks of {chunk_rows} rows, "
        f"source I/O {'O_DIRECT' if src.direct else 'buffered+DONTNEED'}, pack I/O {'O_DIRECT' if dst_direct else 'buffered+DONTNEED'}")

    vbuf, sbuf = Buf(chunk_rows * ROW_V + 2 * ALIGN), Buf(chunk_rows * ROW_S + 2 * ALIGN)
    obuf, rbuf, hbuf = Buf(chunk_rows * STRIDE + ALIGN), Buf(chunk_rows * STRIDE + ALIGN), Buf(HEADER_BYTES)
    try:
        hasher = hashlib.sha256()
        prefix: list = []
        done = 0
        if resume:
            log(f"{tag}: journal found with {len(jr['done'])} finished chunks: re-hashing them against the journal")
            for k, want in enumerate(jr["done"]):
                n = min(chunk_rows, rows - k * chunk_rows)
                if n <= 0 or have < DATA_OFF + (k * chunk_rows + n) * STRIDE:
                    break
                mv = rbuf.view(0, n * STRIDE)
                got = 0
                base = DATA_OFF + k * chunk_rows * STRIDE
                rd = (n * STRIDE + ALIGN - 1) & ~(ALIGN - 1) if dst_direct else n * STRIDE
                while got < rd:
                    r = os.preadv(dst_fd, [rbuf.view(got, rd - got)], base + got)
                    if r <= 0:
                        break
                    got += r
                if got < n * STRIDE:
                    break
                trial = hasher.copy()
                trial.update(mv)
                if trial.hexdigest() != want:
                    log(f"{tag}: chunk {k} does not match the journal: redoing from there")
                    break
                hasher = trial
                prefix.append(want)
                done += 1
            if not dst_direct:
                fadvise_dontneed(dst_fd)
            log(f"{tag}: resuming at chunk {done}/{nchunks}")
        else:
            if jr or have:
                log(f"{tag}: starting over (no usable journal / --force)")
            os.ftruncate(dst_fd, 0)
            if hasattr(os, "posix_fallocate"):
                try:
                    os.posix_fallocate(dst_fd, 0, total)
                except OSError as e:
                    if e.errno == 28:  # ENOSPC
                        raise PackError(f"{out_dir}: no space left to preallocate {total / 2**30:.2f} GiB")
                    # EOPNOTSUPP/EINVAL: the filesystem has no preallocation, the writes will extend the file
            hdr = encode_header(layer=layer, rank=rank, tp=tp, flags=0, row_lo=row_lo, row_hi=row_hi, num_rows=tbl.num_rows,
                                values_off=tbl.values_off, scales_off=tbl.scales_off, source_size=tbl.file_size,
                                source_mtime_ns=tbl.mtime_ns, source_header_sha256=tbl.header_sha256, data_bytes=data_bytes)
            hbuf.view(0, HEADER_BYTES)[:] = hdr
            os.pwritev(dst_fd, [hbuf.view(0, HEADER_BYTES)], 0)
            fdatasync(dst_fd)
        if done == 0 and resume:  # resumed but nothing usable: same as a fresh start (the header is rewritten below at the end anyway)
            pass

        throttle = Throttle(max_mbps)
        t_start = time.monotonic()
        t_log = t_start
        moved = 0
        for k in range(done, nchunks):
            r0 = row_lo + k * chunk_rows
            n = min(chunk_rows, row_hi - r0)
            v = src.read_into(vbuf, tbl.values_off + r0 * ROW_V, n * ROW_V)
            s = src.read_into(sbuf, tbl.scales_off + r0 * ROW_S, n * ROW_S)
            transform_chunk(v, s, n, obuf)
            del v, s
            nb = n * STRIDE
            wb = (nb + ALIGN - 1) & ~(ALIGN - 1) if dst_direct else nb
            if wb > nb:  # O_DIRECT tail: whole 4 KiB blocks; the zero tail is cut off again at the end
                obuf.view(nb, wb - nb)[:] = bytes(wb - nb)
            hasher.update(obuf.view(0, nb))
            digest = hasher.hexdigest()
            off = DATA_OFF + k * chunk_rows * STRIDE
            mv = obuf.view(0, wb)
            put = 0
            while put < wb:
                put += os.pwritev(dst_fd, [mv[put:]], off + put)
            fdatasync(dst_fd)
            if not dst_direct:
                fadvise_dontneed(dst_fd, off, wb)
            prefix.append(digest)
            jr_new = dict(params, done=prefix)
            write_json_atomic(journal_path, jr_new)
            moved += nb
            throttle.account(nb)
            now = time.monotonic()
            if now - t_log >= 5 or k + 1 == nchunks:
                t_log = now
                log(f"{tag}: chunk {k + 1}/{nchunks}  {moved / max(now - t_start, 1e-9) / 1e6:.0f} MB/s written")
            if _stop_after_chunks is not None and k + 1 >= _stop_after_chunks and k + 1 < nchunks:
                raise _TestStop()
        data_sha = hasher.hexdigest()
        hdr = encode_header(layer=layer, rank=rank, tp=tp, flags=FLAG_COMPLETE, row_lo=row_lo, row_hi=row_hi, num_rows=tbl.num_rows,
                            values_off=tbl.values_off, scales_off=tbl.scales_off, source_size=tbl.file_size,
                            source_mtime_ns=tbl.mtime_ns, source_header_sha256=tbl.header_sha256, data_sha256=data_sha, data_bytes=data_bytes)
        hbuf.view(0, HEADER_BYTES)[:] = hdr
        os.pwritev(dst_fd, [hbuf.view(0, HEADER_BYTES)], 0)
        fdatasync(dst_fd)
        os.ftruncate(dst_fd, total)  # exact size: drops the O_DIRECT zero tail
        fdatasync(dst_fd)
        if not dst_direct:
            fadvise_dontneed(dst_fd)
    finally:
        os.close(dst_fd)
        src.close()
        for b in (vbuf, sbuf, obuf, rbuf, hbuf):
            b.close()

    if os.path.getsize(packed_path) != total:
        raise PackError(f"{packed_path}: size {os.path.getsize(packed_path)} != expected {total}")
    sample = chunk_edge_rows(row_lo, row_hi, chunk_rows)
    checked, bad = verify_sample(tbl, packed_path, DATA_OFF, row_lo, row_hi, verify_rows, verify_threads, seed, extra_rows=sample)
    if bad:
        raise PackError(f"{tag}: end-of-run verification FAILED: {len(bad)} of {checked} sampled rows differ from the original shard "
                        f"(first: row {bad[0]}); NOT writing the manifest")
    log(f"{tag}: verified {checked} rows (random + slice/chunk edges) against the original shard: identical")
    pst = os.stat(packed_path)
    m = dict(
        format=FORMAT, layout_version=LAYOUT_VERSION, complete=True, created_utc=utc_now(), tool=TOOL, host=socket.gethostname(),
        layer=layer, rank=rank, tp=tp, head_start=head_range[0], head_end=head_range[1],
        row_lo=row_lo, row_hi=row_hi, rows=rows, num_rows=tbl.num_rows, data_off=DATA_OFF, row_stride=STRIDE, value_bytes=ROW_V,
        scale_bytes=ROW_S, pad_bytes=PAD, data_bytes=data_bytes, file=stem, file_size=pst.st_size, file_mtime_ns=pst.st_mtime_ns,
        data_sha256=data_sha, chunk_rows=chunk_rows, prefix_sha256=prefix, custom_range=bool(name_suffix),
        source=dict(path=os.path.realpath(tbl.path), shard=tbl.shard, size=tbl.file_size, mtime_ns=tbl.mtime_ns, header_len=tbl.header_len,
                    header_sha256=tbl.header_sha256, values_off=tbl.values_off, scales_off=tbl.scales_off, num_rows=tbl.num_rows,
                    values_tensor=f"layers.{layer}.engram.embed.weight", scales_tensor=f"layers.{layer}.engram.embed.scale"),
        verify=dict(rows_checked=checked, ok=True, seed=seed),
    )
    write_json_atomic(manifest_path, m)
    fsync_dir(out_dir)
    try:
        os.unlink(journal_path)
    except OSError:
        pass
    log(f"{tag}: DONE {packed_path}  data sha256 {data_sha[:16]}..  manifest {manifest_path}")
    return m


# ---------------------------------------------------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------------------------------------------------
def chunk_edge_rows(row_lo: int, row_hi: int, chunk_rows: int) -> list:
    out = {row_lo, row_hi - 1}
    for r0 in range(row_lo, row_hi, chunk_rows):
        out.update((r0, min(r0 + chunk_rows, row_hi) - 1))
        if r0 > row_lo:
            out.add(r0 - 1)
    return sorted(out)


def verify_sample(tbl: Table, packed_path: str, data_off: int, row_lo: int, row_hi: int, nrows: int, threads: int = 4,
                  seed: int | None = None, extra_rows=()):
    """Compare records of the packed file with the ORIGINAL shard (buffered pread, independent of the pack's I/O path).

    Returns (rows_checked, [global ids that differ])."""
    rng = random.Random(seed)
    rows = row_hi - row_lo
    pick = set(extra_rows)
    pick.update(rng.sample(range(row_lo, row_hi), min(rows, max(0, nrows))))
    ids = sorted(pick)
    src_fd = os.open(tbl.path, os.O_RDONLY)
    pk_fd = os.open(packed_path, os.O_RDONLY)

    def check(batch):
        bad = []
        for g in batch:
            rec = os.pread(pk_fd, STRIDE, data_off + (g - row_lo) * STRIDE)
            v = os.pread(src_fd, ROW_V, tbl.values_off + g * ROW_V)
            s = os.pread(src_fd, ROW_S, tbl.scales_off + g * ROW_S)
            if len(rec) != STRIDE or len(v) != ROW_V or len(s) != ROW_S or rec[:ROW_V] != v or rec[ROW_V:USED] != s or rec[USED:] != ZERO_PAD:
                bad.append(g)
        fadvise_dontneed(src_fd)
        fadvise_dontneed(pk_fd)
        return bad

    try:
        step = 2048
        batches = [ids[i:i + step] for i in range(0, len(ids), step)]
        bad: list = []
        with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
            for b in ex.map(check, batches):
                bad.extend(b)
    finally:
        os.close(src_fd)
        os.close(pk_fd)
    return len(ids), sorted(bad)


def verify_pack(tbl: Table, layer, tp, rank, row_lo, row_hi, out_dir: str, *, rows: int = 65536, threads: int = 4, seed=None,
                full_hash: bool = False, full_compare: bool = False, io: str = "auto", max_mbps: float = 0.0, name_suffix: str = "") -> list:
    """All checks the reader makes at engine start, plus a sample (and optionally everything). Returns a list of problems."""
    stem, mname, _ = pack_names(layer, tp, rank)
    stem, mname = stem + name_suffix, mname.replace(".pack.json", f".pack{name_suffix}.json")
    packed_path, manifest_path = os.path.join(out_dir, stem), os.path.join(out_dir, mname)
    tag = f"L{layer} r{rank}/{tp}"
    problems: list = []
    try:
        with open(manifest_path) as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        return [f"{tag}: manifest {manifest_path}: {e}"]
    why = manifest_matches(m, tbl, layer, tp, rank, row_lo, row_hi, packed_path)
    if why:
        return [f"{tag}: {why}"]
    try:
        with open(packed_path, "rb") as f:
            hd = decode_header(f.read(HEADER_BYTES))
    except (OSError, ValueError) as e:
        return [f"{tag}: header of {packed_path}: {e}"]
    src = m["source"]
    for key, have, want in (
            ("layer", hd["layer"], m["layer"]), ("rank", hd["rank"], m["rank"]), ("tp", hd["tp"], m["tp"]),
            ("row_lo", hd["row_lo"], m["row_lo"]), ("row_hi", hd["row_hi"], m["row_hi"]), ("data_off", hd["data_off"], m["data_off"]),
            ("data_bytes", hd["data_bytes"], m["data_bytes"]), ("data_sha256", hd["data_sha256"], m["data_sha256"]),
            ("num_rows", hd["num_rows"], m["num_rows"]), ("values_off", hd["values_off"], src["values_off"]),
            ("scales_off", hd["scales_off"], src["scales_off"]), ("source_size", hd["source_size"], src["size"]),
            ("source_mtime_ns", hd["source_mtime_ns"], src["mtime_ns"])):
        if have != want:
            problems.append(f"{tag}: header {key} {have} != manifest {want}")
    if not hd["complete"]:
        problems.append(f"{tag}: header says the pack is incomplete")
    if hd["source_header_sha256"] != m["source"]["header_sha256"]:
        problems.append(f"{tag}: header source sha256 != manifest")
    if problems:
        return problems
    log(f"{tag}: manifest, header and source identity agree")
    checked, bad = verify_sample(tbl, packed_path, m["data_off"], row_lo, row_hi, rows, threads, seed,
                                 extra_rows=chunk_edge_rows(row_lo, row_hi, m["chunk_rows"]))
    log(f"{tag}: sample of {checked} rows vs the original shard: {'identical' if not bad else f'{len(bad)} DIFFER (first: row {bad[0]})'}")
    if bad:
        problems.append(f"{tag}: {len(bad)} sampled rows differ from the shard (first: row {bad[0]})")
    if full_hash or full_compare:
        problems.extend(_verify_full(tbl, m, packed_path, tag, full_compare, io, max_mbps))
    return problems


def _verify_full(tbl: Table, m: dict, packed_path: str, tag: str, compare: bool, io: str, max_mbps: float) -> list:
    chunk_rows, rows, row_lo = m["chunk_rows"], m["rows"], m["row_lo"]
    nchunks = (rows + chunk_rows - 1) // chunk_rows
    pk_fd, pk_direct = open_file(packed_path, os.O_RDONLY, io)
    src = Source(tbl.path, io) if compare else None
    rbuf = Buf(chunk_rows * STRIDE + ALIGN)
    vbuf, sbuf, obuf = (Buf(chunk_rows * ROW_V + 2 * ALIGN), Buf(chunk_rows * ROW_S + 2 * ALIGN), Buf(chunk_rows * STRIDE + ALIGN)) if compare else (None,) * 3
    hasher = hashlib.sha256()
    problems: list = []
    throttle = Throttle(max_mbps)
    t0 = time.monotonic()
    try:
        for k in range(nchunks):
            n = min(chunk_rows, rows - k * chunk_rows)
            nb = n * STRIDE
            rd = (nb + ALIGN - 1) & ~(ALIGN - 1) if pk_direct else nb
            got = 0
            base = m["data_off"] + k * chunk_rows * STRIDE
            while got < rd:
                r = os.preadv(pk_fd, [rbuf.view(got, rd - got)], base + got)
                if r <= 0:
                    break
                got += r
            if got < nb:
                problems.append(f"{tag}: packed file ends inside chunk {k}")
                break
            data = rbuf.view(0, nb)
            hasher.update(data)
            if k < len(m["prefix_sha256"]) and hasher.hexdigest() != m["prefix_sha256"][k]:
                problems.append(f"{tag}: running sha256 differs from the manifest at chunk {k} (rows {row_lo + k * chunk_rows}..): the data was modified")
                break
            if compare:
                r0 = row_lo + k * chunk_rows
                v = src.read_into(vbuf, tbl.values_off + r0 * ROW_V, n * ROW_V)
                s = src.read_into(sbuf, tbl.scales_off + r0 * ROW_S, n * ROW_S)
                transform_chunk(v, s, n, obuf)
                del v, s
                exp = obuf.view(0, nb)
                step = 8 << 20
                for i in range(0, nb, step):
                    if bytes(exp[i:i + step]) != bytes(data[i:i + step]):
                        problems.append(f"{tag}: chunk {k} differs from a fresh transform of the source near byte {i} (row {r0 + i // STRIDE})")
                        break
                del exp
            del data
            if not pk_direct:
                fadvise_dontneed(pk_fd, base, rd)
            throttle.account(nb * (2 if compare else 1))
            if problems:
                break
        else:
            if hasher.hexdigest() != m["data_sha256"]:
                problems.append(f"{tag}: sha256 of the data region != manifest data_sha256")
    finally:
        os.close(pk_fd)
        if src:
            src.close()
        for b in (rbuf, vbuf, sbuf, obuf):
            if b:
                b.close()
    if not problems:
        log(f"{tag}: full {'hash + compare with a fresh transform of the source' if compare else 'hash'}: OK ({time.monotonic() - t0:.0f} s)")
    return problems


# ---------------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------------
def _layers_arg(s):
    return tuple(int(x) for x in s.split(",") if x.strip()) if s else None


def _cfg_overrides(a) -> dict:
    return dict(vocab_size=a.vocab_size, n_heads=a.n_heads, max_ngram_size=a.max_ngram, head_dim=a.head_dim,
                layer_ids=_layers_arg(a.layer_ids) if a.layer_ids else None,
                num_embeddings=tuple(int(x) for x in a.num_embeddings.split(",")) if a.num_embeddings else None)


def cmd_plan(a) -> int:
    ec, items = derive_slices(a.model_dir, a.tp, _layers_arg(a.layers), **_cfg_overrides(a))
    out = []
    for tbl, sizes in items:
        print(f"layer {tbl.layer}: {tbl.shard}  values @{tbl.values_off}  scales @{tbl.scales_off}  {tbl.num_rows} rows  "
              f"(header {tbl.header_len} B, sha256 {tbl.header_sha256[:16]}..)")
        for r in range(a.tp):
            h0, h1, lo, hi = rank_slice(sizes, a.tp, r)
            n = hi - lo
            print(f"  rank {r}: heads {h0}-{h1 - 1}  rows [{lo}, {hi}) n={n}  values {tbl.values_off + lo * ROW_V}+{n * ROW_V}  "
                  f"scales {tbl.scales_off + lo * ROW_S}+{n * ROW_S}  -> packed {(DATA_OFF + n * STRIDE) / 1e9:.2f} GB "
                  f"({stem_name(tbl.layer, a.tp, r)})")
            out.append(dict(layer=tbl.layer, rank=r, heads=[h0, h1], row_lo=lo, row_hi=hi, rows=n))
    if a.json:
        print(json.dumps(out))
    return 0


def stem_name(layer, tp, rank) -> str:
    return pack_names(layer, tp, rank)[0]


def _slices_for(a):
    ec, items = derive_slices(a.model_dir, a.tp, _layers_arg(a.layers), **_cfg_overrides(a))
    for tbl, sizes in items:
        h0, h1, lo, hi = rank_slice(sizes, a.tp, a.rank)
        suffix = ""
        if a.row_range:
            if len(items) != 1:
                raise PackError("--row-range needs exactly one layer (--layers N)")
            lo, hi = (int(x) for x in a.row_range.split(":"))
            suffix = f"-x{lo}-{hi}"
        yield tbl, lo, hi, (h0, h1), suffix


def cmd_pack(a) -> int:
    for tbl, lo, hi, heads, suffix in _slices_for(a):
        pack_slice(tbl, tbl.layer, a.tp, a.rank, lo, hi, a.out_dir, chunk_rows=a.chunk_rows, io=a.io, max_mbps=a.max_mbps,
                   verify_rows=a.verify_rows, verify_threads=a.verify_threads, seed=a.seed, force=a.force, name_suffix=suffix, head_range=heads,
                   reserve_bytes=a.reserve_mib << 20)
    return 0


def cmd_verify(a) -> int:
    bad = []
    for tbl, lo, hi, heads, suffix in _slices_for(a):
        bad += verify_pack(tbl, tbl.layer, a.tp, a.rank, lo, hi, a.out_dir, rows=a.rows, threads=a.verify_threads, seed=a.seed,
                           full_hash=a.full_hash or a.full_compare, full_compare=a.full_compare, io=a.io, max_mbps=a.max_mbps, name_suffix=suffix)
    for p in bad:
        log("PROBLEM: " + p)
    log("verify: " + ("FAILED" if bad else "OK"))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="engram_pack.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_out):
        p.add_argument("--model-dir", required=True, help="checkpoint directory (model.safetensors.index.json, config.json, shards)")
        p.add_argument("--tp", type=int, default=8)
        p.add_argument("--layers", default="", help="comma separated Engram layer ids (default: all of the model, e.g. 1,14)")
        p.add_argument("--vocab-size", type=int, help="override engram_vocab_size (tests)")
        p.add_argument("--n-heads", type=int, help="override engram_n_heads (tests)")
        p.add_argument("--max-ngram", type=int, help="override engram_max_ngram_size (tests)")
        p.add_argument("--head-dim", type=int, help="override engram_head_dim (tests)")
        p.add_argument("--layer-ids", help="override engram_layer_ids, comma separated (tests)")
        p.add_argument("--num-embeddings", help="override engram_num_embeddings, comma separated (tests)")
        if need_out:
            p.add_argument("--out-dir", required=True, help="where the .pack/.pack.json files go (must be on the node's NVMe)")
            p.add_argument("--rank", type=int, required=True, help="TP rank of the node this runs on (NODE1 is rank 0, NODE2 rank 1, ...) = which rank's slice to pack")
            p.add_argument("--row-range", help="LO:HI debugging override of the slice (single layer only; the file gets a -x<lo>-<hi> name suffix)")
            p.add_argument("--io", choices=("auto", "direct", "buffered"), default="auto")
            p.add_argument("--max-mbps", type=float, default=0.0, help="cap the average rate (MB/s) so a pack can run next to a serving engine")
            p.add_argument("--verify-threads", type=int, default=4)
            p.add_argument("--seed", type=int, default=None, help="seed of the random verification sample")

    p = sub.add_parser("plan", help="print the rank slices and file offsets (headers only; reads no table data)")
    common(p, False)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("pack", help="pack one rank's slice of the requested layers (resumable)")
    common(p, True)
    p.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    p.add_argument("--verify-rows", type=int, default=65536, help="random rows compared with the original shard at the end")
    p.add_argument("--force", action="store_true", help="repack even if a matching manifest exists / discard the journal")
    p.add_argument("--reserve-mib", type=int, default=1024, help="free space that must remain after the pack (default 1024)")
    p.set_defaults(fn=cmd_pack)
    p = sub.add_parser("verify", help="check an existing pack against the manifest and the source shard")
    common(p, True)
    p.add_argument("--rows", type=int, default=65536, help="random rows compared with the original shard")
    p.add_argument("--full-hash", action="store_true", help="re-read the whole pack and check the running sha256 against the manifest")
    p.add_argument("--full-compare", action="store_true", help="also re-transform the whole source and compare byte for byte (reads source + pack)")
    p.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except PackError as e:
        log(f"FAILED: {e}")
        return 1
    except OSError as e:  # disk full, I/O error, missing file: the journal is consistent, the same command resumes once the cause is fixed
        log(f"FAILED: {type(e).__name__}: {e}")
        return 1
    except KeyboardInterrupt:
        log("interrupted: the journal is consistent, re-run the same command to resume")
        return 1


if __name__ == "__main__":
    sys.exit(main())
