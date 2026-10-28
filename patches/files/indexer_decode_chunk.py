# SPDX-License-Identifier: Apache-2.0
"""[patch-043 graphpool] Row-chunked decode path of the DSA sparse-attention indexer (installed as vllm/_indexer_decode_chunk.py).

WHY: in the flattened decode layout (one Q row per decode token; GB10 / sm_121, 128 requests x 6 tokens = 768 rows in the largest
CUDA graph) DeepGEMM's fp8_fp4_paged_mqa_logits allocates a [rows, max_model_len // compress_ratio] fp32 logits tensor -- 3 GiB at
ratio 1 and max_model_len 1,048,576 -- and the candidate selection of the two-level (v4.1) indexer adds a [rows, ceil(width / 8)]
fp32 score tensor (layer 20) or a [rows, ceil(width / 8) + 1] uint8 flag tensor (layers 24..36).  Inside a CUDA graph that memory
lives in the graph pool for the lifetime of the engine (3.50 GiB per rank, measured by the 10-28 pool analysis) and is lost to the KV
cache.  Every row is independent, so the rows can be processed in chunks: logits -> candidate select / mask -> top-k for rows
[r0, r1), writing into the caller's slices, then the chunk's temporaries are freed (and, inside a graph capture, reused by the next
chunk), so the pool needs only ONE chunk's worth.

Setting (read once at import; unset / empty / 0 / off = the stock code path, this module is never entered):
  VLLM_GB10_INDEXER_DECODE_CHUNK_MB=<cap>   cap, in MiB, for the logits + candidate temporaries of one chunk (1 .. 65536)
  VLLM_GB10_INDEXER_DECODE_CHUNK_SHARE_META=1   (default 1) compute each chunk's paged-MQA schedule once per forward context and
                                                share it between the indexer layers of one KV-cache group; 0 = recompute per layer
  VLLM_GB10_INDEXER_DECODE_CHUNK_SCOPE=all|cand   (default all) cand = the SAFE SUBSET: logits and the final top-k run exactly as stock (one
                                                full-width call each), only the candidate select / mask kernels run in row chunks (saves the
                                                ~0.4 GiB of scores / flags, not the 3 GiB of logits); the cap then counts candidate temporaries only

The hook (run_decode) is called from sparse_attn_indexer's decode branch right before the logits are computed.  It returns False --
and the stock code runs unchanged -- unless ALL of these hold: cap set; flattened layout (next_n == 1, seq_lens (rows, 1), no
varlen `indices`, no padded/packed path, no DCP); top-k backend one of auto (non-cooperative platforms) / persistent / per_row /
flashinfer; the unchunked size of the logits + candidate temporaries exceeds the cap; at least two chunks of MIN_CHUNK_ROWS rows fit.
So every graph whose unchunked temporaries fit in the cap stays byte-for-byte stock.

EXACTNESS (what the argument rests on -- the GPU test in tests/043 checks each point):
  * every kernel on the path is row-local: paged-MQA logits (per (row, key): the same fp8 MMA over the head dimension; the schedule
    metadata only decides WHICH SM does the work, never the arithmetic), _block_scores_kernel / _candidate_flags_kernel /
    _mask_candidates_kernel (one row per program), the top-k kernels (per-row selection).  The chunk's schedule metadata is
    DeepGEMM's own get_paged_mqa_logits_metadata on the chunk's context lengths, with the same block size / SM count as the builder.
  * columns past a row's context length hold uninitialised memory in BOTH variants and are read by no consumer (they are never
    inside [start, end) of any kernel above).
  * torch.topk of select_candidate_blocks (layer 20) picks its algorithm from the NUMBER OF ROWS (full sort for <= 10 slices of
    >= 100000 elements, radix select otherwise): chunks therefore keep >= MIN_CHUNK_ROWS (16) rows, never fewer than the radix-select
    threshold, whenever the unchunked call would have used radix select.  The order of the candidate ids is irrelevant to the consumers
    (flags), their SET is the same for both algorithms (lowest index wins ties).
  * the final top-k (persistent_topk on GB10) gets the SAME max_seq_len host bound as the unchunked call; its per-row result must not
    depend on the row count -- UNVERIFIED here (kernel source not in the tree): tests/043 compares it, order included, on tie-heavy data.
"""
from __future__ import annotations

import os

ENV_CAP_MB = "VLLM_GB10_INDEXER_DECODE_CHUNK_MB"
ENV_SHARE_META = "VLLM_GB10_INDEXER_DECODE_CHUNK_SHARE_META"
TAG = "[patch-043]"

# Smallest chunk (rows).  torch.topk switches from radix select to a full sort at <= 10 slices; staying above that keeps the
# candidate selection on the same algorithm as the unchunked call.  Module attribute: tests lower it to probe tiny chunks.
MIN_CHUNK_ROWS = 16

# top-k backends the chunked path may drive: their per-row result is by construction independent of the batch (modulo the
# UNVERIFIED point above).  "cooperative" (<= 64 rows only), "deep_select" (bf16 logits, SM100), "aiter" and "torch" stay stock.
_UNSUPPORTED_BACKENDS = ("cooperative", "deep_select", "aiter", "torch")


def _parse_cap_mb(raw: str | None) -> int:
    if raw is None:
        return 0
    s = raw.strip().lower()
    if s in ("", "0", "off", "false", "no"):
        return 0
    try:
        mb = int(s)
    except ValueError:
        raise ValueError(f"{TAG} {ENV_CAP_MB}={raw!r}: expected a whole number of MiB (1 .. 65536) or unset/0 = off") from None
    if not 1 <= mb <= 65536:
        raise ValueError(f"{TAG} {ENV_CAP_MB}={mb}: outside 1 .. 65536 MiB")
    return mb


def _parse_flag(name: str, raw: str | None, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{TAG} {name}={raw!r}: expected 0 or 1")


ENV_SCOPE = "VLLM_GB10_INDEXER_DECODE_CHUNK_SCOPE"


def _parse_scope(raw: str | None) -> str:
    s = (raw or "").strip().lower()
    if s in ("", "all"):
        return "all"
    if s == "cand":
        return "cand"
    raise ValueError(f"{TAG} {ENV_SCOPE}={raw!r}: expected all or cand")


CAP_MB = _parse_cap_mb(os.environ.get(ENV_CAP_MB))
CAP_BYTES = CAP_MB << 20
SHARE_META = _parse_flag(ENV_SHARE_META, os.environ.get(ENV_SHARE_META), True)
SCOPE = _parse_scope(os.environ.get(ENV_SCOPE))

_logged: set = set()
_verified: set = set()
_disabled = False  # set when the one-time schedule cross-check fails: the stock path runs from then on
CALLS = 0  # number of run_decode calls that processed their rows in chunks (tests assert it moved)


def _log_once(key, msg: str, *args) -> None:
    if key in _logged:
        return
    _logged.add(key)
    try:
        from vllm.logger import init_logger

        init_logger(__name__).info(msg, *args)
    except Exception:  # noqa: BLE001 -- logging must never break a step
        pass


def row_bytes(width: int, candidate_blocks, candidate_block_size: int, candidate_write: bool) -> int:
    """Bytes of the temporaries one decode row needs: fp32 logits + the candidate scores (source layer) or flags (consumers)."""
    rb = 4 * int(width)
    if candidate_blocks is not None and candidate_block_size > 0:
        nblocks = -(-int(width) // int(candidate_block_size))
        rb += 4 * nblocks if candidate_write else nblocks + 1
    return rb


def cand_row_bytes(width: int, candidate_block_size: int, candidate_write: bool) -> int:
    """Bytes of the candidate temporaries of one decode row (scope cand): fp32 scores (source layer) or uint8 flags (consumers)."""
    nblocks = -(-int(width) // int(candidate_block_size))
    return 4 * nblocks if candidate_write else nblocks + 1


def plan_chunks(num_rows: int, row_bytes_: int, cap_bytes: int, min_rows: int | None = None):
    """Balanced chunk bounds [(r0, r1), ...] (bigger chunks first), or None = do not chunk.

    Chunks: as few as keep each chunk's temporaries <= cap_bytes, but never a chunk of fewer than `min_rows` rows (a cap below
    min_rows rows of temporaries is raised to min_rows rows) and at least two chunks.
    """
    if min_rows is None:
        min_rows = MIN_CHUNK_ROWS
    min_rows = max(1, int(min_rows))
    if cap_bytes <= 0 or row_bytes_ <= 0 or num_rows < 2 * min_rows:
        return None
    if num_rows * row_bytes_ <= cap_bytes:
        return None
    rows_cap = max(min_rows, cap_bytes // row_bytes_)
    n = min(-(-num_rows // rows_cap), num_rows // min_rows)
    if n < 2:
        return None
    base, extra = divmod(num_rows, n)
    out = []
    r0 = 0
    for i in range(n):
        r1 = r0 + base + (1 if i < extra else 0)
        out.append((r0, r1))
        r0 = r1
    return out


def _capturing() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 -- no CUDA runtime (CPU tests): nothing is capturing
        return False


def _meta_cache(g):
    """Per-forward-context cache of chunk schedules (one dict on the ForwardContext object, which vLLM creates per forward).

    Entries filled by an EAGER call must never serve a graph CAPTURE (the capture would record no schedule kernel and every replay
    would read the eager call's stale values), and vice versa: the dict is dropped whenever the context is used in the other mode.
    Within one capture the first layer of a KV-cache group records the schedule kernels, the later layers read their result."""
    if not SHARE_META:
        return None
    try:
        ctx = g["get_forward_context"]()
        mode = _capturing()
        held = getattr(ctx, "_p043_chunk_meta", None)
        if held is None or held[0] != mode:
            held = (mode, {})
            setattr(ctx, "_p043_chunk_meta", held)
        return held[1]
    except Exception:  # noqa: BLE001 -- no context / frozen object: recompute
        return None


def chunk_schedule(g, cache, decode_metadata, sl_c, r0: int, r1: int, block_size: int, num_sms: int):
    """DeepGEMM schedule metadata for the context lengths of rows [r0, r1) (what the builder computes for the whole batch)."""
    key = (id(decode_metadata), int(sl_c.data_ptr()), r0, r1, block_size, num_sms)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None and hit[0] is decode_metadata:
            return hit[1]
    fn = g.get("get_paged_mqa_logits_metadata")
    if fn is None:
        from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata as fn
    meta = fn(sl_c, block_size, num_sms)
    if cache is not None:
        cache[key] = (decode_metadata, meta)  # keeps decode_metadata alive, so its id() cannot be recycled
    return meta


def _verify_block_size(g, decode_metadata, seq_lens, block_size: int, num_sms: int) -> bool:
    """One-time cross-check (eager calls only: it synchronises) that DeepGEMM's metadata for the WHOLE batch, computed with the block
    size taken from the KV-cache view and the builder's SM count, equals what the builder stored.  Proves the two parameters the
    chunk schedules use are the builder's."""
    key = (block_size, num_sms, int(decode_metadata.schedule_metadata.data_ptr()))  # per builder: its schedule buffer is persistent
    if key in _verified:
        return True
    import torch

    if _capturing():
        return True  # cannot synchronise inside a capture; the eager warm-up of the same graph size has checked it
    fn = g.get("get_paged_mqa_logits_metadata")
    if fn is None:
        from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata as fn
    full = fn(seq_lens, block_size, num_sms)
    stored = decode_metadata.schedule_metadata
    ok = tuple(full.shape) == tuple(stored.shape) and bool(torch.equal(full, stored))
    if ok:
        if len(_verified) > 64:
            _verified.clear()
        _verified.add(key)
    return ok


def _one_chunk(g, r0, r1, q_quant, q_scale, weights, seq_lens, kv_cache, block_table, meta, *, max_model_len, topk_tokens, topk_backend,
               topk_indices_buffer, max_seq_len, candidate_blocks, candidate_block_size, candidate_write):
    """Logits -> candidate select / mask -> top-k for rows [r0, r1).  Its own function so that the chunk's temporaries are
    released when it returns, before the next chunk allocates (inside a graph capture the pool reuses the freed blocks)."""
    sl_c = seq_lens[r0:r1]
    q_c = q_quant[r0:r1]
    qs_c = q_scale[r0:r1] if q_scale is not None else None
    qs_c, w_c = g["mqa_logits_mx_inputs"](q_c, qs_c, weights[r0:r1])
    logits = g["fp8_fp4_paged_mqa_logits"](
        (q_c, qs_c),
        kv_cache,
        w_c,
        sl_c,
        block_table[r0:r1],
        meta,
        max_model_len=max_model_len,
        indices=None,
    )
    # stock: `if topk_backend != "deep_select" or dcp_world_size != 1` -- both true here (those cases never reach this function)
    logits = g["as_fp32_logits"](logits, sl_c.view(-1))
    if candidate_blocks is not None:
        vis = sl_c.reshape(-1)  # stock: row_repeat = 1 because seq_lens has one entry per row
        cand = candidate_blocks[r0:r1]
        if candidate_write:
            g["_select_candidate_blocks"](logits, None, vis, cand.shape[1], candidate_block_size, cand, 1)
        else:
            g["_apply_candidate_mask"](logits, None, vis, cand, candidate_block_size, 1)
    g["get_indexer_topk"](topk_backend)(
        logits, sl_c, 1, topk_indices_buffer[r0:r1, :topk_tokens], topk_tokens, max_seq_len
    )


def _run_cand_only(g, cap, q_quant, q_scale, weights, seq_lens, kv_cache, decode_metadata, *, rows, max_model_len, topk_tokens, topk_backend,
                   topk_indices_buffer, max_seq_len, candidate_blocks, candidate_block_size, candidate_write) -> bool:
    """Scope cand: stock logits (one call, the builder's schedule), candidate select / mask in row chunks, stock top-k over all rows."""
    global CALLS
    if candidate_blocks is None or candidate_block_size <= 0:
        return False
    chunks = plan_chunks(rows, cand_row_bytes(max_model_len, candidate_block_size, candidate_write), cap)
    if chunks is None:
        return False
    _log_once(("cand", int(max_model_len), bool(candidate_write)),
              "%s decode indexer, scope cand (first use: %d rows x width %d): candidate %s in %d chunks of %d..%d rows (cap %d MiB)", TAG, rows,
              int(max_model_len), "scores" if candidate_write else "flags", len(chunks), min(b - a for a, b in chunks), max(b - a for a, b in chunks), cap >> 20)
    qs, dw = g["mqa_logits_mx_inputs"](q_quant, q_scale, weights[:rows])
    logits = g["fp8_fp4_paged_mqa_logits"]((q_quant, qs), kv_cache, dw, seq_lens, decode_metadata.block_table, decode_metadata.schedule_metadata,
                                           max_model_len=max_model_len, indices=None)
    logits = g["as_fp32_logits"](logits, seq_lens.view(-1))
    vis = seq_lens.reshape(-1)
    for r0, r1 in chunks:
        cand = candidate_blocks[r0:r1]
        if candidate_write:
            g["_select_candidate_blocks"](logits[r0:r1], None, vis[r0:r1], cand.shape[1], candidate_block_size, cand, 1)
        else:
            g["_apply_candidate_mask"](logits[r0:r1], None, vis[r0:r1], cand, candidate_block_size, 1)
    g["get_indexer_topk"](topk_backend)(logits, seq_lens, 1, topk_indices_buffer[:rows, :topk_tokens], topk_tokens, max_seq_len)
    CALLS += 1
    return True


def run_decode(g, q_quant, q_scale, weights, seq_lens, kv_cache, decode_metadata, *, next_n, max_model_len, topk_tokens,
               topk_backend, topk_indices_buffer, max_seq_len, candidate_blocks, candidate_block_size, candidate_write,
               needs_padded_path, dcp_world_size) -> bool:
    """Chunked replacement of sparse_attn_indexer's decode block (logits, candidate stage, top-k).  True = every row is done and the
    caller returns; False = nothing was touched and the stock code must run.  `g` = the caller module's globals (the exact function
    objects the stock code would call)."""
    cap = CAP_BYTES
    if cap <= 0 or needs_padded_path or dcp_world_size != 1 or next_n != 1:
        return False
    if getattr(decode_metadata, "global_seq_lens", None) is not None or getattr(decode_metadata, "indices", None) is not None:
        return False
    if g["current_platform"].is_xpu():
        return False
    if topk_backend == "deep_select":  # bf16 logits on SM100: the stock code upcasts only below the row ends
        return False
    if q_quant.dim() != 4 or int(q_quant.shape[1]) != 1:
        return False
    rows = int(q_quant.shape[0])
    if seq_lens.dim() != 2 or tuple(seq_lens.shape) != (rows, 1) or int(weights.shape[0]) < rows:
        return False
    if candidate_blocks is not None and candidate_block_size <= 0:
        return False
    if SCOPE == "cand":
        return _run_cand_only(g, cap, q_quant, q_scale, weights, seq_lens, kv_cache, decode_metadata, rows=rows, max_model_len=max_model_len,
                              topk_tokens=topk_tokens, topk_backend=topk_backend, topk_indices_buffer=topk_indices_buffer, max_seq_len=max_seq_len,
                              candidate_blocks=candidate_blocks, candidate_block_size=candidate_block_size, candidate_write=candidate_write)
    if topk_backend in _UNSUPPORTED_BACKENDS:
        return False
    if topk_backend == "auto" and getattr(g["get_indexer_topk"](topk_backend), "_cooperative_capable", False):
        return False  # "auto" would pick cooperative_topk for chunks of <= 64 rows but persistent_topk for the whole batch
    chunks = plan_chunks(rows, row_bytes(max_model_len, candidate_blocks, candidate_block_size, candidate_write), cap)
    if chunks is None:
        return False

    block_size = int(kv_cache.shape[1])
    nsm = g.get("num_compute_units")
    if nsm is None:
        from vllm.utils.platform_utils import num_compute_units as nsm
    num_sms = int(nsm(q_quant.device.index))
    if int(decode_metadata.schedule_metadata.shape[0]) != num_sms + 1:
        _log_once(("sched", num_sms), "%s stock path: schedule metadata has %d slots, builder SM count %d", TAG,
                  int(decode_metadata.schedule_metadata.shape[0]), num_sms)
        return False

    global _disabled, CALLS
    if _disabled:
        return False
    if not _verify_block_size(g, decode_metadata, seq_lens, block_size, num_sms):
        _disabled = True
        _log_once("verify", "%s stock path for good: the whole-batch schedule recomputed with block size %d / %d SMs differs from the "
                  "builder's -- the chunk schedules would not match the kernel", TAG, block_size, num_sms)
        return False
    _log_once(("run", int(max_model_len), candidate_blocks is not None, bool(candidate_write)),  # once per kind, not per graph size
              "%s decode indexer chunked (first use: %d rows x width %d%s) -> %d chunks of %d..%d rows (cap %d MiB, schedule %s)", TAG, rows,
              int(max_model_len), (" + candidate " + ("scores" if candidate_write else "flags")) if candidate_blocks is not None else "",
              len(chunks), min(b - a for a, b in chunks), max(b - a for a, b in chunks), cap >> 20,
              "shared per forward" if SHARE_META else "per layer")
    cache = _meta_cache(g)
    for r0, r1 in chunks:
        meta = chunk_schedule(g, cache, decode_metadata, seq_lens[r0:r1], r0, r1, block_size, num_sms)
        _one_chunk(g, r0, r1, q_quant, q_scale, weights, seq_lens, kv_cache, decode_metadata.block_table, meta,
                   max_model_len=max_model_len, topk_tokens=topk_tokens, topk_backend=topk_backend,
                   topk_indices_buffer=topk_indices_buffer, max_seq_len=max_seq_len, candidate_blocks=candidate_blocks,
                   candidate_block_size=candidate_block_size, candidate_write=candidate_write)
    CALLS += 1
    return True
