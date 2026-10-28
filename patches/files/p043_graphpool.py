# SPDX-License-Identifier: Apache-2.0
"""Patch 043 -- shrink the CUDA-graph pool of the DeepSeek-V4.1 decode graphs (two independent opt-in levers).

Measured / derived on 2026-10-28 (pool analysis of the same day): every rank's graph pool holds 3.50 GiB, almost all of it the
decode sparse indexer's full-width fp32 logits inside the 768-token decode graph (DeepGEMM allocates [rows, max_model_len //
compress_ratio] fp32: 3 GiB at ratio 1) plus the candidate scores (384 MiB, layer 20) / flags (96 MiB); a second pool of 0.29 GiB
belongs to the DSpark drafter graphs, which cannot reuse the main graphs' free blocks because the caching allocator keys the free
blocks of a shared pool by stream and every graph_capture(device) call opens a NEW capture stream.  Pool memory is lost to the KV cache.

Lever (a)  VLLM_GB10_INDEXER_DECODE_CHUNK_MB=<cap MiB>   (model_executor/layers/sparse_attn_indexer.py + vllm/_indexer_decode_chunk.py)
  The decode block of sparse_attn_indexer runs, for the flattened layout, in row chunks whose logits + candidate temporaries stay
  below the cap; per chunk: DeepGEMM schedule metadata for the chunk's context lengths (shared between the layers of one KV-cache
  group inside a forward), paged-MQA logits, candidate select / mask, top-k into the caller's slice; the chunk's tensors are freed
  before the next chunk (the graph pool reuses them).  Graphs whose unchunked temporaries fit the cap, and every layout that is not
  the flattened one-row-per-token decode (padded/packed, varlen, DCP, bf16 logits, cooperative / deep_select top-k), run the stock
  code.  Exactness argument and the points the GPU test must confirm: see the docstring of files/indexer_decode_chunk.py.
  Edits: (1) one import line after `logger = init_logger(__name__)`; (2) a guarded early return in front of the decode block's
  `if current_platform.is_xpu():` (the code after it is untouched: when the chunked path has processed every row the only things
  left in the stock function are branches that are provably skipped -- no DCP merge, no unpack -- and `return topk_indices_buffer`).

Lever (b)  VLLM_GB10_SHARED_CAPTURE_STREAM=1   (distributed/parallel_state.py, module-level graph_capture(device) only)
  One cached torch.cuda.Stream per device is used for every graph_capture(device) context instead of a fresh stream per call, so the
  graphs of the DSpark drafter (a separate CudaGraphManager, same global pool id) are captured on the main graphs' stream and can
  reuse the blocks the main graphs left free (expected -0.29 GiB).  Patch 018 edits GroupCoordinator.graph_capture (three anchors with
  8/12-space indentation inside the class: maybe_gb10_roce_context); this patch touches only the module-level function and
  its own anchors do not overlap 018's, but it is pinned to the file as 018 leaves it.

Both levers are inert unless their variable is set: without them the stock calls are made (lever b evaluates the stock
`torch.cuda.Stream(device=device)`; lever a's hook is `if _p043_chunk.CAP_BYTES and ...`, i.e. one attribute read).
"""
ID = "043"
TAG = "graphpool"
M = f"# [patch-{ID} {TAG}]"

# model_executor/layers/sparse_attn_indexer.py: pristine image file == the file after every patch of patches-i063 (028 pins it as a
# read-only dependency and hashes only PrefillRowShard.exchange_topk / balanced_prefill_row_shard at run time; no patch edits it)
SAI_SHA = "ed9cfdd83544d15c088b052e287971610d0de96062493b6c757d16ea2248c6aa"
# distributed/parallel_state.py after patch 018 (pristine 862b7220.. + 018's three GroupCoordinator.graph_capture edits)
PS_SHA = "a048d45869b27c3869fa1231211596f234b6eab69e2fcaf069d4f99dcbb16b72"

_SAI = "model_executor/layers/sparse_attn_indexer.py"
_PS = "distributed/parallel_state.py"

# ---- lever (a) ---------------------------------------------------------------------------------------------------------------
OLD_IMPORT = (
    "logger = init_logger(__name__)\n"
    "\n"
    "# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.\n"
)
NEW_IMPORT = (
    "logger = init_logger(__name__)\n"
    "\n"
    f"{M} VLLM_GB10_INDEXER_DECODE_CHUNK_MB: row-chunked decode logits / candidates / top-k (vllm/_indexer_decode_chunk.py); inert when unset\n"
    "from vllm import _indexer_decode_chunk as _p043_chunk  # noqa: E402\n"
    "\n"
    "# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.\n"
)

OLD_HOOK = (
    "        if current_platform.is_xpu():\n"
    "            if padded_q_scale is not None:\n"
    "                raise RuntimeError(\"XPU fp8_paged_mqa_logits does not support FP4 Q\")\n"
)
NEW_HOOK = (
    f"        {M} flattened decode rows in chunks: logits, candidate select/mask and top-k per chunk; True = all rows done\n"
    "        if _p043_chunk.CAP_BYTES and _p043_chunk.run_decode(\n"
    "            globals(),\n"
    "            padded_q_quant_cast,\n"
    "            padded_q_scale,\n"
    "            weights,\n"
    "            seq_lens,\n"
    "            kv_cache,\n"
    "            decode_metadata,\n"
    "            next_n=next_n,\n"
    "            max_model_len=max_model_len,\n"
    "            topk_tokens=topk_tokens,\n"
    "            topk_backend=topk_backend,\n"
    "            topk_indices_buffer=topk_indices_buffer,\n"
    "            max_seq_len=attn_metadata_narrowed.max_seq_len,\n"
    "            candidate_blocks=candidate_blocks,\n"
    "            candidate_block_size=candidate_block_size,\n"
    "            candidate_write=candidate_write,\n"
    "            needs_padded_path=needs_padded_path,\n"
    "            dcp_world_size=dcp_world_size,\n"
    "        ):\n"
    "            return topk_indices_buffer\n"
) + OLD_HOOK

# ---- lever (b) ---------------------------------------------------------------------------------------------------------------
OLD_DEF = (
    "@contextmanager\n"
    "def graph_capture(device: torch.device):\n"
)
NEW_DEF = (
    f"{M} VLLM_GB10_SHARED_CAPTURE_STREAM=1: one capture stream per device for every graph_capture(device) caller, so graphs captured by\n"
    "# different CudaGraphManagers (main model, DSpark drafter) share the free blocks of the common graph pool (the allocator keys them by stream)\n"
    "import os as _p043_os  # noqa: E402\n"
    "\n"
    "_P043_SHARED_STREAM = _p043_os.environ.get(\"VLLM_GB10_SHARED_CAPTURE_STREAM\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\", \"on\")\n"
    "_P043_STREAMS: dict = {}\n"
    "\n"
    "\n"
    "def _p043_capture_stream(device):\n"
    "    dev = torch.device(device)\n"
    "    key = (dev.type, dev.index if dev.index is not None else torch.cuda.current_device())\n"
    "    stream = _P043_STREAMS.get(key)\n"
    "    if stream is None:\n"
    "        stream = _P043_STREAMS[key] = torch.cuda.Stream(device=device)\n"
    "    return stream\n"
    "\n"
    "\n"
) + OLD_DEF

OLD_CTX = "    context = GraphCaptureContext(torch.cuda.Stream(device=device))\n"
NEW_CTX = (
    f"    {M} VLLM_GB10_SHARED_CAPTURE_STREAM=1: the cached capture stream of this device (stock: a new stream per call)\n"
    "    context = GraphCaptureContext(\n"
    "        _p043_capture_stream(device) if _P043_SHARED_STREAM else torch.cuda.Stream(device=device)\n"
    "    )\n"
)

EDITS = [
    (_SAI, SAI_SHA, OLD_IMPORT, NEW_IMPORT),
    (_SAI, SAI_SHA, OLD_HOOK, NEW_HOOK),
    (_PS, PS_SHA, OLD_DEF, NEW_DEF),
    (_PS, PS_SHA, OLD_CTX, NEW_CTX),
]
