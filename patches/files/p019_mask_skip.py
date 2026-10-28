# SPDX-License-Identifier: Apache-2.0
"""Patch 019 -- skip the candidate-mask tiles past each row's end (applied by xpatch.py).

WHY: DeepSeek-V4.1's two-level sparse indexer masks the indexer logits to the candidate blocks before the row top-k
(_mask_candidates_kernel). The logits buffer is max_model_len wide (CUDA-graph static), and the kernel writes -inf to
every column outside [start, end) -- at max_model_len 1M that is ~4 MB of writes per decode row, 4 calls per step:
473 us per step at one user in b023 (1.4 % of the step), for a 300-token context.

EXACT: every consumer of the masked logits reads only [start, end) of a row -- the decode top-k backends clamp to the
row length (persistent / per-row / cooperative / deep_select / flashinfer take seq_lens or row ends), the prefill
top_k_per_row_prefill takes row starts and ends, and as_fp32_logits upcasts only below row_ends ("the span the
consumers read"). A tile that starts at or past the row's end therefore holds nothing a consumer sees; it now returns
without writing. Columns inside [start, end) are computed exactly as before (bit-identical), including the
last-column edge case, which can only matter when end == width.
"""
ID = "019"
TAG = "mask-skip"
M = f"# [patch-{ID} {TAG}]"

CB_SHA = "604f588fcf96d546fdf230a7b225384a27952acedbcefbd7ea5c181f18e5c621"

OLD = """    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
"""
NEW = """    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    """ + M + """ consumers read only [start, end): a tile at or past the row's end has nothing to mask
    if tl.program_id(1) * TILE >= end:
        return
    valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
"""

EDITS = [("model_executor/kernels/attention/dsa/candidate_blocks.py", CB_SHA, OLD, NEW)]
