# SPDX-License-Identifier: Apache-2.0
"""GB10 split configs for the fused mHC post + pre-norm GEMM at decode token counts (patch 025; installed as vllm/_mhc_split.py).

WHY: mhc_fused_post_pre_split_config() picks (tile_n, n_splits, n_thr) from GB300 tuning: (2, 8, 128) below 16 tokens. On GB10
(48 SMs) that op runs at about half the DRAM rate at one user (16.8 us per call, 82 calls per decode step; it streams ~2 MB of
F32 projection per call). A sweep on the real op (tests/025/mhc_split_bench.py) measures the candidates.

WHAT: VLLM_GB10_MHC_SPLIT="6=4,8,128/12=6,8,128" (";" also accepted) maps exact token counts to configs; every other count keeps the upstream
choice. A config is used only if it is valid for the shape (hidden % (n_splits * n_thr) == 0, (hc*(hc+2)) % tile_n == 0);
otherwise the upstream choice stands (logged once). The same function feeds the dispatch, the compile key, the warmup and the
launch, so they stay consistent. Same math; only the fp32 accumulation order of the k-split partials changes.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENV = "VLLM_GB10_MHC_SPLIT"
_TABLE: dict[int, tuple[int, int, int]] | None = None
_LOGGED: set = set()


def _table() -> dict[int, tuple[int, int, int]]:
    global _TABLE
    if _TABLE is None:
        t: dict[int, tuple[int, int, int]] = {}
        raw = os.environ.get(_ENV, "").strip()
        for item in filter(None, (s.strip() for s in raw.replace("/", ";").split(";"))):  # "/" is shell-safe in recipes
            k, v = item.split("=")
            tile_n, n_splits, n_thr = (int(z) for z in v.split(","))
            t[int(k)] = (tile_n, n_splits, n_thr)
        _TABLE = t
        if t:
            logger.info("[patch-025] GB10 fused-mHC split configs: %s", t)
    return _TABLE


def override(num_tokens: int, hidden_size: int, hc_mult: int) -> tuple[int, int, int] | None:
    cfg = _table().get(num_tokens)
    if cfg is None:
        return None
    tile_n, n_splits, n_thr = cfg
    if hidden_size % (n_splits * n_thr) or (hc_mult * (hc_mult + 2)) % tile_n:
        if (num_tokens, hidden_size) not in _LOGGED:
            _LOGGED.add((num_tokens, hidden_size))
            logger.warning("[patch-025] %s for %d tokens does not fit hidden %d / hc %d: upstream config kept", cfg,
                           num_tokens, hidden_size, hc_mult)
        return None
    return cfg
