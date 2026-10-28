# SPDX-License-Identifier: Apache-2.0
"""Patch 037 -- M64 split kernels for large launches on b12x's compact W4A8-MX layouts (DSV4.1-Flash TP8 prefill).

Applies on top of patch 036 (the files it pins are the 036-patched ones; fail-closed otherwise).

b12x's compact W4A8-MX path (n % 128 == 64: the padded 320 of patch 033; n % 128 == 32 with patch 036: the native 288) runs every
dynamic launch (M >= 9 tokens) on the split-materialized M16 kernels: a front-end routing kernel + phase 1 (FC1) + phase 2 (FC2),
tile M = 16, grouped routing -- _heuristic_moe_decode_config pins dynamic_tile_m = 16 and validate_moe_decode_config refuses anything
else.  The ceil-tiled stock path instead moves to the M64 split kernels once the average routed rows per expert reach 36
(_select_dynamic_tile_mn), because at 128 rows per expert (an 8192-token prefill chunk) an M16 tile re-reads the expert's weight
tile 8 times and dequantizes it for 16 rows only.  The phase kernels (W4A8MaterializedPhase1/Phase2Kernel) take source_tile_m
16 / 64 / 128 AND the compact layout flags (n64_repacked, n_tail_rows) independently; only the policy tied them to M16.

What changes (opt-in; without B12X_COMPACT_PREFILL_TILE_M every decision, query digest and kernel argument is the old one):
  * b12x._lib.env: compact_prefill_tile_m() (B12X_COMPACT_PREFILL_TILE_M: 0 / unset = off, 64), compact_prefill_min_rows_per_expert()
    (B12X_COMPACT_PREFILL_MIN_ROWS_PER_EXPERT, default 36 = the stock M32 -> M64 crossover; 0 = every dynamic launch) and
    w4a8_compact_dynamic_tile_m(routed_rows, num_experts) -> 16 or 64.
  * _impl._heuristic_moe_decode_config: a compact dynamic launch takes that tile (grouped routing as before); plan, scratch sizing and
    the kernel build already derive everything from the planned tile (the same path the stock M64 launches use).
  * _tuning.validate_moe_decode_config: compact queries accept dynamic_tile_m 16, and 64 when the variable is set.
  * _preparation._control_snapshot: the setting joins the query controls only when set (stock selection-cache digests unchanged).
Nothing in the kernels changes.  The variable is read at every call (plan time); launches below the threshold, the compact micro
path (M <= 8) and CUDA-graph decode sizes keep their kernels.  Changing b12x sources changes the package fingerprint: one more
cold JIT of the b12x kernels after installing.

Apply:  python3 xpatch.py apply p037_b12x_compact_prefill.py /usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x
"""
ID = "037"
TAG = "b12x-compact-prefill"
M = f"# [patch-{ID} {TAG}]"

# sha256 of the files as left by patch 036 (b12x-n32) on image dsv41-flash-sm121:n1 (vLLM 0.32.1rc1.dev466+ge4a5b5537, b12x = flashinfer/experimental/b12x)
SHA_LIB_ENV = "60a0d3664ebaf639182385c4170c0de11eb77739b748c5944be8fece332ef4db"
SHA_MOE_FUSED_MOE_IMPL = "da547275399314af080d07fa4e8cc67a7bbcdedfb7b7ce11b853ed46deb0657a"
SHA_MOE_FUSED_MOE_PREPARATION = "31be17b53c9e372b60cb7832adb150e0b913ab38d76913ad1a8bbac80c3a0cf2"
SHA_MOE_FUSED_MOE_TUNING = "3d7e69d1ae251ad8268310391e60c778761e4cb4ee95be00847f496351c7f108"

# --- _lib/env.py (1/2)
OLD_LIB_ENV_1 = r'''
__all__ = [
'''
NEW_LIB_ENV_1 = r'''
# [patch-037 b12x-compact-prefill] Large compact W4A8-MX launches on the M64 split kernels (opt-in, size-gated).
COMPACT_PREFILL_TILES = (64,)
COMPACT_PREFILL_DEFAULT_MIN_ROWS_PER_EXPERT = 36


def compact_prefill_tile_m() -> int:
    """B12X_COMPACT_PREFILL_TILE_M: M tile of the compact W4A8-MX split kernels (phase 1 / phase 2) for large launches.

    Unset, empty, 0 or false: off -- compact launches keep the M16 split kernels, the stock policy.  64: launches whose average
    routed rows per expert reach B12X_COMPACT_PREFILL_MIN_ROWS_PER_EXPERT use the M64 split kernels (the tile the stock
    ceil-tiled path picks for dense prefill).  Anything else is an error (fail closed).  Read at every call.
    """
    raw = env_raw("COMPACT_PREFILL_TILE_M")
    if raw is None:
        return 0
    text = raw.strip().lower()
    if text in {"", "0", "false", "no", "off"}:
        return 0
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value not in COMPACT_PREFILL_TILES:
        raise ValueError(
            f"B12X_COMPACT_PREFILL_TILE_M must be 0 or one of {COMPACT_PREFILL_TILES}, got {raw!r}"
        )
    return value


def compact_prefill_min_rows_per_expert() -> int:
    """B12X_COMPACT_PREFILL_MIN_ROWS_PER_EXPERT: average routed rows per expert from which the prefill tile applies.

    Default 36, the stock M32 -> M64 crossover of the dynamic W4A8 tile ladder (_select_dynamic_tile_mn).  0 applies the
    prefill tile to every dynamic compact launch (benchmarking).  The comparison is routed_rows >= min_rows * num_experts, so
    DSV4.1-Flash TP8 (384 experts, top-6) crosses at 36 * 384 / 6 = 2304 tokens, its drafter (128 experts, top-3) at 1536.
    """
    raw = env_raw("COMPACT_PREFILL_MIN_ROWS_PER_EXPERT")
    if raw is None or not raw.strip():
        return COMPACT_PREFILL_DEFAULT_MIN_ROWS_PER_EXPERT
    try:
        value = int(raw.strip())
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(
            f"B12X_COMPACT_PREFILL_MIN_ROWS_PER_EXPERT must be a non-negative integer, got {raw!r}"
        )
    return value


def w4a8_compact_dynamic_tile_m(routed_rows: int, num_experts: int) -> int:
    """Dynamic M tile of a compact W4A8-MX launch: 16 (stock), or the prefill tile when enabled and the launch is dense enough."""
    tile = compact_prefill_tile_m()
    if tile and int(routed_rows) >= compact_prefill_min_rows_per_expert() * max(1, int(num_experts)):
        return tile
    return 16


__all__ = [
'''

# --- _lib/env.py (2/2)
OLD_LIB_ENV_2 = r'''    "w4a8_compact_tail_rows",
]
'''
NEW_LIB_ENV_2 = r'''    "w4a8_compact_tail_rows",
    "compact_prefill_tile_m",  # [patch-037 b12x-compact-prefill]
    "compact_prefill_min_rows_per_expert",  # [patch-037 b12x-compact-prefill]
    "w4a8_compact_dynamic_tile_m",  # [patch-037 b12x-compact-prefill]
]
'''

# --- moe/fused_moe/_impl.py (1/2)
OLD_MOE_FUSED_MOE_IMPL_1 = r'''from b12x._lib.env import w4a8_compact_tail_rows  # [patch-036 b12x-n32]
from b12x._lib.compiler import (
'''
NEW_MOE_FUSED_MOE_IMPL_1 = r'''from b12x._lib.env import w4a8_compact_tail_rows  # [patch-036 b12x-n32]
from b12x._lib.env import w4a8_compact_dynamic_tile_m  # [patch-037 b12x-compact-prefill]
from b12x._lib.compiler import (
'''

# --- moe/fused_moe/_impl.py (2/2)
OLD_MOE_FUSED_MOE_IMPL_2 = r'''            # The direct compact micro path owns one through eight live tokens.
            dynamic_tile_m = 16
            dynamic_route_mode = "grouped"
'''
NEW_MOE_FUSED_MOE_IMPL_2 = r'''            # The direct compact micro path owns one through eight live tokens.
            # [patch-037 b12x-compact-prefill] ... or the M64 ones when B12X_COMPACT_PREFILL_TILE_M=64 and the launch is dense
            dynamic_tile_m = w4a8_compact_dynamic_tile_m(query.routed_rows, query.num_experts)
            dynamic_route_mode = "grouped"
'''

# --- moe/fused_moe/_preparation.py (1/2)
OLD_MOE_FUSED_MOE_PREPARATION_1 = r'''from b12x._lib.env import compact_n32_enabled, w4a8_compact_tail_rows  # [patch-036 b12x-n32]
from b12x._lib.program_cache import program_cache
'''
NEW_MOE_FUSED_MOE_PREPARATION_1 = r'''from b12x._lib.env import compact_n32_enabled, w4a8_compact_tail_rows  # [patch-036 b12x-n32]
from b12x._lib.env import compact_prefill_min_rows_per_expert, compact_prefill_tile_m  # [patch-037 b12x-compact-prefill]
from b12x._lib.program_cache import program_cache
'''

# --- moe/fused_moe/_preparation.py (2/2)
OLD_MOE_FUSED_MOE_PREPARATION_2 = r'''            **({"w4a8_compact_n32": True} if compact_n32_enabled() else {}),
        }
'''
NEW_MOE_FUSED_MOE_PREPARATION_2 = r'''            **({"w4a8_compact_n32": True} if compact_n32_enabled() else {}),
            # [patch-037 b12x-compact-prefill] only when set (same reason): the tile policy is part of the query contract
            **(
                {"w4a8_compact_prefill": (compact_prefill_tile_m(), compact_prefill_min_rows_per_expert())}
                if compact_prefill_tile_m()
                else {}
            ),
        }
'''

# --- moe/fused_moe/_tuning.py (1/3)
OLD_MOE_FUSED_MOE_TUNING_1 = r'''from b12x._lib.env import w4a8_compact_tail_rows  # [patch-036 b12x-n32]
from b12x._lib.quant.block_codec import BLOCK_CODECS, block_codec
'''
NEW_MOE_FUSED_MOE_TUNING_1 = r'''from b12x._lib.env import w4a8_compact_tail_rows  # [patch-036 b12x-n32]
from b12x._lib.env import compact_prefill_tile_m  # [patch-037 b12x-compact-prefill]
from b12x._lib.quant.block_codec import BLOCK_CODECS, block_codec
'''

# --- moe/fused_moe/_tuning.py (2/3)
OLD_MOE_FUSED_MOE_TUNING_2 = r'''        and config.backend == "dynamic"
        and (config.dynamic_tile_m != 16 or config.dynamic_route_mode != "grouped")
    ):
'''
NEW_MOE_FUSED_MOE_TUNING_2 = r'''        and config.backend == "dynamic"
        # [patch-037 b12x-compact-prefill] M64 too when B12X_COMPACT_PREFILL_TILE_M=64 (compact_prefill_tile_m() is 0 otherwise)
        and (
            config.dynamic_tile_m not in (16, compact_prefill_tile_m())
            or config.dynamic_route_mode != "grouped"
        )
    ):
'''

# --- moe/fused_moe/_tuning.py (3/3)
OLD_MOE_FUSED_MOE_TUNING_3 = r'''            "compact N64 W4A8 dynamic execution requires grouped M16 routing"
        )
'''
NEW_MOE_FUSED_MOE_TUNING_3 = r'''            "compact N64 W4A8 dynamic execution requires grouped M16 routing"
            + (" or M64" if compact_prefill_tile_m() else "")  # [patch-037 b12x-compact-prefill]
        )
'''

EDITS = [
    ("_lib/env.py", SHA_LIB_ENV, OLD_LIB_ENV_1, NEW_LIB_ENV_1),
    ("_lib/env.py", SHA_LIB_ENV, OLD_LIB_ENV_2, NEW_LIB_ENV_2),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_1, NEW_MOE_FUSED_MOE_IMPL_1),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_2, NEW_MOE_FUSED_MOE_IMPL_2),
    ("moe/fused_moe/_preparation.py", SHA_MOE_FUSED_MOE_PREPARATION, OLD_MOE_FUSED_MOE_PREPARATION_1, NEW_MOE_FUSED_MOE_PREPARATION_1),
    ("moe/fused_moe/_preparation.py", SHA_MOE_FUSED_MOE_PREPARATION, OLD_MOE_FUSED_MOE_PREPARATION_2, NEW_MOE_FUSED_MOE_PREPARATION_2),
    ("moe/fused_moe/_tuning.py", SHA_MOE_FUSED_MOE_TUNING, OLD_MOE_FUSED_MOE_TUNING_1, NEW_MOE_FUSED_MOE_TUNING_1),
    ("moe/fused_moe/_tuning.py", SHA_MOE_FUSED_MOE_TUNING, OLD_MOE_FUSED_MOE_TUNING_2, NEW_MOE_FUSED_MOE_TUNING_2),
    ("moe/fused_moe/_tuning.py", SHA_MOE_FUSED_MOE_TUNING, OLD_MOE_FUSED_MOE_TUNING_3, NEW_MOE_FUSED_MOE_TUNING_3),
]
