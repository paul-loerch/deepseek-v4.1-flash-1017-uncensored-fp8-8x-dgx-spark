# SPDX-License-Identifier: Apache-2.0
"""Patch 036 -- native compact W4A8-MX path for intermediate widths with n % 128 == 32 (DSV4.1-Flash TP8: 2304 / 8 = 288).

b12x stores a W4A8-MX expert slice whose width is not a multiple of 128 "ceil-tiled" (288 -> 384, +33 %) and launches its
dynamic kernel at the padded width; n % 128 == 64 (320) selects the exact compact N64 path instead (exact storage, bounded
reads, micro-compact kernels for M <= 8, split front-end + phase1 + phase2 M16 kernels above).  This patch makes the compact
path accept n % 128 == 32 as well, with NO padding: 288 is stored and read exactly (2.35 MB per expert and rank instead of
2.61 MB at 320 or 3.13 MB ceil-tiled).

What changes (opt-in with B12X_COMPACT_N32=1 -- without it every decision is the old one, the kernels see identical arguments):
  * one predicate, b12x._lib.env.w4a8_compact_tail_rows(n): 64 for n % 128 == 64, 32 for n % 128 == 32 when the variable is
    set, else 0.  It replaces the `n % 128 == 64` tests of weight preparation, planning, scratch sizing and launch
    (_impl.py x6: micro policy, decode config, micro scratch, prepare, plan, workspace check; _preparation.py x2; _tuning.py x3)
    so prepare, plan, scratch sizing, micro-vs-dynamic policy and launch agree.  The variable is read at every call; a guard in
    b12x_moe_fp4 raises if the weights' compact layout and the variable disagree (it must not change between prepare and run),
    and it joins the query controls only when set (stock query/selection-cache digests stay byte-identical).
  * validation widened from 64 to 32 where the layout is exact for a 32-row tail (the in-place repack of weights and E8M0
    scales: _impl.py, the three compact micro kernels, the dynamic kernel ctor).
  * kernels: the tail tile is parametric.  Projection (FC1, micro) and phase 1 (FC1, split path) stage a last N128 tile of
    32 rows (one N32 chunk; the other three chunks are zero-filled in shared memory, exactly like the 64-row tail's missing
    two); phase 2 (FC2, both paths) gets a K tail of ONE K32 block (the last K128 slice carries 1 instead of 2 live K32
    blocks, the other three are zero in shared memory: weights, scales and the MXFP8 activation pad).  The compute loops are
    untouched: zero blocks multiply to exact +0.0.
  * MXFP4-CSF A8 preparation (inline scale tiles, 64-row tails only) refuses the new width instead of mis-laying it out.

Applied by xpatch.py to the b12x package (root = .../dist-packages/flashinfer/experimental/b12x):
  python3 xpatch.py apply p036_b12x_n32.py /usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x
Changing ANY b12x source file changes the package fingerprint that keys the CuTe object cache: after installing this patch
every b12x kernel recompiles once (cold JIT on the first boot / first call).
"""
ID = "036"
TAG = "b12x-n32"
M = f"# [patch-{ID} {TAG}]"

# sha256 of the unpatched files of image dsv41-flash-sm121:n1 (vLLM 0.32.1rc1.dev466+ge4a5b5537, b12x = flashinfer/experimental/b12x)
SHA_LIB_ENV = "5e7af97f1427917b8e840040b35b558e64882be9c0404e999d5bd0e1680c069d"
SHA_MOE_SHARED_KERNELS_DYNAMIC = "60fe09636c3892bf7af2e55e7992dca51e7743d334909a47374dce787ee14369"
SHA_MOE_SHARED_KERNELS_W4A8_COMPACT_ACTIVATION = "1b480da749392fc91d598f70ccbdddf3594394530071c107da4e4a145d9780bb"
SHA_MOE_SHARED_KERNELS_W4A8_COMPACT_MICRO = "e574f77448980465bc35572dc63c15ceec09020677a1c11fb586bcfd82a1b3df"
SHA_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION = "cc90193f8cce960c55df27644175986c174f6b6b272385ae7fc3de2ef0b60ffe"
SHA_MOE_SHARED_KERNELS_W4A8_PHASE1 = "e59aef6a5f2f29794f49e85e13ab5cab87cdb54767fd13e0d94d631543c0e7d5"
SHA_MOE_SHARED_KERNELS_W4A8_PHASE2 = "d8607e4c951908d0c51887c5baafa9ea7abc6603e6ade4bf297a5a6194e2ca38"
SHA_MOE_FUSED_MOE_IMPL = "8329746c3eb8b4de751e256d4a8a2a5ede7a2e11be58659080a1d1103ea24d3c"
SHA_MOE_FUSED_MOE_PREPARATION = "713b00af5c29a229be64a871e71c9c5f4ed785d6dea15b57a5d8328b65b00759"
SHA_MOE_FUSED_MOE_TUNING = "c29319c6bfa0b39de0188801b8fbd6fc1364572c09ea6aeb870e862095d9bf6a"
SHA_MOE_FUSED_MOE_PLANNING = "b3a14ca01531e91d1a3be5ca2bb8876788f2f1382aac57eab389ac2f70abafaa"

# --- _lib/env.py (1/1)
OLD_LIB_ENV_1 = r'''
__all__ = ["PREFIX", "env_raw", "env_flag"]
'''
NEW_LIB_ENV_1 = r'''
# [patch-036 b12x-n32] Native compact W4A8-MX path for intermediate widths with n % 128 == 32 (TP8 of 2304 = 288).
def compact_n32_enabled() -> bool:
    """B12X_COMPACT_N32=1: widths with n % 128 == 32 take the exact compact path (read at every call)."""
    return env_flag("COMPACT_N32")


# [patch-036 b12x-n32] The one predicate behind every `n % 128 == 64` decision of the compact W4A8-MX path.
def w4a8_compact_tail_rows(n: int) -> int:
    """Rows of the compact tail tile (n % 128) when the width takes b12x's exact compact W4A8-MX path, else 0.

    n % 128 == 64 always did (compact N64: exact storage, bounded reads, micro-compact for M <= 8, M16 split kernels
    above).  n % 128 == 32 does with B12X_COMPACT_N32=1; every other width keeps the ceil-tiled N128 storage.  Decisions
    are made at prepare, plan and launch time, so the variable must not change between preparing the weights and
    running them.
    """
    tail = int(n) % 128
    if tail == 64 or (tail == 32 and compact_n32_enabled()):
        return tail
    return 0


__all__ = [
    "PREFIX",
    "env_raw",
    "env_flag",
    "compact_n32_enabled",
    "w4a8_compact_tail_rows",
]
'''

# --- moe/_shared/kernels/dynamic.py (1/3)
OLD_MOE_SHARED_KERNELS_DYNAMIC_1 = r'''        self.w4a8_n64_tail = bool(w4a8_n64_tail)
        if self.w4a8_n64_repacked and not self.w4a8_repacked:
'''
NEW_MOE_SHARED_KERNELS_DYNAMIC_1 = r'''        self.w4a8_n64_tail = bool(w4a8_n64_tail)
        # [patch-036 b12x-n32] True keeps meaning a 64-row tail; an int is the tail width in rows (32 or 64)
        self.w4a8_n_tail_rows = (
            64 if (w4a8_n64_tail is True or not w4a8_n64_tail) else int(w4a8_n64_tail)
        )
        if self.w4a8_n64_tail and self.w4a8_n_tail_rows not in (32, 64):
            raise ValueError(
                f"w4a8_n64_tail must be a 32- or 64-row tail, got {w4a8_n64_tail!r}"
            )
        if self.w4a8_n64_repacked and not self.w4a8_repacked:
'''

# --- moe/_shared/kernels/dynamic.py (2/3)
OLD_MOE_SHARED_KERNELS_DYNAMIC_2 = r'''                num_topk=self.num_topk,
                n64_repacked=self.w4a8_n64_repacked,
                n64_tail=self.w4a8_n64_tail,
                csf_inline=bool(w4a8_csf_inline),
                trellis_bits=(
                    trellis_bits
'''
NEW_MOE_SHARED_KERNELS_DYNAMIC_2 = r'''                num_topk=self.num_topk,
                n64_repacked=self.w4a8_n64_repacked,
                n64_tail=self.w4a8_n64_tail,
                n_tail_rows=self.w4a8_n_tail_rows,  # [patch-036 b12x-n32]
                csf_inline=bool(w4a8_csf_inline),
                trellis_bits=(
                    trellis_bits
'''

# --- moe/_shared/kernels/dynamic.py (3/3)
OLD_MOE_SHARED_KERNELS_DYNAMIC_3 = r'''                deterministic_output=bool(deterministic_output),
                n64_repacked=self.w4a8_n64_repacked,
                n64_tail=self.w4a8_n64_tail,
                csf_inline=bool(w4a8_csf_inline),
                trellis_bits=(
                    trellis_bits
'''
NEW_MOE_SHARED_KERNELS_DYNAMIC_3 = r'''                deterministic_output=bool(deterministic_output),
                n64_repacked=self.w4a8_n64_repacked,
                n64_tail=self.w4a8_n64_tail,
                n_tail_rows=self.w4a8_n_tail_rows,  # [patch-036 b12x-n32]
                csf_inline=bool(w4a8_csf_inline),
                trellis_bits=(
                    trellis_bits
'''

# --- moe/_shared/kernels/w4a8_compact_activation.py (1/1)
OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_ACTIVATION_1 = r'''        self.fast_math = bool(fast_math)
        if self.n < 64 or self.n % 64:
            raise ValueError("compact W4A8 micro activation requires N divisible by 64")
        if self.num_experts < 1:
'''
NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_ACTIVATION_1 = r'''        self.fast_math = bool(fast_math)
        # [patch-036 b12x-n32] compact tails are 32 or 64 rows wide
        if self.n < 32 or self.n % 32:
            raise ValueError("compact W4A8 micro activation requires N divisible by 32")
        if self.num_experts < 1:
'''

# --- moe/_shared/kernels/w4a8_compact_micro.py (1/2)
OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_MICRO_1 = r'''        raise ValueError("max_tokens, num_topk, k, and n must be positive")
    if k % 128 or n < 64 or n % 64:
        raise ValueError(
            "compact W4A8 micro requires K divisible by 128 and N divisible by 64"
        )
'''
NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_MICRO_1 = r'''        raise ValueError("max_tokens, num_topk, k, and n must be positive")
    # [patch-036 b12x-n32] compact tails are 32 or 64 rows wide
    if k % 128 or n < 32 or n % 32 or n % 128 == 96:
        raise ValueError(
            "compact W4A8 micro requires K divisible by 128 and N divisible by 32 "
            "(n % 128 in {0, 32, 64})"
        )
'''

# --- moe/_shared/kernels/w4a8_compact_micro.py (2/2)
OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_MICRO_2 = r'''            direct_routes=True,
            n64_repacked=self.n % 128 == 64,
            n64_tail=self.n % 128 == 64,
            csf_inline=self.csf_inline,
'''
NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_MICRO_2 = r'''            direct_routes=True,
            # [patch-036 b12x-n32] a 64- or 32-row tail selects the exact N128-tile layout
            n64_repacked=self.n % 128 in (32, 64),
            n64_tail=self.n % 128 in (32, 64),
            n_tail_rows=(self.n % 128) or 64,
            csf_inline=self.csf_inline,
'''

# --- moe/_shared/kernels/w4a8_compact_projection.py (1/4)
OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_1 = r'''    def __init__(self, k: int, n: int, num_topk: int, csf_inline: bool = False):
        if k <= 0 or n < 64 or k % 128 or n % 64:
            raise ValueError(
                "compact W4A8 projection requires K divisible by 128 and "
                "N divisible by 64"
            )
'''
NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_1 = r'''    def __init__(self, k: int, n: int, num_topk: int, csf_inline: bool = False):
        # [patch-036 b12x-n32] compact tails are 32 or 64 rows wide
        if k <= 0 or n < 32 or k % 128 or n % 32 or n % 128 == 96:
            raise ValueError(
                "compact W4A8 projection requires K divisible by 128 and "
                "N divisible by 32 (n % 128 in {0, 32, 64})"
            )
'''

# --- moe/_shared/kernels/w4a8_compact_projection.py (2/4)
OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_2 = r'''        self.n_tiles = (self.n + self.tile_n - 1) // self.tile_n
        self.n64_tail = self.n % self.tile_n == 64
        # The SFB operand is inline MXFP4-CSF storage (mxfp4_csf_inline): stages
'''
NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_2 = r'''        self.n_tiles = (self.n + self.tile_n - 1) // self.tile_n
        # [patch-036 b12x-n32] rows of the last N128 tile: 64 (n % 128 == 64) or 32 (n % 128 == 32); 0 = no tail
        self.n_tail_rows = self.n % self.tile_n
        self.n64_tail = self.n_tail_rows in (32, 64)
        # The SFB operand is inline MXFP4-CSF storage (mxfp4_csf_inline): stages
'''

# --- moe/_shared/kernels/w4a8_compact_projection.py (3/4)
OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_3 = r'''        if output_tile == Int32(self.n_tiles - 1):
            n_rows = Int32(64)
        for i in cutlass.range_constexpr(2):
'''
NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_3 = r'''        if output_tile == Int32(self.n_tiles - 1):
            n_rows = Int32(self.n_tail_rows)  # [patch-036 b12x-n32] was Int32(64)
        for i in cutlass.range_constexpr(2):
'''

# --- moe/_shared/kernels/w4a8_compact_projection.py (4/4)
OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_4 = r'''        if output_tile == Int32(self.n_tiles - 1):
            n_rows = Int32(64)
        if tid < Int32(32):
'''
NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_4 = r'''        if output_tile == Int32(self.n_tiles - 1):
            n_rows = Int32(self.n_tail_rows)  # [patch-036 b12x-n32] was Int32(64)
        if tid < Int32(32):
'''

# --- moe/_shared/kernels/w4a8_phase1.py (1/5)
OLD_MOE_SHARED_KERNELS_W4A8_PHASE1_1 = r'''        csf_inline: bool = False,
    ):
'''
NEW_MOE_SHARED_KERNELS_W4A8_PHASE1_1 = r'''        csf_inline: bool = False,
        n_tail_rows: int = 64,  # [patch-036 b12x-n32]
    ):
'''

# --- moe/_shared/kernels/w4a8_phase1.py (2/5)
OLD_MOE_SHARED_KERNELS_W4A8_PHASE1_2 = r'''        self.n64_tail = bool(n64_tail)
        # The SFB operand is inline MXFP4-CSF storage (mxfp4_csf_inline): stages
'''
NEW_MOE_SHARED_KERNELS_W4A8_PHASE1_2 = r'''        self.n64_tail = bool(n64_tail)
        # [patch-036 b12x-n32] rows of the last N128 intermediate tile in compact storage (64 as before; 32 for n % 128 == 32)
        self.n_tail_rows = int(n_tail_rows)
        if self.n64_tail and self.n_tail_rows not in (32, 64):
            raise ValueError(f"n_tail_rows must be 32 or 64, got {n_tail_rows}")
        # The SFB operand is inline MXFP4-CSF storage (mxfp4_csf_inline): stages
'''

# --- moe/_shared/kernels/w4a8_phase1.py (3/5)
OLD_MOE_SHARED_KERNELS_W4A8_PHASE1_3 = r'''            raise ValueError("inline MXFP4-CSF scales require compact N64 W4A8 weights")
        if source_tile_m not in (16, 64, 128):
'''
NEW_MOE_SHARED_KERNELS_W4A8_PHASE1_3 = r'''            raise ValueError("inline MXFP4-CSF scales require compact N64 W4A8 weights")
        if self.csf_inline and self.n64_tail and self.n_tail_rows != 64:
            raise ValueError("inline MXFP4-CSF scales require a 64-row compact tail")  # [patch-036 b12x-n32]
        if source_tile_m not in (16, 64, 128):
'''

# --- moe/_shared/kernels/w4a8_phase1.py (4/5)
OLD_MOE_SHARED_KERNELS_W4A8_PHASE1_4 = r'''        total_k32: cutlass.Constexpr,
        tid: Int32,
    ):
        logical_n = intermediate_tiles * Int32(128)
        if cutlass.const_expr(self.n64_tail):
            logical_n -= Int32(64)
        n_rows = Int32(128)
        if cutlass.const_expr(self.n64_tail):
            if output_tile == intermediate_tiles - Int32(1):
                n_rows = Int32(64)
        transfers = 2 * 4 * 32
'''
NEW_MOE_SHARED_KERNELS_W4A8_PHASE1_4 = r'''        total_k32: cutlass.Constexpr,
        tid: Int32,
    ):
        logical_n = intermediate_tiles * Int32(128)
        if cutlass.const_expr(self.n64_tail):
            logical_n -= Int32(128 - self.n_tail_rows)  # [patch-036 b12x-n32] was Int32(64)
        n_rows = Int32(128)
        if cutlass.const_expr(self.n64_tail):
            if output_tile == intermediate_tiles - Int32(1):
                n_rows = Int32(self.n_tail_rows)  # [patch-036 b12x-n32] was Int32(64)
        transfers = 2 * 4 * 32
'''

# --- moe/_shared/kernels/w4a8_phase1.py (5/5)
OLD_MOE_SHARED_KERNELS_W4A8_PHASE1_5 = r'''        total_k32: cutlass.Constexpr,
        tid: Int32,
    ):
        logical_n = intermediate_tiles * Int32(128)
        if cutlass.const_expr(self.n64_tail):
            logical_n -= Int32(64)
        n_rows = Int32(128)
        if cutlass.const_expr(self.n64_tail):
            if output_tile == intermediate_tiles - Int32(1):
                n_rows = Int32(64)
        transfers = 512 // 16
'''
NEW_MOE_SHARED_KERNELS_W4A8_PHASE1_5 = r'''        total_k32: cutlass.Constexpr,
        tid: Int32,
    ):
        logical_n = intermediate_tiles * Int32(128)
        if cutlass.const_expr(self.n64_tail):
            logical_n -= Int32(128 - self.n_tail_rows)  # [patch-036 b12x-n32] was Int32(64)
        n_rows = Int32(128)
        if cutlass.const_expr(self.n64_tail):
            if output_tile == intermediate_tiles - Int32(1):
                n_rows = Int32(self.n_tail_rows)  # [patch-036 b12x-n32] was Int32(64)
        transfers = 512 // 16
'''

# --- moe/_shared/kernels/w4a8_phase2.py (1/6)
OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_1 = r'''        csf_inline: bool = False,
    ):
'''
NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_1 = r'''        csf_inline: bool = False,
        n_tail_rows: int = 64,  # [patch-036 b12x-n32]
    ):
'''

# --- moe/_shared/kernels/w4a8_phase2.py (2/6)
OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_2 = r'''        self.n64_tail = bool(n64_tail)
        # The SFB operand is inline MXFP4-CSF storage (mxfp4_csf_inline): stages
'''
NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_2 = r'''        self.n64_tail = bool(n64_tail)
        # [patch-036 b12x-n32] width of the compact tail tile in rows (64 as before; 32 for n % 128 == 32).  In FC2 the
        # intermediate width is the K extent, so the tail is its last K128 slice holding n_tail_rows // 32 K32 blocks.
        self.n_tail_rows = int(n_tail_rows)
        if self.n64_tail and self.n_tail_rows not in (32, 64):
            raise ValueError(f"n_tail_rows must be 32 or 64, got {n_tail_rows}")
        self.tail_k32 = self.n_tail_rows // 32
        # The SFB operand is inline MXFP4-CSF storage (mxfp4_csf_inline): stages
'''

# --- moe/_shared/kernels/w4a8_phase2.py (3/6)
OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_3 = r'''            raise ValueError("inline MXFP4-CSF scales require compact N64 W4A8 weights")
        self.direct_routes = bool(direct_routes)
'''
NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_3 = r'''            raise ValueError("inline MXFP4-CSF scales require compact N64 W4A8 weights")
        if self.csf_inline and self.n64_tail and self.n_tail_rows != 64:
            raise ValueError("inline MXFP4-CSF scales require a 64-row compact tail")  # [patch-036 b12x-n32]
        self.direct_routes = bool(direct_routes)
'''

# --- moe/_shared/kernels/w4a8_phase2.py (4/6)
OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_4 = r'''            if k128_slice == Int32(total_k32 // 4):
                k32_count = Int32(2)
        for i in cutlass.range_constexpr(
'''
NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_4 = r'''            if k128_slice == Int32(total_k32 // 4):
                k32_count = Int32(self.tail_k32)  # [patch-036 b12x-n32] was Int32(2)
        for i in cutlass.range_constexpr(
'''

# --- moe/_shared/kernels/w4a8_phase2.py (5/6)
OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_5 = r'''            if k128_slice == Int32(total_k32 // 4):
                tile_cols = Int32(2)
        for i in cutlass.range_constexpr(
'''
NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_5 = r'''            if k128_slice == Int32(total_k32 // 4):
                tile_cols = Int32(self.tail_k32)  # [patch-036 b12x-n32] was Int32(2)
        for i in cutlass.range_constexpr(
'''

# --- moe/_shared/kernels/w4a8_phase2.py (6/6)
OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_6 = r'''            if cutlass.const_expr(self.n64_tail):
                total_k32 -= Int32(2)
            self._stage_n64_b(
'''
NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_6 = r'''            if cutlass.const_expr(self.n64_tail):
                total_k32 -= Int32(4 - self.tail_k32)  # [patch-036 b12x-n32] was Int32(2)
            self._stage_n64_b(
'''

# --- moe/fused_moe/_impl.py (1/12)
OLD_MOE_FUSED_MOE_IMPL_1 = r'''from b12x._lib.program_cache import register_program_cache
from b12x._lib.compiler import (
'''
NEW_MOE_FUSED_MOE_IMPL_1 = r'''from b12x._lib.program_cache import register_program_cache
from b12x._lib.env import w4a8_compact_tail_rows  # [patch-036 b12x-n32]
from b12x._lib.compiler import (
'''

# --- moe/fused_moe/_impl.py (2/12)
OLD_MOE_FUSED_MOE_IMPL_2 = r'''        )
    if source_format == "fp4_e8m0_k32" and n % 128 == 64:
        return bool(
'''
NEW_MOE_FUSED_MOE_IMPL_2 = r'''        )
    # [patch-036 b12x-n32] n % 128 == 64, or == 32 with B12X_COMPACT_N32=1
    if source_format == "fp4_e8m0_k32" and w4a8_compact_tail_rows(n):
        return bool(
'''

# --- moe/fused_moe/_impl.py (3/12)
OLD_MOE_FUSED_MOE_IMPL_3 = r'''            and query.source_format == "fp4_e8m0_k32"
            and query.intermediate_size % 128 == 64
        )
'''
NEW_MOE_FUSED_MOE_IMPL_3 = r'''            and query.source_format == "fp4_e8m0_k32"
            # [patch-036 b12x-n32]
            and w4a8_compact_tail_rows(query.intermediate_size) != 0
        )
'''

# --- moe/fused_moe/_impl.py (4/12)
OLD_MOE_FUSED_MOE_IMPL_4 = r'''                )
            if source_format == "fp4_e8m0_k32" and n % 128 == 64:
                from b12x.moe._shared.kernels.w4a8_compact_micro import (
'''
NEW_MOE_FUSED_MOE_IMPL_4 = r'''                )
            # [patch-036 b12x-n32]
            if source_format == "fp4_e8m0_k32" and w4a8_compact_tail_rows(n):
                from b12x.moe._shared.kernels.w4a8_compact_micro import (
'''

# --- moe/fused_moe/_impl.py (5/12)
OLD_MOE_FUSED_MOE_IMPL_5 = r'''    group_rows = size_n if group_rows is None else int(group_rows)
    if size_n % 64 or size_k % 64 or group_rows % 64 or size_n % group_rows:
        raise ValueError(
            "exact W4A8 repack requires N, K, and group rows divisible by 64 "
            f"with whole row groups; got N={size_n}, K={size_k}, "
'''
NEW_MOE_FUSED_MOE_IMPL_5 = r'''    group_rows = size_n if group_rows is None else int(group_rows)
    # [patch-036 b12x-n32] 32-wide compact tails (n % 128 == 32) are as valid as 64-wide ones
    if size_n % 32 or size_k % 32 or group_rows % 32 or size_n % group_rows:
        raise ValueError(
            "exact W4A8 repack requires N, K, and group rows divisible by 32 "
            f"with whole row groups; got N={size_n}, K={size_k}, "
'''

# --- moe/fused_moe/_impl.py (6/12)
OLD_MOE_FUSED_MOE_IMPL_6 = r'''    group_rows = rows if group_rows is None else int(group_rows)
    if rows % 64 or k_dim % 64 or group_rows % 64 or rows % group_rows:
        raise ValueError(
            "exact W4A8 scale repack requires N, K, and group rows divisible "
            f"by 64 with whole row groups; got N={rows}, K={k_dim}, "
            f"group_rows={group_rows}"
'''
NEW_MOE_FUSED_MOE_IMPL_6 = r'''    group_rows = rows if group_rows is None else int(group_rows)
    # [patch-036 b12x-n32] 32-wide compact tails (n % 128 == 32) are as valid as 64-wide ones
    if rows % 32 or k_dim % 32 or group_rows % 32 or rows % group_rows:
        raise ValueError(
            "exact W4A8 scale repack requires N, K, and group rows divisible "
            f"by 32 with whole row groups; got N={rows}, K={k_dim}, "
            f"group_rows={group_rows}"
'''

# --- moe/fused_moe/_impl.py (7/12)
OLD_MOE_FUSED_MOE_IMPL_7 = r'''            w13_layout=plan.w13_layout,
            n64_repack=plan.intermediate_size % 128 == 64,
        )
'''
NEW_MOE_FUSED_MOE_IMPL_7 = r'''            w13_layout=plan.w13_layout,
            n64_repack=bool(w4a8_compact_tail_rows(plan.intermediate_size)),  # [patch-036 b12x-n32]
        )
'''

# --- moe/fused_moe/_impl.py (8/12)
OLD_MOE_FUSED_MOE_IMPL_8 = r'''        and source_format == "fp4_e8m0_k32"
        and n % 128 == 64
    )
'''
NEW_MOE_FUSED_MOE_IMPL_8 = r'''        and source_format == "fp4_e8m0_k32"
        and w4a8_compact_tail_rows(n) != 0  # [patch-036 b12x-n32]
    )
'''

# --- moe/fused_moe/_impl.py (9/12)
OLD_MOE_FUSED_MOE_IMPL_9 = r'''        and plan.quant_mode == "w4a8_mx"
        and plan.n % 128 == 64
    ):
'''
NEW_MOE_FUSED_MOE_IMPL_9 = r'''        and plan.quant_mode == "w4a8_mx"
        and w4a8_compact_tail_rows(plan.n) != 0  # [patch-036 b12x-n32]
    ):
'''

# --- moe/fused_moe/_impl.py (10/12)
OLD_MOE_FUSED_MOE_IMPL_10 = r'''        raise ValueError("inline MXFP4-CSF scales require compact N64 W4A8 weights")
    if w4a8_n64_repacked and n % 64 != 0:
        raise ValueError("w4a8_n64_repacked requires N divisible by 64")
    if w4a8_n64_repacked:
'''
NEW_MOE_FUSED_MOE_IMPL_10 = r'''        raise ValueError("inline MXFP4-CSF scales require compact N64 W4A8 weights")
    # [patch-036 b12x-n32] compact tails are 32 or 64 rows wide
    if w4a8_n64_repacked and n % 32 != 0:
        raise ValueError("w4a8_n64_repacked requires N divisible by 32")
    if w4a8_n64_repacked:
'''

# --- moe/fused_moe/_impl.py (11/12)
OLD_MOE_FUSED_MOE_IMPL_11 = r'''        kernel_kwargs["w4a8_n64_repacked"] = w4a8_n64_repacked
        kernel_kwargs["w4a8_n64_tail"] = bool(w4a8_n64_repacked and n % 128)
    elif is_w6a8 or is_w8a8:
'''
NEW_MOE_FUSED_MOE_IMPL_11 = r'''        kernel_kwargs["w4a8_n64_repacked"] = w4a8_n64_repacked
        # [patch-036 b12x-n32] the tail width in rows (True meant 64 before): 32 or 64
        kernel_kwargs["w4a8_n64_tail"] = (
            (n % 128) if (w4a8_n64_repacked and n % 128) else False
        )
    elif is_w6a8 or is_w8a8:
'''

# --- moe/fused_moe/_impl.py (12/12)
OLD_MOE_FUSED_MOE_IMPL_12 = r'''    routed_rows = m * num_topk
    if apply_router_weight_on_input and quant_mode != "w4a16":
'''
NEW_MOE_FUSED_MOE_IMPL_12 = r'''    routed_rows = m * num_topk
    # [patch-036 b12x-n32] the compact layout of a W4A8-MX expert slice is fixed when the weights are prepared; B12X_COMPACT_N32
    # is read again at plan and launch time, so a change in between would run kernels against the wrong storage layout.
    if (
        quant_mode == "w4a8_mx"
        and source_format == "fp4_e8m0_k32"
        and prepared_payload is not None
        and bool(getattr(prepared_payload, "n64_repack", False))
        != bool(w4a8_compact_tail_rows(n))
    ):
        raise RuntimeError(
            "B12X_COMPACT_N32 changed after the W4A8-MX weights were prepared "
            f"(n={n}, compact layout={bool(getattr(prepared_payload, 'n64_repack', False))}): "
            "prepare, plan and run with the same setting"
        )
    if apply_router_weight_on_input and quant_mode != "w4a16":
'''

# --- moe/fused_moe/_preparation.py (1/4)
OLD_MOE_FUSED_MOE_PREPARATION_1 = r'''from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
'''
NEW_MOE_FUSED_MOE_PREPARATION_1 = r'''from b12x._lib.compile_pool import CompileJob
from b12x._lib.env import compact_n32_enabled, w4a8_compact_tail_rows  # [patch-036 b12x-n32]
from b12x._lib.program_cache import program_cache
'''

# --- moe/fused_moe/_preparation.py (2/4)
OLD_MOE_FUSED_MOE_PREPARATION_2 = r'''            ),
        }
'''
NEW_MOE_FUSED_MOE_PREPARATION_2 = r'''            ),
            # [patch-036 b12x-n32] only when set: stock query digests (selection caches) stay byte-identical
            **({"w4a8_compact_n32": True} if compact_n32_enabled() else {}),
        }
'''

# --- moe/fused_moe/_preparation.py (3/4)
OLD_MOE_FUSED_MOE_PREPARATION_3 = r'''        and caps.weight_plan.source_format == "fp4_e8m0_k32"
        and int(n) % 128 == 64
    )
'''
NEW_MOE_FUSED_MOE_PREPARATION_3 = r'''        and caps.weight_plan.source_format == "fp4_e8m0_k32"
        # [patch-036 b12x-n32] n % 128 == 64, or == 32 with B12X_COMPACT_N32=1
        and w4a8_compact_tail_rows(int(n)) != 0
    )
'''

# --- moe/fused_moe/_preparation.py (4/4)
OLD_MOE_FUSED_MOE_PREPARATION_4 = r'''        and caps.weight_plan.source_format == "fp4_e8m0_k32"
        and plan.n % 128 == 64
    ):
'''
NEW_MOE_FUSED_MOE_PREPARATION_4 = r'''        and caps.weight_plan.source_format == "fp4_e8m0_k32"
        and w4a8_compact_tail_rows(plan.n) != 0  # [patch-036 b12x-n32]
    ):
'''

# --- moe/fused_moe/_tuning.py (1/4)
OLD_MOE_FUSED_MOE_TUNING_1 = r'''
from b12x._lib.quant.block_codec import BLOCK_CODECS, block_codec
'''
NEW_MOE_FUSED_MOE_TUNING_1 = r'''
from b12x._lib.env import w4a8_compact_tail_rows  # [patch-036 b12x-n32]
from b12x._lib.quant.block_codec import BLOCK_CODECS, block_codec
'''

# --- moe/fused_moe/_tuning.py (2/4)
OLD_MOE_FUSED_MOE_TUNING_2 = r'''        and query.source_format == "fp4_e8m0_k32"
        and query.intermediate_size % 128 == 64
    )
'''
NEW_MOE_FUSED_MOE_TUNING_2 = r'''        and query.source_format == "fp4_e8m0_k32"
        and w4a8_compact_tail_rows(query.intermediate_size) != 0  # [patch-036 b12x-n32]
    )
'''

# --- moe/fused_moe/_tuning.py (3/4)
OLD_MOE_FUSED_MOE_TUNING_3 = r'''        and query.source_format == "fp4_e8m0_k32"
        and query.intermediate_size % 128 == 64
        and config.backend == "dynamic"
'''
NEW_MOE_FUSED_MOE_TUNING_3 = r'''        and query.source_format == "fp4_e8m0_k32"
        and w4a8_compact_tail_rows(query.intermediate_size) != 0  # [patch-036 b12x-n32]
        and config.backend == "dynamic"
'''

# --- moe/fused_moe/_tuning.py (4/4)
OLD_MOE_FUSED_MOE_TUNING_4 = r'''        and effective_query.quant_mode == "w4a8_mx"
        and query.intermediate_size % 128 == 64
        and query.io_dtype != "bfloat16"
'''
NEW_MOE_FUSED_MOE_TUNING_4 = r'''        and effective_query.quant_mode == "w4a8_mx"
        and w4a8_compact_tail_rows(query.intermediate_size) != 0  # [patch-036 b12x-n32]
        and query.io_dtype != "bfloat16"
'''

# --- moe/fused_moe/planning.py (1/2)
OLD_MOE_FUSED_MOE_PLANNING_1 = r'''from b12x._lib.quant.block_codec import BLOCK_CODECS

'''
NEW_MOE_FUSED_MOE_PLANNING_1 = r'''from b12x._lib.quant.block_codec import BLOCK_CODECS
from b12x._lib.env import w4a8_compact_tail_rows  # [patch-036 b12x-n32]

'''

# --- moe/fused_moe/planning.py (2/2)
OLD_MOE_FUSED_MOE_PLANNING_2 = r'''            )
            planes = (weights.w13_scales, weights.w2_scales)
'''
NEW_MOE_FUSED_MOE_PLANNING_2 = r'''            )
            # [patch-036 b12x-n32] the MXFP4-CSF repack and its inline scale tiles know compact N64 tails only
            if n % 128 == 32 and w4a8_compact_tail_rows(n):
                raise ValueError(
                    "MXFP4-CSF A8 preparation does not support the native compact "
                    "n % 128 == 32 path (B12X_COMPACT_N32)"
                )
            planes = (weights.w13_scales, weights.w2_scales)
'''

EDITS = [
    ("_lib/env.py", SHA_LIB_ENV, OLD_LIB_ENV_1, NEW_LIB_ENV_1),
    ("moe/_shared/kernels/dynamic.py", SHA_MOE_SHARED_KERNELS_DYNAMIC, OLD_MOE_SHARED_KERNELS_DYNAMIC_1, NEW_MOE_SHARED_KERNELS_DYNAMIC_1),
    ("moe/_shared/kernels/dynamic.py", SHA_MOE_SHARED_KERNELS_DYNAMIC, OLD_MOE_SHARED_KERNELS_DYNAMIC_2, NEW_MOE_SHARED_KERNELS_DYNAMIC_2),
    ("moe/_shared/kernels/dynamic.py", SHA_MOE_SHARED_KERNELS_DYNAMIC, OLD_MOE_SHARED_KERNELS_DYNAMIC_3, NEW_MOE_SHARED_KERNELS_DYNAMIC_3),
    ("moe/_shared/kernels/w4a8_compact_activation.py", SHA_MOE_SHARED_KERNELS_W4A8_COMPACT_ACTIVATION, OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_ACTIVATION_1, NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_ACTIVATION_1),
    ("moe/_shared/kernels/w4a8_compact_micro.py", SHA_MOE_SHARED_KERNELS_W4A8_COMPACT_MICRO, OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_MICRO_1, NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_MICRO_1),
    ("moe/_shared/kernels/w4a8_compact_micro.py", SHA_MOE_SHARED_KERNELS_W4A8_COMPACT_MICRO, OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_MICRO_2, NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_MICRO_2),
    ("moe/_shared/kernels/w4a8_compact_projection.py", SHA_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION, OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_1, NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_1),
    ("moe/_shared/kernels/w4a8_compact_projection.py", SHA_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION, OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_2, NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_2),
    ("moe/_shared/kernels/w4a8_compact_projection.py", SHA_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION, OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_3, NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_3),
    ("moe/_shared/kernels/w4a8_compact_projection.py", SHA_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION, OLD_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_4, NEW_MOE_SHARED_KERNELS_W4A8_COMPACT_PROJECTION_4),
    ("moe/_shared/kernels/w4a8_phase1.py", SHA_MOE_SHARED_KERNELS_W4A8_PHASE1, OLD_MOE_SHARED_KERNELS_W4A8_PHASE1_1, NEW_MOE_SHARED_KERNELS_W4A8_PHASE1_1),
    ("moe/_shared/kernels/w4a8_phase1.py", SHA_MOE_SHARED_KERNELS_W4A8_PHASE1, OLD_MOE_SHARED_KERNELS_W4A8_PHASE1_2, NEW_MOE_SHARED_KERNELS_W4A8_PHASE1_2),
    ("moe/_shared/kernels/w4a8_phase1.py", SHA_MOE_SHARED_KERNELS_W4A8_PHASE1, OLD_MOE_SHARED_KERNELS_W4A8_PHASE1_3, NEW_MOE_SHARED_KERNELS_W4A8_PHASE1_3),
    ("moe/_shared/kernels/w4a8_phase1.py", SHA_MOE_SHARED_KERNELS_W4A8_PHASE1, OLD_MOE_SHARED_KERNELS_W4A8_PHASE1_4, NEW_MOE_SHARED_KERNELS_W4A8_PHASE1_4),
    ("moe/_shared/kernels/w4a8_phase1.py", SHA_MOE_SHARED_KERNELS_W4A8_PHASE1, OLD_MOE_SHARED_KERNELS_W4A8_PHASE1_5, NEW_MOE_SHARED_KERNELS_W4A8_PHASE1_5),
    ("moe/_shared/kernels/w4a8_phase2.py", SHA_MOE_SHARED_KERNELS_W4A8_PHASE2, OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_1, NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_1),
    ("moe/_shared/kernels/w4a8_phase2.py", SHA_MOE_SHARED_KERNELS_W4A8_PHASE2, OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_2, NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_2),
    ("moe/_shared/kernels/w4a8_phase2.py", SHA_MOE_SHARED_KERNELS_W4A8_PHASE2, OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_3, NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_3),
    ("moe/_shared/kernels/w4a8_phase2.py", SHA_MOE_SHARED_KERNELS_W4A8_PHASE2, OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_4, NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_4),
    ("moe/_shared/kernels/w4a8_phase2.py", SHA_MOE_SHARED_KERNELS_W4A8_PHASE2, OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_5, NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_5),
    ("moe/_shared/kernels/w4a8_phase2.py", SHA_MOE_SHARED_KERNELS_W4A8_PHASE2, OLD_MOE_SHARED_KERNELS_W4A8_PHASE2_6, NEW_MOE_SHARED_KERNELS_W4A8_PHASE2_6),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_1, NEW_MOE_FUSED_MOE_IMPL_1),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_2, NEW_MOE_FUSED_MOE_IMPL_2),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_3, NEW_MOE_FUSED_MOE_IMPL_3),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_4, NEW_MOE_FUSED_MOE_IMPL_4),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_5, NEW_MOE_FUSED_MOE_IMPL_5),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_6, NEW_MOE_FUSED_MOE_IMPL_6),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_7, NEW_MOE_FUSED_MOE_IMPL_7),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_8, NEW_MOE_FUSED_MOE_IMPL_8),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_9, NEW_MOE_FUSED_MOE_IMPL_9),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_10, NEW_MOE_FUSED_MOE_IMPL_10),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_11, NEW_MOE_FUSED_MOE_IMPL_11),
    ("moe/fused_moe/_impl.py", SHA_MOE_FUSED_MOE_IMPL, OLD_MOE_FUSED_MOE_IMPL_12, NEW_MOE_FUSED_MOE_IMPL_12),
    ("moe/fused_moe/_preparation.py", SHA_MOE_FUSED_MOE_PREPARATION, OLD_MOE_FUSED_MOE_PREPARATION_1, NEW_MOE_FUSED_MOE_PREPARATION_1),
    ("moe/fused_moe/_preparation.py", SHA_MOE_FUSED_MOE_PREPARATION, OLD_MOE_FUSED_MOE_PREPARATION_2, NEW_MOE_FUSED_MOE_PREPARATION_2),
    ("moe/fused_moe/_preparation.py", SHA_MOE_FUSED_MOE_PREPARATION, OLD_MOE_FUSED_MOE_PREPARATION_3, NEW_MOE_FUSED_MOE_PREPARATION_3),
    ("moe/fused_moe/_preparation.py", SHA_MOE_FUSED_MOE_PREPARATION, OLD_MOE_FUSED_MOE_PREPARATION_4, NEW_MOE_FUSED_MOE_PREPARATION_4),
    ("moe/fused_moe/_tuning.py", SHA_MOE_FUSED_MOE_TUNING, OLD_MOE_FUSED_MOE_TUNING_1, NEW_MOE_FUSED_MOE_TUNING_1),
    ("moe/fused_moe/_tuning.py", SHA_MOE_FUSED_MOE_TUNING, OLD_MOE_FUSED_MOE_TUNING_2, NEW_MOE_FUSED_MOE_TUNING_2),
    ("moe/fused_moe/_tuning.py", SHA_MOE_FUSED_MOE_TUNING, OLD_MOE_FUSED_MOE_TUNING_3, NEW_MOE_FUSED_MOE_TUNING_3),
    ("moe/fused_moe/_tuning.py", SHA_MOE_FUSED_MOE_TUNING, OLD_MOE_FUSED_MOE_TUNING_4, NEW_MOE_FUSED_MOE_TUNING_4),
    ("moe/fused_moe/planning.py", SHA_MOE_FUSED_MOE_PLANNING, OLD_MOE_FUSED_MOE_PLANNING_1, NEW_MOE_FUSED_MOE_PLANNING_1),
    ("moe/fused_moe/planning.py", SHA_MOE_FUSED_MOE_PLANNING, OLD_MOE_FUSED_MOE_PLANNING_2, NEW_MOE_FUSED_MOE_PLANNING_2),
]
