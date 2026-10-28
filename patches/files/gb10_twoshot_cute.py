# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernel of the GB10 two-shot RoCE all-reduce (see gb10_twoshot.py).

Derives from FlashInfer's b12x RoCEnante one-shot kernel (b12x/comm/roce/_oneshot_cute.py, Apache-2.0): the staging,
the doorbell, the multi-block counter scheme, the epoch and the poison/timeout handling are adapted from it.  The
reduction is NOT done here: the proxy thread of each rank reduces its own chunk on the CPU.

One launch is one complete all-reduce for one message:

1. stage:    copy the device input into ``send[seq & 1]`` in pinned host memory;
2. doorbell: the last block to finish staging publishes ``nbytes`` and the dtype code in the per-slot words and then
             ``seq`` in the control record that the proxy polls (a level, not a queue: a proxy that was descheduled
             across two doorbells finds ``seq`` ahead and posts both slots, which is why the words are per slot);
3. wait:     spin (``ld.acquire.sys``) on ``flag2[q][seq & 1][rail] == seq`` for EVERY rank q, this rank included.  Flag
             ``q`` appears after rank q's reduced chunk was written to ``recv2[q]`` (by q's NIC, or by this rank's own
             proxy thread for q == this rank).  A wait that exceeds ``spin_limit`` polls, or that finds the proxy's
             error word set, records ``seq`` in the control record and poisons the runtime;
4. copy:     ``recv2[q]`` -> the output at pack offset ``q * c`` for every chunk q (c = ceil(packs / world)); no
             arithmetic on the GPU;
5. epoch:    the last block to finish advances the device-resident epoch (unless an error was recorded), which makes the
             sequence number a runtime value rather than a launch argument and keeps CUDA-graph replay correct.

Staging arrivals and tail arrivals use two separate counters per power-of-two grid size, exactly as in RoCEnante: a block
that stages nothing could otherwise reach the tail before a slower block has staged and ring the doorbell early.

Control record (u32 words): 0 doorbell seq, 1 nbytes, 2 GPU error seq, 3 GPU error peer, 4-5 per-slot nbytes, 6 GPU
error rail, 7 GPU error code (1 = flag timeout, 2 = proxy failed), 8-9 per-slot dtype code; line 1 (byte 128+): proxy
error word, peer, seq, last completed seq.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from flashinfer.experimental.b12x.comm.roce._cute_intrinsics import (
    atomic_add_relaxed_gpu_u32,
    fence_sc_gpu,
    fence_sc_sys,
    ld_global_v4_u32,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    ld_relaxed_sys_v4_u32,
    st_global_v4_u32,
    st_release_gpu_u32,
    st_relaxed_sys_u32,
)

PACK_BYTES = 16
UNROLL = 4  # independent 16-byte loads in flight per thread in the staging and copy loops
PROXY_ERR_OFFSET = 128  # byte offset of the proxy error word in the control record
_SRC_HASH = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]  # a source edit invalidates the disk cache


@dsl_user_op
def spin_until_eq_acquire_sys_abortable(
    addr: Int64, expected: Uint32, limit: Uint32, abort_addr: Int64, *, loc=None, ip=None
) -> Uint32:
    """Spin until the word at ``addr`` equals ``expected`` (acquire, system scope).

    Returns 0 on success, 1 after ``limit`` polls without a match, and 2 as soon as the word at ``abort_addr`` (the
    proxy's error word, polled every 128th iteration) is non-zero: a dead proxy can never produce the flag, so the
    kernel gives up at once instead of after the full limit.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Uint32(expected).ir_value(loc=loc, ip=ip),
                Uint32(limit).ir_value(loc=loc, ip=ip),
                Int64(abort_addr).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred pending, expired, skip, aborted;
                .reg .b32 seen, polls, low, err, res;
                mov.u32 polls, 0;
                mov.u32 res, 0;
            ts_wait:
                ld.acquire.sys.global.u32 seen, [$1];
                setp.ne.u32 pending, seen, $2;
                @!pending bra ts_done;
                add.u32 polls, polls, 1;
                and.b32 low, polls, 127;
                setp.ne.u32 skip, low, 0;
                @skip bra ts_noabort;
                ld.relaxed.sys.global.u32 err, [$4];
                setp.ne.u32 aborted, err, 0;
                @!aborted bra ts_noabort;
                mov.u32 res, 2;
                bra ts_done;
            ts_noabort:
                setp.ge.u32 expired, polls, $3;
                @!expired bra ts_wait;
                mov.u32 res, 1;
            ts_done:
                mov.u32 $0, res;
            }
            """,
            "=r,l,r,r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def globaltimer_ns(*, loc=None, ip=None) -> Int64:
    """The GPU's nanosecond global timer (profiling builds only)."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_u64(addr: Int64, value: Int64, *, loc=None, ip=None) -> None:
    """Plain 64-bit global store (profiling builds only)."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _TwoShotLaunch:
    def __init__(
        self, world_size: int, threads: int, slots: int, flag_stride: int, rails: int, profile: bool = False
    ) -> None:
        """Bind one kernel specialization: world size and layout constants (the kernel does not depend on the rank)."""
        if int(threads) < int(world_size) * int(rails):
            raise ValueError(
                "two-shot kernel needs threads >= world_size * rails (one thread per flag), got "
                f"threads={threads} world_size={world_size} rails={rails}"
            )
        self._world_size = int(world_size)
        self._threads = int(threads)
        self._slots = int(slots)
        self._flag_stride = int(flag_stride)
        self._rails = int(rails)
        self._profile = bool(profile)

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        op_code: Int32,
        send_base: Int64,
        recv2_base: Int64,
        flag2_base: Int64,
        ctrl_base: Int64,
        slot_bytes: Int64,
        cslot_bytes: Int64,
        epoch_ptr: Int64,
        stage_counter_ptr: Int64,
        tail_counter_ptr: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        prof_base: Int64,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: launch the all-reduce kernel with runtime scalars."""
        self.kernel(
            input_ptr,
            output_ptr,
            size_packs,
            nbytes,
            op_code,
            send_base,
            recv2_base,
            flag2_base,
            ctrl_base,
            slot_bytes,
            cslot_bytes,
            epoch_ptr,
            stage_counter_ptr,
            tail_counter_ptr,
            poison_ptr,
            spin_limit,
            prof_base,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        op_code: Int32,
        send_base: Int64,
        recv2_base: Int64,
        flag2_base: Int64,
        ctrl_base: Int64,
        slot_bytes: Int64,
        cslot_bytes: Int64,
        epoch_ptr: Int64,
        stage_counter_ptr: Int64,
        tail_counter_ptr: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        prof_base: Int64,
    ) -> None:
        """Device kernel: stage, doorbell, wait for every rank's reduced chunk, copy out, advance the epoch."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        # Every block reads the epoch before any block can advance it: the advance happens only after all blocks
        # arrived at the tail counter.
        epoch = ld_relaxed_gpu_u32(epoch_ptr)
        seq = epoch + Uint32(1)
        slot = Int64(seq & Uint32(1))
        send_slot = send_base + slot * slot_bytes

        index = Int32(bidx) * Int32(self._threads) + Int32(tidx)
        stride = Int32(gdim) * Int32(self._threads)
        # profiling builds: one 64-byte row of GPU timestamps per sequence (block 0 thread 0 unless noted)
        prow = prof_base + Int64(seq & Uint32(8191)) * Int64(64)
        if cutlass.const_expr(self._profile):
            if Int32(tidx) == Int32(0):
                if Int32(bidx) == Int32(0):
                    st_global_u64(prow, Int64(seq))
                    st_global_u64(prow + Int64(8), globaltimer_ns())

        # A recorded timeout poisons the runtime: later launches do nothing, so the host sees the failure without
        # waiting another spin limit per op.
        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # 1. stage the input into the pinned send slot (UNROLL independent loads in flight per thread)
            stage_index = index
            while stage_index < size_packs:
                staged = cute.make_rmem_tensor((UNROLL * 4,), cutlass.Uint32)
                for u in cutlass.range_constexpr(UNROLL):
                    g = stage_index + Int32(u) * stride
                    if g < size_packs:
                        words = ld_global_v4_u32(input_base + Int64(g) * Int64(PACK_BYTES))
                        for k in cutlass.range_constexpr(4):
                            staged[u * 4 + k] = words[k]
                for u in cutlass.range_constexpr(UNROLL):
                    g = stage_index + Int32(u) * stride
                    if g < size_packs:
                        st_global_v4_u32(
                            send_slot + Int64(g) * Int64(PACK_BYTES),
                            staged[u * 4],
                            staged[u * 4 + 1],
                            staged[u * 4 + 2],
                            staged[u * 4 + 3],
                        )
                stage_index += Int32(UNROLL) * stride
            cute.arch.sync_threads()
            if cutlass.const_expr(self._profile):
                if Int32(tidx) == Int32(0):
                    if Int32(bidx) == Int32(0):
                        st_global_u64(prow + Int64(16), globaltimer_ns())

            # 2. the last block to finish staging rings the proxy doorbell
            if Int32(tidx) == Int32(0):
                fence_sc_sys()
                prior = atomic_add_relaxed_gpu_u32(stage_counter_ptr, Uint32(1))
                if cutlass.const_expr(self._profile):
                    if Int32(bidx) == Int32(0):
                        st_global_u64(prow + Int64(24), globaltimer_ns())
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    if cutlass.const_expr(self._profile):
                        st_global_u64(prow + Int64(32), globaltimer_ns())
                    st_relaxed_sys_u32(ctrl_base + Int64(4), Uint32(nbytes))
                    st_relaxed_sys_u32(ctrl_base + Int64(16) + slot * Int64(4), Uint32(nbytes))
                    st_relaxed_sys_u32(ctrl_base + Int64(32) + slot * Int64(4), Uint32(op_code))
                    fence_sc_sys()
                    st_relaxed_sys_u32(ctrl_base, seq)

            # 3. wait for every rank's phase-2 flag, this rank's own included
            if Int32(tidx) < Int32(self._world_size * self._rails):
                peer = Int32(tidx) // Int32(self._rails)
                rail = Int32(tidx) - peer * Int32(self._rails)
                flag_addr = flag2_base + (
                    (Int64(peer) * Int64(self._slots) + slot) * Int64(self._rails) + Int64(rail)
                ) * Int64(self._flag_stride)
                status = spin_until_eq_acquire_sys_abortable(
                    flag_addr, seq, spin_limit, ctrl_base + Int64(PROXY_ERR_OFFSET)
                )
                if status != Uint32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(12), Uint32(peer))
                    st_relaxed_sys_u32(ctrl_base + Int64(24), Uint32(rail))
                    st_relaxed_sys_u32(ctrl_base + Int64(28), status)
                    st_relaxed_sys_u32(ctrl_base + Int64(8), seq)
                    st_release_gpu_u32(poison_ptr, seq)
            cute.arch.sync_threads()
            if cutlass.const_expr(self._profile):
                if Int32(tidx) == Int32(0):
                    if Int32(bidx) == Int32(0):
                        st_global_u64(prow + Int64(40), globaltimer_ns())
            # A wait that gave up in this block leaves the reduced chunks unreliable: skip the copy.
            failed = ld_relaxed_gpu_u32(poison_ptr)
            if failed == Uint32(0):
                # 4. copy the W reduced chunks into the output: pack g belongs to chunk q = g // c at offset g - q * c
                chunk = (size_packs + Int32(self._world_size - 1)) // Int32(self._world_size)
                copy_index = index
                while copy_index < size_packs:
                    moved = cute.make_rmem_tensor((UNROLL * 4,), cutlass.Uint32)
                    for u in cutlass.range_constexpr(UNROLL):
                        g = copy_index + Int32(u) * stride
                        if g < size_packs:
                            q = g // chunk
                            src = (
                                recv2_base
                                + (Int64(q) * Int64(self._slots) + slot) * cslot_bytes
                                + Int64(g - q * chunk) * Int64(PACK_BYTES)
                            )
                            words = ld_relaxed_sys_v4_u32(src)
                            for k in cutlass.range_constexpr(4):
                                moved[u * 4 + k] = words[k]
                    for u in cutlass.range_constexpr(UNROLL):
                        g = copy_index + Int32(u) * stride
                        if g < size_packs:
                            st_global_v4_u32(
                                output_base + Int64(g) * Int64(PACK_BYTES),
                                moved[u * 4],
                                moved[u * 4 + 1],
                                moved[u * 4 + 2],
                                moved[u * 4 + 3],
                            )
                    copy_index += Int32(UNROLL) * stride

            # 5. the last block to finish publishes the next epoch
            fence_sc_gpu()
            cute.arch.sync_threads()
            if cutlass.const_expr(self._profile):
                if Int32(tidx) == Int32(0):
                    if Int32(bidx) == Int32(0):
                        st_global_u64(prow + Int64(48), globaltimer_ns())
            if Int32(tidx) == Int32(0):
                prior = atomic_add_relaxed_gpu_u32(tail_counter_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    fence_sc_gpu()
                    if cutlass.const_expr(self._profile):
                        st_global_u64(prow + Int64(56), globaltimer_ns())
                    # Every block's timeout store precedes its tail arrival, so the error word is final here.  A failed
                    # sequence keeps the epoch, which makes every later launch a no-op until the host raises.
                    if ld_relaxed_sys_u32(ctrl_base + Int64(8)) == Uint32(0):
                        st_release_gpu_u32(epoch_ptr, seq)


def _dummy(dtype, alignment: int):
    """A pointer of ``dtype`` used to trace launcher argument types."""
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=alignment)


_LAUNCHERS: dict[tuple, Callable[..., None]] = {}


def get_launcher(
    world_size: int, threads: int, slots: int, flag_stride: int, rails: int, device_index: int, profile: bool = False
) -> Callable[..., None]:
    """Compile (or load from the disk cache) the launcher for one specialization and return it.

    Not allowed inside a CUDA graph capture (b12x's kernel-resolution guard refuses it): the runtime calls it from
    its constructor.
    """
    key = (int(world_size), int(threads), int(slots), int(flag_stride), int(rails), int(device_index), bool(profile))
    cached = _LAUNCHERS.get(key)
    if cached is not None:
        return cached
    launch = _TwoShotLaunch(world_size, threads, slots, flag_stride, rails, profile)
    cache_key = (int(world_size), int(threads), int(slots), int(flag_stride), int(rails), UNROLL, int(bool(profile)), _SRC_HASH)
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=cache_key)
    raw = b12x_compile(
        launch,
        _dummy(cutlass.Uint32, 16),  # input
        _dummy(cutlass.Uint32, 16),  # output
        1,  # size_packs
        16,  # nbytes
        1,  # op_code
        16,  # send_base
        16,  # recv2_base
        16,  # flag2_base
        16,  # ctrl_base
        4096,  # slot_bytes
        256,  # cslot_bytes
        16,  # epoch
        16,  # stage counter
        16,  # tail counter
        16,  # poison
        1,  # spin_limit
        16,  # prof_base
        1,  # grid_x
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("comm.gb10_twoshot", 1, cache_key),
    )

    def run(
        input_address: int,
        output_address: int,
        size_packs: int,
        nbytes: int,
        op_code: int,
        send_base: int,
        recv2_base: int,
        flag2_base: int,
        ctrl_base: int,
        slot_bytes: int,
        cslot_bytes: int,
        epoch_address: int,
        stage_counter_address: int,
        tail_counter_address: int,
        poison_address: int,
        spin_limit: int,
        prof_address: int,
        grid_x: int,
    ) -> None:
        """Launch the compiled kernel with runtime scalar arguments."""
        raw(
            make_ptr(cutlass.Uint32, input_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16),
            int(size_packs),
            int(nbytes),
            int(op_code),
            int(send_base),
            int(recv2_base),
            int(flag2_base),
            int(ctrl_base),
            int(slot_bytes),
            int(cslot_bytes),
            int(epoch_address),
            int(stage_counter_address),
            int(tail_counter_address),
            int(poison_address),
            int(spin_limit),
            int(prof_address),
            int(grid_x),
            current_cuda_stream(),
        )

    launcher = attach_programs(run, raw)
    _LAUNCHERS[key] = launcher
    return launcher


__all__ = ["PACK_BYTES", "UNROLL", "PROXY_ERR_OFFSET", "get_launcher"]
