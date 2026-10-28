# SPDX-License-Identifier: Apache-2.0
"""Engram rows from node-local NVMe -- GPU side (recipe patch 016).

One `EngramNvmeChannel` per Engram layer served from NVMe. Per forward:

  publish(hash_ids)  right after the hash kernel (Engram.prepare_embeddings): ONE single-CTA kernel copies this rank's
                     `local_heads` hash columns into pinned host memory and rings the doorbell (`req_seq`, system-scope
                     release store of the new sequence number).
  [native reader     sees the doorbell, reads the rows with O_DIRECT (engram_nvme.c), compacts them into pinned staging in
                     the order the dequant kernel consumes them, publishes `done_seq` (release).]
  fetch(out)         right before the layer's all-gather (Engram._ready_rows): a single-CTA kernel waits (bounded spin,
                     system-scope acquire) for `done_seq >= seq`, then the dequant kernel turns the staged fp8 + e8m0 bytes
                     into the BF16 rows `staged_rows[:T]` -- the same arithmetic as ParallelEngramEmbedding's
                     _engram_lookup_kernel, so the rows are bit-identical to the resident path.

Everything is a kernel launch on the current stream with static buffers: it is captured into the FULL decode graph like any
other kernel, nothing runs on the CPU at replay, and no extra collective is issued. The sequence number lives in device memory
(`dev_seq`, incremented by the publish kernel) so every replay of a captured graph gets a fresh number.

Fail closed (see engram_nvme.c): a wait that times out, or one that is not preceded by a new publish (a stale request), sets
`gpu_abort`; the reader's watchdog then ends the process. While `status`/`gpu_abort` are set the dequant kernel writes NaN.

The pinned host tensors are kept alive for the life of the process (`_KEEPALIVE`): the native threads hold raw pointers to them.
"""
from __future__ import annotations

import os

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.models.deepseek_v41.common import engram_nvme_io as nio
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

_KEEPALIVE: list[torch.Tensor] = []

# word offsets into the int64 / int32 views of the control page (byte offsets / 8 and / 4). Triton kernels may only read
# module globals that are constexpr.
_REQ_SEQ64 = tl.constexpr(nio.OFF_REQ_SEQ // 8)
_DONE_SEQ64 = tl.constexpr(nio.OFF_DONE_SEQ // 8)
_STALL64 = tl.constexpr(nio.OFF_STALL // 8)
_SCRATCH64 = tl.constexpr(nio.OFF_SCRATCH // 8)
_REQ_NTOK32 = tl.constexpr(nio.OFF_REQ_NTOK // 4)
_STATUS32 = tl.constexpr(nio.OFF_STATUS // 4)
_GPU_ABORT32 = tl.constexpr(nio.OFF_GPU_ABORT // 4)


# --------------------------------------------------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["stride_t", "stride_h", "num_tokens", "head_start", "cap_rows"])
def _nvme_publish_kernel(
    hashes,  # int32 [T, >= head_start + LOCAL] row ids from the hash kernel (strided view of one layer)
    stride_t,
    stride_h,
    req_idx,  # int32 pinned [2, cap_rows]
    ctrl64,  # int64 view of the pinned control page
    ctrl32,  # int32 view of the same page
    dev_seq,  # int64[1] device
    cur_seq,  # int64[1] device: the sequence number of the request this forward waits for
    num_tokens,
    head_start,
    cap_rows,
    LOCAL: tl.constexpr,
    BLOCK: tl.constexpr,
    FLAG_XCHG: tl.constexpr,
):
    n = num_tokens * LOCAL
    seq = tl.load(dev_seq) + 1  # one program, one stream: no concurrent publishers, so no atomic needed
    tl.store(dev_seq, seq)
    slot = seq & 1
    base = req_idx + slot * cap_rows
    for start in range(0, n, BLOCK):
        i = start + tl.arange(0, BLOCK)
        m = i < n
        t = (i // LOCAL).to(tl.int64)
        h = i % LOCAL
        v = tl.load(hashes + t * stride_t + (head_start + h) * stride_h, mask=m, other=0)
        tl.store(base + i, v, mask=m)
    tl.debug_barrier()  # every thread's payload stores precede the doorbell
    tl.store(cur_seq, seq)
    tl.store(ctrl32 + _REQ_NTOK32, num_tokens)
    if FLAG_XCHG:
        tl.atomic_xchg(ctrl64 + _REQ_SEQ64, seq, sem="release", scope="sys")
    else:  # fence variant: a system-scope acq_rel RMW on a scratch word orders the stores, then a plain store rings the bell
        tl.atomic_add(ctrl64 + _SCRATCH64, 0, sem="acq_rel", scope="sys")
        tl.store(ctrl64 + _REQ_SEQ64, seq)


@triton.jit(do_not_specialize=["max_spins"])
def _nvme_wait_kernel(
    ctrl64,
    ctrl32,
    cur_seq,
    waited_seq,
    max_spins,
    VOLATILE_POLL: tl.constexpr,
):
    seq = tl.load(cur_seq)
    prev = tl.load(waited_seq)
    tl.store(waited_seq, seq)
    spins = 0
    if VOLATILE_POLL:
        done = tl.load(ctrl64 + _DONE_SEQ64, volatile=True)
        while (done < seq) & (spins < max_spins):
            done = tl.load(ctrl64 + _DONE_SEQ64, volatile=True)
            spins += 1
        done = tl.atomic_add(ctrl64 + _DONE_SEQ64, 0, sem="acquire", scope="sys")  # acquire ordering for the staging reads
    else:
        done = tl.atomic_add(ctrl64 + _DONE_SEQ64, 0, sem="acquire", scope="sys")
        while (done < seq) & (spins < max_spins):
            done = tl.atomic_add(ctrl64 + _DONE_SEQ64, 0, sem="acquire", scope="sys")
            spins += 1
    bad = (done < seq) | (prev == seq)  # timed out, or waited twice for one publish (the staged rows would be stale)
    if bad:
        tl.atomic_xchg(ctrl32 + _GPU_ABORT32, 1, sem="release", scope="sys")
    tl.store(ctrl64 + _STALL64 + (seq % 512), (seq << 32) | spins)


@triton.jit(do_not_specialize=["num_rows", "cap_rows", "GRID"])
def _nvme_dequant_kernel(
    stage_v,  # float8_e4m3fn pinned [2, cap_rows, DIM]
    stage_s,  # uint8 pinned [2, cap_rows, DIM // QUANT_BLOCK]
    out,  # bfloat16 [num_rows, DIM]  (= staged_rows[:T] viewed as rows = token * LOCAL + head)
    ctrl32,
    cur_seq,
    num_rows,
    cap_rows,
    GRID,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Same arithmetic as _engram_lookup_kernel's non-packed branch (engram.py): scale = ue8m0 byte placed in the fp32
    # exponent field, out = bf16(fp8 value as fp32 * scale). The resident kernel is deliberately left untouched.
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    slot = (tl.load(cur_seq) & 1).to(tl.int64)
    failed = (tl.load(ctrl32 + _STATUS32, volatile=True) != 0) | (tl.load(ctrl32 + _GPU_ABORT32, volatile=True) != 0)
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        srow = slot * cap_rows + rows.to(tl.int64)
        values = tl.load(
            stage_v + srow[:, None] * DIM + cols[None, :],
            mask=valid[:, None],
            other=0.0,
            cache_modifier=".cv",  # host memory the CPU rewrites every step: never serve it from a GPU cache
        )
        scale = tl.load(
            stage_s + srow[:, None] * (DIM // QUANT_BLOCK) + scale_cols[None, :],
            mask=valid[:, None],
            other=0,
            cache_modifier=".cv",
        )
        scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        res = (values.to(tl.float32) * scale).to(tl.bfloat16)
        res = tl.where(failed, float("nan"), res).to(tl.bfloat16)  # a failed channel can never produce plausible rows
        tl.store(out + rows[:, None] * DIM + cols[None, :], res, mask=valid[:, None])


# --------------------------------------------------------------------------------------------------------------------
# channel
# --------------------------------------------------------------------------------------------------------------------
def _pinned(nbytes: int) -> torch.Tensor:
    t = torch.zeros(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    _KEEPALIVE.append(t)
    return t


class EngramNvmeChannel:
    """The NVMe-served table of one Engram layer on this rank."""

    def __init__(
        self,
        engine: nio.NvmeEngine,
        loc: nio.TableLoc,
        *,
        layer_id: int,
        row_lo: int,
        row_hi: int,
        head_start: int,
        local_heads: int,
        dim: int,
        quant_block: int,
        max_tokens: int,
        device: torch.device,
        timeout_s: float,
    ) -> None:
        if dim != nio.ROW_V or dim // quant_block != nio.ROW_S:
            raise NotImplementedError(f"engram nvme reads {nio.ROW_V}+{nio.ROW_S} byte rows; this model has dim={dim}, block={quant_block}")
        self.layer_id = layer_id
        self.head_start = head_start
        self.local_heads = local_heads
        self.dim = dim
        self.quant_block = quant_block
        self.max_tokens = max_tokens
        self.cap_rows = max_tokens * local_heads
        self.device = device
        cap = self.cap_rows
        # pinned host memory: control page, request ids, staging (2 slots each, selected by sequence parity)
        ctrl = _pinned(nio.CTRL_BYTES)
        req = _pinned(2 * cap * 4)
        sv = _pinned(2 * cap * nio.ROW_V)
        ss = _pinned(2 * cap * nio.ROW_S)
        ctrl_gpu = get_accelerator_view_from_cpu_tensor(ctrl)
        self._ctrl64 = ctrl_gpu.view(torch.int64)
        self._ctrl32 = ctrl_gpu.view(torch.int32)
        self._req_idx = get_accelerator_view_from_cpu_tensor(req).view(torch.int32)
        self._stage_v = get_accelerator_view_from_cpu_tensor(sv).view(torch.float8_e4m3fn)
        self._stage_s = get_accelerator_view_from_cpu_tensor(ss)  # uint8
        # device-side sequence state (static addresses: captured graphs read and write them on every replay)
        self._dev_seq = torch.zeros(1, dtype=torch.int64, device=device)
        self._cur_seq = torch.zeros(1, dtype=torch.int64, device=device)
        self._waited_seq = torch.zeros(1, dtype=torch.int64, device=device)
        self._num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        # ~0.5 us per poll at the very least: the host watchdog (engram_nvme.c) is what really ends a hung engine
        self._max_spins = int(min(timeout_s * 2_000_000, 2**31 - 1))
        self._flag_xchg = os.environ.get("ENGRAM_NVME_FLAG", "xchg") != "fence"
        self._volatile_poll = os.environ.get("ENGRAM_NVME_WAIT", "atomic") == "volatile"
        self._pending: int | None = None
        self.engine = engine
        self.chan = engine.add_channel(loc, row_lo, row_hi, ctrl.data_ptr(), req.data_ptr(), sv.data_ptr(), ss.data_ptr(), cap, local_heads)
        logger.info(
            "[engram-nvme] layer %d: rows [%d, %d) of %d from %s (values @%d, scales @%d), %d heads/rank, "
            "staging %.1f MB pinned, backend=%s, flag=%s, wait=%s",
            layer_id, row_lo, row_hi, loc.num_rows, loc.path, loc.values_off, loc.scales_off, local_heads,
            (ctrl.numel() + req.numel() + sv.numel() + ss.numel()) / 2**20, engine.backend,
            "xchg" if self._flag_xchg else "fence", "volatile" if self._volatile_poll else "atomic",
        )

    # -- per forward, both launched on the current stream (so they are captured with the rest of the forward) --------
    def publish(self, hash_ids: torch.Tensor) -> None:
        """Hand this layer's hash ids (int32 [T, n_hash_cols]) to the reader."""
        num_tokens = hash_ids.shape[0]
        if self._pending is not None:  # a forward published but never fetched: harmless (the next wait uses the newest seq)
            logger.warning_once("engram nvme: layer %d published rows that were never fetched", self.layer_id)
        assert num_tokens <= self.max_tokens, f"engram nvme: {num_tokens} tokens > staging capacity {self.max_tokens}"
        assert hash_ids.dtype == torch.int32 and hash_ids.is_cuda
        self._pending = num_tokens
        if num_tokens == 0:
            return
        _nvme_publish_kernel[(1,)](
            hash_ids, hash_ids.stride(0), hash_ids.stride(1), self._req_idx, self._ctrl64, self._ctrl32, self._dev_seq, self._cur_seq,
            num_tokens, self.head_start, self.cap_rows,
            LOCAL=self.local_heads, BLOCK=1024, FLAG_XCHG=self._flag_xchg, num_warps=1,  # one warp: scalar atomics run once, the barrier is a warp barrier
        )

    def fetch(self, out: torch.Tensor) -> None:
        """Wait for the rows published in this forward and dequantize them into `out` ([T, local_heads, dim] BF16)."""
        num_tokens = out.shape[0]
        assert self._pending == num_tokens, f"engram nvme: fetch of {num_tokens} tokens after publish of {self._pending}"
        self._pending = None
        if num_tokens == 0:
            return
        assert out.is_contiguous() and out.dtype == torch.bfloat16 and tuple(out.shape[1:]) == (self.local_heads, self.dim)
        _nvme_wait_kernel[(1,)](
            self._ctrl64, self._ctrl32, self._cur_seq, self._waited_seq, self._max_spins, VOLATILE_POLL=self._volatile_poll, num_warps=1,
        )
        rows = num_tokens * self.local_heads
        grid = min(triton.cdiv(rows, 16), self._num_sms)
        _nvme_dequant_kernel[(grid,)](
            self._stage_v, self._stage_s, out, self._ctrl32, self._cur_seq, rows, self.cap_rows, grid,
            DIM=self.dim, QUANT_BLOCK=self.quant_block, BLOCK_R=16,
        )

    def status(self) -> str:
        return nio.STATUS_NAMES.get(self.chan.status(), str(self.chan.status()))


def create_channel(engram, layout, layer_hash_index: int, max_tokens: int) -> EngramNvmeChannel:
    """Called from Engram.__init__ for a layer whose rows are read from NVMe (EngramConfig.nvme)."""
    vllm_config = get_current_vllm_config()
    ec = vllm_config.engram_config
    emb = engram.embed_tokens
    layer_id = layout.layer_ids[layer_hash_index]
    if not vllm_config.use_v2_model_runner:
        raise NotImplementedError("EngramConfig.nvme needs the V2 model runner (the V1 hash path keeps slot-cache state)")
    if emb.dp_size != 1 or emb.tp_size < 1 or emb.part_n_hash_cols * emb.tp_size != emb.n_hash_cols:
        raise NotImplementedError(
            f"EngramConfig.nvme needs {emb.n_hash_cols} hash heads to divide evenly over TP={emb.tp_size} with no Engram DP group"
        )
    load_config = getattr(vllm_config, "load_config", None)
    strategy = getattr(load_config, "safetensors_load_strategy", None)
    extra = getattr(load_config, "model_loader_extra_config", None)
    if strategy not in (None, "lazy") or (isinstance(extra, dict) and extra.get("enable_multithread_load")):
        # eager / prefetch / the multi-thread iterator read whole shards (the two Engram shards are ~94 GiB each) and ignore the
        # loader skip; fail now, before any weight is read, instead of filling memory or the page cache
        raise NotImplementedError(
            f"EngramConfig.nvme needs the default lazy safetensors loader (got safetensors_load_strategy={strategy!r}, "
            f"enable_multithread_load={bool(isinstance(extra, dict) and extra.get('enable_multithread_load'))})"
        )
    model_dir = ec.nvme_path or vllm_config.model_config.model
    loc = nio.locate_tables(model_dir, layer_id, layout.num_embeddings[layer_hash_index], layout.head_dim, emb.block_size)
    cap_rows = max_tokens * emb.part_n_hash_cols
    engine = nio.get_engine(nio.NvmeSettings.from_engram_config(ec), cap_rows)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    assert emb.vocab_end_idx <= loc.num_rows
    return EngramNvmeChannel(
        engine,
        loc,
        layer_id=layer_id,
        row_lo=emb.vocab_start_idx,
        row_hi=emb.vocab_end_idx,
        head_start=emb.head_start,
        local_heads=emb.part_n_hash_cols,
        dim=emb.dim,
        quant_block=emb.block_size,
        max_tokens=max_tokens,
        device=device,
        timeout_s=ec.nvme_timeout_s,
    )
