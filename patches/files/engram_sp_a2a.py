# SPDX-License-Identifier: Apache-2.0
"""Engram under prefill sequence parallelism without its two TP all-gathers (patch 022 extension; installed as
vllm/_engram_sp_a2a.py; used by vllm/_prefill_sp.py when VLLM_GB10_PREFILL_SP_ENGRAM=a2a).

WHY. Inside an SP region each rank needs the Engram kv only for its own S rows, but the upstream path computes it for all
T rows: the head-sharded n-gram lookup is all-gathered ([T, 3, 256] per rank -> [T, 24, 256], 3.9 ms per 4,096-row chunk)
and the column-parallel wkv output is all-gathered ([T, 3,200] per rank -> [T, 25,600], 15.7 ms). Two Engram layers:
~39 ms of link time per chunk, ~5 % of an SP prefill chunk.

WHAT. (1) An all-to-all on the TP group's own NCCL communicator (pynccl grouped send/recv): rank r sends rank q its looked-up
heads for q's rows, so every rank holds all 24 heads for its S rows ([S, 24, 256]; 7/8 of 6.3 MB per rank instead of
receiving 44 MB). (2) A replicated wkv ([25,600 x 6,144] FP8, loaded once from the checkpoint exactly as vLLM's loader
prepares it: E8M0 scales of 32 x 32 blocks repeated over each block's 32 rows -> MXFP8 [N, K/32]) runs the FlashInfer
CUTLASS W8A8 MXFP8 GEMM -- the kernel the upstream path already uses for these prefill GEMMs (patch 021, >= 512 rows) --
on the S local rows. No all-gather is left; the GEMM FLOPs per rank are unchanged (S x 25,600 instead of T x 3,200).
Memory: +~160 MB per Engram layer per rank.

SAFETY. The first SP forward that reaches an Engram layer with real hashes runs both paths for that layer and compares
this rank's rows (data movement must be bitwise; the GEMM within RTOL); every rank votes over gloo, and on any failure all
ranks fall back to the upstream gather path for the rest of the process (logged). Called only inside an SP region (eager,
no CUDA graph).
"""

from __future__ import annotations

import json
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

TAG = "[patch-022 engram-a2a]"
ENV = "VLLM_GB10_PREFILL_SP_ENGRAM"
RTOL = 2.0e-2          # amax-relative kv difference allowed between the replicated and the column-parallel GEMM


def mode() -> str:
    m = os.environ.get(ENV, "gather").strip().lower()
    if m not in ("gather", "a2a"):
        raise ValueError(f"{TAG} {ENV}={m!r} must be 'gather' or 'a2a'")
    return m


class _FullWkv(torch.nn.Module):
    """The whole wkv of one Engram layer on every rank: FP8 weight + per-row MXFP8 scales, FlashInfer W8A8 MXFP8."""

    def __init__(self, layer_id: int, model_dir: str, device: torch.device) -> None:
        super().__init__()
        from safetensors import safe_open

        from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
        from vllm.model_executor.kernels.linear.mxfp8.flashinfer import FlashInferCutlassMxfp8LinearKernel

        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            wmap = json.load(f)["weight_map"]
        wname, sname = f"layers.{layer_id}.engram.wkv.weight", f"layers.{layer_id}.engram.wkv.scale"
        with safe_open(os.path.join(model_dir, wmap[wname]), framework="pt", device="cpu") as f:
            w = f.get_tensor(wname)
        with safe_open(os.path.join(model_dir, wmap[sname]), framework="pt", device="cpu") as f:
            s = f.get_tensor(sname)
        n, k = w.shape
        if w.dtype != torch.float8_e4m3fn or k % 32 or s.shape[1] * 32 != k or n % s.shape[0]:
            raise RuntimeError(f"{TAG} unexpected wkv layout for layer {layer_id}: weight {tuple(w.shape)} {w.dtype}, "
                               f"scale {tuple(s.shape)} {s.dtype}")
        block_rows = n // s.shape[0]
        s = s.view(torch.uint8).repeat_interleave(block_rows, dim=0)  # as KMxfp8Static.get_scale_weight_loader
        self.weight = torch.nn.Parameter(w.to(device), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(s.to(device).contiguous(), requires_grad=False)
        self.kernel = FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
        ok, why = self.kernel.is_supported()
        if ok:
            ok, why = self.kernel.can_implement(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
        if not ok:
            raise RuntimeError(f"{TAG} FlashInfer MXFP8 GEMM unavailable for wkv {n}x{k}: {why}")
        self.kernel.process_weights_after_loading(self)
        self.out_features = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel.apply_weights(self, x, None)


class _State:
    def __init__(self) -> None:
        self.full: dict[int, _FullWkv] = {}       # id(engram) -> replicated wkv
        self.layer_ids: dict[int, int] = {}       # id(engram) -> decoder layer index
        self.vllm_config = None                   # the model's VllmConfig (no current config inside a forward)
        self.verified = False
        self.disabled_reason: str | None = None
        self.comm = None


_ST = _State()


def register_model(model) -> None:
    """Map every Engram module of the model to its decoder-layer index and keep the model's VllmConfig (called by
    begin_forward, cheap, idempotent): inside a forward there is no current vLLM config to read the model path from."""
    if _ST.layer_ids:
        return
    _ST.vllm_config = getattr(model, "vllm_config", None)
    for i, layer in enumerate(model.layers):
        eng = getattr(layer, "engram", None)
        if eng is not None:
            _ST.layer_ids[id(eng)] = i


def active() -> bool:
    return mode() == "a2a" and _ST.disabled_reason is None


def _comm():
    if _ST.comm is None:
        from vllm.distributed.parallel_state import get_tp_group

        comm = getattr(get_tp_group().device_communicator, "pynccl_comm", None)
        if comm is None or getattr(comm, "disabled", True):
            raise RuntimeError(f"{TAG} the TP group has no usable pynccl communicator")
        _ST.comm = comm
    return _ST.comm


def a2a_rows(rows: torch.Tensor, plan, rank: int) -> torch.Tensor:
    """[T, h, D] (this rank's h heads, all T rows) -> [S, W*h, D] (all heads, this rank's S rows; zero rows past T)."""
    from vllm._prefill_sp import pad_rows

    W, S = plan.W, plan.S
    send = pad_rows(rows, plan.padded).view(W, S, *rows.shape[1:])
    recv = torch.empty_like(send)
    comm = _comm()
    comm.group_start()
    for q in range(W):
        if q != rank:
            comm.send(send[q], q)
            comm.recv(recv[q], q)
    comm.group_end()
    recv[rank].copy_(send[rank])
    return recv.permute(1, 0, 2, 3).reshape(S, W * rows.shape[1], rows.shape[2])


def _full_wkv(engram, device) -> _FullWkv:
    fw = _ST.full.get(id(engram))
    if fw is None:
        from vllm.config import set_current_vllm_config

        layer_id = _ST.layer_ids.get(id(engram))
        if layer_id is None or _ST.vllm_config is None:
            raise RuntimeError(f"{TAG} Engram module or model config not registered (register_model was not called)")
        model_dir = _ST.vllm_config.model_config.model
        with set_current_vllm_config(_ST.vllm_config):  # the kernel classes may read the current config
            fw = _FullWkv(layer_id, model_dir, device)
        _ST.full[id(engram)] = fw
        logger.info("%s layer %d: replicated wkv %s loaded (%.0f MB)", TAG, layer_id, tuple(fw.weight.shape),
                    (fw.weight.numel() + fw.weight_scale.numel()) / 1e6)
    return fw


def local_kv(engram, hash_ids: torch.Tensor, plan, rank: int) -> torch.Tensor:
    """kv [S, (hc+1)*dim] for this rank's rows, without any all-gather."""
    rows = engram._ready_rows(hash_ids.shape[0])            # [T, h, D]: resident lookup or the NVMe reader (patch 016)
    heads = a2a_rows(rows, plan, rank)[:, : engram.embed_tokens.n_hash_cols]
    return _full_wkv(engram, rows.device)(heads.flatten(-2))


def _amax_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float()).abs().max().item()
    m = b.float().abs().max().item()
    return d / m if m > 0 else d


def verified_local_kv(engram, hash_ids: torch.Tensor, plan, rank: int, world) -> torch.Tensor | None:
    """local_kv(); on the first call also the upstream path, compared on this rank's rows and voted over gloo by every rank.
    Returns None when the a2a path is (now) disabled: the caller then runs the upstream gather path."""
    if not active():
        return None
    if _ST.verified:
        return local_kv(engram, hash_ids, plan, rank)
    verdict, kv_new, err_rows, err_kv = "ok", None, -1.0, -1.0
    try:
        from vllm.distributed import tensor_model_parallel_all_gather

        lo, real = plan.start(rank), plan.real_rows(rank)
        n_cols = engram.embed_tokens.n_hash_cols
        # _ready_rows once (the NVMe reader of patch 016 hands each batch over once); then the upstream path on those rows:
        # heads all-gathered over TP (as Engram.embed), column-parallel wkv with its gathered output
        rows = engram._ready_rows(hash_ids.shape[0])
        ref_heads = tensor_model_parallel_all_gather(rows, dim=1)[:, :n_cols]
        ref_kv = engram.wkv(ref_heads.flatten(-2))
        heads = a2a_rows(rows, plan, rank)[:, :n_cols]
        err_rows = 0.0 if torch.equal(heads[:real], ref_heads[lo:lo + real]) else float("inf")
        kv_new = _full_wkv(engram, rows.device)(heads.flatten(-2))
        err_kv = _amax_rel(kv_new[:real], ref_kv[lo:lo + real]) if real > 0 else 0.0
        if err_rows != 0.0:
            verdict = "a2a rows differ from the gathered rows"
        elif not err_kv <= RTOL:
            verdict = f"kv amax-rel error {err_kv:.3e} > {RTOL:.0e}"
    except Exception as e:  # noqa: BLE001 - every rank must still vote
        verdict = f"{type(e).__name__}: {e}"
    votes = world.all_gather_object((verdict, err_kv))
    bad = [(r, v) for r, (v, _) in enumerate(votes) if v != "ok"]
    if bad:
        _ST.disabled_reason = "; ".join(f"rank {r}: {v}" for r, v in bad)[:500]
        logger.error("%s self-test FAILED -> upstream gather path for the rest of this process: %s", TAG,
                     _ST.disabled_reason)
        return None
    _ST.verified = True
    logger.info("%s ACTIVE: first live Engram layer verified on all %d ranks (rows bitwise, kv amax-rel max %.2e); "
                "Engram kv now all-to-all + replicated wkv on each rank's rows (no all-gather)", TAG, len(votes),
                max(e for _, e in votes))
    return kv_new
