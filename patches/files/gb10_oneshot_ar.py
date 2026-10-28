"""GB10 one-shot all-reduce for small tensor-parallel messages (recipe patch 012, opt-in VLLM_GB10_ONESHOT_AR=1).

vLLM nightly e4a5b55 port: the code of this module is unchanged from the v0.32.0 port. The hooks in
distributed/device_communicators/cuda_communicator.py (patches/012-gb10-oneshot-allreduce.sh) sit in CudaCommunicator.__init__
(after the PyNcclCommunicator block) and as the FIRST statement of CudaCommunicator.all_reduce, i.e. before the dispatch
chain (NCCL symmetric memory, quick reduce, FlashInfer PCIe-IPC / FlashInfer all-reduce, AITER, custom all-reduce, torch symm
mem, pynccl): an eligible message never reaches any of those backends whatever VLLM_ALLREDUCE_USE_FLASHINFER says. The recipe
still sets VLLM_ALLREDUCE_USE_FLASHINFER=0 (default 1) and --disable-custom-all-reduce so that the messages this module does
NOT take (bf16 > 64 KiB: prefill chunks; every non-bf16 or non-contiguous tensor) fall through to pynccl, instead of
FlashInfer's all-reduce (for a multi-node TP group FlashInfer supports only its MNNVL backend, which needs NVSwitch multicast;
GB10 has none).

WHY: a TP=8 decode step runs ~90 all-reduces of [tokens, hidden] bf16 (embedding, then attention output projection and MoE
combine of every layer) over 100 GbE RoCE without GPUDirect RDMA. NCCL's tree takes ~6 sequential host-staged hops (73 / 91 /
109 us at 8 / 16 / 32 KiB, 8 nodes). Sending the whole vector to all 7 peers at once (grouped ncclSend/ncclRecv) and summing
locally takes one network step (40 / 37 / 52 us measured with nccl-tests alltoall). With hidden size 5120 one token is 10 KiB,
so the default 64 KiB limit covers decode steps of up to 6 tokens: one request with DSpark's 5 draft tokens (6 verified
positions = 60 KiB). Larger decode batches (12 tokens = 120 KiB, ...) stay on NCCL unless VLLM_GB10_ONESHOT_AR_MAX_BYTES is
raised (the receive buffer is world_size x max_bytes; the crossover against the tree is not measured above 64 KiB).

HOW: every rank sends its input to each peer and receives each peer's input into a persistent buffer, then one Triton
kernel adds the ws vectors in fp32 in canonical rank order 0..ws-1 and rounds to bf16 once. Every rank therefore produces
bit-identical output (required: mHC, MoE routers and the attention latents are replicated per rank and must not drift
apart), and the result is closer to the exact sum than NCCL's per-hop bf16 adds.

SCOPE: only the tensor-parallel group (group name 'tp:<n>'), bf16, contiguous, 0 < bytes <= VLLM_GB10_ONESHOT_AR_MAX_BYTES
(default 65536). Everything else keeps the normal vLLM path. The decision depends only on shape and dtype, so all ranks
always take the same path (a mixed choice would deadlock). CUDA-graph capturable: NCCL p2p inside a group is capturable, the
receive buffer is allocated before capture, and the output comes from the graph pool like pynccl.all_reduce's.
Fail-closed: the constructor runs a collective warm-up (which also opens the lazy NCCL p2p connections and compiles the
Triton kernel before any graph capture) and raises if the result is wrong on any element.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

_ENV = "VLLM_GB10_ONESHOT_AR"
_ENV_MAX = "VLLM_GB10_ONESHOT_AR_MAX_BYTES"
_BLOCK = 1024


@triton.jit
def _oneshot_sum_kernel(x_ptr, buf_ptr, out_ptr, n, rank, stride, WS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for r in tl.static_range(WS):
        # Exactly one of the two loads is live per element: this rank's own input for r == rank, the received copy of
        # rank r's input otherwise. own + 0.0 and 0.0 + received are the same IEEE value, so all ranks add the same
        # numbers in the same order.
        own = tl.load(x_ptr + offs, mask=mask & (r == rank), other=0.0).to(tl.float32)
        got = tl.load(buf_ptr + r * stride + offs, mask=mask & (r != rank), other=0.0).to(tl.float32)
        acc += own + got
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


class GB10OneShotAllReduce:
    def __init__(self, pynccl_comm, device: torch.device, max_bytes: int):
        self.comm = pynccl_comm
        self.rank = pynccl_comm.rank
        self.world_size = pynccl_comm.world_size
        self.max_elems = max_bytes // 2
        self.buf = torch.empty((self.world_size, self.max_elems), dtype=torch.bfloat16, device=device)
        self._warmup()

    def should_use(self, x: torch.Tensor) -> bool:
        return x.dtype == torch.bfloat16 and x.is_contiguous() and 0 < x.numel() <= self.max_elems

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        r, ws = self.rank, self.world_size
        stream = current_stream()
        flat = x.view(-1)
        self.comm.group_start()
        for d in range(1, ws):
            dst, src = (r + d) % ws, (r - d) % ws
            self.comm.send(flat, dst, stream)
            self.comm.recv(self.buf[src, :n], src, stream)
        self.comm.group_end()
        out = torch.empty_like(x)
        _oneshot_sum_kernel[(triton.cdiv(n, _BLOCK),)](
            flat, self.buf, out.view(-1), n, r, self.buf.stride(0), WS=ws, BLOCK=_BLOCK
        )
        return out

    def _warmup(self) -> None:
        # Collective: every rank of the group constructs the communicator together. Each rank contributes (rank + 1); the
        # exact sum ws * (ws + 1) / 2 is representable in bf16 for ws <= 64.
        for n in (1, 4096, self.max_elems):
            x = torch.full((n,), float(self.rank + 1), dtype=torch.bfloat16, device=self.buf.device)
            out = self(x)
            torch.cuda.synchronize(self.buf.device)
            want = float(self.world_size * (self.world_size + 1) // 2)
            if not bool(torch.all(out == want)):
                raise RuntimeError(f"[patch-012] one-shot all-reduce warm-up wrong on rank {self.rank} (n={n})")


def maybe_create(pynccl_comm, device: torch.device, unique_name: str | None):
    if os.environ.get(_ENV, "0") != "1":
        return None
    if pynccl_comm is None or pynccl_comm.disabled or pynccl_comm.world_size < 2:
        return None
    if not unique_name or unique_name.split(":")[0] != "tp":     # exactly the TP group ("tp:0"), like v0.32.0's own gate
        return None
    max_bytes = int(os.environ.get(_ENV_MAX, "65536"))
    if max_bytes <= 0 or max_bytes % 2:
        raise ValueError(f"[patch-012] {_ENV_MAX} must be a positive even number of bytes, got {max_bytes}")
    op = GB10OneShotAllReduce(pynccl_comm, device, max_bytes)
    logger.info(
        "[patch-012] one-shot all-reduce ACTIVE for group %s (world %d, bf16 <= %d bytes; larger messages use NCCL)",
        unique_name, pynccl_comm.world_size, max_bytes,
    )
    return op
