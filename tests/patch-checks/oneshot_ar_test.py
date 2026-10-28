"""oneshot_ar_test.py — standalone multi-rank validation of patch 012 (gb10_oneshot_ar.py) BEFORE it serves traffic.
NEEDS THE GPUs: it is not part of the CPU checks (p012_cpu_check.py in this directory). Ran on eight GB10 ranks on 2026-10-26: PASS
(bit-exact vs the canonical fp32 sum, identical output bits on all ranks; timings in PATCHES.md).

Run one process per rank, in containers of the recipe image on idle nodes with the patch directory mounted at /opt/recipe-patches
(copy this file into the container first, e.g. `docker cp tests/patch-checks/oneshot_ar_test.py <container>:/tmp/`, then
`docker exec -e RANK=r -e WORLD_SIZE=8 -e MASTER_ADDR=<rank-0 fabric IP> -e MASTER_PORT=<port> <rank-r container> python3
/tmp/oneshot_ar_test.py`). Uses the container's NCCL env and vLLM's PyNCCL.
Checks: (1) bit-exact vs a local canonical-order fp32 reference sum, for several sizes; (2) identical output bits on all ranks;
(3) max |error| vs the exact (fp64) sum, one-shot vs NCCL's tree all-reduce; (4) per-op time eager and in a CUDA graph (96 ops
captured, 20 replays) for the DeepSeek-V4.1 decode shapes [1|5|6, 5120] bf16 (one DSpark verification step of one request is
6 tokens = 60 KiB), one-shot vs NCCL.
"""

import hashlib
import os
import sys
import time

import torch
import torch.distributed as dist

# the module under test is patch 012's: inside a rank container the patch directory is mounted at /opt/recipe-patches
for _d in (os.environ.get("PATCH_FILES_DIR"), "/opt/recipe-patches/files",
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "patches", "files")):
    if _d and os.path.isfile(os.path.join(_d, "gb10_oneshot_ar.py")):
        sys.path.insert(0, _d)
        break
import gb10_oneshot_ar as m  # noqa: E402
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # noqa: E402

rank, ws = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
dist.init_process_group("gloo", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                        rank=rank, world_size=ws)
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
comm = PyNcclCommunicator(group=dist.group.WORLD, device=dev)
os.environ["VLLM_GB10_ONESHOT_AR"] = "1"
op = m.maybe_create(comm, dev, "tp:selftest")  # includes the self-checking warm-up
assert op is not None


def inputs(n):
    xs = []
    for r in range(ws):
        g = torch.Generator().manual_seed(1000 * r + n)
        xs.append((torch.randn(n, generator=g) * 3).to(torch.bfloat16))
    return xs


lines, hashes, bad = [], [], 0
for n in (1, 7, 5120, 30720, 32768):
    xs = inputs(n)
    x = xs[rank].to(dev)
    out = op(x)
    torch.cuda.synchronize()
    acc = torch.zeros(n, dtype=torch.float32)
    for r in range(ws):
        acc += xs[r].float()
    ref = acc.to(torch.bfloat16)
    exact = sum(t.double() for t in xs)
    tree = comm.all_reduce(x)
    torch.cuda.synchronize()
    eq = torch.equal(out.cpu(), ref)
    bad += 0 if eq else 1
    e1 = (out.cpu().double() - exact).abs().max().item()
    e2 = (tree.cpu().double() - exact).abs().max().item()
    lines.append(f"n={n:6d} bit-exact={eq} max|err| one-shot {e1:.4f} vs NCCL tree {e2:.4f}")
    hashes.append(hashlib.sha1(out.cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:12])

all_hashes = [None] * ws
dist.all_gather_object(all_hashes, hashes)
same = all(h == all_hashes[0] for h in all_hashes)


def eager_us(fn, x, iters=300):
    for _ in range(20):
        fn(x)
    torch.cuda.synchronize()
    dist.barrier()
    t = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def graph_us(fn, x, per_graph=96, replays=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(per_graph):
            fn(x)
    torch.cuda.synchronize()
    dist.barrier()
    g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    t = time.perf_counter()
    for _ in range(replays):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (replays * per_graph) * 1e6


timing = []
for tokens in (6, 1, 5, 1, 6, 2):
    x = torch.randn(tokens, 5120, device=dev).to(torch.bfloat16)
    row = [tokens, eager_us(lambda t: comm.all_reduce(t), x), eager_us(op, x),
           graph_us(lambda t: comm.all_reduce(t), x), graph_us(op, x)]
    timing.append(row)

if rank == 0:
    print("CORRECTNESS")
    for ln in lines:
        print("  " + ln)
    print(f"  identical output bits on all {ws} ranks: {same}")
    print("TIMING per op (us)   tokens  bytes   NCCL eager  one-shot eager   NCCL graph  one-shot graph")
    for tokens, a, b, c, d in timing:
        print(f"                     {tokens:6d} {tokens * 10240:7d} {a:11.1f} {b:15.1f} {c:12.1f} {d:15.1f}")
    print("RESULT", "PASS" if (bad == 0 and same) else "FAIL")
dist.barrier()
dist.destroy_process_group()
