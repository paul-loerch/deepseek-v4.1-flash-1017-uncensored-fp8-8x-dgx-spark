# SPDX-License-Identifier: Apache-2.0
# [patch-032 memstats] installed as vllm/_memstats.py by 032-memstats.sh (opt-in VLLM_GB10_MEMSTATS=<seconds>; logging only).
"""Periodic CUDA caching-allocator statistics of a worker (diagnostic, changes no computation).

Why: on GB10 every worker holds ~94 GiB of device memory, ~25 GiB more than weights + KV + graphs explain. These lines show how
much of it is live tensors (allocated) vs blocks the caching allocator keeps for reuse (reserved - allocated), and the peaks.
"""
from __future__ import annotations

import os
import threading
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
TAG = "[patch-032]"
ENV = "VLLM_GB10_MEMSTATS"
ENV_HIST = "VLLM_GB10_MEMSTATS_HISTORY"   # 1: record allocation stacks from load_model to warm-up, log the top sites once
_HIST = {"on": False}


def start_history() -> None:
    """At the start of Worker.load_model: record allocation stack traces (stopped again in start())."""
    if os.environ.get(ENV_HIST, "0").strip() not in ("1", "on", "true", "yes"):
        return
    try:
        torch.cuda.memory._record_memory_history(max_entries=500000)
        _HIST["on"] = True
        logger.info("%s recording allocation history from load_model", TAG)
    except Exception as e:  # noqa: BLE001
        logger.warning("%s history not available: %s", TAG, e)


def _site(frames) -> str:
    """The innermost vllm frame of an allocation, plus the next vllm frame outward for context."""
    vf = [f for f in frames if "/vllm/" in f.get("filename", "")]
    if not vf:
        f = frames[0] if frames else {}
        return f"{f.get('filename', '?').split('/')[-1]}:{f.get('line', 0)} {f.get('name', '?')}"
    def fmt(f):
        fn = f["filename"]
        return f"{fn[fn.index('/vllm/') + 6:]}:{f.get('line', 0)} {f.get('name', '?')}"
    return fmt(vf[0]) + (" <- " + fmt(vf[1]) if len(vf) > 1 else "")


def dump_history(label: str, top: int = 45) -> None:
    if not _HIST["on"]:
        return
    try:
        snap = torch.cuda.memory._snapshot()
        by_site: dict[str, list[int]] = {}
        total = 0
        for seg in snap.get("segments", []):
            for blk in seg.get("blocks", []):
                if blk.get("state") != "active_allocated":
                    continue
                size = int(blk.get("size", 0)); total += size
                key = _site(blk.get("frames") or [])
                e = by_site.setdefault(key, [0, 0]); e[0] += size; e[1] += 1
        logger.info("%s %s: live allocations %s GiB in %d sites; top %d by bytes:", TAG, label, _g(total), len(by_site), top)
        for key, (b, n) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[:top]:
            logger.info("%s   %8.3f GiB  %6d x  %s", TAG, b / 2**30, n, key)
    except Exception as e:  # noqa: BLE001
        logger.warning("%s history dump failed: %s", TAG, e)
    finally:
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception:  # noqa: BLE001
            pass
        _HIST["on"] = False


def _g(b: int) -> str:
    return f"{b / 2**30:.2f}"


def snapshot(label: str, dev: torch.device) -> None:
    s = torch.cuda.memory_stats(dev)
    free, total = torch.cuda.mem_get_info(dev)
    logger.info("%s %s: allocated %s GiB (peak %s), reserved %s GiB (peak %s), inactive split %s GiB, alloc retries %d, "
                "cudaMemGetInfo free %s of %s GiB", TAG, label,
                _g(s.get("allocated_bytes.all.current", 0)), _g(s.get("allocated_bytes.all.peak", 0)),
                _g(s.get("reserved_bytes.all.current", 0)), _g(s.get("reserved_bytes.all.peak", 0)),
                _g(s.get("inactive_split_bytes.all.current", 0)), s.get("num_alloc_retries", 0), _g(free), _g(total))


def start(dev: torch.device) -> None:
    try:
        period = float(os.environ.get(ENV, "0") or 0)
    except ValueError:
        period = 0.0
    dump_history("before warm-up (weights + KV + init buffers)")
    if period <= 0:
        return
    snapshot("before warm-up", dev)

    def loop() -> None:
        while True:
            time.sleep(period)
            try:
                snapshot("periodic", dev)
            except Exception as e:  # noqa: BLE001 -- diagnostics must never take the worker down
                logger.warning("%s stopped: %s", TAG, e)
                return

    threading.Thread(target=loop, daemon=True, name="p032-memstats").start()
