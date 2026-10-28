# SPDX-License-Identifier: Apache-2.0
# [patch-031 sched-ctl] installed as vllm/_sched_ctl.py by 031-sched-ctl.sh (opt-in VLLM_GB10_SCHED_CTL=<file>).
"""Runtime scheduler knobs for same-boot A/B tests (rank 0's EngineCore only; scheduling only, every output stays exact).

The control file holds `key=value` tokens; it is re-read when its mtime changes (checked at most once a second, at the start of
Scheduler.schedule):
    interval=N        prefill_schedule_interval: admit prefill once every N engine steps while requests are decoding (1 = off)
    long_prefill=T    long_prefill_token_threshold: cap a long prefill's chunk at T tokens while other requests are active
                      (0 = off; the nightly never applies it to a lone request)
A malformed file is refused with an error line and the previous values stay.
"""
from __future__ import annotations

import os
import time

from vllm.logger import init_logger

logger = init_logger(__name__)
TAG = "[patch-031]"
ENV = "VLLM_GB10_SCHED_CTL"
PATH = os.environ.get(ENV) or None
_STATE = {"mtime": None, "next": 0.0}


def parse(text: str) -> tuple[int, int]:
    kv = {}
    for tok in text.split():
        k, sep, v = tok.partition("=")
        if not sep or k not in ("interval", "long_prefill"):
            raise ValueError(f"bad token {tok!r}")
        kv[k] = int(v)
    interval, thr = kv.get("interval", 1), kv.get("long_prefill", 0)
    if not (1 <= interval <= 64 and 0 <= thr <= (1 << 20)):
        raise ValueError(f"out of range: interval={interval} long_prefill={thr}")
    return interval, thr


def poll(sched) -> None:
    if PATH is None:
        return
    now = time.monotonic()
    if now < _STATE["next"]:
        return
    _STATE["next"] = now + 1.0
    try:
        mtime = os.stat(PATH).st_mtime_ns
    except OSError:
        return
    if mtime == _STATE["mtime"]:
        return
    _STATE["mtime"] = mtime
    try:
        with open(PATH) as f:
            interval, thr = parse(f.read())
    except Exception as e:  # noqa: BLE001 -- keep the previous values
        logger.error("%s control file %s refused: %s", TAG, PATH, e)
        return
    if sched.parallel_config.data_parallel_size == 1:
        sched.local_prefill_interval = interval
    sched.scheduler_config.long_prefill_token_threshold = thr
    logger.info("%s scheduler: prefill_schedule_interval=%d long_prefill_token_threshold=%d", TAG,
                sched.local_prefill_interval, sched.scheduler_config.long_prefill_token_threshold)
