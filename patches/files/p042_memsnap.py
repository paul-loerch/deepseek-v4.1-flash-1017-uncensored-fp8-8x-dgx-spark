# SPDX-License-Identifier: Apache-2.0
"""Patch 042 -- on-demand caching-allocator snapshot for patch 032's memstats thread (logging only).

Why: at steady state every rank's CUDA caching allocator holds 4.5-6 GiB more than its live tensors (reserved 92.1 vs allocated
86.1 GiB on rank 0 of b067; 3.7 GiB above the peak allocation), memory that on GB10 comes straight out of the KV budget.  The
periodic line cannot say where it sits; this dump groups the segments by (stream, pool), so per-stream caching, graph pools and
private pools can be told apart.

Edit (vllm/_memstats.py as installed by 032): when the trigger file exists (VLLM_GB10_MEMSNAP_TRIGGER, default
/cache/huggingface/dsv41-memsnap.trigger, i.e. per node), the periodic thread deletes it and logs one summary of
torch.cuda.memory_snapshot().  Costs nothing until triggered; any error ends only the diagnostics thread (032's try/except).
"""
ID = "042"
TAG = "memsnap"
M = f"# [patch-{ID} {TAG}]"

# sha256 of files/memstats.py (patch 032, installed as vllm/_memstats.py)
MEMSTATS_SHA = "45574ef0789d9d5357e6890662b42457b9b9f29ce393607ad7d9e30809ac7b37"
_P = "_memstats.py"

OLD_START = "def start(dev: torch.device) -> None:\n"
NEW_START = (
    f"{M} on demand: if the trigger file exists, delete it and log the allocator's segments grouped by (stream, pool)\n"
    "_P042_TRIGGER = os.environ.get(\"VLLM_GB10_MEMSNAP_TRIGGER\", \"/cache/huggingface/dsv41-memsnap.trigger\")\n"
    "\n"
    "\n"
    "def _p042_maybe_dump() -> None:\n"
    "    if not _P042_TRIGGER or not os.path.exists(_P042_TRIGGER):\n"
    "        return\n"
    "    try:\n"
    "        os.unlink(_P042_TRIGGER)\n"
    "    except OSError:\n"
    "        pass\n"
    "    t0 = time.perf_counter()\n"
    "    segs = torch.cuda.memory_snapshot()\n"
    "    ms = (time.perf_counter() - t0) * 1e3\n"
    "    by: dict[tuple, list[int]] = {}\n"
    "    for s in segs:\n"
    "        key = (int(s.get(\"stream\", 0) or 0), str(s.get(\"segment_pool_id\")), str(s.get(\"segment_type\", \"?\")),\n"
    "               bool(s.get(\"is_expandable\", False)))\n"
    "        e = by.setdefault(key, [0, 0, 0, 0, 0, 0])  # segments, reserved, allocated, live blocks, inactive, largest inactive\n"
    "        e[0] += 1\n"
    "        e[1] += int(s.get(\"total_size\", 0))\n"
    "        e[2] += int(s.get(\"allocated_size\", 0))\n"
    "        for b in s.get(\"blocks\", []):\n"
    "            if b.get(\"state\") == \"active_allocated\":\n"
    "                e[3] += 1\n"
    "            elif b.get(\"state\") == \"inactive\":\n"
    "                e[4] += int(b.get(\"size\", 0))\n"
    "                e[5] = max(e[5], int(b.get(\"size\", 0)))\n"
    "    logger.info(\"%s [patch-042] allocator snapshot (%.0f ms): %d segments in %d (stream, pool) groups, reserved %s GiB, \"\n"
    "                \"allocated %s GiB\", TAG, ms, len(segs), len(by), _g(sum(v[1] for v in by.values())),\n"
    "                _g(sum(v[2] for v in by.values())))\n"
    "    for (stream, pool, typ, exp), v in sorted(by.items(), key=lambda kv: -kv[1][1])[:16]:\n"
    "        logger.info(\"%s [patch-042]   stream %#x pool %s %s%s: %d segments, reserved %s GiB, allocated %s GiB, %d live \"\n"
    "                    \"blocks, inactive %s GiB (largest %s GiB)\", TAG, stream, pool, typ, \" expandable\" if exp else \"\",\n"
    "                    v[0], _g(v[1]), _g(v[2]), v[3], _g(v[4]), _g(v[5]))\n"
    "\n"
    "\n"
    "def start(dev: torch.device) -> None:\n"
)

OLD_LOOP = "                snapshot(\"periodic\", dev)\n"
NEW_LOOP = (
    "                snapshot(\"periodic\", dev)\n"
    f"                {M} logs (and deletes) the trigger file's snapshot when present\n"
    "                _p042_maybe_dump()\n"
)

EDITS = [(_P, MEMSTATS_SHA, OLD_START, NEW_START), (_P, MEMSTATS_SHA, OLD_LOOP, NEW_LOOP)]
