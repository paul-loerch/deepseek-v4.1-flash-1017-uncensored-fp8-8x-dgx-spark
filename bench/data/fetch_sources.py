#!/usr/bin/env python3
"""Download the pinned Hugging Face source files that the prep scripts need into the cache directory.

    python3 fetch_sources.py [--only w2,w3] [--cache-dir DIR]

Groups (each file is fetched at the revision pinned in _common.py):
  w2  nebius/SWE-rebench-openhands-trajectories   recorded coding-agent sessions        -> prep_w2_agent.py
  w3  zai-org/LongBench-v2, princeton-nlp/SWE-bench_Lite_bm25_27K   long documents      -> prep_w3_docqa.py
  w1  allenai/WildChat-4.8M (11 of its 86 parquet shards)   real chat prompts           -> prep_w1_chat.py

The benchmark tools in bench/tools need w2 and w3 only, which is the default. w1 is optional: nothing in bench/tools reads
it, and WildChat contains real user prompts (see data/README.md). Everything is about 3.5 GB; run it detached on a
slow link. Re-running is cheap: files already in the cache are not downloaded again. A dataset that Hugging Face reports as
gated needs its terms accepted on the dataset page and a login first (huggingface-cli login, or HF_TOKEN).
"""
from __future__ import annotations

import argparse
import sys
import time

import _common as C
from _common import WILDCHAT_SHARDS, hf_file

# (group, repo, file) in download order
FILES: list[tuple[str, str, str]] = [
    ("w3", "princeton-nlp/SWE-bench_Lite_bm25_27K", "data/test-00000-of-00001.parquet"),
    ("w2", "nebius/SWE-rebench-openhands-trajectories", "tools.json"),
    ("w2", "nebius/SWE-rebench-openhands-trajectories", "config.toml"),
    ("w3", "zai-org/LongBench-v2", "data.json"),
    *[("w1", "allenai/WildChat-4.8M", f"data/train-{i:05d}-of-00086.parquet") for i in WILDCHAT_SHARDS],
    ("w2", "nebius/SWE-rebench-openhands-trajectories", "trajectories.parquet"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="w2,w3", help="comma-separated groups to fetch: w1, w2, w3 (default: w2,w3)")
    C.add_location_args(ap)
    args = ap.parse_args()
    groups = {g.strip() for g in args.only.split(",") if g.strip()}
    unknown = groups - {"w1", "w2", "w3"}
    if unknown or not groups:
        ap.error(f"--only: unknown group(s) {sorted(unknown) or args.only!r}; choose from w1, w2, w3")
    C.configure(args)
    failed = 0
    for group, repo, fn in FILES:
        if group not in groups:
            continue
        t0 = time.time()
        try:
            p = hf_file(repo, fn)
            print(f"ok   {repo}/{fn} -> {p} ({p.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:  # keep going; report at the end
            failed += 1
            print(f"FAIL {repo}/{fn}: {exc!r}", flush=True)
    print("done, failures:", failed, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
