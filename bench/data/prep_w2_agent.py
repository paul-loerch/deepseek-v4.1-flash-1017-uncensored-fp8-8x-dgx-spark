#!/usr/bin/env python3
"""W2 — coding-agent sessions from nebius/SWE-rebench-openhands-trajectories.

Filters:
  * resolved == 1 and exit_status contains "submit"
  * at most one trajectory per repository (the only cross-session shared prefix is the
    system prompt + tools preamble)
  * cut each session at the last assistant turn whose prompt is ≤ 48K (49152) tokens of the model under test
    (a compaction boundary that keeps sessions inside the safety envelope)
  * ≥ 15 assistant turns after the cut
  * 64 sessions, 32 with final context 16–32K and 32 with 32–48K, spread round-robin across
    repository languages (inferred from the files the resolving patch touches)

Stored per session: the full cut message array (system, task, assistant turns with tool calls,
tool observations), the tools schema, and per-turn token accounting so the driver can
estimate prompt sizes for admission control without tokenizing on the hot path:
  assistant_idx[k]  message index of recorded assistant turn k
  prefix_tokens[k]  templated prompt tokens to generate turn k (history before it + gen prompt)
  asst_tokens[k]    tokens of the recorded assistant message k as rendered in history
  obs_tokens[k]     tokens of the messages between assistant k and k+1 (tool results, user nudges)

The preamble is OpenHands' system prompt + tools.json; its size is recorded in the manifest.

    python3 prep_w2_agent.py [--per-bucket 32] [--data-dir DIR] [--cache-dir DIR] [--tokenizer DIR]

Needs the w2 files from fetch_sources.py (the default). The result, w2_agent-v1.jsonl, is what decode-bench-w2.py replays
and what ctx-curve.py appends to the W3 text for prompts beyond about 0.9 M tokens.
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_MISSING = None
try:
    import pyarrow.parquet as pq
except ImportError as _exc:  # reported by main(), after --help has been handled
    pq = None
    _MISSING = _exc.name

import _common as C
from _common import (REVISIONS, _template_messages, file_entry, hf_file, histogram,
                     stable_hash, token_stats, tokenizer, update_manifest, write_jsonl)

REPO = "nebius/SWE-rebench-openhands-trajectories"
MAX_CTX = 49152
BUCKETS = {"16-32K": (16384, 32768), "32-48K": (32768, MAX_CTX + 1)}
MIN_TURNS = 15
REASONING_EFFORT = "high"

EXT_LANG = {".py": "python", ".pyx": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
            ".tsx": "typescript", ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".rb": "ruby",
            ".php": "php", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp",
            ".scala": "scala", ".swift": "swift", ".sh": "shell", ".r": "r", ".jl": "julia"}


def strip_none(obj: Any) -> Any:
    """Parquet structs fill absent keys with None; drop them to recover the original JSON."""
    if isinstance(obj, dict):
        return {k: strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [strip_none(v) for v in obj]
    return obj


def patch_language(patch: str | None) -> str:
    exts = Counter()
    for path in re.findall(r"^diff --git a/(\S+)", patch or "", re.MULTILINE):
        suffix = Path(path).suffix.lower()
        if suffix in EXT_LANG and "test" not in path.lower():
            exts[EXT_LANG[suffix]] += 1
    if not exts:
        for path in re.findall(r"^diff --git a/(\S+)", patch or "", re.MULTILINE):
            suffix = Path(path).suffix.lower()
            if suffix in EXT_LANG:
                exts[EXT_LANG[suffix]] += 1
    return exts.most_common(1)[0][0] if exts else "other"


def render(messages: list[dict], tools: list[dict], agp: bool) -> str:
    return tokenizer().apply_chat_template(_template_messages(messages), tools=tools, tokenize=False,
                                           add_generation_prompt=agp, reasoning_effort=REASONING_EFFORT)


class Accountant:
    """Token counts at message boundaries of one rendered trajectory (one tokenization)."""

    def __init__(self, messages: list[dict], tools: list[dict]):
        self.messages, self.tools = messages, tools
        self.full = render(messages, tools, agp=False)
        enc = tokenizer()(self.full, add_special_tokens=False, return_offsets_mapping=True)
        self.starts = [s for s, _ in enc["offset_mapping"]]
        self.n_tokens = len(self.starts)
        self.unstable = 0
        self._cache: dict[int, int] = {}
        gp = render(messages[:1], tools, agp=True)
        self.gen_prompt_tokens = (len(tokenizer().encode(gp, add_special_tokens=False))
                                  - len(tokenizer().encode(render(messages[:1], tools, agp=False),
                                                           add_special_tokens=False)))

    def tokens_before(self, i: int) -> int:
        """Tokens of the rendered history messages[:i] (no generation prompt)."""
        if i in self._cache:
            return self._cache[i]
        text = render(self.messages[:i], self.tools, agp=False)
        if self.full.startswith(text):
            n = bisect.bisect_left(self.starts, len(text))
        else:  # template not prefix-stable at this boundary: tokenize directly
            self.unstable += 1
            n = len(tokenizer().encode(text, add_special_tokens=False))
        self._cache[i] = n
        return n


def account(messages: list[dict], tools: list[dict]) -> dict | None:
    """Per-turn token accounting and the ≤48K cut; None if the session does not qualify."""
    a_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if len(a_idx) < MIN_TURNS:
        return None
    acc = Accountant(messages, tools)
    prefix, asst, obs = [], [], []
    kept = 0
    for k, a in enumerate(a_idx):
        p = acc.tokens_before(a) + acc.gen_prompt_tokens
        if p > MAX_CTX:
            break
        prefix.append(p)
        kept = k + 1
    if kept < MIN_TURNS:
        return None
    for k in range(kept):
        a = a_idx[k]
        asst.append(acc.tokens_before(a + 1) - acc.tokens_before(a))
        if k + 1 < kept:
            obs.append(acc.tokens_before(a_idx[k + 1]) - acc.tokens_before(a + 1))
        else:
            obs.append(0)  # observations after the last kept turn are dropped (next prompt > 48K)
    cut = a_idx[kept - 1] + 1  # keep the last kept recorded assistant turn (teacher-forcing fallback)
    return {"messages": messages[:cut], "assistant_idx": a_idx[:kept], "prefix_tokens": prefix,
            "asst_tokens": asst, "obs_tokens": obs, "final_context_tokens": prefix[-1],
            "n_turns": kept, "unstable_boundaries": acc.unstable, "full_tokens": acc.n_tokens}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-bucket", type=int, default=32)
    ap.add_argument("--max-candidates", type=int, default=500)
    C.add_location_args(ap)
    args = ap.parse_args()
    if pq is None:
        sys.exit(f"missing Python package: {_MISSING} (pip install pyarrow huggingface_hub transformers jinja2)")
    C.configure(args)
    OUT = C.DATA_DIR / "w2_agent-v1.jsonl"

    path = hf_file(REPO, "trajectories.parquet")
    tools_json = json.loads(hf_file(REPO, "tools.json").read_text())
    tool_names = sorted(t["function"]["name"] for t in tools_json)
    pf = pq.ParquetFile(path)
    meta_cols = ["trajectory_id", "instance_id", "repo", "resolved", "exit_status", "model_patch"]
    meta = pf.read(columns=meta_cols).to_pylist()
    rg_starts = []
    off = 0
    for rg in range(pf.num_row_groups):
        rg_starts.append(off)
        off += pf.metadata.row_group(rg).num_rows
    reasons: Counter = Counter(total=len(meta))

    # 1) cheap filters, one candidate list per repo in a fixed pseudo-random order
    by_repo: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(meta):
        if r["resolved"] != 1 or "submit" not in (r["exit_status"] or ""):
            reasons["not_resolved_submit"] += 1
            continue
        by_repo[r["repo"]].append(i)
    for rows in by_repo.values():
        rows.sort(key=lambda i: stable_hash("w2", meta[i]["trajectory_id"]))
    repo_lang = {repo: patch_language(meta[rows[0]]["model_patch"]) for repo, rows in by_repo.items()}
    # 2) stratify repos by language (round-robin over languages, each list in hash order)
    lang_repos: dict[str, list[str]] = defaultdict(list)
    for repo in sorted(by_repo, key=lambda r: stable_hash("w2-repo", r)):
        lang_repos[repo_lang[repo]].append(repo)
    order: list[str] = []
    langs = sorted(lang_repos, key=lambda l: (-len(lang_repos[l]), l))
    while any(lang_repos.values()):
        for l in langs:
            if lang_repos[l]:
                order.append(lang_repos[l].pop(0))
    print(f"[w2] {len(meta)} trajectories, {len(by_repo)} repos with resolved submits; "
          f"languages: {dict(Counter(repo_lang.values()))}", flush=True)

    # 3) read the top-ranked candidates' trajectories, each row group read once
    cand_rows = [by_repo[repo][0] for repo in order][: args.max_candidates]
    wanted: dict[int, list[int]] = defaultdict(list)
    for i in cand_rows:
        wanted[bisect.bisect_right(rg_starts, i) - 1].append(i)
    trajs: dict[int, list[dict]] = {}
    tools_mismatch = 0
    for rg in sorted(wanted):
        t = pf.read_row_group(rg, columns=["trajectory", "tools"])
        for i in wanted[rg]:
            j = i - rg_starts[rg]
            trajs[i] = strip_none(t.column("trajectory")[j].as_py())
            tools = strip_none(t.column("tools")[j].as_py())
            if sorted(x["function"]["name"] for x in tools) != tool_names:
                tools_mismatch += 1
        del t
    print(f"[w2] loaded {len(trajs)} candidate trajectories", flush=True)

    # accounting in stratified rank order until both buckets are full
    results: dict[int, dict] = {}
    per_bucket: Counter = Counter()
    picked: list[int] = []
    examined = 0
    unstable = 0
    for n, i in enumerate(cand_rows):
        if all(per_bucket[b] >= args.per_bucket for b in BUCKETS):
            break
        examined = n + 1
        traj = trajs.pop(i)
        if sum(1 for m in traj if m.get("role") == "assistant") < MIN_TURNS:
            reasons["few_turns"] += 1
            continue
        acc = account(traj, tools_json)
        if acc is None:
            reasons["few_turns_under_48k"] += 1
            continue
        ctx = acc["final_context_tokens"]
        bucket = next((b for b, (lo, hi) in BUCKETS.items() if lo <= ctx < hi), None)
        if bucket is None:
            reasons["final_ctx_below_16k"] += 1
            continue
        if per_bucket[bucket] >= args.per_bucket:
            reasons[f"bucket_full_{bucket}"] += 1
            continue
        results[i] = {**acc, "bucket": bucket}
        unstable += acc["unstable_boundaries"]
        per_bucket[bucket] += 1
        picked.append(i)
        if len(picked) % 8 == 0:
            print(f"[w2] candidate {n}: picked {dict(per_bucket)}", flush=True)
    trajs.clear()

    rows = []
    for i in picked:
        r = results[i]
        m = meta[i]
        rows.append({"id": m["trajectory_id"], "instance_id": m["instance_id"], "repo": m["repo"],
                     "language": repo_lang[m["repo"]], "bucket": r["bucket"], "n_turns": r["n_turns"],
                     "final_context_tokens": r["final_context_tokens"], "tools": tools_json,
                     "messages": r["messages"], "assistant_idx": r["assistant_idx"],
                     "prefix_tokens": r["prefix_tokens"], "asst_tokens": r["asst_tokens"],
                     "obs_tokens": r["obs_tokens"]})
    rows.sort(key=lambda r: stable_hash("w2-order", r["id"]))
    n = write_jsonl(OUT, rows)

    # preamble size (system prompt + tools, as every session starts)
    sys_msgs = {r["messages"][0]["content"] for r in rows if r["messages"][0].get("role") == "system"}
    first_sys = rows[0]["messages"][0]
    acc_pre = len(tokenizer().encode(render([first_sys], tools_json, agp=False), add_special_tokens=False))
    first_prompt = [r["prefix_tokens"][0] for r in rows]
    entry = {
        "workload": "w2", "source": REPO, "revision": REVISIONS[REPO],
        "source_files": ["trajectories.parquet", "tools.json"],
        "filters": {"resolved": 1, "exit_status": "contains 'submit'", "one_per_repo": True,
                    "max_context_tokens": MAX_CTX, "min_assistant_turns": MIN_TURNS,
                    "buckets": {b: list(v) for b, v in BUCKETS.items()},
                    "language": "round-robin over languages inferred from the resolving patch"},
        "counts": {"sessions": n, "buckets": dict(per_bucket),
                   "languages": dict(Counter(r["language"] for r in rows)),
                   "candidates_examined": examined, "template_unstable_boundaries": unstable, "tools_schema_mismatch": tools_mismatch,
                   "distinct_system_prompts": len(sys_msgs)},
        "rejections": dict(reasons),
        "preamble": {"source": "OpenHands system prompt + tools.json",
                     "tokens": acc_pre, "reasoning_effort": REASONING_EFFORT},
        "token_stats": {"final_context": token_stats([r["final_context_tokens"] for r in rows]),
                        "first_prompt": token_stats(first_prompt),
                        "turns": token_stats([r["n_turns"] for r in rows]),
                        "obs_tokens": token_stats([t for r in rows for t in r["obs_tokens"][:-1]]),
                        "recorded_asst_tokens": token_stats([t for r in rows for t in r["asst_tokens"]])},
        "histograms": {"final_context": histogram([r["final_context_tokens"] for r in rows],
                                                  [16384, 20480, 24576, 28672, 32768, 36864, 40960, 45056, 49153])},
        **file_entry(OUT),
    }
    update_manifest("w2_agent-v1", entry)
    print(f"[w2] wrote {n} sessions -> {OUT}")
    print(json.dumps({k: entry[k] for k in ("counts", "preamble", "token_stats", "rejections")}, indent=1))
    return 0 if all(per_bucket[b] >= args.per_bucket for b in BUCKETS) else 2


if __name__ == "__main__":
    sys.exit(main())
