#!/usr/bin/env python3
"""W1 — real multi-turn chat from allenai/WildChat-4.8M (user turns only).

WildChat contains real user prompts. Only the user turns are written to the prepared file (assistant text is read
for timing fields only and never stored), conversations flagged toxic or redacted are dropped, and the prepared file
is for local use: it is git-ignored and must not be redistributed. See data/README.md.

Filters:
  * conversation-level toxic=false and redacted=false, no message flagged toxic/redacted
  * language English, plus a configurable German share (default 20 %)
  * 2–6 user turns; first user message ≥ 20 tokens; every user message ≤ 4096 tokens
  * dedupe: conversation_hash + normalised first message + MinHash/LSH near-duplicates of the
    first user message (WildChat repeats prompts heavily; duplicates would fake cache hits)
  * 50/50 technical (fenced code or code vocabulary) vs general, per language
  * 600 conversations, fixed order (seed 20260925)

Only user turns are stored; assistant turns are generated live by the model under test.
Also writes the fixed ~600-token chat system prompt used by W1/W3 and fits the WildChat
think-time distribution (user send − previous assistant reply) for the manifest.

    python3 prep_w1_chat.py [--n 600] [--german-share 0.2] [--data-dir DIR] [--cache-dir DIR] [--tokenizer DIR]

Needs the w1 files from fetch_sources.py --only w1. The result, w1_chat-v1.jsonl, is not read by any tool in bench/tools.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_MISSING = None
try:
    import numpy as np
    import pyarrow.parquet as pq
except ImportError as _exc:  # reported by main(), after --help has been handled
    np = pq = None
    _MISSING = _exc.name

import _common as C
from _common import (REVISIONS, SYSTEM_PROMPT, WILDCHAT_SHARDS, chat_tokens, file_entry, hf_file,
                     histogram, rng, stable_hash, text_tokens, token_stats, update_manifest,
                     write_jsonl)

REPO = "allenai/WildChat-4.8M"

TECH_RE = re.compile(
    r"```|\bdef \w+\(|\bclass \w+|\bimport \w+|#include|\bfunction\s*\w*\(|console\.log|SELECT .+ FROM|"
    r"\bpublic static\b|\bstd::|\bnpm\b|\bpip install\b|\bdocker\b|\bkubectl\b|\bgit (?:clone|commit|push)|"
    r"\bpython\b|\bjavascript\b|\btypescript\b|\bjava\b|\bc\+\+|\brust\b|\bgolang\b|\bsql\b|\bregex\b|"
    r"\bbash\b|\blinux\b|\bapi\b|\bjson\b|\bhtml\b|\bcss\b|\breact\b|\bexception\b|\btraceback\b|"
    r"\berror:|\bcompile|\balgorithm\b|\bfunction\b|\bvariable\b|\bdatabase\b|\bserver\b|\bscript\b",
    re.IGNORECASE)

LANGS = {"English": "en", "German": "de"}


# --------------------------------------------------------------------------- MinHash / LSH
_P = (1 << 61) - 1
_NPERM = 64
_BANDS, _ROWS = 16, 4
_AB = None


def _tables():
    """The two hash-parameter vectors, drawn once with the same seed as always."""
    global _AB
    if _AB is None:
        r = np.random.default_rng(20260925)
        _AB = (r.integers(1, _P - 1, size=_NPERM, dtype=np.uint64),
               r.integers(0, _P - 1, size=_NPERM, dtype=np.uint64))
    return _AB


def _shingles(text: str, k: int = 5) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {int.from_bytes(hashlib.blake2b(" ".join(words).encode(), digest_size=8).digest(), "big") >> 3}
    return {int.from_bytes(hashlib.blake2b(" ".join(words[i:i + k]).encode(), digest_size=8).digest(), "big") >> 3
            for i in range(len(words) - k + 1)}


def minhash(text: str) -> np.ndarray:
    _A, _B = _tables()
    sh = np.fromiter(_shingles(text), dtype=np.uint64)
    # (a*x + b) mod p with python ints would be slow; uint64 wraps, which is fine for hashing
    vals = (_A[:, None] * sh[None, :] + _B[:, None]) % np.uint64(_P)
    return vals.min(axis=1)


class NearDupIndex:
    """LSH over 64-permutation MinHash signatures; a hit is confirmed at est. Jaccard ≥ 0.7."""

    def __init__(self, threshold: float = 0.7):
        self.buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
        self.sigs: list[np.ndarray] = []
        self.threshold = threshold

    def seen_or_add(self, sig: np.ndarray) -> bool:
        keys = [(b, sig[b * _ROWS:(b + 1) * _ROWS].tobytes()) for b in range(_BANDS)]
        cands = {i for k in keys for i in self.buckets.get(k, ())}
        for i in cands:
            if float(np.mean(self.sigs[i] == sig)) >= self.threshold:
                return True
        idx = len(self.sigs)
        self.sigs.append(sig)
        for k in keys:
            self.buckets[k].append(idx)
        return False


# --------------------------------------------------------------------------- scan
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def scan(shards: list[int]) -> tuple[list[dict], list[float], Counter]:
    """Yield candidate conversations passing the cheap (non-token) filters."""
    cands: list[dict] = []
    gaps: list[float] = []
    reasons: Counter = Counter()
    cols = ["conversation_hash", "model", "conversation", "turn", "language", "toxic", "redacted"]
    for shard in shards:
        path = hf_file(REPO, f"data/train-{shard:05d}-of-00086.parquet")
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg, columns=cols).to_pylist():
                reasons["seen"] += 1
                lang = LANGS.get(row["language"] or "")
                if lang is None:
                    reasons["language"] += 1
                    continue
                if row["toxic"] or row["redacted"]:
                    reasons["toxic_or_redacted"] += 1
                    continue
                turns = row["turn"] or 0
                if not 2 <= turns <= 6:
                    reasons["turns"] += 1
                    continue
                conv = row["conversation"] or []
                if any(m.get("toxic") or m.get("redacted") for m in conv):
                    reasons["msg_toxic_or_redacted"] += 1
                    continue
                users = [m for m in conv if m.get("role") == "user"]
                if len(users) != turns or any(not (m.get("content") or "").strip() for m in users):
                    reasons["malformed"] += 1
                    continue
                # think-time sample: WildChat stores no user timestamps, but each assistant reply has
                # `created` (request start, unix s) and `timestamp` (reply finished); the user's think
                # time is created[k+1] - timestamp[k]
                asst = [m for m in conv if m.get("role") == "assistant"]
                for a, b in zip(asst, asst[1:]):
                    if a.get("timestamp") is not None and b.get("created"):
                        g = b["created"] - a["timestamp"].replace(tzinfo=_dt.timezone.utc).timestamp()
                        if 0 < g < 3600:
                            gaps.append(g)
                cands.append({
                    "conversation_hash": row["conversation_hash"],
                    "source_model": row["model"],
                    "lang": lang,
                    "user_turns": [m["content"] for m in users],
                    "shard": shard,
                })
        print(f"[w1] shard {shard}: {len(cands)} candidates so far", flush=True)
    return cands, gaps, reasons


def fit_lognormal(gaps: list[float], lo: float = 1.0, hi: float = 600.0) -> dict:
    g = np.array([x for x in gaps if lo <= x <= hi], dtype=float)
    if len(g) < 50:
        return {"n": int(len(g)), "note": "too few per-message timestamps to fit"}
    lg = np.log(g)
    return {"n": int(len(g)), "range_s": [lo, hi], "median_s": round(float(np.exp(np.median(lg))), 2),
            "sigma": round(float(np.std(lg)), 3),
            "p10_s": round(float(np.percentile(g, 10)), 1), "p90_s": round(float(np.percentile(g, 90)), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--german-share", type=float, default=0.2)
    ap.add_argument("--shards", type=str, default=",".join(map(str, WILDCHAT_SHARDS)))
    C.add_location_args(ap)
    args = ap.parse_args()
    if np is None or pq is None:
        sys.exit(f"missing Python package: {_MISSING} (pip install numpy pyarrow huggingface_hub transformers jinja2)")
    C.configure(args)
    OUT = C.DATA_DIR / "w1_chat-v1.jsonl"
    SYSTEM_PROMPT_FILE = C.DATA_DIR / "w1_system_prompt.txt"

    SYSTEM_PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SYSTEM_PROMPT_FILE.write_text(SYSTEM_PROMPT, encoding="utf-8")
    sys_tokens = text_tokens(SYSTEM_PROMPT)
    print(f"[w1] system prompt: {sys_tokens} tokens", flush=True)

    cands, gaps, reasons = scan([int(s) for s in args.shards.split(",")])

    # deterministic order, then dedupe + token filters, filling the (lang, category) buckets
    n_de = round(args.n * args.german_share)
    quota = {("de", "technical"): n_de // 2, ("de", "general"): n_de - n_de // 2,
             ("en", "technical"): (args.n - n_de) // 2, ("en", "general"): args.n - n_de - (args.n - n_de) // 2}
    cands.sort(key=lambda c: stable_hash("w1", c["conversation_hash"]))
    seen_hash: set[str] = set()
    seen_text: set[str] = set()
    lsh = NearDupIndex()
    picked: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cands:
        if all(len(picked[k]) >= q for k, q in quota.items()):
            break
        cat = "technical" if any(TECH_RE.search(u) for u in c["user_turns"]) else "general"
        key = (c["lang"], cat)
        if len(picked[key]) >= quota[key]:
            continue
        if c["conversation_hash"] in seen_hash:
            reasons["dup_hash"] += 1
            continue
        first = _norm(c["user_turns"][0])
        if first in seen_text:
            reasons["dup_first_msg"] += 1
            continue
        toks = [text_tokens(u) for u in c["user_turns"]]
        if toks[0] < 20:
            reasons["first_msg_short"] += 1
            continue
        if max(toks) > 4096:
            reasons["msg_too_long"] += 1
            continue
        if lsh.seen_or_add(minhash(c["user_turns"][0])):
            reasons["near_dup_first_msg"] += 1
            continue
        seen_hash.add(c["conversation_hash"])
        seen_text.add(first)
        c["category"] = cat
        c["user_tokens"] = toks
        picked[key].append(c)

    # German is scarce in multi-turn WildChat: top up any German shortfall with English
    # (same technical/general split) so the set stays at --n conversations.
    deficit = {cat: quota[("de", cat)] - len(picked[("de", cat)]) for cat in ("technical", "general")}
    if any(d > 0 for d in deficit.values()):
        for c in cands:
            if all(deficit[cat] <= 0 for cat in deficit):
                break
            if c["lang"] != "en" or c["conversation_hash"] in seen_hash:
                continue
            cat = "technical" if any(TECH_RE.search(u) for u in c["user_turns"]) else "general"
            if deficit[cat] <= 0:
                continue
            first = _norm(c["user_turns"][0])
            if first in seen_text:
                continue
            toks = [text_tokens(u) for u in c["user_turns"]]
            if toks[0] < 20 or max(toks) > 4096 or lsh.seen_or_add(minhash(c["user_turns"][0])):
                continue
            seen_hash.add(c["conversation_hash"])
            seen_text.add(first)
            c["category"], c["user_tokens"] = cat, toks
            picked[("en", cat)].append(c)
            deficit[cat] -= 1
    rows: list[dict] = [c for k in quota for c in picked[k]]
    rng("w1-order").shuffle(rows)
    out = []
    for c in rows:
        first_prompt = chat_tokens([{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": c["user_turns"][0]}])
        out.append({"id": f"wc-{c['conversation_hash'][:16]}", "conversation_hash": c["conversation_hash"],
                    "lang": c["lang"], "category": c["category"], "source_model": c["source_model"],
                    "n_turns": len(c["user_turns"]), "user_turns": c["user_turns"],
                    "user_tokens": c["user_tokens"], "first_prompt_tokens": first_prompt,
                    "source_shard": c["shard"]})
    n = write_jsonl(OUT, out)
    counts = Counter((r["lang"], r["category"]) for r in out)
    short = [k for k, q in quota.items() if counts[k] < q]
    german_share_actual = round(sum(1 for r in out if r["lang"] == "de") / max(1, len(out)), 3)
    all_user = [t for r in out for t in r["user_tokens"]]
    conv_user = [sum(r["user_tokens"]) for r in out]
    entry = {
        "workload": "w1", "source": REPO, "revision": REVISIONS[REPO],
        "source_files": [f"data/train-{s:05d}-of-00086.parquet" for s in map(int, args.shards.split(","))],
        "filters": {"toxic": False, "redacted": False, "languages": {"en": 1 - args.german_share,
                                                                     "de": args.german_share},
                    "user_turns": [2, 6], "first_msg_min_tokens": 20, "msg_max_tokens": 4096,
                    "dedupe": "conversation_hash + normalised first msg + MinHash(5-word shingles, 64 perm, "
                              "LSH 16x4, est. Jaccard>=0.7) of first user msg",
                    "mix": "50/50 technical/general per language (fenced code or code vocabulary)"},
        "counts": {"conversations": n, **{f"{l}_{c}": counts[(l, c)] for l, c in quota},
                   "turns": dict(sorted(Counter(r["n_turns"] for r in out).items())),
                   "quota_short": [f"{l}_{c}" for l, c in short],
                   "german_share_actual": german_share_actual},
        "rejections": dict(reasons),
        "system_prompt": {"file": SYSTEM_PROMPT_FILE.name, "tokens": sys_tokens,
                          "sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()},
        "token_stats": {"user_msg": token_stats(all_user), "first_prompt": token_stats([r["first_prompt_tokens"] for r in out]),
                        "user_tokens_per_conv": token_stats(conv_user)},
        "histograms": {"user_msg": histogram(all_user, [0, 32, 64, 128, 256, 512, 1024, 2048, 4097])},
        "think_time_fit": fit_lognormal(gaps),
        **file_entry(OUT),
    }
    update_manifest("w1_chat-v1", entry)
    print(f"[w1] wrote {n} conversations -> {OUT}", flush=True)
    print({k: v for k, v in entry.items() if k in ("counts", "token_stats", "think_time_fit", "rejections")})
    return 0 if n == args.n else 2


if __name__ == "__main__":
    sys.exit(main())
