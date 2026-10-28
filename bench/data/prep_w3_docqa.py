#!/usr/bin/env python3
"""W3 — long-document and code-repo QA: one cold question per unique document, then warm
follow-ups on the same document.

Sources:
  * zai-org/LongBench-v2: 12 English items of length class "short" whose templated prompt is
    12–56K tokens, 3 each from single-doc QA, multi-doc QA, code-repo understanding and
    structured data (length-stratified within each domain)
  * princeton-nlp/SWE-bench_Lite_bm25_27K (test): 12 items from distinct repositories
    (issue + BM25-retrieved files, ≈27K tokens)
  * 2 fixed follow-up questions per item
  * plus a "long" pool of 4 English LongBench "medium" documents that the driver only uses cut to
    the 56K / 60K single-stream targets (pool = "long"; never sent uncut)

Each record keeps the user message split into prefix / doc / suffix so that a driver can shorten
the document to a target prompt size (16K, 32K, 48K, 56K, 60K, ...) at a precomputed character
offset ("cuts"), without a tokenizer on the hot path. prompt_tokens are measured with the fixed
chat system prompt (_common.SYSTEM_PROMPT), as a chat client would send it. The benchmark tools use
the documents as plain text: ctx-curve.py slices them into long prompts and drift-probe.py takes the
longest ones.

    python3 prep_w3_docqa.py [--data-dir DIR] [--cache-dir DIR] [--tokenizer DIR]

Needs the w3 files from fetch_sources.py (the default); it does not need the WildChat data.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import argparse

_MISSING = None
try:
    import pyarrow.parquet as pq
except ImportError as _exc:  # reported by main(), after --help has been handled
    pq = None
    _MISSING = _exc.name

import _common as C
from _common import (REVISIONS, chat_tokens, file_entry, hf_file, histogram, stable_hash,
                     token_stats, tokenizer, update_manifest, write_jsonl)

LB = "zai-org/LongBench-v2"
SWE = "princeton-nlp/SWE-bench_Lite_bm25_27K"
MIN_TOK, MAX_TOK = 12288, 57344
CUT_TARGETS = [8192, 12288, 16384, 24576, 32768, 40960, 49152, 57344, 61440]
LB_DOMAINS = ["Single-Document QA", "Multi-Document QA", "Code Repository Understanding",
              "Long Structured Data Understanding"]
LB_STRATA = [(12288, 24576), (24576, 40960), (40960, MAX_TOK)]

# LongBench-v2's own prompt template (0-shot, no CoT instruction; the model reasons anyway)
LB_PREFIX = "Please read the following text and answer the question below.\n\n<text>\n"
LB_SUFFIX = ("\n</text>\n\nWhat is the correct answer to this question: {question}\nChoices:\n(A) {A}\n(B) {B}\n"
             "(C) {C}\n(D) {D}\n\nFormat your response as follows: \"The correct answer is (insert answer here)\".")
LB_FOLLOWUPS = ["Quote, word for word, the passage from the text that best supports your answer.",
                "For each of the other three choices, explain briefly why it is wrong, citing the text."]
SWE_FOLLOWUPS = ["Which files and functions does your patch change, and why is each change needed?",
                 "Write a pytest regression test that fails before your fix and passes after it."]


def english(text: str) -> bool:
    sample = text[:20000] + text[len(text) // 2: len(text) // 2 + 20000]
    return sum(c.isascii() for c in sample) / max(1, len(sample)) > 0.97


def cuts(system: str, prefix: str, doc: str, suffix: str, full_tokens: int) -> tuple[dict[str, int], int]:
    """Char offsets into doc so that prefix+doc[:cut]+suffix renders to ≈ each target size."""
    enc = tokenizer()(doc, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    overhead = full_tokens - len(offsets)
    out: dict[str, int] = {}
    for t in CUT_TARGETS:
        budget = t - overhead
        if 0 < budget < len(offsets):
            out[str(t)] = offsets[budget][0]
    return out, overhead


def record(system: str, rid: str, source: str, domain: str, prefix: str, doc: str, suffix: str,
           answer: str | None, followups: list[str], extra: dict) -> dict:
    full = chat_tokens([{"role": "system", "content": system}, {"role": "user", "content": prefix + doc + suffix}])
    c, overhead = cuts(system, prefix, doc, suffix, full)
    return {"id": rid, "source": source, "domain": domain, "prompt_tokens": full,
            "overhead_tokens": overhead, "doc_chars": len(doc), "cuts": c, "prefix": prefix, "doc": doc,
            "suffix": suffix, "answer": answer, "followups": followups, **extra}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_location_args(ap)
    args = ap.parse_args()
    if pq is None:
        sys.exit(f"missing Python package: {_MISSING} (pip install pyarrow huggingface_hub transformers jinja2)")
    C.configure(args)
    OUT = C.DATA_DIR / "w3_docqa-v1.jsonl"
    SYSTEM_PROMPT_FILE = C.DATA_DIR / "w1_system_prompt.txt"
    system = C.ensure_system_prompt_file()  # the same fixed text prep_w1_chat.py writes
    reasons: Counter = Counter()

    # ---- LongBench-v2
    lb = json.loads(hf_file(LB, "data.json").read_text())
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for d in lb:
        if d["length"] != "short" or d["domain"] not in LB_DOMAINS:
            continue
        if not english(d["context"]):
            reasons["lb_not_english"] += 1
            continue
        by_domain[d["domain"]].append(d)
    per_dom: dict[str, dict[int, list[dict]]] = {}
    for dom in LB_DOMAINS:
        per_stratum: dict[int, list[dict]] = defaultdict(list)
        for d in sorted(by_domain[dom], key=lambda d: stable_hash("w3-lb", d["_id"])):
            suffix = LB_SUFFIX.format(question=d["question"].strip(), A=d["choice_A"], B=d["choice_B"],
                                      C=d["choice_C"], D=d["choice_D"])
            r = record(system, f"lb-{d['_id']}", "longbench-v2", dom, LB_PREFIX, d["context"], suffix,
                       d["answer"], LB_FOLLOWUPS, {"difficulty": d["difficulty"], "sub_domain": d["sub_domain"]})
            s = next((i for i, (lo, hi) in enumerate(LB_STRATA) if lo <= r["prompt_tokens"] < hi), None)
            if s is None:
                reasons["lb_out_of_range"] += 1
                continue
            per_stratum[s].append(r)
        per_dom[dom] = per_stratum
    # 3 per domain: one per length stratum first, then any stratum (hash order); a domain with
    # too few in-range English items is topped up from the other domains in the same way
    lb_rows: list[dict] = []
    used: set[str] = set()

    def take(dom: str, k: int) -> list[dict]:
        ps = per_dom[dom]
        order = [ps[s][0] for s in range(len(LB_STRATA)) if ps[s]] + \
                [r for s in range(len(LB_STRATA)) for r in ps[s][1:]]
        got = [r for r in order if r["id"] not in used][:k]
        used.update(r["id"] for r in got)
        return got

    for dom in LB_DOMAINS:
        got = take(dom, 3)
        lb_rows += got
        print(f"[w3] {dom}: {[r['prompt_tokens'] for r in got]}", flush=True)
    for dom in LB_DOMAINS * 3:
        if len(lb_rows) >= 12:
            break
        lb_rows += take(dom, 1)
    lb_rows = lb_rows[:12]

    # ---- SWE-bench Lite BM25 27K
    sw = pq.read_table(hf_file(SWE, "data/test-00000-of-00001.parquet")).to_pylist()
    sw.sort(key=lambda d: stable_hash("w3-swe", d["instance_id"]))
    seen_repo: set[str] = set()
    swe_rows: list[dict] = []
    for d in sw:
        if len(swe_rows) >= 12:
            break
        if d["repo"] in seen_repo:
            continue
        text = d["text"]
        a, b = text.find("<code>\n"), text.rfind("\n</code>")
        if a < 0 or b < 0:
            reasons["swe_no_code_block"] += 1
            continue
        prefix, doc, suffix = text[: a + len("<code>\n")], text[a + len("<code>\n"): b], text[b:]
        r = record(system, f"swe-{d['instance_id']}", "swe-bench-lite-bm25-27k", "Repo issue fix", prefix, doc,
                   suffix, None, SWE_FOLLOWUPS, {"repo": d["repo"]})
        if not MIN_TOK <= r["prompt_tokens"] <= MAX_TOK:
            reasons["swe_out_of_range"] += 1
            continue
        seen_repo.add(d["repo"])
        swe_rows.append(r)

    # ---- long pool: 4 English LongBench "medium" docs (64K-160K tokens) that the driver only ever
    # uses *cut* to the 56K/60K single-stream targets (the 12-56K "short" set cannot reach them).
    # Answer accuracy is not scored on cut documents.
    long_rows: list[dict] = []
    for d in sorted((d for d in lb if d["length"] == "medium" and d["domain"] in LB_DOMAINS[:2]),
                    key=lambda d: stable_hash("w3-lb-long", d["_id"])):
        if len(long_rows) >= 4:
            break
        if not english(d["context"]) or not 200_000 < len(d["context"]) < 700_000:
            continue
        suffix = LB_SUFFIX.format(question=d["question"].strip(), A=d["choice_A"], B=d["choice_B"],
                                  C=d["choice_C"], D=d["choice_D"])
        r = record(system, f"lb-{d['_id']}", "longbench-v2", d["domain"], LB_PREFIX, d["context"], suffix,
                   d["answer"], LB_FOLLOWUPS, {"difficulty": d["difficulty"], "sub_domain": d["sub_domain"],
                                               "pool": "long"})
        if 65536 <= r["prompt_tokens"] <= 163840 and "61440" in r["cuts"]:
            long_rows.append(r)
    for r in lb_rows + swe_rows:
        r["pool"] = "main"

    rows = lb_rows + swe_rows + long_rows
    n = write_jsonl(OUT, rows)
    toks = [r["prompt_tokens"] for r in lb_rows + swe_rows]
    entry = {
        "workload": "w3",
        "sources": {LB: REVISIONS[LB], SWE: REVISIONS[SWE]},
        "source_files": {LB: "data.json", SWE: "data/test-00000-of-00001.parquet"},
        "filters": {"longbench": {"length": "short", "english": "ascii ratio > 0.97", "domains": LB_DOMAINS,
                                  "per_domain": 3, "prompt_tokens": [MIN_TOK, MAX_TOK],
                                  "strata": LB_STRATA},
                    "swebench": {"split": "test", "distinct_repos": True, "n": 12}},
        "system_prompt": SYSTEM_PROMPT_FILE.name,
        "followups": {"longbench": LB_FOLLOWUPS, "swebench": SWE_FOLLOWUPS},
        "cut_targets": CUT_TARGETS,
        "counts": {"items": n, "longbench": len(lb_rows), "swebench": len(swe_rows), "long_pool": len(long_rows),
                   "domains": dict(Counter(r["domain"] for r in rows))},
        "rejections": dict(reasons),
        "token_stats": {"prompt": token_stats(toks), "longbench": token_stats([r["prompt_tokens"] for r in lb_rows]),
                        "swebench": token_stats([r["prompt_tokens"] for r in swe_rows]),
                        "long_pool_uncut": token_stats([r["prompt_tokens"] for r in long_rows])},
        "histograms": {"prompt": histogram(toks, [12288, 16384, 24576, 32768, 40960, 49152, 57345])},
        **file_entry(OUT),
    }
    update_manifest("w3_docqa-v1", entry)
    print(f"[w3] wrote {n} items -> {OUT}")
    print(json.dumps({k: entry[k] for k in ("counts", "token_stats", "histograms", "rejections")}, indent=1))
    return 0 if len(lb_rows) == 12 and len(swe_rows) == 12 and len(long_rows) == 4 else 2


if __name__ == "__main__":
    sys.exit(main())
