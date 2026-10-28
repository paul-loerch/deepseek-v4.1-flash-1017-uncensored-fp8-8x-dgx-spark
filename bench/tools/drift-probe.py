#!/usr/bin/env python3
"""drift-probe.py -- cold-vs-warm prefix-cache drift test.

    drift-probe.py OUT.json [--base-url URL] [--data FILE] [--docs N] [--splits P,P,...] [--q N] [--gen N]

Question: does a request whose prompt prefix is served from the prefix cache (for a model with sliding-window, compressed or
state-space caches: all of that state restored at a cached block boundary) compute the same thing as the same request
computed cold? A state restored at the wrong position (cf. the public report "KDA state resumed 1,152 tokens early on
prefix-cache hits", for another model) makes warm outputs diverge early and strongly; plain floating-point
nondeterminism makes two COLD runs diverge only later and weakly.

Per document and split point p (deliberately NOT aligned to cache block boundaries):
  cold A : prompt T[:p+q]  with a fresh random cache_salt          (nothing cached)
  cold B : prompt T[:p+q]  with another fresh salt                  (the noise floor)
  prime  : prompt T[:p]    with salt S, max_tokens 1                (puts the prefix into the cache)
  warm   : prompt T[:p+q]  with salt S                              (prefix hit; usage reports cached_tokens)
T is the tokenized document text. The documents are the N longest items of the prepared W3 document-QA data
(w3_docqa-v1.jsonl, built by data/prep_w3_docqa.py). Each scored request generates 48 tokens greedily (temperature 0)
with top-5 logprobs, via /v1/completions on token ids.

Scores per pair (A-B, A-warm, B-warm): the number of leading identical tokens, the mean |delta logprob| of the chosen
token over that identical leading part, and the KL divergence over the first position's top-5 logprobs. Reading it:
A-warm and B-warm should look like A-B (identical, or a late weak divergence). An early, strong divergence only on the
warm side means the cache restored wrong state. warm_cache_hits must be > 0, otherwise the warm request never used the
cache and the test says nothing.

OUT.json: {"summary": ..., "results": [one object per document and split]}; the summary is also printed on a line that
starts with DRIFT-SUMMARY. The server root is --base-url (alias --base), else the BASE_URL environment variable, else
http://127.0.0.1:8000; the model name comes from GET /v1/models. --data defaults to w3_docqa-v1.jsonl in
$BENCH_DATA_DIR, else in bench/data/prepared. Python 3.9+, standard library only.
"""
import argparse, json, math, os, random, sys, time, urllib.request

DATA_DIR = os.environ.get("BENCH_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out", help="output JSON file")
ap.add_argument("--base-url", "--base", dest="base", default=os.environ.get("BASE_URL") or "http://127.0.0.1:8000",
                help="server root, without /v1 (default: $BASE_URL, else http://127.0.0.1:8000)")
ap.add_argument("--data", default=os.path.join(DATA_DIR, "w3_docqa-v1.jsonl"),
                help="prepared W3 document-QA data (default: %(default)s)")
ap.add_argument("--docs", type=int, default=4, help="number of documents, longest first (default 4)")
ap.add_argument("--splits", default="9001,17333,23011", help="prefix lengths in tokens, not block-aligned (default 9001,17333,23011)")
ap.add_argument("--q", type=int, default=211, help="uncached tail after the prefix, in tokens (default 211)")
ap.add_argument("--gen", type=int, default=48, help="tokens generated per scored request (default 48)")
a = ap.parse_args()
a.base = a.base.rstrip("/")
if not os.path.isfile(a.data):
    sys.exit(f"{a.data} not found: build it with bench/data/prep_w3_docqa.py (see bench/data/README.md) or pass --data")

def post(path, body, timeout=1800):
    r = urllib.request.Request(a.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

model = json.loads(urllib.request.urlopen(a.base + "/v1/models").read())["data"][0]["id"]
rows = [json.loads(l) for l in open(a.data)]
rows = sorted(rows, key=lambda r: -r["prompt_tokens"])[: a.docs]

def complete(ids, salt, gen):
    body = {"model": model, "prompt": ids, "max_tokens": gen, "temperature": 0, "logprobs": 5, "cache_salt": salt}
    t0 = time.time(); d = post("/v1/completions", body); dt = time.time() - t0
    ch = d["choices"][0]; lp = ch.get("logprobs") or {}
    toks = lp.get("tokens") or []; tlp = lp.get("token_logprobs") or []; top = lp.get("top_logprobs") or []
    cached = ((d.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"tokens": toks, "lp": tlp, "top": top, "cached": cached, "s": round(dt, 2)}

def kl_top5(p, q):
    if not p or not q: return None
    keys = set(p) | set(q); floor = min(list(p.values()) + list(q.values())) - 2.0
    P = {k: math.exp(p.get(k, floor)) for k in keys}; Q = {k: math.exp(q.get(k, floor)) for k in keys}
    zp, zq = sum(P.values()), sum(Q.values())
    return sum((P[k] / zp) * math.log((P[k] / zp) / (Q[k] / zq)) for k in keys)

def compare(x, y):
    n = min(len(x["tokens"]), len(y["tokens"])); lead = 0
    while lead < n and x["tokens"][lead] == y["tokens"][lead]: lead += 1
    m = lead if lead else 1
    dlp = [abs(x["lp"][i] - y["lp"][i]) for i in range(m) if x["lp"][i] is not None and y["lp"][i] is not None]
    return {"lead_identical": lead, "of": n, "mean_abs_dlp_common": round(sum(dlp) / len(dlp), 5) if dlp else None,
            "first_pos_kl_top5": round(kl_top5(x["top"][0], y["top"][0]), 6) if x["top"] and y["top"] else None}

results = []
for r in rows:
    text = r["prefix"] + r["doc"] + r.get("suffix", "")
    ids = post("/tokenize", {"model": model, "prompt": text})["tokens"]
    for p in [int(s) for s in a.splits.split(",")]:
        if p + a.q > len(ids): continue
        full = ids[: p + a.q]
        rnd = lambda: f"drift-{random.getrandbits(64):016x}"
        cold_a = complete(full, rnd(), a.gen)
        cold_b = complete(full, rnd(), a.gen)
        s = rnd(); prime = complete(ids[:p], s, 1)
        warm = complete(full, s, a.gen)
        res = {"doc": r["id"], "prefix": p, "len": len(full), "warm_cached_tokens": warm["cached"],
               "A_vs_B": compare(cold_a, cold_b), "A_vs_warm": compare(cold_a, warm), "B_vs_warm": compare(cold_b, warm),
               "secs": [cold_a["s"], cold_b["s"], prime["s"], warm["s"]]}
        results.append(res)
        print(json.dumps(res), flush=True)

def agg(key):
    v = [x[key]["lead_identical"] for x in results]; k = [x[key]["first_pos_kl_top5"] for x in results if x[key]["first_pos_kl_top5"] is not None]
    return {"lead_identical_median": sorted(v)[len(v) // 2] if v else None, "lead_identical_min": min(v) if v else None,
            "first_pos_kl_max": max(k) if k else None}
summary = {"n": len(results), "cold_vs_cold": agg("A_vs_B"), "cold_vs_warm": {"A": agg("A_vs_warm"), "B": agg("B_vs_warm")},
           "warm_cache_hits": sum(1 for x in results if (x["warm_cached_tokens"] or 0) > 0)}
json.dump({"summary": summary, "results": results}, open(a.out, "w"), indent=1)
print("DRIFT-SUMMARY", json.dumps(summary))
