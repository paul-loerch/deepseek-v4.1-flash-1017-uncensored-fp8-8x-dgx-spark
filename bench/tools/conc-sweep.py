#!/usr/bin/env python3
"""conc-sweep.py -- closed-loop concurrency sweep for chat decode throughput.

    conc-sweep.py OUT.json [LEVELS] [SECS] [--base-url URL]

LEVELS is a comma-separated list of user counts (default 1,2,4,8,12,16); SECS is the length of the measured window per
level (default 90). The release sequence uses 1,2,3,4,6,8 and 120.

For every level, N client threads each loop over a fixed set of 8 short chat prompts, one request at a time per thread,
with no think time: "N users" means N requests in flight at all times. Thinking off, temperature 1.0 / top_p 0.95,
max_tokens 300, fixed seeds, no ignore_eos (answers end when the model ends them, at most 300 tokens). The first 15 s
of a level are a ramp and are not measured; the window that follows lasts SECS seconds.

Per level: aggregate generated tok/s from the SERVER counter vllm:generation_tokens_total over the window (this avoids
the client-side first-chunk artefact), the median per-request decode rate, TTFT p50 and p90, the mean number of
requests running (vllm:num_requests_running, sampled every 5 s), preemptions during the window and KV-cache usage at the
end. agg_tok_s is the whole system's throughput; per_user_tps_p50 is what one user sees. Do not quote one for the other.

Reading per_user_tps_p50: it is (streamed chunks - 1) / (last chunk time - first chunk time) per request, i.e. chunks
per second, not tokens per second. With speculative decoding one chunk usually carries more than one token, so it reads
low by about the mean acceptance length. agg_tok_s / users is the token-based per-user figure at steady state. The
per-request figures (per_user_tps_p50, ttft_p50, ttft_p90, n_req) cover every request that finished between the start of
the level (ramp included) and the end of the window; requests still running at the end are dropped.

OUT.json is a list with one object per level (written when all levels are done; the same objects are printed as they
finish). The server must be quiet while this runs.

The server root is --base-url, else the BASE_URL environment variable, else http://127.0.0.1:8000; the model name
comes from GET /v1/models. Python 3.9+, standard library only.
"""
import argparse, json, os, re, statistics as st, sys, threading, time, urllib.request
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out", help="output JSON file")
ap.add_argument("levels", nargs="?", default="1,2,4,8,12,16", help="comma-separated user counts (default 1,2,4,8,12,16)")
ap.add_argument("secs", nargs="?", type=int, default=90, help="measured window per level in seconds (default 90)")
ap.add_argument("--base-url", default=os.environ.get("BASE_URL") or "http://127.0.0.1:8000",
                help="server root, without /v1 (default: $BASE_URL, else http://127.0.0.1:8000)")
args = ap.parse_args()
B = args.base_url.rstrip("/")
M = json.loads(urllib.request.urlopen(B + "/v1/models").read())["data"][0]["id"]
PROMPTS = [
 "Write a Python function that parses an nginx access log line and returns a dict. Explain edge cases.",
 "Explain how TCP congestion control (CUBIC vs BBR) works, with a worked example.",
 "Write a bash script that rotates logs older than 7 days, compresses them, and keeps 30 archives.",
 "Summarise the causes and consequences of the 1929 stock market crash in a structured essay.",
 "Implement a thread-safe LRU cache in Go with generics and write table-driven tests for it.",
 "Explain the difference between processes and threads, with examples from Linux.",
 "Write a SQL schema for a library system and three example queries with explanations.",
 "Describe how a hash map handles collisions, compare chaining and open addressing.",
]
def metrics():
    t = urllib.request.urlopen(B + "/metrics", timeout=30).read().decode()
    def g(n):
        return sum(float(m) for m in re.findall(rf"^{n}(?:{{[^}}]*}})? ([0-9.e+]+)$", t, re.M))
    return {"gen": g("vllm:generation_tokens_total"), "running": g("vllm:num_requests_running"),
            "preempt": g("vllm:num_preemptions_total"), "kv": g("vllm:kv_cache_usage_perc")}
def one(i, seed, rec, stop):
    body = {"model": M, "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], "max_tokens": 300,
            "temperature": 1.0, "top_p": 0.95, "seed": seed, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); ts = []
    r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"}), timeout=900)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]": continue
        d = json.loads(line[5:])
        if d.get("choices") and (d["choices"][0].get("delta", {}).get("content") or d["choices"][0].get("delta", {}).get("reasoning_content")):
            ts.append(time.time())
    if len(ts) > 1 and not stop.is_set():
        rec.append({"ttft": ts[0] - t0, "tps": (len(ts) - 1) / (ts[-1] - ts[0])})
def level(n, secs):
    rec, stop = [], threading.Event()
    def worker(k):
        j = 0
        while not stop.is_set():
            one(k * 101 + j, 10_000 * n + 100 * k + j, rec, stop); j += 1
    th = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(n)]
    for t in th: t.start()
    time.sleep(15)
    m0, t0, run = metrics(), time.time(), []
    while time.time() - t0 < secs:
        time.sleep(5); run.append(metrics()["running"])
    m1, t1 = metrics(), time.time()
    stop.set()
    for t in th: t.join(timeout=600)
    agg = (m1["gen"] - m0["gen"]) / (t1 - t0)
    tps = [r["tps"] for r in rec]; ttft = sorted(r["ttft"] for r in rec)
    return {"users": n, "agg_tok_s": round(agg, 1), "per_user_tps_p50": round(st.median(tps), 1) if tps else None,
            "ttft_p50": round(ttft[len(ttft) // 2], 2) if ttft else None, "ttft_p90": round(ttft[int(len(ttft) * 0.9)], 2) if ttft else None,
            "running_mean": round(st.mean(run), 1), "preemptions": m1["preempt"] - m0["preempt"], "kv_usage_end": round(m1["kv"], 3), "n_req": len(rec)}
levels = [int(x) for x in args.levels.split(",")]
secs = args.secs
out = []
for n in levels:
    r = level(n, secs); out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(args.out, "w"), indent=1)
