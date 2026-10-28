#!/usr/bin/env python3
"""decode-bench.py -- single-user decode speed on short chat prompts.

    decode-bench.py OUT.json [REPS] [--base-url URL]

Sends 6 fixed chat prompts (code, networking, history, ...), REPS times each (default 2), one request at a time,
streamed. Every request uses temperature 1.0 and top_p 0.95 (the model's default sampling), a fixed seed (1000 * rep +
prompt index), thinking off, max_tokens 400 and ignore_eos, so every request decodes exactly 400 tokens whatever the
model would have liked to say.

Per request: decode tok/s = 399 / (time of the last streamed chunk - time of the first one), i.e. the first token is
excluded. With speculative decoding one chunk can carry several tokens, which is why the formula uses the known token
count (400) and not the number of chunks. Speculative-decoding acceptance comes from the deltas of the server's
/metrics counters over the whole run: AL (mean acceptance length, tokens per draft step) = 1 + accepted / drafts, and
accept_rate = accepted / proposed draft tokens.

OUT.json holds one row per request plus decode_tok_s_median / _mean / _all, AL and accept_rate; the same summary
(without the rows) is printed. The server must be quiet while this runs: other traffic changes the numbers.

The server root is --base-url, else the BASE_URL environment variable, else http://127.0.0.1:8000. The model name is
read from GET /v1/models. Python 3.9+, standard library only.
"""
import argparse, json, os, re, statistics, sys, time, urllib.request
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out", help="output JSON file")
ap.add_argument("reps", nargs="?", type=int, default=2, help="repetitions of the 6 prompts (default 2)")
ap.add_argument("--base-url", default=os.environ.get("BASE_URL") or "http://127.0.0.1:8000",
                help="server root, without /v1 (default: $BASE_URL, else http://127.0.0.1:8000)")
args = ap.parse_args()
B = args.base_url.rstrip("/")
M = json.loads(urllib.request.urlopen(B + "/v1/models").read())["data"][0]["id"]
PROMPTS = [
 "Write a Python function that parses an nginx access log line and returns a dict. Explain edge cases.",
 "Explain how TCP congestion control (CUBIC vs BBR) works, with a worked example.",
 "Refactor this JavaScript into modern async/await style and explain each change:\nfunction load(cb){fs.readFile('a.json',function(e,d){if(e)return cb(e);cb(null,JSON.parse(d))})}",
 "Write a bash script that rotates logs in /var/log/myapp older than 7 days, compresses them, and keeps 30 archives.",
 "Summarise the causes and consequences of the 1929 stock market crash in a structured essay.",
 "Implement a thread-safe LRU cache in Go with generics and write table-driven tests for it.",
]
def metrics():
    t = urllib.request.urlopen(B + "/metrics", timeout=30).read().decode()
    g = lambda n: sum(float(m) for m in re.findall(rf"^{n}(?:{{[^}}]*}})? ([0-9.e+]+)$", t, re.M))
    return g("vllm:spec_decode_num_accepted_tokens_total"), g("vllm:spec_decode_num_drafts_total"), g("vllm:spec_decode_num_draft_tokens_total")
reps = args.reps
a0 = metrics(); rows = []
for rep in range(reps):
    for i, p in enumerate(PROMPTS):
        body = {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 1.0,
                "top_p": 0.95, "seed": 1000 * rep + i, "stream": True, "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(),
                                   {"Content-Type": "application/json"}), timeout=600)
        ts = []
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("choices") and (d["choices"][0].get("delta", {}).get("content") or d["choices"][0].get("delta", {}).get("reasoning_content")):
                ts.append(time.time())
        rows.append({"p": i, "rep": rep, "chunks": len(ts), "decode_s": round(ts[-1] - ts[0], 3)})
a1 = metrics()
acc, drafts, dtok = (a1[0] - a0[0]), (a1[1] - a0[1]), (a1[2] - a0[2])
# tokens per request is 400 (ignore_eos); chunks < tokens under spec decode (several tokens per chunk)
tps = [399 / r["decode_s"] for r in rows]
res = {"rows": rows, "decode_tok_s_median": round(statistics.median(tps), 2), "decode_tok_s_mean": round(statistics.mean(tps), 2),
       "decode_tok_s_all": [round(x, 2) for x in tps], "AL": round(1 + acc / drafts, 3) if drafts else None,
       "accept_rate": round(acc / dtok, 3) if dtok else None}
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
