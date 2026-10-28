#!/usr/bin/env python3
"""decode-bench-w2.py -- single-user decode speed on recorded coding-agent contexts.

    decode-bench-w2.py OUT.json [REPS] [--base-url URL] [--data FILE]

Takes the first 8 sessions of the prepared W2 agent data (w2_agent-v1.jsonl, built by data/prep_w2_agent.py from
recorded OpenHands trajectories). From each session it replays two turns: the first and the last turn whose recorded
assistant message has at least 100 tokens, i.e. one early and one late turn. A request carries the recorded conversation
up to that turn (7k-49k prompt tokens) and the tool definitions. One request at a time, streamed, temperature 1.0 /
top_p 0.95 (the model's default sampling), seed 7000 + 100 * rep + turn number, max_tokens 256 and ignore_eos, so every
request decodes the same 256 tokens. The request does not set enable_thinking, so the chat template's default applies
(thinking is on for this model). Each repetition uses its own cache_salt: the first request of every turn is a cold
prefill and repetitions do not reuse each other's prefix cache.

Decode tok/s per request comes from the server's own token counts: the request asks for usage on every streamed chunk
(stream_options.continuous_usage_stats), and the rate is (completion tokens at the last chunk - at the first chunk that
carries a token) / (time between those chunks). Timing visible deltas instead is fooled when the tool parser holds
output back. The mean acceptance length AL of speculative decoding (1 + accepted / drafts) comes from the
deltas of the server's /metrics counters, per request and over the whole run. The prompts are identical on every
run, so two configurations can be compared turn by turn.

OUT.json: one row per request (turn id, context tokens, decode tok/s, AL) plus median, mean, AL and n.

The server root is --base-url, else the BASE_URL environment variable, else http://127.0.0.1:8000; the model name
comes from GET /v1/models. --data defaults to w2_agent-v1.jsonl in $BENCH_DATA_DIR, else in bench/data/prepared.
Python 3.9+, standard library only.
"""
import argparse, json, os, re, statistics, sys, time, urllib.request
DATA_DIR = os.environ.get("BENCH_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out", help="output JSON file")
ap.add_argument("reps", nargs="?", type=int, default=2, help="repetitions of the 16 turns (default 2)")
ap.add_argument("--base-url", default=os.environ.get("BASE_URL") or "http://127.0.0.1:8000",
                help="server root, without /v1 (default: $BASE_URL, else http://127.0.0.1:8000)")
ap.add_argument("--data", default=os.path.join(DATA_DIR, "w2_agent-v1.jsonl"),
                help="prepared W2 agent sessions (default: %(default)s)")
args = ap.parse_args()
B = args.base_url.rstrip("/")
if not os.path.isfile(args.data):
    sys.exit(f"{args.data} not found: build it with bench/data/prep_w2_agent.py (see bench/data/README.md) or pass --data")
M = json.loads(urllib.request.urlopen(B + "/v1/models").read())["data"][0]["id"]
rows = [json.loads(l) for l in open(args.data)]
TURNS = []
for r in rows[:8]:                       # first 8 sessions; 2 turns each: one early, one late in the session
    idx = [i for i, a in enumerate(r["asst_tokens"]) if a >= 100]
    for i in (idx[:1] + idx[-1:]) if idx else []:
        TURNS.append((r["id"], i, r["messages"][: r["assistant_idx"][i]], r["tools"], r["prefix_tokens"][i]))
def metrics():
    t = urllib.request.urlopen(B + "/metrics", timeout=30).read().decode()
    g = lambda n: sum(float(m) for m in re.findall(rf"^{n}(?:{{[^}}]*}})? ([0-9.e+]+)$", t, re.M))
    return g("vllm:spec_decode_num_accepted_tokens_total"), g("vllm:spec_decode_num_drafts_total")
reps = args.reps
out = []; acc_all = drafts_all = 0.0
for rep in range(reps):
    for k, (sid, ti, msgs, tools, ptok) in enumerate(TURNS):
        body = {"model": M, "messages": msgs, "tools": tools, "max_tokens": 256, "temperature": 1.0, "top_p": 0.95,
                "seed": 7000 + 100 * rep + k, "stream": True, "ignore_eos": True, "cache_salt": f"w2dec-{rep}",
                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        a0 = metrics()
        r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(),
                                   {"Content-Type": "application/json"}), timeout=900)
        pts = []                         # (arrival time, completion tokens so far) of every chunk that reports usage
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            ct = (d.get("usage") or {}).get("completion_tokens")
            if ct is not None:
                pts.append((time.time(), ct))
        a1 = metrics(); acc, dr = a1[0] - a0[0], a1[1] - a0[1]; acc_all += acc; drafts_all += dr
        first = next((p for p in pts if p[1] >= 1), None)
        last = pts[-1] if pts else None
        tps = ((last[1] - first[1]) / (last[0] - first[0])
               if first and last and last[0] > first[0] and last[1] > first[1] else None)
        out.append({"rep": rep, "turn": f"{sid[-8:]}#{ti}", "ctx": ptok, "decode_tok_s": round(tps, 2) if tps else None,
                    "AL": round(1 + acc / dr, 3) if dr else None})
        print(json.dumps(out[-1]), flush=True)
v = [o["decode_tok_s"] for o in out if o["decode_tok_s"]]
res = {"rows": out, "median": round(statistics.median(v), 2), "mean": round(statistics.mean(v), 2),
       "AL": round(1 + acc_all / drafts_all, 3) if drafts_all else None, "n": len(v)}
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("median", "mean", "AL", "n")}))
