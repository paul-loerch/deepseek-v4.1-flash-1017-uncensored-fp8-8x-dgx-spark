#!/usr/bin/env python3
"""decode-bench-w2-tc.py -- decode-bench-w2.py with an explicit tool_choice.

    decode-bench-w2-tc.py OUT.json TOOL_CHOICE [IGNORE_EOS] [REPS] [SALT] [--base-url URL] [--data FILE]

TOOL_CHOICE is auto or none. IGNORE_EOS is 1 (default) or 0; REPS defaults to 1 and SALT to w2tc.

Same recorded agent turns, sampling and seeds as decode-bench-w2.py; only tool_choice (and optionally ignore_eos)
changes. The tools are sent in both cases and the chat template renders them with tool_choice none too, so the prompt
and the prefix-cache key stay identical across the arms.

Purpose: isolate the cost of vLLM's default tool-call enforcement from everything else. In v0.32 the default
--tool-strict-level attaches a FUNCTION structural tag (a grammar for the tool-call format) to every request that has
tools and tool_choice auto; tool_choice none does not. Run the tool once with auto and once with none and compare the
decode speeds.

Decode tok/s per request = (completion_tokens - 1) / (time of the last delta - time of the first delta), counting
content, reasoning and tool-call deltas; completion_tokens comes from the streamed usage. The mean acceptance length AL
of speculative decoding (1 + accepted / drafts) comes from the deltas of the server's /metrics counters. Each row also
records TTFT, the finish reason, the number of tool-call deltas and which kinds of delta arrived.

OUT.json: the settings, one row per request, and median, mean, AL and n.

The server root is --base-url, else the BASE_URL environment variable, else http://127.0.0.1:8000; the model name
comes from GET /v1/models. --data defaults to w2_agent-v1.jsonl in $BENCH_DATA_DIR, else in bench/data/prepared.
Python 3.9+, standard library only.
"""
import argparse, json, os, re, statistics, sys, time, urllib.request
DATA_DIR = os.environ.get("BENCH_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out", help="output JSON file")
ap.add_argument("tool_choice", help="tool_choice sent with the request: auto or none")
ap.add_argument("ignore_eos", nargs="?", default="1", help="1 = send ignore_eos (default), 0 = let the model stop")
ap.add_argument("reps", nargs="?", type=int, default=1, help="repetitions of the 16 turns (default 1)")
ap.add_argument("salt", nargs="?", default="w2tc", help="cache_salt prefix (default w2tc)")
ap.add_argument("--base-url", default=os.environ.get("BASE_URL") or "http://127.0.0.1:8000",
                help="server root, without /v1 (default: $BASE_URL, else http://127.0.0.1:8000)")
ap.add_argument("--data", default=os.path.join(DATA_DIR, "w2_agent-v1.jsonl"),
                help="prepared W2 agent sessions (default: %(default)s)")
args = ap.parse_args()
B = args.base_url.rstrip("/")
if not os.path.isfile(args.data):
    sys.exit(f"{args.data} not found: build it with bench/data/prep_w2_agent.py (see bench/data/README.md) or pass --data")
OUT, TC = args.out, args.tool_choice
IGN = args.ignore_eos == "1"
REPS = args.reps
SALT = args.salt
M = json.loads(urllib.request.urlopen(B + "/v1/models").read())["data"][0]["id"]
rows = [json.loads(l) for l in open(args.data)]
TURNS = []
for r in rows[:8]:
    idx = [i for i, a in enumerate(r["asst_tokens"]) if a >= 100]
    for i in (idx[:1] + idx[-1:]) if idx else []:
        TURNS.append((r["id"], i, r["messages"][: r["assistant_idx"][i]], r["tools"], r["prefix_tokens"][i]))
def metrics():
    t = urllib.request.urlopen(B + "/metrics", timeout=30).read().decode()
    g = lambda n: sum(float(m) for m in re.findall(rf"^{n}(?:{{[^}}]*}})? ([0-9.e+]+)$", t, re.M))
    return g("vllm:spec_decode_num_accepted_tokens_total"), g("vllm:spec_decode_num_drafts_total")
out = []; acc_all = drafts_all = 0.0
for rep in range(REPS):
    for k, (sid, ti, msgs, tools, ptok) in enumerate(TURNS):
        body = {"model": M, "messages": msgs, "tools": tools, "tool_choice": TC, "max_tokens": 256, "temperature": 1.0,
                "top_p": 0.95, "seed": 7000 + 100 * rep + k, "stream": True, "stream_options": {"include_usage": True},
                "cache_salt": f"{SALT}-{rep}"}
        if IGN:
            body["ignore_eos"] = True
        a0 = metrics(); t0 = time.time()
        r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(),
                                   {"Content-Type": "application/json"}), timeout=900)
        ts = []; ntc = 0; fin = None; ctoks = None; kinds = set()
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): ctoks = d["usage"].get("completion_tokens")
            if d.get("choices"):
                c = d["choices"][0]; dl = c.get("delta", {}); fin = c.get("finish_reason") or fin
                for kk in ("content", "reasoning_content", "reasoning", "tool_calls"):
                    if dl.get(kk): kinds.add(kk)
                if dl.get("tool_calls"): ntc += 1
                if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning") or dl.get("tool_calls"):
                    ts.append(time.time())
        a1 = metrics(); acc, dr = a1[0] - a0[0], a1[1] - a0[1]; acc_all += acc; drafts_all += dr
        n = ctoks or 256
        tps = (n - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else None
        out.append({"rep": rep, "turn": f"{sid[-8:]}#{ti}", "ctx": ptok, "decode_tok_s": round(tps, 2) if tps else None,
                    "AL": round(1 + acc / dr, 3) if dr else None, "ttft_s": round(ts[0] - t0, 2) if ts else None,
                    "completion_tokens": ctoks, "finish": fin, "tool_call_deltas": ntc, "kinds": sorted(kinds)})
        print(json.dumps(out[-1]), flush=True)
v = [o["decode_tok_s"] for o in out if o["decode_tok_s"]]
res = {"tool_choice": TC, "ignore_eos": IGN, "rows": out, "median": round(statistics.median(v), 2) if v else None,
       "mean": round(statistics.mean(v), 2) if v else None,
       "AL": round(1 + acc_all / drafts_all, 3) if drafts_all else None, "n": len(v)}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("tool_choice", "ignore_eos", "median", "mean", "AL", "n")}))
