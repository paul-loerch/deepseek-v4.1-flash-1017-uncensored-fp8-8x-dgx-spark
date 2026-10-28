#!/usr/bin/env python3
"""agent-conc.py -- multi-user coding-agent throughput: N users, each replaying its own recorded agent session turn by turn.

    agent-conc.py OUT.json [LEVELS] [SECS] [--base-url URL] [--data FILE] [--max-tokens 256] [--start-turn 4]

LEVELS: comma-separated user counts (default 1,2,4,8,16). SECS: measured window per level (default 150), after a 30 s ramp.
User u replays session u of w2_agent-v1.jsonl (64 recorded OpenHands sessions, 7k-49k tokens; built on your machine by
bench/data/prep_w2_agent.py, see bench/data/README.md; default location bench/data/prepared/, or $BENCH_DATA_DIR): request t = the recorded
conversation up to assistant turn t plus the session's tool definitions; then t+1, t+2, ... (wrapping to --start-turn), one
request at a time per user, no think time. Every user has its own cache_salt per level, so users never share a prefix, while
one user's consecutive turns reuse its own prefix (realistic agent prefix caching: each turn prefills the recorded assistant
message + observation that were added since the previous turn). Temperature 1.0 / top_p 0.95, thinking per the template
default, max_tokens 256 with ignore_eos (every request decodes the same number of tokens, so builds compare like for like).

Per level, from the SERVER counters over the window: generated tok/s (vllm:generation_tokens_total), prompt tok/s actually
computed (prompt tokens minus prefix-cache hits), preemptions; from the client (requests that started and finished inside
the level): TTFT p50/p90, per-request decode tok/s p50 (completion tokens / (last - first token time)), requests. Results:
one JSON object per level, printed as it finishes and written to OUT.json at the end.

The server root is --base-url, else the BASE_URL environment variable, else http://127.0.0.1:8000; the model name comes from
GET /v1/models. The server must be quiet while this runs. Python 3.9+, standard library only.
"""
import argparse, json, os, re, statistics as st, sys, threading, time, urllib.request, uuid

DATA_DIR = os.environ.get("BENCH_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out")
ap.add_argument("levels", nargs="?", default="1,2,4,8,16")
ap.add_argument("secs", nargs="?", type=int, default=150)
ap.add_argument("--base-url", default=os.environ.get("BASE_URL") or "http://127.0.0.1:8000")
ap.add_argument("--data", default=os.path.join(DATA_DIR, "w2_agent-v1.jsonl"))
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--start-turn", type=int, default=4)
ap.add_argument("--ramp", type=int, default=30, help="minimum ramp (s); the window also waits until every user finished its first (cold) turn, at most --max-ramp")
ap.add_argument("--max-ramp", type=int, default=600)
ap.add_argument("--inject-tokens", type=int, default=0,
                help="also send a cold synthetic prompt of about this many tokens (max_tokens 1, fresh cache_salt) every --inject-every s")
ap.add_argument("--inject-every", type=int, default=45)
args = ap.parse_args()
if not os.path.isfile(args.data):
    sys.exit(f"{args.data} not found: build it with bench/data/prep_w2_agent.py (see bench/data/README.md) or pass --data")
B = args.base_url.rstrip("/")
M = json.loads(urllib.request.urlopen(B + "/v1/models").read())["data"][0]["id"]
SESS = [json.loads(l) for l in open(args.data)]


def metrics():
    t = urllib.request.urlopen(B + "/metrics", timeout=30).read().decode()
    def g(n):
        return sum(float(m) for m in re.findall(rf"^{n}(?:{{[^}}]*}})? ([0-9.e+]+)$", t, re.M))
    return {"gen": g("vllm:generation_tokens_total"), "prompt": g("vllm:prompt_tokens_total"),
            "hits": g("vllm:prompt_tokens_cached_total"), "preempt": g("vllm:num_preemptions_total")}


def one(sess, t, salt, rec, stop):
    msgs = sess["messages"][: sess["assistant_idx"][t]]
    body = {"model": M, "messages": msgs, "tools": sess["tools"], "max_tokens": args.max_tokens, "ignore_eos": True,
            "temperature": 1.0, "top_p": 0.95, "seed": 1000 + t, "stream": True, "cache_salt": salt,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    t0 = time.time(); first = last = None; c_first = c_last = 0
    r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"}), timeout=1800)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        u = d.get("usage") or {}
        c = u.get("completion_tokens") or 0
        if c > 0:
            now = time.time()
            if first is None:
                first, c_first = now, c
            last, c_last = now, c
    if first is not None and not stop.is_set():
        rec.append({"t0": t0, "ttft": first - t0, "dec": (c_last - c_first) / (last - first) if last > first else None,
                    "tokens": c_last})


WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()


def injector(stop, out):
    """One cold long prompt every --inject-every s (never two at once); records its time to first (= only) token."""
    text = " ".join(WORDS[i % len(WORDS)] + str(i % 97) for i in range(int(args.inject_tokens / 2.35)))
    while not stop.is_set():
        t0 = time.time()
        body = {"model": M, "prompt": text, "max_tokens": 1, "temperature": 0, "cache_salt": uuid.uuid4().hex}
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(B + "/v1/completions", json.dumps(body).encode(),
                           {"Content-Type": "application/json"}), timeout=1800).read())
            out.append({"t0": t0, "ttft": time.time() - t0, "prompt_tokens": r["usage"]["prompt_tokens"]})
        except Exception as e:  # noqa: BLE001
            out.append({"t0": t0, "error": str(e)[:200]})
        stop.wait(max(0.0, args.inject_every - (time.time() - t0)))


def level(n):
    rec, stop, inj = [], threading.Event(), []
    warm = threading.Semaphore(0)
    def user(u):
        sess = SESS[u % len(SESS)]; salt = uuid.uuid4().hex
        turns = [t for t in range(len(sess["assistant_idx"])) if t >= min(args.start_turn, len(sess["assistant_idx"]) - 1)]
        k = 0
        while not stop.is_set():
            try:
                one(sess, turns[k % len(turns)], salt, rec, stop)
            except Exception as e:  # noqa: BLE001 -- count, keep the user going
                rec.append({"error": str(e)[:200]})
                time.sleep(1)
            if k == 0:
                warm.release()
            k += 1
    th = [threading.Thread(target=user, args=(u,), daemon=True) for u in range(n)]
    for t in th:
        t.start()
    if args.inject_tokens > 0:
        ti = threading.Thread(target=injector, args=(stop, inj), daemon=True); ti.start(); th.append(ti)
    tr = time.time()
    time.sleep(args.ramp)
    for _ in range(n):  # every user past its cold first turn (the prefix cache is warm from here on)
        warm.acquire(timeout=max(1.0, args.max_ramp - (time.time() - tr)))
    ramp_s = time.time() - tr
    m0, t0 = metrics(), time.time()
    time.sleep(args.secs)
    m1, t1 = metrics(), time.time()
    stop.set()
    for t in th:
        t.join(timeout=900)
    w = t1 - t0
    ok = [r for r in rec if "error" not in r and r["t0"] >= t0]
    ttft = sorted(r["ttft"] for r in ok); dec = [r["dec"] for r in ok if r["dec"]]
    return {"users": n, "ramp_s": round(ramp_s, 1), "gen_tok_s": round((m1["gen"] - m0["gen"]) / w, 1),
            "prefill_tok_s": round(((m1["prompt"] - m0["prompt"]) - (m1["hits"] - m0["hits"])) / w, 1),
            "prompt_tok_s_incl_cached": round((m1["prompt"] - m0["prompt"]) / w, 1),
            "ttft_p50": round(ttft[len(ttft) // 2], 2) if ttft else None,
            "ttft_p90": round(ttft[int(len(ttft) * 0.9)], 2) if ttft else None,
            "dec_tok_s_p50": round(st.median(dec), 1) if dec else None, "requests": len(ok),
            "errors": sum(1 for r in rec if "error" in r), "preemptions": m1["preempt"] - m0["preempt"],
            **({"inject_ttft": [round(r["ttft"], 2) for r in inj if "ttft" in r],
                "inject_prompt_tokens": max((r["prompt_tokens"] for r in inj if "ttft" in r), default=0),
                "inject_errors": sum(1 for r in inj if "error" in r)} if args.inject_tokens > 0 else {})}


out = []
for n in [int(x) for x in args.levels.split(",")]:
    r = level(n); out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(args.out, "w"), indent=1)
