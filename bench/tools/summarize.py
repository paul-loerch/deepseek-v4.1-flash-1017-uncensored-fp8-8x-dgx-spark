#!/usr/bin/env python3
"""summarize.py -- turn the JSON files of one benchmark run into summary.md.

    summarize.py RUN_DIR [--out FILE] [--stdout]

RUN_DIR is the output directory of run-bench.sh: run-info.json, steps.tsv and the JSON files of the individual tools
(drift.json, ctx-curve.json, ctx-conc.json, decode-agent.json, decode-chat.json, decode-conc.json, vllm-bench-c*.json,
canary/report.json). Files that are missing are skipped, so it also works on a partial run. The summary is written to
RUN_DIR/summary.md unless --out says otherwise; --stdout prints it instead.

Every table is followed by a "Disclosure" line with the eight things a published number needs: date, engine version,
prompt class, n, temperature, thinking state, GPU clock cap and the definition of concurrency. Date, engine and clock cap
come from run-info.json (the clock cap is whatever you passed to run-bench.sh --clock-cap; "NOT RECORDED" otherwise); the
rest comes from the tool's own output and fixed settings. A cell or step that failed, was aborted or was skipped is shown
as such, never left out. Python 3.9+, standard library only.
"""
import argparse
import glob
import json
import os
import re
import sys


UNREADABLE = []    # files that exist but are not valid JSON: reported at the end, never dropped silently


def load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        parent = os.path.basename(os.path.dirname(path))
        UNREADABLE.append(f"{parent + '/' if parent == 'canary' else ''}{os.path.basename(path)}: {type(exc).__name__}")
        return None


def num(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return str(x)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def pick(d, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


class Summary:
    def __init__(self, run_dir):
        self.dir = run_dir
        self.info = load(os.path.join(run_dir, "run-info.json")) or {}
        self.lines = []

    # ---- pieces ----------------------------------------------------------------------------------------------
    def p(self, *ls):
        self.lines.extend(ls)

    def disclosure(self, prompt_class, n, temperature, thinking, concurrency):
        i = self.info
        self.p("", "*Disclosure.* "
               f"Date: {i.get('date_utc', 'NOT RECORDED')}. Engine: {i.get('engine', 'NOT RECORDED')}. "
               f"Prompt class: {prompt_class}. n: {n}. Temperature: {temperature}. Thinking: {thinking}. "
               f"GPU clock cap: {i.get('clock_cap') or 'NOT RECORDED'}. Concurrency: {concurrency}.", "")

    def header(self):
        i = self.info
        quick = i.get("profile") == "quick"
        self.p("# Benchmark summary", "")
        if quick:
            self.p("> **QUICK RUN.** A smoke test with fewer and smaller cells. It checks that the tools and the server work; "
                   "its numbers are not comparable with a full run and must not be published.", "")
        rows = [("Profile", i.get("profile", "unknown")),
                ("Date (UTC)", i.get("date_utc", "NOT RECORDED")),
                ("Engine", i.get("engine", "NOT RECORDED")),
                ("Model", i.get("model", "NOT RECORDED")),
                ("max_model_len", num(i.get("max_model_len"))),
                ("GPU clock cap", i.get("clock_cap") or "NOT RECORDED (pass --clock-cap to run-bench.sh)"),
                ("Node guard (ctx-curve)", "on" if i.get("guard") else "off"),
                ("Fabric counters (decode arm)", "on" if i.get("fabric") else "off")]
        self.p(*table(["", ""], rows), "")
        skipped = i.get("cells_skipped") or []
        if skipped:
            self.p("Cells skipped because the served max_model_len is too small: " + ", ".join(skipped) + ".", "")

    def steps(self):
        path = os.path.join(self.dir, "steps.tsv")
        rows = []
        try:
            for ln in open(path, encoding="utf-8"):
                parts = ln.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    rows.append((parts[0], parts[1], parts[3] + " s"))
        except OSError:
            return
        if rows:
            self.p("## Steps", "", *table(["step", "result", "time"], rows), "")

    def drift(self):
        d = load(os.path.join(self.dir, "drift.json"))
        if not d:
            return
        s = d.get("summary", {})
        res = d.get("results", [])
        cc, cw = s.get("cold_vs_cold", {}), s.get("cold_vs_warm", {})
        self.p("## Cold vs warm prefix-cache drift (drift-probe.py)", "")
        rows = [("cold A vs cold B (noise floor)", num(cc.get("lead_identical_median")), num(cc.get("lead_identical_min")), num(cc.get("first_pos_kl_max"), 6)),
                ("cold A vs warm", num(cw.get("A", {}).get("lead_identical_median")), num(cw.get("A", {}).get("lead_identical_min")), num(cw.get("A", {}).get("first_pos_kl_max"), 6)),
                ("cold B vs warm", num(cw.get("B", {}).get("lead_identical_median")), num(cw.get("B", {}).get("lead_identical_min")), num(cw.get("B", {}).get("first_pos_kl_max"), 6))]
        self.p(*table(["pair", "identical leading tokens, median", "minimum", "max KL at position 1"], rows), "")
        hits = s.get("warm_cache_hits")
        n = s.get("n")
        self.p(f"{n} comparisons; warm requests that reported cached tokens: {hits} of {n}." +
               ("  **No warm request used the cache, so this test says nothing.**" if n and not hits else ""))
        self.p("", "How to read it: warm should look like the noise floor (identical, or a late and weak divergence). An early, strong "
               "divergence on the warm side only means the prefix cache restored the wrong state.")
        self.disclosure("document text (prepared W3 corpus), prefix of 9001 / 17333 / 23011 tokens plus a 211-token tail",
                        f"{n} comparisons ({len(res)} document/split pairs, 4 requests each)", "0 (greedy), 48 tokens, top-5 logprobs",
                        "not applicable (completions on token ids, no chat template)", "1 (requests are sequential)")

    def ctx(self, fname, title):
        d = load(os.path.join(self.dir, fname))
        if not d:
            return
        cells = d.get("cells", [])
        args = d.get("args", {})
        reps = args.get("reps", 1)
        mt = args.get("max_tokens", 512)
        self.p(f"## {title}", "")
        one = [c for c in cells if c.get("U") == 1]
        many = [c for c in cells if (c.get("U") or 0) > 1]

        def note(c):
            if c.get("aborted"):
                return "ABORTED: " + str(c["aborted"])[:80]
            errs = [r.get("err") for r in c.get("reqs", []) if r.get("err")]
            if errs:
                return "ERROR: " + str(errs[0])[:80]
            wn = (c.get("window") or {}).get("note")
            return f"ok ({wn}: no window)" if wn else "ok"

        guard_on = bool(d.get("guard"))

        def thermal(c):
            """hottest zone, hottest GPU, lowest MemAvailable over the nodes during the cell (guard only)"""
            mem = c.get("mem_avail_min_gib") or {}
            return [num(c.get("zmax_c")), num(c.get("gpu_max_c")), num(min(mem.values())) if mem else "-"]
        extra_h = ["hottest zone C", "hottest GPU C", "min MemAvailable GiB"] if guard_on else []

        if one:
            rows = []
            for c in one:
                r = (c.get("reqs") or [{}])[0]
                md = c.get("metrics_delta", {})
                rows.append((num(c["N"]), c["rep"], num(r.get("prompt_tokens")), num(r.get("ttft_s"), 2), num(r.get("prefill_tok_s")),
                             num(r.get("decode_tok_s")), num(c.get("step_ms"), 2), num(c.get("mean_accept_len"), 2),
                             num(100 * c["kv_usage_max"]) if "kv_usage_max" in c else "-", num(md.get("prefix_cache_hits_total")),
                             num(md.get("num_preemptions_total")), *(thermal(c) if guard_on else []), note(c)))
            self.p(*table(["N", "rep", "prompt tokens", "TTFT s", "prefill tok/s", "decode tok/s", "step ms", "mean accept len",
                           "KV max %", "cache hits", "preempt"] + extra_h + ["result"], rows), "")
        if many:
            rows = []
            for c in many:
                rs = c.get("reqs", [])
                w = c.get("window", {})
                md = c.get("metrics_delta", {})
                ttfts = [r["ttft_s"] for r in rs if r.get("ttft_s")]
                per = w.get("per_user_tok_s")
                rows.append((num(c["N"]), c["U"], c["rep"], "/".join(num(r.get("prompt_tokens")) for r in rs[:1]),
                             f"{num(min(ttfts), 1)}-{num(max(ttfts), 1)}" if ttfts else "-",
                             f"{num(min(per))}-{num(max(per))}" if per else "-", num(w.get("agg_tok_s")), num(w.get("s")),
                             num(100 * c["kv_usage_max"]) if "kv_usage_max" in c else "-", num(md.get("num_preemptions_total")),
                             *(thermal(c) if guard_on else []), note(c)))
            self.p(*table(["N", "users", "rep", "prompt tokens per user", "TTFT s (min-max)", "decode tok/s per user (min-max)",
                           "aggregate tok/s", "all-decoding window s", "KV max %", "preempt"] + extra_h + ["result"], rows), "")
        if not cells:
            self.p("No cell finished.", "")
        guard = "on" if d.get("guard") else "off"
        if one:
            self.p("TTFT is the time to the first streamed token (it includes queueing); prefill tok/s = prompt tokens / TTFT; decode tok/s "
                   "counts tokens from the first to the last streamed chunk using the server's exact usage counts; step ms and mean accept "
                   f"len come from the speculative-decoding counters. Node guard: {guard}.")
        if many:
            self.p("Per-user decode tok/s and the aggregate are measured only inside the window in which all users are decoding "
                   "(none is recorded when the users overlap for less than 5 s); TTFT includes queueing behind the other users' prefills. "
                   f"Node guard: {guard}.")
        self.disclosure("document text (prepared W3 corpus, continued by W2 agent traces beyond about 0.9 M tokens), chat API",
                        f"{reps} repetition(s); one request per user per cell, so n = users per cell",
                        "0.6 (top_p 0.95, per-user seed)", "off (enable_thinking false)",
                        f"U users submit together, one request each; every prefill cold (fresh cache_salt); ignore_eos, max_tokens {mt}")

    def decode(self):
        chat = load(os.path.join(self.dir, "decode-chat.json"))
        agent = load(os.path.join(self.dir, "decode-agent.json"))
        if not chat and not agent:
            return
        self.p("## Decode speed, one user (decode-bench.py, decode-bench-w2.py)", "")
        rows = []
        if chat:
            rows.append(("chat, 6 short prompts", num(len(chat.get("rows", []))), num(chat.get("decode_tok_s_median"), 2),
                         num(chat.get("decode_tok_s_mean"), 2), num(chat.get("AL"), 3), num(chat.get("accept_rate"), 3)))
        if agent:
            rows.append(("agent turns, 7k-49k context", num(agent.get("n")), num(agent.get("median"), 2), num(agent.get("mean"), 2),
                         num(agent.get("AL"), 3), "-"))
        self.p(*table(["workload", "n", "median tok/s", "mean tok/s", "mean accept len (AL)", "accept rate"], rows), "")
        self.p("Chat: tok/s = 399 / (last chunk - first chunk) per request. Agent: tok/s = 255 / (last delta - first delta) per request.")
        self.disclosure("chat: 6 fixed short prompts; agent: 16 recorded coding-agent turns with tool definitions",
                        f"chat {len((chat or {}).get('rows', []))}, agent {(agent or {}).get('n', '-')} requests",
                        "1.0 (top_p 0.95)", "chat: off; agent: chat-template default (the request does not set it; on for this model)",
                        "1 (one request at a time); ignore_eos, 400 / 256 tokens")

    def sweep(self):
        d = load(os.path.join(self.dir, "decode-conc.json"))
        if not d:
            return
        self.p("## Chat concurrency sweep (conc-sweep.py)", "")
        rows = []
        for r in d:
            u = r.get("users")
            per_tok = (r["agg_tok_s"] / u) if u and r.get("agg_tok_s") is not None else None
            rows.append((u, num(r.get("agg_tok_s")), num(per_tok), num(r.get("per_user_tps_p50")), num(r.get("ttft_p50"), 2),
                         num(r.get("ttft_p90"), 2), num(r.get("running_mean")), num(r.get("preemptions")), num(r.get("kv_usage_end"), 3),
                         num(r.get("n_req"))))
        self.p(*table(["users", "aggregate tok/s (server counter)", "aggregate / users", "per-request chunks/s p50", "TTFT p50 s",
                       "TTFT p90 s", "running (mean)", "preemptions", "KV usage at end", "requests"], rows), "")
        self.p("Aggregate tok/s is the whole system; aggregate / users is the token-based figure per user. 'chunks/s' counts streamed chunks, "
               "not tokens: with speculative decoding it reads lower than tokens per second by about the acceptance length.")
        self.disclosure("8 fixed short chat prompts, answers end naturally (max_tokens 300)", "requests per level in the last column",
                        "1.0 (top_p 0.95)", "off (enable_thinking false)",
                        "N client threads, each with exactly one request in flight at all times (closed loop, no think time)")

    def serve(self):
        files = sorted(glob.glob(os.path.join(self.dir, "vllm-bench-c*.json")),
                       key=lambda p: int(re.findall(r"c(\d+)\.json$", p)[0]) if re.findall(r"c(\d+)\.json$", p) else 0)
        runs = [(f, load(f)) for f in files]
        runs = [(f, d) for f, d in runs if isinstance(d, dict)]
        if not runs:
            return
        self.p("## vllm bench serve (synthetic reference)", "")

        def conc_of(f, d):
            c = d.get("max_concurrency")
            if c is None:
                m = re.search(r"c(\d+)\.json$", os.path.basename(f))
                c = m.group(1) if m else "?"
            return c
        heads = ["metric"] + [f"c={conc_of(f, d)}" for f, d in runs]
        rows = []

        def row(label, *keys, nd=1):
            rows.append([label] + [num(pick(d, *keys), nd) for _, d in runs])
        row("prompts (num_prompts)", "num_prompts")
        row("completed", "completed")
        row("duration s", "duration", nd=1)
        row("request throughput req/s", "request_throughput", nd=3)
        row("output throughput tok/s", "output_throughput", nd=1)
        row("total token throughput tok/s", "total_token_throughput", nd=1)
        row("mean accept len (server counters)", "spec_decode_acceptance_length", nd=3)
        for m, label in (("ttft", "TTFT"), ("tpot", "TPOT"), ("itl", "ITL"), ("e2el", "E2EL")):
            row(f"{label} mean ms", f"mean_{m}_ms")
            row(f"{label} p50 ms", f"p50_{m}_ms", f"median_{m}_ms")
            row(f"{label} p90 ms", f"p90_{m}_ms")
            row(f"{label} p99 ms", f"p99_{m}_ms")
        self.p(*table(heads, rows), "")
        self.p("Random token ids are not natural text: speculative-decoding acceptance is low, so decode is slower than on real prompts. "
               "The four runs use the same seed, so later runs can reuse prefixes cached by earlier ones (check the server's prefix-cache hit rate). "
               "The tail of a short run is part of the duration; use the concurrency sweep for steady-state throughput.")
        self.disclosure("random token ids (vllm bench serve --dataset-name random), 8192 in / 1024 out",
                        "num_prompts = max(8, 4 x concurrency), seed 1234",
                        "not set by the client: the server default (the model's generation config) applies",
                        "not applicable (completions endpoint, no chat template)",
                        "--max-concurrency c, unlimited request rate; ignore_eos")

    def canary(self):
        r = load(os.path.join(self.dir, "canary", "report.json"))
        if not r:
            return
        st = r.get("stats", {})
        evs = r.get("events", [])
        crit = sum(1 for e in evs if e.get("severity") == "critical")
        high = sum(1 for e in evs if e.get("severity") == "high")
        kinds = st.get("event_kinds", {})
        self.p("## Cross-request canary (canary-leak-test.py)", "")
        rows = [("verdict", st.get("verdict", "-") + (" -- " + "; ".join(st.get("reasons", [])) if st.get("reasons") else "")),
                ("duration", f"{num(r.get('seconds'), 0)} s"),
                ("requests (warm incl. forks / cold replays)", f"{num(st.get('requests'))} ({num(st.get('warm'))} / {num(st.get('cold'))})"),
                ("sessions finished", num(len(r.get("sessions", [])))),
                ("leak events (critical)", num(crit)),
                ("critical + high events", num(crit + high)),
                ("95 % upper bound, critical events per request", f"{100 * st['upper95_critical']:.2f} %" if st.get("upper95_critical") is not None else "-"),
                ("event kinds", ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items())) or "none")]
        self.p(*table(["", ""], rows), "")
        self.p("A zero count only certifies a rate of about 3 / requests (95 % upper bound); a short run proves little.")
        a = r.get("args", {})
        self.disclosure("synthetic long-lived agent-shaped sessions with unique canary strings, "
                        f"{a.get('ctx_min', '-')}-{a.get('ctx_max', '-')} token contexts",
                        f"{num(st.get('requests'))} requests", f"0 and 1 (fraction at 1: {a.get('t1_frac', '-')})",
                        "off when run-bench.sh starts it (enable_thinking false)",
                        f"{a.get('sessions', '-')} concurrent sessions, at most {a.get('max_inflight', '-')} requests in flight")

    def build(self):
        self.header()
        self.steps()
        self.drift()
        self.ctx("ctx-curve.json", "Cold prefill, TTFT and decode against context length, one user (ctx-curve.py)")
        self.ctx("ctx-conc.json", "Several users at 32k context (ctx-curve.py)")
        self.decode()
        self.sweep()
        self.serve()
        self.canary()
        if UNREADABLE:
            self.p("## Files that could not be read", "", *[f"* {u}" for u in UNREADABLE], "")
        return "\n".join(self.lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="output directory of run-bench.sh")
    ap.add_argument("--out", default=None, help="summary file (default: RUN_DIR/summary.md)")
    ap.add_argument("--stdout", action="store_true", help="print the summary instead of writing a file")
    a = ap.parse_args()
    if not os.path.isdir(a.run_dir):
        sys.exit(f"{a.run_dir}: not a directory")
    text = Summary(a.run_dir).build()
    if a.stdout:
        sys.stdout.write(text)
        return 0
    out = a.out or os.path.join(a.run_dir, "summary.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"summary written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
