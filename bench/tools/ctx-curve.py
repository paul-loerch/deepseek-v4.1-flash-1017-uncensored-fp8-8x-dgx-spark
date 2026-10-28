#!/usr/bin/env python3
"""ctx-curve.py -- cold prefill, TTFT and decode speed against context length and number of users.

  ctx-curve.py OUT.json --cells 8192x1,32768x1,32768x4 [--reps 2] [--max-tokens 512] [--base-url URL] [--data-dir DIR]
  ctx-curve.py --build-only --cells ...    # tokenize/detokenize the prompts and print their sizes; no chat request
  ctx-curve.py --guard --dry-run           # take two readings from every node and print them; no request to the server

A cell NxU is U simultaneous users, each with its own distinct document slice (the prepared W3 document-QA text,
continued by the W2 agent traces beyond about 0.9 M tokens; N prompt tokens in total), chat API, thinking off,
temperature 0.6 / top_p 0.95, a per-user seed, a fresh cache_salt per request (so every prefill is cold) and ignore_eos
(every request decodes exactly --max-tokens). Repetitions run as rounds: all cells, then all cells again.

Per request: TTFT, prompt tokens, and decode tok/s from the first to the last streamed token. The server sends usage
statistics with every chunk, so tokens are counted exactly even when a chunk carries several.
Per cell: U = 1 -> prefill tok/s (prompt tokens / TTFT) and, from the server's speculative-decoding counters, the mean
acceptance length (tokens per step) and the decode step time (decode window / drafts): the context dependence of a step
without the acceptance noise. U > 1 -> the all-decoding window [latest first token, earliest last token] with per-user
and aggregate tok/s inside it. Always: preemptions, prefix-cache hits (must be about 0: the prompts are cold), the
maximum KV-cache usage, running and waiting requests.

Stall protection (always on). No first token within 120 + 2 x (queued prompt tokens / 1500 tok/s) seconds, or no
streamed token for --stall-post seconds after the first one: the cell is aborted (the sockets are shut, so vLLM aborts
the requests), the run stops and the exit status is 3 ("possible wedge": check the engine before anything else). A cell
that runs longer than --max-cell-s is aborted too (exit status 2).

Optional node guard (off by default; --guard). Every 5 s each node of your cluster is read over one persistent ssh
connection: the hottest ACPI thermal zone, the GPU temperature and power draw, and MemAvailable. A node at zone
>= --zstop or GPU >= --gstop, or (when --memstop > 0) with MemAvailable below --memstop GiB, on two readings in a row
trips the watchdog: the cell is aborted and the run stops (exit status 2). Before each cell every node must be freshly
read at <= --cool (zone) and <= --gcool (GPU) degrees C, for at most --cool-max seconds, otherwise the run stops.
The node list comes from --nodes, else $NODES, else cluster.env (NODES, or NODE1..NODE8); ssh options come from
SSH_OPTS and SSH_USER there ($CLUSTER_ENV or --cluster-env points at the file, default: cluster.env in the repository
root). A node whose name is this machine's hostname is read directly instead of over ssh. Needs passwordless ssh to
every node and nvidia-smi on the nodes. Without --guard none of this runs, and the run says so at the start.
The output records which node had which peak by position (node1 = first entry), never by host name.

OUT.json: {"version", "model", "guard", "args", "cells": [...]}, rewritten after every cell. One line per cell is printed.
The server root is --base-url, else the BASE_URL environment variable, else http://127.0.0.1:8000; the model name comes
from GET /v1/models. --data-dir defaults to $BENCH_DATA_DIR, else bench/data/prepared. Python 3.9+, standard library only.
Exit status: 0 done, 2 stopped by the guard or by --max-cell-s, 3 stall or setup error.
"""
import argparse, http.client, json, os, re, shlex, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.parse, uuid
from concurrent.futures import ThreadPoolExecutor

VERSION = "ctx-curve/1.2"
SCHEME, HOST, PORT, PREFIX = "http", "127.0.0.1", 8000, ""      # set from --base-url by set_endpoint()
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                    # <repo>/bench/tools/ctx-curve.py -> <repo>
DEFAULT_DATA_DIR = os.environ.get("BENCH_DATA_DIR") or os.path.join(os.path.dirname(HERE), "data", "prepared")
DEFAULT_SSH_OPTS = "-o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=3"
HEAD, INSTR = "<document>\n", "\n</document>\n\nSummarize the document above in detail, section by section."
DOC_OVERHEAD = 40          # chat template + instruction tokens, subtracted from N for the document slice
KEYS = ["vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total", "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
        "vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
        "vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting"]


def set_endpoint(base_url):
    global SCHEME, HOST, PORT, PREFIX
    u = urllib.parse.urlsplit(base_url if "://" in base_url else "http://" + base_url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SystemExit(f"bad --base-url {base_url!r}: expected http://HOST:PORT or https://HOST:PORT")
    SCHEME, HOST = u.scheme, u.hostname
    PORT = u.port or (443 if u.scheme == "https" else 80)
    PREFIX = u.path.rstrip("/")


def connect(timeout=None):
    cls = http.client.HTTPSConnection if SCHEME == "https" else http.client.HTTPConnection
    return cls(HOST, PORT, timeout=timeout)


def call(path, body=None, timeout=300):
    c = connect(timeout)
    try:
        if body is None:
            c.request("GET", PREFIX + path)
        else:
            c.request("POST", PREFIX + path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
    finally:
        c.close()
    if r.status != 200:
        raise RuntimeError(f"{path} HTTP {r.status}: {data[:300]!r}")
    return data


def metrics():
    out = {}
    for ln in call("/metrics", timeout=10).decode().splitlines():
        if ln.startswith("#"):
            continue
        name = ln.split("{", 1)[0].split(" ", 1)[0]
        if name in KEYS:
            try:
                out[name] = out.get(name, 0.0) + float(ln.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


# ------------------------------------------------------------------------------ optional node guard
def read_env_file(path):
    """KEY=value lines of a cluster.env (plain assignments only: no variable expansion, no commands)."""
    vals = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.startswith("export "):
                ln = ln[7:].lstrip()
            k, sep, v = ln.partition("=")
            if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k.strip()):
                continue
            v = v.strip()
            if v[:1] in ("'", '"'):
                end = v.find(v[0], 1)
                v = v[1:end] if end > 0 else v[1:]
            else:
                v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
            vals[k.strip()] = v
    return vals


def is_local_host(host):
    """True when `host` names this machine; such a node is read directly instead of over ssh."""
    names = {"localhost", "127.0.0.1", "::1"}
    try:
        full = socket.gethostname()
        names.update({full, full.split(".")[0]})
    except OSError:
        pass
    return host in names


def guard_config(nodes_arg, cluster_env):
    """-> (hosts, ssh option list, ssh user) for --guard, or SystemExit with a message that says what to fix."""
    env = {}
    if cluster_env and os.path.isfile(cluster_env):
        env = read_env_file(cluster_env)
    txt = nodes_arg or os.environ.get("NODES") or env.get("NODES")
    if txt:
        hosts = txt.replace(",", " ").split()
        where = "--nodes / $NODES / NODES in cluster.env"
    else:
        hosts = [env.get(f"NODE{i}", "").strip() for i in range(1, 9)]
        hosts = [h for h in hosts if h]
        where = f"NODE1..NODE8 in {cluster_env}"
    if not hosts:
        raise SystemExit("guard: no node list. Pass --nodes \"host1 host2 ...\", set NODES, or copy cluster.env.example to "
                         f"cluster.env and fill in NODE1..NODE8 (looked for {cluster_env}). Run without --guard to skip the guard.")
    bad = [h for h in hosts if "CHANGE_ME" in h or "<" in h]
    if bad:
        raise SystemExit(f"guard: the node list ({where}) still has placeholder values: {' '.join(bad)}")
    ssh_opts = shlex.split(os.environ.get("SSH_OPTS") or env.get("SSH_OPTS") or DEFAULT_SSH_OPTS)
    ssh_user = (os.environ.get("SSH_USER") or env.get("SSH_USER") or "").strip()
    return hosts, ssh_opts, ssh_user


class Fleet:
    CMD = ("cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1; "
           "nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits; "
           "awk '/MemAvailable/{print $2}' /proc/meminfo")   # MemAvailable (kB) as a third line

    def __init__(self, hosts, ssh_opts, ssh_user, zstop, gstop, interval=5.0, memstop=0.0):
        self.hosts, self.ssh_opts, self.ssh_user = list(hosts), list(ssh_opts), ssh_user
        self.names = [f"node{i + 1}" for i in range(len(self.hosts))]   # the labels used in every output
        self.local = [is_local_host(h) for h in self.hosts]
        self.zstop, self.gstop, self.interval, self.memstop = zstop, gstop, interval, memstop
        self.memmin = {}
        self.cm = tempfile.mkdtemp(prefix="ctxcm-", dir="/tmp" if os.path.isdir("/tmp") else None)
        self.lock, self.stop_ev = threading.Lock(), threading.Event()
        self.last, self.peak, self.breach, self.fails, self.trip = {}, {}, {}, {}, None
        for h, loc in zip(self.hosts, self.local):   # one persistent master per remote node; samples reuse it (ControlMaster=no never hangs a pipe)
            if loc:
                continue
            subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", *self.ssh_opts, "-MNf",
                            "-o", f"ControlPath={self.cm}/%C", "-o", "ControlPersist=900", self.target(h)],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=20, check=False)
        self.pool = ThreadPoolExecutor(len(self.hosts))
        self.th = threading.Thread(target=self.loop, daemon=True)
        self.th.start()

    def target(self, host):
        return f"{self.ssh_user}@{host}" if self.ssh_user else host

    def one(self, i):
        node = self.names[i]
        if self.local[i]:
            argv = ["sh", "-c", self.CMD]
        else:
            argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "ControlMaster=no",
                    "-o", f"ControlPath={self.cm}/%C", *self.ssh_opts, self.target(self.hosts[i]),
                    "timeout -k 1 4 sh -c " + shlex.quote(self.CMD)]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=6, stdin=subprocess.DEVNULL)
            ln = p.stdout.strip().split("\n")
            z = int(ln[0]) / 1000.0
            g, w = [float(x) for x in ln[1].split(",")]
            mem = int(ln[2]) / 1048576.0 if len(ln) > 2 and ln[2].strip().isdigit() else None
            return node, z, g, w, mem
        except Exception:
            return node, None, None, None, None

    def loop(self):
        while not self.stop_ev.is_set():
            t0 = time.time()
            for node, z, g, w, mem in self.pool.map(self.one, range(len(self.hosts))):
                with self.lock:
                    if z is None:
                        self.fails[node] = self.fails.get(node, 0) + 1
                        continue
                    self.last[node] = (t0, z, g, w)
                    pk = self.peak.setdefault(node, [0.0, 0.0, 0.0])
                    pk[0], pk[1], pk[2] = max(pk[0], z), max(pk[1], g), max(pk[2], w)
                    if mem is not None:
                        self.memmin[node] = min(self.memmin.get(node, 1e9), mem)
                    lowmem = bool(self.memstop) and mem is not None and mem < self.memstop
                    bad = z >= self.zstop or g >= self.gstop or lowmem
                    self.breach[node] = self.breach.get(node, 0) + 1 if bad else 0
                    if self.breach[node] >= 2 and not self.trip:
                        self.trip = (f"{node}: zone {z:.1f} C / GPU {g:.0f} C / MemAvailable "
                                     f"{(mem if mem is not None else float('nan')):.1f} GiB on 2 samples in a row "
                                     f"(stop {self.zstop} / {self.gstop} / mem {self.memstop})")
            self.stop_ev.wait(max(0.5, self.interval - (time.time() - t0)))

    def first_readings(self, timeout=25.0):
        """Wait for one reading of every node; return the labels of the nodes that never answered."""
        t0 = time.time()
        while True:
            with self.lock:
                missing = [n for n in self.names if n not in self.last]
            if not missing or time.time() - t0 > timeout:
                return missing
            time.sleep(0.5)

    def reset_peaks(self):
        with self.lock:
            self.peak = {}
            self.memmin = {}

    def peaks(self):
        with self.lock:
            return {n: [round(v, 1) for v in pk] for n, pk in self.peak.items()}

    def snapshot(self):
        with self.lock:
            return {n: (round(time.time() - v[0], 1), v[1], v[2], v[3]) for n, v in self.last.items()}

    def wait_cool(self, zcool, gcool, max_s):
        t0 = time.time()
        while time.time() - t0 < max_s:
            if self.trip:
                return None
            with self.lock:
                fresh = {n: v for n, v in self.last.items() if time.time() - v[0] < 15}
            if len(fresh) == len(self.names) and all(v[1] <= zcool and v[2] <= gcool for v in fresh.values()):
                return round(time.time() - t0, 1)
            time.sleep(2)
        return None

    def close(self):
        self.stop_ev.set()
        self.th.join(timeout=10)
        for h, loc in zip(self.hosts, self.local):
            if not loc:
                subprocess.run(["ssh", "-o", f"ControlPath={self.cm}/%C", "-O", "exit", self.target(h)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=10, check=False)
        self.pool.shutdown(wait=False)
        shutil.rmtree(self.cm, ignore_errors=True)


class Req:
    def __init__(self, u, prompt, max_tokens, seed, model, t0):
        self.u, self.prompt, self.max_tokens, self.seed, self.model, self.T0 = u, prompt, max_tokens, seed, model, t0
        self.t_send = self.t_first = self.t_last_tok = self.t_end = None
        self.t_last = None
        self.ct, self.ct_first, self.prompt_tokens, self.finish = 0, None, None, None
        self.tl, self.err, self.aborted, self.conn, self.usage_seen = [], None, None, None, False
        self.th = threading.Thread(target=self.run, daemon=True)

    def run(self):
        body = {"model": self.model, "messages": [{"role": "user", "content": self.prompt}],
                "max_tokens": self.max_tokens, "temperature": 0.6, "top_p": 0.95, "seed": self.seed,
                "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True},
                "chat_template_kwargs": {"enable_thinking": False}, "cache_salt": f"ctx-{uuid.uuid4().hex}"}
        self.t_send = self.t_last = time.time()
        try:
            self.conn = connect()
            self.conn.request("POST", PREFIX + "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = self.conn.getresponse()
            if r.status != 200:
                self.err = f"HTTP {r.status}: {r.read()[:400]!r}"
                return
            while True:
                ln = r.readline()
                if not ln:
                    break
                ln = ln.strip()
                if not ln.startswith(b"data:"):
                    continue
                d = ln[5:].strip()
                if d == b"[DONE]":
                    break
                ch = json.loads(d)
                now = time.time()
                self.t_last = now
                u = ch.get("usage") or {}
                if u:
                    self.usage_seen = True
                    if u.get("prompt_tokens"):
                        self.prompt_tokens = u["prompt_tokens"]
                chs = ch.get("choices") or []
                if chs and chs[0].get("finish_reason"):
                    self.finish = chs[0]["finish_reason"]
                ct = u.get("completion_tokens")
                if ct is not None and ct > self.ct:
                    if self.t_first is None:
                        self.t_first, self.ct_first = now, ct
                    self.ct, self.t_last_tok = ct, now
                    self.tl.append((round(now - self.T0, 3), ct))
            self.t_end = time.time()
        except Exception as e:
            if not self.aborted:
                self.err = repr(e)[:300]
        finally:
            try:
                self.conn.close()
            except Exception:
                pass

    def abort(self, why):
        self.aborted = why
        try:
            self.conn.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass

    def result(self):
        r = {"u": self.u, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.ct, "finish": self.finish,
             "err": self.err, "aborted": self.aborted, "usage_seen": self.usage_seen}
        if self.t_first:
            r["ttft_s"] = round(self.t_first - self.t_send, 3)
            if self.prompt_tokens:
                r["prefill_tok_s"] = round(self.prompt_tokens / (self.t_first - self.t_send), 1)
            if self.t_last_tok and self.t_last_tok > self.t_first and self.ct > self.ct_first:
                r["decode_s"] = round(self.t_last_tok - self.t_first, 3)
                r["decode_tok_s"] = round((self.ct - self.ct_first) / (self.t_last_tok - self.t_first), 2)
        r["tl"] = self.tl
        return r


def tokens_at(tl, t):
    """completion tokens streamed by time t (t relative to the cell start), linear inside a gap."""
    prev = (None, 0)
    for tt, ct in tl:
        if tt >= t:
            if prev[0] is None:
                return 0.0
            return prev[1] + (ct - prev[1]) * (t - prev[0]) / max(1e-9, tt - prev[0])
        prev = (tt, ct)
    return float(prev[1])


def build_prompts(cells, model, docqa_path, agent_path):
    need = max((u_count - 1) * (n + 4096) + n for n, u_count in cells) + 2048
    src = "\n\n".join(json.loads(ln)["doc"] for ln in open(docqa_path))
    if need > 900_000:   # prompts beyond the W3 corpus continue into the W2 agent traces (prefix unchanged)
        extra = []
        for ln in open(agent_path):
            for m in json.loads(ln).get("messages", []):
                c = m.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                if isinstance(c, str) and c:
                    extra.append(c)
        src += "\n\n" + "\n\n".join(extra)
    toks, pos, step = [], 0, 300_000
    while len(toks) < need and pos < len(src):
        piece = src[pos:pos + step]
        cut = piece.rfind("\n") if pos + step < len(src) else len(piece)
        piece = piece[:cut if cut > step // 2 else len(piece)]
        toks += json.loads(call("/tokenize", {"model": model, "prompt": piece, "add_special_tokens": False}))["tokens"]
        pos += len(piece)
    if len(toks) < need:
        raise SystemExit(f"source has {len(toks)} tokens, need {need}")
    prompts = {}
    for n, u_count in cells:
        for u in range(u_count):
            off = u * (n + 4096)
            text = json.loads(call("/detokenize", {"model": model, "tokens": toks[off:off + n - DOC_OVERHEAD]}))["prompt"]
            prompts[(n, u)] = HEAD + text + INSTR
    return prompts, len(toks)


def run_cell(n, u_count, rep, args, prompts, model, fleet, out):
    cell = {"N": n, "U": u_count, "rep": rep, "max_tokens": args.max_tokens}
    if fleet:
        waited = fleet.wait_cool(args.cool, args.gcool, args.cool_max)
        if waited is None:
            cell["aborted"] = "fleet did not reach the cool level" + (f" (watchdog: {fleet.trip})" if fleet.trip else "")
            cell["fleet_last"] = fleet.snapshot()
            return cell, True
        cell["waited_cool_s"] = waited
        fleet.reset_peaks()
    m0, mx, stop_poll = metrics(), {"kv": 0.0, "run": 0.0, "wait": 0.0}, threading.Event()

    def poll():
        while not stop_poll.wait(2):
            try:
                m = metrics()
                mx["kv"] = max(mx["kv"], m.get("vllm:kv_cache_usage_perc", 0))
                mx["run"] = max(mx["run"], m.get("vllm:num_requests_running", 0))
                mx["wait"] = max(mx["wait"], m.get("vllm:num_requests_waiting", 0))
            except Exception:
                pass
    pt = threading.Thread(target=poll, daemon=True)
    pt.start()
    T0 = time.time()
    cell["t_start_unix"] = round(T0, 1)
    reqs = [Req(u, prompts[(n, u)], args.max_tokens, 1000 + u, model, T0) for u in range(u_count)]
    for r in reqs:
        r.th.start()
    pre_limit = 120 + 2 * (u_count * n / 1500.0)
    abort = None
    while any(r.th.is_alive() for r in reqs):
        now = time.time()
        if fleet and fleet.trip and not abort:
            abort = "thermal: " + fleet.trip
        for r in reqs:
            if abort:
                break
            if r.t_first is None and r.t_send and now - r.t_send > pre_limit:
                abort = f"stall: user {r.u} has no first token after {now - r.t_send:.0f} s (limit {pre_limit:.0f} s)"
            elif r.t_first is not None and r.th.is_alive() and now - r.t_last > args.stall_post:
                abort = f"stall: user {r.u} streamed nothing for {now - r.t_last:.0f} s"
        if now - T0 > args.max_cell_s and not abort:
            abort = f"cell exceeded {args.max_cell_s} s"
        if abort:
            for r in reqs:
                if r.th.is_alive():
                    r.abort(abort)
            for r in reqs:
                r.th.join(timeout=15)
            break
        time.sleep(1)
    stop_poll.set()
    pt.join(timeout=15)
    try:
        m1 = metrics()
        cell["metrics_delta"] = {k.replace("vllm:", ""): round(m1.get(k, 0) - m0.get(k, 0), 4) for k in KEYS
                                 if k.endswith("_total")}
    except Exception as e:
        cell["metrics_error"] = repr(e)[:200]
    cell.update({"kv_usage_max": round(mx["kv"], 4), "running_max": mx["run"], "waiting_max": mx["wait"],
                 "aborted": abort})
    if fleet:
        cell.update({"fleet_peak": fleet.peaks(),
                     "mem_avail_min_gib": {n: round(v, 1) for n, v in fleet.memmin.items()}})
    res = [r.result() for r in reqs]
    cell["reqs"] = res
    pk = cell.get("fleet_peak")
    if pk:
        zn = max(pk, key=lambda k: pk[k][0])
        gn = max(pk, key=lambda k: pk[k][1])
        cell.update({"zmax_c": pk[zn][0], "zmax_node": zn, "gpu_max_c": pk[gn][1], "gpu_max_node": gn,
                     "fleet_power_peak_w": round(sum(v[2] for v in pk.values()), 1)})
    md = cell.get("metrics_delta", {})
    drafts = md.get("spec_decode_num_drafts_total", 0)
    if u_count == 1 and drafts and res[0].get("decode_s"):
        cell["mean_accept_len"] = round(1 + md.get("spec_decode_num_accepted_tokens_total", 0) / drafts, 3)
        cell["step_ms"] = round(1000 * res[0]["decode_s"] / drafts, 2)
    if u_count > 1 and all(r.get("ttft_s") for r in res):
        w0 = max(r.t_first for r in reqs) - T0
        w1 = min(r.t_last_tok for r in reqs) - T0
        if w1 - w0 >= 5:
            per = [round((tokens_at(r.tl, w1) - tokens_at(r.tl, w0)) / (w1 - w0), 2) for r in reqs]
            cell["window"] = {"t0": round(w0, 2), "t1": round(w1, 2), "s": round(w1 - w0, 2), "per_user_tok_s": per,
                              "agg_tok_s": round(sum(per), 2)}
        else:
            cell["window"] = {"t0": round(w0, 2), "t1": round(w1, 2), "s": round(w1 - w0, 2), "note": "overlap < 5 s"}
    stop = bool(abort)
    return cell, stop


def line(c):
    s = f"N={c['N']} U={c['U']} rep{c['rep']}: "
    if c.get("aborted") and not c.get("reqs"):
        return s + "ABORTED " + c["aborted"]
    rs = c["reqs"]
    s += "TTFT " + "/".join(f"{r['ttft_s']:.1f}" if r.get("ttft_s") else "-" for r in rs) + " s"
    s += ", prompt " + "/".join(str(r.get("prompt_tokens")) for r in rs)
    if c["U"] == 1 and rs[0].get("decode_tok_s"):
        s += f", prefill {rs[0].get('prefill_tok_s')} tok/s, decode {rs[0]['decode_tok_s']} tok/s"
        if c.get("step_ms"):
            s += f", step {c['step_ms']} ms, AL {c['mean_accept_len']}"
    elif c.get("window", {}).get("agg_tok_s"):
        w = c["window"]
        s += f", all-decoding {w['s']:.0f} s: {'/'.join(str(x) for x in w['per_user_tok_s'])} tok/s, agg {w['agg_tok_s']}"
    md = c.get("metrics_delta", {})
    s += (f", kv max {100 * c.get('kv_usage_max', 0):.1f} %, preempt {md.get('num_preemptions_total')}, cache hits "
          f"{md.get('prefix_cache_hits_total')}")
    if "zmax_c" in c:
        s += f", zmax {c.get('zmax_c')} ({c.get('zmax_node')}), gpu {c.get('gpu_max_c')}"
    if c.get("aborted"):
        s += " ABORTED: " + c["aborted"]
    errs = [r["err"] for r in rs if r.get("err")]
    if errs:
        s += " ERR: " + errs[0]
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", help="output JSON file (not needed with --dry-run or --build-only)")
    ap.add_argument("--cells", default="8192x1,16384x1,32768x1,49152x1,65536x1",
                    help="comma-separated NxU cells: N prompt tokens per user, U simultaneous users (default: %(default)s)")
    ap.add_argument("--reps", type=int, default=2, help="rounds over all cells (default 2)")
    ap.add_argument("--max-tokens", type=int, default=512, help="tokens every request decodes (default 512)")
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL") or "http://127.0.0.1:8000",
                    help="server root, without /v1 (default: $BASE_URL, else http://127.0.0.1:8000)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="folder with w3_docqa-v1.jsonl and w2_agent-v1.jsonl (default: $BENCH_DATA_DIR, else bench/data/prepared)")
    ap.add_argument("--stall-post", type=float, default=60.0,
                    help="seconds without a streamed token after the first one that count as a stall (default 60)")
    ap.add_argument("--max-cell-s", type=float, default=1500.0, help="abort a cell that runs longer than this (default 1500)")
    g = ap.add_argument_group("optional node guard (off by default; reads the nodes over ssh)")
    g.add_argument("--guard", action="store_true", help="enable the thermal / memory watchdog described above")
    g.add_argument("--nodes", default=None, help="node ssh names, space or comma separated (default: $NODES, else cluster.env)")
    g.add_argument("--cluster-env", default=os.environ.get("CLUSTER_ENV") or os.path.join(REPO, "cluster.env"),
                   help="cluster description with NODES or NODE1..NODE8, SSH_OPTS, SSH_USER (default: $CLUSTER_ENV, else <repo>/cluster.env)")
    g.add_argument("--zstop", type=float, default=87.0, help="hottest thermal zone that stops the run, degrees C (default 87)")
    g.add_argument("--gstop", type=float, default=83.0, help="GPU temperature that stops the run, degrees C (default 83)")
    g.add_argument("--memstop", type=float, default=0.0, help="MemAvailable in GiB below which the run stops; 0 = off (default 0)")
    g.add_argument("--cool", type=float, default=65.0, help="every node's hottest zone must be at most this before a cell (default 65)")
    g.add_argument("--gcool", type=float, default=60.0, help="every GPU must be at most this before a cell (default 60)")
    g.add_argument("--cool-max", type=float, default=600.0, help="longest wait for the nodes to cool, seconds (default 600)")
    ap.add_argument("--dry-run", action="store_true", help="with the guard: take two readings from every node, print them, send nothing")
    ap.add_argument("--build-only", action="store_true", help="build the prompts and print their sizes; no chat request")
    a = ap.parse_args()
    try:
        cells = [(int(x.split("x")[0]), int(x.split("x")[1])) for x in a.cells.split(",")]
    except (ValueError, IndexError):
        ap.error(f"--cells {a.cells!r}: expected a comma-separated list of NxU, for example 8192x1,32768x4")
    set_endpoint(a.base_url)
    if a.dry_run:
        a.guard = True
    hosts = None
    if a.guard:
        hosts, ssh_opts, ssh_user = guard_config(a.nodes, a.cluster_env)
        print(f"guard: ON, {len(hosts)} node(s) read every 5 s; stop at zone >= {a.zstop} C, GPU >= {a.gstop} C"
              f"{', MemAvailable < ' + str(a.memstop) + ' GiB' if a.memstop else ''}", flush=True)
    else:
        print("guard: OFF -- no node temperature / memory watchdog and no cool-down wait before cells. Long cells heat the "
              "nodes: watch them yourself, or pass --guard (needs ssh to the nodes, see bench/README.md).", flush=True)
    if a.dry_run:
        f = Fleet(hosts, ssh_opts, ssh_user, a.zstop, a.gstop, memstop=a.memstop)
        time.sleep(11)
        print(json.dumps({"snapshot": f.snapshot(), "memmin": f.memmin, "fails": f.fails, "trip": f.trip}))
        missing = f.first_readings(timeout=0)
        f.close()
        if missing:
            shown = ", ".join(f"{n} ({hosts[f.names.index(n)]})" for n in missing)
            print(f"guard: no reading from {shown}. Check that `ssh <node> true` and `nvidia-smi` work without a prompt.",
                  file=sys.stderr, flush=True)
            return 3
        return 0
    if not a.out and not a.build_only:
        ap.error("OUT.json is required (except with --dry-run or --build-only)")
    docqa = os.path.join(a.data_dir, "w3_docqa-v1.jsonl")
    agent = os.path.join(a.data_dir, "w2_agent-v1.jsonl")
    need_agent = max((u_count - 1) * (n + 4096) + n for n, u_count in cells) + 2048 > 900_000
    for path, prep in ((docqa, "prep_w3_docqa.py"),) + (((agent, "prep_w2_agent.py"),) if need_agent else ()):
        if not os.path.isfile(path):
            raise SystemExit(f"{path} not found: build it with bench/data/{prep} (see bench/data/README.md) or pass --data-dir")
    model = json.loads(call("/v1/models", timeout=30))["data"][0]["id"]
    t = time.time()
    prompts, ntok = build_prompts(cells, model, docqa, agent)
    print(f"{VERSION}: source {ntok} tokens, {len(prompts)} prompts built in {time.time() - t:.1f} s", flush=True)
    if a.build_only:
        for (n, u), p in sorted(prompts.items()):
            cnt = json.loads(call("/tokenize", {"model": model, "messages": [{"role": "user", "content": p}],
                                                "chat_template_kwargs": {"enable_thinking": False}}))["count"]
            print(f"  N={n} u={u}: {len(p)} chars, {cnt} chat prompt tokens", flush=True)
        return 0
    keep = ["cells", "reps", "max_tokens", "stall_post", "max_cell_s"] + (["zstop", "gstop", "memstop", "cool", "gcool", "cool_max"] if a.guard else [])
    out = {"version": VERSION, "model": model, "guard": bool(a.guard), "args": {k: getattr(a, k) for k in keep}, "cells": []}
    fleet = None
    if a.guard:
        fleet = Fleet(hosts, ssh_opts, ssh_user, a.zstop, a.gstop, memstop=a.memstop)
        missing = fleet.first_readings()
        if missing:
            shown = ", ".join(f"{n} ({hosts[fleet.names.index(n)]})" for n in missing)
            print(f"guard: no reading from {shown}. Check that `ssh <node> true` and `nvidia-smi` work without a prompt, or run "
                  "without --guard. Not starting without the guard you asked for.", file=sys.stderr, flush=True)
            fleet.close()
            return 3
    rc = 0
    try:
        for rep in range(a.reps):
            for n, u_count in cells:
                c, stop = run_cell(n, u_count, rep, a, prompts, model, fleet, out)
                out["cells"].append(c)
                with open(a.out, "w") as fh:
                    json.dump(out, fh, indent=1)
                print(line(c), flush=True)
                if stop:
                    rc = 3 if "stall" in (c.get("aborted") or "") else 2
                    print(f"STOP: {c.get('aborted')}", flush=True)
                    return rc
    finally:
        if fleet:
            fleet.close()
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
