#!/usr/bin/env python3
"""canary-leak-test.py [BASE_URL] [options]  -- cross-request leakage / corruption canary for prefix cache + speculative decoding.

Why: serving stacks that combine prefix caching, speculative decoding (MTP, DSpark) and async scheduling over hybrid or sparse
attention caches belong to a class
with public field reports of (a) degenerate output ("!!!!", "= = =", empty), (b) CROSS-REQUEST content exposure and (c) warm-cache
outputs that differ from the same request computed cold (vllm-project/vllm#53912 and friends).  The reports are rare (~0.3-3 % of
requests), silent (no error, normal finish_reason) and load dependent: a synthetic one-shot test is clean (the field reporters saw
exactly that).  This tool therefore runs N long-lived, concurrent, agent-shaped sessions against ANY OpenAI-compatible
/v1/chat/completions endpoint for a fixed time budget and checks every response:

  (a) LEAK       a marker / access code / invented word that belongs to ANOTHER session (or to another client family's preamble) shows
                 up in a response (content or reasoning).  Every session carries unique random canaries in its system prompt and in
                 every tool output; sessions of one family share a long preamble with their siblings (cross-session prefix hits and
                 shared-prefix junctions) but never a canary.  A PHANTOM (canary-shaped string nobody was ever given) is a corrupted copy.
  (b) DEGENERATE repeated punctuation / n-gram loops / garbage script / empty answer with finish_reason=stop / failed exact-copy tasks
                 (count-to-N, recall of an 8-digit code from deep in the context, echo of the session marker and of the latest tool id).
  (c) COLD/WARM  a sample of the requests is replayed with a fresh cache_salt (guaranteed cold prefill; usage.prompt_tokens_details.
                 cached_tokens must be 0) and compared with the warm answer: greedy answers are compared token by token with logprobs,
                 so a near-tie flip (benign numerics) is separated from a hard divergence (wrong restored state).  Cold replays are also
                 the control group for the recall/echo/degenerate failure rates (Fisher exact test, warm vs cold).
  Extras: forks (same history, different question, fired concurrently with the main turn -> same-step writer/reader and partial-tail
  copy-on-write), temperature 0 and 1, contexts from --ctx-min to --ctx-max, per-session decode-token accounting (the "accumulated
  decode" effect of #56868), Prometheus deltas, warm-vs-warm determinism.

Statistics: with n requests and no event the exact 95 % upper bound on the per-request rate is ~3/n, so certifying < 0.5 % needs
n >= 600 and < 1 % needs n >= 300 (printed at the end and by --dry-run).
Exit codes: 0 pass, 1 FAIL (leak / degenerate / hard divergence / warm-vs-cold failure excess), 2 WARN or underpowered, 3 harness error.

Safe against production: read-only chat requests, the only side effects are KV-cache pressure and load.  Stdlib only (python >= 3.8).
Requests go only to BASE_URL (/v1/chat/completions, /v1/models, /metrics).  An API key (if any) is read from an environment variable
and never printed or stored.  BASE_URL is the server root without /v1; if it is left out, $BASE_URL is used, else http://127.0.0.1:8000.
The report does not record the URL, the CA file or the output path.

Examples:
  canary-leak-test.py https://HOST --ca-cert ca.pem --sessions 8 --time-budget 3600 --out-dir canary-out
  canary-leak-test.py http://127.0.0.1:8000 --dry-run                  # plan + sample prompt, no network
  canary-leak-test.py --mock --mock-faults leak=0.03,degenerate=0.03 --time-budget 20 --ctx-min 2000 --ctx-max 6000   # detector check
  canary-leak-test.py --selftest                                        # mock: clean run must be clean, faulted runs must be caught
Thinking models: if most requests end as BUDGET (content empty, finish_reason=length) raise --max-tokens or pass
--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'.
"""
from __future__ import annotations

import argparse
import string
import collections
import hashlib
import json
import math
import os
import random
import re
import signal
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

VERSION = "1.1"


# --------------------------------------------------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------------------------------------------------

def now() -> float:
    return time.time()


def iso(ts=None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts if ts is not None else now()))


def sha(s: str, n: int = 12) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:n]


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Bin(n, p), computed in log space (exact, stdlib only)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = []
    for i in range(0, min(k, n) + 1):
        lc = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        terms.append(lc + i * lp + (n - i) * lq)
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def upper95(k: int, n: int) -> float:
    """One-sided exact (Clopper-Pearson) 95 % upper bound for a rate with k events in n trials."""
    if n <= 0 or k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


def fisher_greater(a_fail: int, a_n: int, b_fail: int, b_n: int) -> float:
    """One-sided Fisher exact p-value that group A's failure rate exceeds group B's (hypergeometric tail)."""
    if a_n <= 0 or b_n <= 0:
        return 1.0
    n_all, t = a_n + b_n, a_fail + b_fail
    if t == 0:
        return 1.0
    p = 0
    for x in range(a_fail, min(a_n, t) + 1):
        if t - x <= b_n:
            p += math.comb(a_n, x) * math.comb(b_n, t - x)
    return min(1.0, p / math.comb(n_all, t))


def need_n(rate: float) -> int:
    return int(math.ceil(3.0 / rate))


# --------------------------------------------------------------------------------------------------------------------------------
# text generation (deterministic from a seed; every canary is unique across the whole run)
# --------------------------------------------------------------------------------------------------------------------------------

COMMON = (
    "the of and to in is it that for on with as are this be at or by from an we not have more can will one all there which their about "
    "service request response latency queue worker cluster node shard replica index schema column table record field value update "
    "deploy release rollback incident alert metric trace span log error warning retry timeout budget limit quota policy account team "
    "owner review ticket status report summary change config setting default option flag version build test check result failure "
    "success pending running stopped scheduled planned approved rejected migrated archived restored backup snapshot volume network "
    "gateway proxy cache store memory disk cpu thread process batch stream window offset cursor token session client server host port "
    "route path header cookie secret key cert rotation audit access role group member admin user guest invoice payment ledger balance "
    "region zone rack power cooling sensor reading threshold anomaly baseline trend forecast capacity usage cost saving target goal "
    "milestone sprint backlog story epic task bug fix patch hotfix feature module package library import export parse render compile "
    "optimize measure compare verify validate monitor notify escalate resolve close open assign merge branch commit push pull tag"
).split()
COMMON_SET = set(COMMON)

SYL = ["ka", "ve", "lo", "mi", "ra", "zu", "te", "no", "shi", "bor", "dan", "fel", "gur", "hax", "jin", "kor", "lum", "mar",
       "nel", "ost", "pri", "quo", "rin", "sel", "tor", "ulm", "vex", "wyn", "yor", "zan", "bri", "cal", "dro", "eth", "fyn"]

MARK_RE = re.compile(r"\bzk[a-z0-9]{10}\b")
CODE_RE = re.compile(r"(?<![0-9])[0-9]{8}(?![0-9])")
WORD_RE = re.compile(r"[a-z]{6,14}")
CJK_RE = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯�]")
RUN_PUNCT_RE = re.compile(r"([!?.,;:'\"`^&@$%()\[\]{}<>/\\|+])\1{24,}")      # 25+ identical punctuation chars ("!!!!!!")
RUN_ANY_RE = re.compile(r"(\S)\1{79,}")                                         # 80+ identical non-space chars
SP_LOOP_RE = re.compile(r"(\S{1,24}?)(?:\s+\1){11,}")                           # "= = = = ..." / "the the the ..."
ADJ_LOOP_RE = re.compile(r"(\S{2,16}?)\1{11,}")                                 # ".0.0.0.0.0..." / "abcabcabc..."


class Registry:
    """Everything the harness ever handed to a model, keyed to its owner (a session id; "F<n>" for family preambles)."""

    def __init__(self, rng: random.Random):
        self.lock = threading.Lock()
        self.rng = rng
        self.marker_owner: dict[str, str] = {}
        self.code_owner: dict[str, tuple] = {}            # code -> (owner session, label)
        self.word_owner: dict[str, str] = {}
        self.session_family: dict[str, str] = {}

    def new_marker(self, owner: str) -> str:
        with self.lock:
            while True:
                m = "zk" + "".join(self.rng.choice(string.ascii_lowercase + string.digits) for _ in range(10))
                if m not in self.marker_owner:
                    self.marker_owner[m] = owner
                    return m

    def new_code(self, owner: str, label: str) -> str:
        with self.lock:
            while True:
                c = "%08d" % self.rng.randrange(10000000, 99999999)
                if c not in self.code_owner:
                    self.code_owner[c] = (owner, label)
                    return c

    def new_word(self, owner: str) -> str:
        with self.lock:
            while True:
                w = "".join(self.rng.choice(SYL) for _ in range(self.rng.choice((3, 4)))) + self.rng.choice(("ex", "on", "ar", "ik"))
                if 6 <= len(w) <= 14 and w not in self.word_owner and w not in COMMON_SET:
                    self.word_owner[w] = owner
                    return w


def filler(rng: random.Random, lexicon, target_chars: int, inserts=None) -> str:
    """Agent-shaped filler: prose, log lines, JSON and code-ish blocks, sprinkled with the session lexicon.
    `inserts` (notes / ids) are dropped at random depths so that retrieval needs the whole context."""
    out, size = [], 0
    while size < target_chars:
        style = rng.random()
        if style < 0.40:      # prose
            words = [rng.choice(lexicon) if rng.random() < 0.012 else rng.choice(COMMON) for _ in range(rng.randint(30, 90))]
            para = " ".join(words).capitalize() + ".\n"
        elif style < 0.65:    # log lines
            lines = []
            for _ in range(rng.randint(4, 10)):
                lines.append("2026-10-%02dT%02d:%02d:%02dZ %s-%02d %s req=%06x lat=%dms path=/api/v%d/%s/%d status=%d" % (
                    rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59), rng.choice(COMMON), rng.randint(1, 40),
                    rng.choice(("INFO", "INFO", "WARN", "ERROR", "DEBUG")), rng.getrandbits(24), rng.randint(2, 900), rng.randint(1, 3),
                    rng.choice(COMMON), rng.randint(1, 99999), rng.choice((200, 200, 200, 204, 404, 500))))
            para = "\n".join(lines) + "\n"
        elif style < 0.85:    # json
            items = ['{"id": %d, "name": "%s", "tag": "%s", "score": %.3f, "ok": %s}' % (
                rng.randint(1, 999999), rng.choice(COMMON), rng.choice(lexicon) if rng.random() < 0.15 else rng.choice(COMMON),
                rng.random() * 100, rng.choice(("true", "false"))) for _ in range(rng.randint(3, 8))]
            para = "[" + ",\n ".join(items) + "]\n"
        else:                 # code-ish
            fn = rng.choice(COMMON) + "_" + rng.choice(COMMON)
            para = ("def %s(%s, %s):\n    # %s\n    total = 0\n    for i in range(%d):\n        total += %s(i) * %d\n    return total\n" % (
                fn, rng.choice(COMMON), rng.choice(COMMON), " ".join(rng.choice(COMMON) for _ in range(6)), rng.randint(3, 99),
                rng.choice(COMMON), rng.randint(2, 97)))
        out.append(para)
        size += len(para)
    text = "".join(out)
    for m in inserts or ():
        cut = rng.randrange(0, max(1, len(text)))
        nl = text.find("\n", cut)
        nl = len(text) if nl < 0 else nl + 1
        text = text[:nl] + m + "\n" + text[nl:]
    return text


class Cpt:
    """chars-per-token estimate, learned from usage.prompt_tokens (thread-safe EMA)."""

    def __init__(self, v: float = 3.5):
        self.v, self.lock = v, threading.Lock()

    def get(self) -> float:
        return self.v

    def update(self, chars: int, tokens: int) -> None:
        if tokens > 500 and chars > 1000:
            with self.lock:
                self.v = 0.9 * self.v + 0.1 * max(1.5, min(8.0, chars / tokens))


# --------------------------------------------------------------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------------------------------------------------------------

class Session:
    def __init__(self, sid: str, family: str, rng: random.Random, reg: Registry, args, shared: dict, cpt: Cpt):
        self.sid, self.family, self.rng, self.reg, self.args, self.cpt = sid, family, rng, reg, args, cpt
        reg.session_family[sid] = family
        self.lexicon = [reg.new_word(sid) for _ in range(40)]
        self.marker = reg.new_marker(sid)
        self.latest_marker = self.marker
        self.labels: list = []                                    # (label, code, turn defined)
        self.used_labels: set = set()
        self.turn = 0
        self.decode_tokens = 0
        self.prompt_tokens_last = 0
        lo, hi = args.ctx_min, args.ctx_max
        self.ctx_target = rng.randint(min(hi, lo + (hi - lo) // 3), hi)
        self.start_tokens = rng.randint(lo, max(lo, min(hi, int(lo * 2.2))))
        pre = shared["global"] + shared["family"][family]
        sysmsg = (pre + "\n\n== Session ==\nSession identifier: %s\nYou are the operations assistant of team %s. Answer exactly what is asked, "
                  "never invent identifiers, and quote identifiers and codes exactly as written.\n" % (self.marker, self.lexicon[0]))
        self.messages: list = [{"role": "system", "content": sysmsg}]
        self.pending_user = self._new_user(first=True)

    def _tool_output(self, tokens: int, first: bool) -> str:
        rid = self.reg.new_marker(self.sid)
        self.latest_marker = rid
        notes = []
        for _ in range(self.rng.randint(2, 4) if first else self.rng.randint(0, 2)):
            label = self.rng.choice(self.lexicon[1:]) + "-" + str(self.rng.randint(10, 99))
            while label in self.used_labels:                      # a label must identify exactly one code
                label = self.rng.choice(self.lexicon[1:]) + "-" + str(self.rng.randint(10, 999))
            self.used_labels.add(label)
            code = self.reg.new_code(self.sid, label)
            self.labels.append((label, code, self.turn))
            notes.append("NOTE: the access code for label %s is %s." % (label, code))
        body = filler(self.rng, self.lexicon, int(tokens * self.cpt.get()), notes)
        return "TOOL OUTPUT (RESULT-ID: %s)\n%s\nEND OF TOOL OUTPUT (RESULT-ID: %s)\n" % (rid, body, rid)

    def _new_user(self, first: bool) -> str:
        if first:
            toks = max(500, self.start_tokens - int(len(self.messages[0]["content"]) / self.cpt.get()))
        else:
            toks = self.rng.randint(self.args.grow_min, self.args.grow_max)
        return self._tool_output(toks, first)

    def est_tokens(self) -> int:
        n = sum(len(m["content"]) for m in self.messages) + len(self.pending_user or "")
        return int(n / self.cpt.get())

    def done(self) -> bool:
        return self.est_tokens() >= self.ctx_target

    def question(self):
        """-> (kind, question text, expectation dict, temperature)"""
        r = self.rng
        kinds = [("recall", 0.30), ("echo", 0.20), ("summary", 0.20), ("counting", 0.15), ("chat", 0.15)]
        old = [x for x in self.labels if x[2] < self.turn] or self.labels
        if not old:
            kinds = [k for k in kinds if k[0] != "recall"]
        x = r.random() * sum(w for _, w in kinds)
        kind = kinds[-1][0]
        for k, w in kinds:
            x -= w
            if x <= 0:
                kind = k
                break
        temp = 0.0 if (kind == "counting" or r.random() >= self.args.t1_frac) else 1.0
        if kind == "recall":
            label, code, _ = (r.choice(old[: max(1, len(old) // 2)]) if r.random() < 0.6 else r.choice(old))
            return kind, ("Question: according to the notes in this conversation, what is the access code for label %s? "
                          "Reply with the eight digits only." % label), {"code": code, "label": label}, temp
        if kind == "echo":
            return kind, ("Question: reply with exactly one line of the form  SESSION=<session identifier> LATEST=<RESULT-ID of the most "
                          "recent tool output>  copying both identifiers exactly."), {"markers": [self.marker, self.latest_marker]}, temp
        if kind == "summary":
            return kind, ("Question: summarize the most recent tool output in at most 70 words and quote its RESULT-ID exactly once."), \
                   {"markers": [self.latest_marker]}, temp
        if kind == "counting":
            a, n = r.randint(1000, 90000), r.randint(150, 300)
            return kind, ("Question: count from %d to %d, one number per line, with no other text." % (a, a + n - 1)), {"start": a, "n": n}, 0.0
        return "chat", ("Question: write a short status update (about 120 words, no bullet points) for the team about the work "
                        "described so far."), {}, temp

    def build_messages(self, question: str) -> list:
        return self.messages + [{"role": "user", "content": self.pending_user + "\n" + question}]

    def commit(self, question: str, answer: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.messages.append({"role": "user", "content": self.pending_user + "\n" + question})
        self.messages.append({"role": "assistant", "content": answer if answer.strip() else "(no answer)"})
        self.turn += 1
        self.decode_tokens += completion_tokens
        self.prompt_tokens_last = prompt_tokens
        self.pending_user = self._new_user(first=False)


def make_shared(rng: random.Random, reg: Registry, args, cpt: Cpt) -> dict:
    """Preambles shared between sessions: one global (same for everybody) and one per client family (jittered length so that the
    shared-prefix junction lands at different offsets relative to the 1,280-token cache blocks)."""
    glob_words = [reg.new_word("global") for _ in range(20)]
    g = "== Tool documentation (shared by all clients) ==\n" + filler(rng, glob_words, int(args.preamble_tokens * 0.5 * cpt.get())) + "\n"
    fam = {}
    for i in range(args.families):
        name = "F%d" % (i + 1)
        jitter = rng.choice((0, 1, 17, 640, 1279, 333, 2048, 5))
        lex = [reg.new_word(name) for _ in range(20)]
        fam[name] = "== Client profile %s ==\n" % name + filler(rng, lex, int((args.preamble_tokens * 0.5 + jitter) * cpt.get())) + "\n"
    return {"global": g, "family": fam}


# --------------------------------------------------------------------------------------------------------------------------------
# analysis of one response
# --------------------------------------------------------------------------------------------------------------------------------

def tokens_of(resp: dict) -> list:
    try:
        return resp["choices"][0]["logprobs"]["content"] or []
    except Exception:       # noqa: BLE001
        return []


def analyze(text: str, reasoning: str, finish: str, kind: str, exp: dict, sess: Session, reg: Registry, prompt_ascii: bool) -> list:
    """Return findings [{kind, severity, detail}].  text = message content, reasoning = reasoning channel."""
    f: list = []
    both = text + "\n" + reasoning

    def add(k, sev, detail):
        f.append({"kind": k, "severity": sev, "detail": detail})

    # (a) leaks ---------------------------------------------------------------------------------------------------------------
    for m in sorted(set(MARK_RE.findall(both))):
        owner = reg.marker_owner.get(m)
        if owner is None:
            add("PHANTOM_MARKER", "high", {"marker": m, "note": "canary-shaped string nobody was given (corrupted copy?)"})
        elif owner != sess.sid:
            add("LEAK_MARKER", "critical", {"marker": m, "owner_session": owner, "owner_family": reg.session_family.get(owner),
                                            "victim_session": sess.sid, "victim_family": sess.family,
                                            "same_family": reg.session_family.get(owner) == sess.family})
    for c in sorted(set(CODE_RE.findall(both))):
        own = reg.code_owner.get(c)
        if own and own[0] != sess.sid:
            add("LEAK_CODE", "critical", {"code": c, "owner_session": own[0], "owner_family": reg.session_family.get(own[0]),
                                          "label": own[1], "victim_session": sess.sid, "victim_family": sess.family})
    foreign = sorted({w for w in WORD_RE.findall(both.lower())
                      if w in reg.word_owner and reg.word_owner[w] not in (sess.sid, sess.family, "global")})
    if foreign:
        add("LEAK_WORD", "high", {"words": foreign[:8], "owners": sorted({reg.word_owner[w] for w in foreign}), "victim_session": sess.sid})

    # (b) degenerate output ----------------------------------------------------------------------------------------------------
    if not text.strip():
        if finish == "length":
            add("BUDGET", "info", {"note": "content empty, finish_reason=length (reasoning used the budget)"})
        else:
            add("EMPTY", "high", {"finish_reason": finish, "reasoning_chars": len(reasoning)})
    probe = text if text.strip() else reasoning
    if kind != "counting" and probe:
        sq = re.sub(r"[ \t]+", " ", probe)
        why = None
        if RUN_PUNCT_RE.search(probe) or RUN_ANY_RE.search(probe):
            why = "long run of one repeated character"
        elif len(sq) > 200 and (SP_LOOP_RE.search(sq) or ADJ_LOOP_RE.search(sq)):
            why = "same unit repeated 12+ times"
        elif len(probe) > 400:
            raw = probe.encode("utf-8", "replace")
            ratio = len(zlib.compress(raw, 6)) / max(1, len(raw))
            if ratio < 0.08:
                why = "compression ratio %.3f" % ratio
        if why:
            add("DEGENERATE", "high", {"why": why, "head": probe[:80]})
    if prompt_ascii and probe:
        cj = len(CJK_RE.findall(probe))
        if cj >= 4:
            add("GARBAGE", "high", {"why": "%d CJK/replacement chars in an ASCII conversation" % cj, "head": probe[:80]})

    # task checks ---------------------------------------------------------------------------------------------------------------
    if kind == "recall":
        got = CODE_RE.findall(text)
        if text.strip() and exp["code"] not in got:
            add("RECALL_FAIL", "medium", {"expected": exp["code"], "label": exp["label"], "got": got[:4], "head": text[:60]})
    elif kind in ("echo", "summary"):
        miss = [m for m in exp["markers"] if m not in text]
        if text.strip() and miss:
            add("ECHO_FAIL" if kind == "echo" else "SUMMARY_FAIL", "medium", {"missing": miss, "head": text[:90]})
    elif kind == "counting":
        nums = [int(x) for x in re.findall(r"(?m)^\s*(\d{1,7})\s*$", text)]
        a, n = exp["start"], exp["n"]
        if text.strip():
            bad = None
            for i, v in enumerate(nums):
                if v != a + i:
                    bad = {"index": i, "expected": a + i, "got": v, "context": nums[max(0, i - 2): i + 3]}
                    break
            if not nums or nums[0] != a:
                add("COUNT_FAIL", "medium", {"why": "does not start at %d" % a, "head": text[:60]})
            elif bad:
                add("COUNT_FAIL", "medium", bad)
            elif len(nums) < n and finish != "length":
                add("COUNT_FAIL", "medium", {"why": "stopped early", "got": len(nums), "expected": n})
    return f


def compare_tokens(w: list, c: list, tie_gap: float) -> dict:
    """Compare two greedy answers (logprobs.content lists)."""
    n = min(len(w), len(c))
    lead = 0
    while lead < n and w[lead].get("token") == c[lead].get("token"):
        lead += 1
    dl = [abs((w[i].get("logprob") or 0.0) - (c[i].get("logprob") or 0.0)) for i in range(lead)]
    out = {"n_a": len(w), "n_b": len(c), "lead_identical": lead, "mean_abs_dlp": round(sum(dl) / len(dl), 5) if dl else 0.0,
           "max_abs_dlp": round(max(dl), 5) if dl else 0.0}
    if lead == n and len(w) == len(c):
        out["verdict"] = "IDENTICAL"
    elif lead >= n:
        out["verdict"] = "LENGTH_ONLY"
    else:
        wt, ct = w[lead], c[lead]
        b_top = {t.get("token"): t.get("logprob") for t in (ct.get("top_logprobs") or [])}
        a_top = {t.get("token"): t.get("logprob") for t in (wt.get("top_logprobs") or [])}
        gap_b = (ct.get("logprob") or 0.0) - b_top.get(wt.get("token"), -99.0)
        gap_a = (wt.get("logprob") or 0.0) - a_top.get(ct.get("token"), -99.0)
        out.update({"diverge_at": lead, "gap_in_b": round(gap_b, 4), "gap_in_a": round(gap_a, 4)})
        out["verdict"] = "TIE_FLIP" if max(gap_b, gap_a) <= tie_gap else "HARD_DIVERGENCE"
    return out


# --------------------------------------------------------------------------------------------------------------------------------
# transports: real HTTP, and an in-process mock that can inject faults (detector self-test)
# --------------------------------------------------------------------------------------------------------------------------------

class HttpTransport:
    def __init__(self, base: str, api_key, ca_cert, insecure: bool, timeout: float):
        self.base, self.timeout = base.rstrip("/"), timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key
        self.ctx = None
        if self.base.startswith("https"):
            self.ctx = ssl.create_default_context(cafile=ca_cert) if ca_cert else ssl.create_default_context()
            if insecure:
                self.ctx.check_hostname = False
                self.ctx.verify_mode = ssl.CERT_NONE

    def get(self, path: str, raw: bool = False, timeout: float = 30):
        req = urllib.request.Request(self.base + path, headers=self.headers)
        with urllib.request.urlopen(req, timeout=timeout, context=self.ctx) as r:
            data = r.read()
        return data.decode("utf-8", "replace") if raw else json.loads(data)

    def chat(self, body: dict) -> dict:
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.ctx) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError("HTTP %d: %s" % (e.code, e.read()[:300].decode("utf-8", "replace")))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise ConnectionError("transport: %r" % (e,))


class MockTransport:
    """A fake model good enough to exercise the harness: answers recall / echo / count questions from the conversation, simulates
    prefix-cache accounting, and injects faults on request (leak, degenerate, empty, diverge, wrong)."""

    def __init__(self, faults: dict, seed: int = 1):
        self.f = faults
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.seen: dict = collections.defaultdict(list)
        self.prev_prompt: dict = {}
        self.n = 0

    def get(self, path: str, raw: bool = False, timeout: float = 30):
        if path == "/v1/models":
            return {"data": [{"id": "mock-model"}]}
        if path.startswith("/metrics"):
            return "# mock\nvllm:prefix_cache_hits_total 0\n" if raw else {}
        raise RuntimeError("mock: no such path " + path)

    def chat(self, body: dict) -> dict:
        with self.lock:
            self.n += 1
            r = random.Random(self.rng.random())
        msgs = body["messages"]
        alltext = "\n".join(m["content"] for m in msgs)
        sysm = msgs[0]["content"]
        km = re.search(r"Session identifier: (zk[a-z0-9]{10})", sysm)
        key = km.group(1) if km else sha(sysm, 10)
        last = msgs[-1]["content"]
        q = last[last.rfind("Question:"):]
        marks = MARK_RE.findall(alltext)
        with self.lock:
            self.seen[key] = list(dict.fromkeys(self.seen[key] + marks))[-300:]
        if "access code for label" in q:
            lab = re.search(r"label (\S+)\?", q).group(1)
            m = re.search(r"access code for label %s is ([0-9]{8})" % re.escape(lab), alltext)
            text = m.group(1) if m else "00000000"
            if r.random() < self.f.get("wrong", 0.0):
                text = text[:-1] + str((int(text[-1]) + 1) % 10)
        elif "SESSION=" in q:
            text = "SESSION=%s LATEST=%s" % (km.group(1), MARK_RE.findall(last)[0])
        elif "count from" in q:
            a, b = [int(x) for x in re.search(r"count from (\d+) to (\d+)", q).groups()]
            text = "\n".join(str(i) for i in range(a, b + 1))
        elif "summarize" in q:
            text = "The tool output lists service records and log lines; RESULT-ID %s. Nothing needs escalation." % MARK_RE.findall(last)[0]
        else:
            h0 = int(sha(json.dumps(msgs[-1]) + str(body.get("temperature")), 8), 16)
            br = random.Random(h0) if body.get("temperature") == 0 else r        # greedy answers are deterministic
            text = " ".join(br.choice(COMMON) for _ in range(110)) + "."
        cold = "cache_salt" in body
        h = int(sha(json.dumps(msgs[-1]) + str(body.get("temperature")), 8), 16)
        if r.random() < self.f.get("leak", 0.0):
            others = [m for k, v in list(self.seen.items()) if k != key for m in v]
            if others:
                text += " " + r.choice(others)
        if r.random() < self.f.get("degenerate", 0.0):
            text = r.choice(["!" * 200, "= " * 120, "あいうえお" * 20])
        if r.random() < self.f.get("empty", 0.0):
            text = ""
        content_lp = []
        if body.get("logprobs"):
            toks = re.findall(r"\S+\s*", text) or [""]
            for i, t in enumerate(toks[: body.get("max_tokens", 64)]):
                lp = -0.02 * ((h + i * 7) % 5)
                content_lp.append({"token": t, "logprob": lp, "top_logprobs": [{"token": t, "logprob": lp}, {"token": t + "_", "logprob": lp - 0.15}]})
            if not cold and r.random() < self.f.get("diverge", 0.0) and len(content_lp) > 6:
                old = content_lp[4]["token"]
                content_lp[4] = {"token": "WRONG", "logprob": -0.1, "top_logprobs": [{"token": "WRONG", "logprob": -0.1}, {"token": old, "logprob": -5.0}]}
        ptok = int(len(alltext) / 3.5)
        cached = 0
        if not cold:
            pp = self.prev_prompt.get(key, "")
            i, lim = 0, min(len(pp), len(alltext))
            while i < lim and pp[i] == alltext[i]:
                i += 1
            cached = int(i / 3.5) // 8 * 8
        with self.lock:
            self.prev_prompt[key] = alltext
        time.sleep(0.002)
        resp = {"model": "mock-model", "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "reasoning_content": ""},
                                                     "finish_reason": "stop"}],
                "usage": {"prompt_tokens": ptok, "completion_tokens": max(1, len(text) // 4), "prompt_tokens_details": {"cached_tokens": cached}}}
        if body.get("logprobs"):
            resp["choices"][0]["logprobs"] = {"content": content_lp}
        return resp


# --------------------------------------------------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------------------------------------------------

class Run:
    def __init__(self, args, transport):
        self.a, self.tr = args, transport
        self.t0 = now()
        self.deadline = self.t0 + args.time_budget
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.events: list = []
        self.records: list = []
        self.cnt = collections.Counter()
        self.fail = {"warm": collections.Counter(), "cold": collections.Counter()}
        self.n = {"warm": collections.Counter(), "cold": collections.Counter()}
        self.cmp = collections.Counter()
        self.cpt = Cpt(args.chars_per_token)
        self.rng = random.Random(args.seed)
        self.reg = Registry(random.Random(args.seed + 1))
        self.shared = make_shared(random.Random(args.seed + 2), self.reg, args, self.cpt)
        self.sid_counter = 0
        self.sem = threading.BoundedSemaphore(max(1, args.max_inflight))
        self.consec_err = 0
        self.have_cached = True
        self.session_summaries: list = []
        self.printed = 0
        self.flag_dir = None
        if args.out_dir:
            self.flag_dir = os.path.join(args.out_dir, "flagged")
            os.makedirs(self.flag_dir, exist_ok=True)

    # ---- request plumbing
    def body(self, messages, temp, max_tokens, logprobs=False, salt=None) -> dict:
        b = {"model": self.a.model, "messages": messages, "temperature": temp, "max_tokens": max_tokens, "stream": False}
        if logprobs:
            b["logprobs"] = True
            b["top_logprobs"] = 5
        if salt:
            b["cache_salt"] = salt
        for k, v in (self.a.extra_body or {}).items():
            b.setdefault(k, v)
        return b

    def send(self, body, tag):
        t0 = now()
        err = None
        for attempt in (0, 1):
            try:
                with self.sem:                      # never more requests in flight than the server has sequence slots
                    t0 = now()
                    resp = self.tr.chat(body)
                resp["_latency"] = now() - t0
                self.consec_err = 0
                return resp
            except ConnectionError as e:        # transient transport problem: one retry
                err = e
                time.sleep(2.0)
            except Exception as e:              # noqa: BLE001
                err = e
                break
        with self.lock:
            self.cnt["http_errors"] += 1
            self.consec_err += 1
            self.events.append({"t": round(now() - self.t0, 1), "kind": "HTTP_ERROR", "severity": "info", "tag": tag, "detail": str(err)[:300]})
            if self.consec_err >= self.a.max_consecutive_errors:
                self.stop.set()
                self.cnt["aborted_on_errors"] = 1
        return None

    @staticmethod
    def parts(resp):
        ch = resp["choices"][0]
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        u = resp.get("usage") or {}
        det = u.get("prompt_tokens_details") or {}
        return text, reasoning, ch.get("finish_reason") or "", int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0), det.get("cached_tokens")

    def record(self, s: Session, tag, kind, temp, resp, findings, mode):
        _, _, fin, ptok, ctok, cached = self.parts(resp)
        rec = {"t": round(now() - self.t0, 1), "session": s.sid, "family": s.family, "turn": s.turn, "tag": tag, "kind": kind, "temp": temp,
               "mode": mode, "prompt_tokens": ptok, "completion_tokens": ctok, "cached_tokens": cached, "latency_s": round(resp["_latency"], 2),
               "finish": fin, "findings": [x["kind"] for x in findings]}
        with self.lock:
            self.records.append(rec)
            self.cnt["requests"] += 1
            self.cnt["prompt_tokens"] += ptok
            self.cnt["completion_tokens"] += ctok
            self.cnt["cached_tokens"] += cached or 0
            self.n[mode][kind] += 1
            self.n[mode]["_all"] += 1
            if ptok < 1000:
                self.cnt["_small_prompts"] += 1
            if mode == "warm" and ptok >= 1000 and s.turn >= 1:           # a follow-up turn should hit the prefix cache
                self.cnt["warm_followups"] += 1
                if not cached:
                    self.cnt["warm_no_hit"] += 1
            if findings and any(x["kind"] == "BUDGET" for x in findings):
                self.cnt["budget"] += 1
            for x in findings:
                if x["severity"] != "info":
                    self.fail[mode][x["kind"]] += 1
                    self.fail[mode]["_any"] += 1
                self.events.append({"t": rec["t"], "kind": x["kind"], "severity": x["severity"], "session": s.sid, "family": s.family,
                                    "turn": s.turn, "mode": mode, "tag": tag, "detail": x["detail"], "prompt_tokens": ptok,
                                    "session_decode_tokens": s.decode_tokens})

    def say(self, msg: str, critical: bool = False) -> None:
        """Console line for a finding; after the first 40 only criticals and every 25th are printed (details stay in report.json)."""
        with self.lock:
            self.printed += 1
            if critical or self.printed <= 40 or self.printed % 25 == 0:
                print(msg, flush=True)

    def save_flag(self, name, payload):
        if self.flag_dir and self.a.save_prompts != "none":
            try:
                with open(os.path.join(self.flag_dir, name + ".json"), "w") as fh:
                    json.dump(payload, fh)
            except OSError:
                pass

    def flag_findings(self, s, tag, findings, body, resp):
        bad = [f for f in findings if f["severity"] in ("critical", "high")]
        if not bad:
            return
        name = "%s_%s_t%d_%s_%s" % (time.strftime("%H%M%S"), s.sid, s.turn, tag, bad[0]["kind"])
        payload = {"findings": findings, "response": {k: v for k, v in resp.items() if k != "_latency"}}
        if self.a.save_prompts in ("flagged", "all"):
            payload["request"] = body
        self.save_flag(name, payload)
        self.say("[%s] %s %s turn %d %s %s: %s" % (iso(), s.sid, tag, s.turn, bad[0]["severity"].upper(), bad[0]["kind"],
                                                    json.dumps(bad[0]["detail"])[:220]), critical=bad[0]["severity"] == "critical")

    def compare(self, s, label, resp_a, resp_b, ptok, body):
        w, c = tokens_of(resp_a), tokens_of(resp_b)
        if not w or not c:
            with self.lock:
                self.cnt["compare_no_logprobs"] += 1
            return
        r = compare_tokens(w, c, self.a.tie_gap)
        v = r["verdict"]
        with self.lock:
            self.cmp[(label, v)] += 1
            self.cmp[(label, "_all")] += 1
        suspect = v == "HARD_DIVERGENCE" or (r["mean_abs_dlp"] > self.a.max_mean_dlp and v != "IDENTICAL")
        if suspect:
            kind = "COLD_WARM_SUSPECT" if label == "cold" else "NONDET_" + label.upper().replace("-", "_")
            sev = "high" if label in ("cold", "warm-vs-warm2") else "medium"
            with self.lock:
                self.events.append({"t": round(now() - self.t0, 1), "kind": kind, "severity": sev, "session": s.sid, "turn": s.turn,
                                    "mode": label, "detail": r, "prompt_tokens": ptok, "session_decode_tokens": s.decode_tokens})
                self.fail["warm"][kind] += 1
            if body is not None:
                self.save_flag("%s_%s_t%d_%s" % (time.strftime("%H%M%S"), s.sid, s.turn, label),
                               {"compare": r, "request": body if self.a.save_prompts in ("flagged", "all") else None,
                                "a": resp_a.get("choices"), "b": resp_b.get("choices")})
            self.say("[%s] %s turn %d %s %s: %s" % (iso(), s.sid, s.turn, label, v, json.dumps(r)))

    # ---- one turn (+ fork + replays)
    def turn(self, s: Session) -> bool:
        a = self.a
        kind, qtext, exp, temp = s.question()
        msgs = s.build_messages(qtext)
        small = s.est_tokens() <= a.replay_max_ctx
        replay = small and a.replay_rate > 0 and self.rng.random() < a.replay_rate
        warm2 = small and a.warm2_rate > 0 and self.rng.random() < a.warm2_rate
        want_lp = temp == 0.0 and (replay or warm2)
        body = self.body(msgs, temp, a.max_tokens, logprobs=want_lp)
        prompt_ascii = all(ord(ch) < 128 for ch in qtext)
        fork = None
        if a.fork_rate > 0 and self.rng.random() < a.fork_rate:
            fkind, fq, fexp, ftemp = s.question()
            fork = (fkind, fexp, ftemp, self.body(s.build_messages(fq), ftemp, a.max_tokens))
        fork_res: dict = {}
        th = None
        if fork:
            th = threading.Thread(target=lambda: fork_res.__setitem__("resp", self.send(fork[3], "fork")), daemon=True)
            th.start()
        resp = self.send(body, "main")
        if th:
            th.join()
        if resp is None:
            return False
        text, reasoning, fin, ptok, ctok, cached = self.parts(resp)
        self.cpt.update(sum(len(m["content"]) for m in msgs), ptok)
        if cached is None:
            self.have_cached = False
        findings = analyze(text, reasoning, fin, kind, exp, s, self.reg, prompt_ascii)
        self.record(s, "main", kind, temp, resp, findings, "warm")
        self.flag_findings(s, "main", findings, body, resp)
        if fork and fork_res.get("resp") is not None:
            fr = fork_res["resp"]
            ft, frs, ff, _, _, _ = self.parts(fr)
            ffind = analyze(ft, frs, ff, fork[0], fork[1], s, self.reg, prompt_ascii)
            self.record(s, "fork", fork[0], fork[2], fr, ffind, "warm")
            self.flag_findings(s, "fork", ffind, fork[3], fr)
        if replay:
            cbody = self.body(msgs, temp, a.max_tokens, logprobs=want_lp, salt="canary-%016x" % self.rng.getrandbits(64))
            cresp = self.send(cbody, "cold")
            if cresp is not None:
                ct, crs, cf, _, _, ccached = self.parts(cresp)
                cfind = analyze(ct, crs, cf, kind, exp, s, self.reg, prompt_ascii)
                if ccached:
                    cfind.append({"kind": "COLD_NOT_COLD", "severity": "info", "detail": {"cached_tokens": ccached}})
                self.record(s, "cold", kind, temp, cresp, cfind, "cold")
                self.flag_findings(s, "cold", cfind, cbody, cresp)
                if want_lp and not ccached:
                    self.compare(s, "cold", resp, cresp, ptok, body)
                    if self.rng.random() < a.cold2_rate:
                        c2 = self.send(self.body(msgs, temp, a.max_tokens, logprobs=True, salt="canary-%016x" % self.rng.getrandbits(64)), "cold2")
                        if c2 is not None:
                            self.compare(s, "cold-vs-cold2", cresp, c2, ptok, None)
        if warm2 and want_lp:
            w2 = self.send(body, "warm2")
            if w2 is not None:
                self.compare(s, "warm-vs-warm2", resp, w2, ptok, body)
        s.commit(qtext, text, ptok, ctok)
        return True

    # ---- session loop (one thread each)
    def worker(self, idx: int):
        rng = random.Random(self.a.seed * 1000 + idx)
        time.sleep(rng.uniform(0, self.a.stagger))
        fams = list(self.shared["family"].keys())
        while not self.stop.is_set() and now() < self.deadline:
            with self.lock:
                self.sid_counter += 1
                k = self.sid_counter
            s = Session("S%03d" % k, fams[(k - 1) % len(fams)], random.Random(rng.getrandbits(32)), self.reg, self.a, self.shared, self.cpt)
            while not self.stop.is_set() and now() < self.deadline:
                if self.a.max_turns and self.cnt["turns"] >= self.a.max_turns:
                    self.stop.set()
                    break
                ok = self.turn(s)
                with self.lock:
                    self.cnt["turns"] += 1
                if (not ok and self.stop.is_set()) or s.done():
                    break
                time.sleep(rng.uniform(0, self.a.think))
            with self.lock:
                self.session_summaries.append({"session": s.sid, "family": s.family, "turns": s.turn, "est_ctx_tokens": s.est_tokens(),
                                               "last_prompt_tokens": s.prompt_tokens_last, "decode_tokens": s.decode_tokens})

    def progress(self):
        while not self.stop.is_set() and now() < self.deadline + 5:
            time.sleep(self.a.progress)
            with self.lock:
                print("[%s] t=%4ds requests=%d turns=%d events=%d (critical %d) http_errors=%d tokens prompt/cached/completion=%d/%d/%d" % (
                    iso(), now() - self.t0, self.cnt["requests"], self.cnt["turns"], sum(1 for e in self.events if e["severity"] != "info"),
                    sum(1 for e in self.events if e["severity"] == "critical"), self.cnt["http_errors"], self.cnt["prompt_tokens"],
                    self.cnt["cached_tokens"], self.cnt["completion_tokens"]), flush=True)

    def go(self):
        threads = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(self.a.sessions)]
        for t in threads:
            t.start()
        threading.Thread(target=self.progress, daemon=True).start()
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(0.3)
                if now() > self.deadline + self.a.grace:
                    print("grace period over; abandoning in-flight requests", flush=True)
                    break
        except KeyboardInterrupt:
            print("interrupted; writing the partial report", flush=True)
        self.stop.set()
        time.sleep(0.2)


def preflight(run: Run) -> None:
    """Two identical requests and one salted: checks that usage reports cached_tokens and that cache_salt is accepted."""
    a = run.a
    rng = random.Random(a.seed + 99)
    lex = [run.reg.new_word("preflight") for _ in range(5)]
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": filler(rng, lex, int(1800 * run.cpt.get())) + "\nQuestion: reply with the single word OK."}]
    res = []
    for salt in (None, None, "canary-preflight-%016x" % rng.getrandbits(64)):
        r = run.send(run.body(msgs, 0.0, 16, salt=salt), "preflight")
        res.append(None if r is None else run.parts(r)[5])
    print("preflight: cached_tokens first/second/salted = %s" % res, flush=True)
    if run.cnt["http_errors"] >= 3:
        print("  WARNING: all preflight requests failed (%s); continuing, the run will stop after %d consecutive errors" % (
            run.events[-1]["detail"][:120] if run.events else "?", a.max_consecutive_errors))
    elif res[1] is None and res[0] is None:
        print("  WARNING: usage.prompt_tokens_details.cached_tokens missing (server needs --enable-prompt-tokens-details): hit accounting off")
        run.have_cached = False
    elif (res[1] or 0) <= (res[0] or 0):
        print("  WARNING: the repeated request did not hit the prefix cache; the run will not exercise it (caching off or evicted?)")
    if res[2] is None and res[0] is not None:
        print("  WARNING: a request carrying cache_salt failed: cold replays disabled")
        a.replay_rate = a.cold2_rate = 0.0
    elif (res[2] or 0) > 0:
        print("  WARNING: cache_salt did not isolate the prefix cache (cached_tokens>0 on a salted request); cold replays are not cold")
    run.cnt["http_errors"] = 0
    run.events[:] = [e for e in run.events if e.get("tag") != "preflight"]


# --------------------------------------------------------------------------------------------------------------------------------
# metrics scrape, report, verdict
# --------------------------------------------------------------------------------------------------------------------------------

def scrape(tr) -> dict:
    try:
        raw = tr.get("/metrics", raw=True, timeout=10)
    except Exception:       # noqa: BLE001
        return {}
    out: dict = collections.defaultdict(float)
    for line in raw.splitlines():
        m = re.match(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)$", line.strip())
        if m and re.search(r"prefix_cache|spec_decode|preempt|nan|num_requests_(running|waiting)|corrupt", m.group(1)):
            try:
                out[m.group(1)] += float(m.group(3))
            except ValueError:
                pass
    return dict(out)


def verdict(run: Run):
    a = run.a
    lines: list = []
    evs = [e for e in run.events]
    crit = [e for e in evs if e["severity"] == "critical"]
    sev = collections.Counter(e["severity"] for e in evs)
    kinds = collections.Counter(e["kind"] for e in evs if e["severity"] != "info")
    n_req = run.cnt["requests"]
    n_warm, n_cold = run.n["warm"]["_all"], run.n["cold"]["_all"]
    stats: dict = {"requests": n_req, "warm": n_warm, "cold": n_cold, "event_kinds": dict(kinds)}
    tests = {}
    task_of = {"RECALL_FAIL": "recall", "ECHO_FAIL": "echo", "SUMMARY_FAIL": "summary", "COUNT_FAIL": "counting"}
    for k in ("RECALL_FAIL", "ECHO_FAIL", "SUMMARY_FAIL", "COUNT_FAIL", "DEGENERATE", "EMPTY", "GARBAGE"):
        wf, cf = run.fail["warm"][k], run.fail["cold"][k]
        wn, cn = (run.n["warm"][task_of[k]], run.n["cold"][task_of[k]]) if k in task_of else (n_warm, n_cold)
        if wn and cn:
            tests[k] = {"warm": [wf, wn], "cold": [cf, cn], "p_warm_worse": round(fisher_greater(wf, wn, cf, cn), 5)}
    stats["warm_vs_cold"] = tests
    lines.append("== canary summary (%s) ==" % iso())
    lines.append("requests %d (warm incl. forks %d, cold replays %d)  turns %d  sessions %d  http errors %d" % (
        n_req, n_warm, n_cold, run.cnt["turns"], len(run.session_summaries), run.cnt["http_errors"]))
    ptok, ctok, cached = run.cnt["prompt_tokens"], run.cnt["completion_tokens"], run.cnt["cached_tokens"]
    lines.append("tokens: prompt %d (cached %d = %.1f %%), completion %d; follow-up turns (>=1k tokens) without any cache hit: %d of %d; BUDGET (empty, length): %d" % (
        ptok, cached, 100.0 * cached / max(1, ptok), ctok, run.cnt["warm_no_hit"], run.cnt["warm_followups"], run.cnt["budget"]))
    if run.session_summaries:
        lines.append("sessions finished %d; longest prompt %d tokens; most decode tokens in one session %d" % (
            len(run.session_summaries), max(x["last_prompt_tokens"] for x in run.session_summaries),
            max(x["decode_tokens"] for x in run.session_summaries)))
    lines.append("events: critical %d, high %d, medium %d, low %d, info %d" % (sev["critical"], sev["high"], sev["medium"], sev["low"], sev["info"]))
    for k, v in sorted(kinds.items()):
        lines.append("   %-22s %d" % (k, v))
    for lab in ("cold", "cold-vs-cold2", "warm-vs-warm2"):
        tot = run.cmp[(lab, "_all")]
        if tot:
            lines.append("greedy %-14s n=%d identical=%d length-only=%d tie-flips=%d hard-divergences=%d" % (
                lab, tot, run.cmp[(lab, "IDENTICAL")], run.cmp[(lab, "LENGTH_ONLY")], run.cmp[(lab, "TIE_FLIP")], run.cmp[(lab, "HARD_DIVERGENCE")]))
    for k, t in tests.items():
        if t["warm"][0] or t["cold"][0]:
            lines.append("  %-13s warm %d/%d  cold %d/%d  P(warm worse)=%.4f" % (k, t["warm"][0], t["warm"][1], t["cold"][0], t["cold"][1], t["p_warm_worse"]))
    bad = sum(1 for e in evs if e["severity"] in ("critical", "high"))
    stats["upper95_critical"] = upper95(len(crit), max(1, n_req))
    stats["upper95_critical_or_high"] = upper95(bad, max(1, n_req))
    lines.append("leak events %d in %d requests -> 95 %% upper bound %.2f %% per request; critical+high %d -> %.2f %%" % (
        len(crit), n_req, 100 * stats["upper95_critical"], bad, 100 * stats["upper95_critical_or_high"]))
    lines.append("power: certifying < 1.0 %% needs >= %d requests, < 0.5 %% needs >= %d (this run: %d)" % (need_n(0.01), need_n(0.005), n_req))

    rc, reasons = 0, []
    if run.cnt["aborted_on_errors"]:
        rc = 3
        reasons.append("aborted after %d consecutive HTTP errors" % a.max_consecutive_errors)
    if n_req == 0:
        rc = 3
        reasons.append("no successful request")
    if crit and rc != 3:
        rc = 1
        reasons.append("%d LEAK events (another session's canary appeared in a response)" % len(crit))
    deg = kinds["DEGENERATE"] + kinds["EMPTY"] + kinds["GARBAGE"]
    if deg > a.tolerate_degenerate and rc != 3:
        rc = 1
        reasons.append("%d degenerate / empty / garbage outputs" % deg)
    if kinds["COLD_WARM_SUSPECT"] > a.tolerate_divergence and rc != 3:
        rc = 1
        reasons.append("%d warm-vs-cold hard divergences" % kinds["COLD_WARM_SUSPECT"])
    for k, t in tests.items():
        if t["p_warm_worse"] < 0.01 and t["warm"][0] > t["cold"][0] and k not in ("DEGENERATE", "EMPTY", "GARBAGE") and rc != 3:
            rc = 1
            reasons.append("%s fails more often on warm than on cold requests (P=%.4f)" % (k, t["p_warm_worse"]))
    if rc == 0:
        anomalies = sorted(k for k in kinds)
        if anomalies:
            rc = 2
            reasons.append("non-critical findings: " + ", ".join(anomalies))
        elif n_req < need_n(0.01):
            rc = 2
            reasons.append("no events, but only %d requests: underpowered (< %d for a 1 %% bound)" % (n_req, need_n(0.01)))
        elif run.cnt["budget"] > 0.3 * max(1, n_req):
            rc = 2
            reasons.append("%.0f %% of the requests ended as BUDGET (reasoning used max_tokens): tasks were mostly not evaluated" % (100.0 * run.cnt["budget"] / n_req))
        elif run.have_cached and run.cnt["warm_followups"] >= 20 and run.cnt["warm_no_hit"] > 0.5 * run.cnt["warm_followups"]:
            rc = 2
            reasons.append("most warm requests had no prefix-cache hit: the cache was not exercised")
    name = ("PASS", "FAIL", "WARN", "ERROR")[rc]
    lines.append("VERDICT: %s%s" % (name, ("  -- " + "; ".join(reasons)) if reasons else ""))
    stats["verdict"], stats["reasons"] = name, reasons
    return rc, lines, stats


def write_report(run, rc, lines, stats, before, after, args, started):
    deltas = {k: round(after.get(k, 0.0) - before.get(k, 0.0), 3) for k in set(after) | set(before) if after.get(k, 0.0) != before.get(k, 0.0)}
    if deltas:
        lines.insert(len(lines) - 1, "server metric deltas: " + ", ".join("%s=%+g" % (k, v) for k, v in sorted(deltas.items())[:12]))
    rep = {"tool": "canary-leak-test", "version": VERSION, "started": iso(started), "finished": iso(), "seconds": round(now() - started, 1),
           "args": {k: v for k, v in vars(args).items() if k not in ("base_url", "ca_cert", "out_dir", "api_key_env")}, "stats": stats, "metrics_deltas": deltas, "sessions": run.session_summaries,
           "events": run.events, "chars_per_token": round(run.cpt.get(), 3)}
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
            json.dump(rep, fh, indent=1)
        with open(os.path.join(args.out_dir, "requests.jsonl"), "w") as fh:
            for r in run.records:
                fh.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    return rep


# --------------------------------------------------------------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------------------------------------------------------------

def dry_run(args) -> int:
    cpt = Cpt(args.chars_per_token)
    reg = Registry(random.Random(args.seed + 1))
    shared = make_shared(random.Random(args.seed + 2), reg, args, cpt)
    s = Session("S001", sorted(shared["family"])[0], random.Random(args.seed + 3), reg, args, shared, cpt)
    kind, q, exp, temp = s.question()
    msgs = s.build_messages(q)
    est = sum(len(m["content"]) for m in msgs) / cpt.get()
    new_tok = (args.grow_min + args.grow_max) / 2
    turn_s = new_tok / 2000.0 + args.max_tokens * 0.6 / 18.0 + 1.0 + args.think / 2     # prefill of new tokens + decode + overhead
    turns = args.sessions * args.time_budget / turn_s
    reqs = turns * (1 + args.fork_rate + args.replay_rate * (1 + args.cold2_rate) + args.warm2_rate)
    print("DRY RUN (no network).  canary-leak-test %s" % VERSION)
    print("endpoint       : %s   model=%s   seed=%d" % (args.base_url, args.model or "(first id of /v1/models)", args.seed))
    print("load           : %d sessions in flight, %d client families, budget %ds, max_tokens %d, T=0 %.0f %% / T=1 %.0f %%" % (
        args.sessions, args.families, args.time_budget, args.max_tokens, 100 * (1 - args.t1_frac), 100 * args.t1_frac))
    print("contexts       : %d .. %d tokens (new tool output %d..%d tokens/turn), shared preamble ~%d tokens (global + per family, jittered)" % (
        args.ctx_min, args.ctx_max, args.grow_min, args.grow_max, args.preamble_tokens))
    print("replays        : cold %.0f %% of turns (<= %d tokens), 2nd cold %.0f %%, warm2 %.0f %%, concurrent forks %.0f %%" % (
        100 * args.replay_rate, args.replay_max_ctx, 100 * args.cold2_rate, 100 * args.warm2_rate, 100 * args.fork_rate))
    print("expected volume: ~%d turns, ~%d requests (model: new tokens prefilled at 2k tok/s, decode 18 tok/s per session)" % (turns, reqs))
    print("power          : zero events in n requests -> 95 %% upper bound ~3/n = %.2f %% for n=%d;  n>=%d for <1 %%, n>=%d for <0.5 %%" % (
        100 * 3 / max(1, reqs), reqs, need_n(0.01), need_n(0.005)))
    print("checks         : LEAK (marker / code / invented word of another session or family), PHANTOM, DEGENERATE / GARBAGE / EMPTY,")
    print("                 RECALL / ECHO / SUMMARY / COUNT failures, cold-vs-warm greedy comparison (logprobs; tie flip vs hard divergence),")
    print("                 cold-vs-cold2 noise floor, warm-vs-warm2 determinism, warm-vs-cold failure rates (Fisher)")
    print("sample request : kind=%s temp=%.1f  ~%d prompt tokens; system message %d chars, last user message %d chars" % (
        kind, temp, est, len(msgs[0]["content"]), len(msgs[-1]["content"])))
    print("expectation    : %s" % json.dumps(exp))
    print("--- system message (head) ---\n" + msgs[0]["content"][:420].rstrip() + "\n[...]")
    print("--- last user message (head / tail) ---\n" + msgs[-1]["content"][:260].rstrip() + "\n[...]\n" + msgs[-1]["content"][-250:].rstrip())
    print("out dir        : %s" % (args.out_dir or "(none: pass --out-dir to keep report.json / requests.jsonl / flagged/)"))
    return 0


# --------------------------------------------------------------------------------------------------------------------------------
# main / selftest
# --------------------------------------------------------------------------------------------------------------------------------

def parse_faults(spec: str) -> dict:
    out = {}
    for part in filter(None, (spec or "").split(",")):
        k, _, v = part.partition("=")
        out[k.strip()] = float(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("base_url", nargs="?", default=None, help="server root without /v1, e.g. http://127.0.0.1:8000 or https://host (default: $BASE_URL, else http://127.0.0.1:8000)")
    p.add_argument("--model", default=None, help="served model name (default: first id from GET /v1/models)")
    p.add_argument("--sessions", type=int, default=8, help="concurrent sessions (1..8 = max_num_seqs); default 8")
    p.add_argument("--max-inflight", type=int, default=8, help="global cap on requests in flight incl. forks and replays (= max_num_seqs); default 8")
    p.add_argument("--time-budget", type=int, default=1800, help="seconds after which no new turn starts (default 1800)")
    p.add_argument("--grace", type=int, default=900, help="extra seconds to wait for in-flight requests after the budget (default 900)")
    p.add_argument("--max-turns", type=int, default=0, help="stop after this many turns in total (0 = no limit)")
    p.add_argument("--ctx-min", type=int, default=8000)
    p.add_argument("--ctx-max", type=int, default=100000)
    p.add_argument("--grow-min", type=int, default=1500, help="min tokens of new tool output per turn")
    p.add_argument("--grow-max", type=int, default=6000, help="max tokens of new tool output per turn")
    p.add_argument("--preamble-tokens", type=int, default=3000, help="shared (cross-session) preamble length in tokens")
    p.add_argument("--families", type=int, default=3, help="client families sharing a preamble")
    p.add_argument("--t1-frac", type=float, default=0.5, help="fraction of (non-counting) turns sampled at temperature 1 (default 0.5)")
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--replay-rate", type=float, default=0.15, help="fraction of turns replayed cold with a fresh cache_salt")
    p.add_argument("--replay-max-ctx", type=int, default=40000, help="only replay turns whose context is <= this many tokens (cold prefill cost)")
    p.add_argument("--cold2-rate", type=float, default=0.3, help="fraction of cold replays that get a second cold replay (noise floor)")
    p.add_argument("--warm2-rate", type=float, default=0.05, help="fraction of greedy replay candidates repeated warm (warm determinism)")
    p.add_argument("--fork-rate", type=float, default=0.1, help="fraction of turns that fire a concurrent fork (same history, other question)")
    p.add_argument("--tie-gap", type=float, default=0.35, help="logprob gap (nats) below which a greedy divergence counts as a near-tie flip")
    p.add_argument("--max-mean-dlp", type=float, default=0.25, help="mean |dlogprob| over the identical prefix above which warm/cold is suspect")
    p.add_argument("--tolerate-degenerate", type=int, default=0)
    p.add_argument("--tolerate-divergence", type=int, default=0)
    p.add_argument("--think", type=float, default=2.0, help="max client think time between turns (s)")
    p.add_argument("--stagger", type=float, default=15.0, help="sessions start at random offsets within this many seconds")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--chars-per-token", type=float, default=3.5, help="initial estimate; learned from usage.prompt_tokens")
    p.add_argument("--extra-body", default=None, help='JSON merged into every request, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    p.add_argument("--api-key-env", default="CANARY_API_KEY", help="environment variable holding a bearer token (optional)")
    p.add_argument("--ca-cert", default=None, help="CA bundle for https")
    p.add_argument("--insecure", action="store_true", help="do not verify the https certificate")
    p.add_argument("--timeout", type=int, default=1800, help="per-request timeout (s)")
    p.add_argument("--max-consecutive-errors", type=int, default=10)
    p.add_argument("--no-preflight", action="store_true")
    p.add_argument("--out-dir", default=None, help="write report.json, requests.jsonl, summary.txt and flagged/ here")
    p.add_argument("--save-prompts", choices=("flagged", "none", "all"), default="flagged")
    p.add_argument("--progress", type=int, default=30, help="progress line every N seconds")
    p.add_argument("--dry-run", action="store_true", help="print the plan and a sample prompt; no network")
    p.add_argument("--mock", action="store_true", help="use the in-process mock model instead of HTTP (detector self-test; no network)")
    p.add_argument("--mock-faults", default="", help="e.g. leak=0.03,degenerate=0.03,empty=0.01,diverge=0.05,wrong=0.01 (mock only)")
    p.add_argument("--selftest", action="store_true", help="mock runs: a clean run must stay clean, faulted runs must be caught")
    return p


def run_main(args) -> int:
    started = now()
    if args.mock:
        tr = MockTransport(parse_faults(args.mock_faults), args.seed)
    else:
        if args.base_url.startswith("mock://"):
            print("base_url required (or use --mock / --selftest / --dry-run)", file=sys.stderr)
            return 3
        tr = HttpTransport(args.base_url, os.environ.get(args.api_key_env) or None, args.ca_cert, args.insecure, args.timeout)
    if not args.model:
        try:
            args.model = tr.get("/v1/models")["data"][0]["id"]
        except Exception as e:      # noqa: BLE001
            print("cannot determine the model name (GET /v1/models failed: %s); pass --model" % e, file=sys.stderr)
            return 3
    run = Run(args, tr)
    before = scrape(tr)
    print("canary-leak-test %s start %s model=%s sessions=%d budget=%ds seed=%d out=%s" % (
        VERSION, iso(), args.model, args.sessions, args.time_budget, args.seed, args.out_dir), flush=True)
    if not args.no_preflight:
        preflight(run)
    try:
        signal.signal(signal.SIGTERM, lambda *_: run.stop.set())
    except ValueError:
        pass
    run.go()
    after = scrape(tr)
    rc, lines, stats = verdict(run)
    write_report(run, rc, lines, stats, before, after, args, started)
    print("\n".join(lines))
    return rc


def selftest(args) -> int:
    base = argparse.Namespace(**vars(args))
    base.mock, base.dry_run, base.selftest = True, False, False
    base.time_budget, base.grace, base.sessions = 8, 30, 6
    base.ctx_min, base.ctx_max, base.grow_min, base.grow_max = 1500, 5000, 300, 900
    base.preamble_tokens, base.max_tokens, base.stagger, base.think, base.progress = 400, 256, 0.5, 0.02, 5
    base.replay_rate, base.cold2_rate, base.warm2_rate, base.fork_rate, base.replay_max_ctx = 0.5, 0.5, 0.3, 0.3, 10 ** 6
    base.out_dir, base.model, base.max_turns, base.extra_body = None, "mock-model", 0, None
    ok = True

    def one(name, faults, expect):
        nonlocal ok
        a = argparse.Namespace(**vars(base))
        a.mock_faults, a.seed = faults, (args.seed or 1) + len(name)
        print("\n##### selftest case: %s (%s)" % (name, faults or "no faults"), flush=True)
        run = Run(a, MockTransport(parse_faults(faults), a.seed))
        run.go()
        rc, lines, _ = verdict(run)
        print("\n".join(lines))
        kinds = collections.Counter(e["kind"] for e in run.events if e["severity"] != "info")
        good = bool(expect(rc, kinds, run))
        print("selftest %s: %s" % (name, "ok" if good else "FAILED (kinds=%s rc=%d requests=%d)" % (dict(kinds), rc, run.cnt["requests"])), flush=True)
        ok = ok and good

    one("clean", "", lambda rc, k, r: rc in (0, 2) and not k and r.cnt["requests"] > 60 and r.cmp[("cold", "IDENTICAL")] > 0)
    one("leak", "leak=0.08", lambda rc, k, r: rc == 1 and k["LEAK_MARKER"] > 0)
    one("degenerate", "degenerate=0.06,empty=0.04", lambda rc, k, r: rc == 1 and k["DEGENERATE"] + k["GARBAGE"] > 0 and k["EMPTY"] > 0)
    one("diverge", "diverge=0.25", lambda rc, k, r: rc == 1 and k["COLD_WARM_SUSPECT"] > 0)
    one("wrong", "wrong=0.25", lambda rc, k, r: k["RECALL_FAIL"] > 0)
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.base_url is None:
        args.base_url = os.environ.get("BASE_URL") or "http://127.0.0.1:8000"
    if args.seed is None:
        args.seed = int(time.time()) % 1000003
    args.sessions = max(1, min(args.sessions, 64))
    args.extra_body = json.loads(args.extra_body) if args.extra_body else None
    if args.selftest:
        return selftest(args)
    if args.dry_run:
        return dry_run(args)
    return run_main(args)


if __name__ == "__main__":
    sys.exit(main())
