"""Shared helpers for the benchmark dataset preparation scripts (fetch_sources.py, prep_w*.py).

Every prep script:
  * downloads its source from Hugging Face at a *pinned revision* into the cache directory,
  * filters and orders the items deterministically (seed SEED),
  * measures lengths with the tokenizer + chat template of the model under test,
  * writes compact JSONL to the data directory and records a manifest entry (dataset id, revision,
    filters, counts, token stats, sha256) in MANIFEST.json there.

Where things go (a flag beats the environment variable, which beats the default):
  --data-dir   BENCH_DATA_DIR       prepared JSONL files and MANIFEST.json      bench/data/prepared
  --cache-dir  BENCH_HF_CACHE       Hugging Face download cache                 bench/data/cache
  --tokenizer  BENCH_TOKENIZER_DIR  tokenizer files + chat template of the      MODEL_DIR from cluster.env
                                    checkpoint you serve (any directory that has them)

Python packages: huggingface_hub, transformers (with jinja2), pyarrow, numpy, plus whatever the checkpoint's own
tokenizer needs. The two default folders are git-ignored; nothing downloaded or prepared is ever committed.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")   # downloads need no usage reporting

SEED = 20260925
HERE = Path(__file__).resolve().parent            # <repo>/bench/data
REPO_DIR = HERE.parent.parent                     # <repo>
DATA_DIR = Path(os.environ.get("BENCH_DATA_DIR") or HERE / "prepared")
HF_CACHE = Path(os.environ.get("BENCH_HF_CACHE") or HERE / "cache")
TOKENIZER_DIR: str | None = os.environ.get("BENCH_TOKENIZER_DIR") or None   # else MODEL_DIR from cluster.env

# Pinned upstream revisions (HF dataset commit shas, resolved 2026-09-25).
REVISIONS: dict[str, str] = {
    "allenai/WildChat-4.8M": "c827c6df8fcf008219ffaffa4d1dd77491099367",
    "nebius/SWE-rebench-openhands-trajectories": "35455389ab51bf5e2306bfd436ef72d0f98bf882",
    "zai-org/LongBench-v2": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
    "princeton-nlp/SWE-bench_Lite_bm25_27K": "19519c4a85faf070faaa7055026de59c6c3454d1",
}

# WildChat is 86 parquet shards (15 GB). The early shards (2023, GPT-3.5/4 era) hold most of the
# multi-turn conversations; later shards are ~99 % single-turn. A spread of early shards is plenty.
WILDCHAT_SHARDS = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17]


# --------------------------------------------------------------------------- locations
def add_location_args(ap) -> None:
    """--data-dir, --cache-dir and --tokenizer for a prep script."""
    g = ap.add_argument_group("locations")
    g.add_argument("--data-dir", default=None,
                   help=f"prepared JSONL files and MANIFEST.json (default: $BENCH_DATA_DIR, else {HERE / 'prepared'})")
    g.add_argument("--cache-dir", default=None,
                   help=f"Hugging Face download cache (default: $BENCH_HF_CACHE, else {HERE / 'cache'})")
    g.add_argument("--tokenizer", default=None,
                   help="directory with the tokenizer files and chat template of the checkpoint you serve "
                        "(default: $BENCH_TOKENIZER_DIR, else MODEL_DIR from cluster.env)")


def configure(args) -> None:
    """Apply the flags of add_location_args(); call once, right after parse_args()."""
    global DATA_DIR, HF_CACHE, TOKENIZER_DIR
    if getattr(args, "data_dir", None):
        DATA_DIR = Path(args.data_dir).expanduser()
    if getattr(args, "cache_dir", None):
        HF_CACHE = Path(args.cache_dir).expanduser()
    if getattr(args, "tokenizer", None):
        TOKENIZER_DIR = args.tokenizer


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=value lines of a cluster.env (plain assignments only: no expansion, no commands)."""
    vals: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = raw.strip()
        if ln.startswith("export "):
            ln = ln[7:].lstrip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, _, v = ln.partition("=")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k.strip()):
            continue
        v = v.strip()
        if v[:1] in ("'", '"'):
            end = v.find(v[0], 1)
            v = v[1:end] if end > 0 else v[1:]
        else:
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
        vals[k.strip()] = v
    return vals


def tokenizer_dir() -> str:
    """The tokenizer directory: --tokenizer / BENCH_TOKENIZER_DIR, else MODEL_DIR from cluster.env."""
    if TOKENIZER_DIR:
        return TOKENIZER_DIR
    env_file = Path(os.environ.get("CLUSTER_ENV") or REPO_DIR / "cluster.env")
    model_dir = ""
    if env_file.is_file():
        model_dir = read_env_file(env_file).get("MODEL_DIR", "")
    if model_dir and "CHANGE_ME" not in model_dir and Path(model_dir).is_dir():
        return model_dir
    why = (f"MODEL_DIR in {env_file} is {model_dir!r}, which is not a directory on this machine" if model_dir
           else f"{env_file} has no MODEL_DIR" if env_file.is_file() else f"{env_file} does not exist")
    sys.exit("no tokenizer: pass --tokenizer DIR (the checkpoint directory, or any directory with its tokenizer files "
             f"and chat template) or set BENCH_TOKENIZER_DIR. ({why})")


# --------------------------------------------------------------------------- downloads
def hf_file(repo_id: str, filename: str) -> Path:
    """Download (or reuse from cache) one file of a dataset repo at its pinned revision."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset",
                                revision=REVISIONS[repo_id], cache_dir=str(HF_CACHE)))


# --------------------------------------------------------------------------- tokenizer
_TOK = None


def tokenizer():
    """The tokenizer of the model under test (loaded once)."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(tokenizer_dir(), trust_remote_code=True)
    return _TOK


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
                   "vocab.json", "merges.txt")


def tokenizer_info() -> dict:
    """Identifies the tokenizer + chat template behind the token counts (folder name and file hashes, no path)."""
    d = Path(tokenizer_dir())
    return {"dir_name": d.name, "files_sha256": {f: sha256_file(d / f) for f in TOKENIZER_FILES if (d / f).is_file()}}


def _template_messages(messages: Sequence[dict]) -> list[dict]:
    """Messages in the shape a Jinja chat template expects.

    The OpenAI wire format carries tool-call arguments as JSON *strings*; vLLM parses them
    into dicts before rendering (the template iterates ``arguments.items()``). We do the same
    so offline token counts match what the server renders.
    """
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args) if args.strip() else {}
                        fn["arguments"] = parsed if isinstance(parsed, dict) else {"value": parsed}
                    except json.JSONDecodeError:
                        fn["arguments"] = {"raw": args}
                tcs.append(tc)
            m["tool_calls"] = tcs
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def chat_tokens(messages: Sequence[dict], tools: Sequence[dict] | None = None,
                reasoning_effort: str | None = "high", add_generation_prompt: bool = True) -> int:
    """Prompt length in tokens of the model under test, as the server would render it (chat template applied)."""
    tok = tokenizer()
    kwargs: dict[str, Any] = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    ids = tok.apply_chat_template(_template_messages(messages), tools=list(tools) if tools else None,
                                  add_generation_prompt=add_generation_prompt, tokenize=True,
                                  return_dict=False, **kwargs)
    if isinstance(ids, dict):  # some transformers versions return BatchEncoding
        ids = ids["input_ids"]
    return len(ids)


def text_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


def encode(text: str) -> list[int]:
    return tokenizer().encode(text, add_special_tokens=False)


def decode(ids: Sequence[int]) -> str:
    return tokenizer().decode(list(ids))


# --------------------------------------------------------------------------- utilities
def rng(tag: str) -> random.Random:
    """Deterministic RNG per purpose, derived from the global seed."""
    h = hashlib.sha256(f"{SEED}:{tag}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def stable_hash(*parts: Any) -> int:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def token_stats(values: Sequence[int | float]) -> dict[str, float]:
    """min/p10/p50/p90/max/mean of a list of token counts (for the manifest and the report)."""
    if not values:
        return {"n": 0}
    v = sorted(values)

    def q(p: float) -> float:
        k = (len(v) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return float(v[lo] + (v[hi] - v[lo]) * (k - lo))

    return {"n": len(v), "min": float(v[0]), "p10": round(q(0.10), 1), "p50": round(q(0.50), 1),
            "p90": round(q(0.90), 1), "max": float(v[-1]), "mean": round(statistics.fmean(v), 1)}


def histogram(values: Sequence[int | float], edges: Sequence[int]) -> dict[str, int]:
    """Counts per [edge_i, edge_i+1) bucket, keyed 'lo-hi'."""
    out: dict[str, int] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        out[f"{lo}-{hi}"] = sum(1 for x in values if lo <= x < hi)
    return out


def update_manifest(name: str, entry: dict) -> None:
    """Merge one dataset's entry into MANIFEST.json in the data directory (atomic rewrite)."""
    entry = dict(entry)
    entry.setdefault("prepared_utc", _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    entry.setdefault("seed", SEED)
    entry.setdefault("tokenizer", tokenizer_info())
    path = DATA_DIR / "MANIFEST.json"
    try:
        cur = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        cur = {}
    cur.setdefault("_about", "Pinned benchmark datasets. The JSONL files are generated locally by bench/data/prep_*.py and "
                             "are not committed. File names, counts, token statistics and sha256 identify them; no paths "
                             "are recorded.")
    cur.setdefault("datasets", {})[name] = entry
    try:
        write_json(path, cur)
    except OSError as exc:
        print(f"[manifest] could not write {path}: {exc}")


def file_entry(path: Path) -> dict:
    return {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


# --------------------------------------------------------------------------- the chat system prompt
# A realistic, fixed chat-UI system prompt (~600 tokens). Shared by W1 and W3 so the
# benchmark sends the envelope a real chat front-end sends.
SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant running on a private research cluster. You talk with engineers, researchers and students who use you for programming, systems administration, data analysis, writing, translation and general questions.

How to answer:
- Answer the question that was asked. If the request is ambiguous, state the most reasonable interpretation, answer it, and mention briefly what you assumed. Ask a clarifying question only when an answer would be useless without it.
- Lead with the answer or the result, then give the reasoning or steps that support it. Keep simple answers short; give longer, structured answers for complex tasks.
- Use Markdown when it helps readability: headings for long answers, numbered steps for procedures, bullet lists for options, and tables for comparisons. Do not use Markdown decoration for one-line answers.
- Put all code, commands, configuration and file contents in fenced code blocks with a language tag. Code must be complete enough to run or paste; do not elide essential parts with comments like "rest of code here". Prefer the standard library and well-known packages. Explain non-obvious lines briefly after the block, not inside long comments.
- For shell commands, assume a Linux system (Ubuntu 24.04) unless the user says otherwise. Point out commands that are destructive, need root, or cannot be undone before showing them.
- For math and quantitative questions, show the key steps and check units and orders of magnitude. Give the final number clearly.
- When you are not sure, say so and explain what would settle the question. Never invent citations, APIs, command-line flags, package names or version numbers. If something depends on a version, say which version you mean.
- If the user writes in a language other than English, reply in that language unless asked otherwise. Keep technical identifiers, code and error messages unchanged.
- When the user shares an error message or a stack trace, identify the most likely cause first, then give a concrete fix and a way to verify it.
- When asked to review or improve text or code, preserve the author's intent and style; list the most important changes first.
- Respect privacy: do not ask for passwords, keys or personal data, and warn the user if they paste secrets.
- Stay neutral and factual on contested topics, and present the main positions fairly.
- Keep the conversation context in mind: refer back to earlier messages when relevant and do not repeat long passages you already gave unless asked.
- When comparing tools, libraries or approaches, name the trade-offs that matter for the user's situation (performance, maintenance, licensing, learning curve) and give a recommendation.
- For writing tasks, match the requested tone and audience; for translations, keep the meaning and register rather than translating word for word.

Formatting limits: keep answers under roughly 800 words unless the user asks for more detail or the task (for example a full program or document) needs it. Do not end with generic offers of further help; end when the answer is complete.
"""


def ensure_system_prompt_file() -> str:
    """Write the fixed chat system prompt to DATA_DIR/w1_system_prompt.txt (as prep_w1_chat.py always did) and return it."""
    p = DATA_DIR / "w1_system_prompt.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(SYSTEM_PROMPT, encoding="utf-8")
    return SYSTEM_PROMPT
