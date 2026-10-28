# Benchmark data

The benchmarks in [`../tools`](../tools) replay recorded agent turns and slice long documents into prompts. Those
inputs are built from public Hugging Face datasets by the scripts in this directory. **This repository contains no
data files**: you download the datasets from Hugging Face yourself and build the files locally. The two folders the
scripts write to (`prepared/` and `cache/`) are git-ignored.

| What | Script | Source dataset (pinned revision) | Output | Used by |
|---|---|---|---|---|
| W2, recorded coding-agent sessions | `prep_w2_agent.py` | [`nebius/SWE-rebench-openhands-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) `35455389ab51bf5e2306bfd436ef72d0f98bf882` | `w2_agent-v1.jsonl` (64 sessions) | `decode-bench-w2.py`, `decode-bench-w2-tc.py`; `ctx-curve.py` for prompts beyond about 0.9 M tokens |
| W3, long documents and repositories | `prep_w3_docqa.py` | [`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2) `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9` and [`princeton-nlp/SWE-bench_Lite_bm25_27K`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite_bm25_27K) `19519c4a85faf070faaa7055026de59c6c3454d1` | `w3_docqa-v1.jsonl` (28 items) | `ctx-curve.py`, `drift-probe.py` |
| W1, real chat prompts (optional) | `prep_w1_chat.py` | [`allenai/WildChat-4.8M`](https://huggingface.co/datasets/allenai/WildChat-4.8M) `c827c6df8fcf008219ffaffa4d1dd77491099367` | `w1_chat-v1.jsonl` (600 conversations), `w1_system_prompt.txt` | nothing in `bench/tools` |

The revisions are exact commit hashes (`REVISIONS` in `_common.py`), so everybody builds from the same bytes. The
selection is deterministic (seed 20260925), so the same tokenizer gives the same files.

## Licenses

This package redistributes none of the datasets, so it does not restate their licenses. For all four: see the dataset
card (linked above) for the license and any access terms. Some Hugging Face datasets are gated: if a download is
refused, accept the terms on the dataset page while logged in and run `huggingface-cli login` (or set `HF_TOKEN`).

## WildChat

WildChat contains **real user prompts**. `prep_w1_chat.py` uses only the *user turns*: assistant text from the dataset
is read for its timing fields only and is never written out (the model under test generates its own answers),
conversations flagged toxic or redacted are dropped, and near-duplicates are removed. The prepared file is for local
use. It is **not redistributed here** and must not be committed or published; `.gitignore` in this directory keeps it
out of git. None of the tools in `bench/tools` reads it, so you can skip W1 and its roughly 2 GB download entirely
(`fetch_sources.py` leaves it out by default).

## Build the files

```bash
pip install huggingface_hub transformers jinja2 pyarrow numpy      # plus whatever your checkpoint's tokenizer needs

python3 bench/data/fetch_sources.py                # downloads the W2 and W3 sources into bench/data/cache
python3 bench/data/prep_w2_agent.py                # -> bench/data/prepared/w2_agent-v1.jsonl
python3 bench/data/prep_w3_docqa.py                # -> bench/data/prepared/w3_docqa-v1.jsonl

# only if you want W1 as well:
python3 bench/data/fetch_sources.py --only w1
python3 bench/data/prep_w1_chat.py                 # -> bench/data/prepared/w1_chat-v1.jsonl
```

Every script takes the same three location options (a flag beats the environment variable, which beats the default):

| Option | Environment variable | Default |
|---|---|---|
| `--data-dir DIR` prepared files and `MANIFEST.json` | `BENCH_DATA_DIR` | `bench/data/prepared` |
| `--cache-dir DIR` Hugging Face download cache | `BENCH_HF_CACHE` | `bench/data/cache` |
| `--tokenizer DIR` tokenizer files and chat template | `BENCH_TOKENIZER_DIR` | `MODEL_DIR` from `cluster.env` (or `$CLUSTER_ENV`); for this model pass a directory with a chat template, see below |

The benchmark tools read the prepared files from `--data-dir` / `BENCH_DATA_DIR` as well, so if you move them, set the
variable once. `fetch_sources.py --only` takes `w1`, `w2`, `w3` (default `w2,w3`); the download is about 3.5 GB with
all three, and re-running it only fetches what is missing.

### Use the tokenizer of the checkpoint you serve

The scripts measure lengths with the checkpoint's own tokenizer and chat template (`reasoning_effort=high`), and those
lengths decide which sessions and documents are kept (for example sessions are cut at the last turn of at most 49,152
tokens). Point `--tokenizer` at a directory that holds the checkpoint's `tokenizer.json`, `tokenizer_config.json` and a
chat template (the next section says where to get one for this model). A different tokenizer builds different files, and
the numbers are then not comparable. The tokenizer is loaded with `trust_remote_code=True`, which runs Python code from
that directory: only point it at a directory you trust.

### This checkpoint has no chat template

The scripts render prompts with `tokenizer.apply_chat_template`. The DeepSeek-V4.1-Flash-1017 repository contains no Jinja
template: DeepSeek ships a Python encoder (`encoding/encoding.py`) instead, and vLLM renders prompts with its own copy of
it (`--tokenizer-mode deepseek_v41`). With `MODEL_DIR` as the tokenizer directory the scripts therefore stop with the
transformers error "chat_template is not set".

The older `deepseek-ai/DeepSeek-V4.1-Flash` repository does ship a `chat_template.jinja` (a community contribution), and its
`tokenizer.json` and `tokenizer_config.json` are byte-identical to the 1017 ones. Put these three files in a directory of
their own and pass it as `--tokenizer`:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash tokenizer.json tokenizer_config.json chat_template.jinja \
    --local-dir bench/data/cache/tok-dsv41
python3 bench/data/prep_w2_agent.py --tokenizer bench/data/cache/tok-dsv41
python3 bench/data/prep_w3_docqa.py --tokenizer bench/data/cache/tok-dsv41
```

That template predates DeepSeek's encoder fix of 16 October 2026, so the token counts of long tool-heavy sessions can differ
slightly from what the server renders. This moves the nominal sizes of the prepared prompts and nothing else: `ctx-curve.py`
and the other tools report the server's own token counts, except the `ctx` column of `decode-bench-w2.py`, which is the
count recorded here.

TODO: the scripts were not re-run with this template for this release. Build the files once and compare a few `ctx` values
with the server's `prompt_tokens` before you publish numbers.

## What comes out

`MANIFEST.json` in the data directory records, per file, the source dataset and revision, the filters, counts, token
statistics, the sha256 and size of the output, the seed, the preparation time, and the tokenizer (folder name and sha256
of its files; no paths). Keep it next to your results; if you publish numbers, publish it too, so a reader can check
that they built the same inputs. The JSONL files themselves are not meant to be published.

| File | One line is | Fields the tools read |
|---|---|---|
| `w2_agent-v1.jsonl` | one agent session cut at the last turn whose prompt is at most 49,152 tokens | `id`, `messages`, `tools`, `assistant_idx`, `asst_tokens`, `prefix_tokens` |
| `w3_docqa-v1.jsonl` | one document (12 LongBench, 12 SWE-bench Lite, 4 long ones) | `id`, `prompt_tokens`, `prefix`, `doc`, `suffix` |

`w1_system_prompt.txt` holds the fixed chat system prompt (`_common.SYSTEM_PROMPT`) that W1 and W3 use when they count
tokens. `prep_w3_docqa.py` writes it itself, so W3 does not need the WildChat download.
