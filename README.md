# DeepSeek-V4.1-Flash-1017 on eight DGX Sparks

Serve **DeepSeek-V4.1-Flash-1017** in its native FP8 / MXFP4 precision with **vLLM** across **eight NVIDIA DGX Spark**
nodes: tensor parallel 8 over a 100 GbE RoCE fabric, DSpark speculative decoding, and the model's full **1,048,576-token**
context window. The weights are the official checkpoint plus a small, published refusal-removal overlay (26 tensors); the
engine is a pinned vLLM nightly image plus one build-time fix and 29 documented runtime patches for the GB10 (sm_121).

[Results](#results-at-a-glance) · [What it is](#what-it-is) · [Requirements](#requirements) · [Quick start](#quick-start) ·
[Configuration](#configuration) · [Operations](#operations) · [How it works](#how-it-works) · [Patches](#patches) ·
[Known limitations](#known-limitations) · [Layout](#repository-layout)

## Results at a glance

Build **b072**, measured on 28 October 2026 on eight DGX Sparks (tensor parallel 8, 100 GbE, GPU clocks capped at 2,200 MHz).
Speeds are generated tokens per second, measured with the tools in [bench/](bench/).

| | |
|---|---|
| **One user, coding agent** | **~140 tok/s** while generating |
| **Many users at once** (total) | 16 users **392** · 64 users **716** · 128 users **~900** tok/s |
| **Coding agents in parallel** (total) | 4 agents **201** · 16 agents **330** tok/s, first token after 0.5 s / 1.1 s |
| **Reading a prompt** (prefill) | **~6,700-6,900 tok/s** for prompts of 8k-128k tokens |
| **Long prompts**, wait for the first token | 128k tokens **19 s** · 512k **81 s** · 1M **172 s** |
| **Context window** | **1,048,576 tokens** per request |
| **KV cache** | **20.4 M tokens** in total, about 19 requests at the full 1M context at once |
| **Quality** | Same as the official model: GSM8K **49/50**, HumanEval+ **19/20**, tool calls **19/24**, long-context recall **12/12** (official: 49, 18, 19, 12) |

## What it is

| | |
|---|---|
| Model | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at revision `1bae0e10` (MIT; 475 GiB: FP8 attention, MXFP4 experts, two Engram n-gram tables, DSpark draft blocks) **plus** the refusal-direction overlay [`drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated`](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated) at revision `de412d22` (MIT, gated): 26 attention output projections (layers 10-35) replaced, merged locally by `tools/build_overlay_variant.py` |
| Hardware | 8 x DGX Spark (GB10, 128 GB unified memory each), one tensor-parallel rank per node, 100 GbE RoCE through one switch |
| Engine | vLLM nightly `e4a5b55` (0.32.1rc1.dev466, torch 2.14, FlashInfer 0.7.2.post1, NCCL 2.30.7), official image by digest, plus one build-time GB10 edit and 29 runtime patches that are applied when each container starts |
| Decoding | DSpark speculative decoding (5 drafted tokens per step, from the checkpoint), masked verification, FP8 KV cache in DeepSeek's own format, prefix caching, CUDA graphs for decode (up to 128 requests) |
| Memory plan | experts stored at their native 288-column width (b12x), Engram layers 1 and 14 read from each node's NVMe instead of memory, a CUDA-graph pool of 0.7 GiB (patch 043), a pinned 43.5 GiB KV pool per node |
| License | Apache-2.0 for this repository; the model (MIT), the overlay (MIT, gated) and the benchmark datasets have their own terms ([NOTICE](NOTICE)) |

**What the recipe changes.** Released vLLM (v0.32.0) does not start DeepSeek-V4.1 on GB10, and a stock nightly runs it at a
fraction of this speed. The recipe changes vLLM in three ways, each documented in [PATCHES.md](PATCHES.md) with its reason,
its effect on numerics and its measured effect: one build-time edit (64-token kernel blocks), 29 runtime patches
(faster collectives over RoCE, cheaper drafting, fewer expert reads per step, a sequence-parallel and overlapped prefill,
Engram rows from NVMe, native-width experts), and a long list of settings in [config/recipe.env](config/recipe.env).

## Requirements

- **Eight DGX Spark nodes** running DGX OS, with Docker and the NVIDIA container toolkit (both ship with DGX OS).
- **One 100 GbE link per node** to one switch, with jumbo frames (MTU 9000) end to end and RoCE working.
- **Local NVMe on every node, about 750 GB free while you prepare the weights** (the official checkpoint is 475 GiB; the merged
  checkpoint adds 179 GiB of patched shards, the rest are hardlinks; plus about 49 GB of packed Engram rows), about 570 GB
  afterwards, plus about 40 GB for the image and a few GB of JIT caches. The Engram rows are read from this disk at run time:
  use the node's own NVMe, not a network share.
- **Nothing else on the nodes.** The recipe leaves 8-12 GiB of memory free per node under load (see
  [Operations](#operations)); another workload on a node breaks that.
- **Key-based ssh** from the first node (NODE1) to the other seven.
- **A Hugging Face account** that has accepted the terms of the overlay repository (it is gated, free).
- **Recommended: the GPU clock capped at 2200 MHz on every node** (`sudo nvidia-smi -lgc 0,2200`; lost on reboot). At
  TP=8 the hottest node sets the pace, and every number here was taken at this cap.

## Quick start

Run everything on NODE1, from a clone of this repository at the same path on every node (or copy `patches/` to that path
on the other nodes, step 5).

**1. Describe your cluster.**

```bash
cp cluster.env.example cluster.env
$EDITOR cluster.env        # node names, fabric addresses, interface and RoCE device, model path, packed-Engram path
```

**2. Build the served weights and put them on every node,** at the path `MODEL_DIR` names:

```bash
pipx install huggingface_hub        # provides the `hf` command; then `hf auth login` (the overlay is gated)
. ./cluster.env
BASE=/models/DeepSeek-V4.1-Flash-1017          # any path on the NVMe, not MODEL_DIR itself
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir "$BASE"
hf download drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated \
    --revision de412d2249632903dace23399526e00a145aada5 \
    --include wo_b_l10_35_1017.safetensors ABLIT_META.json --local-dir "$BASE-keys-overlay"
python3 tools/build_overlay_variant.py --base "$BASE" \
    --overlay "$BASE-keys-overlay/wo_b_l10_35_1017.safetensors" \
    --out "$MODEL_DIR" --extra "$BASE-keys-overlay/ABLIT_META.json"
# ends with "verify: 0 problem(s); 26 shards patched, 52 tensors replaced"
for n in "$NODE2" "$NODE3" "$NODE4" "$NODE5" "$NODE6" "$NODE7" "$NODE8"; do
  rsync -a --mkpath "$MODEL_DIR/" "$n:$MODEL_DIR/"
done
```

The overlay file must have sha256 `88324ceaa2c0754a2bdb4ef6bb388a088438e352358343e6c2c882009cdde864`. The builder never
writes to the official checkpoint: unchanged files are hardlinked, the 26 shards that hold a replaced tensor are copied,
and every byte is verified afterwards.

**3. Build the image and copy it to all nodes** (about 38 GB):

```bash
image/build.sh --distribute
```

**4. Pack each node's slice of the Engram tables.** Layers 1 and 14 of the Engram tables (23.6 GiB per node) are not kept in
memory; each node reads its own rank's rows from its NVMe. `tools/engram_pack.py` writes them as one 512-byte record per row
(about 22.9 GiB per layer and node, about 35 s per layer on the NVMe we used). Run it once per node, after the weights are in
place at `MODEL_DIR`, with the rank of that node (NODE1 = 0 ... NODE8 = 7). It needs only Python 3, so it can be streamed
over ssh:

```bash
. ./cluster.env
python3 tools/engram_pack.py pack   --model-dir "$MODEL_DIR" --out-dir "$ENGRAM_PACKED_DIR" --rank 0      # NODE1 itself
python3 tools/engram_pack.py verify --model-dir "$MODEL_DIR" --out-dir "$ENGRAM_PACKED_DIR" --rank 0
rank=1
for n in "$NODE2" "$NODE3" "$NODE4" "$NODE5" "$NODE6" "$NODE7" "$NODE8"; do
  ssh "$n" python3 - pack   --model-dir "$MODEL_DIR" --out-dir "$ENGRAM_PACKED_DIR" --rank "$rank" < tools/engram_pack.py
  ssh "$n" python3 - verify --model-dir "$MODEL_DIR" --out-dir "$ENGRAM_PACKED_DIR" --rank "$rank" < tools/engram_pack.py
  rank=$((rank + 1))
done
```

Each pack records the size, modification time and header hash of the checkpoint shard it came from, and the engine refuses to
start with a pack that no longer matches: repack (`pack --force`) after you change a shard. `launch/up.sh` checks that every
node has its own rank's packs before it starts anything.

**5. Make the patch directory identical on every node** (the launcher checks this and refuses otherwise):

```bash
for n in "$NODE2" "$NODE3" "$NODE4" "$NODE5" "$NODE6" "$NODE7" "$NODE8"; do
  rsync -a --delete --mkpath patches/ "$n:$PWD/patches/"
done
```

**6. Start it.** The copy in step 2 leaves the weights in every node's page cache, and on GB10's shared memory CUDA does
not count that cache as free memory, so release it before every start: set `DROP_CACHES=1` in `cluster.env` (it needs
the one-line sudoers rule described there), or run once by hand:

```bash
for n in "$NODE1" "$NODE2" "$NODE3" "$NODE4" "$NODE5" "$NODE6" "$NODE7" "$NODE8"; do
  ssh -t "$n" "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
done
```

A boot took 6.0 minutes (358 s) in our 28 October run. The first boot of a node also fills its kernel caches (TileLang,
Triton, FlashInfer, b12x) and takes longer.

```bash
launch/up.sh --dry-run     # read-only preflight on all eight nodes, prints what it would start
launch/up.sh --wait        # start the eight ranks and wait until the API answers
```

**7. Warm it up, then use it.** About ten kernels compile at the first real requests after every boot, which makes the first
users wait and briefly takes about 2 GiB of free memory; the two-half prefill overlap (patch 028) also runs its self-test at the
first prompt of 2,048 tokens or more. Send a few requests yourself first, including one prompt of about 8,000 tokens.
The API is OpenAI-compatible, on NODE1 at `API_PORT`
(default 8000):

```bash
curl -s http://NODE1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dsv41-flash-1017-uncen",
  "messages": [{"role": "user", "content": "Explain tensor parallelism in two sentences."}]
}'
```

The model thinks first: its reasoning arrives in a separate `reasoning` field, then the answer in `content`. It can think
for thousands of tokens, so give it a generous `max_tokens` (DeepSeek recommends 256K or more; without one, vLLM lets the
answer run to the end of the context window), choose a lower effort with
`"chat_template_kwargs": {"reasoning_effort": "low"}`, or switch thinking off with
`"chat_template_kwargs": {"enable_thinking": false}` for a short reply. The API has no key and speaks plain HTTP: put access
control and TLS in front of it before anyone but you can reach it.

Stop it with `launch/down.sh` (it refuses while requests are running; `--force` after a crash; it also removes the head node's
stale shared-memory files). Check it with `launch/status.sh`.

## Is it working?

- `launch/up.sh --wait` only returns success when rank 0's log shows every expected line: all 29 patches applied, the ones that
  switch on at run time reporting ACTIVE, the Engram rows coming from the packed files, the KV cache size, the API on its
  port. The rules are in `launch/readiness.txt`.
- `launch/status.sh` shows every rank container, the model and its 1,048,576-token limit, and each GPU's power and
  temperature.
- After your first prompt of 2,048 tokens or more, rank 0's log shows `[patch-028] prefill two-half overlap ACTIVE ... self-test
  OK on all 8 ranks`.
- `bench/run-bench.sh --quick` runs a short version of the published benchmarks, after you have prepared the benchmark
  data ([bench/README.md](bench/README.md)).

## Configuration

`cluster.env` says where the recipe runs; `config/recipe.env` says what is served. The settings you are most likely to
change:

| Setting | Where | Value | Note |
|---|---|---|---|
| Context length | `--max-model-len` | 1048576 | Lowering it saves a few GiB of workspace per node; it barely changes speed |
| Concurrent requests | `--max-num-seqs` | 128 | The decode CUDA graphs are captured up to 768 tokens (128 requests x 6). 256 did not fit next to a 40 GiB pool |
| Prefill chunk | `--max-num-batched-tokens`, `--long-prefill-token-threshold` | 8192, 4096 | 8,192-token chunks; a long prompt gets 4,096 tokens per step while other requests are active |
| KV cache per node | `--kv-cache-memory-bytes` | 46707769344 (43.5 GiB) | 20,411,335 tokens. It leaves the head node 8.08 GiB free in the worst-case order, 0.08 GiB above the rule: this is the limit, not a starting point. It needs patches 043 and 045 (below); without them use 40 GiB (42949672960; 8.9 GiB, b070). 41 GiB with patch 015's replicated embedding took the head node to 7.2 GiB (b069) |
| Memory patches | `VLLM_GB10_EMBED_TP`, `VLLM_GB10_INDEXER_DECODE_CHUNK_MB`, `VLLM_GB10_SHARED_CAPTURE_STREAM` in `M_DOCKER_EXTRA` | 1, 512, 1 | Patches 045 and 043 give 1.07 GiB and 3.1 GiB per node back; without them keep the KV pool at 40 GiB |
| Engram tables | `--engram-config`, `ENGRAM_PACKED_DIR` in `cluster.env` | layers 1 and 14 on NVMe, 1 % verified | Needs the packs of step 4 on every node |
| Speculative decoding | `--speculative-config` | DSpark, 5 tokens, draft top-k 2048 | The draft blocks are part of the checkpoint |
| Prefill overlap switch | file `dsv41-p028.ctl` in `HF_CACHE_DIR` on NODE1 | absent = on | Write `off` into it to fall back to the plain sequence-parallel prefill without a restart |
| Page cache | `DROP_CACHES` in `cluster.env` | 1 recommended | Needed after the weights were copied or read |
| Headroom warning | `MEM_SERVING_WARN_GIB` in `cluster.env` | 8 | The launcher warns after boot when a node has less memory available |
| Shared-memory cleanup | `CLEAN_SHM` in `cluster.env` | 1 | `down.sh` removes the head node's stale `/dev/shm/psm_*` files |

Most of the 29 patches are switched by an environment variable in `M_DOCKER_EXTRA` of `config/recipe.env` (or by a vLLM
argument: 013, 016); a patch whose switch is absent is inert. Patches 001, 014, 015, 019 and 034 have no switch and apply once
installed ([PATCHES.md](PATCHES.md) names the switch of each). Set the same variables on all eight ranks: the launcher gives
the file to every rank unchanged. Patch 015 (replicated token embedding) stays installed, but patch 045
switches it off in this configuration; to switch it back on, remove `VLLM_GB10_EMBED_TP` from `M_DOCKER_EXTRA` and lower the
KV pool by 1 GiB.

## Operations

- **Keep at least 8 GiB of MemAvailable on every node, always.** CPU and GPU share one memory pool on GB10, so a node that
  runs out is killed, not slowed down. The shipped configuration is sized for it: in the worst-case order (a long prompt right
  after a busy period; the 1M-token prompt right after 128 users and 16 agents) the head node's lowest point was 8.08 GiB, which
  is 0.08 GiB above the rule.
  We run a small guard on every node that kills the engine when MemAvailable stays below 6 GiB for 2 s, so that a node never
  freezes; run something like it. Do not give the engine more KV cache, more sequences or a profiler without re-running the
  worst-case order and watching the lowest free memory on every node.
- **Warm up after every boot** (step 7): the first requests compile kernels and the first long prompt runs patch 028's
  self-test; both cost latency and a few GiB of free memory for a short time.
- **Restart with `launch/down.sh`**, not with `docker rm`. It refuses while requests are running, removes the workers first and
  rank 0 last, waits until every GPU is idle, and deletes the head node's stale `/dev/shm/psm_*` files with a throwaway
  container of the recipe image (vLLM does not unlink its message rings when a container is removed; 136 of them, 243 MB of
  RAM, had piled up on our head node after two days of restarts). A GPU that stays busy with no container is an orphaned
  context: reboot that node.
- **Watch the logs**: patch 032 logs the CUDA allocator's state once a minute; `touch`
  `$HF_CACHE_DIR/dsv41-memsnap.trigger` on a node for a full allocator snapshot (patch 042).
- **Do not expose the API** beyond a network you control (no API key, plain HTTP).

## How it works

- **Eight small GPUs on one switch.** Each DGX Spark is a 48-SM GB10 with 128 GB of memory shared with the CPU; the model is
  cut into eight, so every layer ends in network all-reduces. A decode step is limited by the memory bandwidth of each node
  (every weight byte it needs is read once per step) and by the latency of about 86 all-reduces over 100 GbE. The recipe's
  decode work shortens both: experts at native width, a fused b12x MoE kernel, one-shot and two-shot RoCE all-reduces
  (the two-shot reduces on the Grace CPU) that sum in a fixed rank order, a cheaper drafter, and masked verification that
  does not read experts for drafts that are about to be rejected.
- **Speculative decoding.** DSpark drafts 5 tokens per step; the model verifies all 6 positions at once, exactly (block
  verification keeps the output distribution the model's own). A step yields 4.0-4.25 tokens (of at most 6) on coding-agent turns.
- **Prefill is link-bound.** A prefill chunk runs sequence-parallel (mHC on one eighth of the rows per rank), with the chunk
  cut into two halves whose network traffic hides behind the other half's compute, MXFP8 W8A8 GEMMs, and the Engram layers
  read from NVMe. That took prefill from 3.7k to 6.7-6.9k tokens per second.
- **Memory.** Weights take 38.7 GiB per node, the KV pool 43.5 GiB, and the rest is the CUDA graphs (0.7 GiB private pool), the allocator's cache
  and the working set of a prefill. The Engram tables (23.6 GiB per node) and the expert padding that b12x used to keep
  (11.5 GiB of zeros per node) are gone from memory, the decode graphs' private pool shrank from 3.8 to 0.7 GiB (patch 043) and the
  token embedding is sharded again (patch 045, 1.07 GiB); that is where the larger KV pool comes from.
- **Exact or gated.** Every patch is either bit-exact, lossless for the output distribution (drafting changes), or it changes
  the numerics slightly (a floating-point reduction order or kernel; MXFP8 activations in prefill-size GEMMs) and passed the
  quality gate (KL divergence from the official model at most 0.0185). [PATCHES.md](PATCHES.md) says which is which.

## Patches

Everything is in [PATCHES.md](PATCHES.md): for each patch the problem, the change, the switch, the numerics, the measured
effect and the boot log line that shows it is active. "Exact" means bit-identical outputs or an unchanged output
distribution; "order" means a different but valid floating-point summation order or kernel and "numerics" a changed activation
precision, both admitted by the quality gate. Effects are single-patch or same-boot figures from our build log, with the
build they come from.

| ID | What it does | Measured effect | Exact? |
|---|---|---|---|
| B1 | build time: sparse-attention and indexer backends offer 64-token kernel blocks on SM12x (vLLM PR #60762) | enabler: the engine does not start without it | yes |
| 001 | inter-process wait on the head node sleeps after 2 ms instead of spinning 1 s | idle reader CPU 1.001 s → 0.004 s per 1.6 s | yes |
| 012 | one-shot all-reduce over NCCL for a decode step of up to 6 tokens | 130 → 90 µs per collective; step 58.8 → 54.1 ms (b004) | order |
| 013 | DSpark draft top-k for DeepSeek-V4.1 (k = 2048) | enabler for 024, 029, 038; k = 64 had cut acceptance by about 9 % (b011) | yes |
| 014 | attention `wo_a` as one plain MXFP8 GEMM (through Marlin here) instead of an emulated BF16 batched matmul | with 015: step 40.1 → 39.1 ms (b013) | order |
| 015 | replicated token embedding: no all-reduce at the start of each graph | with 014, see above; installed, but switched off by 045 in this configuration | yes |
| 016 | Engram layers read from node-local NVMe (O_DIRECT, fail-closed) | 11.8 GiB per layer and node freed: layer 14 doubled the KV pool (b019); layer 1 as well took it 7.5 M → 13.1 M tokens (b048) | yes |
| 018 | RoCE one-shot / two-shot all-reduce and all-gather for small messages | step 38.1 → 36.1 ms (b020), → 33.6 ms with the two-shot (b021) | order |
| 019 | candidate-mask tiles past each row's end are skipped | 147 → 12 µs per call; step 33.9 → 33.3 ms (b024) | yes |
| 020 | DSpark drafter: MXFP8 draft head, column-parallel `main_proj` | step 33.3 → 32.5 ms (b025) | yes (drafts only) |
| 021 | MXFP8 W8A8 GEMMs for prefill-sized batches (Marlin W8A16 below 512 rows) | prefill +5 % (b025) | numerics, quality-gated |
| 022 | sequence-parallel mHC in prefill, layers 0-19 | prefill +13 % (b029) | order, quality-gated |
| 023 | router GEMM through the CuTe dot-product kernel for up to 8 rows | 16.2 vs 25.8 µs per call at 6 rows | order |
| 024 | the request's top-p applied to the draft distribution | acceptance +5 % (b026/b027) | yes (drafts only) |
| 025 | split configs of the fused mHC kernel for 6 and 12 tokens | op −15 % / −8 %; −0.13 ms per step (b034) | yes (bitwise) |
| 026 | masked verification: only drafts that are likely to be accepted read experts (chain policy) | +3.2 % tokens per second at equal acceptance (b038) | yes |
| 028 | two-half communication/compute overlap for the sequence-parallel prefill | prefill +21 % (5.4-5.7k → 6.6-6.9k tok/s); 1M prompt 200 → 172 s (b064) | order, self-tested at run time |
| 029 | DSpark top-k Markov bias in one Triton launch | 64-user step 316 → 303 ms (b049) | yes (drafts only) |
| 030 | packed Engram rows: one 512-byte read per row | gave back the 6-8 % of prefill that layer 1 on NVMe had cost: KV pool 7.5 M → 12.2 M tokens at the prefill speed of b042 (b054) | yes |
| 031 | scheduler knobs from a control file | inert in this configuration | yes |
| 032 | allocator statistics every 60 s | diagnostics only | yes |
| 033 | zero-padding of the expert slice 288 → 320 | inert here (replaced by 036); b056: step 31.5 → 29.1 ms | yes |
| 034 | b12x compact micro launchers attached to vLLM's scratch plans | enabler for 033/036 | yes |
| 036 | b12x compact path at the native 288-column width | step 29.1 → 28.5 ms, experts 38.3 → 34.5 GiB, KV +11 % (b061) | yes (bitwise to 033) |
| 037 | 64-row tiles for prefill-size expert launches | +0.5 % prefill under the overlap (b065) | yes |
| 038 | the drafter gathers each rank's top-2048 candidates, not the vocabulary | 64 / 128 users +1.7 / +3.7 % (b066) | yes (bitwise) |
| 040 | patch 026's confidence chain only in the graphs that use it | 64 / 128 users +3.5 / +3.9 % (b067) | yes |
| 042 | allocator snapshot on demand | diagnostics only | yes |
| 043 | the decode sparse indexer's rows run in chunks of at most 512 MiB of temporaries inside the CUDA graphs; the drafter's graphs share the main graphs' capture stream | graph pool 3.8 → 0.68 GiB per rank, which allowed 41 → 43.5 GiB of KV pool (b071 → b072); about 1 % slower at 64 users (against b071) | yes (logits bit-identical; top-k set equal) |
| 045 | switches patch 015's replicated embedding off (`VLLM_GB10_EMBED_TP=1`): vocab-parallel again | 1.07 GiB per node back (38.69 vs 39.76 GiB), which allowed 40 → 41 GiB of KV pool (b070 → b071); prefill about −1 % | yes |

## Known limitations

- **It thinks long.** In our quality runs, which capped each answer at `max_tokens` 8,192, 6 of 169 answers ran out of tokens
  while still thinking (`finish_reason: "length"`, no final answer). The cap is per request, not a server limit: any value is
  accepted, and one larger than what is left of the 1,048,576-token window is cut to it. A higher cap reserves no memory and
  does not slow decoding; only an answer that really runs that long takes the time (32k tokens at the one-user speed is about
  4 minutes). DeepSeek recommends `max_tokens` of 256K or more; lower the effort or switch thinking off for short answers.
- **Temperature 0 is not bit-reproducible.** Two identical prompts can produce different text at a near-tie token: vLLM is
  not batch-invariant by default and the b12x MoE kernel is not bit-deterministic from run to run. The quality gate's KL
  divergence varied between 0.0170 and 0.0183 for the same served code.
- **Text only.** The vision encoder is switched off (`--language-model-only`).
- **Long prompts take minutes.** A 1M-token prompt needs about 172 s before the first token; while other requests are active
  a long prompt gets 4,096 tokens per step instead of 8,192.
- **Memory is tight by design.** 8.08 GiB free on the head node in the worst case, 8-12 GiB per node under load. More KV cache, 256 sequences, a profiler or another
  workload on the nodes can break the 8 GiB rule (and the engine with it).
- **The Engram layers 1 and 14 come from NVMe.** The engine reads about one 512-byte record per looked-up row; any read error,
  timeout or changed shard ends the process (fail closed).
- **A network outage of 6 s or more breaks a running engine** even after the link recovers: restart it.
- **Every boot needs the same image ID and patch directory on all eight nodes,** and the patches pin the files of the image
  they were written against: a different vLLM image makes them refuse to start.
- **First requests after a boot are slow** while about ten kernels compile.
- **Quality was gated at 4 concurrent requests.** Patch 021 switches the dense GEMMs to MXFP8 W8A8 for batches of 512 rows or
  more, which a decode step reaches from about 86 running requests (5 drafts each); the quality runs do not cover that regime
  (PATCHES.md, patch 021).
- **Quality is gated, not identical.** Some patches change a floating-point reduction order (table above); the KL divergence
  from the official model is 0.0180 including the overlay's own contribution (0.0141 on the bring-up engine).

## Repository layout

```
cluster.env.example        your nodes, addresses and paths (copy to cluster.env)
config/recipe.env          the served configuration: image, vLLM arguments, patch switches
recipe.lock.yaml           every version and pin behind the published numbers
launch/                    up.sh, down.sh, status.sh and the readiness rules
image/                     Dockerfile, the build-time GB10 edit and build.sh (IMAGE.md)
patches/                   the 29 runtime patches, applied at container start
tools/build_overlay_variant.py  merges the overlay into the official checkpoint (hardlinks, verified)
tools/engram_pack.py       writes each node's packed Engram rows
tools/check-public.sh      scan for private addresses, keys and paths before you publish a fork
bench/                     benchmark tools and dataset preparation (bench/README.md)
tests/                     offline test of the launcher (stubbed ssh, docker and nvidia-smi), CPU checks of patches 001 and 012
PATCHES.md                 every change to vLLM, why, and its measured effect
CREDITS.md, NOTICE, LICENSE
```

## Credits and license

This recipe builds on DeepSeek's model and its DSpark draft blocks, the drowzeys Keys overlay, vLLM, FlashInfer (including its
b12x kernels and RoCE communication runtime) and the DGX Spark community; see [CREDITS.md](CREDITS.md). The code in this
repository is Apache-2.0 ([LICENSE](LICENSE)); third-party terms are in [NOTICE](NOTICE).

## Responsible use

The served weights have had their refusal behaviour removed, so the model will answer requests the original declines.
You are responsible for how it is used: follow the law and the model licenses (the overlay's terms also exclude any use
the DeepSeek license prohibits), and add your own safeguards (access control, usage policies, logging) before you let
anyone else use the endpoint. The recipe starts the API without an API key (vLLM's `--api-key` can add one); do not
expose it beyond a network you control.
