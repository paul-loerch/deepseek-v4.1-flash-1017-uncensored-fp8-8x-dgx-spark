# Patches

Everything this recipe changes in vLLM and in the FlashInfer `b12x` package inside the image, and why. For each change you get
the problem, what the change does, how to switch it, what it does to numerics, what we measured, and the boot line that shows it
is active.

The recipe serves **DeepSeek-V4.1-Flash-1017** (official checkpoint at revision `1bae0e10` plus the Keys overlay) with a
**vLLM nightly** (commit `e4a5b5537344`, 0.32.1rc1.dev466) at tensor parallel 8 across eight NVIDIA DGX Spark (GB10, sm_121)
nodes joined by 100G RoCE, one rank per node, with DSpark speculative decoding (5 draft tokens) and a 1,048,576-token context.
The base image is `vllm/vllm-openai@sha256:27aa6d814d1fc44ff8f13bb50c291cb23f54e15544ace8b699e0b29bdcc9451b` (CUDA 13.0, torch
2.14.0, FlashInfer 0.7.2.post1, NCCL 2.30.7). Pin it by digest: every file hash the patches check is the hash of a file in that
image.

The patch directory `patches/` holds 29 runtime patches (`001` to `045`, with gaps) and `entrypoint.sh`. It is the directory the
numbers in this repository were measured with (build b072); three comments in it were reworded for publication and no code was
changed (`files/prefill_w8a8.py`, `files/prefill_sp_overlap.py`, `files/p043_graphpool.py`; none of them is hash-pinned by another
patch).

Contents: [1 Reading guide](#1-reading-guide) · [2 Summary table](#2-summary-table) ·
[3 The changes](#3-the-changes-one-by-one) · [4 Considered and not shipped](#4-considered-and-not-shipped)

## 1. Reading guide

### 1.1 Kinds of change

| Kind | What | Where | Applies |
|---|---|---|---|
| Build-time bake, B1 | source edits for sm_121 in two vLLM files | `image/Dockerfile`, `image/dsv41_bake.py` | once, when the image is built |
| Runtime patches, 001 to 045 | 29 patch scripts that edit vLLM's Python files, install new modules into the `vllm` package, and (036, 037) edit the FlashInfer `b12x` package | `patches/NNN-*.sh`, `patches/files/`, `patches/entrypoint.sh` | at every container start, most behind their own environment variable |
| Configuration | vLLM flags and environment variables | `config/recipe.env` | at every launch |

Patches 001, 012 and 022 were developed first for our GLM-5.3-Flash recipe and were ported to this model and vLLM build. The
gaps in the numbering are patches that are not in this release (section 4).

### 1.2 The hardware and the model in a few lines

- **GB10 (sm_121).** One GPU per node with 48 SMs, about 121 GiB of unified memory shared with the CPU, a practical memory
  bandwidth of about 244 GB/s (273 GB/s peak), 101,376 B of shared memory per block, no GPUDirect RDMA.
- **Eight ranks over RoCE.** At TP=8 every layer ends in network all-reduces: 80 per decode step for the target model plus 6 in
  the drafter. At decode sizes they are latency-bound; prefill chunks move tens of MB per message at about 92 % of the 100 GbE
  line rate and are link-bound.
- **Host CPU.** Spinning waits keep performance cores at full clock, and the CPU-cluster sensor, not the GPU, is the hottest
  thermal zone.
- **The model.** DeepSeek-V4.1-Flash-1017: 40 layers, hidden size 5120, 64 query heads sharing one KV head, 384 routed experts
  (FP4) plus one shared expert (FP8) with 6 routed experts per token, sliding-window plus compressed sparse attention with an
  indexer, mHC hyper-connections, two Engram layers (n-gram embedding tables, 188.8 GiB), and a three-stage DSpark draft that
  proposes 5 tokens per step.
- **Terms.** TP is the tensor-parallel size. A rank is one TP process, one per node; rank 0 also serves the API. M = tokens in one
  decode-time forward pass, drafts included: one request with 5 drafts is M = 6. Acceptance length (AL) = mean tokens produced per
  speculative step. KL5 = the truncated top-5 KL divergence in nats per token from the official model on a fixed corpus.

### 1.3 How the runtime patches are applied, and why that is safe

`patches/entrypoint.sh` runs every top-level `patches/NNN-*.sh` in numeric order inside each rank's container, printing
`[recipe-entrypoint] applying <script>` before each, and then starts `vllm serve`. A script that exits non-zero makes it print
`[recipe-entrypoint] FATAL: <script> failed`, and the container stops. The patch directory is mounted read-only, containers are
created fresh at every start (the image is never modified), and the launcher refuses to start unless the directory is identical
on every node. Every installer is idempotent. The b12x patches (036, 037) change the FlashInfer package fingerprint, so b12x's
kernels are compiled again once after the first boot that installs them.

1. **Vendored, not fetched.** Nothing is downloaded at boot.
2. **Pinned.** Almost every edit goes through `patches/files/xpatch.py`: a patch module lists, per file, the sha256 of the
   expected original, the text to replace (which must occur exactly once) and the replacement. A file that is neither the pinned
   original nor already carries this patch's marker is refused, so a different image stops the boot instead of being patched
   blindly. Patches 001, 012, 022, 028, 030, 040 and 042 additionally pin the hashes of their own modules or of modules that earlier
   patches installed, and 022 pins a list of read-only vLLM files its data flow depends on.
3. **Anchored and atomic.** Edits are computed in memory, checked with Python's parser, written atomically and re-hashed.
   Re-running is a no-op (`[skip] ... already patched`).
4. **Gated.** Every behaviour change has an environment variable or a flag in the vLLM arguments (section 2 lists them). With the
   variable unset the hooks return at once. Patches 022 and 028 go further and touch nothing when their variable is unset. Set the
   same variables on every rank: a rank that lacks a gate does not join the collectives the others enter, and the group hangs.
   The launcher passes `M_DOCKER_EXTRA` of `config/recipe.env` to all eight containers, so this holds as long as you edit that one
   place.
5. **Fail closed at run time.** Patches that add communication, a native reader or a numerics change check themselves on the real
   model during the boot or at their first use, and stop the process when something is wrong instead of serving wrong values (026
   switches itself off instead, 043 falls back to the stock indexer path, and 028 in `report` mode, as shipped, logs a purely
   numeric failure and keeps serving):
   collective warm-up with an exact expected sum (012, 018), a self-check of the MXFP8 draft head on the real weights (020),
   boot self-tests on the first eligible forward (022, 026, 028), a bitwise check of the first live Engram layer (022), a shadow
   check of the first calls (038), manifest, header and sampled-row checks of the packed Engram rows (030), a watchdog in the
   NVMe reader that ends the process on any I/O error or timeout (016) and a one-time cross-check of the chunk schedule against
   the attention builder's (043).
6. **Checked before they served.** Patches 001 and 012 come with CPU checks (44 and 81 on the image's full vLLM tree) and, for 012,
   a GPU test; they are in [tests/patch-checks/](tests/patch-checks/) (`python3 -I -B tests/patch-checks/p001_cpu_check.py --vllm-root
   <vllm package dir>`, any CPU, no GPU). The other patches came with CPU and/or GPU test suites, some of which
   are named in patch comments (for example `tests/038`, `tests/028-overlap`, `tests/roce2`); those suites are not part of this
   release. The measurements in this file are the evidence we ship.

Boot lines: install lines quote the words ACTIVE and INACTIVE, so when you check a boot automatically, count only the runtime
`[patch-NNN] ... ACTIVE` lines that come after `all patches applied`. The ones that matter are listed in the patch sections below;
`launch/readiness.txt` requires them.

### 1.4 What "numerics" means here

Every patch is put in one of these classes:

| Class | Meaning |
|---|---|
| **Identical** | Bit-identical results to the stock path for the same inputs (or bit-identical where the result is read). |
| **Lossless for the output** | Changes something only the drafter sees (the draft tokens or their probabilities) or which drafts are verified. Speculative decoding with block verification keeps the output distribution exactly the target model's for any draft distribution, so only the speed and the acceptance length can change. |
| **Exact arithmetic, different order or kernel** | The same mathematical result, but a different summation order or tiling, so bits can differ in the last place (and near-tie tokens can flip). Admitted by the quality comparison below. |
| **Changes numerics** | A deliberately different computation (a different reduction order, activations quantized to MXFP8 in prefill). Each of these was admitted by a quality gate before it served: KL5 against the official model not above 0.0185, and task scores level within noise. |

The same served code measures between 0.0170 and 0.0183 KL5 from run to run (b067 0.0170 and b070 0.0183; b072, which adds only
the two memory patches 043 and 045, 0.0180), and even one boot does not reproduce
itself bit for bit (the MoE kernel is not bit-deterministic; two identical prefills agree on every top-1 token but their
log-probabilities differ by up to 0.004 to 0.010). A change is therefore judged by comparing next-token distributions over a fixed
corpus of code and prose: top-1 agreement, KL5, and the mean change in log-likelihood, plus GSM8K, HumanEval+, tool calls and
needle retrieval. The results are summarized in the README.

### 1.5 How to read the measurements

All measurements use eight nodes with the GPU clock capped at 2,200 MHz. Decode step times are the per-step times from the server's
speculative-decoding counters or from torch profiles of one user, so they do not depend on how many drafts were accepted;
"agent turns" are 16 recorded coding-agent turns decoded to 256 tokens, "users" are closed-loop chat users. Single runs of
agent turns swing by ±3 % with the sampled content, so effects of about 1 % to 3 % were only trusted from same-boot alternating
A/B runs.

`bNNN` is a build: one boot of one configuration, each normally the previous build plus the change named. The builds cited in
this file:

| Build | What it was |
|---|---|
| b001 | bring-up: stock nightly plus B1, Engram tables resident, DSpark k=5, 262,144-token context, KV pool 8 GiB |
| b004 | b001 + 001 + 012 + 16 sequences |
| b006 | b004 + 1,048,576-token context + 32 sequences, with the Keys weights (profiled as b010) |
| b011 / b012 | + b12x fused MoE kernel (`--moe-backend b12x`) + 013 with top-k 64 (b011), top-k 2048 (b012) |
| b013 / b015 | + 014 + 015 (b013); + `--linear-backend marlin` (b015) |
| b017 / b019 | + 64 sequences (b017); + 016 for Engram layer 14, KV pool 16 GiB (b019) |
| b020 / b021 / b022 | + 018 RoCE one-shot (b020); + two-shot up to 4 MiB (b021); two-shot limit 1 MiB (b022) |
| b024 / b025 | + 019 (b024); + 020 + 021 (b025) |
| b026 / b027 | + 024 + 023 (b026); 024 as one Triton launch (b027) |
| b029 / b030 / b032 | + 022 (b029); + 8,192-token prefill chunks (b030); + Engram all-to-all under sequence parallelism (b032) |
| b034 | + 025 |
| b037 / b038 / b040 / b042 | + 026 (b037 measurement build, b038 exact first-draft policy, b040 chain policy); b042 = production build with the chain policy |
| b048 / b049 | + 128 sequences + Engram layer 1 on NVMe + KV 28 GiB (b048); + 029 (b049) |
| b051 / b052 | + 031 + 032 (b051); + indexer row sharding from 2,048 rows (b052) |
| b054 | + 030 packed Engram rows + `--long-prefill-token-threshold 4096` + KV 26 GiB |
| b056 | + 033 (expert slice padded to 320) + 034, KV 32 GiB |
| b058 | b056 at KV 36 GiB (27 October) |
| b061 | 036 (experts at native width 288) instead of 033's padding, KV 40 GiB |
| b064 / b065 | + 028 revision 4 with the indexer split on its own communicator (b064); + 037 switched on (b065) |
| b066 / b067 | + 038 (b066); + 040 (b067) |
| b069 / b070 | b067 + 028 revision 5 + 042; KV 41 GiB (b069, rejected) and 40 GiB (b070) |
| b071 / b072 | b070 + 045 (patch 015 switched off) at KV 41 GiB (b071, not served); + 043 and KV 43.5 GiB (**b072, the shipped configuration**) |

### 1.6 If you change the image or a patch

Because the pins name exact files, a different vLLM image makes the affected patches refuse to start. That is intended.
Re-validate each affected patch against the new source (its anchors, then its boot self-test, then the quality comparison) before
you update its hashes. A new patch takes the next free number, is inert unless its own variable is set, pins its targets,
requires every anchor to match exactly once, is idempotent, and ends with a verification step that exits non-zero on failure.

Files that more than one patch edits are edited in installer order, and each later patch pins the result of the earlier one.
Paths are under `vllm/` unless noted:

| File | Edited by (in order) |
|---|---|
| `models/deepseek_v41/nvidia/flashinfer_sparse.py`, `v1/attention/backends/mla/indexer.py` | B1 |
| `distributed/device_communicators/shm_broadcast.py` | 001 |
| `distributed/device_communicators/cuda_communicator.py` | 012 (which also installs `_gb10_oneshot_ar.py`), 018 |
| `distributed/parallel_state.py` | 018, then 043 (the module-level `graph_capture`) |
| `v1/worker/gpu/async_utils.py` | 018 |
| `config/speculative.py` | 013 |
| `models/deepseek_v41/nvidia/dspark.py` | 013, 020, 029 |
| `model_executor/kernels/linear/__init__.py` | 014, 021 |
| `model_executor/layers/quantization/modelopt.py`, `models/deepseek_v4/nvidia/ops/o_proj.py` | 014 |
| `models/deepseek_v41/nvidia/model.py` | 015, 022, 028, then 045 |
| `config/engram.py`, `models/deepseek_v41/common/engram.py`, `model_executor/model_loader/ep_weight_filter.py` | 016 (which also installs its NVMe reader into `models/deepseek_v41/common/`; 030 replaces two of those files) |
| `model_executor/kernels/attention/dsa/candidate_blocks.py` | 019 |
| `model_executor/layers/sparse_attn_indexer.py` | 043 (which also installs `_indexer_decode_chunk.py`) |
| `model_executor/layers/fused_moe/router/gate_linear.py` | 023 |
| `v1/worker/gpu/model_runner.py` | 024, 026, 028 (runner half) |
| `v1/worker/gpu/spec_decode/dspark/speculator.py` | 024, 026, 038 |
| `model_executor/kernels/mhc/tilelang_kernels.py` | 025 |
| `model_executor/layers/fused_moe/router/fused_moe_router.py` | 026 |
| `v1/core/sched/scheduler.py` | 031 |
| `v1/worker/gpu_worker.py` | 032 |
| `model_executor/layers/fused_moe/oracle/mxfp4.py` | 033 |
| `model_executor/layers/fused_moe/b12x.py` | 034 |
| `flashinfer/experimental/b12x/...` (11 files) | 036, then 037 (4 of them) |
| `vllm/_mv.py` (installed by 026), `vllm/_memstats.py` (installed by 032) | 040, 042 |

## 2. Summary table

"Numerics" uses the classes of section 1.4. "Build" is the build the effect was measured on (section 1.5). "Switch" is what
`config/recipe.env` sets; a dash means the patch has no switch and is always on once installed.

| # | Area | What it does | Switch in `config/recipe.env` | Numerics | Measured effect | Status |
|---|---|---|---|---|---|---|
| B1 | attention, indexer | the sparse-attention and indexer backends offer 64-token kernel blocks on SM12x | build time; `--block-size 64` | KV layout only | enabler: the engine does not start without it | active (in the image) |
| 001 | CPU | the inter-process wait sleeps after 2 ms instead of spinning for 1 s | - | identical | idle reader CPU 1.001 s → 0.004 s per 1.6 s (head node only) | active |
| 012 | communication | one-shot all-reduce for decode-size messages (up to 64 KiB, 6 tokens) | `VLLM_GB10_ONESHOT_AR=1` | exact arithmetic, different order: fp32 rank-order sum, one bf16 rounding | step 58.8 → 54.1 ms, with 001 (b004) | active; the fallback beneath 018 |
| 013 | drafter | DSpark draft top-k for the V4.1 draft | `"dspark_draft_topk":2048` in `--speculative-config` | lossless for the output | top-k 64 cut acceptance by about 9 %; 2048 keeps it (AL 3.88 → 4.26, b011 → b012) | active |
| 014 | dense GEMM | `wo_a` as a plain GEMM through the MXFP8 linear kernel instead of a BF16 emulation | - | exact arithmetic, different kernel | 40.14 → 39.06 ms per step together with 015 (b012 → b013) | active |
| 015 | embedding | token embedding replicated on every rank, no all-reduce at the start of each graph | - (045 switches it off) | identical | counted with 014 above; costs +1.07 GiB per rank, given back by 045 | installed; switched off by 045 in this configuration |
| 016 | memory | Engram tables of chosen layers read from node-local NVMe | `--engram-config {"nvme":true,"nvme_layers":[1,14],"nvme_verify_pct":1}` | identical (rows bit-identical) | frees 11.8 GiB per node and layer; decode step unchanged (38.1 ms, layer 14, b019) | active |
| 018 | communication | RoCE one-shot all-reduce (up to 128 KiB) and all-gather (shard up to 512 KiB), and a two-shot all-reduce up to 1 MiB reduced on the Grace CPU | `VLLM_GB10_ROCE=1`, `VLLM_GB10_ROCE_TWOSHOT_MAX_BYTES=1048576`, `GB10_TWOSHOT_PROXY_CPU=auto` | same bits as 012 | step 38.06 → 36.12 ms (one-shot, b020) → 33.63 ms (two-shot, b021) | active |
| 019 | indexer | the candidate mask skips tiles past each row's end | - | identical where read | 147 → 12 µs per call; step 33.86 → 33.30 ms (b024) | active |
| 020 | drafter | MXFP8 copy of the lm_head for the draft logits; column-parallel `main_proj` | `VLLM_DSPARK_FP8_HEAD=1`, `VLLM_DSPARK_MAIN_PROJ_TP=1` | lossless for the output | drafter graph 3.05 → 2.44 ms per step (b025) | active |
| 021 | dense GEMM, prefill | MXFP8 W8A8 for GEMMs of 512 rows or more, Marlin W8A16 below | `VLLM_MXFP8_PREFILL_W8A8=1`, `--linear-backend marlin` | changes numerics for batches of 512 rows or more (activations in MXFP8); gate KL5 0.0180 (b025) | prefill +5 % (b025) | active |
| 022 | prefill | sequence-parallel mHC in layers 0-19: reduce-scatter + all-gather instead of all-reduce | `VLLM_GB10_PREFILL_SP=1`, `VLLM_GB10_PREFILL_SP_ENGRAM=a2a` | changes numerics (prefill reduction order); gate KL5 0.0180 (b029) | prefill +13 % at 32k, 4,957 tok/s (b029); 5,353 with 8,192-token chunks (b030); 5,719 with the Engram all-to-all (b032) | active |
| 023 | router | CuTe dot-product kernel for the router GEMM at M <= 8 | `VLLM_GB10_ROUTER_LL=1` | exact arithmetic, different kernel | kernel 16.2 vs 25.8 µs at M=6; no measurable effect on the step in the engine (b026) | active |
| 024 | drafter | the request's top-p applied to the draft distribution | `VLLM_DSPARK_DRAFT_TOP_P=1` | lossless for the output | agent-turn acceptance 4.01 → 4.20 (b026) | active (sharpening knobs off) |
| 025 | mHC | split configs of the fused mHC kernel at 6 and 12 tokens | `VLLM_GB10_MHC_SPLIT=6=4,8,128/12=6,8,128` | identical outputs for these configs | 16.9 → 15.4 µs per call, -0.13 ms per step (b034) | active |
| 026 | speculative decoding | masked verification: drafts past a per-request cap cost no expert reads | `VLLM_GB10_MV=chain` | lossless for the output | +3.3 % tokens per second of step time vs off (b040; +3.2 % with the first-draft policy, b038) | active |
| 028 | prefill | two-half overlap of communication and compute inside 022's region, with the indexer row split on its own NCCL communicator | `VLLM_GB10_PREFILL_SP_OVERLAP=1` and six more (listed under 028), `indexer_tp_shard_min_rows` 2048 | changes numerics (reduction order and tiling, same class as 022) | prefill 5.4-5.7k → 6.6-6.9k tok/s, +21 %; 1M prompt 200 → 171.7 s (b064) | active |
| 029 | drafter | the top-k Markov bias of the draft in one Triton launch | `VLLM_DSPARK_MARKOV_FUSED=1` | lossless for the output | 64-user drafter graph 40.8 → 31.4 ms (b049) | active |
| 030 | Engram | packed rows: one 512-byte read per row; the 1 % verification re-read after the hand-over | `ENGRAM_NVME_PACKED_DIR`, `ENGRAM_NVME_PACKED=require`, `ENGRAM_NVME_VERIFY_AFTER=1` | identical (same bytes) | removes about 80-100 ms per prefill chunk; prefill back at b042 level with layer 1 on NVMe (b054) | active |
| 031 | scheduler | scheduler knobs from a control file | `VLLM_GB10_SCHED_CTL` (not set) | scheduling only | A/B instrument; its result is the `--long-prefill-token-threshold 4096` in the recipe | inert in this configuration |
| 032 | memory | periodic CUDA allocator statistics per worker | `VLLM_GB10_MEMSTATS=60` | none (logging only) | - | **diagnostics only** |
| 033 | MoE | expert slice zero-padded 288 → 320 | `VLLM_GB10_B12X_PAD64` (not set) | identical (zero padding) | step 31.50 → 29.12 ms with 034 (b056); replaced by 036 | inert in this configuration |
| 034 | MoE | attaches b12x's compact micro launchers to vLLM's scratch-plan bindings | - | identical | required for batches of 8 tokens or fewer on the compact path (the b055 boot failed without it) | active |
| 036 | MoE (b12x) | native compact path for widths with n % 128 == 32: the 288-wide expert slice stored and read exactly | `B12X_COMPACT_N32=1`, `--moe-backend b12x` | exact arithmetic; bitwise equal to the 320 padding at decode sizes | experts 38.3 → 34.5 GiB per node, step 29.12 → 28.48 ms, KV pool +11 % (b061) | active |
| 037 | MoE (b12x) | 64-row split kernels for large launches on the compact path | `B12X_COMPACT_PREFILL_TILE_M=64`, `B12X_COMPACT_PREFILL_MIN_ROWS_PER_EXPERT=16` | identical (tile policy only) | bench: 0.815× the 16-row time at 8,192 tokens; engine +0.5 % prefill (b065) | active |
| 038 | drafter | all-gather each rank's top-k candidates instead of the full vocabulary | `VLLM_DSPARK_TOPK_GATHER=1`, `..._MIN_ROWS=5`, `..._CHECK=4` | identical to the stock path (bit for bit) | 64 / 128 users +1.7 / +3.7 % (b066) | active |
| 040 | speculative decoding | 026's chain confidences are computed only in drafter graphs that can use them | `VLLM_GB10_MV_SKIP_INELIGIBLE=1` | identical | 64 / 128 users +3.5 / +3.9 % (b067) | active |
| 042 | memory | on-demand caching-allocator snapshot in 032's thread | trigger file (default `/cache/huggingface/dsv41-memsnap.trigger`), needs `VLLM_GB10_MEMSTATS` | none (logging only) | - | **diagnostics only** |
| 043 | memory | the decode sparse indexer's rows run in chunks of at most 512 MiB inside the CUDA graphs; the drafter's graphs share the main graphs' capture stream | `VLLM_GB10_INDEXER_DECODE_CHUNK_MB=512`, `VLLM_GB10_SHARED_CAPTURE_STREAM=1` | identical where read: logits per row bit-identical; the top-k index set per row equals stock's (the order of stock's own top-k is not reproducible either) | graph pool 3.79 → 0.68 GiB per rank; against b071 about 1 % at 64 users, nothing measurable at 1 or 128 users (b072) | active |
| 045 | memory | makes 015's replicated embedding switchable: vocab-parallel again | `VLLM_GB10_EMBED_TP=1` | identical | model memory 39.76 → 38.69 GiB per node; one-user decode unchanged, prefill about -1 % (b071) | active |

How these add up, end to end: the single-user decode step went from 58.75 ms (b001) to 28.3 ms (28.28 ms in the b066 profile,
28.2-28.6 ms in the profiles of b070 to b072), prefill from 3.6-3.8k to 6.7-6.9k tok/s (b072: 6,695 / 6,901 / 6,778 at 8k / 32k /
128k), and the KV pool from 3.1 M to 20.4 M tokens (20,411,335 in b072). The final b072 numbers are in the README ("Results at a glance").

## 3. The changes, one by one

Each section says what the patch changes, why, what it does to numerics, which settings it has (and what the recipe sets), what
we measured (with the build), and the boot line to look for. Every patch that goes through `xpatch.py` also prints
`[OK] [NNN] <file> patched (sha256 ...)` for each file it edits and `[skip] [NNN] <file> already patched` on a second run.

### 3.1 Build-time bake

#### B1 — 64-token kernel blocks on SM12x (vLLM PR #60762)

- **Problem.** On SM12x the DeepSeek-V4.1 sparse-attention backend (`FLASHINFER_MLA_SPARSE_DSV41`) and the V4.1 indexer backend
  declare 128-token kernel blocks. The SM12x kernels, FlashInfer's sparse-MLA decode and DeepGEMM's paged-MQA logits, take 32
  or 64. The engine therefore stops while it negotiates the block size (`No common block size for 64
  (FLASHINFER_MLA_SPARSE_DSV41: [128])`) or in DeepGEMM (`block_kv == 32 or block_kv == 64`).
- **Change.** Two anchored edits in `models/deepseek_v41/nvidia/flashinfer_sparse.py` and `v1/attention/backends/mla/indexer.py`:
  on compute-capability family 12x both backends declare `[64]` and keep their earlier values elsewhere. The recipe passes
  `--block-size 64`.
- **Switch.** Build time; no switch. `dsv41_bake.py` pins the sha256 of both files, requires each anchor to match exactly once
  and writes nothing if anything is off. Check: `image/build.sh` reports `already applied` for both files
  ([image/IMAGE.md](image/IMAGE.md)).
- **Numerics.** The edits change which page size the backends offer; the kernels are the same ones.
- **Effect.** An enabler; no isolated measurement. Upstream reports show the engine failing in the block-size negotiation without
  it, and the launcher's readiness rules fail the boot on both messages (b001 booted with it).
- **Upstream.** Vendored from vLLM PR #60762, which was open and reported mergeable on 25 October. The test reports in its thread
  are from SM120 GPUs; this recipe runs it on SM121. When it lands in a release the bake becomes unnecessary.

### 3.2 Runtime patches

Applied by `patches/entrypoint.sh` in numeric order.

#### 001 — IPC spin-wait grace (`001-gb10-spin-wait.sh`)

- **Problem.** vLLM's cross-process wait (`shm_broadcast.SpinCondition`) spins on a core for `busy_loop_s` (default 1 s) before it
  sleeps. Decode messages arrive faster than that, so the sleep path never engages and performance cores sit at full clock.
- **Change.** One value in `distributed/device_communicators/shm_broadcast.py`: `busy_loop_s: float = 1` becomes `0.002` (a 2 ms
  grace, then sleep through the notify socket). A wait policy, not a computation. With one GPU per node only the head node has a
  shared-memory queue (the engine core and rank 0's worker); the other ranks block in a network receive. The patch is applied on
  every node anyway, since the containers are identical.
- **Switch.** None; always applied. The script accepts only the original file and its own result, and warns if
  `VLLM_USE_SPINLOOP_EXT` is set (that vLLM extension polls for up to 100 ms before the 2 ms value is reached). Boot line:
  `[OK] busy_loop_s = 0.002 — IPC wait sleeps after 2ms instead of spinning 1s.`
- **Numerics.** None.
- **Effect.** Measured on the image's own `SpinCondition` class with real sockets: an idle reader used 1.001 s of CPU in 1.6 s with
  stock vLLM and 0.004 s patched. A sleeping reader wakes in a median of 846 µs (p95 1,053 µs, p99 1,202 µs), which is paid
  whenever a message arrives more than 2 ms after the last one. Its effect on tokens per second on its own was not measured.
- **Upstream.** No knob: `busy_loop_s` is unchanged on main. Origin: the DSpark recipe by MiaAI-Lab and Anemll (their issue #79).

#### 012 — One-shot all-reduce for small tensor-parallel messages (`012-gb10-oneshot-allreduce.sh`)

- **Problem.** Decode all-reduces are latency-bound. NCCL's tree takes about six sequential host-staged hops over 100 GbE because
  GB10 has no GPUDirect RDMA. Sending the vector to all seven peers at once and summing locally is one network step.
- **Change.** Installs `vllm/_gb10_oneshot_ar.py` and makes two anchored edits in
  `distributed/device_communicators/cuda_communicator.py`. `__init__` creates the helper for the TP group; its constructor runs a
  collective, self-checking warm-up. `all_reduce` hands contiguous bf16 tensors of at most 64 KiB to it as its first statement:
  the vector goes to all seven peers in one grouped send/recv, is summed locally in fp32 in rank order and rounded once to bf16,
  giving identical bits on every rank. Everything else stays on NCCL.
- **Scope.** With hidden size 5120 one token is 10 KiB of bf16, so 64 KiB covers a decode step of up to 6 tokens: **one request
  with its 5 DSpark drafts (60 KiB)**. A step of two requests (12 tokens, 120 KiB), prefill chunks and non-bf16 tensors stay on
  NCCL. In the shipped configuration patch 018 sits in front of this one: its hook is the first statement of `all_reduce` and it
  takes the messages 012 would take (bf16, contiguous, up to 64 KiB, subject to RoCEnante's own size rules). 012's one-shot is
  therefore the layer beneath 018 and runs only for tensors 018 declines. It stays installed because 018's edit is applied on top of
  012's result; its boot warm-up still runs and `launch/readiness.txt` still requires its ACTIVE line.
- **Switch.** `VLLM_GB10_ONESHOT_AR=1` on every rank. `VLLM_GB10_ONESHOT_AR_MAX_BYTES` (a positive even number, default 65536)
  moves the limit. The recipe sets `VLLM_ALLREDUCE_USE_FLASHINFER=0` and `--disable-custom-all-reduce`, so that the messages the
  one-shot does not take end in NCCL and not in a FlashInfer all-reduce that a multi-node group on GB10 cannot use; the script
  warns if the gate is on and that variable is not 0. A wrong sum in the warm-up stops the boot. Boot line:
  `[patch-012] one-shot all-reduce ACTIVE for group tp:0`.
- **Numerics.** Exact arithmetic, different order: the result differs from NCCL's per-hop bf16 additions, so greedy outputs are
  not bit-equal to a stock all-reduce. The one-shot sum is bit-exact against a local fp32 reference, identical on all eight ranks,
  and has a smaller error than NCCL's tree (GPU test on eight nodes, 26 October, `tests/patch-checks/oneshot_ar_test.py`). Logit
  comparison on the code and prose corpus (quick test on a sibling checkpoint with identical tensor shapes): the boot with 001 and
  012 (and the 1M window and 32 sequences) against the first boot without them gives 98.5 % top-1 agreement, a top-5 KL of 0.0026
  nats per token and a mean log-likelihood change of 0.0004 in absolute value.
- **Effect.** Per collective, in a CUDA graph on 8 nodes (throwaway containers on idle nodes):

  | Tokens per message | Message | NCCL | One-shot |
  |---|---|---|---|
  | 1 | 10 KiB | 78 µs | 46 µs |
  | 2 | 20 KiB | 99 µs | 53 µs |
  | 5 | 50 KiB | 126 µs | 81 µs |
  | 6 | 60 KiB | 130 µs | 90 µs |
  | 12 | 120 KiB | 127 µs | 134 µs (no gain) |
  | 18 and more | | | slower than NCCL |

  About 40 µs saved on each of about 85-90 collectives is roughly 3.5 ms per decode step. Measured at batch 1, the decode step went
  from 58.8 ms (b001) to 54.1 ms (b004), mean over prompts of 8k, 32k and 128k tokens: +8 % for one user, with 001 and 012 switched
  on together and `--max-num-seqs` raised from 8 to 16 in that boot. Quick test, sibling checkpoint, total tokens/s for 1, 2, 4
  and 8 chat users: 55.3 → 61.3, 73.3 → 76.8, 97.2 → 97.0 and 124.7 → 123.4. From four users on the steps are too big for the
  one-shot. `NCCL_ALGO=Tree` alone gave 122 µs for the 6-token message, `Ring` 133-139 µs, and the channel count made no
  difference (2 channels 131 µs, 1 channel 135 µs).
- **Upstream.** Nothing upstream for multi-node RoCE: FlashInfer's all-reduce, the PCIe-IPC route and symmetric memory are
  single-node.

#### 013 — DSpark draft top-k for DeepSeek-V4.1 (`013-dspark-draft-topk.sh`)

- **Problem.** The DSpark drafter samples its 5 draft tokens left to right; each step adds a Markov bias, `markov_w2 @
  embed(previous token)`, over the full 129,280-token vocabulary. `markov_w2` is a replicated BF16 [129,280 × 256] matrix (66 MB),
  so every rank reads it 5 times per decode step: about 300 µs each on GB10, about 1.5 ms of a 54 ms step (profile of 26 October).
- **Change.** vLLM already has the cheaper form for Qwen3 DSpark (`SpeculativeConfig.dspark_draft_topk`,
  `DSparkSpeculator._sample_sequential_topk`): the base draft logits pick their top-k candidates once and the Markov bias is applied
  only to those. Two anchored edits: `config/speculative.py` accepts `dspark_draft_topk` for the DeepSeek-V4.1 draft, and
  `models/deepseek_v41/nvidia/dspark.py` gets `apply_markov_bias_gathered` (fp32 product, result in the logits' dtype, same scale
  as the stock bias).
- **Switch.** `"dspark_draft_topk":2048` in `--speculative-config`; the patch is inert unless the speculative config sets it.
- **Numerics.** Lossless for the output. The rejection sampler uses the same truncated draft distribution the draft token was
  sampled from, so the served distribution stays the target model's; only the acceptance rate can change.
- **Effect.** The value matters: top-k 64 lowered acceptance by about 9 % (AL 3.88 in b011); top-k 2048 keeps it (AL 4.26, agent
  turns 101 tok/s in b012; step 40.06 vs 40.14 ms).
- **Upstream.** vLLM gates the option to the Qwen3 architectures; the DeepSeek-V4.1 draft model lacks the gathered-bias hook.

#### 014 — `wo_a` as a plain MXFP8 GEMM (`014-wo-a-plain-mxfp8.sh`)

- **Problem.** The attention output projection `wo_a` is a grouped GEMM (8 groups of [1024 × 4096]); vLLM runs it through an MXFP8
  BMM kernel. On SM12x the only BMM kernel DeepGEMM offers needs Blackwell datacenter parts (family 100), so vLLM falls back to an
  emulation that dequantizes the FP8 weight to BF16 once and runs a BF16 `torch.bmm` per layer: about 50 µs per layer, twice the
  weight bytes (profile of 26 October: 45 launches, 3.9 ms of a 54 ms decode step).
- **Change.** At TP=8 every rank holds exactly one group, so the BMM is a plain [T, 4096] × [4096, 1024] GEMM. Three anchored edits
  (`model_executor/kernels/linear/__init__.py`, `model_executor/layers/quantization/modelopt.py`,
  `models/deepseek_v4/nvidia/ops/o_proj.py`) select the plain MXFP8 linear kernels for such a layer, mark it, and run it as a
  linear: inverse RoPE without quantization, then `wo_a`, then `wo_b`.
- **Switch.** None; it applies when a layer has one local group, no DeepGEMM BMM support is present and the attention kernel did
  not already produce a quantized activation. At TP < 8 or on SM100/103 the stock path stays.
- **Numerics.** Exact arithmetic, different kernel. The plain kernel is whichever one `--linear-backend` selects for the model's
  other MXFP8 linears: Marlin W8A16 in this recipe (activations stay BF16), and the FlashInfer CUTLASS W8A8 kernel for prefill-size
  batches with patch 021.
- **Effect.** Together with 015: step 40.14 → 39.06 ms (b012 → b013); KL5 0.0182 in b013 (b011: 0.0164). The b015 switch to Marlin
  then lowered it to 0.0160.

#### 015 — Replicated token embedding (`015-replicated-embedding.sh`)

- **Problem.** The vocab-parallel embedding ends in an all-reduce, which is the first collective of both decode CUDA graphs (the
  target and the DSpark drafter, which shares the embedding module). In the profiles of 26 October the first all-reduce of each
  graph ended 0.75-1.0 ms (target) and 1.2-1.5 ms (drafter) after the graph started, against about 65 µs for every later one.
- **Change.** `DeepseekV4Model.__init__` in `models/deepseek_v41/nvidia/model.py` builds the embedding with `disable_tp=True`: the
  full 129,280 × 5,120 BF16 table (1.32 GB) on every rank, no all-reduce. A vocab-parallel rank holds an eighth of it, so
  replication adds 1.16 GB per rank, which is 1.08 GiB; "Model loading took" fell by 1.07 GiB (39.76 → 38.69 GiB) when 045
  switched the replication off.
- **Switch.** None of its own: the installer always applies it. In the shipped configuration patch 045 switches it off again with
  `VLLM_GB10_EMBED_TP=1` (the installer still runs at every start; with the variable unset 015 behaves as described here). To go
  back to the replicated table, unset `VLLM_GB10_EMBED_TP` in `config/recipe.env` and lower the KV pool by 1 GiB (see 045).
- **Numerics.** Identical: each row has one owner in the vocab-parallel version and the all-reduce only adds zeros.
- **Effect.** Counted with 014 above. The graph-start wait did not go away: it belongs to NCCL itself. The replicated table costs
  1.07 GiB of memory per rank, which is about 0.5 M KV tokens. Patch 045 gives that memory back, and the b071 measurement of that
  trade-off was about 80 µs per step of GPU-busy time (about 0.3 % of single-user decode) and about 1 % of prefill, a trade this
  release makes (045, below).

#### 016 — Engram tables on node-local NVMe (`016-engram-nvme.sh`)

- **Problem.** The two Engram tables are 188.8 GiB; at TP=8 every rank keeps 23.6 GiB of them resident (11.8 GiB per layer). On
  GB10 that memory is the same DRAM as the KV cache, and vLLM's own `cpu_offload` (pinned host memory) saves nothing there.
- **Change.** The rows a step needs are known as soon as the hash kernel has run; layer 14's are needed 11-16 ms later, layer 1's
  about 1 ms later. This patch serves the rows of chosen layers from the checkpoint shards on each node's own NVMe: the GPU hands
  the row ids to a native reader thread through pinned memory (a doorbell kernel inside the CUDA graph), the reader reads the 256 B
  values and 8 B scales with `O_DIRECT` (Linux native AIO; a pread thread pool as fallback; no page cache, no mmap), stages them in
  pinned memory, and a wait kernel in the graph releases the dequant kernel, which runs the same arithmetic as the resident lookup.
  Anchored edits in `config/engram.py` (the `nvme*` fields), `models/deepseek_v41/common/engram.py` and
  `model_executor/model_loader/ep_weight_filter.py` (the NVMe-served layers' `embed.weight` and `embed.scale` are never loaded);
  the wrapper installs `engram_nvme.py`, `engram_nvme_io.py` and `engram_nvme.c` into `models/deepseek_v41/common/` and builds
  `libengram_nvme.so` there with gcc (raw syscalls only: no libaio, no io_uring).
- **Switch.** `--engram-config` fields; inert unless `"nvme": true`. The recipe sets `{"cpu_offload":false, "nvme":true,
  "nvme_layers":[1,14], "nvme_verify_pct":1}`. Defaults of the rest: `nvme_layers` none (layer 14 only), `nvme_io_backend` auto,
  `nvme_queue_depth` 256, `nvme_threads` 64, `nvme_poll` auto, `nvme_keepwarm_us` 500 (one 4 KiB read every 500 µs while requests
  come, so the SSD stays out of its low-power states), `nvme_timeout_s` 30, `nvme_dedupe` true, `nvme_verify_pct` 0. It needs
  `cpu_offload` false, the V2 model runner and tensor parallelism only.
- **Numerics.** Identical: the rows are byte-identical to the resident path. Fail closed: an I/O error, a short read, a protocol
  violation, a request pending longer than `nvme_timeout_s`, a GPU-side wait timeout or a changed shard file ends the process with
  exit status 86 (the dequant kernel writes NaN rows meanwhile); rows are never guessed. `nvme_verify_pct` re-reads that share of
  the delivered rows from the original shard and compares.
- **Effect.** Layer 14 (b019): decode step unchanged (38.1 ms) and 11.8 GiB per node freed, which went into a 16 GiB KV pool
  (8.34 M tokens). Layer 1 as well (b048): another 11.8 GiB freed, decode unchanged (the wait kernel takes 1.7 µs at the median),
  but each 8,192-token prefill chunk read about 25,000 layer-1 rows right after layer 0 and prefill lost 6-8 %; patch 030 removes
  that cost. At 40 GiB of KV pool (b070) the pool held 18.77 M tokens; at the shipped 43.5 GiB (b072) it holds 20.41 M.
- **Boot lines.** `[engram-nvme] layer 14: rows ...`, `[engram-nvme] layer 1: rows ...`, `[engram-nvme] reader started: backend=aio`.
- **Upstream.** No disk option for the Engram tables is merged in vLLM, and its host-memory option (`cpu_offload`) saves nothing on
  GB10's unified memory.

#### 018 — RoCE one-shot and two-shot collectives for small TP messages (`018-gb10-roce.sh`)

- **Problem.** On eight Sparks (one 100 GbE RoCE port each, no GPUDirect RDMA) NCCL costs about 130 µs per decode all-reduce in a
  CUDA graph (patch 012's one-shot over NCCL send/recv: about 88 µs) and stalls about 0.5 ms at the first NCCL operation of every
  graph launch.
- **Change.** Needs patch 012 first. Installs `vllm/_gb10_roce.py` (a thin layer over FlashInfer's `b12x` RoCEnante runtime,
  `flashinfer.experimental.b12x.comm.roce`, API version 1, shipped in the image) and `vllm/_gb10_twoshot/` (our two-shot runtime,
  with a C proxy library built at container start). Anchored edits: `cuda_communicator.py` (`__init__`; `all_reduce` takes eligible
  messages before 012's one-shot and NCCL; `all_gather` takes eligible shards), `parallel_state.py` (`graph_capture` enters the
  runtime's capture context) and `v1/worker/gpu/async_utils.py` (a health check once per step).
  - *One-shot* (RoCEnante): one fused kernel stages the input in pinned host memory, rings a C proxy thread that posts the RDMA
    writes, waits for every peer's payload and reduces in rank order.
  - *Two-shot* (ours): every rank sends each peer only its eighth of the message in small pieces; the Grace CPU adds its eighth in
    rank order as the pieces arrive (NEON, fp32, one round-to-nearest-even) and sends the result to everyone. About 1.75× the
    message instead of 7× crosses the wire.
- **Switch.** `VLLM_GB10_ROCE=1` on every rank. `VLLM_GB10_ROCE_AR_MAX_BYTES` (default 131072, 12 tokens of this model) and
  `VLLM_GB10_ROCE_AG_MAX_BYTES` (default 524288, per-rank shard) set the one-shot limits; a limit of 0 leaves that collective on
  NCCL. `VLLM_GB10_ROCE_TWOSHOT_MAX_BYTES` (default 0 = off; the recipe sets 1048576) turns the two-shot on for every eligible
  all-reduce up to that size; all-gathers stay on RoCEnante. `GB10_TWOSHOT_PROXY_CPU` (default -1 = not pinned; the recipe sets
  `auto`, the last of the fast Cortex-X925 cores) pins the proxy thread. The piece size (4,096 B) and prefetch (32 KiB) stay at
  their defaults.
- **Numerics.** The all-reduce is an fp32 sum in rank order 0..W-1 with one rounding to the input dtype: the same bits as patch 012
  (tested bit-exact against 012 and the fp32 reference, identical on all 8 ranks). All-gathers are plain copies.
- **Safety.** The constructor is collective and runs a warm-up that raises on any wrong element (one-shot, two-shot and
  all-gather); every eager call and every step check the runtime's health and raise if a wait timed out or the proxy died
  (RoCEnante poisons itself instead of returning data).
- **Effect.** Standalone, 8 ranks, CUDA graphs, next to the idle engine (26 October): one-shot 23 / 61 / 100 µs at 1 / 6 / 12
  tokens (NCCL 81 / 130 / 131, patch 012 47 / 88); all-gather of a [6, 768] bf16 shard 26 µs (NCCL 72). Two-shot 19 / 32 / 49 / 69
  / 117 / 355 µs at 1 / 6 / 12 / 24 / 48 / 192 tokens (one-shot 23 / 61 / 99 / 175 / 329 / 1,249; NCCL 73 / 125 / 124 / 148 / 184
  / 359). In the engine: the first collective of each graph fell from 472 to 115 µs, the step from 38.06 to 36.12 ms, agent turns to
  115.7 tok/s (b020, one-shot); with the two-shot 35 µs per all-reduce at one user (68 in b020) and a step of 33.63 ms (b021). At
  32 and 64 users (2-4 MB per all-reduce) the two-shot lost 2 % to NCCL, because the CPU reduction competes with the GPU for
  memory bandwidth; limiting it to 1 MiB (b022) gave total tokens/s at 1 / 2 / 3 / 4 / 8 / 16 / 32 / 64 users of 93 / 124 / 147 /
  170 / 223 / 299 / 405 / 569, against b020 +5 / +4 / +8 / +13 / +5 / +2 / -1 / +1 %. Pinning the proxy to a fast core (b023) changed
  nothing measurable. Moving every other thread off the proxy's core also changed nothing: the latency is the protocol's own.
- **Boot lines.** `[OK] [018] installed _gb10_roce.py`, `[OK] [018] installed _gb10_twoshot, proxy library ...`,
  `[patch-018] RoCEnante ACTIVE for group tp:0 (...)`, `[patch-018] two-shot all-reduce ACTIVE for group tp:0 (all-reduce <=
  1048576 B; ...)`. Failure lines the readiness rules watch for: `RoCE proxy failed`, `RoCE collective on rank N timed out`.

#### 019 — Skip candidate-mask tiles past each row's end (`019-mask-skip.sh`)

- **Problem.** DeepSeek-V4.1's two-level sparse indexer masks the indexer logits to the candidate blocks before the row top-k. The
  logits buffer is `max_model_len` wide (static for CUDA graphs) and the mask kernel writes -inf to every column outside the
  row's span: at 1M context about 4 MB of writes per decode row, 4 calls per step, 473 µs per step at one user (1.4 % of the step)
  for a 300-token context.
- **Change.** One anchored edit in `model_executor/kernels/attention/dsa/candidate_blocks.py`: a tile that starts at or past the row's
  end returns without writing. Every consumer of the masked logits reads only the row's span (the decode top-k backends clamp to
  the row length, the prefill top-k takes row starts and ends, the fp32 upcast covers only the span).
- **Switch.** None; always applied.
- **Numerics.** Identical where read: columns inside the span are computed as before, including the last-column edge case.
- **Effect.** 147 → 12 µs per call at short contexts; decode step 33.86 → 33.30 ms, 64 users 594 tok/s (b024).

#### 020 — DSpark drafter cost cuts (`020-dspark-lean.sh`)

- **Problem.** The drafter graph took 3.05 ms of a 33.9 ms step at one user (b023 profile): the draft logits read the target's BF16
  lm_head (16,192 × 5,120 per rank, 166 MB, 736 µs per step), and `main_proj` (15,360 → 5,120, MXFP8, 78.6 MB) is a replicated
  layer, so every rank streams all of it (about 320 µs).
- **Change.** Needs patch 013. Installs `vllm/_dspark_lean.py` and edits `models/deepseek_v41/nvidia/dspark.py`.
  `VLLM_DSPARK_FP8_HEAD=1`: the draft logits use an MXFP8 copy of the lm_head shard (E8M0 scales per 32 inputs), through the same
  Marlin MXFP8 kernel as the dense layers (about 2× fewer bytes), built lazily on the first eager call; the target's BF16 lm_head
  is untouched. `VLLM_DSPARK_MAIN_PROJ_TP=1`: `main_proj` becomes column-parallel (each rank reads 1/8 of the weights) with an
  all-gather of the [T, 5,120] output (a RoCE all-gather through patch 018); each output column is the same dot product as before.
- **Switch.** Both variables, opt-in; the recipe sets both.
- **Numerics.** Lossless for the output: only the draft tokens and their probabilities change slightly; block verification keeps
  the output the target's. The lm_head copy is self-checked on the real weights at start (cosine 0.99961, top-1 agreement 1.00).
- **Effect.** Drafter graph 3.05 → 2.44 ms per step (b025). A BF16 draft head instead (b039) was no better in acceptance (3.92 vs
  3.96 tokens per step) and cost 0.45 ms more per step.
- **Boot lines.** `[patch-020] DSpark draft lm_head: MXFP8 copy ...`, `[patch-020] DSpark main_proj column-parallel (15360 -> 5120, output all-gathered)`.

#### 021 — MXFP8 W8A8 for prefill-sized GEMMs (`021-prefill-w8a8.sh`)

- **Problem.** `--linear-backend marlin` keeps activations in BF16 (W8A16): the most exact arithmetic and the fastest decode GEMMs on
  GB10, but prefill pays about 4-5 % against FlashInfer's CUTLASS W8A8 MXFP8 GEMM (b013: 4,363 tok/s at 32k, Marlin 4,139).
- **Change.** Needs patch 014. Installs `vllm/_prefill_w8a8.py`; `init_mxfp8_linear_kernel` returns a hybrid kernel that chooses by
  the number of rows of each call: below `VLLM_MXFP8_PREFILL_W8A8_MIN_TOKENS` (default 512) Marlin W8A16 runs unchanged, at or above
  it the CUTLASS W8A8 kernel runs (activations quantized to MXFP8 by the kernel, the model's own FP8 recipe). The default was chosen
  when the engine took at most 64 sequences, so that every CUDA-graph decode batch (at most 64 × 6 = 384 rows) stayed on Marlin;
  the installer refuses a threshold of 384 or less. The W8A8 path keeps its own copy of the FP8 weights and E8M0 scales: about
  +1.2 GB per rank. A step that mixes decode rows into a prefill chunk of at least the threshold runs W8A8 for the whole batch.
- **Switch.** `VLLM_MXFP8_PREFILL_W8A8=1` (the recipe sets it; the threshold stays at 512); inert otherwise.
- **Numerics.** Changes numerics for batches of 512 rows or more (activations quantized); smaller batches stay W8A16 bit for bit.
  Admitted by a quality gate set on 26 October: keep it only if KL5 against the official model is at most 0.0185 and the task scores
  stay level. Passed: b025 KL5 0.0180, GSM8K 49/50, HumanEval+ 19/20, tool calls 19/24, needles 12/12 (b029: 0.0180; b032's prefill
  path: 0.0171).
- **Note for 128 sequences (derived from the code and the recipe, not a separate measurement).** The recipe runs
  `--max-num-seqs 128` with CUDA-graph sizes up to 768 rows. A decode step of 512 rows or more (about 86 or more running requests with
  5 drafts each) therefore takes the W8A8 path in its dense GEMMs too, because the choice is made per captured batch size. The
  quality runs use 4 requests at a time, which stay on Marlin; the 96- and 128-user throughput runs of b070 to b072 include the W8A8 path.
- **Effect.** Prefill +5 % (b025: 8k 4,358, 32k 4,382, 128k 4,218 tok/s).
- **Boot line.** `[patch-021] MXFP8 prefill W8A8 ACTIVE: Marlin W8A16 below 512 rows, FlashInfer CUTLASS W8A8 from 512 rows`.

#### 022 — Sequence-parallel mHC in prefill (`022-prefill-sp.sh`)

- **Problem.** At TP=8 every rank runs the whole mHC (`mhc_post`, the TF32 pre-norm GEMM, `mhc_pre_big_fuse_with_norm`) on all T rows
  of a prefill chunk: about 18 % of the GPU time of a 4,096-token chunk, replicated eight times. The two row-parallel outputs of
  each layer (attention `wo_b`, MoE) end in an all-reduce (about 34 % of the chunk).
- **Change.** Needs patch 015 (it pins `model.py` as 015 leaves it). Installs `vllm/_prefill_sp.py` (and `_engram_sp_a2a.py`) and
  makes four anchored edits in `models/deepseek_v41/nvidia/model.py`. For an eligible forward (eager, no CUDA graph, at least
  `VLLM_GB10_PREFILL_SP_MIN_T` rows, default 1,024) the layers 0-19 (up to the bounded decoder replay) run sequence-parallel: rank r
  owns rows [r·S, (r+1)·S), each all-reduce becomes a reduce-scatter (every rank keeps the reduced rows it owns) plus an all-gather
  (the next block needs all rows), and the mHC between them runs on T/8 rows. At the end of the region the five carried states are
  all-gathered to full T and everything after (the bounded replay, the collapse, the final norm, the logits) runs as before. Decode,
  short prefills and CUDA-graph captures and replays never enter a region. It is written for TP=8 without expert parallelism and
  refuses other layouts.
- **Engram under sequence parallelism** (`VLLM_GB10_PREFILL_SP_ENGRAM`, default `gather` = the upstream path; the recipe sets
  `a2a`). Inside a region each rank needs the Engram kv only for its own rows, but the upstream path computes it for all rows and
  moves it with two TP all-gathers (about 39 ms of link time per 4,096-row chunk over both Engram layers, about 5 % of a chunk). In
  `a2a` mode one all-to-all hands each rank all 24 hash heads of its own rows and a replicated `wkv` (+160 MB per Engram layer per
  rank) runs the CUTLASS W8A8 GEMM on those rows. The first live use compares both paths on all ranks (data movement bitwise, the
  GEMM within a tolerance); on any failure all ranks fall back to the gather path for the rest of the process and log it (this
  happened in b030 because of a config lookup, and was fixed in b032).
- **Switch.** `VLLM_GB10_PREFILL_SP=1` on every rank; with it unset the installer touches nothing. `VLLM_GB10_PREFILL_SP_MIN_T`
  (default 1,024).
- **Numerics.** Changes numerics in prefill only: reduce-scatter changes the order of the 8-way sum, and the pre-norm GEMM runs with
  the split-K factor of S rows instead of T rows. Every row is computed on exactly one rank and gathered by copy, so all ranks hold
  identical tensors at every point. Admitted by the quality gate: b029 KL5 0.0180, top-1 agreement 0.953, GSM8K 49/50, HumanEval+
  19/20, tool calls 20/24, needles 12/12.
- **Boot self-test.** On the first eligible forward (the profile run): collective, fail closed, bounded waits; synthetic
  reduce-scatter/all-gather cases, the live layers' mHC operations on a shard against the full call, the Engram kernel on a shard
  against the full one (bitwise), and the live `wo_b` and live MoE with the suppressed all-reduce plus the sequence-parallel
  reduction against the unsuppressed output. A failure raises a `[patch-022]` error on every rank. On the eight GPUs (b029): mHC
  activations bitwise, live module against all-reduce 7.8e-3.
- **Effect.** mHC per chunk 134 → 25 ms; prefill 4,899 / 4,957 / 4,701 tok/s at 8k / 32k / 128k (b029, +13 % at 32k). With
  8,192-token chunks (b030) 5,343 / 5,353 / 5,302; with the Engram all-to-all (b032) 5,709 / 5,719 / 5,590 and a 46k-token prompt
  in 7.6 s. Decode and concurrency unchanged.
- **Boot lines.** `[OK] [022] installed _prefill_sp.py`, `[patch-022] prefill sequence-parallel ACTIVE (rank r/8, T >= 1024, eager
  prefill only, layers 0-19, fingerprint ...)`, `[patch-022 engram-a2a] ACTIVE: first live Engram layer verified on all 8 ranks`.

#### 023 — Router GEMM through the CuTe dot-product kernel (`023-router-ll.sh`)

- **Problem.** `GateLinear`'s fast router tier (a CuTe `ll_bf16_gemm`, bf16 in, fp32 out) is enabled only on SM90/SM100, so on GB10 the
  MoE router runs cuBLAS bf16 → fp32 (a WMMA kernel plus a split-K reduce): 44 × about 36 µs plus 44 × 2.9 µs per decode step (b024).
- **Change.** One guarded fast path in `model_executor/layers/fused_moe/router/gate_linear.py`, `GateLinear.forward`, using vLLM's
  CuTe dot-product kernel for M ≤ 8 (vLLM routes only M ≤ 4 there; the split-K variant is slower on GB10). Installs
  `vllm/_router_ll.py`.
- **Switch.** `VLLM_GB10_ROUTER_LL=1` (the recipe sets it).
- **Numerics.** Exact arithmetic, different order: fp32 accumulation either way (errors against fp64 about 1e-6 relative, the
  dot-product kernel's slightly lower), but the summation order differs from cuBLAS, so router logits can differ in the last bits.
- **Effect.** At the kernel: N=384, M=6 16.2 µs against 25.8 µs for cuBLAS; M=12 19.0 against 20.6 (no gain); the DSpark router
  (N=128) M=6 5.0 against 7.1, M=12 10.2 against 7.5 (worse), hence the limit of 8. In the engine it saved nothing measurable
  (b026/b027): the shared-expert GEMM on its side stream now overlaps the routed MoE, so wall time did not move.

#### 024 — The request's top-p on the DSpark draft distribution (`024-draft-top-p.sh`)

- **Problem.** vLLM samples draft tokens with the request temperature only and ignores top-k and top-p. Our requests run top-p 0.95:
  every draft token sampled outside the target's nucleus is rejected for sure, so the probability mass the draft puts there is wasted.
- **Change.** Installs `vllm/_draft_top_p.py`; anchored edits in `v1/worker/gpu/model_runner.py` (register the sampler's states before
  both `propose()` calls) and `v1/worker/gpu/spec_decode/dspark/speculator.py` (`_sample_sequential_topk`: truncate each step's
  candidates to the nucleus). It runs as one Triton launch (a bisection, no sort).
- **Switch.** `VLLM_DSPARK_DRAFT_TOP_P=1` (the recipe sets it). The module also has draft *sharpening* knobs, which the recipe does
  not set: `VLLM_DSPARK_DRAFT_SHARPEN=s` (draft temperature = request temperature × s, 0.2 to 2, default 1 = off),
  `VLLM_DSPARK_DRAFT_SHARPEN_RUNTIME` and `VLLM_DSPARK_DRAFT_SHARPEN_FILE` (A/B sweeps).
- **Numerics.** Lossless for the output: the truncated logits are what the drafts are sampled from and what the rejection sampler
  receives, so verification keeps the target's distribution; requests with top-p 1 are unchanged and greedy requests keep their argmax.
- **Effect.** Agent-turn acceptance 4.01 → 4.20 and agent turns 121.9 tok/s (b026); the sort cost 0.3 ms per step, the Triton
  version (b027) 9.4 µs per call. Sharpening did not help: acceptance 4.21 / 4.07 / 4.03 / 4.25 / 4.19 at s = 1.0 / 0.85 / 0.7 /
  0.55 / 1.0 in one boot (b028), and 3.97 / 4.04 / 3.87 / 4.02 at s = 0.9 / 1.0 / 1.15 / 1.3 on a later check. s stays 1.

#### 025 — Split configs for the fused mHC kernel at decode token counts (`025-mhc-split.sh`)

- **Problem.** `mhc_fused_post_pre_split_config()` picks its tiling from GB300 tuning: (2, 8, 128) below 16 tokens. On GB10 (48 SMs)
  that op ran at about half the DRAM rate at one user: 16.9 µs per call, 82 calls per decode step.
- **Change.** Installs `vllm/_mhc_split.py` and edits `model_executor/kernels/mhc/tilelang_kernels.py` to consult an override first.
  A sweep of 52 configs on the real op found wider output tiles 15 % (6 tokens) and 8 % (12 tokens) faster at the same 8 k-splits.
- **Switch.** `VLLM_GB10_MHC_SPLIT="6=4,8,128/12=6,8,128"` maps exact token counts to (tile_n, n_splits, n_thr); every other count keeps
  the upstream choice, and a config that does not fit the shape is ignored with a log line. Unset: upstream configs.
- **Numerics.** Identical outputs for these two configs (the k-splits stay at 8; only the output tile width changes). A different
  number of k-splits would change the fp32 accumulation order.
- **Effect.** 16.9 → 15.4 µs per call, 0.13 ms per step (b034). It is not a lever at larger batches: the unsplit default was fastest at
  M = 96 to 768 (50.0 vs 113.4 µs at M = 96).
- **Boot line.** `[patch-025] GB10 fused-mHC split configs: ...`.

#### 026 — Masked verification for DSpark (`026-mv.sh`)

- **Problem.** A single-user decode step verifies 1 + 5 tokens. Every target MoE layer streams the experts of all 6 tokens, which was
  about 44 % of the step (14 ms of 32) when the patch was written. A draft that will be rejected costs the same expert reads as one
  that is accepted: at an acceptance length of about 4.2, about 1.8 of the 6 tokens per step are verified for nothing, each costing
  about 2 ms of expert reads (step 22.5 ms with only the anchor's experts, 32.9 ms with all six tokens; b037).
- **Change.** Installs `vllm/_mv.py`; anchored edits in `v1/worker/gpu/model_runner.py`, `v1/worker/gpu/spec_decode/dspark/speculator.py`
  and `model_executor/layers/fused_moe/router/fused_moe_router.py`. After DSpark drafts, each request gets a cap c in [0, 5] from
  the drafter's confidence head. In the next target forward the tokens after the cap keep their place in the batch (so CUDA-graph
  shapes do not change) but take their request anchor's experts in every target MoE layer, so they add no expert reads;
  verification treats drafts after the cap as absent.
- **Exactness.** Tokens 0..c compute what they compute without the cap (attention is causal, every other op is per token, the MoE
  kernel computes each token's rows independently), and verification reads only logits rows 0..c. The cap must not depend on the
  drafted tokens, because under block verification a block length chosen from them would bias the output. The shipped *chain* policy
  runs the confidence head along the drafter's argmax chain (computed before any draft is sampled) and picks the cap that maximizes
  expected tokens per millisecond with the measured per-cap step costs (22.5 / 25.1 / 27.5 / 29.5 / 31.2 / 32.9 ms for 0-5 verified
  drafts). The chain confidences, like the first draft's, are fixed before the drafts are drawn, so the output distribution is the
  target model's. The first version (b037), which also used the later drafts' confidences, could bias the output under block
  verification (a block length that depends on the drafted tokens) and was replaced before any serving use. A self-test on the live MoE layer at the first eligible step, every rank voting, checks the
  per-token independence (the b12x kernel is not bit-deterministic run to run, so the check accepts its own noise); on failure MV
  switches itself off for the process.
- **Switch.** `VLLM_GB10_MV=chain` (the recipe). Other values: `auto` (thresholds on the first draft's confidence: caps 1 to 5 at
  0 / 0.3 / 0.6 / 0.7 / 0.8), `fixed:<c>`, `off`; unset behaves like off (every hook returns at once). Not set by the recipe:
  `VLLM_GB10_MV_CTL` (a control file that switches modes at run time, broadcast from rank 0 at a fixed step count),
  `VLLM_GB10_MV_LOG` (per-step log), `VLLM_GB10_MV_SYNC_STEPS`. `VLLM_GB10_MV_MAX_REQS` (default 1, left at the default): only pure
  decode steps with at most that many requests, each with exactly K drafts, are eligible; other steps run unchanged.
- **Numerics.** Lossless for the output.
- **Effect.** b038 (first-draft policy), three alternating rounds of 12,288 logged steps: acceptance 3.94 vs 3.95 with MV off, mean
  step 32.62 vs 33.75 ms, tokens per second of step time 120.8 vs 117.1 (+3.2 %). b040 (chain policy): +3.3 % against MV off per
  step (the first-draft thresholds +2.5 % in the same boot). b042 (production build): step 31.17 ms (b034: 31.99). The oracle bound
  (cap = accepted) is about +14 %.
- **Boot lines.** `[patch-026 mv] N target MoE routers remap capped draft tokens (...)`, `[patch-026 mv] mode chain (from
  VLLM_GB10_MV): kind=4 ...`, `[patch-026 mv] ACTIVE: self-test passed on all 8 ranks (...)`.

#### 028 — Two-half communication/compute overlap for the sequence-parallel prefill (`028-prefill-sp-overlap.sh`, revision 5)

- **Problem.** Patch 022's region runs, per layer, all-gather → attention → reduce-scatter → mHC → all-gather → MoE → reduce-scatter →
  mHC with every collective exposed: about 534 ms of the about 1.31 s of kernel time of an 8,192-token chunk is NCCL, and nothing
  overlaps it (b042 profile of a 46k-token prompt).
- **Change.** Needs 022. Installs `vllm/_prefill_sp_overlap.py` and edits `models/deepseek_v41/nvidia/model.py` (four anchored edits,
  as 022 leaves it) and `v1/worker/gpu/model_runner.py` (`files/p028_runner.py`: a hook that hands the region the per-step data). The
  forward's T rows are cut into two halves (at a request boundary or, inside a prefilling request, at a position that is a multiple
  of 128; both halves keep at least 1,024 rows). One half's reduce-scatter → mHC → all-gather chain runs on a dedicated
  high-priority communication stream behind the other half's attention or MoE block on the compute stream; every cross-stream
  dependency is a CUDA event. Only forwards of at least `VLLM_GB10_PREFILL_SP_OVERLAP_MIN_T` (default 2,048) rows take part: warm
  agent turns (about 270 new tokens) stay on the plain path, so it helps cold long prompts.
- **Indexer row sharding.** The recipe also sets `indexer_tp_shard_min_rows` 2048 in `--attention-config`: vLLM's sparse-attention
  indexer then splits a prefill chunk's rows over the ranks and exchanges the top-k indices (exact; b052: 128k +4 %, a 512k prompt at
  5,163 tok/s). Without it a 1M prompt slows from 200 to 244 s and needs about 5 GiB more memory per node. The exchange runs on the
  compute stream while the overlap drives the TP communicator from the communication stream, which could deadlock one NCCL
  communicator, so `VLLM_GB10_PREFILL_SP_OVERLAP_INDEXER=comm2` gives the exchange its own second NCCL communicator (about 0.5 GiB
  per node), created and byte-verified collectively at the preflight.
- **Memory.** PyTorch's caching allocator keeps one pool per stream, and on GB10 reserved memory is gone from `MemAvailable`. The
  first version of the pipeline (revision 3) reserved about 5 GiB more than the plain path (95.5 against 90.3 GiB on rank 0; b060: one
  worker fell to 6.0 GiB free on the first 128k prompt). Revision 4 serves the large gather outputs from the compute stream's pool
  and releases cross-stream buffers when the compute stream has waited for them (+1.7 GiB reserved instead of +5.2; b064: rank 0
  reserved 89.51 GiB after a 1M prompt). Revision 5 shrinks the boot self-test
  to about 0.45 GiB above a normal forward (revision 4: about 0.77 GiB) and logs its own memory.
- **Settings.** Every rank needs the same values (a fingerprint is checked at the first eligible forward). With the gate unset the
  installer touches nothing.

  | Variable (suffixes of `VLLM_GB10_PREFILL_SP_OVERLAP`) | Default | Recipe |
  |---|---|---|
  | the gate itself, `VLLM_GB10_PREFILL_SP_OVERLAP` (needs `VLLM_GB10_PREFILL_SP=1`) | off | `1` |
  | `_MIN_T` | 2048 | default |
  | `_CTL` (file read by rank 0 before every eligible forward: `on`, `off`, `log=N`; a missing or empty file keeps the mode, which starts as on) | none | a file in the cache directory (`dsv41-p028.ctl`) |
  | `_LOG_EVERY` | 20 | 10 |
  | `_SELFTEST` | `enforce` | `report` |
  | `_CRITERION` | `absolute` | `noise` |
  | `_INDEXER` | `off` | `comm2` |
  | `_NOISE_K`, `_RTOL`, `_SELFTEST_REPEAT`, `_TRACE`, `_POOL` (`compute`), `_EMPTY_CACHE` (0), `_MEM_GUARD_GIB` (2), `_PRIORITY` (-1), `_IDX_MIN_POS` | defaults | defaults |

- **Numerics.** Changes numerics, in the same class as 022: the same math per row, but the NCCL reduction order, the mHC pre-norm GEMM
  split and the MoE/dense GEMM tiling at the half's row count differ, and top-6 expert routing can flip a few tokens on tiny
  differences. The standard quality gate applies: b064 KL5 0.0180, b065 0.0174, b070 0.0183 (the same code measured 0.0170 as b067),
  b072 0.0180.
- **Boot self-test.** On the first eligible overlapped forward the region runs plain, plain again, overlap and overlap again on the
  same inputs and compares the carried states (and the run-to-run noise of each path, because the MoE kernel is not bit-deterministic
  and routing flips make a single layer's output differ by tens of percent between two runs of the same path). The default criterion is
  absolute (RMS-relative error at most 0.10 per state); the recipe uses `noise` (at most the larger of 0.02 and 2 × the run-to-run
  noise, per state and per half). With `_SELFTEST=report`, as shipped, a purely numeric failure is logged with the full report and
  not enforced; structural failures (a rank error, a non-finite state, ranks that disagree, a different collective schedule) still
  refuse to serve. The first version (b053) failed its absolute test at 14 % RMS, which led to the noise-based criterion. The test
  costs about 0.45 GiB above a normal forward (revision 5); we recommend keeping it, because it guards against a
  changed image.
- **Effect.** Same boot, plain → overlap: 5.4-5.7k → 6.6-6.9k tok/s (b064, +21 %); an earlier revision without the sharding gave
  5.5k / 5.6k / 5.4k → 6.5k / 6.7k / 6.4k at 8k / 32k / 128k (b057, +18 %), but without the indexer sharding a 1M prompt took 244 s
  against 200 s in b056; with the sharding on its own communicator a 512k prompt takes 81.6 s (96.4) and 1M 171.7 s (200.0). b070: 6,655 / 6,991 / 6,837 tok/s, 512k 80.6 s, 1M 170.2 s. b072: 6,695 / 6,901 / 6,778 tok/s, 512k 81.4 s,
  1M 172.0 s.
  Decode and concurrency are unchanged. Only 0.5 s of 3.9 s of NCCL time stays exposed (b057).
- **Boot lines.** `[patch-028] prefill two-half overlap ACTIVE (rank r/8): self-test OK on all 8 ranks in X s - ...`,
  `[patch-028] self-test memory (rank N, ...)` on every rank, then a `stats:` line every `_LOG_EVERY` regions. A failed test raises
  `[patch-028] ... BOOT SELF-TEST FAILED ... refusing to serve`.

#### 029 — DSpark top-k Markov bias in one Triton launch (`029-markov-fused.sh`)

- **Problem.** With 013, every one of the 5 sequential draft steps corrects the 2,048 candidate logits of each request by
  `markov_w2[cand] · markov_embed(prev)`; patch 013 does it with torch ops (gather → fp32 copy → batched GEMV), about 0.2 GB of
  temporaries per draft step at 64 requests, about 10 ms of the drafter graph (b042 profile).
- **Change.** Needs 013 and 020 (it pins `dspark.py` as they leave it). Installs `vllm/_markov_fused.py`; one anchored edit in
  `apply_markov_bias_gathered` tries the fused kernel first: one launch per draft step, one program per (request, block of
  candidates), reading about k × 256 × 2 B per request.
- **Switch.** `VLLM_DSPARK_MARKOV_FUSED=1` (the recipe sets it); unset runs patch 013's torch ops.
- **Numerics.** Lossless for the output. The same fp32 operations per candidate except the order of the 256-term sum (cuBLAS GEMV vs a
  Triton tree reduction), so a corrected draft logit can differ in its last fp32 bit; the drafts are still sampled from, and verified
  against, the logits this function writes.
- **Effect.** 2.13 → 0.27 ms per draft step at 64 users (unit test); the 64-user drafter graph 40.8 → 31.4 ms and the step 316 →
  303 ms; 128 users 739 tok/s (b049).

#### 030 — Packed Engram rows (`030-engram-packed.sh`)

- **Problem.** The checkpoint keeps an Engram row in two tensors, 256 B of fp8 values and 8 B of e8m0 scales, 98 GB apart at unaligned
  offsets, so patch 016's reader needs two `O_DIRECT` reads per row. With layer 1 on NVMe as well, each 8,192-token prefill chunk read
  about 25,000 layer-1 rows (two 4 KiB reads each) right after layer 0 and prefill lost 6-8 % (b048). Part of that was not the disk:
  `nvme_verify_pct` re-read 1 % of the rows from the original file before handing them to the GPU, 80-100 ms per chunk.
- **Change.** Needs 016. Replaces two of the files 016 installed (`engram_nvme.c`, rebuilding `libengram_nvme.so`, and
  `engram_nvme_io.py`) by their patch-030 versions (`files/engram_nvme/*.p030.*`); `engram_nvme.py`, the GPU side, stays exactly as
  016 installed it (patch 028 pins its hash). A per-node *pack* written by `tools/engram_pack.py` holds one rank's slice of a layer
  as 512-byte records `[256 B values | 8 B scales | 248 B zero]`, so the reader issues one read per unique row. Build the packs once per
  node (the tool is resumable, verifies against the original shard, and caps its write rate with `--max-mbps`): `engram_pack.py plan |
  pack | verify --model-dir ... --out-dir ... --rank R`; the files are `engram_L<layer>_tp<tp>_r<rank>.pack` and `.pack.json`.
  b054 packed about 49 GB per node, on all 8 nodes in about 2 minutes.
- **Switch.** `ENGRAM_NVME_PACKED_DIR` (the directory, as seen inside the container; unset: the two-read layout), `ENGRAM_NVME_PACKED`
  = `auto` (default when the directory is set: use a valid pack, otherwise fall back and log why) | `require` (the recipe: a missing,
  stale or invalid pack ends the start-up) | `off`; `ENGRAM_NVME_PACKED_SPOT` (default 256 rows compared with the shard at start);
  `ENGRAM_NVME_VERIFY_AFTER=1` (the recipe) moves the `nvme_verify_pct` re-reads behind the publication of the rows: they cost about
  0.3-0.4 ms per verified row, 80-100 ms for 1 % of an 8,192-token request, now off the critical path. A mismatch is still fatal
  (exit status 86), but the GPU may already have used that request's rows. The pack is found by content (layer and row slice), its
  manifest must be complete and the source shard unchanged (size, mtime, header hash).
- **Numerics.** Identical: the delivered bytes are the same either way.
- **Effect.** Prefill at the b042 level with both layers on NVMe, and +4 % at 128k with the indexer split (b054: 5.6k / 5.9k / 5.85k
  tok/s at 8k / 32k / 128k); KV pool 12.2 M tokens at 26 GiB there, 18.8 M at 40 GiB (b070), 20.4 M at the shipped 43.5 GiB (b072).
- **Boot lines.** `[OK] [030] <dir>: N manifest(s); ...`, `[OK] [030] libengram_nvme.so loads (ABI 2, packed-row reader, ...)`,
  `[engram-nvme] layer 1 ... layout=packed` and `layer 14 ... layout=packed`. A line with `layout=two-read` means the packs were not
  used (the readiness rules fail on it).

#### 031 — Scheduler knobs from a control file (`031-sched-ctl.sh`) — inert in this configuration

- **What.** Installs `vllm/_sched_ctl.py` and edits `Scheduler.schedule` in `v1/core/sched/scheduler.py` to poll a control file (rank 0's
  engine core only; at most once a second; a malformed file is refused and the previous values stay): `interval=N`
  (`prefill_schedule_interval`) and `long_prefill=T` (`long_prefill_token_threshold`). Scheduling only: every output stays exact.
- **Switch.** `VLLM_GB10_SCHED_CTL=<file>`. The recipe does not set it, so the hook returns at once.
- **Why it is here.** It is the instrument of a same-boot A/B (b051) with 8 and 16 coding agents and a cold 32k prompt every minute:
  capping a long prompt's chunk at 4,096 tokens while other requests are active gave equal or up to +7 % tokens and equal or better
  time to first token, the long prompt about 1 s later; holding prefill to every fourth step did not help. The result is in the
  recipe as vLLM's own `--long-prefill-token-threshold 4096` together with `--max-num-batched-tokens 8192` (8,192-token chunks alone,
  4,096 when shared; since b054).

#### 032 — Allocator statistics (`032-memstats.sh`) — DIAGNOSTICS ONLY

- **What.** Installs `vllm/_memstats.py` and edits `Worker.compile_or_warm_up_model` in `v1/worker/gpu_worker.py` to start a logger thread.
  Logging only; it changes no computation.
- **Switch.** `VLLM_GB10_MEMSTATS=<seconds>` (the recipe sets 60). Every period each worker logs `[patch-032] periodic: allocated X GiB
  (peak ...), reserved ... GiB, inactive split ..., alloc retries n, cudaMemGetInfo free ...`. `VLLM_GB10_MEMSTATS_HISTORY=1` (not
  set) also records allocation stacks from model load to warm-up and logs the top allocation sites once.
- **Why it is here.** When the patch was written (b051) every worker held about 94 GiB of device memory, about 25 GiB more than
  weights, KV pool and graphs explain, and on GB10 that memory is missing from the OS. The allocation trace (b052) put the routed experts at 45.9 GiB per node against
  34.5 GiB of MXFP4 weights, which led to 033 and 036.

#### 033 — Expert slice zero-padded to 320 (`033-b12x-pad.sh`) — inert in this configuration

- **Problem.** At TP=8 each rank holds all 384 routed experts at intermediate width 2304/8 = 288 = 2 × 128 + 32. b12x stores such a width
  "ceil-tiled" (288 → 384, +33 %: 45.9 GiB instead of 34.5 GiB) and streams the zero tail on every step. A width with n % 128 == 64
  (320) selects b12x's exact compact N64 path.
- **Change.** One anchored edit in `model_executor/layers/fused_moe/oracle/mxfp4.py`: for the b12x W4A8 backend, round the per-rank
  intermediate size up to a multiple of 64 (288 → 320). The extra gate/up rows and `w2` columns are FP4 zeros with zero E8M0 scales,
  so the math is exact.
- **Switch.** `VLLM_GB10_B12X_PAD64=1` (so one patch directory serves both sides of an A/B). The recipe does not set it, and
  must not combine it with 036: the 320-wide slice would keep 036's native 288 path from applying. Nothing else in this directory
  pins 033's result, so removing it would not affect the other patches; it is shipped because the directory is shipped exactly as it
  was measured.
- **Effect (when it was on).** b056 (with 034): experts 45.9 → 38.3 GiB per node, single-user step 31.50 → 29.12 ms (agent turns 126 →
  137 tok/s), 2-32 users +11-15 %, 128 users 817 tok/s, KV pool 15.0 M tokens; prefill paid 1-3 %. The first boot with it (b055)
  failed at its first launch with 8 tokens or fewer, which patch 034 fixes. Replaced by 036 in b061.

#### 034 — b12x compact micro launchers on vLLM's bindings (`034-b12x-compact.sh`)

- **Problem.** For batches of up to 8 tokens (single-user decode: 5-6 rows per step) b12x's compact W4A8 path uses micro kernels that
  read `binding.compact_launches`. Only b12x's own prepared-state bind sets that field; vLLM plans with `fused_moe.plan` and binds
  with `fused_moe.bind`, which leaves it `None`, so the compact path failed at its first launch (the b055 boot).
- **Change.** Edits `model_executor/layers/fused_moe/b12x.py` (three anchored edits): `_p034_compact_launches(plan)` calls b12x's own
  builder once per plan (cached by plan identity; it returns `None` for any plan that is not compact micro); `B12xExperts._plan`
  builds them when the plan is created (never during CUDA capture, because the builder JIT-compiles); `_run_b12x_moe_plan`
  attaches them with `dataclasses.replace`, as b12x's own bind does. In this configuration it is what makes 036's 288-wide path run
  at decode sizes.
- **Switch.** None; always applied. `VLLM_GB10_B12X_COMPACT_MICRO=0` (not set) would plan b12x's dynamic compact M16 kernels instead of the
  micro kernels for M ≤ 8.
- **Numerics.** Identical: it only attaches launchers b12x itself builds.
- **Boot line.** `[patch-034] b12x compact W4A8 micro launchers built (n=288, rows=..., topk=...)` (up to four times).

#### 036 — b12x native compact path for n % 128 == 32 (`036-b12x-n32.sh`)

- **Problem.** See 033. Padding to 320 stores 2.61 MB per expert and rank and still reads zeros; the exact 288 needs 2.35 MB (ceil-tiled
  384: 3.13 MB).
- **Change.** Edits the FlashInfer `b12x` package inside the image (`flashinfer/experimental/b12x`, override with `B12X_ROOT`): 11 files,
  44 anchored edits, the sha256 of every original pinned. One predicate, `b12x._lib.env.w4a8_compact_tail_rows(n)` (64 for n % 128 ==
  64, 32 for n % 128 == 32 when the variable is set, else 0), replaces the `n % 128 == 64` tests of weight preparation, planning,
  scratch sizing and launch, so prepare, plan, policy and launch agree; the kernels take a parametric tail tile (a last N128 tile of
  32 rows in FC1, a K tail of one K32 block in FC2; the other blocks are zero in shared memory and multiply to exact +0.0). A guard in
  `b12x_moe_fp4` raises if the weights' layout and the variable disagree. MXFP4-CSF A8 preparation refuses the new width instead of
  mis-laying it out. The installer ends with a predicate self-test.
- **Switch.** `B12X_COMPACT_N32=1` (the recipe), together with `--moe-backend b12x`. Without the variable every b12x decision is the stock
  one. Changing any b12x source file changes the package fingerprint that keys b12x's object cache, so every b12x kernel compiles
  once more after installing.
- **Numerics.** Exact arithmetic. Bitwise equal to the 320-padded path at decode sizes and within its own run-to-run noise above (b12x is
  not bit-deterministic run to run); quality b061: KL5 0.0175, GSM8K 49/50, HumanEval+ 18/20, tool calls 19/24, needles 12/12.
- **Effect (b061).** Experts 38.3 → 34.5 GiB per node, which went into the KV pool (16.9 → 18.8 M tokens, +11 %); expert reads per step
  down about 10 %: MoE 11.2 → 10.4 ms at one user and 101 → 86 ms at 64 users; single-user step 29.12 → 28.48 ms; 64 / 96 / 128 users
  682 / 783 / 852 tok/s (64-user step 281 → 255 ms). Prefill unchanged.
- **Boot line.** `[OK] [036] b12x native compact n32 edit in place (inert until B12X_COMPACT_N32=1)`.

#### 037 — 64-row split kernels for large launches on the compact path (`037-b12x-compact-prefill.sh`)

- **Problem.** b12x's compact W4A8-MX path runs every dynamic launch (9 or more tokens) on 16-row tiles; the stock ceil-tiled path moves to
  64-row tiles once the average routed rows per expert reach 36, because at 128 rows per expert (an 8,192-token chunk) a 16-row tile
  re-reads the expert's weight tile 8 times. Only the policy tied the compact layout to 16-row tiles.
- **Change.** Applies on top of 036 (it pins the 036-patched files). A compact dynamic launch takes the 64-row tile when the average
  routed rows per expert reach the threshold; plan, scratch sizing and kernel build already follow the planned tile. Nothing in the
  kernels changes. Same package-fingerprint note as 036.
- **Switch.** `B12X_COMPACT_PREFILL_TILE_M=64` (0 or unset: off) and `B12X_COMPACT_PREFILL_MIN_ROWS_PER_EXPERT` (default 36 = the stock
  crossover; 0 = every dynamic launch); the recipe sets 64 and 16. Launches below the threshold, the micro path (M ≤ 8) and CUDA-graph
  decode sizes keep their kernels.
- **Numerics.** Identical according to our checks: a tile policy only (the kernels already took 64-row tiles).
- **Effect.** Bench: the target's launch at 8,192 tokens takes 0.815× the 16-row time (0.905× stock's path), the drafter's 0.781×. In the
  engine under the overlap each expert launch is a 4,096-row half, where the bench shows about 1 %: prefill +0.5 % (b065), a 512k
  prompt 81.6 → 79.9 s and a 1M prompt 171.7 → 168.7 s. Kept because it is exact and free.

#### 038 — DSpark drafter gathers per-rank top-k candidates (`038-dspark-topk-gather.sh`)

- **Problem.** With 013 the drafter needs only the top-2,048 base logits of each of its requests × 5 draft positions. Stock vLLM
  computes the vocab-parallel head on every rank, all-gathers the full vocabulary (129,280 bf16 per row), copies it into rank-major order
  and takes `torch.topk` over it: on the 8 × 100 GbE fabric that all-gather runs at line rate (torch profile of b056 at 64 users: 6.10
  ms for 82.8 MB, plus about 0.8 ms of layout copy and 1.46 ms of top-k kernels per step).
- **Change.** Needs 013, 020, 024 and 026 (it pins `speculator.py` as they leave it). Installs `vllm/_dspark_topk_gather.py`; one
  anchored edit in `_sample_sequential_topk`. Every rank takes the top-k of its own shard, the ranks all-gather only those candidates
  (one int32 per candidate: the bf16 bits of the logit in the high half, the shard-local index in the low half; 4× less traffic), and every
  rank selects the global top-k from the 8 × k candidates. An element outside its shard's top-k has at least k elements ahead of it, so it
  cannot be in the global top-k; the result (values, ids, order) is the one `torch.topk` gives over the gathered vocabulary.
- **Switch.** `VLLM_DSPARK_TOPK_GATHER=1` (the recipe; `keys` makes the tie order canonical by construction at about 4× the traffic);
  `..._MIN_ROWS` (default 16; the recipe sets 5, so a single user's 5 draft rows already use it) and `..._CHECK=N` (default 0; the recipe
  sets 4: a shadow check that also runs the stock path on the first capture warm-ups and the first N eager calls and compares values,
  ids and order bit for bit). Set the same values on every rank: a rank without the hook issues a different collective and the group
  hangs. It falls back to the stock path with one log line when its conditions do not hold (not V4.1 DSpark, TP 1, a padded vocabulary,
  shard-local indices that do not fit 15 bits).
- **Numerics.** Identical to the stock path, bit for bit (bf16 ties resolve in the canonical order); in any case it only changes how the
  draft candidates are produced, so the output distribution cannot change. Tests: 126/126 single-GPU cases and 24/24 eight-GPU
  cases bit-identical.
- **Effect.** The replaced section per step: 1 user 0.23 → 0.17 ms, 4 users 1.22 → 0.27, 12 users 2.28 → 0.57, 64 users 9.52 → 2.99, 128
  users 18.11 → 5.60 ms. Measured: 64 / 128 users +1.7 / +3.7 % (b066).
- **Boot line.** `[patch-038] DSpark top-k gather ACTIVE (mode 1): shard 16160 rows (...)`.

#### 040 — Skip 026's chain where it cannot be used (`040-mv-skip.sh`)

- **Problem.** Patch 026's masks apply only to steps that pass its eligibility test (one request, pure decode). But its chain
  confidences run inside the DSpark drafter graph, which vLLM captures once per batch size, so the graphs for 2 to 256 requests carried the
  chain too: per draft position a gather of 2,048 Markov rows, an fp32 GEMV and an argmax, then the confidence head: about 9.6 ms of a
  33.7 ms drafter graph at 64 users (it grows with the batch), about 0 at one user.
- **Change.** Edits `vllm/_mv.py` (installed by 026; 040 pins its hash): `chain_confidence()` returns before the chain when `num_reqs >
  VLLM_GB10_MV_MAX_REQS`, so the larger graphs are captured without it.
- **Switch.** `VLLM_GB10_MV_SKIP_INELIGIBLE=1` (the recipe), read once at import; without it the module behaves exactly as 026 left it.
- **Numerics.** Identical: the confidences only choose which drafts are verified, are never read for an ineligible step, and the cap is
  fixed before the drafts are drawn.
- **Effect.** 64 / 128 users +3.5 / +3.9 % on b066 and 32 users +5 %; the head node keeps 0.5 GiB more free memory (b067).
- **Boot line.** `[OK] [040] mv-skip edit in place`.

#### 042 — Allocator snapshot on demand (`042-memsnap.sh`) — DIAGNOSTICS ONLY

- **What.** Edits `vllm/_memstats.py` (installed by 032; 042 pins its hash) so that, when a trigger file exists
  (`VLLM_GB10_MEMSNAP_TRIGGER`, default `/cache/huggingface/dsv41-memsnap.trigger` inside the container, i.e. in the node's cache
  directory), the periodic thread deletes it and logs one summary of `torch.cuda.memory_snapshot()` grouped by (stream, pool).
  Costs nothing until triggered; any error ends only the diagnostics thread. Needs `VLLM_GB10_MEMSTATS`; logging only.
- **Use.** `touch <cache directory>/dsv41-memsnap.trigger` on the node you want to inspect; the summary appears in that rank's log
  within one `VLLM_GB10_MEMSTATS` period.
- **What it showed.** On b070, after 64 and 128 users the default pool held 88.4 GiB with only 1.6 GiB unused, and the CUDA-graph pool
  held 3.5 + 0.3 GiB: the decode graphs' private activation memory for the largest captured batch (768 tokens), dominated by the decode
  sparse indexer's full-width fp32 logits (768 rows × 1,048,576 context = 3.0 GiB). The allocator held 4.5-6 GiB more than the live
  tensors at steady state (reserved 92.1 vs allocated 86.1 GiB on rank 0 of b067). That graph pool is what patch 043 shrinks: the same
  snapshot on b072 (ranks 0 and 1 identical, after the concurrency runs) shows the graph pool at 0.68 GiB and the default pool at
  89.86 GiB reserved and 88.21 GiB allocated.

#### 043 — Graph-pool shrink: decode indexer rows in chunks, one capture stream (`043-graphpool.sh`)

- **Problem.** The decode CUDA graphs keep a private memory pool for the life of the engine, and on GB10's unified memory that is
  memory the KV cache cannot have. On b070 each rank's graph pool held 3.50 GiB, plus 0.29 GiB in a second pool of the DSpark
  drafter's graphs (patch 042's snapshot after 64 and 128 users). Almost all of the 3.50 GiB belongs to the decode sparse indexer in the
  largest graph (768 rows: 128 requests × 6 tokens, one row per token, the flattened layout vLLM uses here): DeepGEMM's paged-MQA
  logits kernel allocates a `[rows, max_model_len / compress_ratio]` fp32 tensor, 3 GiB at ratio 1 and a 1,048,576-token window, and the
  two-level indexer's candidate selection adds `[768, 131072]` fp32 scores (384 MiB, layer 20) or uint8 flags (96 MiB, layers 24-36).
  That adds up to 3.46-3.50 GiB. The drafter's graphs sit in a pool of their own because the caching allocator keys the free blocks of
  a shared pool by stream, and every `graph_capture(device)` call opened a new stream.
- **Change.** Two independent levers, both opt-in.
  - *(a) Chunked decode rows.* Installs `vllm/_indexer_decode_chunk.py` and edits `model_executor/layers/sparse_attn_indexer.py` (one
    import and one guarded early return in front of the decode block). For the flattened decode layout the rows are processed in
    chunks whose logits and candidate temporaries stay below the cap. Per chunk: DeepGEMM's own schedule-metadata kernel on the chunk's
    context lengths (computed once per forward context and shared by the layers of one KV-cache group), the paged-MQA logits, candidate
    select and mask, and the top-k into the caller's slice. The chunk's tensors are freed before the next chunk, so the graph pool needs
    one chunk's worth. At a 512 MiB cap the 768-row graph runs as 3 × 256 rows (ratio 2) or 7 × 109-110 rows (ratio 1, with the scores
    or the flags); chunks keep at least 16 rows. Graphs whose unchunked temporaries fit under the cap (below about 114 rows at 512 MiB)
    and every other layout (padded or packed, varlen, DCP, bf16 logits, cooperative or deep-select top-k) run the stock code.
  - *(b) One capture stream.* The module-level `graph_capture(device)` in `distributed/parallel_state.py` uses one cached
    `torch.cuda.Stream` per device instead of a new stream per call, so the drafter's graphs (a separate graph manager with the same
    global pool id) are captured on the main graphs' stream and reuse the blocks those left free. This is safe because both replay on
    one in-order stream and an output that is kept alive is never reused.

  The installer pins `sparse_attn_indexer.py` to the image's file (no earlier patch edits it) and `parallel_state.py` to the file as 018
  leaves it, so it runs after 018 (its anchors do not overlap 018's). It dry-runs every edit before it writes anything, validates the
  settings, and self-tests the installed helper (it plans the 768-row case and leaves small graphs alone).
- **Switch.** `VLLM_GB10_INDEXER_DECODE_CHUNK_MB=<cap in MiB>` (1 to 65536; unset or 0 = off; the recipe sets 512) and
  `VLLM_GB10_SHARED_CAPTURE_STREAM=1` (the recipe). Both are read once, and a value the installer cannot parse stops the start. Not
  set by the recipe: `VLLM_GB10_INDEXER_DECODE_CHUNK_SHARE_META` (default 1: the chunk schedules are computed once per forward context and
  shared; 0 recomputes them per layer, a switch for bisecting) and `VLLM_GB10_INDEXER_DECODE_CHUNK_SCOPE` (default `all`; `cand` is the
  safe subset: logits and top-k run exactly as stock and only the candidate kernels are chunked, which saves about 0.4 of the 3.5 GiB).
- **Numerics.** Identical where read, with one caveat about the order of the final top-k. Every kernel on the path works per row (the
  paged-MQA logits per row and key, the candidate kernels one row per program, the top-k per-row selection); the chunk's schedule only
  decides which SM does the work. What the GPU tests found on 28 October with the engine down:
  - DeepGEMM's logits per row, chunked against the whole batch: bit-identical in 36 of 36 cases (rows 6 to 768, widths 524,288 and
    1,048,576).
  - The full eager decode path: 48 of 48 cases. The final top-k is done by the persistent top-k kernel, which assigns its output slots with
    atomics, so its result is not order-reproducible even for stock against stock. The chunked path reaches exactly the class that stock
    reaches against itself: equal logits at the selected indices in the eager test (the weakest of the test's three classes: same
    order, same index set, same values), and the same index set per row in the graph tests below. Bit-identity of the top-k's order
    does not exist for any change that alters scheduling; the sparse attention over the selected set then sees only floating-point
    summation-order differences.
  - `torch.topk` of the candidate selection (layer 20) showed no sensitivity to the row count for 1 to 256 rows (exact). Chunks keep
    at least 16 rows anyway, because PyTorch switches algorithm with the row count.
  - Inside CUDA graphs with inputs and schedule updated in place: the schedule kernel captured on a row slice replays exactly with new
    context lengths; the chunk logits with the in-graph schedule had 0 bad rows on the second replay; the whole chunked op is set-equal
    to eager chunked and to eager stock on every row, with schedule sharing on and off. A first graph-mode pass that morning had failed
    10 of 107 checks; the debug session found them to be faults of the test (the two variants drew different random inputs before the
    second replay, the expected launch counts were wrong, tie cases need the comparison by logit values), not of the patch.

  A one-time eager cross-check recomputes the whole-batch schedule with the block size taken from the KV view and the attention
  builder's SM count, and compares it with the builder's. On a mismatch chunking switches itself off for good and logs `[patch-043] stock
  path ...`. The readiness rules fail the start on that line, because the 43.5 GiB KV pool assumes the shrunken graph pool.
- **Effect.** Graph pool 3.79 → 0.68 GiB per rank (patch 042's snapshot after the concurrency runs, ranks 0 and 1 identical; before:
  3.50 + 0.29 GiB; the drafter's separate pool is gone with lever (b)). In the eager GPU test the 768-row decode needed 3.10-3.40 GiB
  of temporaries before and 0.39-0.57 GiB after. With 045 this paid for a KV pool of 43.5 GiB (46,707,769,344 bytes), 20,411,335
  tokens against b070's 18,769,055 (+8.8 %). The cost in throughput against b071 is about 1 % at 64 users and nothing measurable at 1 or 128
  users: in the b072 chain 64 users gave 713.5 / 714.9 and 128 users 908.6 / 910.9 tok/s (two runs each), against the means of two
  runs of 722.7 / 905.8 for b071 and 719.2 / 916.5 for b070 (60 s windows; repeated runs on one boot differ by about 2 %). Against b070 the three b072 runs average 714.7 tok/s at 64 users (b070: 720.3, -0.8 %) and 906.2 at 128 users (b070: 919.9, -1.5 %). The boot took 358 s (b070: 378 s). In the worst-case memory order (a 1M-token prompt right after
  128 users and 16 coding agents) the head node's lowest MemAvailable was **8.08 GiB** and the other nodes' 9.9-11.9 GiB. The rule of
  this recipe is at least 8 GiB, so the margin on the head node is **0.08 GiB**: nothing else that uses memory fits next to a 43.5 GiB
  pool. Quality on b072: KL5 0.01796, top-1 agreement 0.9548, GSM8K 49/50, HumanEval+ 19/20, tool calls 19/24, needles 12/12; no error
  line in the log.
- **Boot lines.** At container start: `[OK] [043] installed _indexer_decode_chunk.py`, `[OK] [043]
  model_executor/layers/sparse_attn_indexer.py patched (sha256 ...)`, `[OK] [043] distributed/parallel_state.py patched (sha256 ...)`,
  `[OK] [043] helper self-test: cap 512 MiB, 768 rows -> 7 chunks of 109..110 rows (at 512 MiB), share_meta=True` and `[OK] [043]
  graph-pool edits in place (chunking 512, shared capture stream 1)`. While the decode graphs are captured: `[patch-043] decode
  indexer chunked (first use: 768 rows x width 524288) -> 3 chunks of 256..256 rows (cap 512 MiB, schedule shared per forward)` and
  two more lines for width 1048576 (with candidate scores, with candidate flags) with 7 chunks of 109..110 rows. A `[patch-043] stock
  path ...` line means chunking fell back. vLLM also prints `Unknown vLLM environment variable detected: VLLM_GB10_SHARED_CAPTURE_STREAM`,
  which is harmless noise.

#### 045 — Switchable token embedding (`045-embed-tp.sh`)

- **Problem.** Patch 015 replicates the token embedding on every rank so that the decode graphs do not start with an all-reduce. The
  graph-start wait did not go away (it belongs to NCCL), and the replica costs 1.07 GiB of memory per rank (the 1.16 GB difference to
  the vocab-parallel layout; the patch's docstring writes "GiB" for it), about 0.5 M KV tokens.
- **Change.** One anchored edit in `models/deepseek_v41/nvidia/model.py` as the whole patch set up to 028 leaves it (028 is the last patch
  that edits that file, so 045 runs after it): 015's literal `disable_tp=True,` becomes an expression of an environment variable that
  is read once, when the model is built. With the variable unset or 0 the module is 015's, byte for byte.
- **Switch.** `VLLM_GB10_EMBED_TP=1` (the recipe): the embedding is vocab-parallel again, an eighth of the table per rank and one
  all-reduce per forward, as in stock vLLM. 015 stays installed and its installer still runs; it is switched off by this variable. To go
  back to the replicated table, remove `VLLM_GB10_EMBED_TP` from `config/recipe.env`, lower `--kv-cache-memory-bytes` by 1 GiB and run
  the worst-case memory order again (a 1M-token prompt right after a busy period, watching every node's lowest free memory).
- **Numerics.** Identical: each embedding row has exactly one owner in the vocab-parallel layout, and the all-reduce only adds zeros.
- **Effect.** Measured on b071 (b070 + 045 at KV 41 GiB, not served) against b070 in the same chain. "Model loading took" fell from 39.76
  to 38.69 GiB per node. One-user decode did not change: the GPU busy time per step in the profile was 28.04 against 27.96 ms (the
  wall-time medians 28.22 and 28.35 ms), and agent turns made 34.33 against 34.29 tok/s per accepted token per step (speed divided by
  acceptance length). The price is in prefill and at many users: 6,694 / 6,867 / 6,718 against 6,655 / 6,991 / 6,837 tok/s at 8k /
  32k / 128k (-1.0 % in sum; the embedding all-reduce moves 84 MB per 8k chunk), a 1M prompt in 171.7 s (+0.9 %; 512k 80.8 s), 128
  users 907.4 / 904.1 against 916.5 tok/s (b070, mean of two runs; about -1.2 %), 64 users 725.7 / 719.7 against 719.2 (+0.5 %), 16 coding agents 327.1
  tok/s with first token 0.97 s (b070: 330.2 and 0.96 s). In exchange the KV pool can grow by 1 GiB (+0.47 M tokens, +2.5 %). Quality
  on b071: KL5 0.01746, top-1 0.9544, GSM8K 48/50, HumanEval+ 18/20, tool calls 20/24, needles 12/12. Worst-case head-node memory
  at KV 41 GiB: 8.04 GiB (b069 at the same pool with 015 on: 7.2 GiB; b070 at 40 GiB: 8.87 GiB).
- **Boot lines.** `=== Patch 045: switchable token-embedding sharding (VLLM_GB10_EMBED_TP=1) ===`, `[OK] [045]
  models/deepseek_v41/nvidia/model.py patched (sha256 ...)`, `[OK] [045] embed-tp edit in place`. At model load, `Model loading took
  38.69 GiB` (39.76 GiB with 015's replica).

### 3.3 About the file names and the references in the patch files

- Some patch files refer to test suites that are not part of this release (`tests/038/...`, `tests/028-overlap/NOTES.md`, `tests/roce2/NOTES.md`, `tests/025/...`,
  `tests/029/...`, `tests/043/...`). They are not part of this release.
- Patch 030's installer and docstrings refer to `engram-nvme/tools/engram_pack.py`; in this repository that tool is
  `tools/engram_pack.py`. The installer's comment on `tools/mkpatchdir.sh` refers to a helper (not shipped) that flattens `files/`; the installers
  accept both the shipped layout (`files/engram_nvme/...`) and a flat one.
- The numbering has gaps (for example 017, 027, 039 and 041): those are patches that were staged, rejected or kept out of this
  release (section 4), or that belong to the GLM recipe.

## 4. Considered and not shipped

Things that were measured, staged or tried and are not in `patches/`. The numbers named here are the gaps in the numbering
(section 3.3). Numbers are from the builds named.

### Communication

- **003, CUDA blocking sync for the worker processes.** Ported to this vLLM build and passing its 44 CPU checks, but never run on the
  GPUs. It makes host waits sleep instead of spin (`VLLM_GB10_BLOCKING_SYNC=1`), but vLLM's Model Runner V2 already waits for its
  per-step output copy on a blocking CUDA event, which sleeps whatever the context flag says, so the benefit here is unproven.
- **017, our own RDMA all-reduce.** Correct after a fix of the packet sequence numbers, but every GPU → CPU → NIC handoff cost about one
  whole one-shot: 163-263 µs at 6 tokens against 88 µs for patch 012's one-shot. It did show that the graph-start stall is avoidable; the
  RoCEnante runtime (018) then removed most of it (the first collective of each graph fell from 472 to 115 µs, b020).
- **A larger one-shot limit and global NCCL overrides.** Raising `VLLM_GB10_ONESHOT_AR_MAX_BYTES` so that two-request decode steps use the
  one-shot: 127 µs for NCCL against 134 µs at 12 tokens, no gain. `NCCL_ALGO=Tree` was 8 µs faster for the 6-token message in a graph (122
  against 130 µs); `Ring`, `NCCL_PROTO=LL` and fewer channels were not faster. For back-to-back all-reduces of 1-8 MB (16 to 128 users)
  the default (4 channels, NCCL's own protocol choice) is best or tied at every size: 3.9 MB in about 690 µs, 7.9 MB in about 1.25 ms,
  85-90 % of the 100 GbE line rate; 2 or 8 channels are slower and Tree is 2-3× slower. The recipe sets only `NCCL_MAX_NCHANNELS=4`.
- **Cores.** Moving every vLLM thread to the fast Cortex-X925 cores from boot (b036) measured the unpinned numbers (32.17 ms per step);
  moving the other threads off the two-shot proxy's core in a live test on all 8 nodes left the all-reduce at 46 µs. Disabling deep CPU idle states changed
  nothing (b006).

### Drafting and verification

- **Adaptive verification length** (`enable_adaptive_verification`). vLLM can size each step's verification from DSpark's confidence head,
  but on SM12x it is refused (vLLM issue #63343); at boot (b031) the DSV4.1 indexer attention backend cannot trim verification requests
  on the GPU. The verification length is static at 5 drafted tokens. Masked verification (026) is the lossless way this recipe skips
  drafts that will probably be rejected.
- **A per-batch-size speculative length** (`num_speculative_tokens_per_batch_size`): supported for DSpark on the V2 runner (it drafts all 5
  and truncates). A cost model from the 128-user profile puts K=4 and K=3 at about neutral (-1 to +1 %) at 128 users, where chat load
  accepts .78 / .57 / .41 / .30 / .22 per draft position. Not pursued.
- **Copy-aware drafting (027)**, b043 and b044. Where an agent turn repeats its context, propose the tokens that followed the last
  occurrence as a one-hot draft (exact under block verification). The copies are right (72 % of copy steps accept all 5), but steps got
  1.5-2.2 % slower per token: at matched positions DSpark already makes 5.2 tokens per step, at least as many as a copy, and a copy gated
  to where DSpark is unsure yields 3.4. Rejected; not shipped.
- **A fused kernel for 026's confidence chain (039).** Staged; no longer needed after 040 removed the chain from the graphs that cannot
  use it.
- **A long-run cap rule for 026** (maximize tokens - λ·time), b046: +0.9 % in an offline replay, +0.1 % per step live over four alternating
  rounds, inside the run-to-run noise of about 2 %. Not promoted.
- **Sharper or flatter drafts** (b028 and a later check): no gain, s stays 1 (section 3, patch 024).
- **A BF16 draft head** (b039): acceptance not higher (3.92 against 3.96 tokens per step), 0.45 ms more per step. A target lm_head in FP8
  would be lossy (the checkpoint's head is BF16) and was not done.
- **10 drafts per step** (b045). DSpark drafts a non-causal block of 5; asked for 10, its acceptance collapses (1.74 tokens per step at a
  fixed cap of 5), agent turns 93.5 tok/s against 128; a 10-token verification costs 3.1-3.8 ms more per step at the same cap.
- **FLy verification** (vLLM's entropy-gated deferred acceptance, window 2), b033. It accepts a draft the exact test rejected when the target
  is uncertain there: agent turns 129.5 / 132.1 tok/s (b032: 127.8 / 127.7), acceptance 4.35 / 4.44 (4.22 / 4.25), chat decode +9-13 %,
  task scores level within noise. It changes the outputs (no longer exactly the model's distribution), so it is off. The drafter is trained
  for 5 positions, which caps the gain.

### Memory and scheduling

- **malloc_trim after warm-up (041)**, b068. Every worker carries about 3.6 GiB of glibc heap; a `malloc_trim` after start-up
  returned 0.01 GiB per rank, so the heap is live data (compiled-kernel state, Python). Dropped.
- **256 sequences** (b062a, graphs up to 1,536 rows). Capturing the larger graphs took the head node to 5.7 GiB free and a node-level OOM
  guard stopped the engine; 256 sequences need about 4 GiB of the KV pool back.
- **16,384-token prefill chunks** (b059): prefill +1.6 % at 32k, +4.2 % at 128k, +4.8 % at 512k, +7.1 % at 1M, but the same 36 GiB of KV
  holds 14.1 M instead of 16.9 M tokens (vLLM sizes the sliding-window cache groups per scheduled chunk), the CUDA graphs need 1.8 GiB more
  and the head node's lowest free memory fell from 10.7 to 7.1 GiB. Chunks stay at 8,192 (4,096 when shared).
- **KV pool of 41 GiB without the memory patches** (b069, 19.24 M tokens, patch 015's replica and the full-size graph pool still in
  place). Throughput and agents passed, but a 1M-token prompt right after the 128-user and 16-agent runs took the head node to 7.2 GiB
  free for about 20 s (the rule is at least 8 GiB; other nodes 8.0-9.9). At 40 GiB the same order left 8.87 GiB (b070). More KV had to
  come out of the allocator, not from squeezing, and it did: 045 (1.07 GiB per node) allowed 41 GiB with 8.04 GiB at the lowest point
  (b071), and 043 (3.1 GiB of graph pool) with 045 allowed the shipped 43.5 GiB with 8.08 GiB on the head node (b072; the other nodes
  9.9-11.9 GiB). That is a margin of 0.08 GiB over the 8 GiB rule. Do not raise the pool, the sequence count or the chunk size, or add
  a memory user, without running that order again.
- **Batching prefill turns every fourth step** (b051, patch 031): did not help.

### Kernels and settings

- **`--linear-backend b12x`** (b014): a segfault in `cuGraphLaunch` on the first request that left orphaned GPU contexts. `marlin` is used.
- **A Triton FP8 GEMV instead of Marlin**: +2-9 % per GEMM (wq_a + wkv 43.9 against 46.2 µs, wq_b 26.2 against 28.1, wo_a 22.5 against
  23.3, wo_b 26.7 against 28.8) because both stop at 185-210 GB/s; about 0.3 ms per step. Parked.
- **L2 prefetch during all-reduce waits.** The mHC op takes 20.9 µs cold and still 17.0 µs with its weights hot, the router 20.0 against
  10.3; prefetching from a side stream moved neither by more than about 2.5 µs. At most about 0.2 ms per step.
- **mHC split configs beyond 6 and 12 tokens** (patch 025): the unsplit default was fastest at M = 96 to 768.
- **GPU clocks above 2,200 MHz.** 2,300 MHz: decode step 32.14 / 32.05 → 31.61 / 31.91 ms (-1.0 %), prefill +1.1-1.5 % (b034). Stock clocks
  (about 2,418 MHz): single-user agent turns +1.9 % (b040), hottest zone 78.6 °C. Under an 8-user sweep at stock clocks the hottest
  node reached 85.7 °C and a watchdog dropped all nodes back to 2,200 MHz. All numbers in this release are at the 2,200 MHz cap.
