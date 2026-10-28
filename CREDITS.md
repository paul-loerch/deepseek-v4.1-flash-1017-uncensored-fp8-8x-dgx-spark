# Credits

This recipe stands on other people's work. Thank you.

## Model and weights

- **DeepSeek**: DeepSeek-V4.1-Flash-1017 (MIT) and what makes it run on eight small nodes at all: the compressed sparse
  attention that keeps the KV cache to about 2 KB per token, the Engram n-gram memory, the DSpark draft blocks that ship inside
  the checkpoint, and the reference encoder behind vLLM's prompt rendering.
- **drowzeys**: the Keys overlay (`drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated`, MIT, gated). It changes 26 attention
  output projections with a rank-1 edit, which makes a 1.1 GB sidecar instead of a second 476 GB checkpoint. That is why a
  merge on your own disk is possible at all.
- **dealignai**: `dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8`, a full abliterated derivative of the same model, in the
  same format and with the same tensor shapes. We compared it with the overlay on one engine, and several of the quick speed
  tests quoted in these docs ran on it.

## Software

- **vLLM** and its contributors: the engine, and the DeepSeek-V4.1 support this recipe builds on (PR #56214 by zyongye and
  the pull requests around it), the Engram implementation and `--engram-config` (PRs #54371 and #56512), DSpark speculative
  decoding, the sparse-attention and indexer backends, and the 1017 prompt-encoder fix (#62199). The build-time edit in
  `image/` carries the change of PR #60762, "declare 64-token kernel blocks for SM120 sparse backends", with thanks to its
  author and to the people who tested it on SM120 in its thread.
- **FlashInfer**: the sparse-MLA decode kernels for DeepSeek-V4.1 on SM120 and SM121 (0.7.1 and 0.7.2), and the planners
  tuned for GB10's 48 SMs. Its experimental **b12x** package supplies the fused MoE kernels that patches 034, 036 and 037 extend
  (native-width experts, 64-row prefill tiles) and the RoCE communication runtime ("RoCEnante", Apache-2.0) on which patch 018's
  one-shot collectives run and from which its two-shot all-reduce and CPU proxy are derived.
- **DeepGEMM** (vLLM's fork): the MXFP4 expert and paged-MQA kernels that the automatic MoE choice and the indexer run on; patch 043 calls its schedule-metadata kernel for every chunk of decode rows.
- **NVIDIA NCCL** and the DGX Spark platform software.
- **Anemll** and MiaAI-Lab's DSpark recipe: the GB10 spin-wait setting that patch 001 carries.
- **TensorFold** (ashhart/TensorFold) and its `ENGRAM_PACKED` option: the idea of storing an Engram row's values and scales
  side by side so that one read fetches a row, which `tools/engram_pack.py` and patch 030 implement for vLLM's NVMe reader.

## Community recipes and benchmarks

We read the public DeepSeek-V4.1 recipes for DGX Spark clusters, and checked our memory plan and our settings against them.
Among others: sumsliu/dgx-spark-deepseek-v41 (an 8-node tensor-parallel deployment, and the report behind vLLM issue
#57156), rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8, im0xMagnus/deepseek-v4.1-flash-uncensored-8x-dgx-spark,
maherajahh/DeepSeek-V4.1-Flash-1017-8x-DGX-Spark, tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks and MiaAI-Lab's
DeepSeek-V4.1 Spark recipes. The benchmark conventions here (state the prompt class, salt every cold prompt, publish raw
results) are the ones of our GLM-5.3-Flash recipe and of the DGX Spark community around it.

## What is original here

- Running DeepSeek-V4.1-Flash-1017 at **tensor parallel 8 across eight Sparks** with the full 1,048,576-token context and up
  to 128 requests, and the memory plan that makes it fit: experts at their native width, Engram layers 1 and 14 read from
  NVMe, a pinned 43.5 GiB KV pool made possible by a small CUDA-graph pool, decode-only CUDA graphs sized for DSpark, and the NCCL and host settings of the multi-node setup.
- `tools/build_overlay_variant.py`: merging the overlay with hardlinks, byte-for-byte verification and a build manifest.
- The GB10 runtime patches ([PATCHES.md](PATCHES.md)): for decode at TP 8 the collectives for small messages (one-shot over
  NCCL, RoCE one-shot and a two-shot all-reduce reduced on the Grace CPU), masked verification, the drafter's candidate gather,
  and the native-width b12x expert path; for prefill the sequence-parallel mHC with the two-half communication/compute overlap
  and the Engram all-to-all; the NVMe-served Engram rows with their packed layout; and the CUDA-graph pool shrink (decode indexer
  rows in chunks, one shared capture stream).
- The memory discipline around it: the measured worst-case order, the 8 GiB headroom rule, and the launcher's checks.
- The comparison of the official weights, a full abliterated derivative and the overlay on one engine with identical
  requests.
