# The serving image

The recipe runs a **pinned vLLM nightly image** with one small build-time source edit for the GB10 GPU (compute capability
12.1, "sm_121") in the DGX Spark. Everything else that the recipe changes happens at container start and is described in
[../PATCHES.md](../PATCHES.md).

| | |
|---|---|
| Base | `vllm/vllm-openai@sha256:27aa6d814d1fc44ff8f13bb50c291cb23f54e15544ace8b699e0b29bdcc9451b`: the arm64 image of the nightly built from vLLM commit `e4a5b5537344` on 25 October 2026 (its tag is `nightly-e4a5b5537344b103ed10de7764378ef09604ef3d`) |
| Inside the base | vLLM 0.32.1rc1.dev466, torch 2.14.0, CUDA 13.0, FlashInfer 0.7.2.post1, NCCL 2.30.7, Triton 3.8.0, Python 3.12, arm64 |
| Result | `dsv41-flash-sm121:n1` (the tag `config/recipe.env` expects) |
| Build inputs | `Dockerfile`, `dsv41_bake.py` (the edit), `.dockerignore` |

The tag `nightly` moves every day. Only the digest above, or the commit tag, names the image the patches were written for.
Do not swap the base for another tag: the `-cu129` images are built for CUDA 12.9 and have no kernels for sm_121, and the
file hashes that the bake and the runtime patches check are those of this exact image.

## Why a nightly build and not the v0.32.0 release

According to upstream reports, the released v0.32.0 (18 October 2026) does not start DeepSeek-V4.1-Flash-1017 on GB10-class
GPUs (we did not try it):

- It ships FlashInfer 0.7.0.post1. FlashInfer's sparse-MLA support for DeepSeek-V4.1 on SM120/SM121 arrived in 0.7.1; with
  0.7.0.post1 the engine stops at start-up with `page_block_size=32 is unsupported; decode kernels are instantiated only
  for page_block_size=64` (reported upstream in the thread of vLLM PR #60762).
- It does not contain vLLM PR #62199, merged on 19 October, which makes the prompt encoder match DeepSeek's 1017 reference
  (a turn that holds only tool results no longer starts a new user turn).

The nightly of 25 October has FlashInfer 0.7.2.post1 and both changes. It is not a release, which is why it is pinned by
digest. Once a vLLM release contains these changes and PR #60762 you can move to it, but every runtime patch pins the
sha256 of the files it edits, so each one has to be re-validated first ([../PATCHES.md](../PATCHES.md), section 1.6).

## What the build-time edit does

One change, two files: the backport of vLLM PR #60762 ("declare 64-token kernel blocks for SM120 sparse backends"). It was
still open upstream on 25 October; the test reports in its thread are from SM120 GPUs (RTX class), and this recipe runs it on
SM121.

| Edit | File (under `vllm/`) | Problem on GB10 | Change |
|---|---|---|---|
| `dsv41_sparse_blocks` | `models/deepseek_v41/nvidia/flashinfer_sparse.py` | The FlashInfer sparse-MLA backend for DeepSeek-V4.1 declares 128-token kernel blocks. The SM12x decode kernels take 64-token pages | Declare 64 on compute-capability family 12x (128 elsewhere) |
| `dsv41_indexer_blocks` | `v1/attention/backends/mla/indexer.py` | The V4.1 indexer backend declares 128-token blocks. DeepGEMM's SM12x paged-MQA kernel accepts only 32 or 64 | Declare 64 on family 12x (the earlier values elsewhere) |

Without the edit the engine stops while it negotiates the KV block size: `No common block size for 64
(FLASHINFER_MLA_SPARSE_DSV41: [128])` when `--block-size 64` is given (the recipe gives it), or an assertion in DeepGEMM,
`block_kv == 32 or block_kv == 64`, when it is not. `launch/readiness.txt` fails the boot on both lines.

The edit only changes which page size the backends offer, so the KV cache is laid out in 64-token blocks; the kernels are
unchanged. Every edit must match its anchor exactly once in a file whose sha256 is pinned (`PINS` in
`dsv41_bake.py`); otherwise the build fails and nothing is written. The bake is idempotent: running it again on a baked
image changes nothing and reports `already applied`.

The Dockerfile also sets a few environment defaults the recipe relies on: `VLLM_ENGINE_READY_TIMEOUT_S=3600` (the
default of 600 s leaves little margin: the current recipe's boot took 358 s with warm kernel caches, the first boot of a node
takes longer, and vLLM's own recipe for this model sets 3600 too), the CUDA arch lists for `12.1a`, and the vLLM
usage-statistics opt-out (`VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`). `docker run -e` overrides any of them; the rest of the
container environment comes from `config/recipe.env` and the launcher.

## Build it

On the head node (NODE1), from the repository root:

```bash
image/build.sh                 # docker build + the check that the edit is in the image
image/build.sh --distribute    # ... then copy the image to NODE2..NODE8 and compare the image IDs (uses cluster.env)
```

The build needs no network beyond pulling the base by digest (10.9 GB compressed), no GPU and no pip. `--distribute` streams
the image to the seven other nodes one after the other (`docker save | ssh ... docker load`), which keeps the image ID.
`launch/up.sh` refuses to start unless all eight nodes report the same image ID, so do not build separately on each node:
two builds of one Dockerfile get different IDs.

Size: 37.6 GB unpacked on disk (`docker images`), 10.9 GB of stored layers.

## How to tell that the image is right

- **Build.** `image/build.sh` prints one `[dsv41-bake] [OK] <file>: <edit> (already applied)` line per edited file and
  stops with an error if any line says `applied now`.
- **Versions.** This prints the four versions that matter, without a GPU:

  ```bash
  docker run --rm --entrypoint python3 dsv41-flash-sm121:n1 -I -c "import importlib.metadata as m
  for p in ('vllm', 'torch', 'flashinfer-python', 'nvidia-nccl-cu13'):
      try: print(p, m.version(p))
      except m.PackageNotFoundError: print(p, 'not found')"
  ```

  Expect `0.32.1rc1.dev466+ge4a5b5537`, `2.14.0+cu130`, `0.7.2.post1` and `2.30.7`.
- **Edits.** `docker run --rm --entrypoint python3 dsv41-flash-sm121:n1 -I /opt/dsv41-bake/dsv41_bake.py --list` lists the two
  edits; `/opt/dsv41-bake/bake-manifest.json` in the image records the pristine and the baked sha256 of each file.
- **Boot.** A missing or failed edit shows up in rank 0's log as one of the two messages above, and the launcher stops the
  wait. A wrong base shows up earlier: the first runtime patch refuses to start when its target file has a different hash
  (`[recipe-entrypoint] FATAL`).

## Credits

The block-size change is the work of the author of vLLM PR #60762 and of the people who tested it in that thread. The
FlashInfer sparse-MLA decode path for SM120 and SM121 that makes it useful is the FlashInfer team's work (0.7.1 and 0.7.2).
