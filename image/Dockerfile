#
# GB10 image for DeepSeek-V4.1-Flash-1017 on vLLM (arm64, CUDA 13.0, FlashInfer 0.7.2.post1) -- DGX Spark.
#
#   build   docker build -t dsv41-flash-sm121:n1 -f Dockerfile .        (context = this directory, see .dockerignore)
#   check   docker run --rm --entrypoint python3 dsv41-flash-sm121:n1 -I /opt/dsv41-bake/dsv41_bake.py --dry-run
#           (prints "already applied" for every file: the bake is in place)
#
# What it does: python3 dsv41_bake.py edits vLLM SOURCE files in site-packages as text (every anchor must match
# exactly once, sha256 pins, nothing is written if anything is off). No network, no pip, no GPU at build time.
# The base is the vLLM nightly of 2026-10-25 by digest: vLLM 0.32.1rc1.dev466+ge4a5b5537 (main e4a5b55), torch 2.14.0,
# FlashInfer 0.7.2.post1, NCCL 2.30.7. The released v0.32.0 lacks what DeepSeek-V4.1 needs on GB10 (FlashInfer >= 0.7.1
# runtime page sizes, the 1017 encoder fix #62199) -- see IMAGE.md.

ARG BASE_IMAGE=vllm/vllm-openai@sha256:27aa6d814d1fc44ff8f13bb50c291cb23f54e15544ace8b699e0b29bdcc9451b
FROM ${BASE_IMAGE}

COPY dsv41_bake.py /opt/dsv41-bake/
RUN python3 -I /opt/dsv41-bake/dsv41_bake.py

# Serving defaults the recipe relies on without passing them itself. All are overridable with `docker run -e`.
ENV VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1
