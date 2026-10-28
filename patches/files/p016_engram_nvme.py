# SPDX-License-Identifier: Apache-2.0
"""Patch 016 -- Engram tables on node-local NVMe for DeepSeek-V4.1 (applied by xpatch.py).

WHY: the two Engram tables are 188.8 GiB; at TP=8 every rank keeps 23.6 GiB of them resident (11.8 GiB per layer). On the
8x GB10 cluster that memory is the same DRAM as the KV cache, and vLLM's own cpu_offload (pinned host + UVA) saves nothing
there. The rows a step needs are known as soon as the hash kernel has run (24 hash heads x 2 layers per token, 3 heads per
rank), and layer 14's are needed ~11-16 ms later, layer 1's ~1 ms later. This patch serves the rows of chosen layers
(default: layer 14) from the checkpoint shards on each node's own NVMe: the GPU hands the row ids to a native reader thread
through pinned memory (a doorbell kernel inside the CUDA graph), the reader reads the 256 B values and 8 B scales with
O_DIRECT (Linux native AIO; pread thread pool as fallback; no page cache, no mmap), stages them in pinned memory, and a wait
kernel in the graph releases the dequant kernel, which runs the same arithmetic as the resident lookup. Rows are
bit-identical; a failure of any kind ends the process (fail closed).

WHAT (anchored edits in three files; the resident _engram_lookup_kernel and every other code path stay untouched):
  config/engram.py                                  EngramConfig: nvme, nvme_layers, nvme_io_backend, nvme_queue_depth,
                                                    nvme_threads, nvme_poll, nvme_keepwarm_us, nvme_keepwarm_idle_s,
                                                    nvme_timeout_s, nvme_verify_pct, nvme_dedupe, nvme_cpu, nvme_path;
                                                    validation; nvme_layers_for(); the nvme_* fields are left out of
                                                    compute_hash() while nvme is off (patched-but-off == unpatched)
  models/deepseek_v41/common/engram.py              ParallelEngramEmbedding(nvme=...): zero-row parameters, no table;
                                                    Engram: _nvme channel (created in __init__), prepare_embeddings ->
                                                    publish, _ready_rows -> fetch (wait + dequant)
  model_executor/model_loader/ep_weight_filter.py   should_skip_weight: the NVMe-served layers' embed.weight/embed.scale
                                                    are never loaded (the small wkv/q/k tensors of those shards still are)
The wrapper 016-engram-nvme.sh also installs files/engram_nvme/{engram_nvme.py, engram_nvme_io.py, engram_nvme.c} into
models/deepseek_v41/common/ and builds libengram_nvme.so there (gcc, raw syscalls only: no libaio, no io_uring).
Inert unless the engram config sets "nvme": true. model.py is not edited (patch 015 pins its original sha).
"""
ID = "016"
TAG = "engram-nvme"
M = f"# [patch-{ID} {TAG}]"

CFG_SHA = "c9247c5bd690de50bb2001d4b35117a740910d88334791793ddc05cc8cca2ba0"

CFG_FIELDS_OLD = (
    "    Always on for private tables on ROCm.\"\"\"\n"
    "\n"
    "    @model_validator(mode=\"after\")\n"
    "    def _validate_shared_memory(self) -> Self:\n"
)
CFG_FIELDS_NEW = (
    "    Always on for private tables on ROCm.\"\"\"\n"
    "\n"
    f"    {M} tables served from node-local NVMe\n"
    "    nvme: bool = False\n"
    "    \"\"\"Read the Engram rows of the layers in `nvme_layers` straight from the checkpoint's safetensors shards on node-local\n"
    "    NVMe (O_DIRECT, native reader thread, rows staged in pinned memory) instead of keeping those tables resident. The rows\n"
    "    are byte-identical to the resident path. Frees the table's memory (11.8 GiB per rank and layer at TP=8). Requires\n"
    "    cpu_offload=false, the V2 model runner, tensor parallelism only (no PP, no Engram DP group).\"\"\"\n"
    "\n"
    "    nvme_layers: list[int] | None = None\n"
    "    \"\"\"Model layer ids whose table is read from NVMe, a subset of the model's engram_layer_ids. None: layer 14 only.\"\"\"\n"
    "\n"
    "    nvme_io_backend: str = \"auto\"\n"
    "    \"\"\"auto | aio | threads. aio is Linux native AIO with O_DIRECT (allowed by Docker's default seccomp profile); threads is\n"
    "    a pread thread pool that auto falls back to when AIO is unavailable.\"\"\"\n"
    "\n"
    "    nvme_queue_depth: int = 256\n"
    "    \"\"\"Reads in flight (aio).\"\"\"\n"
    "\n"
    "    nvme_threads: int = 64\n"
    "    \"\"\"Pool size of the threads backend.\"\"\"\n"
    "\n"
    "    nvme_poll: str = \"auto\"\n"
    "    \"\"\"auto | spin | sleep: how the reader waits for the next request.\"\"\"\n"
    "\n"
    "    nvme_keepwarm_us: int = 500\n"
    "    \"\"\"While requests keep coming, one 4 KiB O_DIRECT read every this many microseconds keeps the SSD and its link out of\n"
    "    their low-power states (a cold device answers a decode-sized batch in ~1.0 ms instead of ~0.4 ms). 0 disables it.\"\"\"\n"
    "\n"
    "    nvme_keepwarm_idle_s: float = 5.0\n"
    "    \"\"\"The heartbeat stops after this long without a request.\"\"\"\n"
    "\n"
    "    nvme_timeout_s: float = 30.0\n"
    "    \"\"\"A request pending this long, or a GPU wait that is not served, ends the process (fail closed).\"\"\"\n"
    "\n"
    "    nvme_verify_pct: int = 0\n"
    "    \"\"\"Re-read this percentage of the delivered rows through a buffered pread and compare (debugging; fatal on mismatch).\"\"\"\n"
    "\n"
    "    nvme_dedupe: bool = True\n"
    "    \"\"\"Read each distinct row of a request once.\"\"\"\n"
    "\n"
    "    nvme_cpu: int = -1\n"
    "    \"\"\"Pin the reader thread to this CPU (-1: no pinning).\"\"\"\n"
    "\n"
    "    nvme_path: str | None = None\n"
    "    \"\"\"Directory holding model.safetensors.index.json and the shards, if it is not the model path.\"\"\"\n"
    "\n"
    "    @model_validator(mode=\"after\")\n"
    "    def _validate_shared_memory(self) -> Self:\n"
)

CFG_VALIDATOR_OLD = (
    "        return self\n"
    "\n"
    "    def verify_model_config(self, model_config: \"ModelConfig | None\") -> None:\n"
)
CFG_VALIDATOR_NEW = (
    "        return self\n"
    "\n"
    "    @model_validator(mode=\"after\")\n"
    f"    def _validate_nvme(self) -> Self:  {M}\n"
    "        if self.nvme_io_backend not in (\"auto\", \"aio\", \"threads\"):\n"
    "            raise ValueError(f\"nvme_io_backend must be auto, aio or threads, not {self.nvme_io_backend!r}\")\n"
    "        if self.nvme_poll not in (\"auto\", \"spin\", \"sleep\"):\n"
    "            raise ValueError(f\"nvme_poll must be auto, spin or sleep, not {self.nvme_poll!r}\")\n"
    "        if not (1 <= self.nvme_queue_depth <= 4096 and 1 <= self.nvme_threads <= 64):\n"
    "            raise ValueError(\"nvme_queue_depth must be 1..4096 and nvme_threads 1..64\")\n"
    "        if not (0 <= self.nvme_verify_pct <= 100 and self.nvme_keepwarm_us >= 0 and self.nvme_timeout_s >= 1.0):\n"
    "            raise ValueError(\"nvme_verify_pct must be 0..100, nvme_keepwarm_us >= 0 and nvme_timeout_s >= 1\")\n"
    "        if not self.nvme:\n"
    "            return self\n"
    "        if self.cpu_offload:\n"
    "            raise ValueError(\n"
    "                \"EngramConfig.nvme needs cpu_offload=false (on a unified-memory machine a pinned host table \"\n"
    "                \"is the same DRAM as the GPU's)\"\n"
    "            )\n"
    "        if self.dp_shared_memory or self.use_thp or self.embedding_across_dp:\n"
    "            raise ValueError(\"EngramConfig.nvme does not combine with dp_shared_memory, use_thp or embedding_across_dp\")\n"
    "        if self.nvme_layers is not None and not self.nvme_layers:\n"
    "            raise ValueError(\"EngramConfig.nvme_layers must not be empty (leave it unset for layer 14 only)\")\n"
    "        return self\n"
    "\n"
    f"    def nvme_layers_for(self, layer_ids) -> tuple[int, ...]:  {M}\n"
    "        \"\"\"The Engram layers (model layer ids) served from NVMe: none unless `nvme`; by default layer 14 only.\"\"\"\n"
    "        if not self.nvme:\n"
    "            return ()\n"
    "        ids = tuple(layer_ids)\n"
    "        if self.nvme_layers is None:\n"
    "            return (14,) if 14 in ids else ids[-1:]\n"
    "        return tuple(i for i in ids if i in self.nvme_layers)\n"
    "\n"
    "    def verify_model_config(self, model_config: \"ModelConfig | None\") -> None:\n"
)

CFG_MODEL_OLD = (
    "                \"embeddings and non-empty n-gram layer ids.\"\n"
    "            )\n"
)
CFG_MODEL_NEW = (
    "                \"embeddings and non-empty n-gram layer ids.\"\n"
    "            )\n"
    f"        if self.nvme:  {M}\n"
    "            ids = tuple(getattr(model_config.hf_text_config, field))\n"
    "            unknown = [i for i in (self.nvme_layers or ()) if i not in ids]\n"
    "            if model_config.architecture != \"DeepseekV41ForCausalLM\" or unknown:\n"
    "                raise ValueError(\n"
    "                    \"EngramConfig.nvme serves DeepSeek V4.1 Engram layers only; \"\n"
    "                    f\"nvme_layers {self.nvme_layers} must be among the model's {list(ids)}\"\n"
    "                )\n"
)

CFG_PARALLEL_OLD = (
    "        \"\"\"Reject unsupported embedding parallel topologies.\"\"\"\n"
    "        if self.dp_shared_memory:\n"
)
CFG_PARALLEL_NEW = (
    "        \"\"\"Reject unsupported embedding parallel topologies.\"\"\"\n"
    f"        if self.nvme and (  {M}\n"
    "            parallel_config.data_parallel_size > 1\n"
    "            or parallel_config.pipeline_parallel_size > 1\n"
    "            or parallel_config.enable_elastic_ep\n"
    "        ):\n"
    "            raise ValueError(\n"
    "                \"EngramConfig.nvme needs data_parallel_size == 1, pipeline_parallel_size == 1 and no elastic EP.\"\n"
    "            )\n"
    "        if self.dp_shared_memory:\n"
)

CFG_HASH_OLD = (
    "        \"\"\"Hash settings that affect embedding execution and graph structure.\"\"\"\n"
    "        return hash_factors(get_hash_factors(self, set()))\n"
)
CFG_HASH_NEW = (
    "        \"\"\"Hash settings that affect embedding execution and graph structure.\"\"\"\n"
    f"        {M} the nvme_* fields enter the hash only when nvme is on, so a patched engine with it off keeps\n"
    "        # the hash (and every cache keyed on it) of an unpatched one\n"
    "        import dataclasses\n"
    "\n"
    "        ignored = set() if self.nvme else {f.name for f in dataclasses.fields(self) if f.name.startswith(\"nvme\")}\n"
    "        return hash_factors(get_hash_factors(self, ignored))\n"
)

ENGRAM_SHA = "1dd1c5717a37b388d87124cb513f95c567da6b9eecf0f5a7e1ab7f9b9866bb08"

E_SIG_OLD = (
    "        dp_shared_memory: bool = False,\n"
    "        use_thp: bool = False,\n"
    "    ) -> None:\n"
    "        super().__init__()\n"
    "        assert head_sizes and all(size > 0 for size in head_sizes)\n"
)
E_SIG_NEW = (
    "        dp_shared_memory: bool = False,\n"
    "        use_thp: bool = False,\n"
    f"        nvme: bool = False,  {M} rows are read from NVMe: no table is allocated here\n"
    "    ) -> None:\n"
    "        super().__init__()\n"
    "        assert head_sizes and all(size > 0 for size in head_sizes)\n"
)

E_ATTR_OLD = (
    "        self.use_thp = use_thp\n"
    "        self.dp_size = get_engram_dp_size()\n"
)
E_ATTR_NEW = (
    "        self.use_thp = use_thp\n"
    f"        self.nvme = nvme  {M}\n"
    "        self.dp_size = get_engram_dp_size()\n"
)

E_ALLOC_OLD = (
    "    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:\n"
    "        if self.dp_shared_memory:\n"
)
E_ALLOC_NEW = (
    "    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:\n"
    f"        if self.nvme:  {M} the rows live in the checkpoint shards; keep the module structure, no data\n"
    "            return (\n"
    "                torch.empty(0, self.dim, dtype=torch.float8_e4m3fn),\n"
    "                torch.empty(0, self.dim // self.block_size, dtype=torch.uint8),\n"
    "            )\n"
    "        if self.dp_shared_memory:\n"
)

E_STORAGE_OLD = (
    "    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:\n"
    "        if self._shared_memory is not None:\n"
)
E_STORAGE_NEW = (
    "    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:\n"
    f"        if self.nvme:  {M}\n"
    "            raise RuntimeError(\"this Engram table is served from NVMe (EngramConfig.nvme): there is no resident storage\")\n"
    "        if self._shared_memory is not None:\n"
)

E_CLASSATTR_OLD = (
    "    _batch: EngramBatch | None = None\n"
    "\n"
    "    def __init__(\n"
    "        self,\n"
    "        config,\n"
)
E_CLASSATTR_NEW = (
    "    _batch: EngramBatch | None = None\n"
    f"    _nvme = None  {M} EngramNvmeChannel when this layer's rows come from NVMe\n"
    "\n"
    "    def __init__(\n"
    "        self,\n"
    "        config,\n"
)

E_CREATE_OLD = (
    "            use_thp=engram_config.use_thp,\n"
    "        )\n"
    "\n"
    "    def _init_staging(self, max_tokens: int, head_dim: int) -> None:\n"
)
E_CREATE_NEW = (
    "            use_thp=engram_config.use_thp,\n"
    f"            nvme=layout.layer_ids[layer_hash_index]  {M}\n"
    "            in engram_config.nvme_layers_for(layout.layer_ids),\n"
    "        )\n"
    "\n"
    "    def _init_staging(self, max_tokens: int, head_dim: int) -> None:\n"
)

E_INIT_OLD = (
    "        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens\n"
    "        self._init_staging(max_tokens, layout.head_dim)\n"
    "\n"
    "    def _create_embedding(\n"
)
E_INIT_NEW = (
    "        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens\n"
    "        self._init_staging(max_tokens, layout.head_dim)\n"
    f"        if self.embed_tokens.nvme:  {M} this layer's rows are read from NVMe, not looked up\n"
    "            from vllm.models.deepseek_v41.common.engram_nvme import create_channel\n"
    "\n"
    "            self._nvme = create_channel(self, layout, layer_hash_index, max_tokens)\n"
    "\n"
    "    def _create_embedding(\n"
)

E_PREPARE_OLD = (
    "        assert rows.shape[0] == hash_ids.shape[0], \"engram staging buffer too small\"\n"
    "        if self._prefetch_stream is None:\n"
)
E_PREPARE_NEW = (
    "        assert rows.shape[0] == hash_ids.shape[0], \"engram staging buffer too small\"\n"
    f"        if self._nvme is not None:  {M} hand the row ids to the NVMe reader; _ready_rows waits for it\n"
    "            self._nvme.publish(hash_ids)\n"
    "            return\n"
    "        if self._prefetch_stream is None:\n"
)

E_READY_OLD = (
    "    def _ready_rows(self, num_tokens: int) -> torch.Tensor:\n"
    "        if self._prefetch_done is not None:\n"
)
E_READY_NEW = (
    "    def _ready_rows(self, num_tokens: int) -> torch.Tensor:\n"
    f"        if self._nvme is not None:  {M} wait for the reader, dequantize the staged rows\n"
    "            self._nvme.fetch(self.staged_rows[:num_tokens])\n"
    "            return self.staged_rows[:num_tokens]\n"
    "        if self._prefetch_done is not None:\n"
)

EPFILTER_SHA = "c6e020abeed0d9d78eb6f6705376ea6b63abe95fbbadace6f7faec17c69396dc"

EPF_OLD = (
    "def should_skip_weight(\n"
    "    weight_name: str,\n"
    "    local_expert_ids: set[int] | None,\n"
    ") -> bool:\n"
    "    \"\"\"Return ``True`` if *weight_name* is an expert weight that does not\n"
    "    belong to the local rank and should be skipped during loading.\"\"\"\n"
    "    if local_expert_ids is None:\n"
    "        return False\n"
)
EPF_NEW = (
    f"_NVME_CFG = None  {M} the vLLM config the regex below was derived from\n"
    "_NVME_TABLES = False  # compiled regex of the NVMe-served Engram tables of that config, or False when there are none\n"
    "\n"
    "\n"
    f"def _nvme_engram_table(weight_name: str) -> bool:  {M}\n"
    "    \"\"\"True for the value/scale tensors of an Engram layer served from NVMe (EngramConfig.nvme): they are read at run\n"
    "    time from the shard, never loaded, so the loader must not even map them (each is ~94 GiB).\"\"\"\n"
    "    global _NVME_CFG, _NVME_TABLES\n"
    "    from vllm.config import get_current_vllm_config_or_none\n"
    "\n"
    "    cfg = get_current_vllm_config_or_none()\n"
    "    if cfg is None:\n"
    "        return False\n"
    "    if cfg is not _NVME_CFG:  # the target model and the draft model load under different configs\n"
    "        ec = cfg.engram_config\n"
    "        ids = getattr(getattr(cfg.model_config, \"hf_text_config\", None), \"engram_layer_ids\", None) or ()\n"
    "        layers = ec.nvme_layers_for(ids) if ec is not None else ()\n"
    "        _NVME_TABLES = (\n"
    "            re.compile(r\"^layers\\.(?:\" + \"|\".join(str(i) for i in layers) + r\")\\.engram\\.embed\\.(?:weight|scale)$\")\n"
    "            if layers\n"
    "            else False\n"
    "        )\n"
    "        _NVME_CFG = cfg\n"
    "    return bool(_NVME_TABLES) and _NVME_TABLES.match(weight_name) is not None\n"
    "\n"
    "\n"
    "def should_skip_weight(\n"
    "    weight_name: str,\n"
    "    local_expert_ids: set[int] | None,\n"
    ") -> bool:\n"
    "    \"\"\"Return ``True`` if *weight_name* is an expert weight that does not\n"
    "    belong to the local rank and should be skipped during loading.\"\"\"\n"
    f"    if _nvme_engram_table(weight_name):  {M}\n"
    "        return True\n"
    "    if local_expert_ids is None:\n"
    "        return False\n"
)

EDITS = [
    ("config/engram.py", CFG_SHA, CFG_FIELDS_OLD, CFG_FIELDS_NEW),
    ("config/engram.py", CFG_SHA, CFG_VALIDATOR_OLD, CFG_VALIDATOR_NEW),
    ("config/engram.py", CFG_SHA, CFG_MODEL_OLD, CFG_MODEL_NEW),
    ("config/engram.py", CFG_SHA, CFG_PARALLEL_OLD, CFG_PARALLEL_NEW),
    ("config/engram.py", CFG_SHA, CFG_HASH_OLD, CFG_HASH_NEW),
    ("models/deepseek_v41/common/engram.py", ENGRAM_SHA, E_SIG_OLD, E_SIG_NEW),
    ("models/deepseek_v41/common/engram.py", ENGRAM_SHA, E_ATTR_OLD, E_ATTR_NEW),
    ("models/deepseek_v41/common/engram.py", ENGRAM_SHA, E_ALLOC_OLD, E_ALLOC_NEW),
    ("models/deepseek_v41/common/engram.py", ENGRAM_SHA, E_STORAGE_OLD, E_STORAGE_NEW),
    ("models/deepseek_v41/common/engram.py", ENGRAM_SHA, E_CLASSATTR_OLD, E_CLASSATTR_NEW),
    ("models/deepseek_v41/common/engram.py", ENGRAM_SHA, E_CREATE_OLD, E_CREATE_NEW),
    ("models/deepseek_v41/common/engram.py", ENGRAM_SHA, E_INIT_OLD, E_INIT_NEW),
    ("models/deepseek_v41/common/engram.py", ENGRAM_SHA, E_PREPARE_OLD, E_PREPARE_NEW),
    ("models/deepseek_v41/common/engram.py", ENGRAM_SHA, E_READY_OLD, E_READY_NEW),
    ("model_executor/model_loader/ep_weight_filter.py", EPFILTER_SHA, EPF_OLD, EPF_NEW),
]
