# llama-superfast-v100-BWE-q8-dflash2

A fork of [bitexploder/llama-superfast-v100](https://github.com/bitexploder/llama-superfast-v100), the llama.cpp fork
tuned for the Tesla V100 (Volta, `sm_70`), made for **one V100 card**, a **single user**, **up to four concurrent slots**
on a **unified KV cache**, and the **DFlash2 speculative draft**: agentic work made palatable at the model's whole
262,144-token context on one 32 GB card. Some ideas were stolen from [ReVolta](https://github.com/Leonccaa/ReVolta).

The model is Qwen3.8-27B (`UD-Q4_K_XL`) on a V100 PCIe 32 GB, decoded with [DFlash2](https://github.com/z-lab/dflash),
z-lab's speculative draft for it: a small draft model that reads the target's own layer inputs and proposes a block of
up to 7 tokens, which the 27B verifies in one step; about half are accepted, and that is what takes decode from the
~40 tok/s a 27B manages alone on this card to ~105. Upstream ships DFlash2 beside Qwen's MTP head; this fork runs
DFlash2 by default, and much of what it adds (features kept on the GPU, the rollback by replay, the scheduler's jobs)
is built around it. The slots serve one user: the main thinking thread, and the side requests an agent harness sends
in parallel with it. Our card sits behind a PCIe Gen3 x4 eGPU link (~3.3 GB/s), so anything that crosses the host link
is on the critical path; the gains from keeping work on the GPU are largest on a link like that.

The rest of upstream (the tensor-core matvec on repacked K-quants, MTP, the two-user scheduler's design, two-card
tensor parallel, Qwen3.8-Flash-Next) is documented in
[upstream's README](https://github.com/bitexploder/llama-superfast-v100) and in [`README-27B.md`](README-27B.md).

Part of this work was sponsored by [Inscora](https://inscora.com). If you are an MSP or MSSP and want to offer cyber
insurance services to your clients, have a look at [inscora.com](https://inscora.com).

## What this fork adds

Against upstream at `403f06f`, one line each. The design, the measurements and the switches of every item are in
[CHANGES.md](CHANGES.md); the numbers are from one V100 PCIe 32 GB behind a PCIe Gen3 x4 eGPU link.

**Context: the model's whole 262,144 tokens on one 32 GB card, shared by four slots**

- **`q8_0` KV cache on the Volta attention kernels**: the CUTLASS prefill and the streaming verify kernels read 8-bit K/V directly, ~3% of decode for 2.7× the context. ([§1](CHANGES.md#1-q8_0-kv-cache-on-the-volta-attention-paths))
- **One unified pool at the split cache's speed**: each batch attends over its own sequence's cells only, with device-built masks for every slot. ([§4](CHANGES.md#4-unified-kv-at-full-speed))
- **Rollback by replay on its own stream**: one kept recurrent state a slot, the draft's accepted tokens replayed under the draft. ([§15](CHANGES.md#15-rollback-by-replay-one-state-a-slot-instead-of-a-snapshot-a-token))
- **A capped CUDA graph cache**: 48 graphs for 120 s instead of 512 for 600 s, ~1.2 GB of VRAM that only grew under real traffic. ([§5](CHANGES.md#5-a-smaller-cuda-graph-cache))
- **A conversation's cells stay one run of the pool**: placement right after its last cell or in the lowest free run, a boxed-in prefix moved before its ubatch, new conversations low in the pool. ([§6](CHANGES.md#6-cell-placement-in-a-unified-cache), [§11](CHANGES.md#11-compaction-of-a-boxed-in-prefix), [§14](CHANGES.md#14-a-prompts-cells-stay-one-run-low-in-the-pool))
- **Nothing is cleared for a fragmented pool**: the idle conversations pack down (23.7k cells in 14 ms, without stopping the GPU; a fragmented pack of 357 runs in 16 ms, in one launch) before one is ever cleared for a prompt, and a conversation that is cleared comes back from host RAM in under a second instead of a re-read of minutes, while the cache keeps 4 GiB of the host's memory free. ([§14](CHANGES.md#14-a-prompts-cells-stay-one-run-low-in-the-pool), [§16](CHANGES.md#16-the-prompt-cache-gives-back-the-conversations-that-left-the-slots), [§18](CHANGES.md#18-four-agents-at-once-packs-that-pay-the-joint-steps-empty-tiles-image-prompts-as-jobs), [§19](CHANGES.md#19-row-moves-in-one-launch-and-a-long-idle-conversation-cleared-once-it-is-cached))

**Speed**

- **Faster prefill on repacked weights**: the fp16 expansion decodes straight from the repacked layout, products of 129–191 tokens go to MMQ. ([§2](CHANGES.md#2-faster-prefill-on-repacked-weights))
- **Prompt work kept on the GPU**: DFlash2 features, checkpoints and prompt-cache saves no longer cross the host link every ubatch. ([§3](CHANGES.md#3-prompt-work-kept-on-the-gpu))
- **Short prompts**: 32-token matvec passes and no CUDA graph capture for prompt shapes, the first token 15–20% sooner. ([§9](CHANGES.md#9-short-prompts-32-token-passes-no-graph-capture-for-prompts))
- **Checkpoints from the rollback state**: the tail checkpoint without a 50 ms decode of its own, two checkpoints kept on the device. ([§10](CHANGES.md#10-checkpoints))
- **Four slots that share the verify step**: the verify rows budgeted across the generating slots, the draft run one width at a time, and the joint step skipping the KV tiles none of its rows can see; 151 tok/s combined with four requests in flight against upstream's 89. ([§12](CHANGES.md#12-four-slots-the-verify-rows-budgeted-the-draft-one-width-at-a-time), [§18](CHANGES.md#18-four-agents-at-once-packs-that-pay-the-joint-steps-empty-tiles-image-prompts-as-jobs))

**Several agents at once**

- **The two-user scheduler with DFlash2 on a unified cache**: a side prompt is read beside the generating slot from the moment it arrives, and the generating slot gets more rounds per group as the read deepens. ([§8](CHANGES.md#8-the-two-user-scheduler-with-dflash2-on-a-unified-cache))
- **A full pool fails fast and keeps what it holds**: a prompt that cannot fit is refused at once, a request never pushes out a big conversation while a slot is empty, and a returning conversation gets room before its restore. ([§17](CHANGES.md#17-a-full-pool-the-slot-kept-the-prompt-that-cannot-fit-refused-room-made-for-a-restore))
- **Vision with the scheduler on**: a loaded projector does not turn the scheduler off, and a prompt holding an image is read as a job like any other, so the other slots keep streaming. ([Vision](#vision), [§18](CHANGES.md#18-four-agents-at-once-packs-that-pay-the-joint-steps-empty-tiles-image-prompts-as-jobs))
- **Two or more cards by code reading**: every addition adapted to upstream's tensor-parallel path, untested here. ([below](#two-or-four-cards-tensor-parallel-untested))

**Build**

- **The tensor-core matvec's ~1,000 kernels in per-instance files**: the 14-minute serial compile gone, the longest file ~90 s. ([§7](CHANGES.md#7-build-time))

## Results

[BetterBench](https://github.com/bitexploder/betterbench) on one V100 PCIe 32 GB (200 W cap), Qwen3.8-27B
`UD-Q4_K_XL` with the DFlash2 draft, the same corpus for both (20 passes per category):

- **Upstream**, [bitexploder/llama-superfast-v100](https://github.com/bitexploder/llama-superfast-v100) at `403f06f`:
  its f16 KV cache and recipe (`-b 4096 -ub 2048`), two slots of 49,152 tokens, with a 47k-token conversation resident
  in one of them, sampling at temperature 0.7. That is the largest two-slot context that survived the run: at 2 × 65,536 the card ran out of memory
  four minutes in, as its graph cache grew under traffic ([§5](CHANGES.md#5-a-smaller-cuda-graph-cache)). Its
  scheduler is off (`LLAMA_TU=0`): on, its prompt jobs refuse the DFlash2 draft and long prompts fail.
- **This fork**: the [Run](#run) line as it serves in production, `q8_0` KV, four slots on one 262,144-token pool, the
  host prompt cache on, sampling at the server's temperature, 1.0.

| BetterBench | Upstream, f16 KV, 2 × 49,152 | This fork, `q8_0` KV, 4 slots on 262,144 |
|---|---|---|
| Context | 2 slots of 49,152 tokens (98,304 in all) | 4 slots sharing one 262,144-token pool |
| Time to first token, single stream (p50) | 610 ms | 191 ms |
| Single-stream decode (weighted median)* | 108.9 tok/s at temperature 0.7 | 102.2 tok/s at temperature 1.0 |
| Single-stream decode, median of 160 runs* | — | 109.1 tok/s |
| Cold prefill, 1.5k / 6k / 12k / 24k / 47k-token prompts | 785 / 856 / 877 / 858 / 795 tok/s | 950 / 975 / 945 / 902 / 826 tok/s |
| Aggregate decode, 1 / 2 / 4 requests in flight* | 90 / 101 / 89 tok/s | 90 / 124 / 156 tok/s |
| Per-request decode, 1 / 2 / 4 requests in flight (median)* | 130 / 102 / 123 tok/s | 111 / 71 / 46 tok/s |
| Time to first token, 1 / 2 / 4 requests in flight (p50) | 0.66 / 2.26 / 12.07 s | 0.19 / 0.30 / 0.36 s |
| Peak VRAM | 31,044 MiB at 98,304 tokens | 32,260 MiB at 262,144 tokens (production, under agents) |

\* The decode rows are not like for like: decode follows how many draft tokens are accepted, which falls as the
temperature rises, and upstream ran at 0.7. Under upstream's conditions (temperature 0.7, the 47k-token conversation
resident), a lone stream decodes 3% slower here, the cost of the 8-bit cache, and four requests in flight give 158 tok/s
against 89. Prefill and the time to the first token do not depend on the temperature. Upstream's per-request decode stays
high with four requests because its two slots serve them two at a time: each decodes almost alone, after waiting up to
12 s to start.

The first token comes three times sooner, prompts are read 4–21% faster (the most for short ones), the context is 2.7×
larger, and with four requests in flight the fork serves 1.8× the throughput, the fourth request starting in 0.4 s
instead of 12 s. The 262,144-token pool peaks at ~32.3 GB of the card's 32.5 in production under four agents.

What the bench does not show, because its conversations are short and never return:

- **Prompts far past the bench's 47k.** A 128k-token prompt is read cold at 629 tok/s and a 244k one, the most a
  slot holds, at 459 tok/s, the attention at depth running on tensor cores
  ([§27](CHANGES.md#27-the-streaming-kernel-takes-prompts-of-up-to-64-tokens), [§28](CHANGES.md#28-the-attentions-structure-at-depth-branch-attn-depth)).

- **A side prompt beside a generating slot.** Upstream's two-user scheduler refuses DFlash2 prompt jobs, so a side
  request's prompt waits until the other slot finishes generating. Here it starts at once and the generating slot keeps
  ~34 tok/s while it is read ([§8](CHANGES.md#8-the-two-user-scheduler-with-dflash2-on-a-unified-cache)).
- **Four long conversations generating together.** Four ~30k-token conversations in the pool decode 151 tok/s
  combined, at 76.6 ms a joint step ([§18](CHANGES.md#18-four-agents-at-once-packs-that-pay-the-joint-steps-empty-tiles-image-prompts-as-jobs)).
- **A conversation that comes back.** One that left its slot is restored from the host prompt cache in 0.97 s for 24k
  tokens over this link, against a re-read of 26 s; when it is cleared for room, its save runs in the background, ~35 ms of
  the server's time ([§16](CHANGES.md#16-the-prompt-cache-gives-back-the-conversations-that-left-the-slots),
  [§20](CHANGES.md#20-a-conversation-cleared-for-room-is-saved-in-the-background)).

## Build

Requirements:

- A Tesla V100 (`sm_70`). The changes here are Volta-specific; other GPUs are untested.
- CUDA 12.x (12.8 tested; CUDA 13 dropped Volta), gcc/g++ 14, CMake, Ninja. ccache is optional and recommended.
- The NVIDIA container toolkit, for the Docker route.

**Native:**

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70-real -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CCACHE=ON \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build --target llama-server -j 16
```

- `GGML_CUDA_FA_ALL_QUANTS=OFF` still builds the `f16` and `q8_0` KV pairs this setup reads.
- `GGML_CUDA_CUB_3DOT2=ON` fetches CCCL 3.2 at configure time.
- A cold build takes about 15 minutes on 16 threads. The CUTLASS attention units need several GB of RAM per `nvcc`;
  lower `-j` on a machine with less than ~32 GB.
- Add `--target test-backend-ops` (with `-DLLAMA_BUILD_TESTS=ON`) for the kernel tests.

**Docker:**

```bash
DOCKER_BUILDKIT=1 docker build -f .devops/cuda-v100.Dockerfile -t llama-superfast-v100 .devops
```

The image clones this repository at `--build-arg LLAMA_REF=` (default `main`; pin a commit for a reproducible image).
Other build args: `BUILD_JOBS` (default 8), `CUDA_VERSION` (12.8.1). Its compiler cache persists across builds, so
rebuilding after a small change takes a few minutes.

## Models

```bash
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf --local-dir models
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models
hf download unsloth/Qwen3.8-27B-GGUF mmproj-F16.gguf --local-dir models   # for images (the Vision section)
```

Put the files flat in `./models` (move `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` up). DFlash2 needs the MTP file too: its output
head reads that file's rows. The draft vocabulary ranking, `models/draft-vocab-qwen3.8-27b.txt`, ships in this
repository.

## Run

One card, four slots, DFlash2, one 262,144-token pool:

```bash
LLAMA_DFLASH2_HEAD_FILE=models/mtp-Qwen3.8-27B-Q4_0.gguf LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server -m models/Qwen3.8-27B-UD-Q4_K_XL.gguf -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ubatch 512 -c 262144 --parallel 4 --kv-unified -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 -t 12 -b 2048 -ub 2048 --cache-ram 15360 --no-cache-idle-slots --ctx-checkpoints 16 --jinja --metrics --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```

The same with the Docker image (models in `$PWD/models`):

```bash
docker run -d --name llama-superfast --init --gpus all -p 8080:8080 -v "$PWD/models:/models:ro" -e LLAMA_DFLASH2_HEAD_FILE=/models/mtp-Qwen3.8-27B-Q4_0.gguf llama-superfast-v100 --host 0.0.0.0 --port 8080 -m /models/Qwen3.8-27B-UD-Q4_K_XL.gguf -md /models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ubatch 512 -c 262144 --parallel 4 --kv-unified -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 -t 12 -b 2048 -ub 2048 --cache-ram 15360 --no-cache-idle-slots --ctx-checkpoints 16 --jinja --metrics --temp 1.0 --top-p 0.95 --top-k 20
```

What the settings do:

- The two-user scheduler is on ([§8](CHANGES.md#8-the-two-user-scheduler-with-dflash2-on-a-unified-cache)): a long prompt in one slot does not stall the other slots' decode.
  `LLAMA_TU=0` turns it off.
- `--parallel 4` ([§12](CHANGES.md#12-four-slots-the-verify-rows-budgeted-the-draft-one-width-at-a-time), [§15](CHANGES.md#15-rollback-by-replay-one-state-a-slot-instead-of-a-snapshot-a-token)): four slots, each with one kept recurrent state; the draft's rollbacks replay.
- `LLAMA_DFLASH2_HEAD_FILE` points DFlash2's output head at the MTP file's rows.
- `LLAMA_SPEC_DRAFT_VOCAB_FILE` (set by the Docker image) restricts the draft to the top 98,304 ids of the file's
  131,072-id ranking. Without it, decode is ~5–10% slower.
- `--kv-unified` gives the slots one 262,144-token pool: any slot can take most of it (one took a 157k-token prompt
  while another answered side requests). `--kv-unified-per-slot N` caps each slot inside the pool;
  `--no-kv-unified` gives each slot a fixed share (with two slots 2 × 122,880) at the same speed.
- `-b 2048` equal to `-ub` keeps the DFlash2 features on the GPU during prefill ([§3](CHANGES.md#3-prompt-work-kept-on-the-gpu)).
- `--spec-draft-ubatch 512` sizes the draft's compute buffer for 512-token batches, which frees 1.7 GB.
- `--cache-ram 15360 --no-cache-idle-slots` ([§16](CHANGES.md#16-the-prompt-cache-gives-back-the-conversations-that-left-the-slots)): a host prompt cache of
  15 GiB for the conversations that leave the slots. **`--cache-ram 15360` is sized for a host with 32 GB of RAM; with
  less, lower it.** One cleared from the pool for a bigger prompt, or replaced in its slot, is restored when it comes
  back: 0.97 s for 24k tokens over this link, against a re-read of 26 s. One cleared for room is saved in the background
  (~35 ms of the server's time, [§20](CHANGES.md#20-a-conversation-cleared-for-room-is-saved-in-the-background)); one
  replaced in its slot pauses the server once, ~0.64 s for 47k tokens; conversations under 8,192 tokens are not saved (`LLAMA_CACHE_MIN_TOKENS`), an entry keeps its
  conversation's newest two checkpoints (`LLAMA_CACHE_CKPT_KEEP`), and the cache drops its oldest entries to keep
  4 GiB of the host's memory available whatever its limit (`LLAMA_CACHE_RESERVE_MIB`). Without
  `--no-cache-idle-slots` the server saves and clears every idle slot at every new task. `--cache-ram 0` turns the
  cache off. It is RAM: a restart loses it.
- `--ctx-checkpoints 16` ([§10](CHANGES.md#10-checkpoints)): the recurrent-state checkpoints each slot keeps,
  150 MiB each in pinned host memory reserved at start (16 × 4 slots + 3, 10 GB; the default of 32 reserves 19.6 GB).
  The latest resumes the next turn; the older ones resume a prompt edited mid-way.
- Host memory: these settings take ~12.5 GB at start and up to ~28 GB with a full cache, on a 32 GB host. With less,
  lower `--cache-ram` (or 0) and `--ctx-checkpoints`.
- 262,144 tokens with four slots peaks at ~32.3 GB under BetterBench. VRAM keeps growing for the first minutes of real
  traffic, so size a different context from a long mixed run, not a fill test ([§5](CHANGES.md#5-a-smaller-cuda-graph-cache)).

The host side of this setup — powering an OcuLink eGPU dock from a USB relay, training the PCIe link through an S3
suspend, loading the driver with the 200 W cap, and cooling the passive card with a PWM fan driven from its
temperature, as systemd units an inference service chains after — is published separately as
[v100-egpu-host](https://github.com/gabtremblay/v100-egpu-host), with the bill of materials.

Check the startup log for `two-user scheduling (LLAMA_TU): on`, the draft model loading,
`prompt checkpoints: 67 pinned buffers` and `n_slots = 4, n_ctx_slot = 262144`. Draft acceptance on this target is
~0.4–0.5; near zero means the draft did not load. In traffic, `kv_pack` lines mean the pool was packed instead of a
conversation cleared, and `prompt_cache: ... loaded=1` a conversation restored from the cache.

For MTP instead of DFlash2: `-md models/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp`, without
`LLAMA_DFLASH2_HEAD_FILE` or `--spec-draft-ubatch`. It decodes slower here.

### Two or four cards (tensor parallel, untested)

This fork was built and measured on one card. Upstream runs the 27B on two cards in tensor parallel (`--split-mode
tensor`, the ggml meta backend, `GGML_CUDA_P2P=1`, `iommu=pt`; see its `README-27B.md`), and the fork's additions were
adapted to that path, for any number of cards, so far only by reading the code: there is no multi-card machine here to
run it on. What was done, and what to expect:

- **The unified pool, cell placement, compaction, idle-slot eviction, the q8_0 cache and the per-batch window** work per
  device: a split cache keeps every cell on every card (each holds its heads of it), so the cell bookkeeping is the same
  and the row moves of a compaction run on each card's own slice. The device mask builder was already split-aware.
- **DFlash2's device-fed features** (the draft reading the target's layer inputs on the GPU) copy card to card: with the
  draft split like the target, each card feeds its own copy; with the draft pinned to one card (`--device-draft`), that
  card's copy is read. A split target with a draft split differently refuses the device path and says so; set
  `LLAMA_DFLASH2_FEAT_DEV=0` then.
- **The two-user scheduler** runs its prompt jobs on the plain layer loop under the split (upstream's interleaved
  two-half prefill is off while the draft reads layer inputs, as upstream decided), with MTP or DFlash2.
- **The rollback's replay** restores and backs up its rows per card; the fused write is a per-card kernel already.
- **The pool's packing** moves a split cache's rows device by device, with a synchronous copy each (slower than on one
  card).
- **Checkpoints** on a split cache go through upstream's path; the fork's staged copy (`LLAMA_CKPT_STAGE`) steps aside
  for them.

The line is upstream's two-card line with this fork's switches, for example on two cards. It leaves the prompt cache
off (`--cache-ram 0`): its save and restore on a split cache are untested.

```bash
CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_P2P=1 LLAMA_DFLASH2_HEAD_FILE=models/mtp-Qwen3.8-27B-Q4_0.gguf LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server -m models/Qwen3.8-27B-UD-Q4_K_XL.gguf -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ubatch 512 -c 491520 --parallel 4 --kv-unified --split-mode tensor -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 -t 12 -b 2048 -ub 2048 --cache-ram 0 --jinja --metrics --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```

Two 32 GB cards hold about twice the pool (the weights are split too), so `-c` can double; start lower and raise it
while the server fits. Expect upstream's gains (1.35× decode, 1.7× prefill on two cards) on top of this fork's, not
more: the draft's acceptance and the verify budget are the same. If something fails, the one-card line in `## Run`
with `CUDA_VISIBLE_DEVICES=0` is the fallback, and a report with the server log is welcome.

## Vision

The server takes images when Qwen3.8-27B's vision projector is loaded: `--mmproj models/mmproj-F16.gguf`
(from the same Unsloth repository, 884 MiB; a BF16 one is there too, and any Qwen3.8-27B projector fits, a Q8_0 one
saving ~440 MiB). Chat requests then carry `image_url` parts (data: URLs or http). The projector and its encode buffers
take ~2 GB of the card, so the pool is 196,608 tokens instead of 262,144:

```bash
LLAMA_DFLASH2_HEAD_FILE=models/mtp-Qwen3.8-27B-Q4_0.gguf LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server -m models/Qwen3.8-27B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ubatch 512 -c 196608 --parallel 4 --kv-unified -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 -t 12 -b 2048 -ub 2048 --cache-ram 15360 --no-cache-idle-slots --ctx-checkpoints 16 --jinja --metrics --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```

(the Docker line takes the same two additions: `--mmproj /models/mmproj-F16.gguf` and `-c 196608`).

- 196,608 tokens keeps a margin: the card peaked at 31.6 GB of its 32.5 over 50 minutes of concurrent image and text
  requests, 47k-token prompts and four-slot decode. Larger pools with the projector are untested.
- A 2048 × 1536 image is ~3,100 prompt tokens, read at ~520 tok/s; an 800 × 600 one ~500 tokens. Its answer decodes at
  85–108 tok/s with the draft (a red square, a blue circle and a white band in a test image are all named).
- Text requests run as without the projector: four in flight 130–134 tok/s combined, 47k-token prompts at 843 tok/s.
  Upstream turns the two-user scheduler off as soon as `--mmproj` is given, and four text requests in flight then fall
  to 71 tok/s combined; here it stays on.
- A prompt holding an image is read as scheduler jobs, each image decoded when the read reaches it: with three slots
  streaming, an image plus a 20k-token log took 56.8 s, and the streaming slots paused at most 1.1 s and streamed 3,422
  tokens during the read. `LLAMA_TU_MEDIA_JOBS=0` reads it in the batch instead: 39.1 s, but the other slots pause up to
  3.7 s and stream 295 tokens.
- With a projector loaded, llama.cpp's prompt cache reuse and context shift are off for every slot (upstream's rule), and
  so is the fork's tail-snapshot checkpoint for prompts that hold media. The host prompt cache has run with images in
  production use, not under a probe.

## Known limits

- Tested only on V100 PCIe 32 GB with Qwen3.8-27B. The window and placement changes are on for the qwen35 model line
  only.
- Combined decode tops out near 130–160 tok/s with four slots: a verify step costs about the same up to 8 rows, so the
  slots share 16 verify rows a step ([§12](CHANGES.md#12-four-slots-the-verify-rows-budgeted-the-draft-one-width-at-a-time)), and each slot drafts less when more generate.
- Four slots at 262,144 tokens (the model's training context) use ~32.3 GB of the card's 32.5 under BetterBench. The
  recurrent rows are 150 MiB each, 10 of them ([§15](CHANGES.md#15-rollback-by-replay-one-state-a-slot-instead-of-a-snapshot-a-token)).
- The first two prompts after a start pay ~0.4 s once each to allocate a device checkpoint copy.
- The checkpoint pool and the prompt cache live in host memory: ~10 GB pinned and up to 15 GiB with the Run line, on a
  32 GB host. A restart loses every cached conversation, so the next prompt of a long conversation is read in full (a
  190k-token conversation took 6 minutes).
- A conversation replaced in its slot, or cleared to make room for a restore, is saved while every slot waits, ~0.64 s
  for 47k tokens, and a restore holds them ~0.8-1 s for 20k tokens; only one cleared for a new prompt is saved in the
  background.
- A prompt that cannot fit the pool beside the other conversations is refused at once with the context-size error
  ([§17](CHANGES.md#17-a-full-pool-the-slot-kept-the-prompt-that-cannot-fit-refused-room-made-for-a-restore)):
  the client has to compact it or wait. Idle conversations are cleared for it first, a longer one only once the prompt
  cache holds it ([§19](CHANGES.md#19-row-moves-in-one-launch-and-a-long-idle-conversation-cleared-once-it-is-cached)),
  so it is refused when the generating conversations hold the pool. Each agent's conversation counts against the one pool, so a client that
  lets every conversation grow toward 262,144 tokens (opencode's `limit.context`) runs into it with several agents.
- Two or more cards (`--split-mode tensor`) are supported by code reading only (the section after `## Run`): nothing
  here has run on more than one card.

## Credits and license

MIT, like [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml). Built on
[bitexploder/llama-superfast-v100](https://github.com/bitexploder/llama-superfast-v100); its credits for the models,
the draft vocabulary and ported kernels are in [`README-27B.md`](README-27B.md). DFlash2 is
[z-lab/dflash](https://github.com/z-lab/dflash). Some ideas were stolen from [ReVolta](https://github.com/Leonccaa/ReVolta). Part of this work was sponsored by
[Inscora](https://inscora.com).
