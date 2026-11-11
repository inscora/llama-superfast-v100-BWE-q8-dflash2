# What this fork changed, and why

The design notes of every change on top of [bitexploder/llama-superfast-v100](https://github.com/bitexploder/llama-superfast-v100),
in the order they landed, each with what it replaced, what was measured and the switches that turn it off or tune it.
All of it is in `main`; where a later section replaced an earlier one as the default, the section says so and the earlier
path stays behind its switch. The [README](README.md) lists these one line each and holds the build and run guide.

Measurements are on one V100 PCIe 32 GB behind a PCIe Gen3 x4 eGPU link, Qwen3.8-27B `UD-Q4_K_XL` with the DFlash2
draft, unless a section says otherwise.

## 1. Q8_0 KV cache on the Volta attention paths

Commits `bcf1877`, `5c15ec5`.

Upstream's fast paths, the CUTLASS 6-head prefill attention (`fattn-gqa6`) and the streaming decode/verify kernel
(`fattn-verify-stream`), took only an f16 KV cache. With `-ctk q8_0 -ctv q8_0`, every step fell to the generic kernels
over a temporary f16 copy of the cache.

- Both paths now read `q8_0` directly. The prefill kernel widens key blocks to f16 in shared memory; the verify kernel
  has a `KVQ8` variant with 8-byte row loads and aligned windows.
- Allocations no longer reserve the f16 copy on those routes.
- `test-backend-ops` has `q8_0` cases for both kernels, including the 27B's shapes.
- Switches: `LLAMA_FA_GQA6_Q8=0` and `LLAMA_FATTN_VERIFY_STREAM_Q8=0` fall back to the generic kernels;
  `LLAMA_FATTN_VERIFY_STREAM_Q8_KP` and `LLAMA_FATTN_VERIFY_STREAM_Q8_W` tune the verify kernel.

With sections 3 and 5, this took the server from upstream's 2 × 65,536 tokens (f16 cache, DFlash2) to one
245,760-token pool.

## 2. Faster prefill on repacked weights

Commits `044ef3b`.

- Long prompts expand each weight to fp16 for cuBLAS. The expansion now decodes straight from the repacked layout with
  16-byte stores, instead of one warp rebuilding the GGUF blocks while the rest of the block waits. It matches ggml's
  dequantizers bit for bit.
- Products of 129 to 191 tokens go to MMQ on a device-side unrepack instead of fp16 + cuBLAS. The Volta MMQ crossover is
  192 tokens (64 tokens: 0.5 vs 1.25 ms per 27B FFN product; 256 tokens: 1.72 vs 1.50 ms).
- The tensor-core matvec takes up to 128 tokens by default.
- Switches: `LLAMA_QPN_MAX_TOKENS` (default 128, at most 256); `GGML_CUDA_MMQ_MAX_BATCH` overrides the crossover;
  `LLAMA_QPN_MMQ=0` and `LLAMA_QPN_FAST_EXPAND=0` restore the earlier routes; `LLAMA_QPN_EXPAND_CHECK=1` runs the
  earlier kernels alongside and reports any difference.

## 3. Prompt work kept on the GPU

Commits `64107c5`.

- **DFlash2 features stay on the device.** The draft reads the target's layer inputs. For any decode that is one
  ubatch they now stay on the GPU, and the draft copies only the rows its 2,048-token window can see, device to
  device. Before, every prefill ubatch sent all of them to the host and back (about 2.3 GB for a 24k prompt). Verify
  steps with two slots take this path too. Switches: `LLAMA_DFLASH2_FEAT_DEV=0`, `LLAMA_DFLASH2_FEAT_DEV_ROWS`.
- **Checkpoints copy in the background.** A prompt checkpoint (the 27B's recurrent state, ~150 MiB) goes into a GPU
  staging buffer, then to the host on a second stream, so the next compute does not wait for the link. The draft's
  small state is read first. Switch: `LLAMA_CKPT_STAGE=0`.
- **No duplicate checkpoints.** A checkpoint at the position of an existing one is kept, not copied again; before, every
  new conversation re-saved the shared system prompt's checkpoint. Switch: `LLAMA_CKPT_KEEP_SAME=0`.
- **Cheaper prompt-cache saves.** A save reuses evicted entries' buffers, and moves checkpoints instead of copying them
  when the next prompt won't need them.
- **Two-user scheduler.** A lone prompt is no longer cut into small slices when no other slot is waiting
  (`LLAMA_TU_IDLE_JOB_MIN` sets the minimum slice).
- **Smaller draft compute buffer** with `--spec-draft-ubatch 512`: 1.7 GB freed at no measured speed cost.

## 4. Unified KV at full speed

Commits `146f3c4`, `9a20120`.

- **A per-batch window.** With `--kv-unified`, every batch attended over the other slot's cells too, masked out but
  still paid for. A batch of one sequence now reads only the cells from its first to its last (aligned to 256): the
  K/V views and the mask start there, and graph reuse keys on the window. Batches holding both slots skip it (their
  window would span everything, and the scan cost ~4% of two-slot decode). The window is on for the qwen35 line only,
  whose attention uses nothing else. Switches: `LLAMA_KV_WINDOW=0`, `LLAMA_KV_WINDOW_MIN_TOKENS`.
- **Both slots' masks on the GPU.** The device mask builder handled only sequence 0 of a unified cache, so slot 1's
  masks were built on the host and sent over the link every batch. It now keeps one cell mirror per sequence and
  builds mixed batches run by run.
- With the other slot holding a long conversation, prefill and decode in the unified cache run at the split cache's
  speed.
- Verified: device masks match the host's bit for bit (`LLAMA_KQ_MASK_DEVICE_CHECK=1`, 711 masks under two-slot load),
  and temperature-0 outputs are identical with the window on and off.

## 5. A smaller CUDA graph cache

Commits `0b1a768`.

- Under real traffic VRAM grew ~2 GB over the first ~25 minutes, which no fill test showed. `GGML_CUDA_MEM_TRACE`
  traced it to the CUDA graph cache: each instantiated graph of the 27B holds 8 MiB, decode graphs are keyed by cache
  length in steps of 256, and the cache kept up to 512 graphs for 600 s.
- The defaults are now 48 graphs for 120 s (`GGML_CUDA_GRAPH_MAX`, `GGML_CUDA_GRAPH_TTL_S`). A conversation moves to the
  next step every ~256 tokens, so old steps are rarely met again. Decode and TTFT are unchanged.
- Growth over a long run of mixed agentic traffic is now +784 MiB (temporary-buffer pools plus the capped cache),
  against ~2 GB with upstream's defaults, which made room for 32k more tokens of context.
- Diagnostics: `GGML_CUDA_MEM_TRACE=1` logs every drop of free device memory of 8 MiB or more and where it happened
  (`=2` also per node); `GGML_CUDA_POOL_LOG=1` logs each pool growth; `GGML_CUDA_GRAPH_STATS=1` logs the cached graph
  count at the server's debug level.

## 6. Cell placement in a unified cache

Commits `263398e`, `dd1665f`.

The window of section 4 is only as tight as a sequence's cells are contiguous. llama.cpp's head search fills free
cells from a moving head, token by token, so in a shared pool a conversation's cells scattered into holes left by the
other slot or by its own earlier prompts, and every later window spanned them.

- A ubatch of one sequence in a unified cache now goes right after that sequence's last cell when those cells are free.
  Otherwise it goes to the start of the largest free run that holds it, so a conversation stays one run of cells.
  The head search is the fallback.
- `263398e` does this for prompts and `dd1665f` for decode, which had restarted the search at cell 0 and decoded into
  small holes beside the other slot's conversation.
- Known trade-off: a new conversation placed right after the other slot's cells can block that slot's next prompt
  chunk, which then jumps past it and slows its long-prompt prefill. Placing new conversations in the middle of the
  free run avoids that but slows two-slot decode, so the current build keeps the tight placement; section 14 moves or
  clears what blocks a prompt's start.
- Switches: `LLAMA_KV_PLACE_MIN_TOKENS` (default 1) sets the smallest ubatch placed this way; `LLAMA_KV_WINDOW_LOG=1`
  logs placements and windows.

## 7. Build time

Commits `716cd08`.

`mmvq-qpn.cu`, upstream's tensor-core matvec, instantiates ~1,000 kernels in one file, which took 14 minutes on one core
and set the length of every build. Its kernels now live in `mmvq-qpn-impl.cuh`, with each launch entry point
instantiated in its own file under `ggml/src/ggml-cuda/template-instances/` (`generate_qpn_cu_files.py`), like upstream
llama.cpp's MMQ. The longest of those files takes ~90 s. Every kernel's PTX is unchanged.

## 8. The two-user scheduler with DFlash2 on a unified cache

The two-user scheduler reads a long prompt as a job of layer groups interleaved with the other slot's decode. It refused
every DFlash2 job (the draft needs the target's layer inputs) and did not run on a unified cache, so with DFlash2 a side
request's prompt waited until the other slot finished generating, or (`LLAMA_TU_FIX=0`) the generating slot fell to
~3 tok/s while the prompt was read.

- A job's groups copy each draft layer's input into a persistent f32 buffer (~210 MiB at `-b 2048`), which becomes the
  device feature path's source when the job publishes. Greedy outputs and draft acceptance are identical with the
  scheduler on and off.
- On a unified cache jobs are one ubatch (`-b <= -ub`), and the job's cell hash covers only its own sequence.
- With a slot generating, a read of at most 256 rows joins the decode round (`LLAMA_TU_GEN_JOB_MIN`), a longer one runs
  as jobs of at most 1,024 rows (`LLAMA_TU_GEN_JOB_ROWS`), and a shorter job's groups take more layers. A group's time
  grows with the depth of the prompt read so far (1,024 rows over 4 layers: 82 ms at 8k tokens, 138 ms at 125k), so
  the generating slot gets one round per 80 ms of a group's measured time (`LLAMA_TU_GEN_ROUND_MS`, at most
  `LLAMA_TU_GEN_ROUNDS_MAX` 4): one at 8k as before, two from ~60k. Beside a 123k-token read, at 60-90k of depth the
  generating slot 5.5 -> 8.1 rounds/s (30 -> 46 tok/s) and the read 375 -> 318 tok/s, at 90-120k 4.8 -> 7.6 (26 -> 43)
  and 327 -> 287; the whole read 308 -> 322 s. A 29k-token read is unchanged (529 tok/s, the generating slot 38).
  Shrinking the job's rows instead (`LLAMA_TU_GEN_GROUP_MS`, a target group time, default off) costs the read more:
  a group's cost has a part per group (the context's KV), and at 60-90k of depth a 55 ms target took the read from
  375 to 182 tok/s for the generating slot's 5.5 -> 9.3 rounds/s, an 80 ms target to 241 tok/s for 7.6. Idle, a job's
  slices and groups are longer (`LLAMA_TU_IDLE_SLICE_MS`, `LLAMA_TU_IDLE_GROUP`), so a lone long prompt reads within
  ~1% of the scheduler being off.
- A 24k-token side prompt beside a generating slot: it starts at once, and the generating slot keeps ~34 tok/s with
  gaps of at most 0.22 s.

## 9. Short prompts (32-token passes, no graph capture for prompts)

- The tensor-core matvec's single products take passes of 32 tokens (four 8-token tiles over one read and decode of the
  weights; each column summed in the same order as before, so outputs are bit-identical), and products up to 224
  tokens stay on it: prompts of 68 / 113 / 171 tokens 202 / 310 / 421 ms -> 168 / 240 / 361. `LLAMA_QPN_W32=0` restores
  passes of 16.
- A graph whose products with a weight take more than 64 tokens is no longer captured as a CUDA graph
  (`GGML_CUDA_GRAPH_MAX_TOKENS`): each prompt length was captured on its second occurrence and seldom replayed.

## 10. Checkpoints

- The checkpoint 4 tokens before a prompt's end, which the next turn of the conversation restores, is written from the
  recurrent state's rollback snapshot after the first token is out (`llama_memory_rs_view_back`), instead of reading the
  last 4 tokens as a decode of their own (~50 ms of every prompt's first token). `LLAMA_CKPT_TAIL_SNAP=0` restores it.
- Two checkpoints are kept on the device as well (`LLAMA_CKPT_DEV`, default 2): a slot's latest tail checkpoint and the
  last prefix restored from the host, such as a shared system prompt. A restore of a held prefix takes 1.5 ms instead of
  50 ms over the host link.

## 11. Compaction of a boxed-in prefix

A new conversation keeps the cells of the prefix it restored. When the other slot's conversation sits right after them,
its prompt went past that conversation and every window spanned it. Before a ubatch of a sequence that cannot grow in
place, up to 1,024 of its own cells move next to the free run its ubatch will use (`LLAMA_KV_COMPACT_MAX`): a 15.6k-token
prompt beside a 37k-token resident conversation, 708 -> 933 tok/s.

## 12. Four slots: the verify rows budgeted, the draft one width at a time

Commits `100cd2c`.

A verify step of the 27B costs ~40 ms whatever its width up to 8 rows, +15% to 16 rows and +35% to 32 (the tensor-core
matvec's 32-token kernel runs one block per SM). Three slots at the draft's 7 tokens made 24 rows a step, and a third
slot added nothing (93 vs 93 tok/s combined).

- The generating slots share a budget of verify rows, 8 up to two slots and 16 from three (`LLAMA_SPEC_WIDTH_SLOTS=8,16`):
  each drafts `budget / generating − 1` tokens, so two slots keep 7, three draft 4 and four draft 3.
- A slot alone always drafts the full 7. The older adaptive policy, which narrowed the draft to 3 tokens when its cost
  table said a wide verify was too dear, is off by default (`LLAMA_SPEC_ADAPT_WIDTH=1` turns it back on): its table
  predates the streaming verify kernel, which reads the cache once a step whatever the width. On a 148k-token
  conversation it drafted 3 tokens most rounds for 52 tok/s; the full width at the same depth gives 80–113 tok/s.
- The DFlash2 draft graph needs equal block widths in one ubatch, so the draft now runs one width at a time when slots
  draft different widths. Before, a narrowed slot beside full ones mis-blocked silently at two slots and asserted at
  three.

## 13. A shared pool of rollback snapshots

Commits `36bd290`, `f8aaffa`.

Each slot kept its own rollback snapshots of the recurrent state, as many rows as the draft's tokens: 150 MiB a row,
1.2 GB a slot at 7, so four slots at full depth would have cost 2.4 GB more (~70k tokens of context), and four slots at
a depth of 3 made a lone slot draft 3 as well (decode 74–82 tok/s instead of 90–116).

- `LLAMA_RS_POOL=12` makes the rollback rows one pool the slots share: a slot alone takes 7, two take 6 each, three 4
  and four 3. The runs are planned when the set of active slots changes, and a snapshot still to be read is applied to
  the live row before its run moves. Four live rows and 12 shared ones are 2.4 GB, the same as two slots before, so
  four slots fit the 245,760-token pool.
- A fifth live cell (`LLAMA_RS_SPARE=0` drops it, 150 MiB) keeps a prompt job's state where it is: llama.cpp gathers
  the cells of the sequences decoding together into one run by swapping cells, and with a cell a slot, three slots
  decoding while the fourth held a prompt job swapped the job's cell into their run, and the next decode copied another
  slot's state over it (the job's row check aborted the server about one run of the 8-in-flight probe in three). A prompt
  job's sequence now moves to the fifth cell, at the top, when the job begins (one copy of its state row), so the runs of
  the other sequences, all below it, never straddle it.
- The graph writes each sequence's snapshots as a copy of its own (the fused GDN cache write takes per-sequence bases),
  and the snapshot rows belong to the sequence, not to the storage cell: `f8aaffa` fixed a lookup through the cell,
  which read another sequence's snapshot once the cache had moved a sequence's state (three pinned slots generating
  together: one answer ended after 4 tokens; a released slot's cell tripped an assert in BetterBench's concurrency
  sweep).
- Greedy outputs of a lone slot are identical with the pool on and off (600 tokens, the same draft counts).
- Switches: `LLAMA_RS_POOL=N` (default 0: the per-slot layout), `LLAMA_RS_POOL_LOG=1`, `LLAMA_GDN_CACHE_FUSE=0` (the
  unfused writes, ~10% slower with several slots).

## 14. A prompt's cells stay one run, low in the pool

Commits `39dfacc`, `5e4a216`.

With four slots, a prompt's cached prefix (a shared system prompt, or the longest common prefix of an earlier prompt in
that slot) sat in cells far from the free run its body took, and every ubatch of the prompt read the window spanning
both: a 6k-token prompt read 50k cells, and cold 47k-token prompts fell from 716 to 491 tok/s. The per-ubatch compaction
of section 11 skipped those prefixes (its 1,024-cell cap).

- At a prompt's start the server asks the cache how many tokens the sequence can add while staying one run: the free
  cells after its last cell, or the largest free run when it has none.
- When the prompt does not fit, its whole cached prefix moves to the start of the largest free run that holds it and
  the prompt (a copy of a few ms, once per prompt; `LLAMA_KV_COMPACT_PROMPT_MAX` caps the cells moved).
- If it still does not fit, the idle slots' conversations are cleared, the least recently used first and only those
  shorter than this prompt, so a long resident conversation is never dropped for a shorter prompt
  (`LLAMA_KV_EVICT_IDLE=0` keeps them all; with `--cache-ram` they are saved to the prompt cache first).
- A new conversation takes the lowest free run that holds its whole prompt with room to grow
  (`LLAMA_KV_PLACE_MARGIN` cells, default 16,384), the largest run only when none does. The largest run alone sent new
  conversations to the top of an emptying pool, and the slots' decodes then read the whole used range: after four
  47k-token prompts, four short conversations decoded at 95 tok/s combined instead of 131.
- BetterBench's prefill sweep on four slots: 12k / 24k / 47k tokens 961 / 915 / 830 tok/s, against 920 / 836 / 716 on
  two slots before; its concurrency sweep 154 tok/s with four requests in flight.
- Before an idle conversation is cleared, the idle slots' cells and the prompt's own move down over the free cells
  below them (`llama_kv_pack_seqs`: the idle ones first, the prompt's on top, a device copy of the rows in ranges), so
  the free space is one run right after the slot that asked, and a conversation is cleared only when the pool is
  short, not fragmented. Four agents on the 262,144-token pool had cleared 37-52k-token conversations for follow-ups
  of 29-254 tokens, re-read minutes later beside three generating slots (135-427 s each). On a 65,536-token pool with
  four conversations, two replaced by shorter ones and one growing past the hole after it: without the packing two
  17.8k-token conversations were cleared and 41.6k tokens re-read (47 s); with it 23,739 cells moved in 24.4 ms and
  nothing was cleared, the others' follow-ups fully cached, their greedy answers identical to a fresh read.
  `LLAMA_KV_PACK=0` turns it off; the `kv_pack` log line gives the cells moved, the time and the room gained.
  The copies go to the device backend's stream and synchronize once: a move by a short distance is cut into chunks no
  longer than that distance, and the synchronous copy (one synchronize each) took 1.5 s for 23.7k cells moved by 2,048.

## 16. The prompt cache gives back the conversations that left the slots

With `--cache-ram` and `--no-cache-idle-slots`, the host prompt cache held what `kv_evict_idle` saved before clearing a
conversation and never read it back: slot selection refused every cache use with idle-slot caching off, and with it
on (the server's default) every idle slot's conversation was saved and cleared at every new task, then restored on its
next request, ~0.5 s each way over the eGPU link for 40k tokens, a pause of all slots. Now, with idle-slot caching off,
a slot chosen by LRU, or whose conversation the prompt would mostly discard, saves its own conversation and loads a
better entry when the cache holds one (`server_prompt_cache::load` takes only a strictly better entry and consumes
it), whether or not a peer is busy; nothing is saved or cleared otherwise. The LRU choice prefers an empty slot, so a
returning conversation does not displace a live one. On a 65,536-token pool: a 23.8k-token conversation cleared for a
34k-token prompt came back in 0.97 s with 23,819 tokens restored (26.4 s re-read without the cache), its 48 greedy
tokens identical. Saves cost ~0.2 s per 6k tokens and ~0.35 s per 24k. `LLAMA_CACHE_RETURN=0` restores upstream's
rule; the `prompt_cache` log line gives what was saved and loaded.

The cache must not take the host down. Its limit counts its state vectors and checkpoints, but its entries take
checkpoint blocks from the pinned reserve, the slots' next checkpoints then take fresh anonymous memory, and a recycle
pool keeps more: with `--cache-ram 15360` beside the 10.5 GB reserve the server held 17.5 GB of anonymous memory for
a 13.9 GB cache, and under BetterBench the host ran out (OOM-killed in production; on the dev build a watchdog stopped
it at 1.5 GB available, 16 requests into the concurrency sweep). A new entry now also needs the host's `MemAvailable`
to stay above `LLAMA_CACHE_RESERVE_MIB` (default 4096) after it: the oldest entries go first, and the save is skipped
when none are left. A conversation shorter than `LLAMA_CACHE_MIN_TOKENS` (default 8192) is not saved: every
single-stream request of the bench paid ~150 ms to first token saving the previous one-off conversation (its
recurrent state and tail checkpoint, ~300 MiB over the link; 248 ms for a 612-token one), and it never came back.
The production launcher runs `--cache-ram 15360 --no-cache-idle-slots` and 16 checkpoints per slot (the default 32
reserved 19.6 GB of pinned host memory).

An entry keeps only its conversation's newest `LLAMA_CACHE_CKPT_KEEP` checkpoints (default 2; negative: all). Each is
150 MiB of the recurrent state: a 48k-token entry was 4.5 GiB, about half of it checkpoints, so the 15 GiB cache held
three such conversations, and the entries' checkpoint blocks came from the pinned reserve, so the slots' next checkpoints
took fresh anonymous memory the cache does not count (under four vision agents the host went down to 3.1 GB available
and 2.9 GB of swap before the guard dropped two entries). A returning conversation resumes from its end or its newest
checkpoint; an older checkpoint only shortens the re-read of a history edited in the middle, which with the trim is
read from the start up to the edit (the same answer, read again: ~1 s per 900 tokens alone, 1.3-3 s under load).

Measured on a 65,536-token pool (`ckpttrim.sh`): a 38,359-token conversation with its follow-ups, saved when another
prompt took its slot and restored on its return, is a 2,602 MiB entry with every checkpoint and 1,844 MiB with the
newest two; the save takes 853 / 758 ms, the restore 528 / 498 ms, and both give back 38,308 tokens and the same 48
greedy tokens as before the save.

## 15. Rollback by replay: one state a slot instead of a snapshot a token

A verify step runs the draft's 8 tokens through the recurrent layers at once, and the rollback to the accepted count
needed the state after each of them: the snapshots of section 13, 12 shared rows of 150 MiB plus the four live rows,
2.4 GB. Now (the default; `LLAMA_RS_REPLAY=0` for the pool) every write keeps, in one row a slot, the state as it was
8 tokens before its end (the GDN kernel's slot 1, written from its registers as it passes that token, the input state
when the ubatch is shorter), and the GDN inputs of those last 8 tokens in two rings (the pre-conv projections and the
raw alpha/beta product: one `set_rows` a tensor a layer, 13 MB a slot). A rollback of n tokens replays the first 8-n
of them from the kept state into the live row before the slot's next step: a graph of the recurrent layers only, the
conv, the gate and the delta rule as the forward ran them, so the state is the one the forward would have reached.
The replay of a whole step reproduces the live state exactly (`LLAMA_RS_REPLAY_SELFTEST=1` checks it before every
rollback; `=2` also checks a replay of several slots against each slot's own).

- All slots with a pending rollback replay in one graph, of a length rounded up to a multiple of 4: the shorter
  replays are padded with a pad token the delta rule leaves the state alone for (the x ring's zero row, the
  alpha/beta ring's -inf row: gate 0, beta 0), and each slot's conv state is gathered from the three input columns
  ending at its last real token. A whole rollback replays 0 tokens. One graph a (length, slots) shape, its rows all
  inputs, so after its first run a replay is inputs and a CUDA graph launch (0.3 ms of host time); the kernel reads
  each slot's kept row in place and writes its live row itself (the `set_rows` after it is fused into the launch, the
  rows read from a device table), so a replay moves each state row once.
- The replay runs on its own backend instance (its own CUDA stream), launched from `seq_rm`, so its GPU time (the
  GDN kernel's serial pass over the replayed tokens, about 4 us a token a layer, and a dozen small kernels a layer)
  overlaps the draft's work instead of running ahead of the next verify graph. Events order it: the side stream first
  waits for the main stream's enqueued graphs (a pipelined decode may still read the rows), the main stream waits for
  the replay before its next graph, and host-side readers of the rows (row copies, the checkpoint view) wait on the
  host. `LLAMA_RS_REPLAY_SIDE=0` keeps the replay on the main stream, batched before the decode (then the cost is
  about 1.5 ms a round, 4% of a lone stream). BetterBench: 105.5 tok/s single stream against 106.1 with the pool's
  snapshots, 158.1 / 153.9 tok/s aggregate with four / eight requests in flight against the pool's 145.6 / 144.3;
  the rows are 10 instead of 17, 1.0 GB back, which is what takes the pool from 245,760 tokens to the model's
  262,144.
- The checkpoint 4 tokens before a prompt's end (section 10) replays into a scratch row instead of reading a snapshot.
- The pool's plan, locks and re-packing are off under the replay; the spare cell of section 13 stays.

## 17. A full pool: the slot kept, the prompt that cannot fit refused, room made for a restore

Four agents on the 262,144-token pool held about 105k, 93k, 58k and 22k tokens: two prompt reads ran for minutes
beside the generating slots and failed when the pool ran out ("Context size has been exceeded"), the 58k request's
retry took by prompt similarity the slot of a 94k-token conversation idle between two of its turns while another slot
was empty, and that conversation, back a second later, could not be restored (94k free cells were not there) and was
re-read beside three other reads.

- **Slot choice.** While a slot is empty, a request no longer takes by similarity a slot whose conversation it would
  mostly discard: it must keep `LLAMA_SLOT_SIM_KEEP` of it (default 0.5; 0: upstream's rule), else the choice falls to
  the least recently used slot, which prefers an empty one.
- **Admission.** A prompt is read only if its new tokens, a margin for its answer (`LLAMA_KV_ADMIT_MARGIN`, default
  2048) and what the other slots still have to read fit the pool's free cells after the packing and the allowed
  evictions (`llama_kv_n_free`); else it is refused at once with the context-size error, so the client can compact or
  retry instead of waiting for a read that cannot finish. It runs before each chunk of a long read, so of two reads
  that cannot both fit, one goes on. The generating slots' growth is not reserved: a prompt that fits now is read.
  `LLAMA_KV_ADMIT=0` turns it off; the `kv_admit_refused` log line gives what was needed and what was free.
- **Restore.** Before a cached conversation is restored into a slot, the idle conversations shorter than it are cleared
  (saved first, the least recently used first) until the pool holds it, and the idle conversations pack down; with no
  room the restore is skipped and the entry stays, and the request goes through the admission check like any prompt.
  `LLAMA_CACHE_ROOM=0` restores as before; the `cache_restore_room` log line gives the entry, the room and the outcome.

Measured on a 65,536-token pool with four slots and the prompt cache on, each case with its fix off then on:

| Case | Fix off | Fix on |
|---|---|---|
| A (24k) idle; B shares A's first 5k and adds 17k | B takes A's slot; A goes to the cache (0.5 s) and back (0.8 s), pushing B out | B takes an empty slot; A's follow-up finds A in place |
| Two slots generate 2,500 tokens while D (41k) arrives with 32k cells free | D is read for 75 s; the two generating requests fail with an HTTP 500 when the pool runs out | D is refused in 0.08 s (HTTP 400, the context-size error); both answers complete |
| A (24k) cleared to the cache, then back with 21k free | the restore fails, C is cleared anyway and A is re-read: 26.4 s | C is saved and cleared first, A is restored: 4.4 s |

With the pool full, the old code did not fail the late prompt but the slots already generating: their verify step found
no cell, the batch was halved down to 2 rows, and the speculative batch bookkeeping threw. The first build of the
admission check crashed the server on its first refusal (the two-user scheduler read the refused slot's task after
its release); `tu_remaining()` now returns 0 for a slot without a task and the refusal clears the round's reader pick.

## 18. Four agents at once: packs that pay, the joint step's empty tiles, image prompts as jobs

Commits `0dd2de6`, `bc26d4f`, `21c9092`, `fef620a` (and `89c8385`, the cache entries' checkpoints, in section 16).

Production logs under four agents showed three costs beside the reads themselves: a 49k-token prompt beside
generating slots packed the pool before each of its 2k chunks (58-661 cells moved, ~6k cells of room kept, up to ~1 s
each), a prompt holding an image held the other slots for each of its chunks, and the joint verify step of four slots
cost 1.6-2.4x a single slot's.

- **A pack only when it gives the room.** A dry run of the server's two steps (the idle conversations, then the
  requester on top) over a shadow of the cell table gives the room the requester would have, counting the compact that
  follows when the room after its cells is still short (`llama_kv_pack_room`); the pack runs only if that holds the
  request. Before a restore, it runs only when it lands the entry in one run. The first build of the check counted only
  the room after the requester's cells: a 3.8k-cell conversation under an idle one it could not pass had none, though
  20k cells were free on top, and two idle 18k-token conversations were cleared instead (the pack probe caught it).
- **Pack copies without stopping the GPU.** With a device backend the moves are queued on its stream: they run after
  every graph already queued there and before every graph queued after, the state reads and writes synchronize the
  context first, and nothing on another stream touches these rows. The stall was the pack waiting for the queued prompt
  batch, not the copy. `LLAMA_KV_PACK_SYNC=1` stops before and after the moves as before; a split cache stays synchronous.
- **The verify kernel skips the tiles no row can see.** A step over several sequences attends over the span of all
  their cells, each row only to its own sequence's, so most 64-key tiles are -inf for a whole 8-row block. A pre-pass
  lists the tiles any row of the block sees (`tiles_used`, one block per 8-row half and stream) and the streaming kernel
  walks that list. It helps one slot too: in a shared pool its span crosses the other conversations' cells.
  `LLAMA_FATTN_VERIFY_STREAM_SKIP=0` walks the whole span. `test-backend-ops` has the verify calls with the mask of a
  joint batch of up to four sequences (shared prefix, owned blocks across tile edges, a hole, causal batch rows).
- **Image prompts as scheduler jobs.** A prompt holding media was kept out of the two-user scheduler, so it waited for
  the running job, then held the other slots for each of its chunks. Its text now goes into jobs like any prompt's;
  each image chunk is decoded on its own when the fill reaches it. `LLAMA_TU_MEDIA_JOBS=0` keeps media prompts in the batch.

Measured on the dev build (one V100, 200 W):

| Case | Before (switch off) | After |
|---|---|---|
| Four ~30k-token conversations generating 1,200 tokens together (`joint4.sh`) | 83.8 ms a step, 132 tok/s in all | 76.6 ms a step, 151 tok/s in all |
| One of them generating alone, the other three idle in the pool | 53.6 ms a step, 62.7 tok/s | 48.0 ms a step, 70.1 tok/s |
| A conversation grows 16k past an idle one (`kvpack.sh`, 65,536-token pool) | 23,739 cells packed in 24.4 ms (stopping the GPU) | 23,739 cells in 13.8 ms; the same greedy answers from moved cells |
| Three slots stream while a fourth reads an image and a 20k-token log (`media_probe.py`, vision launcher) | the image prompt in 39.1 s; the others stall up to 3.7 s, 295 tokens streamed during the read | the image prompt in 56.8 s; the others stall at most 1.1 s, 3,422 tokens streamed during the read; the same answer |

The image prompt takes longer as a job for the reason a text prompt does: the generating slots keep their rounds while
it is read.

## 19. Row moves in one launch, and a long idle conversation cleared once it is cached

Commits `31dc568`, `21e36c7`.

Production under four agents, 28 minutes with the pool near full (four conversations of 47-82k tokens): 15 packs that held
every slot 6.7 s in all, the worst 1.53 s, and one of 4,316 cells 503 ms; a compaction 1.0 s; and a main thread's new
18,478-token prompt refused (8,532 cells free) beside an idle 121k-token conversation and an idle 36k one.

- **The moves in one launch.** Four slots generating at once interleave their conversations' cells in short runs; a pack
  of them was thousands of runs, two copies a run through the scratch for each layer's K and V, and the time went to
  issuing them (58,379 cells in long runs took 33 ms). `ggml_backend_cuda_kv_move_rows` walks the moves in order in one
  launch over every K and V tensor: block (t, c) takes tensor t's columns [c0, c1) of each row (a byte only moves to the
  same column of another row), reads each chunk into registers, syncs, then writes, so a move overlapping its source from
  below, and a later move over rows an earlier one vacated, stay exact. A compaction's destination is a free run, so it
  takes the kernel too and no longer stops the GPU. Moves the kernel does not take (out of range, overlapping from above,
  destinations out of order, reading rows an earlier move wrote) and a split cache copy run by run, where a run that does
  not overlap its destination now takes one copy instead of two. `LLAMA_KV_MOVE_KERNEL=0` copies run by run;
  `LLAMA_KV_MOVE_VERIFY=1` checks the first and the last tensor's moved rows against their sources, byte for byte (it
  stops the stream: for tests). The pack and the compaction log their run count.
- **A longer idle conversation cleared once the prompt cache holds it.** Only idle conversations shorter than the request
  were cleared, a rule from before the host prompt cache, when clearing a long one meant re-reading it for minutes; now
  one comes back in about a second. When the shorter ones are not enough, the longer ones are cleared too, the least
  recently used first, but only those the cache took; without a cache, or when the save fails, a longer one stays and the
  request is refused as before. The clearing also runs while the admission check would refuse, not only while the slot
  lacks room.
- A save's timing (allocation, state, checkpoints) is one log line (`prompt saved:`), for the save that still holds every
  slot.

Measured on the dev build (65,536-token pool):

| Case | Result |
|---|---|
| A fragmented pack (`kvfrag_eq.py`, three conversations generating together) | 357 runs, 35,597 cells in 16 ms, queued in 0.06 ms; every moved conversation's 48 greedy tokens identical to a fresh read |
| `LLAMA_KV_MOVE_VERIFY=1` on every pack and compaction of the probes | identical, byte for byte |
| Three idle conversations of 20k, 19k and 18k tokens, a new 15k one (`kvadmit2_probe.py`, cache 15 GiB) | the new one admitted, the 20k one saved and cleared, and back from the cache when it returns (20,015 tokens cached, 63 read, 1.6 s) |

## 20. A conversation cleared for room is saved in the background

A conversation cleared to make room (the idle ones of section 19) went to the prompt cache while every slot waited: the
device stopped, then one blocking copy per layer and cell range into pageable memory, after a zero-filled allocation and
a copy of its checkpoints. Measured on the dev build, a 20k-token conversation held the server 633 ms (allocation 178,
state 289, checkpoints 166), a fragmented 13k-token one ~800 ms, of which the draft's 40 MiB took 247 ms; production saw
1.29 s and 2.83 s.

- **The copy on a stream of its own.** `llama_state_seq_get_data_bg` writes the header and the cells' metadata at once,
  records an event on the compute stream and hands the tensor ranges to a worker thread, which waits for that event on
  its own stream, then per chunk of at most 32 MiB gathers the chunk's ranges into a device staging half in one launch
  (`ggml_backend_cuda_gather_bytes`), sends it to a pinned host half, and copies it into the entry while the other half's
  chunk crosses the link. The saver costs 2 x 32 MiB of the card's memory and 2 x 32 MiB of pinned host memory, made on
  the first save and shared by the target's and the draft's contexts. `llama_state_seq_bg_poll` reports the copy done.
- **The slot held until it is done.** The cleared conversation's slot keeps its cells and takes no task, pack or clearing
  while its copy runs (`held()`); its cache entry is `busy` (not loaded, dropped or reused). The prompt that needs the
  room waits for it, not refused, and every other slot goes on. Each round polls the copies; a finished one makes the
  entry usable and clears the slot (a failed one drops the entry).
- **Nothing zeroed, nothing copied twice.** An entry's buffers are no longer zero-filled (the worker's writes take the
  page faults), and its newest checkpoints move into it instead of being copied, the slot being cleared after.
- `LLAMA_CACHE_BG_SAVE=0` saves synchronously as before; `LLAMA_CACHE_BG_VERIFY=1` compares each background save with a
  synchronous save of the same, untouched cells (it stops the device: for tests). A save logs its main-thread time
  (`prompt saving in the background`), the copy its throughput (`state of ticket`).

Still synchronous: the save when a task replaces an idle slot's conversation in that slot, the clearing that makes room
for a restore, and the restore itself (~0.8-1 s for 20k tokens).

Measured on the dev build (65,536-token pool, cache 15 GiB):

| Case | Main thread, before | Main thread, after | Copy (background) |
|---|---|---|---|
| A 20k-token conversation cleared for a new prompt (`kvadmit2_probe.py`) | 420 ms | ~35 ms (66 ms on the first save, which makes the saver) | 815 MiB in 360 ms, 2.4 GB/s |
| A fragmented 13k-token one (`kvfrag_eq.py`, 9,920 ranges) | ~800 ms | ~35 ms | 589 MiB in 222 ms, 2.8 GB/s |
| Its draft state alone | 247 ms | 4 ms | 40 MiB, behind the target's copy |

Every background save of the probes was identical, target and draft, to a synchronous save of the same cells, and the
cleared conversation came back from the cache when it returned.

## 21. A recycled cache buffer that would grow is freed first

The cache reuses the state buffers of the entries it drops to make room. When the new conversation's state was larger
than the recycled buffer's capacity, the resize reallocated it and copied its old bytes, so the old and the new buffer
were both resident. In production, a 229k-token conversation's 7.9 GB save recycled the 7.5 GB buffer of the entry it
replaced; with the 15 GiB cache and the pinned checkpoint reserve, the host ran out of memory and swap, and the kernel
killed the server. The host-memory check (`LLAMA_CACHE_RESERVE_MIB`) had counted the recycled buffer as room, which it
was not while both copies lived.

A recycled buffer is now kept only when it holds the new state without growing and with at most an eighth (or 64 MiB)
of slack; otherwise it is freed before the host-memory check reads `MemAvailable`, and the new buffer is allocated
fresh (not zeroed: the background save's worker takes its page faults). The slack bound also stops a large buffer
reused for a small entry from keeping memory the cache's size does not count.

## 22. An outside review's four findings: the save's recurrent rows, short appends' scratch, the restore, the limit

An outside review of 59a46eb found four problems; each held up in the code and is fixed here.

- **A background save could read another conversation's recurrent rows** (`TAG_CACHE_BG_RS`). The save records the
  rows' addresses and the worker copies them later, attention cache first. Meanwhile the slot is held, but other
  sequences' batches move recurrent rows: `find_slot` gathers a batch's sequences into consecutive cells (swapping the
  cells between them) and a prompt job takes the top cell. A saved row could be rewritten before the worker read it,
  and the restored conversation would carry another's state, silently. Now a rollback still to replay for the sequence
  replays before the save records anything; the worker copies the recurrent rows first, ahead of any job's rest (the
  rest goes in 256 MiB slices, so another save's rows never wait behind a whole conversation); and the saving sequence is
  pinned: a gather, or a row copy, that would rewrite its row first waits for those rows (logged as `a recurrent row
  move waited`). Twenty-two background saves under `LLAMA_CACHE_BG_VERIFY=1`, two slots generating beside them, were
  identical to synchronous saves; the probes did not produce a row move during a save, so the wait itself is checked by
  the code, not by a run.
- **A short append at depth took up to 640 MiB of scratch** (`TAG_GQA6_KV_BUDGET`). The 6-head route sized its key
  blocks from the 128 MiB score budget alone, and a 17-token call's block reached 131,072 keys: 512 MiB of F16 keys
  and values beside the scores, on a card with ~0.6 GB to spare (the pool keeps its high-water mark). The F16 K/V now
  stay within 64 MiB too. Peak memory of the perf run 2,464 -> 2,014 MiB; 128 tokens and up unchanged (within 1%), 17
  and 32 tokens 4-7% slower (+0.6 ms a layer at 168k keys). 78 Q8_0 attention cases, two new ones over 40,960 keys, pass.
- **Making room could drop the conversation about to be restored** (`TAG_CACHE_RESTORE_PIN`). The slot's outgoing
  conversation was saved, and idle ones cleared, before the restore candidate was chosen, and the cache drops its
  oldest entries for room. The candidate is now pinned before those saves and unpinned after the restore; a save that
  then has no room is skipped. One slot, a cache holding one conversation, A then B then A: A was re-read in 21.8 s
  before, restored in 0.8 s now.
- **The cache limit did not hold while the remaining entries were being written** (`TAG_CACHE_LIMIT_HOLD`). With no
  idle entry left to drop, the new entry went in over the limit; it is now skipped. The host-memory check also takes
  the entries still being written off `MemAvailable` (their pages are not resident yet).

## 23. The remaining saves and the restore in the background

Section 20 moved the save of a conversation cleared for room to the background. Three copies still stopped the device
for every slot: the save of a slot's conversation when a task replaces it, the saves of the idle conversations cleared
so a restore fits, and the restore itself. All three now run on the saver's worker, and the task that needs them waits
(deferred) while the other slots go on.

- **The replacement save** (`TAG_CACHE_BG_REPLACE`) runs in the background and holds the slot, which keeps its
  conversation (the task may reuse a prefix): the checkpoints past the common prefix move into the entry, the newest
  others are copied, as before. The task is deferred; when the copy ends it comes back, finds the conversation in the
  cache and goes on.
- **The saves that make room for a restore** run in the background too; the task waits for their cells.
- **The restore** (`TAG_CACHE_BG_RESTORE`, `llama_state_seq_set_data_bg`): the entry's metadata are read and its cells and
  rows claimed at once (7.5 ms for a 32k-token conversation), then the worker copies its bytes into a pinned half, sends
  it to a device half and scatters it to the cells and rows in one launch (`ggml_cuda_scatter_bytes`), the next chunk
  staged meanwhile; recurrent rows first and pinned against row moves (`TAG_CACHE_BG_RS`). The slot is held until the
  copy ends, the entry busy until then and dropped after.
- When a background save or restore ends, every deferred task goes back to the queue (`server_queue::pop_deferred_all`),
  and the loop keeps polling while one runs even with every slot idle.
- `LLAMA_CACHE_BG_SAVE=0` keeps the saves synchronous, `LLAMA_CACHE_BG_RESTORE=0` the restore. `LLAMA_CACHE_BG_VERIFY=1`
  now also checks a restored state against its entry's bytes, background or synchronous (an entry from another slot
  differs only in the sequence id of each cell's metadata, which the check allows).

Measured on the dev build (three slots on a 65,536-token pool, `bgrestore_probe.py`): a 32k-token conversation comes back
while another slot streams; its slot's 20k-token conversation is saved, a second idle 20k one is cleared for room, then
it is restored.

| | Before | After |
|---|---|---|
| Longest pause of the streaming slot | 1,706 ms | 354 ms |
| The returning conversation's request | 2.1 s | 2.8 s (it waits for each copy in turn) |
| Main thread for the restore | the whole copy | 7.5 ms (the copy: 1.25 GB in ~0.8 s, 1.65 GB/s) |

Every background save and both restores, target and draft, were identical to their synchronous counterparts.

## 24. A small conversation that sets the top of the pool moves down

Section 23's deferred replacement save cost BetterBench's concurrency levels 10%: the 4-request level began with one slot
held 1.3 s while its 47k-token conversation was saved in the background, and the three other requests placed their new
conversations above those cells meanwhile. They grew in place from there (each request reuses its slot's prefix), so for
the rest of the level every step read ~47k masked cells: 77 ms a step instead of 69.5.

`seq_compact` (`TAG_KV_SPAN`), which runs before a single sequence's prompt batch, now also moves a conversation of at most
`LLAMA_KV_COMPACT_MAX` cells (default 1024) when it sets the top of the used cells and a free run at least
`LLAMA_KV_SPAN_MIN` cells (default 4096; 0: off) below it holds it and its batch: one launch of the row-move kernel.

BetterBench, same day, same launcher (aggregate decode, 1 / 2 / 4 requests; time to first token with 2 / 4):

| Build | Aggregate | TTFT |
|---|---|---|
| Section 21 | 91.3 / 111.8 / 158.3 tok/s | 0.31 / 0.37 s |
| Section 23 | 89.8 / 102.5 / 137.4 tok/s | 0.33 / 0.57 s |
| This one | 90.9 / 117.7 / 156.3 tok/s | 0.31 / 0.37 s |

## 25. Two ideas from 1Cat-vLLM, measured: lookup drafting and the exponential in PV

Measured on the branch 1cat-merge, then merged: the exponential in PV is on by default, lookup drafting stays off.

- **Lookup drafting beside DFlash2** (`LLAMA_SPEC_LOOKUP=1`, `TAG_SPEC_LOOKUP`; after 1Cat's lookup-augmented block
  drafting, itself from syv-ai/HyperQwen). After the draft, the request's own tokens are scanned backwards for the most
  recent earlier occurrence of their last 6 to 12 tokens. A strong match (8 or more) proposes what followed it: the
  drafter's rows that agree stay, the continuation takes over from the first disagreement or past the drafter's last
  token, up to the drafter's width (`LLAMA_SPEC_LOOKUP_WIDE` can ask for more when the slot generates alone, within its
  rollback depth, which caps it at 7 here). A replaced position carries a one-token draft distribution, so the rejection
  and block steps keep the target's output distribution. On a verbatim copy of a 1.6k-token file at temperature 1.0
  (`copy_probe.py`): DFlash2 alone 158 tok/s at 82% acceptance; with lookup 145 tok/s at 72%. DFlash2 stops drafting
  where its confidence drops, the lookup fills those rows, and a one-token row is accepted only with the target's
  probability for its token, where a calibrated row that agrees with the target is accepted with probability ~1. The
  extra rows cost verify time and add no accepted tokens. HyperQwen's +47% on copies was measured greedy, where a
  one-token row is exact.
- **The exponential in PV** (`TAG_GQA6_EXP_PV`, on by default, `LLAMA_FA_GQA6_EXP_PV=0` puts it back in the epilogue; after 1Cat's sm70_79t prefill kernel, which
  stores FP16 logits in its QK epilogue and applies the exponential as PV loads). The QK epilogue stores
  (logit - tile max) * log2(e) as FP16 and no sum; PV's A transform takes exp2 and accumulates each row's mass, folded
  into the running sum after PV. 152 attention cases pass. Prompt attention on the Q8_0 cache: +1.2% at 1,024 and
  2,048-token appends (41.4 -> 41.9 TFLOPS at 168k keys), -2.5% at 17 tokens. The epilogue's exponential and sum were
  not the limit of the QK kernel.

BetterBench (temperature 1.0, the production launcher), against the section 24 build the same day:

| build | weighted decode | aggregate t/s at 1 / 2 / 4 requests | TTFT p50 (1 / 2 / 4) | acceptance |
|---|--:|--:|--:|--:|
| section 24 build (reference, rsT run; rsW ran only the concurrency part) | 102.7 | 91.3 / 111.8 / 158.3 (rsW: 90.9 / 117.7 / 156.3) | 0.19 / 0.31 / 0.37 s | 46.7% |
| exponential in PV | 104.3 | 91.5 / 122.3 / 155.3 | 0.19 / 0.30 / 0.37 s | 47.5% |
| lookup drafting | 101.4 | 88.1 / 106.0 / 154.3 | 0.19 / 0.31 / 0.37 s | 46.4% |
| both | 103.0 | 89.5 / 118.7 / 153.8 | 0.19 / 0.30 / 0.37 s | 46.4% |

The exponential in PV lands inside the run-to-run spread of the reference (its three same-day runs gave 102.5 to
104.4 weighted; the 2-request column moves 10% between runs of the same build). Lookup drafting fired on 3% of the
drafting rounds (962 of 32,734, 3,753 tokens proposed, nearly all strong matches) and cost 1 to 5% at 1 and 2 requests,
as the copy probe predicted: the proposed rows are verified and mostly rejected.

At depth (`deepctx.py`: one conversation whose first turn is a 199,736-token synthetic code base, then copy / rename /
prose turns at that depth, two cycles of the four configurations, temperature 1.0):

| build | cold prefill of 200k tokens | copy at 200k (decode t/s, acceptance) | rename at 200k | prose at 200k |
|---|--:|--:|--:|--:|
| switches off | 393.7 / 395.6 s (505-507 tok/s) | 94-104, 73-79% | 107-110, 82-85% | 44-56, 26-36% |
| exponential in PV | 392.3 / 392.6 s (509 tok/s) | 99-105, 76-79% | 105-111, 81-86% | 43-56, 25-36% |
| lookup drafting | 396.0 / 396.2 s (504 tok/s) | 66-68, 46% | 69-71, 48-50% | 44-47, 25-28% |
| both | 392.6 / 393.2 s (508 tok/s) | 65-68, 45-46% | 68-71, 47-49% | 43-63, 24-43% |

A follow-up turn prefills only its own ~50 tokens (the reply before it is already in the cache); that pass takes
0.45-0.53 s in every configuration. The exponential in PV is 0.5% on the cold prefill, four runs against four, and
nothing on decode. Lookup drafting fires on 79% of the rounds of a copy at this depth (98 of 124, the module being
copied sits in the prompt) and loses a third of the decode speed: it drafts the full 7 rows every round where DFlash2
drafts about 5 and stops where its confidence drops, the accepted count per round comes out the same, and at 200k keys
each verified row is paid in attention. Lookup's 200k-token backward scan per round is not what costs; the rows are.

Adopted: the exponential in PV is the default (0.5% on a 200k-token prefill, inside the noise elsewhere, 152 attention
cases exact). The route's one-time log line names where the exponential runs, though the server does not surface that
line (it never has; the route is live: LLAMA_FA_GQA6_Q8=0 costs 1-3% of prefill at 16k-40k tokens). Lookup drafting
stays in the tree as an opt-in switch, for a greedy or a calibrated-row experiment.

## 26. A cancelled prompt job keeps the conversation

A client cancel that lands inside a two-user prompt job (a 2048-token group takes 1-5 s at depth, so most cancels of a
follow-up on a long conversation do) ended with the slot cleared: "the slot starts over". In production on 2026-11-02
that dropped a 166k-token conversation 3 s into a follow-up; the retry one second later found no slot to match, began
the 173k-token prompt from zero and was cancelled again at 15%; the next morning's query finished it in 289 s.

Now (`TAG_TU_CANCEL_KEEP`) the slot keeps the tokens before the job's batch, the cache keeps the ubatches that
completed (the context ends a cancelled job at a ubatch boundary) and the drafter's state, which never saw the batch,
matches the kept tokens. The next launch meets the same state as after a shortened prompt: the recurrent state sits
past the kept tokens, so it restores the checkpoint below them and re-reads from there, then removes the cells past the
prompt. A cancel costs the distance from the last checkpoint, not the conversation. The log line names the kept count:
`TU job: slot N cancelled after G groups, D of T tokens in the cache; the slot keeps K tokens`.

Probe (`cancel_probe.py`: a 41k-token conversation, then an 8k-token follow-up cancelled 1.5 s and 4 s in, each re-sent):
before, the retry read 51,719 and 62,106 tokens (the conversation gone each time); now 10,348 both times, the follow-up
plus the re-read from the last checkpoint, with the cancelled job's 2,048 cached tokens discarded as the state past
the kept prefix.

## 27. The streaming kernel takes prompts of up to 64 tokens

A profile of production's prefill (2026-11-05, branch `prefill-levers`, `profall.sh`): shallow, the weights GEMM is
62% of the kernel time at 75% of the card's peak, the per-call expansion of the repacked weights to fp16 9%, the gqa6
attention 12%, the GDN kernels 10%; at 174k tokens the attention is 61%. The attention route pads a prompt's rows to
128-row tiles: 17 rows run at 7.5 TFLOPS, 32 at 13, 128 at 35, 1,024 and more at 42.

Now (`TAG_FVS_ROWS`) the decode attention kernel, which streams the Q8_0 cache in 8-token blocks, takes prompts of up to
64 tokens (`LLAMA_FATTN_VERIFY_STREAM_MAX_TOK`, a multiple of 8 from 8 to 64; 16 was the cap, and `=16` restores it).
Above 16 tokens each block streams the keys on its own, as the second block of 9-16 tokens already did. At 172k keys a
17-row call takes 3.2 ms instead of the gqa6 route's 9.5, 32 rows 4.4 instead of 10.4, 64 rows 8.4 instead of 11.4;
128 rows and more stay on gqa6. The eval cases at 17 and 64 rows pass; a 64-row perf case joins the list.

Where it acts, a chat client's follow-ups on a 180k-token conversation (`levers_probe.py`, user messages of 25 to
1,016 prompt tokens, each prefilled on its own after its checkpoint):

| follow-up (prompt tokens) | 25 | 36 | 52 | 68 | 136 | 316 | 1,016 |
|---|--:|--:|--:|--:|--:|--:|--:|
| cap 16 (gqa6 from 17 rows) | 0.45 s | 0.46 s | 0.50 s | 0.51 s | 0.73 s | 1.26 s | 3.23 s |
| cap 64 | 0.37 s | 0.36 s | 0.42 s | 0.52 s | 0.73 s | 1.26 s | 3.23 s |

80-100 ms off every follow-up of up to 64 tokens at that depth (16 to 22%), nothing above the cap, nothing on cold
prefills (950 tok/s at 16k, 538 at 180k, both ways). BetterBench (temperature 1.0, the production launcher, 2026-11-06):
106.3 weighted against the rsZ baseline's 104.6 the same night, 92.4 / 120.2 / 161.0 aggregate tok/s at 1 / 2 / 4
requests against 92.5 / 119.4 / 158.8, prefill 955 / 944 / 820 tok/s at 2k / 16k / 64k against 950 / 942 / 818: inside
the spread, as the benchmark's turns are rarely that small.

The profile's other lever, a CUTLASS GEMM that reads the repacked weights itself instead of expanding them to fp16 per
call, is on branch `prefill-levers` (`qpn-gemm.cu`, `LLAMA_QPN_GEMM=1`, off): correct across 64 shapes, and 23% slower
on prefill at every depth, the per-value decode being 60% of a warp's MMA issue time with no spare warps to hide it.
Its CHANGES section 27 there holds the variants and the four-way BetterBench.

Cold prefill at depth on the shipped image (rsAA, `pf2.sh`, a fresh container per run, 2026-11-06, rsZ the same
morning): 128,009 tokens in 203.5 s (629 tok/s; rsZ 205.8 s, 622), 243,576 tokens in 530.7 s (459 tok/s; rsZ 531.2 s,
459); the slot's limit is 245,760 cells, so 244k is the most one conversation can hold. Both sit on the curve of the
earlier points (818 tok/s at 64k, 538 at 180k, 505-509 at 200k): the cap changes nothing on a cold prefill, which runs
in 2,048-row chunks on gqa6 either way.

The same two prefills with the card's 200 W power limit removed (2026-11-08, rsAB, the card at its 250 W board rating
under a new fan at full speed; `pf2u.sh`, the same fresh-container method): 128,009 tokens in 183.9 s (696 tok/s, +11%),
243,576 in 472.4 s (516 tok/s, +12%). The card drew 231-233 W on average through them, sat at the board limit on
25-33% of the 2-second samples (peaks 265-279 W), and topped out at 62-64 C, against 72 C under the old fan at 200 W.
Deep prefill is now bound by the board's own power limit, not by a cap; BetterBench's 1.5k-47k prefills gained 8-10%
the same way and its decode did not move (it draws about 180 W).

## 28. The attention's structure at depth (branch attn-depth)

The question from section 27: with the attention at 61% of a 174k prefill and 70% at 244k, and the gqa6 route at 42
TFLOPS against 1Cat's 71 (300 W SXM2) and 54 (their same kernel at 185 W), what does the card allow. Measured on the
1,024-row x 172k-key perf case (`test-backend-ops perf`, Q8_0 K/V, exponential in PV), the production shape.

**The GEMMs themselves.** cuBLAS on the two attention shapes (`cublasbench.cu`, fp16 in, fp32 accumulate, 6 heads
packed, batch 4): the 256-deep QK product reaches 52-58 TFLOPS (fp16 out; 42-56 fp32 out) and PV 57-66, both on
64x128 tiles (`volta_s884gemm_fp16_64x128_ldg8`). A 256-deep GEMM runs at half the card's peak however it is written;
the 1Cat design (cuBLAS for both products, a separate softmax pass over an FP32 score buffer) would, on this card,
pay the extra pass out of that and land where the fused-epilogue route already is. Nothing to port.

**Where the QK kernel's time went.** With the epilogue stripped in stages (`LLAMA_FA_GQA6_QK_EXP`, wrong results): the
main loop alone is 30 ms of the kernel's 59 per call (72 TFLOPS-equivalent, tensor pipe 80% active); CUTLASS's
epilogue was the other half: the FP32 tile through 49 KB of shared memory in pieces 12 ms, the FP16 score store 6.5,
the mask, maximum and sums 9. PV is 42 ms (254 registers, no spills, tensor pipe 56%, stalled on the score loads: 57%
L2 hits, the scores stream from DRAM); the exponential in its load costs 5% of it.

**The direct epilogue** (`TAG_GQA6_QK_DIRECT`, `LLAMA_FA_GQA6_QK_DIRECT=0` keeps CUTLASS's): the softmax straight from
the accumulator fragments (the Volta interleaved layout: 8 rows by 16 columns a lane). A first version with 4-byte
loads and stores was 8% slower than CUTLASS (twice the store sectors, an LSU queue stall). With a lane-pair exchange
giving each lane 8 consecutive keys, so the mask is read and the scores stored 16 bytes at a time: 41.6 -> 45.7 TFLOPS
at 1,024 and 2,048 rows, 34.3 -> 37.2 at 128 (QK 59 -> 50 ms); the exact tile maximum, bit-identical scores, the 60
six-head Q8_0 eval cases pass. What the 20 ms of epilogue still are (the same stripping): the score store 8 ms, the
mask and maximum 3.4, the cross-warp exchange 0.6, the rest arithmetic and conversions on a latency-bound chain with
two warps a scheduler. Variants that did not move it: statistics per 64-key warp tile with no barrier (-2%: four
times the scale reloads in PV); the mask tile parked in shared memory before the main loop (0); scores stored in
fragment order with V widened in the same key order, no exchange at all (0, and 1,000 fewer instructions a warp and
tile: the chain is not instruction-bound); the first wave's odd blocks delayed half a tile so the SMs' store phases
stop coinciding (0); evict-first score stores (0, though the score stream does push K and Q out of L2: 75% hits
against 90% without stores, 98 MB of DRAM reads a launch against 39); a 64-deep pipeline stage (CUTLASS's Volta
layouts do not support it: illegal access). On the PV side: prefetching the scores to L2 one to four stages ahead
(-1 to -4%), three or four key splits or a split count that follows the row count (-7% at 1,024 rows, noise at 128).

**The fused kernel, bounded before it was written.** A Volta flash-attention kernel (QK tile, softmax in registers,
P through shared memory into PV, O in registers) removes the score round trip (the 8 ms store, PV's DRAM-latency
loads) but has to hold O (64 rows x 256 dims in FP32 = 64 registers a thread with 8 warps) beside the QK accumulators
and the double-buffered fragments, so its warp tiles are 32x64, not 64x64. The QK main loop alone, re-tiled
(`LLAMA_FA_GQA6_QK_EXP=4`): 128x256 / 64x64 warps 30 ms, 128x128 / 64x32 35.5 (+18%), 64x256 / 32x64 38 (+26%),
64x128 / 32x32 47.5 (+58%). Both GEMM phases at +26% cost about what the round trip saves; the expected net is 0 to
+5% for a multi-day kernel with spill risk. Not written. On this card the two-kernel route with the direct epilogue
is within a few percent of what the register file allows at head dimension 256.

On the server (the dev build, one slot, a cold 128k prefill, `qkdirect-srv.sh`): 643 tok/s against 617 with the
switch off (199.1 s against 207.4), +4% where the attention is about half the prefill; at 244k, where it is 70%, the
cap is about +7%. Candidate for main.

BetterBench (temperature 1.0, the production launcher, 2026-11-06, rsAA the same session): rsAB (this epilogue) 102.2
weighted decode, 90.1 / 123.6 / 156.0 aggregate tok/s at 1 / 2 / 4 requests, prefill 951 / 948 / 826 tok/s at 2k /
16k / 64k; rsAA 101.1, 94.2 / 120.5 / 161.7, 951 / 945 / 816. Inside the spread everywhere but the 64k prefill (+1%),
as expected: the benchmark's depths are where the attention is a small share. No errors, peak VRAM unchanged (31.9 GB).
Adopted: main, image rsAB.

## 29. The Qwen4 line (Qwen3.8-Flash-Next, `qwen4exp`) on this fork, blind

Written for weights that have not shipped: Qwen4-27B was named at Apsara on 2026-09-22 with no specification, and
Qwen3.8-Flash-Next (2026-08-28) is the published preview of its architecture. Nothing in this section has run on a
GPU; the branch was built for the card and tested on the CPU only. The reference is
[bitexploder/fnfl-v100](https://github.com/bitexploder/fnfl-v100), read in full first.

**What the line is.** Qwen3.8-Flash-Next (HF `model_type` `qwen4_exp`, GGUF architecture `qwen4exp`): 48 layers of
width 2,560, a 248,320-token vocabulary, 262,144 tokens of context. The residual is four hyper-connection streams
mixed by low-rank (320) mixers in place of every layer norm. Three of every four layers are gated delta-net linear
attention (48 value heads of 128, 16 key heads, a width-4 convolution: the qwen35 line's GDN); every fourth is QSA,
sparse full attention: 24 query heads over 2 KV heads of dimension 256 with a sigmoid output gate, and an indexer
(4 heads of 128) that scores the history in blocks of 4 cells so each query attends at most 2,048 selected cells
plus the tail. The feed-forward is 512 routed experts (10 a token, 640 wide) and one shared expert; a 26.9 GiB
n-gram table (PLE) in host memory adds 16 rows a token before layer 1; an MTP block (one QSA layer with its own
experts, sharing the trunk's embeddings and head) ships as a separate GGUF and is the model's own draft. The
published quantizations are 75.8 GB (IQ3_XXS) and 111 GB (UD-Q4_K_XL): the model itself does not fit one V100 32 GB
without an expert cache in host memory. Whether Qwen4-27B keeps this architecture, and whether it is dense or a
mixture, is not public.

**What fnfl-v100 is, and what it says.** Not a llama.cpp port: a from-scratch CUDA engine (`fn/`) for exactly this
model on two V100s (one with its expert cache), with llama.cpp's server, tokenizer and GGUF reader around it.
Nothing of it transplants into ggml; what it gives is the architecture (`fn/model/shapes.h`, `docs/fnfl/engine.md`)
and the metadata contract it checks at load (`fn/glue/fnfl_glue.cpp`, `fn/loader/`): `qwen4exp.block_count`,
`.embedding_length`, `.context_length`, `.attention.head_count`, `.attention.head_count_kv`, `.expert_count`,
`.hyper_connection.count`, `.ple.conv_kernel`, `.ple.ngram_size`, and the tensors `blk.N.hc_attn_{norm,down,up}`,
`ffn_{gate,up,down}_exps`, `ffn_{gate,up,down}_shexp`, `ffn_gate_inp`. Our converter writes every one of them
(`conversion/qwen4exp.py`, `gguf-py/gguf/constants.py`): the Unsloth and ISTA-DASLab files fnfl-v100 serves are
what it produces.

**What was already in the tree.** The llama.cpp side of the line came with upstream (Unsloth build b11030 of
ggml-org/llama.cpp, 2026-09-18, through bitexploder's import) and bitexploder's V100 work: the converter and the
GGUF keys above, `src/models/qwen4exp.cpp` with its hyper-connection, GDN, QSA, PLE and MTP graphs, the attention
and indexer caches of `llama-memory-hybrid-idx.cpp`, the compact selected-KV decode path
(`models/qwen4exp-qsa-compact.h`), the block-sparse prompt kernel ported from PentaCoxian's patch set
(`ggml-cuda/qsa-attn.cu`, `qsa-select.cu`, `qsa-union.cu`), the draft vocabulary ranking
`models/draft-vocab-qwen3.8-flash-next.txt` (65,536 ids, taken by default for a qwen4exp MTP draft), and the tests
`test-qsa-compact`, `test-qsa-select`, `test-qsa-host-meta`. Upstream bitexploder served it on two cards with the
MTP head (`--spec-type draft-mtp --spec-draft-n-max 2 -ctk bf16 -ctv bf16`, layer split), never on one. This fork's
27B-only paths were already gated off the line: the KV window and the whole-cache decode graph (`qwen35` only,
sections 4 and 27), the MTP chain on the device (`llama_mtp_chain_supported`, qwen35), block verification's
default, the two-user scheduler (`supports_layer_groups`, qwen35: it logs "not engaged" and prompts run as plain
batches).

**What this branch adds** (branch `qwen4`, each tagged in the source):

- `TAG_QWEN4_KV` (`llama-context.cpp`): the line's sparse paths read an F16 or BF16 cache only (the block-sparse
  prompt kernel and the compact decode path both refuse a quantized K/V), and the 27B's Q8_0 routes (section 1) are
  built for six heads of 256. A quantized `-ctk`/`-ctv` on a qwen4exp model would have left every QSA layer on the
  whole-cache dense attention with an F16 widening of the cache per call, silently. The context now replaces it by
  F16 with a warning; `LLAMA_QWEN4EXP_KV_QUANT=1` keeps the requested types. At 262,144 tokens the line's F16 cache
  is 12 layers x 2 heads x 256 x 2 (K and V) x 2 bytes = 24 KiB a token, 6.3 GB, plus the indexer's keys.
- `TAG_QWEN4_MTP` (`common/arg.cpp`, `common/speculative.cpp`): with a draft file and no `--spec-type`, a qwen4exp
  target drafts with its MTP head before the draft file's metadata is consulted (an MTP file, or one that says
  nothing, is MTP; a file of another draft kind is kept and said so); a `--spec-type` given is kept as given.
  DFlash2 stays the 27B's draft and its default. The launcher's `PHASE=mtp` is the same choice made explicit.
- `TAG_QWEN4_ROUTES` (`fattn-gqa6.cu`, `fattn-verify-stream.cu`): the two 27B kernels say once, in the log, when a
  head configuration is not theirs (the head dimension, the query heads per KV head: the line has 12 per KV head,
  the routes pack 6 of 256) and leave it to the generic Volta kernels. They were silent about it; every other
  assumption of theirs is a shape test that already fell through. The line goes through them to the mma kernel
  for prompts (12 % 4 == 0) and the tile or vec kernels for decode, with the QSA selection in front of both.
- `TAG_QWEN4_DENSE` (`qwen4exp.cpp`, `constants.py`): a qwen4exp file with no `expert_count` loads and runs the qwen35
  line's plain SwiGLU (`ffn_gate`, `ffn_up`, `ffn_down`) in both graphs, the trunk's and the MTP block's, and the
  converter maps `mlp.{gate,up,down}_proj` for it. Should Qwen4-27B be a dense model of this architecture under the
  same class name, it converts and loads; should it come as a new `model_type`, the converter needs its class
  registered (`@ModelBase.register`) and the C++ side a look at what changed, and this section is the map.
- `test-llama-archs` generates the dense variant beside the MoE one (qwen4exp left `moe_mandatory`), so the dense
  graph is built, saved, reloaded and decoded on the CPU by the existing tests.

**What was verified, on the CPU.** The CUDA build (sm_70, the production configuration: `GGML_CUDA_FA_ALL_QUANTS=OFF`, tests on) and a CPU-only build (`GGML_CUDA=OFF`) of `llama-server`, `test-backend-ops` and the tests below compile clean of new warnings. On the CPU build: `test-llama-archs` synthesises a qwen4exp model in both variants (dense 15.7 MB, MoE 19.2 MB; before this branch it generated neither, qwen4exp being MoE-mandatory and not MoE-implemented), and `test-save-load-state` decodes each, saves and restores its state and checks the continuation (7 checks a model, both pass); `test-generate-models` and `test-save-load-state` over every architecture pass, `test-spec-rejection`, `test-spec-draft-grammar`, `test-qsa-host-meta` and `test-qsa-select` (its CPU half) pass. One pre-existing failure, unrelated and the same on the unpatched main: `test-recurrent-state-rollback` on the synthetic qwen35 aborts in the CPU build (`llama_memory_recurrent::state_write`, `rs_depth` assert, reached from `common_prompt_checkpoint::update_tgt`); the CUDA build's test binaries link libcuda and cannot run without a card. The GPU halves of the QSA tests, `test-backend-ops` against the CUDA backend, and everything below are not run.

**What is blind, to verify when the weights arrive** (in this order, each a short run on AgentK with the user's go):

1. The architecture of the weights against this section: `model_type` and class name, dense or mixture, head counts
   and head dimension, hyper-connection count, PLE present or not, MTP block present or not. A head dimension other
   than 256 takes the QSA kernels out (`QSA_ATTN_D` in `qsa-attn.cu`, the compact path's 256-cell padding in
   `llama_qsa_compact_width`); the dense fallback still runs.
2. Memory on one card: the weights, the F16 cache above, the indexer cache, the PLE table in host RAM (26.9 GiB on
   Flash-Next; AgentK's 32 GB holds a 15 GiB prompt cache and 10 GB of checkpoints today), the expert cache if the
   model is a mixture (`-ncmoe`, upstream's `llama_moe_cache`: not in this tree).
3. The GPU halves of `test-qsa-compact` and `test-qsa-select` on the V100, then `test-backend-ops -o FLASH_ATTN_EXT`
   with the line's shapes (24 heads over 2, F16 cache, mask) on the generic Volta kernels.
4. A conversation past 8k with `-ctk f16 -ctv f16 --kv-unified --parallel 4`: the hybrid-idx memory under the fork's
   cell placement, packing and prompt cache (sections 6, 14, 16; `llama-kv-cache.cpp` already keeps the indexer's
   ext through a save), a slot's save and restore, a checkpoint resume.
5. The MTP draft: acceptance in the log with rejection sampling at draft temperature 1.0 (unchanged), the serial
   draft loop (the chain is qwen35-only and says so), `--spec-draft-n-max 2` as upstream served it, the 65,536-id
   subset taken by default.
6. The upstream fixes to `qwen4exp` after b11030 that this tree does not carry, by hand if they matter (our file is
   2,131 lines against upstream's 1,514 and no longer merges): the conv-state tail written without a copy
   (ggml-org 749db7f0), the QSA selection built without the KQ mask (5c311426, 27b7a8a0), the layer inputs for
   DFlash drafters (d9ee007a), the KV shift (6023692b; this fork never shifts), the MTP hidden state saved with a
   slot (0f35d6d7), and tensors merged at convert time (2280f0fd: our loader expects them unmerged, so a GGUF made
   by a newer upstream converter will not load here).
7. The one-time route lines of `TAG_QWEN4_ROUTES` are printed for any Volta attention call outside the 27B's shape,
   the vision projector's included: one line each, once a process.

Launch, when the files are in place: `PHASE=mtp KV_TYPE=f16 DRAFT_N_MAX=2 MODEL_DIR=<dir> MODEL_FILE=<model.gguf>
MTP_FILE=<mtp.gguf> ./run-superfast.sh` (the `dflash` phase needs a DFlash2 draft, which the line does not have;
`KV_TYPE=q8_0` would be replaced by F16 with the warning above).

## 30. Host work in the decode round (2026-11-09: decode +0.4 to +0.8%, outputs identical)

A review of the decode round on 2026-11-09 (the card was off; section 29 is the `qwen4` branch's). The round is GPU
bound: 94% busy, 81% of that in the weight kernels (the 2026-10-23 profile). What the host adds is small, and most of
it hides under the drafter's injection, about 1.9 ms of GPU time issued after each verify. It shows when the host
window is longer than that: requests with a tool list, whose sampler clones are heavy, and deep conversations. Four
changes, each exact (the served tokens do not change) and each behind a switch that restores the previous code:

- **A clone of a grammar shares its rules** (`TAG_GRAMMAR_SHARED_RULES`, `LLAMA_GRAMMAR_SHARE_RULES=0` copies). The
  rules never change after init (upstream's own `TODO: shared ptr`). A clone copied every rule, then compared every
  stack element with every rule element to redirect its pointers, a loop with no break: stack elements x rule
  elements comparisons a clone. A speculative round clones the sampler for the draft's rejection picks and for the
  block verification. Now the clone holds
  the same `shared_ptr` and its copied stacks already point into it. `test-grammar-integration` gains `test_clone`
  (a clone taken mid-input: the same stacks, its own progress, a clone of a clone, a clone outliving its source).
- **The sampler's saved copy is cloned only when it can be read** (`TAG_SPEC_SAVE_LAZY`, `LLAMA_SPEC_SMPL_SAVE=1`
  clones every round). The copy is read on one branch, a partial acceptance that restores the checkpoint, which needs
  a rollback deeper than the context supports. At most n_draft tokens roll back, so with a recurrent target at an
  n-max within the rollback depth (every production round) the branch is unreachable and the third clone of the round
  was dead work.
- **The slot's tokens are copied for the draft only when something reads the copy** (`TAG_SPEC_PROMPT_LAZY`,
  `LLAMA_SPEC_PROMPT_COPY=1` copies every round). `spec_prompt` was rebuilt every round, one pass over every token of
  the conversation; its readers are the ngram drafters and the lookup of section 25
  (`common_speculative_needs_prompt`). A draft model keeps its own state.
- **The drafter's injection is issued with the target's decode** (`TAG_SPEC_PROCESS_EARLY`,
  `LLAMA_SPEC_PROCESS_EARLY=0` restores the two yields). `common_speculative_process` ran in a queue yield of its own
  after the target's synchronize, the metrics in between: the GPU sat idle from the end of the verify graph until the
  injection's host preparation was done. It now runs in the decode's yield, before the synchronize. It needs nothing
  the synchronize delivers: the feature rows go device to device, copied on the target's stream behind its graph, and
  the drafter's stream waits on that copy's event; the host path's getter synchronizes itself. DFlash drafters only
  (`common_speculative_process_early_ok`); `LLAMA_TU_DIAG` keeps the two yields.

Verified here: the CPU and CUDA servers build; `test-grammar-integration`, `test-llama-grammar`,
`test-grammar-parser`, `test-json-schema-to-grammar`, `test-chat`, `test-chat-auto-parser`, `test-chat-peg-parser`,
`test-chat-template`, `test-sampling`, `test-spec-rejection`, `test-spec-draft-grammar` and `test-reasoning-budget`
pass with the defaults and with `LLAMA_GRAMMAR_SHARE_RULES=0`.

Measured on the card on 2026-11-09 (the V100 uncapped, production's launcher settings, `~/src/q8bench/hc.sh`,
results in `results/hc/`). The probe `hcprobe.py` sends 21 fixed requests on one slot, each at temperature 0 or at
temperature 1.0 with a fixed seed: plain chat, a five-turn agent loop with a 14-tool list, a JSON schema, a 1,400-token
file written through a tool call, and a 99,645-token conversation with a verbatim copy, a tool-call write and prose
on top. Three servers ran it with the round timers on: this build with the switches at their defaults (new), this
build with the four switches restored (old), and the previous build.

- Outputs: all 21 requests are identical, character for character, across the three, and across six more passes
  without timers (new and old alternating, three each). No assert or CUDA error in any log; 65 C at most.
- Round time, new against old, microseconds of host round on the same rounds (the outputs being identical, the two
  ran the same rounds): plain chat 41,185 -> 41,026 (-0.4%), the agent loop 42,390 -> 42,050 (-0.8%), the tool-call
  write 42,050 -> 41,740 (-0.75%), the JSON schema 48,370 -> 47,380 (-2.0%), 100k deep 51,640 -> 51,230 (-0.8%).
  GPU busy time is unchanged; the idle time between graphs falls by 0.2 to 0.6 ms a round. The old code and the
  previous build agree within 0.1% on every request: the switches do restore it.
- Decode rate over the alternating passes (three each, ranges that do not overlap on any request but each
  server's first): plain +0.4%, the agent loop +0.7%, the tool-call write +0.7% (142.6 against 141.6 tok/s), the JSON
  schema +2.1% (90.9 against 89.0).
- Where it comes from. Sampler clones a round: 3 -> 2 without a grammar, 2 -> 1 with one. A clone with the 14-tool
  grammar costs 20 us instead of 60 (`cs.clone` 120 -> 20 us a round); with the schema's grammar about 530 us
  either way, because there the stacks are the cost, not the rules. The early injection does not overlap the verify:
  the drafter's decode synchronizes its own stream while it prepares, and that stream waits on the copy behind the
  target's graph, so `spec.process` now blocks for the verify's 34 ms and the synchronize after it finds the work
  done. What it saves is the hand-off: the gap from the target's graph to the drafter's injection falls from
  286-311 us to 102-106 us a round on every request. The gap from the injection to the draft falls with the clones
  (357 -> 280 us in the tool-call write) and, at 100k, with the token copy (556 -> 355 us).
- Several slots at once (`hcconc.sh`, `hcconc2.sh`): a 21k-token prompt read beside a generating slot gives the
  same two outputs on new and old, four runs out of four. Three slots decoding greedily at once are not repeatable
  on any build: of eight runs each against one reference, old matched in three, new in two, the previous build in
  four, always at the same few positions (tokens 35, 36, 70, 120, 191), which move with the order the requests
  arrive in. No error in any of them.
- Seen on the way, not part of this change: with the JSON schema (an array of exactly 12 objects) the grammar work
  of a round takes 6.3 to 7 ms (`srv.accept_block`), against 0.3 ms inside a tool call's body, and the round's idle
  time is 9 ms against 2. Acceptance in the tool-call bodies was 70% shallow and 80% at 100k.

Left out, from the same review: the next draft launched right after the acceptance (the host work it would hide
already runs under the injection), the forward's 96 ring `set_rows` fused into the conv-slots launch (a CUDA kernel
change, 0.3-0.6 ms, to write with the card at hand and `LLAMA_RS_REPLAY_SELFTEST`), the graph compatibility walk
cached per uid (20-50 us). Two findings of the review are not host work and are not in this branch: a drafted row
loses the rejection step once a tool call's grammar has triggered (`common_sampler_rejection_row_ok`), and the
streaming verify kernel reads a Q8_0 cache at about 306 GB/s where it reads an F16 one at 659 (section 1's numbers).

## 31. Rejection sampling inside tool calls, and the graph check kept per graph (2026-11-09: tool-call bodies +4.5%, decode +0.3 to +0.4%)

Two of the review's items of section 30, measured on 2026-11-09 with the same method (fixed requests, identical outputs
where the change is exact, alternating servers).

**The rejection step under a tool call's grammar** (`TAG_SPEC_REJECTION_GRAMMAR`, `LLAMA_SPEC_REJECTION_GRAMMAR`,
default 1; 0: as before; 2: under any grammar). Once a lazy grammar had triggered, a drafted row lost the rejection
step: the draft proposed its argmax and the verify sampled the target and compared, so a tool call's body, where a
coding agent writes its files, decoded on the sample-and-match rule (with `LLAMA_SPEC_REJECTION=0` its output did not
change by a token). `common_sampler_sample` draws from the chain and keeps the token when it fits the grammar; only a
token that does not fit sends it to a second draw with the grammar first. So on a row where every candidate with
probability fits the grammar, the chain's distribution is the row's distribution, and the rejection step is exact
there. Now such a row takes the step (the token step and the block verification both), the draft's q is its candidates
that fit, renormalized, and a row with a candidate that does not fit keeps the two draws. Not at temperature 0 (the
step cannot gain there) and by default not under a grammar that is not lazy: a JSON schema's grammar costs more to
read than the step gains (a schema of 12 fixed objects lost 4% with it, 86.9 against 90.7 tok/s).
- The write-tool task of section 30's probe at temperature 1.0, eight seeds: 142.0 tok/s against 135.8 (+4.5%),
  acceptance 0.710 against 0.669. Every row read through the grammar had only candidates that fit (3,082 of 3,082 in
  the first three seeds), so no row fell back.
- For scale: the same file written as plain content decodes at 160 tok/s with the rejection step (acceptance 0.787)
  and 143 without (0.699); the tool call's JSON string is harder for the draft.
- The distribution is the target's: over 60 seeds the first tool called was write 51, grep 8, glob 1 with the step and
  50, 9, 1 without. The greedy requests and every request without tools give the outputs of the previous build,
  character for character; a seeded request with tools gives another sample of the same distribution, as the
  rejection step always did for plain text.
- `test-spec-draft-grammar` checks the draft's pick under a triggered grammar (q holds only candidates that fit, sums
  to 1, the pick is accepted by the grammar; nothing fits: no pick).

**The graph compatibility walk kept per graph uid** (`TAG_CG_COMPAT_CACHE`, `LLAMA_CG_COMPAT_CACHE=0` walks every
time). Before each launch the backend walked every node of the graph to decide whether it may run as a CUDA graph:
98.5 us on the verify graph's 4,574 nodes, 17 us on each of the drafter's two, all before the launch, the GPU idle.
The answer depends on the nodes' ops, shapes and weights only, so a graph with the uid the walk last ran on keeps its
answer (the reuse `ggml_cuda_graph_update_required` already trusts). Outputs identical on the 17 requests; the walk
falls to 1 us a round, the host round by 118 us (41,716 -> 41,598), and the decode rate rises 0.3 to 0.4% on every
request in two alternating passes each (143.6 against 143.2 tok/s on the tool-call write).

Measured and not built: the forward's ring writes. An nsys node trace of decode shows 96 `k_set_rows` a round at
3.1 us each on the target's stream, 0.3 ms a round, 0.7%: the ceiling of fusing them, half in the conv-slots kernel
(the x ring) and half in the gated delta net kernel (the alpha/beta ring).

The streaming verify kernel on a Q8_0 cache (branch `q8-fvs`, experiments, all off): 462-473 us a call at 65,536 keys
x 8 tokens, an F16 cache 394-399. Fewer or wider loads do not help (two 8-byte loads with a word select 502-506; the
scale by a shuffle 476-481; per-thread offsets 461-467), and neither does reading each row from global memory once
with the raw rows staged in shared memory (K rows 462, V rows 462-468, both exact). The probes say why: the time is in
the PV warps' product loop, their reads and their widening (one read a lane on V alone: 375; no widening on V: 432),
and the QK warps have room. A probe that has the QK warps load and widen 3/8 of each V tile into shared memory as F16,
the PV warps reading it with one load, runs in 429 us (4/8: 431, 5/8: 443): about 8.5% of the kernel, which is 2% of a
round at 100k and 3% at 200k. The exact version needs the V part's hand-over between the two groups.

## 32. The verify kernel on a Q8_0 cache: the V widening shared between the warp groups, and a cheaper tile list (2026-11-10: kernel -8.4%, decode at 100k +1.1 to +1.5%, outputs identical)

Measured 2026-11-10 with `test-backend-ops perf` (4 KV heads x 6, 65,536 keys, 8 tokens, Q8_0), each variant
alternated with the kernel as it was, then on the server with fixed requests whose outputs must be identical.

| kernel | us a call |
|---|---|
| as it was (section 31) | 463-471 |
| the V widening shared (`TAG_FVS_Q8_VB`) | 440-442 |
| the fast tile list (`TAG_FVS_SKIP_FAST`) | 450 |
| both (the default now) | 426 |
| no tile list at all (`LLAMA_FATTN_VERIFY_STREAM_SKIP=0`), V shared | 412 |

**The V widening shared** (`LLAMA_FATTN_VERIFY_STREAM_Q8_VB`, 3 by default; 2 or 4: another share; 0: off). Section
31's probe built exactly. The QK warps load and widen the first 3/8 of each V tile (24 rows, all 256 dims, a lane
taking the 8 dims it takes of a K row) into shared memory as F16; the PV warps read those rows as fragments with one
load each and widen only the other 40 themselves. The words, the widening and the product order are the PV warps'
own, so the output is the same bit for bit. The part has one buffer, handed over at a named barrier: a PV warp
consumes the previous tile's part first in its iteration and arrives; a QK warp, its P written, waits there for the
four PV warps before it lays down the next tile's. Shares 2 and 3 measure the same (442-447), 4 is slower (457): the
QK warps become the slow group. The barrier costs nothing (the same kernel without it, wrong output: 441-444), and
two buffers with no barrier (`_VB_MODE=2`, share 2 only: two buffers of 3/8 do not fit a block's shared memory)
442. The probe's 429-432 is not reached, by 10 us, with the loads and stores in the probe's own order and place
and with the tile list off as well (412 against 401): what the probe gains beyond the exact kernel is not found.

**The tile list** (`LLAMA_FATTN_VERIFY_STREAM_SKIP_FAST`, default on; 0: as before). The list of used tiles
(`TAG_FVS_SKIP`) lets a batch of several conversations read only its own cells. On one conversation every tile is
used and the list is pure cost, and it was 25 us of a 465 us call (made and not used, `_SKIP=2`: 462; not made: 437;
reading it in the kernel: nothing). One block of 256 threads read every 8-key chunk of every row, 32 dependent loads
a thread, and one warp compacted the flags tile after tile. Now the block has 1024 threads; every tile is first
tried on one chunk (its first, in the half's last row), and only the tiles that chunk leaves open are read in full;
and every warp compacts its 32 tiles at once, its place in the list from the counts of the warps before it. The same
list in the same order. 450 us with it: 13 us are left over the kernel with no list, part of it the launch itself
(the perf test launches without a CUDA graph). Not done: one list for all the layers of a round (they share the mask).

**Checked.** The Q8_0 eval cases against the CPU reference: 39 of 39, share 3 three passes, then both changes two
passes. On the server (dev build, 4 slots, unified cache), the 17 shallow requests and the 4 at 100k of section 30's
probe, the kernel as it was (`_SKIP_FAST=0 _Q8_VB=0`) against the defaults, two alternating passes at 100k: every
output identical, no errors. Decode at 100k: 122.8 / 122.5 -> 124.2 / 124.3 tok/s (verbatim copy), 127.7 / 125.9 ->
129.2 / 127.8 (write call, temperature 1), 109.0 / 108.9 -> 110.3 / 110.4 (prose, temperature 1): +1.1 to +1.5%.
Shallow requests are unchanged (the kernel is a small share there).
The release image against the one before it (the launcher, the same 21 requests): every output identical, no
errors; at 100k 124.0 -> 125.6, 122.9 -> 124.5, 108.6 -> 110.0 tok/s. Merged to main.
