// launcher and route of the streaming verify kernel (fattn-verify-stream.cuh)
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-verify-stream.cuh"

// The verify and draft-step call (1 to 8 tokens, a GQA group of 6, D = 256) over an F16 cache of at least LLAMA_FATTN_VERIFY_STREAM keys
// (default 4096; -1 or 0: off, the mma routes are then used). And 9 to 16 tokens (two slots'
// verify), as pairs of 8-token blocks over the same K/V stream (LLAMA_FATTN_VERIFY_STREAM_W16=0: 1 to 8 only, as before). And over
// several KV streams (the non-unified cache, one per sequence: ne[3] of Q, K, V and the mask), each block on its own stream
// (LLAMA_FATTN_VERIFY_STREAM_NS=0: one stream only, as before). And over a Q8_0 cache (K and V both Q8_0, read in place and
// widened to F16 in registers; LLAMA_FATTN_VERIFY_STREAM_Q8=0: not taken, the generic kernels then read an F16 copy).
static bool fvs_q8() {
    static const bool q8 = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_Q8"); return e == nullptr || atoi(e) != 0; }();
    return q8;
}
static int fvs_min_keys() {
    static const int min_keys = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM"); return e ? atoi(e) : 4096; }();
    return min_keys;
}
static int fvs_max_tok() {
    static const int max_tok = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_W16"); return e == nullptr || atoi(e) != 0 ? 16 : 8; }();
    return max_tok;
}

// the kernel's limits, for the KV cache's whole-cache decode graphs: the least key count it takes and the most tokens
// (min_keys = 0: the kernel is off, or a CUDA device is not Volta)
void ggml_cuda_fattn_verify_stream_limits(int * min_keys, int * max_tok) {
    int mk = fvs_min_keys();
    for (int id = 0; id < ggml_cuda_info().device_count; ++id) {
        if (ggml_cuda_info().devices[id].cc != GGML_CUDA_CC_VOLTA) {
            mk = 0;
        }
    }
    *min_keys = std::max(mk, 0);
    *max_tok  = fvs_max_tok();
}

bool ggml_cuda_fattn_verify_stream_applies(const ggml_tensor * dst, int device) {
    const int min_keys = fvs_min_keys();
    if (min_keys <= 0 || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_VOLTA) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f || sinks != nullptr || ggml_get_op_params_i32(dst, 4) != 0) {
        return false;
    }
    const bool kv_q8 = fvs_q8() && K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0;
    if (Q->type != GGML_TYPE_F32 || !((K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) || kv_q8) || !mask ||
            mask->type != GGML_TYPE_F16) {
        return false;
    }
    const int max_tok = fvs_max_tok();
    static const bool multi_stream = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_NS"); return e == nullptr || atoi(e) != 0; }();
    const int64_t ns = Q->ne[3];
    if (Q->ne[0] != fvs::D || K->ne[0] != fvs::D || V->ne[0] != fvs::D || Q->ne[1] < 1 || Q->ne[1] > max_tok ||
            ns < 1 || (ns > 1 && !multi_stream) || K->ne[3] != ns || V->ne[3] != ns ||
            K->ne[2] == 0 || Q->ne[2] != 6*K->ne[2] || V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1]) {
        return false;
    }
    if (K->ne[1] < min_keys || K->ne[1] % FATTN_KQ_STRIDE != 0 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1 ||
            mask->ne[3] != ns) {
        return false;
    }
    // 16-byte loads of Q, the mask and an F16 K/V; a Q8_0 K/V is read in 2-byte words, from whole blocks
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        const size_t align = kv_q8 && (t == K || t == V) ? 2 : 16;
        if ((uintptr_t) t->data % align != 0 || t->nb[1] % align != 0 || t->nb[2] % align != 0 || t->nb[3] % align != 0) {
            return false;
        }
    }
    const size_t kv_elt = kv_q8 ? ggml_type_size(GGML_TYPE_Q8_0) : sizeof(half);
    return Q->nb[0] == 4 && K->nb[0] == kv_elt && V->nb[0] == kv_elt;
}

void ggml_cuda_flash_attn_ext_verify_stream(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    // by default the PV warps load and store the last 8 of the 16 K rows per lane (KP = 8; the same mma sequence and
    // reductions, so bitwise); LLAMA_FATTN_VERIFY_STREAM_KP=0 restores the earlier kernel (all of K in the QK warps)
    static const bool kp8 = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_KP"); return !e || atoi(e) != 0; }();
    constexpr size_t nbytes_shared = sizeof(fvs::smem_t);
    // the shared-memory limit is a per-device attribute (the macro keeps one flag per device), so it is raised on every
    // device that launches the kernel, not once inside the static below (the second card of --split-mode tensor failed the launch)
    CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false>),          nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 8>),       nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, true>),           nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false,  0, true>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false,  8, true>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32, true>), nbytes_shared);
    CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 40, true>), nbytes_shared);
    static int occ = [] {
        int o = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, kp8 ? fvs::verify_stream<6, false, 8> : fvs::verify_stream<6, false>, fvs::NTH, nbytes_shared));
        return std::max(o, 1);
    }();

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_tok  = Q->ne[1];
    const int n_head = Q->ne[2];
    const int nkvh   = K->ne[2];
    const int n_kv   = K->ne[1];
    const int n_strm = Q->ne[3]; // KV streams, one grid row each
    const int ntiles = n_kv / fvs::TK;
    const int nsplit = std::max(1, std::min(occ*nsm / nkvh, ntiles)); // one wave (of 8-token blocks)
    // src[6] is the used key count of a decode graph over the whole cache (read by the kernel). The split above is computed
    // from the whole cache's tiles; under this assert it equals the one computed from any used count of at least min_keys keys
    const int32_t * kv_count = dst->src[6] ? (const int32_t *) dst->src[6]->data : nullptr;
    if (kv_count) {
        GGML_ASSERT(occ*nsm / nkvh <= fvs_min_keys() / fvs::TK);
    }
    // 9 to 16 tokens, two 8-token blocks per (KV head, split), side by side; the split is the 8-token call's, so each token's
    // result is bit-identical to it
    const int nhalf  = (n_tok + 7)/8;

    // [TAG_FVS_SKIP] the used tiles of each (stream, 8-token half): a batch of several sequences reads only its rows' tiles,
    // not the span of all their cells (LLAMA_FATTN_VERIFY_STREAM_SKIP=0: the span). One tiles_used launch per call
    static const bool skip = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_SKIP"); return !e || atoi(e) != 0; }();
    // =2, timing probe: the list is made and not used (the kernel reads the span)
    static const bool skip_unused = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_SKIP"); return e && atoi(e) == 2; }();
    const int tstride = std::min(ntiles, fvs::TMAX);
    const bool use_list = skip && ntiles <= fvs::TMAX;
    const bool pass_list = use_list && !skip_unused;
    ggml_cuda_pool_alloc<int32_t> tlist(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> tcount(ctx.pool());
    if (use_list) {
        tlist.alloc((size_t) n_strm*nhalf*tstride);
        tcount.alloc((size_t) n_strm*nhalf);
        // [TAG_FVS_SKIP_FAST] LLAMA_FATTN_VERIFY_STREAM_SKIP_FAST=0: the list as before (one block of 256, every chunk read, one warp compacts)
        static const bool skip_fast = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_SKIP_FAST"); return !e || atoi(e) != 0; }();
        if (skip_fast) {
            fvs::tiles_used<true><<<dim3(nhalf, n_strm, 1), 1024, 0, stream>>>((const char *) mask->data, mask->nb[1], mask->nb[3], n_tok, nhalf,
                n_kv, kv_count, tlist.ptr, tcount.ptr, tstride);
        } else {
            fvs::tiles_used<false><<<dim3(nhalf, n_strm, 1), 256, 0, stream>>>((const char *) mask->data, mask->nb[1], mask->nb[3], n_tok, nhalf,
                n_kv, kv_count, tlist.ptr, tcount.ptr, tstride);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    ggml_cuda_pool_alloc<float>  parts(ctx.pool());
    ggml_cuda_pool_alloc<float2> meta(ctx.pool());
    if (nsplit > 1) {
        parts.alloc((size_t) n_strm*n_tok*n_head*nsplit*fvs::D);
        meta.alloc((size_t) n_strm*n_tok*n_head*nsplit);
    }

    static const bool stub = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_STUB"); return e && atoi(e) != 0; }(); // ceiling probe, wrong output
    // a Q8_0 cache has no stub variant: the probe is for the F16 data path. Its K-row split between the warp groups is its own
    // switch, since the widening moves work into whichever group stores the K rows: on Q8_0 the QK warps keep all of K (KP = 0;
    // LLAMA_FATTN_VERIFY_STREAM_Q8_KP=8 gives the PV warps half, as on F16), and the codes load as aligned windows (ld_row8;
    // LLAMA_FATTN_VERIFY_STREAM_Q8_W=0: four 2-byte loads). V100, 4 KV heads x gqa 6, 65,536 keys, 8 / 4 / 1 tokens:
    // KP 8 + 2-byte 517 / 490 / 480 us, KP 0 + 2-byte 479 / 456 / 439, KP 8 + window 504 / 478 / 465, KP 0 + window 466 / 445 / 427
    // (F16: 407 / 389 / 375)
    static const bool q8_kp8 = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_Q8_KP"); return e && atoi(e) != 0; }();
    static const bool q8_w   = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_Q8_W");  return !e || atoi(e) != 0; }();
    const bool kv_q8 = K->type == GGML_TYPE_Q8_0;
    // timing probes of the Q8_0 path, wrong output (LLAMA_FATTN_VERIFY_STREAM_Q8_PROBE = 1: one load a lane, 2: no widening, 3: both)
    static const int q8_probe = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_Q8_PROBE"); return e ? atoi(e) & 7 : 0; }();
    if (kv_q8 && q8_probe != 0) {
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 +  64, true>), nbytes_shared);
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 128, true>), nbytes_shared);
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 192, true>), nbytes_shared);
    }
    // [TAG_FVS_Q8_LD3] LLAMA_FATTN_VERIFY_STREAM_Q8_LD = 1: the Q8_0 loads at per-thread offsets with d from the block's
    // first lane by a shuffle (three loads a lane); 2: the same offsets, d loaded (four). KP 0 both
    static const int q8_ld = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_Q8_LD"); return e ? atoi(e) : 0; }();
    if (kv_q8 && q8_ld != 0) {
        GGML_ASSERT(K->nb[1] % 4 == 0 && V->nb[1] % 4 == 0);
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 256, true>), nbytes_shared);
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 512, true>), nbytes_shared);
    }
    // [TAG_FVS_Q8_ROW1] LLAMA_FATTN_VERIFY_STREAM_Q8_ROW1=1: each Q8_0 K row read from global memory once (16-byte loads, raw
    // rows staged in shared memory); the V rows as before. Its own launch: the strip needs more shared memory
    static const int q8_row1 = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_Q8_ROW1"); return e ? atoi(e) : 0; }();
    // [TAG_FVS_Q8_VB] LLAMA_FATTN_VERIFY_STREAM_Q8_VB = 2, 3 or 4: the QK warps load and widen that many eighths of each
    // Q8_0 V tile for the PV warps (exact: the same output bit for bit). 3 by default; 0: off. Its own launch: the part needs more
    // shared memory. A V row is laid out as a K row (the QK lanes read it as they read K)
    static const int q8_vb = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_Q8_VB"); const int v = e ? atoi(e) : 3; return v >= 2 && v <= 4 ? v : 0; }();
    // LLAMA_FATTN_VERIFY_STREAM_Q8_VB_MODE = 0: one buffer, handed over at barrier 3; 2: two buffers, no barrier (share 2
    // only: two buffers of 3 eighths do not fit a block's shared memory); 1: timing probe, wrong output (one buffer, no barrier)
    static const int q8_vbm = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_Q8_VB_MODE"); return e ? atoi(e) : 0; }();
    if (kv_q8 && V->type == GGML_TYPE_Q8_0 && q8_vb != 0 && q8_vbm != 0 && q8_probe == 0 && q8_ld == 0 && q8_row1 == 0 && q8_w && !q8_kp8 && K->nb[1] == V->nb[1]) {
        constexpr int VB = 32 + 131072;
        const bool db = q8_vbm == 2;
        const auto   fn = db         ? fvs::verify_stream<6, false, VB + 262144 + 1048576, true> :
                          q8_vb == 2 ? fvs::verify_stream<6, false, VB + 262144 +  524288, true> : fvs::verify_stream<6, false, VB + 65536 + 524288, true>;
        const size_t sm = db ? fvs::sm_bytes_vb(2, 2) : fvs::sm_bytes_vb(q8_vb == 2 ? 2 : 3);
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, VB + 262144 + 1048576, true>), fvs::sm_bytes_vb(2, 2));
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, VB + 262144 +  524288, true>), fvs::sm_bytes_vb(2));
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, VB +  65536 +  524288, true>), fvs::sm_bytes_vb(3));
        fn<<<dim3(nsplit*nkvh*nhalf, n_strm, 1), fvs::NTH, sm, stream>>>(
            (const float *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            (float *) dst->data, parts.ptr, meta.ptr, n_tok, n_head, nkvh,
            Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1], n_kv, kv_count, nsplit, scale*1.4426950408889634f, nhalf,
            Q->nb[3], K->nb[3], V->nb[3], mask->nb[3], pass_list ? tlist.ptr : nullptr, pass_list ? tcount.ptr : nullptr, tstride);
    } else if (kv_q8 && V->type == GGML_TYPE_Q8_0 && q8_vb != 0 && q8_probe == 0 && q8_ld == 0 && q8_row1 == 0 && q8_w && !q8_kp8 && K->nb[1] == V->nb[1]) {
        constexpr int VB = 32 + 131072;
        const auto   fn = q8_vb == 2 ? fvs::verify_stream<6, false, VB + 262144, true> :
                          q8_vb == 3 ? fvs::verify_stream<6, false, VB +  65536, true> : fvs::verify_stream<6, false, VB, true>;
        const size_t sm = fvs::sm_bytes_vb(q8_vb);
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, VB + 262144, true>), fvs::sm_bytes_vb(2));
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, VB +  65536, true>), fvs::sm_bytes_vb(3));
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, VB, true>),          fvs::sm_bytes_vb(4));
        fn<<<dim3(nsplit*nkvh*nhalf, n_strm, 1), fvs::NTH, sm, stream>>>(
            (const float *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            (float *) dst->data, parts.ptr, meta.ptr, n_tok, n_head, nkvh,
            Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1], n_kv, kv_count, nsplit, scale*1.4426950408889634f, nhalf,
            Q->nb[3], K->nb[3], V->nb[3], mask->nb[3], pass_list ? tlist.ptr : nullptr, pass_list ? tcount.ptr : nullptr, tstride);
    } else if (kv_q8 && q8_probe == 4) {
        // probe 4, wrong output: part of each V tile loaded and widened by the QK warps, read as F16 by the PV warps
        // (LLAMA_FATTN_VERIFY_STREAM_Q8_HB = 3, 4 or 5 eighths of the tile; 4 by default)
        static const int hb = [] { const char * e = getenv("LLAMA_FATTN_VERIFY_STREAM_Q8_HB"); return e ? atoi(e) : 4; }();
        const auto fn = hb == 5 ? fvs::verify_stream<6, false, 32 + 16384 + 32768, true> :
                        hb == 3 ? fvs::verify_stream<6, false, 32 + 16384 + 65536, true> : fvs::verify_stream<6, false, 32 + 16384, true>;
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 16384, true>), fvs::SM_BYTES_HB);
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 16384 + 32768, true>), fvs::SM_BYTES_HB);
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 16384 + 65536, true>), fvs::SM_BYTES_HB);
        fn<<<dim3(nsplit*nkvh*nhalf, n_strm, 1), fvs::NTH, fvs::SM_BYTES_HB, stream>>>(
            (const float *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            (float *) dst->data, parts.ptr, meta.ptr, n_tok, n_head, nkvh,
            Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1], n_kv, kv_count, nsplit, scale*1.4426950408889634f, nhalf,
            Q->nb[3], K->nb[3], V->nb[3], mask->nb[3], pass_list ? tlist.ptr : nullptr, pass_list ? tcount.ptr : nullptr, tstride);
    } else if (kv_q8 && q8_probe == 0 && q8_ld == 0 && q8_row1 == 2 && V->nb[1] == 272) {
        // [TAG_FVS_Q8_ROW1V] =2: the V rows read once (the PV warps' strips), the K rows as usual
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 8192, true>), fvs::SM_BYTES_V);
        fvs::verify_stream<6, false, 32 + 8192, true><<<dim3(nsplit*nkvh*nhalf, n_strm, 1), fvs::NTH, fvs::SM_BYTES_V, stream>>>(
            (const float *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            (float *) dst->data, parts.ptr, meta.ptr, n_tok, n_head, nkvh,
            Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1], n_kv, kv_count, nsplit, scale*1.4426950408889634f, nhalf,
            Q->nb[3], K->nb[3], V->nb[3], mask->nb[3], pass_list ? tlist.ptr : nullptr, pass_list ? tcount.ptr : nullptr, tstride);
    } else if (kv_q8 && q8_probe == 0 && q8_ld == 0 && q8_row1 == 1 && K->nb[1] == 272) {
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 1024, true>), fvs::SM_BYTES);
        fvs::verify_stream<6, false, 32 + 1024, true><<<dim3(nsplit*nkvh*nhalf, n_strm, 1), fvs::NTH, fvs::SM_BYTES, stream>>>(
            (const float *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            (float *) dst->data, parts.ptr, meta.ptr, n_tok, n_head, nkvh,
            Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1], n_kv, kv_count, nsplit, scale*1.4426950408889634f, nhalf,
            Q->nb[3], K->nb[3], V->nb[3], mask->nb[3], pass_list ? tlist.ptr : nullptr, pass_list ? tcount.ptr : nullptr, tstride);
    } else if (kv_q8 && q8_probe >= 5) {
        // V-only probes, wrong output: 5 one load a lane, 6 no widening, 7 one global word and three shared-memory reads
        const auto fn = q8_probe == 5 ? fvs::verify_stream<6, false, 32 + 64 + 2048, true> :
                        q8_probe == 6 ? fvs::verify_stream<6, false, 32 + 128 + 2048, true> : fvs::verify_stream<6, false, 32 + 4096, true>;
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 64 + 2048, true>), nbytes_shared);
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 128 + 2048, true>), nbytes_shared);
        CUDA_SET_SHARED_MEMORY_LIMIT((fvs::verify_stream<6, false, 32 + 4096, true>), nbytes_shared);
        fn<<<dim3(nsplit*nkvh*nhalf, n_strm, 1), fvs::NTH, nbytes_shared, stream>>>(
            (const float *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
            (float *) dst->data, parts.ptr, meta.ptr, n_tok, n_head, nkvh,
            Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1], n_kv, kv_count, nsplit, scale*1.4426950408889634f, nhalf,
            Q->nb[3], K->nb[3], V->nb[3], mask->nb[3], pass_list ? tlist.ptr : nullptr, pass_list ? tcount.ptr : nullptr, tstride);
    } else
    (kv_q8 && q8_probe == 0 && q8_ld == 1 ?
 fvs::verify_stream<6, false, 32 + 256, true> :
     kv_q8 && q8_probe == 0 && q8_ld == 2 ? fvs::verify_stream<6, false, 32 + 512, true> :
     kv_q8 && q8_probe == 1 ? fvs::verify_stream<6, false, 32 +  64, true> :
     kv_q8 && q8_probe == 2 ? fvs::verify_stream<6, false, 32 + 128, true> :
     kv_q8 && q8_probe == 3 ? fvs::verify_stream<6, false, 32 + 192, true> :
     kv_q8 ? (q8_w ? (q8_kp8 ? fvs::verify_stream<6, false, 40, true> : fvs::verify_stream<6, false, 32, true>)
                   : (q8_kp8 ? fvs::verify_stream<6, false,  8, true> : fvs::verify_stream<6, false,  0, true>)) :
     stub  ? fvs::verify_stream<6, true> : kp8 ? fvs::verify_stream<6, false, 8> : fvs::verify_stream<6, false>)<<<dim3(nsplit*nkvh*nhalf, n_strm, 1), fvs::NTH, nbytes_shared, stream>>>(
        (const float *) Q->data, (const char *) K->data, (const char *) V->data, (const char *) mask->data,
        (float *) dst->data, parts.ptr, meta.ptr, n_tok, n_head, nkvh,
        Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2], mask->nb[1], n_kv, kv_count, nsplit, scale*1.4426950408889634f, nhalf,
        Q->nb[3], K->nb[3], V->nb[3], mask->nb[3], pass_list ? tlist.ptr : nullptr, pass_list ? tcount.ptr : nullptr, tstride);
    CUDA_CHECK(cudaGetLastError());

    if (nsplit > 1) {
        const dim3 blocks_num_combine(n_tok, n_head, n_strm);
        flash_attn_combine_results<fvs::D><<<blocks_num_combine, fvs::D, nsplit*sizeof(float2), stream>>>(parts.ptr, meta.ptr, (float *) dst->data, nsplit);
        CUDA_CHECK(cudaGetLastError());
    }
}
