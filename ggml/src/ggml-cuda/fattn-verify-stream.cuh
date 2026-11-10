#pragma once
// The 1..8-row verify's / draft step's attention over an F16 cache at depth, streamed on Volta (sm_70).
//
// Second design. One block (8 warps, one per SM) per (KV head, slice of the sequence), blockIdx.x = kvh + n_kv_heads*split.
// All NR = 48 query rows of the group (8 tokens x G = 6 heads) are resident; Q is staged once in shared memory as f16, pre-scaled
// by scale*log2(e). Keys are streamed in tiles of TK = 64. The two phases run on separate warp groups, one tile apart:
//   warps 0..3 (QK group): K tile (64 keys x 256 dims) loaded ROW-COALESCED (a warp instruction reads one 512 B key row) into
//     registers one tile ahead, stored to shared memory (8-half row pad), read back as m8n8k4 A fragments (LDS.128 per lane),
//     S = K Q^T for 32 keys x 24 rows per warp, then the online softmax; P = exp2(S - m) (f16) and the per-row rescale go through
//     shared memory (P double-buffered).
//   warps 4..7 (PV group): O^T += V^T P^T for tile j-1 while the QK group does tile j. V is read from global memory straight into
//     the A fragments, 16 B per lane (a lane's 8 dims are two M blocks of 4), one tile ahead in registers; warp w owns 64 dims.
//   One __syncthreads per tile, plus a 128-thread barrier in the QK group (the max exchange, which also frees the K tile).
// Key labels in the QK fragments are permuted so that the LDS.128 of K is bank-conflict free; P is written in natural key order.
// The online softmax is exact-exp2 in fp32 per row; row sums are kept per lane (rescaled with the row's factor) and reduced once
// at the end. Each block writes an unnormalised fp32 partial plus (max in natural-log units, sum) per row, combined by
// flash_attn_combine_results; with one split it writes the normalised result directly.
// STUB = true is the ceiling probe: the same data path, both mma phases, the split and the combine, with the softmax replaced by
// P = S/1024 (no max, no exp, no rescale, no sums).
// KVQ8 = true reads a Q8_0 cache instead. Each lane loads the same 8 dims of a K or V row as on F16 (the row-coalesced K
// load, the V fragment load), from its Q8_0 block: the block's d and 8 int8 codes, as five 2-byte loads (a 34-byte block is
// only 2-byte aligned). The raw words wait in the same registers the F16 data would (kreg, vreg) and are widened to F16 where
// those are consumed (the K tile's store to shared memory, the V fragment), so the loads stay one tile ahead and the mma
// sequence is unchanged. The widening is exact up to one rounding of q*d to F16, as the dequantization to an F16 copy.

#include <cuda_fp16.h>
#include <cstdint>

#ifndef FVS_NACC
#define FVS_NACC 2
#endif

namespace fvs {
constexpr int D   = 256;
constexpr int NT  = 6;          // query n-tiles of 8 rows
constexpr int NR  = NT*8;       // 48 query rows
constexpr int TK  = 64;         // keys per tile
constexpr int NW  = 8;
constexpr int NTH = NW*32;
constexpr int QST = D  + 8;     // Q and K row stride in halves (8-half pad: conflict-free fragment reads)
constexpr int KST = D  + 8;
constexpr int PST = TK + 8;     // P row stride in halves
constexpr int MST = TK + 8;     // mask row stride in halves

struct smem_t {
    half  Q[NR*QST];
    half  K[TK*KST];
    half  P[2][NR*PST];
    half  M[2][8*MST];
    float redmax[2][NR];
    float redsum[2][NR];
    float alpha[2][NR];
    int   resc[2][2];        // per tile buffer and row group: does any row of the group need its O rescaled (a factor != 1)
    float m[NR];
};

// [TAG_FVS_Q8_ROW1] the raw K rows of the four QK warps (16 rows of 272 bytes each) follow smem_t
constexpr size_t SM_KR    = (sizeof(smem_t) + 15) & ~(size_t) 15;
constexpr size_t SM_BYTES = SM_KR + 4*16*272;
// [TAG_FVS_Q8_ROW1V] or, in the V variant, the raw V strips of the four PV warps (64 rows x 80 bytes each)
constexpr size_t SM_BYTES_V = SM_KR + 4*64*80;
// probe (VAR & 16384): half a V tile as F16 (32 rows), widened by the QK warps
constexpr size_t SM_BYTES_HB = SM_KR + 40*KST*sizeof(half);   // up to 40 rows (5/8 of a tile)
// [TAG_FVS_Q8_VB] the first c eighths of a V tile (8*c rows) as F16, widened by the QK warps
constexpr size_t sm_bytes_vb(const int c) { return SM_KR + (size_t) 8*c*KST*sizeof(half); }

static __device__ __forceinline__ void mma_rc(float (&c)[8], const uint32_t a0, const uint32_t a1, const uint32_t b0, const uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void mma_cc(float (&c)[8], const uint32_t a0, const uint32_t a1, const uint32_t b0, const uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ float ex2(const float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// 8 dims of a row: on F16 one 16-byte load at p; on Q8_0 p is the Q8_0 block and off (0, 8, 16, 24) the first of the 8 codes,
// returned raw as {codes 0-3, codes 4-7, d, 0}. W (Q8_0 only): the codes as three 4-byte loads of the aligned 12-byte window
// around them and two funnel shifts (they start 2 or 0 bytes past a 4-byte boundary: blocks are 34 bytes, rows 16-aligned),
// instead of four 2-byte loads; the same code on every lane, no branch
// X (Q8_0 only, [TAG_FVS_Q8_LD2]): two 8-byte loads a lane instead of four loads. The lane's 8 codes always lie inside the
// aligned 16-byte window around them. The lane that holds a block's first codes starts its window at the block's d (the
// two bytes before them; the window still holds its codes), takes d from it, and the block's other three lanes get d from
// that lane by a shuffle (src: its lane index; every lane of the warp runs the load). The codes come out of the window's
// four words by a per-lane word select and the same funnel shifts: the same raw words as W, so the same result bit for bit
template <bool Q8, bool W = false, int P = 0, bool X = false>
static __device__ __forceinline__ uint4 ld_row8(const char * p, const int off, const int src = 0) {
    if constexpr (Q8 && X) {
        (void) off; (void) src;
        return make_uint4(0, 0, 0, 0); // not used: the X path loads through ld_row8c
    } else if constexpr (Q8) {
        (void) src;
        if constexpr ((P & 1) != 0) {
            // timing probe, wrong output: one aligned 4-byte load a lane, the load count of the F16 path
            const uint32_t w0 = __ldg((const uint32_t *)((uintptr_t)(p + 2 + off) & ~(uintptr_t) 3));
            return make_uint4(w0, w0, w0 & 0xffffu, 0);
        }
        const uint32_t d = (uint32_t) __ldg((const unsigned short *) p);
        if constexpr (W) {
            const uintptr_t s = (uintptr_t)(p + 2 + off);
            const uint32_t * w = (const uint32_t *)(s & ~(uintptr_t) 3);
            const uint32_t sh = 8*(uint32_t)(s & 3);
            // aligned codes need two words: the third load re-reads the second, so no load leaves the codes' own words
            // (the last block of the last row ends the buffer on an aligned 8 codes)
            const uint32_t w0 = __ldg(w + 0), w1 = __ldg(w + 1), w2 = __ldg(w + 1 + (sh != 0));
            return make_uint4(__funnelshift_r(w0, w1, sh), __funnelshift_r(w1, w2, sh), d, 0);
        } else {
            const unsigned short * qs = (const unsigned short *)(p + 2 + off);
            const uint32_t lo = (uint32_t) __ldg(qs + 0) | ((uint32_t) __ldg(qs + 1) << 16);
            const uint32_t hi = (uint32_t) __ldg(qs + 2) | ((uint32_t) __ldg(qs + 3) << 16);
            return make_uint4(lo, hi, d, 0);
        }
    } else {
        (void) off;
        return __ldg((const uint4 *) p);
    }
}

// [TAG_FVS_Q8_LD3] a lane's Q8_0 load with everything about its place in the row worked out once per thread: the three
// aligned 4-byte words that hold its 8 codes at fixed offsets from the block (o0, o1, o2), their funnel shift (sh: 0, 16 or
// 32 bits), and the block's d. The lane with the block's first codes starts its window on the word that holds d (the two
// bytes before those codes), so its first word carries d (dsh bits up); with SD the block's other three lanes take d
// from that lane by a shuffle (src) instead of loading it, which leaves three loads a lane. Rows must be a multiple of 4
// bytes apart (they are 272), so one row's alignment is every row's. The same raw words as ld_row8 gives, bit for bit
struct q8ld_t {
    int o0, o1, o2;
    uint32_t sh, dsh;
    int src;
};
static __device__ __forceinline__ q8ld_t q8ld_make(const char * p, const int off, const int src) {
    const bool first = off == 0;
    const uintptr_t s = (uintptr_t)(p + (first ? 0 : 2 + off));
    const uint32_t  o = (uint32_t)(s & 3) + (first ? 2u : 0u);   // the codes' byte offset in the window: 0, 2 or 4
    q8ld_t c;
    c.o0  = (int)((s & ~(uintptr_t) 3) - (uintptr_t) p);
    c.o1  = c.o0 + 4;
    c.o2  = c.o0 + (o != 0 ? 8 : 4);   // aligned codes need two words: the third load re-reads the second
    c.sh  = 8*o;
    c.dsh = 8*(uint32_t)((uintptr_t) p & 3);
    c.src = src;
    return c;
}
template <bool SD>
static __device__ __forceinline__ uint4 ld_row8c(const char * p, const q8ld_t & c) {
    const uint32_t w0 = __ldg((const uint32_t *)(p + c.o0)), w1 = __ldg((const uint32_t *)(p + c.o1)), w2 = __ldg((const uint32_t *)(p + c.o2));
    uint32_t d;
    if constexpr (SD) {
        d = __shfl_sync(0xffffffffu, w0 >> c.dsh, c.src) & 0xffffu;
    } else {
        d = (uint32_t) __ldg((const unsigned short *) p);
    }
    return make_uint4(__funnelshift_rc(w0, w1, c.sh), __funnelshift_rc(w1, w2, c.sh), d, 0);
}

// what ld_row8 returned, as 8 F16 values. Q8_0: each int8 code q is biased to an unsigned byte and placed under the F16
// exponent of 1024 (0x64xx = 1024 + q + 128, exact), 1152 is taken off (exact: q itself) and the product with d is the one
// rounding
template <bool Q8, int P = 0>
static __device__ __forceinline__ uint4 to_f16x8(const uint4 r) {
    if constexpr (Q8 && (P & 2) != 0) {
        return make_uint4(r.x & 0x3bff3bffu, r.y & 0x3bff3bffu, r.x & 0x3bff3bffu, r.y & 0x3bff3bffu); // timing probe, wrong output: no widening (finite halves)
    } else if constexpr (Q8) {
        const half2  d    = __half2half2(__ushort_as_half((unsigned short) r.z));
        const half2  bias = __half2half2(__ushort_as_half((unsigned short) 0x6480)); // 1152
        const uint32_t a  = r.x ^ 0x80808080u;
        const uint32_t b  = r.y ^ 0x80808080u;
        uint32_t w[4] = { __byte_perm(a, 0x64646464u, 0x5140), __byte_perm(a, 0x64646464u, 0x7362),
                          __byte_perm(b, 0x64646464u, 0x5140), __byte_perm(b, 0x64646464u, 0x7362) };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const half2 h = __hmul2(__hsub2(*(const half2 *) &w[i], bias), d);
            w[i] = *(const uint32_t *) &h;
        }
        return make_uint4(w[0], w[1], w[2], w[3]);
    } else {
        return r;
    }
}

// [TAG_FVS_SKIP] the tiles of TK keys where any of the 8-token half's rows is unmasked, per (half, stream): blockIdx.x = half,
// blockIdx.y = stream, 256 threads. A masked key is -inf (0xFC00); this kernel is only taken without ALiBi or a softcap, so
// any other value is a key the row reads. The flags go to shared memory, then warp 0 compacts them by ballot into
// list[(stream*nhalf + half)*tstride ..] and their number into count[stream*nhalf + half]. TMAX tiles at most (n_kv/TK)
constexpr int TMAX = 8192;
__global__ void tiles_used(const char * __restrict__ mask, const int64_t nbm1, const int64_t nbm3, const int n_tok_all, const int nhalf,
        const int n_kv, const int32_t * __restrict__ kv_count, int32_t * __restrict__ list, int32_t * __restrict__ count, const int tstride) {
    __shared__ unsigned char used[TMAX];
    const int hf     = blockIdx.x;
    const int strm   = blockIdx.y;
    const int tok0   = 8*hf;
    const int n_tok  = min(8, n_tok_all - tok0);
    const int ntiles = min((kv_count ? *kv_count : n_kv) / TK, TMAX);
    mask += tok0*nbm1 + strm*nbm3;
    // one thread per (tile, 8-key chunk): 8 halves of every row; a tile is used if any chunk of any row holds a non -inf value
    for (int t = threadIdx.x; t < ntiles; t += blockDim.x) {
        used[t] = 0;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < ntiles*(TK/8); idx += blockDim.x) {
        const int t = idx / (TK/8), c = idx % (TK/8);
        bool u = false;
        for (int tok = 0; tok < n_tok && !u; ++tok) {
            const uint4 v = *(const uint4 *)(mask + tok*nbm1 + (int64_t)(t*TK + 8*c)*2);
            const uint32_t w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                u |= (w[k] & 0xFFFFu) != 0xFC00u || (w[k] >> 16) != 0xFC00u;
            }
        }
        if (u) {
            used[t] = 1;   // benign race: every writer writes 1
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        int32_t * out = list + (int64_t)(strm*nhalf + hf)*tstride;
        int n = 0;
        for (int base = 0; base < ntiles; base += 32) {
            const int t = base + lane;
            const bool u = t < ntiles && used[t];
            const unsigned int bal = __ballot_sync(0xffffffff, u);
            if (u) {
                out[n + __popc(bal & ((1u << lane) - 1u))] = t;
            }
            n += __popc(bal);
        }
        if (lane == 0) {
            count[strm*nhalf + hf] = n;
        }
    }
}

// G: GQA ratio (query heads per KV head); rows r < n_tok*G are real: token r/G, head kvh*G + r%G.
// 9 to 16 tokens run as nhalf = 2 blocks per (KV head, split), blockIdx.x = half + nhalf*(kvh + n_kv_heads*split): block
// half takes tokens 8*half .. 8*half + 7 (its own 8-row tile, exactly as an 8-token call of the same split computes them), and the
// two blocks of a pair stream the same K and V tiles side by side, so the second read can come from L2. n_tok_all counts all tokens, n_tok the block's.
// BlockIdx.y is the KV stream (llama.cpp's non-unified cache: one stream per sequence, on ne[3] of Q, K, V and the mask); a block
// reads only its own stream's Q, K, V and mask, at the same split as a call on that stream alone, and writes its tokens at stream*n_tok_all.
// VAR & 31 = KP: of the 16 key rows per lane of a K tile, the PV group loads and stores the last KP (0 = the earlier kernel, all in the QK group)
template <int G, bool STUB, int VAR = 0, bool KVQ8 = false>
__global__ void __launch_bounds__(NTH, 1) verify_stream(
        const float * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, const char * __restrict__ mask,
        float * __restrict__ dst, float * __restrict__ parts, float2 * __restrict__ meta,
        const int n_tok_all, const int n_head, const int n_kv_heads,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbk1, const int64_t nbk2, const int64_t nbv1, const int64_t nbv2,
        const int64_t nbm1, const int n_kv, const int32_t * __restrict__ kv_count, const int nsplit, const float qscale, const int nhalf,
        const int64_t nbq3, const int64_t nbk3, const int64_t nbv3, const int64_t nbm3,
        const int32_t * __restrict__ tlist, const int32_t * __restrict__ tcount, const int tstride) {
#if __CUDA_ARCH__ == 700
    static_assert(G >= NR/8 && G <= 8, "the mask tile holds 8 tokens");
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2   = 0.6931471805599453f;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    smem_t & s = *reinterpret_cast<smem_t *>(smem_raw);
    // a decode graph over the whole cache passes the used key count as data; the tiles are divided over the used keys only
    const int n_kv_used = kv_count ? *kv_count : n_kv;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int w    = tid >> 5;
    const int l4   = lane & 3;
    const int hi   = lane >> 4;
    const int qp   = (lane >> 2) & 3;
    const bool isQK = w < 4;
    constexpr int KP = VAR & 31, KQ = 16 - KP;
    constexpr bool Q8W = (VAR & 32) != 0;   // VAR & 32: the Q8_0 codes as aligned-window loads (ld_row8)
    constexpr int  Q8P = (VAR >> 6) & 3;    // VAR & 192: timing probes with wrong output (1: one load a lane, 2: no widening, 3: both)
    constexpr bool Q8X  = KVQ8 && (VAR & 768) != 0;   // VAR & 256: the Q8_0 loads at per-thread offsets, d by shuffle (ld_row8c, [TAG_FVS_Q8_LD3])
    constexpr bool Q8SD = (VAR & 256) != 0;            // VAR & 512: the same offsets, d loaded
    constexpr bool Q8R  = KVQ8 && (VAR & 1024) != 0;  // VAR & 1024: each K row read from global memory once ([TAG_FVS_Q8_ROW1])
    static_assert(!Q8R || KP == 0, "the staged K rows are loaded by the QK warps");
    constexpr int  Q8PK = (VAR & 2048) != 0 ? 0 : Q8P;   // VAR & 2048: the probes on the V loads only (K as usual)
    constexpr bool Q8RV = KVQ8 && (VAR & 8192) != 0;    // VAR & 8192: each V row's bytes read from global memory once ([TAG_FVS_Q8_ROW1V])
    // [TAG_FVS_Q8_VB] VAR & 131072: the widening of a Q8_0 V tile shared between the groups. The PV warps set the kernel's
    // pace on Q8_0 (their product loop carries the V loads and the widening; the QK warps finish their tile early), so
    // the QK warps load and widen the first HBC eighths of each V tile (rows 0 .. 8*HBC - 1, all 256 dims: a lane takes
    // the 8 dims it takes of a K row) into shared memory as F16, and the PV warps read those rows as fragments with one
    // load each and widen only the rest themselves. The same raw words through the same to_f16x8, in the same product
    // order: the same output bit for bit. The part is single-buffered. A PV warp consumes tile j - 1's part first in
    // iteration j, then arrives at barrier 3; a QK warp, once its P of tile j is written, waits at barrier 3 for the
    // four PV warps, lays down tile j's part (in its registers since the iteration before) and requests tile j + 1's
    constexpr bool Q8VX = KVQ8 && (VAR & 131072) != 0;
    static_assert(!Q8VX || (KP == 0 && (VAR & (768 | 1024 | 4096 | 8192 | 16384)) == 0), "the shared V widening goes with the plain Q8_0 loads");
    constexpr bool Q8HB = KVQ8 && (VAR & (16384 | 131072)) != 0;   // VAR & 16384: the timing probe of the above, wrong output (the part one tile early, no barrier)
    constexpr int  HBC  = (VAR & 32768) != 0 ? 5 : (VAR & 65536) != 0 ? 3 : (VAR & 262144) != 0 ? 2 : 4;   // the QK warps' share of a V tile, in eighths
    constexpr bool Q8VS = KVQ8 && (VAR & 4096) != 0;    // VAR & 4096: probe, wrong output: a V lane's load as one global word and three shared-memory reads

    // this block's 8-token half; Q, the mask and the outputs from its first token on
    const int hf    = blockIdx.x % nhalf;
    const int bxk   = blockIdx.x / nhalf;
    const int tok0  = 8*hf;
    const int n_tok = min(8, n_tok_all - tok0);
    // this block's stream; its tokens are stream*n_tok_all + tok in the output, parts and meta ([stream][token][head])
    const int strm  = blockIdx.y;
    const int64_t otok0 = (int64_t) strm*n_tok_all + tok0;
    Q    = (const float *) ((const char *) Q + tok0*nbq1 + strm*nbq3);
    K    += strm*nbk3;
    V    += strm*nbv3;
    mask += tok0*nbm1 + strm*nbm3;
    if (nsplit > 1) {
        parts += otok0*n_head*nsplit*D;
        meta  += otok0*n_head*nsplit;
    } else {
        dst   += otok0*n_head*D;
    }
    const int kvh   = bxk % n_kv_heads;
    const int split = bxk / n_kv_heads;
    // [TAG_FVS_SKIP] with a list, the splits divide this half's used tiles (tiles_used); tile k of the split is tl[k]
    const int32_t * tl = tlist ? tlist + (int64_t)(strm*nhalf + hf)*tstride : nullptr;
    const int ntiles = tl ? tcount[strm*nhalf + hf] : n_kv_used / TK;
    const int it0 = (int)(((int64_t) split     *ntiles)/nsplit);
    const int it1 = (int)(((int64_t)(split + 1)*ntiles)/nsplit);
    const int nt  = it1 - it0;
    // the first key of the split's tile k (0 <= k < nt)
    auto key0 = [&](int k) -> int64_t { return (int64_t)(tl ? tl[it0 + k] : it0 + k)*TK; };
    const int nrows = n_tok*G;

    // Q -> f16, scaled; padded rows are zero
    for (int idx = tid; idx < NR*(D/4); idx += NTH) {
        const int r = idx / (D/4), c4 = idx % (D/4);
        const int tok = r / G, hl = r % G;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < nrows) {
            v = *(const float4 *)((const char *) Q + tok*nbq1 + (kvh*G + hl)*nbq2 + 16*c4);
        }
        half2 h0 = __floats2half2_rn(v.x*qscale, v.y*qscale);
        half2 h1 = __floats2half2_rn(v.z*qscale, v.w*qscale);
        *(uint2 *) &s.Q[r*QST + 4*c4] = make_uint2(*(uint32_t *) &h0, *(uint32_t *) &h1);
    }
    // mask for the first tile (64 threads: 8 tokens x 8 chunks of 8 keys)
    if (tid < 64 && nt > 0) {
        const int tok = tid >> 3, c = tid & 7;
        uint4 mv = make_uint4(0, 0, 0, 0);
        if (tok < n_tok) mv = *(const uint4 *)(mask + tok*nbm1 + (key0(0) + 8*c)*2);
        *(uint4 *) &s.M[0][tok*MST + 8*c] = mv;
    }
    if (tid < NR) {
        s.m[tid] = -INFINITY;
        s.alpha[0][tid] = 1.0f;
        s.alpha[1][tid] = 1.0f;
        if (tid < 4) s.resc[tid >> 1][tid & 1] = 0;
    }

    const char * Kbase = K + kvh*nbk2;
    const int ka = (w >> 1) & 1;   // key group of the QK phase (32 keys)
    const int rg = w & 1;          // row group of the QK phase (24 rows)
    const int keyA = 32*ka + l4 + 4*(qp & 1) + 8*hi + 16*(qp >> 1);   // this thread's key in the QK A fragment
    // a lane's 8 dims of a row: dims 8*lane.. of K, dims vd.. of V; on Q8_0 the byte offset of their block and the first code
    const int vd   = 64*(w & 3) + 16*qp + 8*hi;
    const int kofs = KVQ8 ? 34*(lane >> 2) : 16*lane;
    const int kq   = 8*(lane & 3);
    const int vq   = vd & 31;
    const char * Vbase = V + kvh*nbv2 + (int64_t) l4*nbv1 + (KVQ8 ? 34*(vd >> 5) : 2*vd);
    // [TAG_FVS_Q8_LD3] the lane with the first codes of this lane's block: lane & 28 in a K row, lane & 11 in a V row (the
    // same row l4 and block, the block's first 8 dims)
    const q8ld_t kld = q8ld_make(Kbase + kofs, kq, lane & 28);
    const q8ld_t vld = q8ld_make(Vbase, vq, lane & 11);

    // The two groups have separate loops (their register state is disjoint); both cross the same barriers nt + 2 times.
    if (isQK) {
        uint4 kreg[KQ > 0 ? KQ : 1];
        // [TAG_FVS_Q8_ROW1] a Q8_0 K row is 272 bytes, 17 chunks of 16. Here the warp reads its 16 rows of a tile as their
        // 272 chunks, one 16-byte load a lane and 9 loads a tile (each row's bytes read from global memory once, where
        // the per-lane loads read them four times over: 64 loads a tile), keeps them in registers one tile ahead as
        // before, then lays the rows down raw in a private strip of shared memory and widens each lane's 8 dims from
        // there (the same words, so the same K tile bit for bit). Chunks past the 272nd repeat the last one
        constexpr int NKR = (16*17 + 31)/32;
        uint4 kraw[Q8R ? NKR : 1];
        uint4 kx[Q8HB ? 2*HBC : 1];
        int   kgo[Q8R ? NKR : 1], kso[Q8R ? NKR : 1];
        unsigned char * const krw = smem_raw + SM_KR + (w & 3)*(16*272);
        q8ld_t krld = {};
        if constexpr (Q8R) {
#pragma unroll
            for (int r = 0; r < NKR; ++r) {
                const int g = min(r*32 + lane, 16*17 - 1);
                kgo[r] = (g/17)*4*(int) nbk1 + 16*(g%17);
                kso[r] = (g/17)*272 + 16*(g%17);
            }
            krld = q8ld_make((const char *) krw + kofs, kq, 0);
        }
        const auto k_load = [&](const int64_t key) {   // the K tile at key into registers
            if constexpr (Q8R) {
                const char * tp = Kbase + (key + w)*nbk1;
#pragma unroll
                for (int r = 0; r < NKR; ++r) kraw[r] = __ldg((const uint4 *)(tp + kgo[r]));
            } else {
                const char * kp = Kbase + (key + w)*nbk1 + kofs;
#pragma unroll
                for (int rr = 0; rr < KQ; ++rr, kp += 4*nbk1) kreg[rr] = (Q8X ? ld_row8c<Q8SD>(kp, kld) : ld_row8<KVQ8, Q8W, Q8PK>(kp, kq));
                if constexpr (Q8HB && !Q8VX) {
                    const char * xp = V + kvh*nbv2 + (key + w)*nbv1 + kofs;
#pragma unroll
                    for (int rr = 0; rr < 2*HBC; ++rr, xp += 4*nbv1) kx[rr] = ld_row8<KVQ8, Q8W, 0>(xp, kq);
                }
            }
        };
        const auto k_store = [&]() {                    // the registers' K tile into shared memory, as F16
            if constexpr (Q8R) {
#pragma unroll
                for (int r = 0; r < NKR; ++r) *(uint4 *)(krw + kso[r]) = kraw[r];
#pragma unroll
                for (int rr = 0; rr < KQ; ++rr) {
                    const unsigned char * bp = krw + rr*272 + kofs;
                    const uint32_t w0 = *(const uint32_t *)(bp + krld.o0), w1 = *(const uint32_t *)(bp + krld.o1), w2 = *(const uint32_t *)(bp + krld.o2);
                    const uint32_t d  = *(const unsigned short *) bp;
                    *(uint4 *) &s.K[(4*rr + w)*KST + 8*lane] =
                        to_f16x8<KVQ8, Q8PK>(make_uint4(__funnelshift_rc(w0, w1, krld.sh), __funnelshift_rc(w1, w2, krld.sh), d, 0));
                }
            } else {
#pragma unroll
                for (int rr = 0; rr < KQ; ++rr) *(uint4 *) &s.K[(4*rr + w)*KST + 8*lane] = to_f16x8<KVQ8, Q8PK>(kreg[rr]);
                if constexpr (Q8HB && !Q8VX) {
                    half * vf = (half *)(smem_raw + SM_KR);
#pragma unroll
                    for (int rr = 0; rr < 2*HBC; ++rr) *(uint4 *) &vf[(4*rr + w)*KST + 8*lane] = to_f16x8<KVQ8, 0>(kx[rr]);
                }
            }
        };
        // [TAG_FVS_Q8_VB] the QK warps' part of the V tile at key: rows w, w + 4, .. of its first 8*HBC, this lane's 8 dims
        const auto x_load = [&](const int64_t key) {
            const char * xp = V + kvh*nbv2 + (key + w)*nbv1 + kofs;
#pragma unroll
            for (int rr = 0; rr < 2*HBC; ++rr, xp += 4*nbv1) kx[rr] = ld_row8<KVQ8, Q8W, 0>(xp, kq);
        };
        const auto x_store = [&]() {
            half * vf = (half *)(smem_raw + SM_KR);
#pragma unroll
            for (int rr = 0; rr < 2*HBC; ++rr) *(uint4 *) &vf[(4*rr + w)*KST + 8*lane] = to_f16x8<KVQ8, 0>(kx[rr]);
        };
        float mrun[3][4], lrun[3][4];
#pragma unroll
        for (int t = 0; t < 3; ++t)
#pragma unroll
            for (int j = 0; j < 4; ++j) { mrun[t][j] = -INFINITY; lrun[t][j] = 0.0f; }
        if (KQ > 0 && nt > 0) {
            k_load(key0(0));
            k_store();
            if (nt > 1) {
                k_load(key0(1));
            }
        }
        if constexpr (Q8VX) { if (nt > 0) x_load(key0(0)); }
        __syncthreads();

        for (int j = 0; j <= nt; ++j) {
            if (j < nt) {
                const int b = j & 1;
                // mask of the next tile: loaded now, stored after the QK phase
                uint4 mnext = make_uint4(0, 0, 0, 0);
                const bool has_next = j + 1 < nt;
                if (tid < 64 && has_next && (tid >> 3) < n_tok) {
                    mnext = *(const uint4 *)(mask + (tid >> 3)*nbm1 + (key0(j + 1) + 8*(tid & 7))*2);
                }

                // ---- S = K Q^T for 32 keys x 24 rows, two accumulator sets over the dims
                float S[FVS_NACC][3][8];
#pragma unroll
                for (int a = 0; a < FVS_NACC; ++a)
#pragma unroll
                    for (int t = 0; t < 3; ++t)
#pragma unroll
                        for (int i = 0; i < 8; ++i) S[a][t][i] = 0.0f;
                {
                    const half * Krow = s.K + keyA*KST;
                    const half * Qrow = s.Q + (24*rg + l4 + 4*hi)*QST;
#ifndef FVS_SKIP_QK
#pragma unroll
                    for (int c = 0; c < D/8; ++c) {
                        const uint4 kf = *(const uint4 *)(Krow + 8*c);
#pragma unroll
                        for (int t = 0; t < 3; ++t) {
                            const uint4 qf = *(const uint4 *)(Qrow + 8*t*QST + 8*c);
                            mma_rc(S[c % FVS_NACC][t], kf.x, kf.y, qf.x, qf.y);
                            mma_rc(S[c % FVS_NACC][t], kf.z, kf.w, qf.z, qf.w);
                        }
                    }
#endif
                }
                if constexpr (KP > 0) { if (j + 1 < nt) asm volatile("bar.arrive 2, 256;" ::: "memory"); }   // this warp is done reading the K tile
                // mask, in log2 units; keys and rows of this lane's accumulators
#pragma unroll
                for (int t = 0; t < 3; ++t)
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int qrow = 24*rg + 8*t + (i & 4) + (lane & 2) + (i & 1);
                        const int key  = 32*ka + (lane & 1) + (i & 2) + 4*(qp & 1) + 8*hi + 16*(qp >> 1);
                        {
#pragma unroll
                            for (int a = 1; a < FVS_NACC; ++a) S[0][t][i] += S[a][t][i];
                        }
                        S[0][t][i] += LOG2E*__half2float(s.M[b][(qrow / G)*MST + key]);
                    }

                if constexpr (!STUB) {
                    float mx[3][4];
#pragma unroll
                    for (int t = 0; t < 3; ++t)
#pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int i = (jj & 1) + 2*(jj & 2); // 0, 1, 4, 5
                            float v = fmaxf(S[0][t][i], S[0][t][i | 2]);
                            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  1));
                            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  4));
                            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  8));
                            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 16));
                            mx[t][jj] = v;
                        }
                    if ((lane & 0x1D) == 0) {
#pragma unroll
                        for (int t = 0; t < 3; ++t)
#pragma unroll
                            for (int jj = 0; jj < 4; ++jj) {
                                const int i = (jj & 1) + 2*(jj & 2);
                                s.redmax[ka][24*rg + 8*t + (i & 4) + (lane & 2) + (i & 1)] = mx[t][jj];
                            }
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");   // the max exchange; every QK warp is also done reading the K tile
                    // P = exp2(S - m); the row's rescale factor; per-lane row sums
                    bool moved = false;
#pragma unroll
                    for (int t = 0; t < 3; ++t)
#pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const int i = (jj & 1) + 2*(jj & 2);
                            const int qrow = 24*rg + 8*t + (i & 4) + (lane & 2) + (i & 1);
                            const int key  = 32*ka + (lane & 1) + (i & 2) + 4*(qp & 1) + 8*hi + 16*(qp >> 1);
                            const float mt = fmaxf(mx[t][jj], s.redmax[ka ^ 1][qrow]);
                            const float mo = mrun[t][jj];
                            const float mn = fmaxf(mo, mt);
                            const float al = mn == -INFINITY ? 1.0f : ex2(mo - mn);
                            mrun[t][jj] = mn;
                            moved |= al != 1.0f;
                            if (ka == 0 && (lane & 0x1D) == 0) s.alpha[b][qrow] = al;
                            const float ms = mn == -INFINITY ? 0.0f : mn;
                            const float p0 = ex2(S[0][t][i]     - ms);
                            const float p1 = ex2(S[0][t][i | 2] - ms);
                            s.P[b][qrow*PST + key]     = __float2half(p0);
                            s.P[b][qrow*PST + key + 2] = __float2half(p1);
                            lrun[t][jj] = lrun[t][jj]*al + (p0 + p1);
                        }
                    const bool any_moved = __any_sync(0xffffffff, moved);
                    if (ka == 0 && lane == 0) s.resc[b][rg] = any_moved;
                } else {
                    asm volatile("bar.sync 1, 128;" ::: "memory");
#pragma unroll
                    for (int t = 0; t < 3; ++t)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int qrow = 24*rg + 8*t + (i & 4) + (lane & 2) + (i & 1);
                            const int key  = 32*ka + (lane & 1) + (i & 2) + 4*(qp & 1) + 8*hi + 16*(qp >> 1);
                            s.P[b][qrow*PST + key] = __float2half(S[0][t][i]*(1.0f/1024.0f));
                        }
                }

                // the next K tile (loaded one iteration ago) into shared memory, then the one after it into registers
                if (has_next) {
                    k_store();
                    if (tid < 64) *(uint4 *) &s.M[b ^ 1][(tid >> 3)*MST + 8*(tid & 7)] = mnext;
                    if (KQ > 0 && j + 2 < nt) {
                        k_load(key0(j + 2));
                    }
                }
                if constexpr (Q8VX) {
                    asm volatile("bar.sync 3, 256;" ::: "memory");   // every PV warp is done reading the part of tile j - 1
                    x_store();                                        // tile j's part, read by the PV warps in iteration j + 1
                    if (j + 1 < nt) {
                        x_load(key0(j + 1));
                    }
                }
            }
            __syncthreads();
        }

        // row sums: per-lane partials -> lanes -> the two key groups
        float lsum[3][4];
#pragma unroll
        for (int t = 0; t < 3; ++t)
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                float v = lrun[t][jj];
                v += __shfl_xor_sync(0xffffffff, v,  1);
                v += __shfl_xor_sync(0xffffffff, v,  4);
                v += __shfl_xor_sync(0xffffffff, v,  8);
                v += __shfl_xor_sync(0xffffffff, v, 16);
                lsum[t][jj] = v;
            }
        if ((lane & 0x1D) == 0) {
#pragma unroll
            for (int t = 0; t < 3; ++t)
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const int i = (jj & 1) + 2*(jj & 2);
                    const int qrow = 24*rg + 8*t + (i & 4) + (lane & 2) + (i & 1);
                    s.redsum[ka][qrow] = STUB ? 0.5f : lsum[t][jj];
                    if (ka == 0) s.m[qrow] = STUB ? 0.0f : mrun[t][jj];
                }
        }
        __syncthreads();
        if (nsplit > 1 && tid < nrows) {
            const int tok = tid / G, head = kvh*G + tid % G;
            const float mm = s.m[tid];
            meta[((int64_t) tok*n_head + head)*nsplit + split] = make_float2(mm == -INFINITY ? -INFINITY : mm*LN2, s.redsum[0][tid] + s.redsum[1][tid]);
        }
    } else {
        float O[2][NT][8];
        uint4 vreg[16];
        const auto v_load = [&](const char * vp) -> uint4 {
            if constexpr (Q8VS) {
                const uint32_t w0 = __ldg((const uint32_t *)((uintptr_t)(vp + 2 + vq) & ~(uintptr_t) 3));
                const unsigned char * sp = (const unsigned char *) &s.Q[0] + ((uintptr_t) vp & 0x3f0) + 16*lane;   // words that vary with the row
                const uint32_t w1 = *(const uint32_t *)(sp + 4), w2 = *(const uint32_t *)(sp + 8), d = *(const unsigned short *) sp;
                return make_uint4(__funnelshift_rc(w0, w1, 16), __funnelshift_rc(w1, w2, 16), (d & 0x03ffu) | 0x3000u, 0);
            } else {
                return (Q8X ? ld_row8c<Q8SD>(vp, vld) : ld_row8<KVQ8, Q8W, Q8P>(vp, vq));
            }
        };
        // [TAG_FVS_Q8_ROW1V] a PV warp owns 64 dims of every V row: two Q8_0 blocks, 68 bytes, inside 5 chunks of 16 (80
        // bytes from byte 64*(w & 3) of the row). Here the warp reads a tile's 64 rows as those 320 chunks, one 16-byte
        // load a lane and 10 loads a tile (where the per-lane loads are 64 a tile), holds them in registers one tile
        // ahead as before, lays them down raw in its own strip of shared memory when the tile's turn comes, and each
        // lane widens its 8 dims from there: the same words, so the same V fragments bit for bit
        constexpr int NVR = 10;
        uint4 vraw[Q8RV ? NVR : 1];
        int   vgo[Q8RV ? NVR : 1], vso[Q8RV ? NVR : 1];
        unsigned char * const vrw = smem_raw + SM_KR + (w & 3)*(64*80);
        const int vbo = 4*(w & 3) + 34*(qp >> 1);   // the lane's block in a strip row
        q8ld_t vrld = {};
        if constexpr (Q8RV) {
#pragma unroll
            for (int r = 0; r < NVR; ++r) {
                const int g = r*32 + lane;
                vgo[r] = (g/5)*(int) nbv1 + 16*(4*(w & 3) + g%5);
                vso[r] = (g/5)*80 + 16*(g%5);
            }
            vrld = q8ld_make((const char *) vrw + vbo, vq, 0);
        }
        const auto vt_load = [&](const int64_t key) {   // the V tile at key into registers
            const char * tp = V + kvh*nbv2 + key*nbv1;
#pragma unroll
            for (int r = 0; r < NVR; ++r) vraw[r] = __ldg((const uint4 *)(tp + vgo[r]));
        };
        const auto vt_store = [&]() {
#pragma unroll
            for (int r = 0; r < NVR; ++r) *(uint4 *)(vrw + vso[r]) = vraw[r];
        };
        const auto vt_row = [&](const int row) -> uint4 {   // the lane's 8 dims of a strip row, raw
            const unsigned char * bp = vrw + row*80 + vbo;
            const uint32_t w0 = *(const uint32_t *)(bp + vrld.o0), w1 = *(const uint32_t *)(bp + vrld.o1), w2 = *(const uint32_t *)(bp + vrld.o2);
            const uint32_t d  = *(const unsigned short *) bp;
            return make_uint4(__funnelshift_rc(w0, w1, vrld.sh), __funnelshift_rc(w1, w2, vrld.sh), d, 0);
        };
        uint4 kreg[KP > 0 ? KP : 1];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
                for (int i = 0; i < 8; ++i) O[u][t][i] = 0.0f;
        if (nt > 0) {
            if constexpr (Q8RV) {
                vt_load(key0(0));
            } else {
                constexpr int q0 = Q8VX ? 2*HBC : 0;   // [TAG_FVS_Q8_VB] the rows before q0 come from the QK warps
                const char * vp = Vbase + (key0(0) + 4*q0)*nbv1;
#pragma unroll
                for (int q = q0; q < 16; ++q, vp += 4*nbv1) vreg[q] = v_load(vp);
            }
        }
        if constexpr (KP > 0) {   // the PV group owns the last KP rows of the K tile's load and its store to shared memory
            const int wk = w & 3;
            if (nt > 0) {
                {
                    const char * kp = Kbase + (key0(0) + wk + 4*KQ)*nbk1 + kofs;
#pragma unroll
                    for (int rr = 0; rr < KP; ++rr, kp += 4*nbk1) kreg[rr] = (Q8X ? ld_row8c<Q8SD>(kp, kld) : ld_row8<KVQ8, Q8W, Q8P>(kp, kq));
                }
#pragma unroll
                for (int rr = 0; rr < KP; ++rr) *(uint4 *) &s.K[(4*(KQ + rr) + wk)*KST + 8*lane] = to_f16x8<KVQ8, Q8P>(kreg[rr]);
                if (nt > 1) {
                    const char * kp = Kbase + (key0(1) + wk + 4*KQ)*nbk1 + kofs;
#pragma unroll
                    for (int rr = 0; rr < KP; ++rr, kp += 4*nbk1) kreg[rr] = (Q8X ? ld_row8c<Q8SD>(kp, kld) : ld_row8<KVQ8, Q8W, Q8P>(kp, kq));
                }
            }
        }
        __syncthreads();

        for (int j = 0; j <= nt; ++j) {
            // [TAG_FVS_Q8_VB] nothing of the part is read in iteration 0: the QK warps may lay down tile 0's
            if constexpr (Q8VX) { if (j == 0 && nt > 0) asm volatile("bar.arrive 3, 256;" ::: "memory"); }
            if (j >= 1) {
                const int b = (j - 1) & 1;
                const half * Prow = s.P[b] + (l4 + 4*hi)*PST;
                // rescale O by the tile's factors, when any is not 1
                if (s.resc[b][0] | s.resc[b][1]) {
#pragma unroll
                for (int t = 0; t < NT; ++t)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const float2 al = *(const float2 *) &s.alpha[b][8*t + 4*h + (lane & 2)];
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            O[u][t][4*h + 0] *= al.x; O[u][t][4*h + 1] *= al.y;
                            O[u][t][4*h + 2] *= al.x; O[u][t][4*h + 3] *= al.y;
                        }
                    }
                }
                // O^T += V^T P^T for 64 dims x 48 rows over the tile's 64 keys; V of the next tile is requested as this one is consumed
                const bool has_next = j < nt;
                const char * vp = Vbase + (has_next ? key0(j) : 0)*nbv1;
#ifndef FVS_SKIP_PV
                if constexpr (Q8RV) {
                    vt_store();
                    if (has_next) {
                        vt_load(key0(j));
                    }
                }
#pragma unroll
                for (int c = 0; c < TK/8; ++c) {
                    uint4 v0, v1;
                    // [TAG_FVS_Q8_VB] this warp has read its fragments of the QK warps' part (chunks 0 .. HBC - 1)
                    if constexpr (Q8VX) { if (c == HBC && j < nt) asm volatile("bar.arrive 3, 256;" ::: "memory"); }
                    if constexpr (Q8RV) {
                        v0 = to_f16x8<KVQ8, Q8P>(vt_row(l4 + 8*c));
                        v1 = to_f16x8<KVQ8, Q8P>(vt_row(l4 + 8*c + 4));
                    } else if (Q8HB && c < HBC) {
                        const half * vf = (const half *)(smem_raw + SM_KR);
                        v0 = *(const uint4 *) &vf[(l4 + 8*c)*KST + vd];
                        v1 = *(const uint4 *) &vf[(l4 + 8*c + 4)*KST + vd];
                    } else {
                        v0 = to_f16x8<KVQ8, Q8P>(vreg[2*c]);
                        v1 = to_f16x8<KVQ8, Q8P>(vreg[2*c + 1]);
                        if (has_next) {
                            vreg[2*c]     = v_load(vp);
                            vreg[2*c + 1] = v_load(vp + 4*nbv1);
                        }
                    }
                    vp += 8*nbv1;
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const uint4 pf = *(const uint4 *)(Prow + 8*t*PST + 8*c);
                        mma_cc(O[0][t], v0.x, v0.y, pf.x, pf.y);
                        mma_cc(O[1][t], v0.z, v0.w, pf.x, pf.y);
                        mma_cc(O[0][t], v1.x, v1.y, pf.z, pf.w);
                        mma_cc(O[1][t], v1.z, v1.w, pf.z, pf.w);
                    }
                }
#else
                if constexpr (Q8VX) { if (j < nt) asm volatile("bar.arrive 3, 256;" ::: "memory"); }
#endif
            }
            if constexpr (KP > 0) {
                if (j + 1 < nt) {
                    const int wk = w & 3;
                    asm volatile("bar.sync 2, 256;" ::: "memory");   // every QK warp is done reading the K tile
#pragma unroll
                    for (int rr = 0; rr < KP; ++rr) *(uint4 *) &s.K[(4*(KQ + rr) + wk)*KST + 8*lane] = to_f16x8<KVQ8, Q8P>(kreg[rr]);
                    if (j + 2 < nt) {
                        const char * kp = Kbase + (key0(j + 2) + wk + 4*KQ)*nbk1 + kofs;
#pragma unroll
                        for (int rr = 0; rr < KP; ++rr, kp += 4*nbk1) kreg[rr] = (Q8X ? ld_row8c<Q8SD>(kp, kld) : ld_row8<KVQ8, Q8W, Q8P>(kp, kq));
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();   // pairs with the QK group's row-sum barrier

#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int qrow = 8*t + (i & 4) + (lane & 2) + (i & 1);
                    const int dim  = 64*(w & 3) + 16*qp + 8*hi + 4*u + (lane & 1) + (i & 2);
                    if (qrow < nrows) {
                        const int tok = qrow / G, head = kvh*G + qrow % G;
                        if (nsplit == 1) {
                            dst[((int64_t) tok*n_head + head)*D + dim] = O[u][t][i] / (s.redsum[0][qrow] + s.redsum[1][qrow]);
                        } else {
                            parts[(((int64_t) tok*n_head + head)*nsplit + split)*D + dim] = O[u][t][i];
                        }
                    }
                }
    }
#else
    (void) Q; (void) K; (void) V; (void) mask; (void) dst; (void) parts; (void) meta; (void) n_tok_all; (void) n_head; (void) n_kv_heads;
    (void) nbq1; (void) nbq2; (void) nbk1; (void) nbk2; (void) nbv1; (void) nbv2; (void) nbm1; (void) n_kv; (void) kv_count; (void) nsplit; (void) qscale; (void) nhalf;
    (void) nbq3; (void) nbk3; (void) nbv3; (void) nbm3; (void) tlist; (void) tcount; (void) tstride;
#endif // __CUDA_ARCH__ == 700
}
} // namespace fvs
