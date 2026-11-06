// [TAG_QPN_GEMM] prompt products on repacked weights without the fp16 expansion (LLAMA_QPN_GEMM=1).
//
// Above GGML_CUDA_QPN_MAX_TOKENS tokens a repacked weight goes to cuBLAS through an expansion node: a CPY that writes the
// whole weight as fp16 once per ubatch (qpn_to_fp16_fast_kernel), 9% of a shallow prefill's kernel time at ub 2048 (the
// GEMM itself, cuBLAS's cutlass_70_tensorop_s884gemm_f16_128x128, is 62% at 75% of the card's peak). Here the same
// CUTLASS Volta GEMM (128x128x32 threadblock, 64x64 warps, 2 stages, fp32 accumulators) reads the repacked chunks
// directly: its B-operand iterator loads one 32-bit code word per 8-element access (the thread map's access is 8
// consecutive K columns of one row, which is exactly one chunk word) and decodes it to fp16 with the arithmetic of
// qpn_to_fp16_fast_kernel (the same float operations in the same order, so the values the GEMM multiplies are the
// expansion's), between the global load and the shared-memory store, where CUTLASS's pipeline applies its operand
// transform. The activations arrive as fp16 (the cuBLAS path's conversion) and the output is fp32, as before.
//
// Q4_K, Q5_K, Q6_K and IQ4_XS (the four types with a fast expansion: 96% of the 27B's bytes). A weight of another type,
// or a product the kernel does not take, runs the expansion on demand into the expansion node and cuBLAS after it.
// LLAMA_QPN_GEMM_CHECK=1 runs both for every product and logs the largest difference per shape once.

#include "common.cuh"
#include "convert.cuh"
#include "mmvq-qpn-impl.cuh"
#include "qpn-gemm.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"

#include <atomic>
#include <mutex>
#include <set>
#include <tuple>

namespace {

using Element = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;     // activations [tokens][K]
using LayoutB = cutlass::layout::ColumnMajor;  // the weight [N][K] read as K x N
using LayoutC = cutlass::layout::RowMajor;     // the product [tokens][N]
using TBShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;  // 16 warps: 32 accumulators a thread, one B access a thread, so other warps' MMAs overlap a warp's decode
using InstShape = cutlass::gemm::GemmShape<8, 8, 4>;
using OutputOp = cutlass::epilogue::thread::LinearCombination<float, 4, float, float>;

using Default = typename cutlass::gemm::kernel::DefaultGemm<
    Element, LayoutA, 8, Element, LayoutB, 8, float, LayoutC, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    TBShape, WarpShape, InstShape, OutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2, false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;
using DefaultMma = typename Default::Mma;
using Epilogue = typename Default::Epilogue;
using OutputIterator = typename Epilogue::OutputTileIterator;
using IteratorA = typename DefaultMma::IteratorA;
using SmemIteratorA = typename DefaultMma::SmemIteratorA;
using SmemIteratorB = typename DefaultMma::SmemIteratorB;
using ThreadMapB = typename DefaultMma::IteratorB::ThreadMap;

static_assert(ThreadMapB::kElementsPerAccess == 8, "one chunk word per access");
static_assert(ThreadMapB::Iterations::kContiguous == 1, "a thread's accesses share their 8 K columns");
static_assert(TBShape::kK % 8 == 0 && 32 % TBShape::kK == 0 || TBShape::kK % 32 == 0, "whole chunk words");

// The raw words one access needs: the code word, its high-bit word (Q5_K, Q6_K), the access's scale and min as floats
// (from the row's header, kept in registers per superblock, so computing them costs no global load) and the word
// position q. The iterator only loads the code words (the pipeline issues these loads a k-step ahead); the transform
// decodes them when the fragment is stored to shared memory, after the current tile's MMAs, as CUTLASS's own operand
// transform runs: per value a shift, a mask, one multiply-subtract and the rounding to fp16
constexpr int kRawWords = 5;

// the row's per-superblock header: Q4_K/Q5_K d, dmin, scales; Q6_K the 16 scales (d in extra); IQ4_XS d, scales_h, scales_l
template <ggml_type type>
struct QpnHeader {
  uint4 w;
  uint32_t extra;
};

template <ggml_type type>
static __device__ __forceinline__ QpnHeader<type> qpn_gemm_header(const char* __restrict__ tsb, const int r) {
  QpnHeader<type> h;
  if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
    h.w = *(const uint4*) (tsb + r * 16);
    h.extra = 0u;
  } else if constexpr (type == GGML_TYPE_Q6_K) {
    h.w = *(const uint4*) (tsb + 12 * QPN_CHUNK + r * 16);
    h.extra = (uint32_t) (uint16_t) qpn_load_tail<type>(tsb, r).x;  // d
  } else {
    h.w = qpn_load_tail<type>(tsb, r);
    h.extra = 0u;
  }
  return h;
}

// byte i of the four words, selected in registers (an indexed local array would live in local memory)
static __device__ __forceinline__ uint32_t qpn_gemm_byte(const uint4 w, const int i) {
  const uint32_t v = i < 4 ? w.x : i < 8 ? w.y : i < 12 ? w.z : w.w;
  return (v >> (8 * (i & 3))) & 0xFF;
}

// the access's code word(s) and, from the row's header, its scale and min: the same float operations as
// qpn_to_fp16_fast_kernel (Q4_K/Q5_K: d = dall*sc, mm = dmin*m with get_scale_min_k4's bytes; Q6_K: d*sc; IQ4_XS: d*(ls - 32))
template <ggml_type type>
static __device__ __forceinline__ void qpn_gemm_load_raw(const char* __restrict__ tsb, const int r, const int j, const int q,
                                                         const QpnHeader<type>& h, uint32_t* raw) {
  float scale = 0.0f, mn = 0.0f;
  if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
    constexpr int ch0 = type == GGML_TYPE_Q4_K ? 1 : 3;
    raw[0] = *(const uint32_t*) (tsb + (ch0 + j) * QPN_CHUNK + r * 16 + 4 * q);
    raw[1] = type == GGML_TYPE_Q5_K ? *(const uint32_t*) (tsb + (1 + j / 4) * QPN_CHUNK + r * 16 + 4 * (j % 4)) : 0u;
    half2 dm;
    memcpy(&dm, &h.w.x, 4);
    const float dall = __low2half(dm);
    const float dmin = __high2half(dm);
    // get_scale_min_k4 on the 12 scale bytes (bytes 4..15 of the header)
    const uint4 sw = make_uint4(h.w.y, h.w.z, h.w.w, 0u);
    uint32_t sc, m;
    if (j < 4) {
      sc = qpn_gemm_byte(sw, j) & 63;
      m = qpn_gemm_byte(sw, j + 4) & 63;
    } else {
      sc = (qpn_gemm_byte(sw, j + 4) & 0xF) | ((qpn_gemm_byte(sw, j - 4) >> 6) << 4);
      m = (qpn_gemm_byte(sw, j + 4) >> 4) | ((qpn_gemm_byte(sw, j) >> 6) << 4);
    }
    scale = dall * (uint8_t) sc;
    mn = dmin * (uint8_t) m;
  } else if constexpr (type == GGML_TYPE_Q6_K) {
    raw[0] = *(const uint32_t*) (tsb + j * QPN_CHUNK + r * 16 + 4 * q);
    raw[1] = *(const uint32_t*) (tsb + (8 + j / 2) * QPN_CHUNK + r * 16 + 4 * ((2 * j) % 4) + (q < 2 ? 0 : 4));
    const float d = __ushort_as_half((uint16_t) h.extra);
    const int ip = j / 4, q4 = j % 4;
    scale = d * (int8_t) qpn_gemm_byte(h.w, 8 * ip + q / 2 + 2 * q4);  // l/16 == q/2 for l = 8q + s
  } else if constexpr (type == GGML_TYPE_IQ4_XS) {
    raw[0] = *(const uint32_t*) (tsb + j * QPN_CHUNK + r * 16 + 4 * q);
    raw[1] = 0u;
    // the tail: d (bytes 0-1), scales_h (2-3), scales_l (4-7)
    const float d = __ushort_as_half((uint16_t) (h.w.x & 0xFFFF));
    const uint32_t scales_h = h.w.x >> 16;
    const uint32_t ls = ((qpn_gemm_byte(h.w, 4 + j / 2) >> (4 * (j % 2))) & 0xf) | (((scales_h >> (2 * j)) & 3) << 4);
    scale = d * ((int) ls - 32);
  }
  raw[2] = __float_as_uint(scale);
  raw[3] = __float_as_uint(mn);
  raw[4] = (uint32_t) q;
}

// the 8 values of the access, as qpn_to_fp16_fast_kernel computes them (the same operations, the same rounding)
template <ggml_type type>
static __device__ __forceinline__ void qpn_gemm_decode8(const uint32_t* raw, cutlass::half_t* out) {
  const uint32_t cw = raw[0];
  const uint32_t hw = raw[1];
  const float scale = __uint_as_float(raw[2]);
  const float mn = __uint_as_float(raw[3]);
  const int q = (int) raw[4];
  float v[8];
  if constexpr (type == GGML_TYPE_Q5_K) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int code = ((cw >> (4 * qpn_nib(s))) & 0xF) + ((hw >> qpn_hb5(q, s)) & 1 ? 16 : 0);
      v[s] = scale * code - mn;
    }
  } else if constexpr (type == GGML_TYPE_Q4_K) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      v[s] = scale * ((cw >> (4 * qpn_nib(s))) & 0xF) - mn;
    }
  } else if constexpr (type == GGML_TYPE_Q6_K) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int c = ((cw >> (4 * qpn_nib(s))) & 0xF) | (((hw >> qpn_hb6(q, s)) & 3) << 4);
      v[s] = scale * ((int8_t) c - 32);
    }
  } else if constexpr (type == GGML_TYPE_IQ4_XS) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      v[s] = scale * kvalues_iq4nl[(cw >> (4 * s)) & 0xF];
    }
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const half h = ggml_cuda_cast<half>(v[s]);
    out[s] = *reinterpret_cast<const cutlass::half_t*>(&h);
  }
}

// The B operand iterator over a repacked weight of N rows and K = 256*nsb columns: the interface MmaPipelined uses
// (load, ++, clear_mask), the thread map of the GEMM's own B iterator, and its fragment order (strided iterations, then
// the access), so the transform's output is stored as the predicated iterator's fragment would be. The fragment holds
// the raw words; the row header (16 bytes, the same for 8 k-steps and the 4 lanes of a row) is read per access
template <ggml_type type>
struct QpnIteratorB {
  using Element = uint32_t;
  using ThreadMap = ThreadMapB;
  using Fragment = cutlass::Array<uint32_t, ThreadMap::Iterations::kStrided * kRawWords>;
  static constexpr int kTSB = qpn_tsb<type>();

  struct Params {
    const char* w;  // the repacked weight
    int nsb;        // superblocks per row (K / 256)
    int n;          // rows
  };

  const char* w_;
  int nsb_;
  int n_;
  int k_;                                        // first K column of this thread's accesses
  int row_[ThreadMap::Iterations::kStrided];     // the rows of this thread's accesses
  bool masked_;

  CUTLASS_DEVICE QpnIteratorB(const Params& p, int thread_idx, int k0, int n0)
      : w_(p.w), nsb_(p.nsb), n_(p.n), masked_(false) {
    const auto offset = ThreadMap::initial_offset(thread_idx);
    k_ = k0 + offset.contiguous();
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      row_[s] = n0 + offset.strided() + s * ThreadMap::Delta::kStrided;
    }
  }

  CUTLASS_DEVICE void load(Fragment& frag) {
    uint32_t* out = frag.data();
    const int sb = k_ / QK_K;
    const int j = (k_ % QK_K) / 32;
    const int q = (k_ % 32) / 8;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      const int row = row_[s];
      uint32_t* raw = out + s * kRawWords;
      if (masked_ || row >= n_) {
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kRawWords; ++e) {
          raw[e] = 0u;
        }
        continue;
      }
      const char* tsb = w_ + ((int64_t) (row / 32) * nsb_ + sb) * kTSB;
      qpn_gemm_load_raw<type>(tsb, row % 32, j, q, qpn_gemm_header<type>(tsb, row % 32), raw);
    }
  }

  CUTLASS_DEVICE QpnIteratorB& operator++() {
    k_ += TBShape::kK;
    return *this;
  }

  CUTLASS_DEVICE void clear_mask(bool enable = true) {
    masked_ = masked_ || enable;
  }
};

// the raw words of a fragment decoded to the fp16 fragment the shared-memory iterator stores (a masked access's words
// are zero: scale zero, code zero, so its values are zero as the padded rows of the predicated iterator would be)
template <ggml_type type>
struct QpnTransformB {
  using InputFragment = typename QpnIteratorB<type>::Fragment;
  using OutputFragment = cutlass::Array<cutlass::half_t, ThreadMapB::Iterations::kStrided * 8>;
  CUTLASS_DEVICE OutputFragment operator()(const InputFragment& in) const {
    OutputFragment out;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMapB::Iterations::kStrided; ++s) {
      qpn_gemm_decode8<type>(in.data() + s * kRawWords, out.data() + s * 8);
    }
    return out;
  }
};

using TransformA = cutlass::NumericArrayConverter<cutlass::half_t, cutlass::half_t, IteratorA::Fragment::kElements>;

template <ggml_type type>
using Mma = cutlass::gemm::threadblock::MmaPipelined<typename DefaultMma::Shape, IteratorA, SmemIteratorA, QpnIteratorB<type>,
                                                     SmemIteratorB, float, LayoutC, typename DefaultMma::Policy, TransformA,
                                                     QpnTransformB<type>>;

template <ggml_type type>
union SharedStorage {
  typename Mma<type>::SharedStorage main_loop;
  typename Epilogue::SharedStorage epilogue;
};

struct GemmParams {
  int m;  // tokens
  int n;  // the weight's rows
  int k;  // columns
  const cutlass::half_t* a;  // [m][k]
  float* d;                  // [m][n]
  typename IteratorA::Params params_A;
  typename OutputIterator::Params params_D;
  const char* w;
  int nsb;
};

template <ggml_type type>
static __global__ void __launch_bounds__(Default::kThreadCount, 1) qpn_gemm_kernel(const GemmParams p) {
  extern __shared__ int4 qpn_gemm_smem[];
  SharedStorage<type>& shared = *reinterpret_cast<SharedStorage<type>*>(qpn_gemm_smem);

  const int n0 = blockIdx.x * TBShape::kN;
  const int m0 = blockIdx.y * TBShape::kM;
  const int thread_idx = threadIdx.x;
  const int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  const int lane_idx = threadIdx.x % 32;

  IteratorA iterator_A(p.params_A, const_cast<cutlass::half_t*>(p.a), {p.m, p.k}, thread_idx, {m0, 0});
  typename QpnIteratorB<type>::Params pb = {p.w, p.nsb, p.n};
  QpnIteratorB<type> iterator_B(pb, thread_idx, 0, n0);

  Mma<type> mma(shared.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma<type>::FragmentC accumulators;
  accumulators.clear();
  mma(p.k / TBShape::kK, accumulators, iterator_A, iterator_B, accumulators);

  OutputOp output_op(typename OutputOp::Params(1.0f, 0.0f));
  OutputIterator iterator_D(p.params_D, p.d, {p.m, p.n}, thread_idx, {m0, n0});
  OutputIterator iterator_C(p.params_D, p.d, {p.m, p.n}, thread_idx, {m0, n0});
  Epilogue epilogue(shared.epilogue, thread_idx, warp_idx, lane_idx);
  epilogue(output_op, iterator_D, accumulators, iterator_C);
}

template <ggml_type type>
static void qpn_gemm_launch(const GemmParams& p, cudaStream_t stream) {
  constexpr size_t smem = sizeof(SharedStorage<type>);
  static std::once_flag once[GGML_CUDA_MAX_DEVICES];
  const int id = ggml_cuda_get_device();
  std::call_once(once[id], [] {
    CUDA_CHECK(cudaFuncSetAttribute(qpn_gemm_kernel<type>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, qpn_gemm_kernel<type>, Default::kThreadCount, smem);
    cudaFuncAttributes attr = {};
    cudaFuncGetAttributes(&attr, qpn_gemm_kernel<type>);
    GGML_LOG_WARN("qpn gemm %s: shared %zu B (mainloop %zu, epilogue %zu), %d regs, %zu B local, %d threads, %d blocks/SM
",
                  ggml_type_name(type), smem, sizeof(typename Mma<type>::SharedStorage), sizeof(typename Epilogue::SharedStorage),
                  attr.numRegs, attr.localSizeBytes, Default::kThreadCount, occ);
  });
  const dim3 grid((p.n + TBShape::kN - 1) / TBShape::kN, (p.m + TBShape::kM - 1) / TBShape::kM, 1);
  qpn_gemm_kernel<type><<<grid, Default::kThreadCount, smem, stream>>>(p);
  CUDA_CHECK(cudaGetLastError());
}

// max |a - b| and max |b| over n values, as non-negative float bits (atomicMax on the int image orders them)
static __global__ void qpn_gemm_compare_kernel(const float* __restrict__ a, const float* __restrict__ b, const int64_t n,
                                               int* __restrict__ out) {
  float md = 0.0f, mb = 0.0f;
  for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
    md = fmaxf(md, fabsf(a[i] - b[i]));
    mb = fmaxf(mb, fabsf(b[i]));
  }
  atomicMax(out + 0, __float_as_int(md));
  atomicMax(out + 1, __float_as_int(mb));
}

}  // namespace

bool ggml_cuda_qpn_gemm_on() {
  static const bool on = [] {
    const char* e = getenv("LLAMA_QPN_GEMM");
    return e != nullptr && atoi(e) != 0;
  }();
  return on;
}

bool ggml_cuda_qpn_gemm_check_on() {
  static const bool on = [] {
    const char* e = getenv("LLAMA_QPN_GEMM_CHECK");
    return e != nullptr && atoi(e) != 0;
  }();
  return on;
}

bool ggml_cuda_qpn_gemm_type(ggml_type type) {
  return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS;
}

bool ggml_cuda_qpn_gemm_takes(const ggml_tensor* w, const ggml_tensor* src1, const ggml_tensor* dst) {
  if (!ggml_cuda_qpn_gemm_on() || !ggml_cuda_qpn_is_repacked(w) || w->view_src != nullptr || !ggml_cuda_qpn_gemm_type(w->type)) {
    return false;
  }
  if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_cuda_qpn_flat_cols(src1) || !ggml_cuda_qpn_flat_cols(dst)) {
    return false;
  }
  if (src1->ne[0] != w->ne[0] || dst->ne[0] != w->ne[1] || ggml_nrows(dst) != ggml_nrows(src1) || src1->nb[0] != sizeof(float) ||
      dst->nb[0] != sizeof(float) || w->ne[0] % QK_K != 0 || w->ne[1] % 32 != 0) {
    return false;
  }
  if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc != GGML_CUDA_CC_VOLTA) {
    return false;
  }
  return true;
}

void ggml_cuda_mul_mat_qpn_gemm(ggml_backend_cuda_context& ctx, const ggml_tensor* w, const ggml_tensor* src1, ggml_tensor* dst) {
  GGML_ASSERT(ggml_cuda_qpn_gemm_takes(w, src1, dst));
  cudaStream_t stream = ctx.stream();
  const int64_t k = w->ne[0];
  const int64_t n = w->ne[1];
  const int64_t m = ggml_nrows(src1);

  // the activations as fp16, as the cuBLAS path converts them
  ggml_cuda_pool_alloc<half> a16(ctx.pool(), (size_t) m * k);
  const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
  to_fp16(src1->data, a16.get(), m * k, stream);

  GemmParams p;
  p.m = (int) m;
  p.n = (int) n;
  p.k = (int) k;
  p.a = reinterpret_cast<const cutlass::half_t*>(a16.get());
  p.d = (float*) dst->data;
  p.params_A = typename IteratorA::Params(LayoutA((int) k));
  p.params_D = typename OutputIterator::Params(LayoutC((int) n));
  p.w = (const char*) w->data;
  p.nsb = (int) (k / QK_K);
  switch (w->type) {
    case GGML_TYPE_Q4_K:   qpn_gemm_launch<GGML_TYPE_Q4_K>(p, stream); break;
    case GGML_TYPE_Q5_K:   qpn_gemm_launch<GGML_TYPE_Q5_K>(p, stream); break;
    case GGML_TYPE_Q6_K:   qpn_gemm_launch<GGML_TYPE_Q6_K>(p, stream); break;
    case GGML_TYPE_IQ4_XS: qpn_gemm_launch<GGML_TYPE_IQ4_XS>(p, stream); break;
    default: GGML_ABORT("qpn gemm: type");
  }
}

void ggml_cuda_qpn_gemm_compare(const ggml_tensor* w, const ggml_tensor* src1, const float* fused, const float* ref, cudaStream_t stream) {
  const int64_t n = (int64_t) w->ne[1] * ggml_nrows(src1);
  int* d = nullptr;
  CUDA_CHECK(cudaMallocAsync((void**) &d, 2 * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d, 0, 2 * sizeof(int), stream));
  qpn_gemm_compare_kernel<<<1024, 256, 0, stream>>>(fused, ref, n, d);
  CUDA_CHECK(cudaGetLastError());
  int h[2];
  CUDA_CHECK(cudaMemcpyAsync(h, d, sizeof(h), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFreeAsync(d, stream));
  float md, mb;
  memcpy(&md, &h[0], 4);
  memcpy(&mb, &h[1], 4);
  static std::mutex mu;
  static std::set<std::tuple<int, int64_t, int64_t, int64_t>> seen;
  static int n_logged = 0;
  const auto key = std::make_tuple((int) w->type, w->ne[0], w->ne[1], ggml_nrows(src1));
  std::lock_guard<std::mutex> lock(mu);
  if (seen.insert(key).second && n_logged++ < 64) {
    GGML_LOG_WARN("qpn gemm check: %s %" PRId64 " x %" PRId64 " at %" PRId64 " tokens: max |fused - cublas| %.3e, max |cublas| %.3e (%.2e relative)\n",
                  ggml_type_name(w->type), w->ne[1], w->ne[0], ggml_nrows(src1), md, mb, mb > 0 ? md / mb : 0.0);
  }
}
