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
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
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

// the 8 values of columns 8q..8q+7 of sub-block j (columns 32j..32j+31 of the superblock) of row r, decoded as
// qpn_to_fp16_fast_kernel decodes them (the same float operations, the same rounding to fp16)
template <ggml_type type>
static __device__ __forceinline__ void qpn_gemm_decode8(const char* __restrict__ tsb, const int r, const int j, const int q,
                                                        cutlass::half_t* out) {
  float v[8];
  if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
    constexpr int ch0 = type == GGML_TYPE_Q4_K ? 1 : 3;
    const uint4 hd = *(const uint4*) (tsb + r * 16);  // d, dmin, scales
    const uint32_t cw = *(const uint32_t*) (tsb + (ch0 + j) * QPN_CHUNK + r * 16 + 4 * q);
    half2 dm;
    memcpy(&dm, &hd.x, 4);
    const uint32_t sw[3] = {hd.y, hd.z, hd.w};
    const float dall = __low2half(dm);
    const float dmin = __high2half(dm);
    uint8_t sc, m;
    get_scale_min_k4(j, (const uint8_t*) sw, sc, m);
    const float d = dall * sc;
    const float mm = dmin * m;
    if constexpr (type == GGML_TYPE_Q5_K) {
      const uint32_t hw = *(const uint32_t*) (tsb + (1 + j / 4) * QPN_CHUNK + r * 16 + 4 * (j % 4));  // w[4 + j]
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int code = ((cw >> (4 * qpn_nib(s))) & 0xF) + ((hw >> qpn_hb5(q, s)) & 1 ? 16 : 0);
        v[s] = d * code - mm;
      }
    } else {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        v[s] = d * ((cw >> (4 * qpn_nib(s))) & 0xF) - mm;
      }
    }
  } else if constexpr (type == GGML_TYPE_Q6_K) {
    const uint32_t cw = *(const uint32_t*) (tsb + j * QPN_CHUNK + r * 16 + 4 * q);
    // w[32 + 2j] holds the high bits of words 0,1 of sub-block j and w[33 + 2j] those of words 2,3
    const uint32_t h = *(const uint32_t*) (tsb + (8 + j / 2) * QPN_CHUNK + r * 16 + 4 * ((2 * j) % 4) + (q < 2 ? 0 : 4));
    const uint4 sv = *(const uint4*) (tsb + 12 * QPN_CHUNK + r * 16);  // the 16 scales
    const int8_t* scales = (const int8_t*) &sv;
    const uint16_t dh = (uint16_t) qpn_load_tail<type>(tsb, r).x;
    const float d = __ushort_as_half(dh);
    const int ip = j / 4, q4 = j % 4;
    const int8_t sc = scales[8 * ip + q / 2 + 2 * q4];  // l/16 == q/2 for l = 8q + s
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int c = ((cw >> (4 * qpn_nib(s))) & 0xF) | (((h >> qpn_hb6(q, s)) & 3) << 4);
      v[s] = d * sc * ((int8_t) c - 32);
    }
  } else if constexpr (type == GGML_TYPE_IQ4_XS) {
    const uint32_t cw = *(const uint32_t*) (tsb + j * QPN_CHUNK + r * 16 + 4 * q);
    const uint4 tl = qpn_load_tail<type>(tsb, r);
    block_iq4_xs b;
    memcpy(&b, &tl, 8);  // d, scales_h, scales_l
    const float d = (float) b.d * ((((b.scales_l[j / 2] >> 4 * (j % 2)) & 0xf) | (((b.scales_h >> 2 * j) & 3) << 4)) - 32);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      v[s] = d * kvalues_iq4nl[(cw >> (4 * s)) & 0xF];
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
// the 8 elements of the access), so SmemIteratorB stores the fragment as it stores the predicated iterator's
template <ggml_type type>
struct QpnIteratorB {
  using Element = cutlass::half_t;
  using ThreadMap = ThreadMapB;
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
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
    Element* out = frag.data();
    const int sb = k_ / QK_K;
    const int j = (k_ % QK_K) / 32;
    const int q = (k_ % 32) / 8;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      const int row = row_[s];
      if (masked_ || row >= n_) {
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < 8; ++e) {
          out[s * 8 + e] = Element(0);
        }
        continue;
      }
      const char* tsb = w_ + ((int64_t) (row / 32) * nsb_ + sb) * kTSB;
      qpn_gemm_decode8<type>(tsb, row % 32, j, q, out + s * 8);
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

template <ggml_type type>
using Mma = cutlass::gemm::threadblock::MmaPipelined<typename DefaultMma::Shape, IteratorA, SmemIteratorA, QpnIteratorB<type>,
                                                     SmemIteratorB, float, LayoutC, typename DefaultMma::Policy>;

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
