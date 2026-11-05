#pragma once

// [TAG_QPN_GEMM] prompt products on repacked weights without the fp16 expansion (qpn-gemm.cu)

#include "common.cuh"

bool ggml_cuda_qpn_gemm_on();        // LLAMA_QPN_GEMM=1
bool ggml_cuda_qpn_gemm_check_on();  // LLAMA_QPN_GEMM_CHECK=1: the expansion and cuBLAS too, compared
bool ggml_cuda_qpn_gemm_type(ggml_type type);
// the fused kernel takes this product: the weight repacked, of a decoded type, the input and output fp32 with their rows one run
bool ggml_cuda_qpn_gemm_takes(const ggml_tensor * w, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_mul_mat_qpn_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cuda_qpn_gemm_compare(const ggml_tensor * w, const ggml_tensor * src1, const float * fused, const float * ref, cudaStream_t stream);
