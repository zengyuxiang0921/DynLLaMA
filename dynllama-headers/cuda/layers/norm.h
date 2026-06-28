//
// CUDA normalization layer: LayerNorm and RMSNorm.
// LayerNorm computes mean/variance per row then applies weight + bias.
// RMSNorm delegates to the rms_norm op.
// GPU path (__CUDACC__): launches CUDA kernels.
// CPU fallback (#else): plain loops with std::vector temps when needed.
//

#pragma once

#include "types.h"
#include <vector>
#include <cmath>

#include "../ops/rms_norm.h"
#include "../ops/add.h"

// ---------------------------------------------------------------------------
// LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
// ---------------------------------------------------------------------------

#ifdef __CUDACC__

extern "C" __global__
void layer_norm_kernel(float * out, const float * x, const float * weight, const float * bias,
                        int n, int n_embd, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    const float * x_row = x + row * n_embd;
    float * out_row = out + row * n_embd;

    // mean
    float mean = 0.0f;
    for (int j = 0; j < n_embd; j++) {
        mean += x_row[j];
    }
    mean /= (float)n_embd;

    // variance
    float var = 0.0f;
    for (int j = 0; j < n_embd; j++) {
        float diff = x_row[j] - mean;
        var += diff * diff;
    }
    var /= (float)n_embd;

    // normalize, scale, shift
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int j = 0; j < n_embd; j++) {
        float val = (x_row[j] - mean) * inv_std * weight[j];
        if (bias) {
            val += bias[j];
        }
        out_row[j] = val;
    }
}

#else // CPU fallback

static inline void layer_norm_cpu(float * out, const float * x, const float * weight, const float * bias,
                                   int n, int n_embd, float eps) {
    for (int i = 0; i < n; i++) {
        const float * x_row = x + i * n_embd;
        float * out_row = out + i * n_embd;

        float mean = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            mean += x_row[j];
        }
        mean /= (float)n_embd;

        float var = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            float diff = x_row[j] - mean;
            var += diff * diff;
        }
        var /= (float)n_embd;

        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < n_embd; j++) {
            float val = (x_row[j] - mean) * inv_std * weight[j];
            if (bias) {
                val += bias[j];
            }
            out_row[j] = val;
        }
    }
}

#endif

static inline void llm_norm_forward(
        float       * out,
        const float * x,
        const float * weight,
        const float * bias,
        int           n,
        int           n_embd,
        float         eps) {

#ifdef __CUDACC__
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    layer_norm_kernel<<<grid, block>>>(out, x, weight, bias, n, n_embd, eps);
#else
    layer_norm_cpu(out, x, weight, bias, n, n_embd, eps);
#endif
}

// ---------------------------------------------------------------------------
// RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
// ---------------------------------------------------------------------------

static inline void llm_rms_norm_forward(
        float       * out,
        const float * x,
        const float * weight,
        int           n,
        int           n_embd,
        float         eps) {

#ifdef __CUDACC__
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    rms_norm<<<grid, block>>>(out, x, weight, n, n_embd, eps);
#else
    rms_norm(out, x, weight, n, n_embd, eps);
#endif
}

// ggml tensor wrapper
static inline void llm_norm_forward(
        struct ggml_tensor * out,
        const struct ggml_tensor * x,
        const struct ggml_tensor * weight,
        const struct ggml_tensor * bias,
        float                   eps) {
    llm_norm_forward(
        (float *)out->data,
        (const float *)x->data,
        (const float *)weight->data,
        bias ? (const float *)bias->data : NULL,
        (int)x->ne[1], (int)x->ne[0], eps);
}

// ggml tensor wrapper
static inline void llm_rms_norm_forward(
        struct ggml_tensor * out,
        const struct ggml_tensor * x,
        const struct ggml_tensor * weight,
        float                   eps) {
    llm_rms_norm_forward(
        (float *)out->data,
        (const float *)x->data,
        (const float *)weight->data,
        (int)x->ne[1], (int)x->ne[0], eps);
}
