//
// CPU normalization layer: LayerNorm and RMSNorm.
// Built as a dynllama_layer holding the norm weight (and optional bias).
//

#pragma once

#include "dynllama_layer.h"
#include <cmath>

#include "../ops/rms_norm.h"

// RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)
static inline void dynllama_rms_norm_forward(
        const dynllama::dynllama_layer & L,
        float       * out,
        const float * x,
        int           n,
        int           n_embd,
        float         eps) {
    rms_norm(out, x, L.weight_ptr("weight"), n, n_embd, eps);
}

// LayerNorm: out = (x - mean) / sqrt(var + eps) * weight (+ bias)
static inline void dynllama_norm_forward(
        const dynllama::dynllama_layer & L,
        float       * out,
        const float * x,
        int           n,
        int           n_embd,
        float         eps) {
    const float * weight = L.weight_ptr("weight");
    const float * bias   = L.weight_ptr("bias");
    for (int i = 0; i < n; i++) {
        const float * x_row   = x + i * n_embd;
        float       * out_row = out + i * n_embd;

        float mean = 0.0f;
        for (int j = 0; j < n_embd; j++) mean += x_row[j];
        mean /= (float) n_embd;

        float var = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            float diff = x_row[j] - mean;
            var += diff * diff;
        }
        var /= (float) n_embd;

        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < n_embd; j++) {
            float val = (x_row[j] - mean) * inv_std * weight[j];
            if (bias) val += bias[j];
            out_row[j] = val;
        }
    }
}

// Build a norm layer holding `weight` [n_embd] and optional `bias` [n_embd].
// rms=true selects RMSNorm (default), rms=false selects LayerNorm.
static inline dynllama::dynllama_layer dynllama_make_norm(
        const float * weight, const float * bias = nullptr, bool rms = true) {
    dynllama::dynllama_layer L("norm");
    L.bind_weight("weight", weight);
    L.bind_weight("bias", bias);   // null bias -> weight_ptr("bias") == nullptr
    if (rms) {
        L.fwd = [](const dynllama::dynllama_layer & L, dynllama::dynllama_fwd_ctx & c) {
            dynllama_rms_norm_forward(L, c.out, c.in, c.n_tokens, c.n_embd, c.eps);
        };
    } else {
        L.fwd = [](const dynllama::dynllama_layer & L, dynllama::dynllama_fwd_ctx & c) {
            dynllama_norm_forward(L, c.out, c.in, c.n_tokens, c.n_embd, c.eps);
        };
    }
    return L;
}
