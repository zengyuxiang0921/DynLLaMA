//
// Linear (dense) op, ggml-compatible weight layout.
//
// Activations are token-major:           x[t * n_in  + i]   (t in tokens, i in features)
// Weights are stored as ggml stores them: w[o * n_in  + i]   (o = output unit)
//   (a ggml tensor with ne = [n_in, n_out] has element (i,o) at o*n_in + i)
//
// Computes:  out[t * n_out + o] = sum_i x[t,i] * w[o,i] (+ bias[o])
//
// This is the natural transformer linear layer and needs no weight transpose:
// the host dequantizes each ggml weight tensor as-is and passes it straight in.
//

#pragma once

#include <cstddef>

static inline void linear(
        const float * x,
        const float * w,
        const float * bias,   // [n_out] or null
        float       * out,
        int           n_tokens,
        int           n_out,
        int           n_in) {
    for (int t = 0; t < n_tokens; t++) {
        const float * xr   = x   + (size_t) t * n_in;
        float       * orow = out + (size_t) t * n_out;
        for (int o = 0; o < n_out; o++) {
            const float * wr = w + (size_t) o * n_in;
            float s = 0.0f;
            for (int i = 0; i < n_in; i++) {
                s += xr[i] * wr[i];
            }
            orow[o] = bias ? s + bias[o] : s;
        }
    }
}
