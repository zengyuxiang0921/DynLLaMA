#pragma once

#include <cmath>

//
// NEOX-style rotary position embedding (llama / qwen2 / gpt-neox), in-place.
//
// x is token-major [n_tokens, n_heads * n_embd_head]. For each token at
// absolute position (pos_offset + t) and each head, rotates the pair
// (i, i + n_rot/2) for i in [0, n_rot/2) by angle pos * freq_base^(-2i/n_rot).
// Dimensions >= n_rot are left unchanged.
//
static inline void rope_neox(float * x,
                             int   n_tokens,
                             int   n_heads,
                             int   n_embd_head,
                             int   n_rot,
                             float freq_base,
                             int   pos_offset) {
    const int half  = n_rot / 2;
    const int hdim  = n_heads * n_embd_head;
    for (int t = 0; t < n_tokens; t++) {
        const int pos = pos_offset + t;
        for (int i = 0; i < half; i++) {
            const float inv_freq = std::pow(freq_base, -2.0f * (float) i / (float) n_rot);
            const float theta = (float) pos * inv_freq;
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            for (int h = 0; h < n_heads; h++) {
                const int base = t * hdim + h * n_embd_head;
                const float x0 = x[base + i];
                const float x1 = x[base + i + half];
                x[base + i]        = x0 * c - x1 * s;
                x[base + i + half] = x0 * s + x1 * c;
            }
        }
    }
}

static inline void rope(float * out, const float * in, const float * freqs, int n_tokens, int n_embd, int n_head, int n_embd_head) {
    for (int i = 0; i < n_tokens; i++) {
        for (int h = 0; h < n_head; h++) {
            for (int j = 0; j < n_embd_head; j += 2) {
                int idx = i * n_embd + h * n_embd_head + j;
                float cos = std::cos(freqs[i * n_embd_head / 2 + j / 2]);
                float sin = std::sin(freqs[i * n_embd_head / 2 + j / 2]);

                float x0 = in[idx];
                float x1 = in[idx + 1];

                out[idx]     = x0 * cos - x1 * sin;
                out[idx + 1] = x0 * sin + x1 * cos;
            }
        }
    }
}
