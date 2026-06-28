#pragma once

#include <cmath>

static inline void rms_norm(float * out, const float * x, const float * weight, int n, int n_embd, float eps) {
    for (int i = 0; i < n; i++) {
        const float * row = x + i * n_embd;
        float * orow = out + i * n_embd;

        float ss = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            ss += row[j] * row[j];
        }
        float rms = std::sqrt(ss / (float)n_embd + eps);
        float inv_rms = 1.0f / rms;

        for (int j = 0; j < n_embd; j++) {
            orow[j] = weight[j] * (row[j] * inv_rms);
        }
    }
}
