#pragma once

#include <cmath>

static inline void gelu(float * out, const float * in, int n) {
    for (int i = 0; i < n; i++) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}
