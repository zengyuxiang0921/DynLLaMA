#pragma once

#include <cmath>

static inline void silu(float * out, const float * in, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] / (1.0f + std::exp(-in[i]));
    }
}
