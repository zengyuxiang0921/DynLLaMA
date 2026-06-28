#pragma once

#include <cmath>

static inline void soft_max(float * out, const float * in, int n, int n_embd) {
    for (int i = 0; i < n; i++) {
        const float * row = in + i * n_embd;
        float * orow = out + i * n_embd;

        float max_val = row[0];
        for (int j = 1; j < n_embd; j++) {
            if (row[j] > max_val) max_val = row[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            float e = std::exp(row[j] - max_val);
            orow[j] = e;
            sum += e;
        }

        for (int j = 0; j < n_embd; j++) {
            orow[j] /= sum;
        }
    }
}
