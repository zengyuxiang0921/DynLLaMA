#pragma once

static inline void get_rows(float * out, const float * embd, const int * ids, int n, int n_embd) {
    for (int i = 0; i < n; i++) {
        int idx = ids[i];
        for (int j = 0; j < n_embd; j++) {
            out[i * n_embd + j] = embd[idx * n_embd + j];
        }
    }
}
