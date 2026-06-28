//
// Repeat a block function N times over a buffer with ping-pong buffering.
// Avoids unnecessary copies between iterations. Pure CPU, no ggml.
//

#pragma once

#include <vector>
#include <cstring>

static inline void llm_block_repeat(
        float       * out,
        const float * in,
        void       (*block_fn)(float * out, const float * in, int il),
        int           n_repeat,
        int           n_embd,
        int           n_tokens) {

    const int ne = n_tokens * n_embd;

    if (n_repeat <= 0) {
        if (out != in) std::memcpy(out, in, ne * sizeof(float));
        return;
    }
    if (n_repeat == 1) {
        block_fn(out, in, 0);
        return;
    }

    std::vector<float> buf_a(ne);
    std::vector<float> buf_b(ne);

    block_fn(buf_a.data(), in, 0);
    for (int i = 1; i < n_repeat - 1; i++) {
        if (i % 2 == 1) block_fn(buf_b.data(), buf_a.data(), i);
        else            block_fn(buf_a.data(), buf_b.data(), i);
    }
    const float * src = (n_repeat % 2 == 0) ? buf_a.data() : buf_b.data();
    block_fn(out, src, n_repeat - 1);
}
