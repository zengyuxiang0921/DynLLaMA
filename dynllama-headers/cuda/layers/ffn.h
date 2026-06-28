//
// CUDA FFN layer: up/gate/down projection + activation (SiLU/GELU/ReLU).
// Composes CUDA ops from ../ops/ into a single forward pass.
// GPU path (__CUDACC__): launches CUDA kernels with <<<>>> syntax.
// CPU fallback (#else): calls static inline op versions with std::vector temps.
//

#pragma once

#include "types.h"
#include <vector>

#include "../ops/mul_mat.h"
#include "../ops/silu.h"
#include "../ops/gelu.h"
#include "../ops/relu.h"
#include "../ops/mul.h"

static inline void llm_ffn_forward(
        float       * out,
        const float * hidden,
        const float * up_w,
        const float * gate_w,
        const float * down_w,
        int           n_tokens,
        int           n_embd,
        int           n_ff,
        int           act_type) { // 0=silu, 1=gelu, 2=relu

    const int ne_ff_tokens = n_ff * n_tokens;

#ifdef __CUDACC__

    std::vector<float> up_buf(ne_ff_tokens);
    std::vector<float> gate_buf(ne_ff_tokens);

    // up projection: up_w [n_ff, n_embd] x hidden [n_embd, n_tokens] -> up [n_ff, n_tokens]
    {
        dim3 block(16, 16);
        dim3 grid((n_tokens + 15) / 16, (n_ff + 15) / 16);
        mul_mat<<<grid, block>>>(up_w, hidden, up_buf.data(), n_ff, n_tokens, n_embd);
    }

    // gate projection: gate_w [n_ff, n_embd] x hidden [n_embd, n_tokens] -> gate [n_ff, n_tokens]
    {
        dim3 block(16, 16);
        dim3 grid((n_tokens + 15) / 16, (n_ff + 15) / 16);
        mul_mat<<<grid, block>>>(gate_w, hidden, gate_buf.data(), n_ff, n_tokens, n_embd);
    }

    // activation on gate_buf
    {
        dim3 block(256);
        dim3 grid((ne_ff_tokens + 255) / 256);
        if (act_type == 0) {
            silu<<<grid, block>>>(gate_buf.data(), gate_buf.data(), ne_ff_tokens);
        } else if (act_type == 1) {
            gelu<<<grid, block>>>(gate_buf.data(), gate_buf.data(), ne_ff_tokens);
        } else {
            relu<<<grid, block>>>(gate_buf.data(), gate_buf.data(), ne_ff_tokens);
        }
    }

    // element-wise multiply: gated = up * gate
    {
        dim3 block(256);
        dim3 grid((ne_ff_tokens + 255) / 256);
        mul_ew<<<grid, block>>>(gate_buf.data(), up_buf.data(), gate_buf.data(), ne_ff_tokens);
    }

    // down projection: down_w [n_embd, n_ff] x gated [n_ff, n_tokens] -> out [n_embd, n_tokens]
    {
        dim3 block(16, 16);
        dim3 grid((n_tokens + 15) / 16, (n_embd + 15) / 16);
        mul_mat<<<grid, block>>>(down_w, gate_buf.data(), out, n_embd, n_tokens, n_ff);
    }

#else // CPU fallback

    std::vector<float> up_buf(ne_ff_tokens);
    std::vector<float> gate_buf(ne_ff_tokens);

    // up projection
    mul_mat(up_w, hidden, up_buf.data(), n_ff, n_tokens, n_embd);

    // gate projection
    mul_mat(gate_w, hidden, gate_buf.data(), n_ff, n_tokens, n_embd);

    // activation
    if (act_type == 0) {
        silu(gate_buf.data(), gate_buf.data(), ne_ff_tokens);
    } else if (act_type == 1) {
        gelu(gate_buf.data(), gate_buf.data(), ne_ff_tokens);
    } else {
        relu(gate_buf.data(), gate_buf.data(), ne_ff_tokens);
    }

    // element-wise multiply: gated = up * gate (reuse gate_buf)
    mul_ew(gate_buf.data(), up_buf.data(), gate_buf.data(), ne_ff_tokens);

    // down projection
    mul_mat(down_w, gate_buf.data(), out, n_embd, n_tokens, n_ff);

#endif
}

// ggml tensor wrapper
static inline void llm_ffn_forward(
        struct ggml_tensor * out,
        const struct ggml_tensor * hidden,
        const struct ggml_tensor * up_w,
        const struct ggml_tensor * gate_w,
        const struct ggml_tensor * down_w,
        int                     n_tokens,
        int                     n_ff,
        int                     act_type) {
    llm_ffn_forward(
        (float *)out->data,
        (const float *)hidden->data,
        (const float *)up_w->data,
        (const float *)gate_w->data,
        (const float *)down_w->data,
        n_tokens, (int)hidden->ne[0], n_ff, act_type);
}
