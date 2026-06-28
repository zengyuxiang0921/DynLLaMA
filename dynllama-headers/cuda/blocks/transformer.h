//
// Transformer decoder block: norm -> attn -> residual -> norm -> ffn -> residual
// Composes CUDA layers from ../layers/ into one forward pass.
// GPU path (__CUDACC__): launches CUDA kernels for element-wise ops.
// CPU fallback (#else): calls static inline versions.
//

#pragma once

#include "../layers/types.h"
#include <vector>

#include "../layers/norm.h"
#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../ops/add.h"

static inline void llm_block_transformer(
        float       * out,
        const float * hidden,
        const float * attn_norm_w,
        const float * wq, const float * wk, const float * wv, const float * wo,
        const float * ffn_norm_w,
        const float * ffn_gate, const float * ffn_up, const float * ffn_down,
        int           n_tokens,
        int           n_embd,
        int           n_ff,
        int           n_head,
        int           n_embd_head,
        int           act_type) {

    const int ne = n_tokens * n_embd;
    const float eps = 1e-5f;

    // temp buffers for intermediate results
    std::vector<float> norm_buf(ne);   // pre-attn norm, reused for pre-ffn norm
    std::vector<float> attn_buf(ne);   // attention output, then residual1
    std::vector<float> ffn_buf(ne);    // FFN output

#ifdef __CUDACC__

    // pre-attention RMS norm: norm_buf = rms_norm(hidden)
    llm_rms_norm_forward(norm_buf.data(), hidden, attn_norm_w, n_tokens, n_embd, eps);

    // multi-head attention: attn_buf = attn(norm_buf)
    llm_attn_forward(attn_buf.data(), norm_buf.data(), wq, wk, wv, wo,
                     n_tokens, n_embd, n_head, n_embd_head);

    // residual: attn_buf = attn_buf + hidden
    {
        dim3 block(256);
        dim3 grid((ne + 255) / 256);
        add_ew<<<grid, block>>>(attn_buf.data(), attn_buf.data(), hidden, ne);
    }

    // pre-FFN RMS norm (reuse norm_buf): norm_buf = rms_norm(attn_buf)
    llm_rms_norm_forward(norm_buf.data(), attn_buf.data(), ffn_norm_w,
                         n_tokens, n_embd, eps);

    // FFN: ffn_buf = ffn(norm_buf)
    llm_ffn_forward(ffn_buf.data(), norm_buf.data(), ffn_up, ffn_gate, ffn_down,
                    n_tokens, n_embd, n_ff, act_type);

    // residual: out = ffn_buf + attn_buf
    {
        dim3 block(256);
        dim3 grid((ne + 255) / 256);
        add_ew<<<grid, block>>>(out, ffn_buf.data(), attn_buf.data(), ne);
    }

#else // CPU fallback

    // pre-attention RMS norm
    llm_rms_norm_forward(norm_buf.data(), hidden, attn_norm_w, n_tokens, n_embd, eps);

    // multi-head attention
    llm_attn_forward(attn_buf.data(), norm_buf.data(), wq, wk, wv, wo,
                     n_tokens, n_embd, n_head, n_embd_head);

    // residual: attn_buf = attn_buf + hidden
    add_ew(attn_buf.data(), attn_buf.data(), hidden, ne);

    // pre-FFN RMS norm (reuse norm_buf)
    llm_rms_norm_forward(norm_buf.data(), attn_buf.data(), ffn_norm_w,
                         n_tokens, n_embd, eps);

    // FFN
    llm_ffn_forward(ffn_buf.data(), norm_buf.data(), ffn_up, ffn_gate, ffn_down,
                    n_tokens, n_embd, n_ff, act_type);

    // residual: out = ffn_buf + attn_buf
    add_ew(out, ffn_buf.data(), attn_buf.data(), ne);

#endif
}

// ggml tensor wrapper
static inline void llm_block_transformer(
        struct ggml_tensor * out,
        const struct ggml_tensor * hidden,
        const struct ggml_tensor * attn_norm_w,
        const struct ggml_tensor * wq,
        const struct ggml_tensor * wk,
        const struct ggml_tensor * wv,
        const struct ggml_tensor * wo,
        const struct ggml_tensor * ffn_norm_w,
        const struct ggml_tensor * ffn_gate,
        const struct ggml_tensor * ffn_up,
        const struct ggml_tensor * ffn_down,
        int                     n_tokens,
        int                     n_ff,
        int                     n_head,
        int                     n_embd_head,
        int                     act_type) {
    llm_block_transformer(
        (float *)out->data,
        (const float *)hidden->data,
        (const float *)attn_norm_w->data,
        (const float *)wq->data,
        (const float *)wk->data,
        (const float *)wv->data,
        (const float *)wo->data,
        (const float *)ffn_norm_w->data,
        (const float *)ffn_gate->data,
        (const float *)ffn_up->data,
        (const float *)ffn_down->data,
        n_tokens, (int)hidden->ne[0], n_ff, n_head, n_embd_head, act_type);
}
