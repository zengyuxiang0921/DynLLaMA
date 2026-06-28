//
// Full transformer pipeline: embedding -> N decoder blocks -> output norm -> output projection.
// Composes blocks and layers into a complete forward pass.
// GPU path (__CUDACC__): launches CUDA kernels for the output projection.
// CPU fallback (#else): calls static inline versions.
//

#pragma once

#include "../layers/types.h"
#include <vector>

#include "../blocks/transformer.h"
#include "../layers/inp.h"
#include "../layers/norm.h"
#include "../ops/mul_mat.h"

static inline void llm_pipeline_forward(
        float       * logits,
        const int   * tokens,
        const float * tok_embd,
        // per-layer weight arrays (n_layer pointers each)
        const float ** attn_norm_w,
        const float ** wq, const float ** wk, const float ** wv, const float ** wo,
        const float ** ffn_norm_w,
        const float ** ffn_gate, const float ** ffn_up, const float ** ffn_down,
        const float * output_norm_w,
        const float * output_w,
        int           n_tokens,
        int           n_embd,
        int           n_ff,
        int           n_layer,
        int           n_head,
        int           n_embd_head,
        int           n_vocab,
        int           act_type) {

    const int ne = n_tokens * n_embd;
    const float eps = 1e-5f;

    // hidden state buffer (embedding output, mutated through blocks)
    std::vector<float> hidden(ne);
    // output norm buffer
    std::vector<float> normed(ne);

    // --- token embedding ---
    llm_inp_embd(hidden.data(), tok_embd, tokens, n_tokens, n_embd);

    // --- transformer decoder layers ---
    for (int i = 0; i < n_layer; i++) {
        llm_block_transformer(
            /* out   */ hidden.data(),
            /* in    */ hidden.data(), // in-place: output feeds into next layer
            /* attn  */ attn_norm_w[i],
                         wq[i], wk[i], wv[i], wo[i],
            /* ffn   */ ffn_norm_w[i],
                         ffn_gate[i], ffn_up[i], ffn_down[i],
                         n_tokens, n_embd, n_ff, n_head, n_embd_head, act_type);
    }

    // --- output RMS norm ---
    llm_rms_norm_forward(normed.data(), hidden.data(), output_norm_w,
                         n_tokens, n_embd, eps);

    // --- output projection: logits = output_w * normed ---
    // output_w: [n_vocab, n_embd], normed: [n_embd, n_tokens], logits: [n_vocab, n_tokens]
#ifdef __CUDACC__
    {
        dim3 block(16, 16);
        dim3 grid((n_tokens + 15) / 16, (n_vocab + 15) / 16);
        mul_mat<<<grid, block>>>(output_w, normed.data(), logits,
                                 n_vocab, n_tokens, n_embd);
    }
#else
    mul_mat(output_w, normed.data(), logits, n_vocab, n_tokens, n_embd);
#endif
}

// ggml tensor wrapper
static inline void llm_pipeline_forward(
        struct ggml_tensor * logits,
        const int          * tokens,
        const struct ggml_tensor * tok_embd,
        const struct ggml_tensor ** attn_norm_w,
        const struct ggml_tensor ** wq,
        const struct ggml_tensor ** wk,
        const struct ggml_tensor ** wv,
        const struct ggml_tensor ** wo,
        const struct ggml_tensor ** ffn_norm_w,
        const struct ggml_tensor ** ffn_gate,
        const struct ggml_tensor ** ffn_up,
        const struct ggml_tensor ** ffn_down,
        const struct ggml_tensor * output_norm_w,
        const struct ggml_tensor * output_w,
        int                     n_tokens,
        int                     n_ff,
        int                     n_layer,
        int                     n_head,
        int                     n_embd_head,
        int                     n_vocab,
        int                     act_type) {
    // Build float* arrays from per-layer ggml tensor pointers
    std::vector<const float *> attn_norm_w_f(n_layer);
    std::vector<const float *> wq_f(n_layer);
    std::vector<const float *> wk_f(n_layer);
    std::vector<const float *> wv_f(n_layer);
    std::vector<const float *> wo_f(n_layer);
    std::vector<const float *> ffn_norm_w_f(n_layer);
    std::vector<const float *> ffn_gate_f(n_layer);
    std::vector<const float *> ffn_up_f(n_layer);
    std::vector<const float *> ffn_down_f(n_layer);
    for (int i = 0; i < n_layer; i++) {
        attn_norm_w_f[i] = (const float *)attn_norm_w[i]->data;
        wq_f[i]          = (const float *)wq[i]->data;
        wk_f[i]          = (const float *)wk[i]->data;
        wv_f[i]          = (const float *)wv[i]->data;
        wo_f[i]          = (const float *)wo[i]->data;
        ffn_norm_w_f[i]  = (const float *)ffn_norm_w[i]->data;
        ffn_gate_f[i]    = (const float *)ffn_gate[i]->data;
        ffn_up_f[i]      = (const float *)ffn_up[i]->data;
        ffn_down_f[i]    = (const float *)ffn_down[i]->data;
    }
    llm_pipeline_forward(
        (float *)logits->data,
        tokens,
        (const float *)tok_embd->data,
        attn_norm_w_f.data(),
        wq_f.data(), wk_f.data(), wv_f.data(), wo_f.data(),
        ffn_norm_w_f.data(),
        ffn_gate_f.data(), ffn_up_f.data(), ffn_down_f.data(),
        (const float *)output_norm_w->data,
        (const float *)output_w->data,
        n_tokens, (int)tok_embd->ne[0], n_ff, n_layer,
        n_head, n_embd_head, n_vocab, act_type);
}
