//
// CPU forward pipeline: embedding -> N transformer blocks -> output norm -> lm_head.
// Pure float-pointer implementation - no ggml.
//
// Two variants:
//   llm_pipeline_forward     - stateless full-sequence pass
//   llm_pipeline_forward_kv  - incremental pass using a host-owned KV cache
//
// NOTE: RoPE is not modelled yet; attention is scaled-dot-product + causal.
//

#pragma once

#include "transformer.h"
#include "../layers/inp.h"
#include "../layers/norm.h"
#include "../ops/linear.h"

#include "dynllama_abi.h"   // dynllama_layer_weights, dynllama_kv_cache (-I dynllama-headers)

#include <vector>

static inline void llm_pipeline_forward(
        float                        * logits,        // [n_vocab, n_tokens]
        const int                    * tokens,        // [n_tokens]
        const float                  * tok_embd,      // [n_embd, n_vocab]
        const dynllama_layer_weights * layers,
        int                            n_layer,
        const float                  * output_norm_w, // [n_embd]
        const float                  * output_w,      // [n_vocab, n_embd]
        int                            n_tokens,
        int                            n_embd,
        int                            n_ff,
        int                            n_head,
        int                            n_head_kv,
        int                            n_embd_head,
        int                            n_vocab,
        int                            n_rot,
        float                          rope_freq_base,
        int                            act_type,
        float                          rms_eps) {

    const int ne = n_tokens * n_embd;

    std::vector<float> hidden(ne);
    std::vector<float> next(ne);
    std::vector<float> normed(ne);

    dynllama::dynllama_layer inp      = dynllama_make_inp(tok_embd);
    dynllama::dynllama_layer out_norm = dynllama_make_norm(output_norm_w);
    dynllama::dynllama_fwd_ctx c;
    c.n_tokens = n_tokens; c.n_embd = n_embd; c.eps = rms_eps;

    c.out = hidden.data(); c.tokens = tokens;
    dynllama::dynllama_forward(inp, c);

    for (int il = 0; il < n_layer; il++) {
        const dynllama_layer_weights & w = layers[il];
        llm_block_transformer(
            next.data(), hidden.data(),
            w.attn_norm,
            w.wq, w.bq, w.wk, w.bk, w.wv, w.bv, w.wo, w.wo_b,
            w.ffn_norm, w.ffn_up, w.ffn_gate, w.ffn_down,
            n_tokens, n_embd, n_ff, n_head, n_head_kv, n_embd_head,
            n_rot, rope_freq_base, act_type, rms_eps);
        hidden.swap(next);
    }

    c.out = normed.data(); c.in = hidden.data();
    dynllama::dynllama_forward(out_norm, c);
    linear(normed.data(), output_w, nullptr, logits, n_tokens, n_vocab, n_embd);
}

// Incremental pipeline: appends the new tokens to the KV cache at n_past.
static inline void llm_pipeline_forward_kv(
        float                        * logits,
        const int                    * tokens,
        const float                  * tok_embd,
        const dynllama_layer_weights * layers,
        int                            n_layer,
        const float                  * output_norm_w,
        const float                  * output_w,
        dynllama_kv_cache            * kv,
        int                            n_tokens,
        int                            n_embd,
        int                            n_ff,
        int                            n_head,
        int                            n_head_kv,
        int                            n_embd_head,
        int                            n_vocab,
        int                            n_rot,
        float                          rope_freq_base,
        int                            n_past,
        int                            act_type,
        float                          rms_eps) {

    const int ne        = n_tokens * n_embd;
    const int n_embd_kv = n_head_kv * n_embd_head;
    const int n_ctx     = kv->n_ctx;
    const int64_t layer_stride = (int64_t) n_embd_kv * n_ctx;

    std::vector<float> hidden(ne);
    std::vector<float> next(ne);
    std::vector<float> normed(ne);

    dynllama::dynllama_layer inp      = dynllama_make_inp(tok_embd);
    dynllama::dynllama_layer out_norm = dynllama_make_norm(output_norm_w);
    dynllama::dynllama_fwd_ctx c;
    c.n_tokens = n_tokens; c.n_embd = n_embd; c.eps = rms_eps;

    c.out = hidden.data(); c.tokens = tokens;
    dynllama::dynllama_forward(inp, c);

    for (int il = 0; il < n_layer; il++) {
        const dynllama_layer_weights & w = layers[il];
        float * k_cache = kv->k + il * layer_stride;
        float * v_cache = kv->v + il * layer_stride;
        llm_block_transformer_kv(
            next.data(), hidden.data(),
            w.attn_norm,
            w.wq, w.bq, w.wk, w.bk, w.wv, w.bv, w.wo, w.wo_b,
            w.ffn_norm, w.ffn_up, w.ffn_gate, w.ffn_down,
            k_cache, v_cache,
            n_tokens, n_embd, n_ff, n_head, n_head_kv, n_embd_head,
            n_rot, rope_freq_base, n_past, n_ctx, act_type, rms_eps);
        hidden.swap(next);
    }

    c.out = normed.data(); c.in = hidden.data();
    dynllama::dynllama_forward(out_norm, c);
    linear(normed.data(), output_w, nullptr, logits, n_tokens, n_vocab, n_embd);
}
