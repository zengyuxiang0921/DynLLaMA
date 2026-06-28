//
// CPU QKV projection (token-major, ggml-compatible weight layout).
// Built as a dynllama_layer holding the Q/K/V weights and optional biases.
//

#pragma once

#include "dynllama_layer.h"
#include "../ops/linear.h"

static inline void dynllama_qkv_forward(
        const dynllama::dynllama_layer & L,
        float       * q_out,
        float       * k_out,
        float       * v_out,
        const float * hidden,
        int           n_tokens,
        int           n_embd,
        int           n_embd_q,
        int           n_embd_kv) {
    linear(hidden, L.weight_ptr("wq"), L.weight_ptr("bq"), q_out, n_tokens, n_embd_q,  n_embd);
    linear(hidden, L.weight_ptr("wk"), L.weight_ptr("bk"), k_out, n_tokens, n_embd_kv, n_embd);
    linear(hidden, L.weight_ptr("wv"), L.weight_ptr("bv"), v_out, n_tokens, n_embd_kv, n_embd);
}

// Build a QKV layer holding the three projection weights and optional biases.
static inline dynllama::dynllama_layer dynllama_make_qkv(
        const float * wq, const float * bq,
        const float * wk, const float * bk,
        const float * wv, const float * bv) {
    dynllama::dynllama_layer L("qkv");
    L.bind_weight("wq", wq); L.bind_weight("bq", bq);
    L.bind_weight("wk", wk); L.bind_weight("bk", bk);
    L.bind_weight("wv", wv); L.bind_weight("bv", bv);
    L.fwd = [](const dynllama::dynllama_layer & L, dynllama::dynllama_fwd_ctx & c) {
        dynllama_qkv_forward(L, c.q, c.k, c.v, c.in,
                             c.n_tokens, c.n_embd, c.n_embd_q, c.n_embd_kv);
    };
    return L;
}
