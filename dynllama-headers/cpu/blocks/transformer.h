//
// CPU transformer decoder block:
//   x  = rms_norm(attn_norm, hidden)
//   x  = attn(qkv(x))           ; residual: a = x + hidden
//   y  = rms_norm(ffn_norm, a)
//   y  = ffn(y)                 ; residual: out = y + a
//
// Each sub-step is a dynllama_layer built from the supplied weights and run
// through the standard dynllama_forward(layer, ctx). Qwen2-style QKV bias is
// supported. RoPE is applied to Q/K between QKV and attention.
//

#pragma once

#include "../layers/norm.h"
#include "../layers/qkv.h"
#include "../layers/attention.h"
#include "../layers/ffn.h"
#include "../ops/add.h"
#include "../ops/rope.h"

#include <vector>

static inline void llm_block_transformer(
        float       * out,
        const float * hidden,
        const float * attn_norm_w,
        const float * wq, const float * bq,
        const float * wk, const float * bk,
        const float * wv, const float * bv,
        const float * wo, const float * wo_b,
        const float * ffn_norm_w,
        const float * ffn_up, const float * ffn_gate, const float * ffn_down,
        int           n_tokens,
        int           n_embd,
        int           n_ff,
        int           n_head,
        int           n_head_kv,
        int           n_embd_head,
        int           n_rot,
        float         rope_freq_base,
        int           act_type,
        float         rms_eps) {

    const int ne        = n_embd * n_tokens;
    const int n_embd_q  = n_head    * n_embd_head;
    const int n_embd_kv = n_head_kv * n_embd_head;

    std::vector<float> norm_buf(ne);
    std::vector<float> q(n_embd_q  * n_tokens);
    std::vector<float> k(n_embd_kv * n_tokens);
    std::vector<float> v(n_embd_kv * n_tokens);
    std::vector<float> attn_out(ne);
    std::vector<float> resid1(ne);
    std::vector<float> ffn_buf(ne);

    dynllama::dynllama_layer attn_norm = dynllama_make_norm(attn_norm_w);
    dynllama::dynllama_layer qkv       = dynllama_make_qkv(wq, bq, wk, bk, wv, bv);
    dynllama::dynllama_layer attn      = dynllama_make_attn(wo, wo_b);
    dynllama::dynllama_layer ffn_norm  = dynllama_make_norm(ffn_norm_w);
    dynllama::dynllama_layer ffn       = dynllama_make_ffn(ffn_up, ffn_gate, ffn_down, act_type);

    dynllama::dynllama_fwd_ctx c;
    c.n_tokens = n_tokens; c.n_embd = n_embd; c.n_ff = n_ff;
    c.n_head = n_head; c.n_head_kv = n_head_kv; c.n_embd_head = n_embd_head;
    c.n_embd_q = n_embd_q; c.n_embd_kv = n_embd_kv;
    c.act_type = act_type; c.eps = rms_eps; c.causal = true;

    // pre-attention norm
    c.out = norm_buf.data(); c.in = hidden;
    dynllama::dynllama_forward(attn_norm, c);

    // QKV projection (with bias)
    c.in = norm_buf.data(); c.q = q.data(); c.k = k.data(); c.v = v.data();
    dynllama::dynllama_forward(qkv, c);

    // rotary position embedding on Q and K (positions 0..n_tokens-1)
    rope_neox(q.data(), n_tokens, n_head,    n_embd_head, n_rot, rope_freq_base, 0);
    rope_neox(k.data(), n_tokens, n_head_kv, n_embd_head, n_rot, rope_freq_base, 0);

    // attention + output projection
    c.out = attn_out.data();
    dynllama::dynllama_forward(attn, c);

    // residual 1
    add_ew(resid1.data(), attn_out.data(), hidden, ne);

    // pre-FFN norm + FFN
    c.out = norm_buf.data(); c.in = resid1.data();
    dynllama::dynllama_forward(ffn_norm, c);
    c.out = ffn_buf.data();  c.in = norm_buf.data();
    dynllama::dynllama_forward(ffn, c);

    // residual 2
    add_ew(out, ffn_buf.data(), resid1.data(), ne);
}

// KV-cache-aware transformer block for incremental decoding.
static inline void llm_block_transformer_kv(
        float       * out,
        const float * hidden,
        const float * attn_norm_w,
        const float * wq, const float * bq,
        const float * wk, const float * bk,
        const float * wv, const float * bv,
        const float * wo, const float * wo_b,
        const float * ffn_norm_w,
        const float * ffn_up, const float * ffn_gate, const float * ffn_down,
        float       * k_cache, float * v_cache,
        int           n_tokens,
        int           n_embd,
        int           n_ff,
        int           n_head,
        int           n_head_kv,
        int           n_embd_head,
        int           n_rot,
        float         rope_freq_base,
        int           n_past,
        int           n_ctx,
        int           act_type,
        float         rms_eps) {

    const int ne        = n_embd * n_tokens;
    const int n_embd_q  = n_head    * n_embd_head;
    const int n_embd_kv = n_head_kv * n_embd_head;

    std::vector<float> norm_buf(ne);
    std::vector<float> q(n_embd_q  * n_tokens);
    std::vector<float> k(n_embd_kv * n_tokens);
    std::vector<float> v(n_embd_kv * n_tokens);
    std::vector<float> attn_out(ne);
    std::vector<float> resid1(ne);
    std::vector<float> ffn_buf(ne);

    dynllama::dynllama_layer attn_norm = dynllama_make_norm(attn_norm_w);
    dynllama::dynllama_layer qkv       = dynllama_make_qkv(wq, bq, wk, bk, wv, bv);
    dynllama::dynllama_layer attn      = dynllama_make_attn(wo, wo_b);
    dynllama::dynllama_layer ffn_norm  = dynllama_make_norm(ffn_norm_w);
    dynllama::dynllama_layer ffn       = dynllama_make_ffn(ffn_up, ffn_gate, ffn_down, act_type);

    dynllama::dynllama_fwd_ctx c;
    c.n_tokens = n_tokens; c.n_embd = n_embd; c.n_ff = n_ff;
    c.n_head = n_head; c.n_head_kv = n_head_kv; c.n_embd_head = n_embd_head;
    c.n_embd_q = n_embd_q; c.n_embd_kv = n_embd_kv;
    c.n_past = n_past; c.n_ctx = n_ctx;
    c.act_type = act_type; c.eps = rms_eps;

    c.out = norm_buf.data(); c.in = hidden;
    dynllama::dynllama_forward(attn_norm, c);

    c.in = norm_buf.data(); c.q = q.data(); c.k = k.data(); c.v = v.data();
    dynllama::dynllama_forward(qkv, c);

    // rotary position embedding on Q and K (absolute positions start at n_past)
    rope_neox(q.data(), n_tokens, n_head,    n_embd_head, n_rot, rope_freq_base, n_past);
    rope_neox(k.data(), n_tokens, n_head_kv, n_embd_head, n_rot, rope_freq_base, n_past);

    // attention over the KV cache (k_cache set -> KV path selected by the layer)
    c.out = attn_out.data(); c.k_cache = k_cache; c.v_cache = v_cache;
    dynllama::dynllama_forward(attn, c);

    add_ew(resid1.data(), attn_out.data(), hidden, ne);

    c.out = norm_buf.data(); c.in = resid1.data();
    dynllama::dynllama_forward(ffn_norm, c);
    c.out = ffn_buf.data();  c.in = norm_buf.data();
    dynllama::dynllama_forward(ffn, c);

    add_ew(out, ffn_buf.data(), resid1.data(), ne);
}
