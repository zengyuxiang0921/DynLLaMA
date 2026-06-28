//
// CPU multi-head attention (token-major) with grouped-query attention (GQA).
// Built as a dynllama_layer holding the output projection (wo, optional wo_b).
//
// Inputs are already-projected Q/K/V, token-major:
//   q [n_tokens, n_head    * n_embd_head]
//   k [n_tokens, n_head_kv * n_embd_head]
//   v [n_tokens, n_head_kv * n_embd_head]
// Query head h reads KV head h / (n_head / n_head_kv). Optional causal masking.
// RoPE is applied by the caller (block) before this forward.
//

#pragma once

#include "dynllama_layer.h"
#include <vector>
#include <cmath>

#include "../ops/linear.h"

static inline void dynllama_attn_forward(
        const dynllama::dynllama_layer & L,
        float       * out,          // [n_tokens, n_embd]
        const float * q,            // [n_tokens, n_head    * n_embd_head]
        const float * k,            // [n_tokens, n_head_kv * n_embd_head]
        const float * v,            // [n_tokens, n_head_kv * n_embd_head]
        int           n_tokens,
        int           n_embd,
        int           n_head,
        int           n_head_kv,
        int           n_embd_head,
        bool          causal = true) {

    const int n_embd_q  = n_head    * n_embd_head;
    const int n_embd_kv = n_head_kv * n_embd_head;
    const int group     = n_head / (n_head_kv > 0 ? n_head_kv : n_head);
    const float scale   = 1.0f / std::sqrt((float) n_embd_head);

    std::vector<float> attn(n_embd_q * n_tokens, 0.0f);
    std::vector<float> scores(n_tokens);

    for (int h = 0; h < n_head; h++) {
        const int kvh    = group > 0 ? h / group : 0;
        const int q_off  = h   * n_embd_head;
        const int kv_off = kvh * n_embd_head;

        for (int t1 = 0; t1 < n_tokens; t1++) {
            const int last = causal ? t1 : (n_tokens - 1);

            float maxv = -3.402823e38f;
            for (int t2 = 0; t2 <= last; t2++) {
                float s = 0.0f;
                for (int d = 0; d < n_embd_head; d++) {
                    s += q[t1 * n_embd_q  + q_off  + d] *
                         k[t2 * n_embd_kv + kv_off + d];
                }
                s *= scale;
                scores[t2] = s;
                if (s > maxv) maxv = s;
            }

            float sum = 0.0f;
            for (int t2 = 0; t2 <= last; t2++) {
                float e = std::exp(scores[t2] - maxv);
                scores[t2] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

            for (int d = 0; d < n_embd_head; d++) {
                float acc = 0.0f;
                for (int t2 = 0; t2 <= last; t2++) {
                    acc += scores[t2] * v[t2 * n_embd_kv + kv_off + d];
                }
                attn[t1 * n_embd_q + q_off + d] = acc * inv;
            }
        }
    }

    linear(attn.data(), L.weight_ptr("wo"), L.weight_ptr("wo_b"),
           out, n_tokens, n_embd, n_embd_q);
}

// KV-cache-aware attention for incremental decoding (declared above make_attn). Appends new K/V into the
// per-layer cache at [n_past, n_past+n_tokens), then each new query attends over
// cached keys [0, n_past+i] (causal). Cache layout per layer: [n_ctx, n_embd_kv].
static inline void dynllama_attn_forward_kv(
        const dynllama::dynllama_layer & L,
        float       * out,
        const float * q,
        const float * k_new,
        const float * v_new,
        float       * k_cache,
        float       * v_cache,
        int           n_tokens,
        int           n_embd,
        int           n_head,
        int           n_head_kv,
        int           n_embd_head,
        int           n_past,
        int           n_ctx) {

    const int n_embd_q  = n_head    * n_embd_head;
    const int n_embd_kv = n_head_kv * n_embd_head;
    const int group     = n_head / (n_head_kv > 0 ? n_head_kv : n_head);
    const float scale   = 1.0f / std::sqrt((float) n_embd_head);

    for (int i = 0; i < n_tokens; i++) {
        const int pos = n_past + i;
        if (pos >= n_ctx) continue;
        for (int d = 0; d < n_embd_kv; d++) {
            k_cache[pos * n_embd_kv + d] = k_new[i * n_embd_kv + d];
            v_cache[pos * n_embd_kv + d] = v_new[i * n_embd_kv + d];
        }
    }

    std::vector<float> attn(n_embd_q * n_tokens, 0.0f);
    std::vector<float> scores(n_past + n_tokens);

    for (int h = 0; h < n_head; h++) {
        const int kvh    = group > 0 ? h / group : 0;
        const int q_off  = h   * n_embd_head;
        const int kv_off = kvh * n_embd_head;

        for (int i = 0; i < n_tokens; i++) {
            const int last = n_past + i;

            float maxv = -3.402823e38f;
            for (int t2 = 0; t2 <= last; t2++) {
                float s = 0.0f;
                for (int d = 0; d < n_embd_head; d++) {
                    s += q[i * n_embd_q + q_off + d] *
                         k_cache[t2 * n_embd_kv + kv_off + d];
                }
                s *= scale;
                scores[t2] = s;
                if (s > maxv) maxv = s;
            }

            float sum = 0.0f;
            for (int t2 = 0; t2 <= last; t2++) {
                float e = std::exp(scores[t2] - maxv);
                scores[t2] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

            for (int d = 0; d < n_embd_head; d++) {
                float acc = 0.0f;
                for (int t2 = 0; t2 <= last; t2++) {
                    acc += scores[t2] * v_cache[t2 * n_embd_kv + kv_off + d];
                }
                attn[i * n_embd_q + q_off + d] = acc * inv;
            }
        }
    }

    linear(attn.data(), L.weight_ptr("wo"), L.weight_ptr("wo_b"),
           out, n_tokens, n_embd, n_embd_q);
}

// Build an attention layer holding the output projection wo (+ optional wo_b).
// Its forward selects the KV-cache path when ctx.k_cache is set.
static inline dynllama::dynllama_layer dynllama_make_attn(
        const float * wo, const float * wo_b = nullptr) {
    dynllama::dynllama_layer L("attn");
    L.bind_weight("wo", wo);
    L.bind_weight("wo_b", wo_b);
    L.fwd = [](const dynllama::dynllama_layer & L, dynllama::dynllama_fwd_ctx & c) {
        if (c.k_cache) {
            dynllama_attn_forward_kv(L, c.out, c.q, c.k, c.v, c.k_cache, c.v_cache,
                                     c.n_tokens, c.n_embd, c.n_head, c.n_head_kv,
                                     c.n_embd_head, c.n_past, c.n_ctx);
        } else {
            dynllama_attn_forward(L, c.out, c.q, c.k, c.v,
                                  c.n_tokens, c.n_embd, c.n_head, c.n_head_kv,
                                  c.n_embd_head, c.causal);
        }
    };
    return L;
}
