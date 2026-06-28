//
// CUDA QKV projection layer: separate Q, K, V projections via mul_mat.
// Each projection is a matrix multiply: weight [embd_out, n_embd] x hidden [n_embd, n_tokens].
// GPU path (__CUDACC__): launches mul_mat CUDA kernels.
// CPU fallback (#else): calls static inline mul_mat directly.
// No temp buffers needed — each projection writes directly to its output.
//

#pragma once

#include "types.h"
#include "../ops/mul_mat.h"

static inline void llm_qkv_forward(
        float       * q_out,
        float       * k_out,
        float       * v_out,
        const float * hidden,
        const float * wq,
        const float * wk,
        const float * wv,
        int           n_tokens,
        int           n_embd,
        int           n_embd_head,
        int           n_head,
        int           n_head_kv,
        int           n_embd_q,
        int           n_embd_kv) {

    (void)n_embd_head;
    (void)n_head;
    (void)n_head_kv;

#ifdef __CUDACC__
    // Q projection: wq [n_embd_q, n_embd] x hidden [n_embd, n_tokens] -> q_out [n_embd_q, n_tokens]
    {
        dim3 block(16, 16);
        dim3 grid((n_tokens + 15) / 16, (n_embd_q + 15) / 16);
        mul_mat<<<grid, block>>>(wq, hidden, q_out, n_embd_q, n_tokens, n_embd);
    }

    // K projection: wk [n_embd_kv, n_embd] x hidden [n_embd, n_tokens] -> k_out [n_embd_kv, n_tokens]
    {
        dim3 block(16, 16);
        dim3 grid((n_tokens + 15) / 16, (n_embd_kv + 15) / 16);
        mul_mat<<<grid, block>>>(wk, hidden, k_out, n_embd_kv, n_tokens, n_embd);
    }

    // V projection: wv [n_embd_kv, n_embd] x hidden [n_embd, n_tokens] -> v_out [n_embd_kv, n_tokens]
    {
        dim3 block(16, 16);
        dim3 grid((n_tokens + 15) / 16, (n_embd_kv + 15) / 16);
        mul_mat<<<grid, block>>>(wv, hidden, v_out, n_embd_kv, n_tokens, n_embd);
    }
#else
    mul_mat(wq, hidden, q_out, n_embd_q, n_tokens, n_embd);
    mul_mat(wk, hidden, k_out, n_embd_kv, n_tokens, n_embd);
    mul_mat(wv, hidden, v_out, n_embd_kv, n_tokens, n_embd);
#endif
}

// ggml tensor wrapper
static inline void llm_qkv_forward(
        struct ggml_tensor * q_out,
        struct ggml_tensor * k_out,
        struct ggml_tensor * v_out,
        const struct ggml_tensor * hidden,
        const struct ggml_tensor * wq,
        const struct ggml_tensor * wk,
        const struct ggml_tensor * wv,
        int                     n_tokens,
        int                     n_embd_head,
        int                     n_head,
        int                     n_head_kv,
        int                     n_embd_q,
        int                     n_embd_kv) {
    llm_qkv_forward(
        (float *)q_out->data,
        (float *)k_out->data,
        (float *)v_out->data,
        (const float *)hidden->data,
        (const float *)wq->data,
        (const float *)wk->data,
        (const float *)wv->data,
        n_tokens, (int)hidden->ne[0],
        n_embd_head, n_head, n_head_kv, n_embd_q, n_embd_kv);
}
