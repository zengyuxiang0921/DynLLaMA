//
// CUDA multi-head attention layer: QKV projection, QK^T scores, softmax, V weighting, output projection.
// Composes CUDA ops from ../ops/ plus a dedicated attention core kernel.
// GPU path (__CUDACC__): launches CUDA kernels including custom attn_core_kernel.
// CPU fallback (#else): plain loops with std::vector temps.
//
// Implementation:
//   1. Q = mul_mat(wq, hidden)  [n_embd_q, n_tokens]
//   2. K = mul_mat(wk, hidden)  [n_embd_kv, n_tokens]
//   3. V = mul_mat(wv, hidden)  [n_embd_kv, n_tokens]
//   4. For each head: S = Q_h^T * K_h / sqrt(d) -> softmax -> * V_h
//   5. out = mul_mat(wo, attn_out)  [n_embd, n_tokens]
//

#pragma once

#include "types.h"
#include <vector>
#include <cmath>

#include "../ops/mul_mat.h"

#ifdef __CUDACC__

// ---------------------------------------------------------------------------
// Attention core CUDA kernel: for one head h and query position t1,
// computes score[t2] = Q_h[d][t1] * K_h[d][t2] / scale,
// applies softmax, then weights V.
// Grid: (n_head, n_tokens), Block: up to 256 threads.
// Uses global memory for scores (via scores_buf).
// ---------------------------------------------------------------------------
extern "C" __global__
void attn_core_kernel(
        float       * out,        // [n_head * n_embd_head, n_tokens]
        const float * q,          // [n_head * n_embd_head, n_tokens]
        const float * k,          // [n_head * n_embd_head, n_tokens]
        const float * v,          // [n_head * n_embd_head, n_tokens]
        float       * scores_buf, // [n_head * n_tokens * n_tokens] temp
        int           n_tokens,
        int           n_embd_head,
        int           n_head,
        float         scale) {

    int h  = blockIdx.x;   // head index
    int t1 = blockIdx.y;   // query token position
    int tid = threadIdx.x;

    // Point to this head's score row for query t1
    float * scores = scores_buf + h * n_tokens * n_tokens + t1 * n_tokens;

    int head_stride = n_embd_head * n_tokens; // stride between heads in q/k/v

    // --- Step 1: compute score[t2] = sum_d Q_h[d][t1] * K_h[d][t2] ---
    for (int t2 = tid; t2 < n_tokens; t2 += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < n_embd_head; d++) {
            int q_idx = h * head_stride + d * n_tokens + t1;
            int k_idx = h * head_stride + d * n_tokens + t2;
            s += q[q_idx] * k[k_idx];
        }
        scores[t2] = s * scale;
    }
    __syncthreads();

    // --- Step 2: softmax (single thread) ---
    if (tid == 0) {
        float maxv = scores[0];
        for (int j = 1; j < n_tokens; j++) {
            if (scores[j] > maxv) maxv = scores[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < n_tokens; j++) {
            scores[j] = expf(scores[j] - maxv);
            sum += scores[j];
        }
        for (int j = 0; j < n_tokens; j++) {
            scores[j] /= sum;
        }
    }
    __syncthreads();

    // --- Step 3: attn_out_h[d][t1] = sum_t2 score[t2] * V_h[d][t2] ---
    for (int d = tid; d < n_embd_head; d += blockDim.x) {
        float val = 0.0f;
        int v_base = h * head_stride + d * n_tokens;
        for (int t2 = 0; t2 < n_tokens; t2++) {
            val += scores[t2] * v[v_base + t2];
        }
        out[v_base + t1] = val;
    }
}

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// llm_attn_forward
// ---------------------------------------------------------------------------

static inline void llm_attn_forward(
        float       * out,
        const float * hidden,
        const float * wq,
        const float * wk,
        const float * wv,
        const float * wo,
        int           n_tokens,
        int           n_embd,
        int           n_head,
        int           n_embd_head) {

    const int n_embd_q = n_head * n_embd_head;

#ifdef __CUDACC__

    // Temporary buffers on device (via std::vector with unified/managed memory)
    std::vector<float> q_buf(n_embd_q * n_tokens);
    std::vector<float> k_buf(n_embd_q * n_tokens);
    std::vector<float> v_buf(n_embd_q * n_tokens);
    std::vector<float> attn_buf(n_embd_q * n_tokens);
    std::vector<float> scores_buf(n_head * n_tokens * n_tokens);

    // Q projection
    {
        dim3 block(16, 16);
        dim3 grid((n_tokens + 15) / 16, (n_embd_q + 15) / 16);
        mul_mat<<<grid, block>>>(wq, hidden, q_buf.data(), n_embd_q, n_tokens, n_embd);
    }

    // K projection
    {
        dim3 block(16, 16);
        dim3 grid((n_tokens + 15) / 16, (n_embd_q + 15) / 16);
        mul_mat<<<grid, block>>>(wk, hidden, k_buf.data(), n_embd_q, n_tokens, n_embd);
    }

    // V projection
    {
        dim3 block(16, 16);
        dim3 grid((n_tokens + 15) / 16, (n_embd_q + 15) / 16);
        mul_mat<<<grid, block>>>(wv, hidden, v_buf.data(), n_embd_q, n_tokens, n_embd);
    }

    // Attention core: QK^T + softmax + V weighting
    {
        float scale = 1.0f / sqrtf((float)n_embd_head);
        dim3 block(256);
        dim3 grid(n_head, n_tokens);
        attn_core_kernel<<<grid, block>>>(
            attn_buf.data(), q_buf.data(), k_buf.data(), v_buf.data(),
            scores_buf.data(), n_tokens, n_embd_head, n_head, scale);
    }

    // Output projection: wo [n_embd, n_embd_q] x attn [n_embd_q, n_tokens] -> out [n_embd, n_tokens]
    {
        dim3 block(16, 16);
        dim3 grid((n_tokens + 15) / 16, (n_embd + 15) / 16);
        mul_mat<<<grid, block>>>(wo, attn_buf.data(), out, n_embd, n_tokens, n_embd_q);
    }

#else // CPU fallback

    std::vector<float> q_buf(n_embd_q * n_tokens);
    std::vector<float> k_buf(n_embd_q * n_tokens);
    std::vector<float> v_buf(n_embd_q * n_tokens);
    std::vector<float> attn_buf(n_embd_q * n_tokens, 0.0f);

    // Q, K, V projections
    mul_mat(wq, hidden, q_buf.data(), n_embd_q, n_tokens, n_embd);
    mul_mat(wk, hidden, k_buf.data(), n_embd_q, n_tokens, n_embd);
    mul_mat(wv, hidden, v_buf.data(), n_embd_q, n_tokens, n_embd);

    // Per-head attention
    const float scale = 1.0f / std::sqrt((float)n_embd_head);
    std::vector<float> scores(n_tokens * n_tokens);

    for (int h = 0; h < n_head; h++) {
        const int head_off = h * n_embd_head;

        // scores[t1][t2] = sum_d Q_h[d][t1] * K_h[d][t2]
        for (int t1 = 0; t1 < n_tokens; t1++) {
            for (int t2 = 0; t2 < n_tokens; t2++) {
                float s = 0.0f;
                for (int d = 0; d < n_embd_head; d++) {
                    s += q_buf[(head_off + d) * n_tokens + t1] *
                         k_buf[(head_off + d) * n_tokens + t2];
                }
                scores[t1 * n_tokens + t2] = s * scale;
            }
        }

        // softmax over each row t1
        for (int t1 = 0; t1 < n_tokens; t1++) {
            float * row = &scores[t1 * n_tokens];
            float maxv = row[0];
            for (int j = 1; j < n_tokens; j++) {
                if (row[j] > maxv) maxv = row[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < n_tokens; j++) {
                row[j] = std::exp(row[j] - maxv);
                sum += row[j];
            }
            for (int j = 0; j < n_tokens; j++) {
                row[j] /= sum;
            }
        }

        // attn_out_h[d][t1] = sum_t2 scores[t1][t2] * V_h[d][t2]
        for (int d = 0; d < n_embd_head; d++) {
            for (int t1 = 0; t1 < n_tokens; t1++) {
                float val = 0.0f;
                int v_base = (head_off + d) * n_tokens;
                for (int t2 = 0; t2 < n_tokens; t2++) {
                    val += scores[t1 * n_tokens + t2] * v_buf[v_base + t2];
                }
                attn_buf[v_base + t1] = val;
            }
        }
    }

    // Output projection
    mul_mat(wo, attn_buf.data(), out, n_embd, n_tokens, n_embd_q);

#endif
}

// ggml tensor wrapper
static inline void llm_attn_forward(
        struct ggml_tensor * out,
        const struct ggml_tensor * hidden,
        const struct ggml_tensor * wq,
        const struct ggml_tensor * wk,
        const struct ggml_tensor * wv,
        const struct ggml_tensor * wo,
        int                     n_tokens,
        int                     n_head,
        int                     n_embd_head) {
    llm_attn_forward(
        (float *)out->data,
        (const float *)hidden->data,
        (const float *)wq->data,
        (const float *)wk->data,
        (const float *)wv->data,
        (const float *)wo->data,
        n_tokens, (int)hidden->ne[0], n_head, n_embd_head);
}
