//
// CUDA MoE FFN layer: router + top-K + per-expert FFN + weighted sum.
// Composes CUDA ops from ../ops/ into a single forward pass.
// GPU path (__CUDACC__): launches CUDA kernels with shared memory.
// CPU fallback (#else): plain loops with std::vector temps.
//
// Supports:
//   - Softmax or sigmoid gating
//   - Top-K expert selection (linear scan insertion)
//   - Per-expert up/gate activation + down projection
//   - Weight normalization
//   - SiLU / GELU / ReLU activation
//
// Weight layouts:
//   gate_inp: [n_embd, n_expert]        router weight
//   up_exps:  [n_ff,  n_embd, n_expert]  expert e at e * n_ff * n_embd
//   gate_exps:[n_ff,  n_embd, n_expert]  (can be NULL for no gating)
//   down_exps:[n_embd, n_ff,  n_expert]  expert e at e * n_embd * n_ff
//

#pragma once

#include "types.h"
#include <vector>
#include <algorithm>
#include <cmath>

#include "../ops/mul_mat.h"
#include "../ops/silu.h"
#include "../ops/gelu.h"
#include "../ops/relu.h"
#include "../ops/mul.h"

#ifdef __CUDACC__

// ---------------------------------------------------------------------------
// Zero a buffer on GPU
// ---------------------------------------------------------------------------
extern "C" __global__
void moe_zero_kernel(float * x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// Router: logits[expert][token] = sum_d gate_inp[d][expert] * hidden[d][token]
//   gate_inp [n_embd, n_expert], hidden [n_embd, n_tokens]
//   logits   [n_expert, n_tokens]
// ---------------------------------------------------------------------------
extern "C" __global__
void moe_router_kernel(
        float       * logits,
        const float * gate_inp,
        const float * hidden,
        int           n_expert,
        int           n_tokens,
        int           n_embd) {

    int e = blockIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (e < n_expert && t < n_tokens) {
        float sum = 0.0f;
        for (int d = 0; d < n_embd; d++) {
            sum += gate_inp[d * n_expert + e] * hidden[d * n_tokens + t];
        }
        logits[e * n_tokens + t] = sum;
    }
}

// ---------------------------------------------------------------------------
// Softmax over experts per token (in-place on probs [n_expert, n_tokens])
// ---------------------------------------------------------------------------
extern "C" __global__
void moe_softmax_kernel(float * probs, int n_expert, int n_tokens) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens) return;

    float maxv = probs[t];
    for (int e = 1; e < n_expert; e++) {
        float v = probs[e * n_tokens + t];
        if (v > maxv) maxv = v;
    }

    float sum = 0.0f;
    for (int e = 0; e < n_expert; e++) {
        float v = expf(probs[e * n_tokens + t] - maxv);
        probs[e * n_tokens + t] = v;
        sum += v;
    }

    for (int e = 0; e < n_expert; e++) {
        probs[e * n_tokens + t] /= sum;
    }
}

// ---------------------------------------------------------------------------
// Sigmoid in-place: x = 1 / (1 + exp(-x))
// ---------------------------------------------------------------------------
extern "C" __global__
void moe_sigmoid_kernel(float * x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

// ---------------------------------------------------------------------------
// Top-K per token: linear scan insertion into sorted list.
//   probs   [n_expert, n_tokens]
//   indices [n_tokens, n_used]
//   weights [n_tokens, n_used]
// ---------------------------------------------------------------------------
extern "C" __global__
void moe_topk_kernel(
        const float * probs,
        int         * indices,
        float       * weights,
        int           n_tokens,
        int           n_expert,
        int           n_used) {

    // Use thread-local arrays (n_used is typically small, e.g. 2-8)

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens) return;

    int   top_idx[64];
    float top_val[64];

    for (int k = 0; k < n_used; k++) {
        top_idx[k] = -1;
        top_val[k] = -__FLT_MAX__;
    }

    for (int e = 0; e < n_expert; e++) {
        float p = probs[e * n_tokens + t];
        for (int k = 0; k < n_used; k++) {
            if (p > top_val[k]) {
                for (int s = n_used - 1; s > k; s--) {
                    top_val[s] = top_val[s - 1];
                    top_idx[s] = top_idx[s - 1];
                }
                top_val[k] = p;
                top_idx[k] = e;
                break;
            }
        }
    }

    for (int k = 0; k < n_used; k++) {
        indices[t * n_used + k] = top_idx[k];
        weights[t * n_used + k] = top_val[k];
    }
}

// ---------------------------------------------------------------------------
// Normalize top-K weights per token: w_k /= max(sum(w), eps)
//   weights [n_tokens, n_used]
// ---------------------------------------------------------------------------
extern "C" __global__
void moe_normalize_weights_kernel(float * weights, int n_tokens, int n_used) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens) return;

    float sum = 0.0f;
    for (int k = 0; k < n_used; k++) {
        sum += weights[t * n_used + k];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;

    for (int k = 0; k < n_used; k++) {
        weights[t * n_used + k] /= sum;
    }
}

// ---------------------------------------------------------------------------
// Fused per-expert FFN + weighted accumulation.
//
// One block per (token, k_slot) pair.  Uses dynamic shared memory for the
// intermediate gated vector [n_ff].
//
// For each pair this computes the full expert FFN:
//   gated[i] = activation( gate_w[i] dot hidden ) * (up_w[i] dot hidden)
//   out[d][t] += w * sum_i down_w[d][i] * gated[i]
// ---------------------------------------------------------------------------
extern "C" __global__
void moe_expert_ffn_kernel(
        float       * out,
        const float * hidden,
        const float * up_exps,
        const float * gate_exps,
        const float * down_exps,
        const int   * selected,
        const float * weights,
        int           n_tokens,
        int           n_embd,
        int           n_ff,
        int           n_used,
        int           act_type) {

    int t = blockIdx.x;
    int k = blockIdx.y;
    if (t >= n_tokens || k >= n_used) return;

    extern __shared__ float gated[];  // [n_ff]

    int expert = selected[t * n_used + k];
    float w = weights[t * n_used + k];

    // skip near-zero weight to save compute
    if (w < 1e-10f) return;

    const float * up_w   = up_exps   + (size_t)expert * n_ff * n_embd;
    const float * gate_w = gate_exps ? (gate_exps + (size_t)expert * n_ff * n_embd) : NULL;
    const float * down_w = down_exps + (size_t)expert * n_embd * n_ff;

    const float * h_col = hidden + t;  // stride n_tokens between embedding dims

    // ---- Phase 1: compute gated[i] = activation(gate[i]) * up[i] ----
    for (int i = threadIdx.x; i < n_ff; i += blockDim.x) {
        float uv = 0.0f;
        for (int d = 0; d < n_embd; d++) {
            uv += up_w[i * n_embd + d] * h_col[d * n_tokens];
        }

        if (gate_w) {
            float gv = 0.0f;
            for (int d = 0; d < n_embd; d++) {
                gv += gate_w[i * n_embd + d] * h_col[d * n_tokens];
            }
            // activate gate
            if (act_type == 0) {
                gv = gv / (1.0f + expf(-gv));
            } else if (act_type == 1) {
                float x = gv;
                gv = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
            } else {
                gv = gv > 0.0f ? gv : 0.0f;
            }
            uv *= gv;
        } else {
            // no gate: activate up directly
            if (act_type == 0) {
                uv = uv / (1.0f + expf(-uv));
            } else if (act_type == 1) {
                float x = uv;
                uv = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
            } else {
                uv = uv > 0.0f ? uv : 0.0f;
            }
        }

        gated[i] = uv;
    }
    __syncthreads();

    // ---- Phase 2: down project and accumulate into out ----
    for (int d = threadIdx.x; d < n_embd; d += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < n_ff; i++) {
            sum += down_w[(size_t)d * n_ff + i] * gated[i];
        }
        out[(size_t)d * n_tokens + t] += w * sum;
    }
}

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// llm_moe_ffn_forward — full MoE FFN forward pass
// ---------------------------------------------------------------------------

static inline void llm_moe_ffn_forward(
        float       * out,                        // output  [n_embd, n_tokens]
        const float * hidden,                     // input   [n_embd, n_tokens]
        const float * gate_inp,                   // router  [n_embd, n_expert]
        const float * up_exps,                    // up weights [n_ff, n_embd, n_expert]
        const float * gate_exps,                  // gate weights [n_ff, n_embd, n_expert] (can be NULL)
        const float * down_exps,                  // down weights [n_embd, n_ff, n_expert]
        int           n_tokens,
        int           n_embd,
        int           n_ff,
        int           n_expert,
        int           n_expert_used,
        int           act_type,                   // 0=silu, 1=gelu, 2=relu
        int           gating_type,                // 0=softmax, 1=sigmoid
        bool          norm_weights) {

    const int total_embd = n_embd * n_tokens;

#ifdef __CUDACC__

    // ---- Temp buffers (unified/managed memory) ----
    std::vector<float>  logits_buf(n_expert * n_tokens);
    std::vector<int>    selected_buf(n_expert_used * n_tokens);
    std::vector<float>  weights_buf(n_expert_used * n_tokens);

    // Step 1: Router logits
    {
        dim3 block(1, 256);
        dim3 grid(n_expert, (n_tokens + 255) / 256);
        moe_router_kernel<<<grid, block>>>(
            logits_buf.data(), gate_inp, hidden,
            n_expert, n_tokens, n_embd);
    }

    // Step 2: Gating
    if (gating_type == 0) {
        dim3 block(256);
        dim3 grid((n_tokens + 255) / 256);
        moe_softmax_kernel<<<grid, block>>>(
            logits_buf.data(), n_expert, n_tokens);
    } else {
        int total = n_expert * n_tokens;
        dim3 block(256);
        dim3 grid((total + 255) / 256);
        moe_sigmoid_kernel<<<grid, block>>>(
            logits_buf.data(), total);
    }

    // Step 3: Top-K selection
    {
        dim3 block(256);
        dim3 grid((n_tokens + 255) / 256);
        moe_topk_kernel<<<grid, block>>>(
            logits_buf.data(),
            selected_buf.data(),
            weights_buf.data(),
            n_tokens, n_expert, n_expert_used);
    }

    // Step 4: Normalize weights (optional)
    if (norm_weights) {
        dim3 block(256);
        dim3 grid((n_tokens + 255) / 256);
        moe_normalize_weights_kernel<<<grid, block>>>(
            weights_buf.data(), n_tokens, n_expert_used);
    }

    // Step 5: Zero output
    {
        dim3 block(256);
        dim3 grid((total_embd + 255) / 256);
        moe_zero_kernel<<<grid, block>>>(out, total_embd);
    }

    // Step 6: Fused per-expert FFN + accumulate (one block per token,k pair)
    {
        const int block_sz = 256;
        dim3 block(block_sz);
        dim3 grid(n_tokens, n_expert_used);
        size_t smem = (size_t)n_ff * sizeof(float);
        moe_expert_ffn_kernel<<<grid, block, smem>>>(
            out, hidden,
            up_exps, gate_exps, down_exps,
            selected_buf.data(), weights_buf.data(),
            n_tokens, n_embd, n_ff, n_expert_used, act_type);
    }

#else // ---- CPU fallback ----

    std::vector<float>  logits_buf(n_expert * n_tokens);
    std::vector<int>    selected_buf(n_expert_used * n_tokens);
    std::vector<float>  weights_buf(n_expert_used * n_tokens);
    std::vector<float>  up_buf(n_ff);
    std::vector<float>  gate_buf(n_ff);
    std::vector<float>  expert_out_buf(n_embd);

    // Step 1: Router logits
    for (int e = 0; e < n_expert; e++) {
        for (int t = 0; t < n_tokens; t++) {
            float sum = 0.0f;
            for (int d = 0; d < n_embd; d++) {
                sum += gate_inp[d * n_expert + e] * hidden[d * n_tokens + t];
            }
            logits_buf[e * n_tokens + t] = sum;
        }
    }

    // Step 2: Gating
    if (gating_type == 0) {
        for (int t = 0; t < n_tokens; t++) {
            float maxv = logits_buf[t];
            for (int e = 1; e < n_expert; e++) {
                float v = logits_buf[e * n_tokens + t];
                if (v > maxv) maxv = v;
            }
            float sum = 0.0f;
            for (int e = 0; e < n_expert; e++) {
                float v = std::exp(logits_buf[e * n_tokens + t] - maxv);
                logits_buf[e * n_tokens + t] = v;
                sum += v;
            }
            for (int e = 0; e < n_expert; e++) {
                logits_buf[e * n_tokens + t] /= sum;
            }
        }
    } else {
        for (int i = 0; i < n_expert * n_tokens; i++) {
            logits_buf[i] = 1.0f / (1.0f + std::exp(-logits_buf[i]));
        }
    }

    // Step 3: Top-K per token
    {
        std::vector<std::pair<float, int>> tmp(n_expert);
        for (int t = 0; t < n_tokens; t++) {
            for (int e = 0; e < n_expert; e++) {
                tmp[e] = {logits_buf[e * n_tokens + t], e};
            }
            std::sort(tmp.begin(), tmp.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });
            for (int k = 0; k < n_expert_used; k++) {
                selected_buf[t * n_expert_used + k] = tmp[k].second;
                weights_buf[t * n_expert_used + k] = tmp[k].first;
            }
        }
    }

    // Step 4: Normalize weights
    if (norm_weights) {
        for (int t = 0; t < n_tokens; t++) {
            float sum = 0.0f;
            for (int k = 0; k < n_expert_used; k++) {
                sum += weights_buf[t * n_expert_used + k];
            }
            if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
            for (int k = 0; k < n_expert_used; k++) {
                weights_buf[t * n_expert_used + k] /= sum;
            }
        }
    }

    // Step 5: Zero output
    std::fill(out, out + total_embd, 0.0f);

    // Step 6: Per-expert FFN + weighted sum (per token, per k-slot)
    for (int t = 0; t < n_tokens; t++) {
        for (int k = 0; k < n_expert_used; k++) {
            int expert = selected_buf[t * n_expert_used + k];
            float w = weights_buf[t * n_expert_used + k];

            if (w < 1e-10f) continue;

            const float * up_w   = up_exps   + (size_t)expert * n_ff * n_embd;
            const float * gate_w = gate_exps ? (gate_exps + (size_t)expert * n_ff * n_embd) : NULL;
            const float * down_w = down_exps + (size_t)expert * n_embd * n_ff;

            // Up projection + gate projection + activation + gated = up*act(gate)
            for (int i = 0; i < n_ff; i++) {
                float uv = 0.0f;
                for (int d = 0; d < n_embd; d++) {
                    uv += up_w[(size_t)i * n_embd + d] * hidden[(size_t)d * n_tokens + t];
                }

                if (gate_w) {
                    float gv = 0.0f;
                    for (int d = 0; d < n_embd; d++) {
                        gv += gate_w[(size_t)i * n_embd + d] * hidden[(size_t)d * n_tokens + t];
                    }
                    // activate gate
                    if (act_type == 0) {
                        gv = gv / (1.0f + std::exp(-gv));
                    } else if (act_type == 1) {
                        float x = gv;
                        gv = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
                    } else {
                        gv = gv > 0.0f ? gv : 0.0f;
                    }
                    uv *= gv;
                } else {
                    if (act_type == 0) {
                        uv = uv / (1.0f + std::exp(-uv));
                    } else if (act_type == 1) {
                        float x = uv;
                        uv = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
                    } else {
                        uv = uv > 0.0f ? uv : 0.0f;
                    }
                }
                up_buf[i] = uv;
            }

            // Down projection
            for (int d = 0; d < n_embd; d++) {
                float sum = 0.0f;
                for (int i = 0; i < n_ff; i++) {
                    sum += down_w[(size_t)d * n_ff + i] * up_buf[i];
                }
                expert_out_buf[d] = sum;
            }

            // Accumulate weighted output
            for (int d = 0; d < n_embd; d++) {
                out[(size_t)d * n_tokens + t] += w * expert_out_buf[d];
            }
        }
    }

#endif
}

// ---------------------------------------------------------------------------
// llm_moe_ffn_forward_simple — simplified wrapper with softmax gating
// ---------------------------------------------------------------------------

static inline void llm_moe_ffn_forward_simple(
        float       * out,
        const float * hidden,
        const float * gate_inp,
        const float * up_exps,
        const float * gate_exps,
        const float * down_exps,
        int           n_tokens,
        int           n_embd,
        int           n_ff,
        int           n_expert,
        int           n_expert_used,
        int           act_type,
        bool          norm_weights) {

    llm_moe_ffn_forward(
        out, hidden, gate_inp, up_exps, gate_exps, down_exps,
        n_tokens, n_embd, n_ff,
        n_expert, n_expert_used,
        act_type,
        0,        // gating_type = softmax
        norm_weights);
}

// ggml tensor wrapper
static inline void llm_moe_ffn_forward(
        struct ggml_tensor * out,
        const struct ggml_tensor * hidden,
        const struct ggml_tensor * gate_inp,
        const struct ggml_tensor * up_exps,
        const struct ggml_tensor * gate_exps,
        const struct ggml_tensor * down_exps,
        int                     n_tokens,
        int                     n_ff,
        int                     n_expert,
        int                     n_expert_used,
        int                     act_type,
        int                     gating_type,
        bool                    norm_weights) {
    llm_moe_ffn_forward(
        (float *)out->data,
        (const float *)hidden->data,
        (const float *)gate_inp->data,
        (const float *)up_exps->data,
        gate_exps ? (const float *)gate_exps->data : NULL,
        (const float *)down_exps->data,
        n_tokens, (int)hidden->ne[0], n_ff,
        n_expert, n_expert_used,
        act_type, gating_type, norm_weights);
}

// ggml tensor wrapper
static inline void llm_moe_ffn_forward_simple(
        struct ggml_tensor * out,
        const struct ggml_tensor * hidden,
        const struct ggml_tensor * gate_inp,
        const struct ggml_tensor * up_exps,
        const struct ggml_tensor * gate_exps,
        const struct ggml_tensor * down_exps,
        int                     n_tokens,
        int                     n_ff,
        int                     n_expert,
        int                     n_expert_used,
        int                     act_type,
        bool                    norm_weights) {
    llm_moe_ffn_forward_simple(
        (float *)out->data,
        (const float *)hidden->data,
        (const float *)gate_inp->data,
        (const float *)up_exps->data,
        gate_exps ? (const float *)gate_exps->data : NULL,
        (const float *)down_exps->data,
        n_tokens, (int)hidden->ne[0], n_ff,
        n_expert, n_expert_used,
        act_type, norm_weights);
}
