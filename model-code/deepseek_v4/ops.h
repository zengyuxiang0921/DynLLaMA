#pragma once
// DeepSeek V4 ops (core architecture), weights loaded from the GGUF.
//
// Each decoder layer is a dynllama_layer built by name prefix "blk.{i}" from the
// host tensor accessor (dynllama_tensors); the forward reads weights by key via
// weight_ptr(). Weights come from the PyTorch model (model.py / train.py)
// converted to GGUF, whose tensor names are already in blk.{i}.* form.
//
// Architecture: Hyper-Connections residual stream; V4 attention (low-rank Q,
// MQA latent KV, attention sink, grouped low-rank O); MoE (sqrtsoftplus gate +
// bias + route_scale, shared expert, swiglu_limit). FP8/FP4, KV compression and
// MTP are omitted. Demo dimensions below must match the trained model's config
// (ModelArgs.tiny); real 4B-class values are in the comments.

#include "dynllama_abi.h"
#include "ops/add.h"
#include "ops/swiglu.h"
#include "ops/hc.h"
#include "layers/dynllama_layer.h"   // dynllama_layer, dynllama_make_layer (-I dynllama-headers/cpu)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace ds4 {

// ---- architecture constants (must match the trained config) ------------------
constexpr int   N_VOCAB       = 1024;   // real: 129280
constexpr int   N_LAYERS      = 4;      // real: 43
constexpr int   D_MODEL       = 256;    // real: 4096
constexpr int   N_HEADS       = 8;      // real: 64
constexpr int   HEAD_DIM      = 64;     // real: 512
constexpr int   ROPE_HEAD_DIM = 16;     // real: 64
constexpr int   NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM;
constexpr int   Q_LORA_RANK   = 64;     // real: 1024
constexpr int   O_GROUPS      = 4;      // real: 8
constexpr int   O_LORA_RANK   = 64;     // real: 1024
constexpr int   ATT_DIM       = N_HEADS * HEAD_DIM;
constexpr int   O_GROUP_IN    = ATT_DIM / O_GROUPS;
constexpr int   N_ROUTED      = 8;      // real: 256
constexpr int   N_ACTIVE      = 2;      // real: 6
constexpr int   MOE_INTER     = 128;    // real: 2048
constexpr float ROUTE_SCALE   = 1.5f;
constexpr float SWIGLU_LIMIT  = 10.0f;
constexpr float RMS_EPS       = 1e-6f;
constexpr float ROPE_BASE     = 10000.0f;
constexpr int   MAX_SEQ       = 512;

// ---- runtime config from GGUF metadata --------------------------------------
struct runtime_config {
    float rope_base = ROPE_BASE;
    float rms_eps   = RMS_EPS;
    int   n_active  = N_ACTIVE;
};
inline runtime_config g_cfg;

inline void load_config(const dynllama_meta * meta) {
    if (!meta) return;
    g_cfg.rope_base = (float) meta->get_float(meta->obj, "deepseek2.rope.freq_base", g_cfg.rope_base);
    g_cfg.rms_eps   = (float) meta->get_float(meta->obj,
                          "deepseek2.attention.layer_norm_rms_epsilon", g_cfg.rms_eps);
    int64_t na = meta->get_int(meta->obj, "deepseek2.expert_used_count", g_cfg.n_active);
    if (na > N_ROUTED) na = N_ROUTED;
    if (na < 1)        na = 1;
    g_cfg.n_active = (int) na;
}

// ---- model state: per-layer dynllama_layer + global tensors ------------------
struct kv_layer {
    std::vector<float> kv_cache;   // [MAX_SEQ x HEAD_DIM]  (MQA shared latent KV)
};

struct model_state {
    const float * tok_embd      = nullptr;   // embed.weight       [N_VOCAB x D_MODEL]
    const float * norm          = nullptr;   // norm.weight        [D_MODEL]
    const float * lm_head       = nullptr;   // lm_head.weight     [N_VOCAB x D_MODEL]
    const float * hc_head_fn    = nullptr;   // head_reduce.fn     [HC_MULT x HC_MULT*D_MODEL]
    const float * hc_head_base  = nullptr;   // head_reduce.base   [HC_MULT]
    const float * hc_head_scale = nullptr;   // head_reduce.scale  [1]
    std::vector<dynllama::dynllama_layer> layers;
    std::vector<kv_layer> kv;
};

inline model_state * g_model = nullptr;

// Build the model from the host tensor accessor. Each layer borrows the tensors
// under "blk.{i}". Returns false if required globals are missing.
inline bool load_weights(const dynllama_tensors * T) {
    if (g_model) return true;
    if (!T || !T->get) { std::fprintf(stderr, "[ds4] no tensor accessor\n"); return false; }

    g_model = new model_state;
    auto & m = *g_model;
    m.tok_embd      = T->get(T->obj, "embed.weight",       nullptr);
    m.norm          = T->get(T->obj, "norm.weight",        nullptr);
    m.lm_head       = T->get(T->obj, "lm_head.weight",     nullptr);
    m.hc_head_fn    = T->get(T->obj, "head_reduce.fn",     nullptr);
    m.hc_head_base  = T->get(T->obj, "head_reduce.base",   nullptr);
    m.hc_head_scale = T->get(T->obj, "head_reduce.scale",  nullptr);

    m.layers.reserve(N_LAYERS);
    m.kv.resize(N_LAYERS);
    for (int i = 0; i < N_LAYERS; ++i) {
        m.layers.push_back(dynllama::dynllama_make_layer("blk." + std::to_string(i), T));
        m.kv[i].kv_cache.assign((size_t) MAX_SEQ * HEAD_DIM, 0.f);
    }
    if (!m.tok_embd || !m.norm || !m.lm_head) {
        std::fprintf(stderr, "[ds4] missing global tensors (embed/norm/lm_head)\n");
        return false;
    }
    return true;
}

// ---- primitive ops -----------------------------------------------------------
inline void rms_norm(float * x, const float * w, int d, float eps) {
    float ss = 0.f;
    for (int i = 0; i < d; ++i) ss += x[i] * x[i];
    float rms_inv = 1.f / std::sqrt(ss / d + eps);
    for (int i = 0; i < d; ++i) x[i] = w[i] * x[i] * rms_inv;
}

inline void matmul(const float * x, const float * W, float * y, int in_dim, int out_dim) {
    for (int m = 0; m < out_dim; ++m) {
        float s = 0.f;
        const float * row = W + (size_t) m * in_dim;
        for (int k = 0; k < in_dim; ++k) s += row[k] * x[k];
        y[m] = s;
    }
}

inline void rope(float * x, int pos, int rope_dim, float base, bool inverse = false) {
    for (int i = 0; i < rope_dim / 2; ++i) {
        float theta = (float) pos / std::pow(base, 2.f * i / rope_dim);
        float c = std::cos(theta), s = std::sin(theta);
        if (inverse) s = -s;
        float x0 = x[2*i], x1 = x[2*i+1];
        x[2*i]   = x0*c - x1*s;
        x[2*i+1] = x0*s + x1*c;
    }
}

inline float sqrtsoftplus(float x) {
    float sp = x > 20.f ? x : std::log1p(std::exp(x));
    return std::sqrt(sp);
}

inline void topk(const float * scores, int n, int k, int * out) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [scores](int a, int b){ return scores[a] > scores[b]; });
    for (int i = 0; i < k; ++i) out[i] = idx[i];
}

// add / swiglu custom ops (JIT from GGUF or builtin fallback)
inline void compile_ops(const dynllama_meta * meta, const dynllama_op_table * host_ops) {
    g_add    = reinterpret_cast<binop_fn>(dynllama_find_op(host_ops, "add"));
    g_swiglu = reinterpret_cast<binop_fn>(dynllama_find_op(host_ops, "swiglu"));
    if (!g_add)    compile_add(meta);
    if (!g_swiglu) compile_swiglu(meta);
}

// ---- V4 attention ------------------------------------------------------------
inline void attention_forward(const dynllama::dynllama_layer & L, const float * x,
                              int pos, kv_layer & kvc, float * out) {
    const float * wq_a   = L.weight_ptr("attn.wq_a.weight");
    const float * q_normw= L.weight_ptr("attn.q_norm.weight");
    const float * wq_b   = L.weight_ptr("attn.wq_b.weight");
    const float * wkv    = L.weight_ptr("attn.wkv.weight");
    const float * kv_normw=L.weight_ptr("attn.kv_norm.weight");
    const float * sink   = L.weight_ptr("attn.attn_sink");
    const float * wo_a   = L.weight_ptr("attn.wo_a");
    const float * wo_b   = L.weight_ptr("attn.wo_b.weight");

    std::vector<float> q_lora(Q_LORA_RANK);
    matmul(x, wq_a, q_lora.data(), D_MODEL, Q_LORA_RANK);
    rms_norm(q_lora.data(), q_normw, Q_LORA_RANK, g_cfg.rms_eps);
    std::vector<float> q(ATT_DIM);
    matmul(q_lora.data(), wq_b, q.data(), Q_LORA_RANK, ATT_DIM);
    for (int h = 0; h < N_HEADS; ++h) {
        float * qh = q.data() + h * HEAD_DIM;
        float ss = 0.f;
        for (int d = 0; d < HEAD_DIM; ++d) ss += qh[d] * qh[d];
        float rinv = 1.f / std::sqrt(ss / HEAD_DIM + g_cfg.rms_eps);
        for (int d = 0; d < HEAD_DIM; ++d) qh[d] *= rinv;
        rope(qh + NOPE_HEAD_DIM, pos, ROPE_HEAD_DIM, g_cfg.rope_base);
    }

    float kv[HEAD_DIM];
    matmul(x, wkv, kv, D_MODEL, HEAD_DIM);
    rms_norm(kv, kv_normw, HEAD_DIM, g_cfg.rms_eps);
    rope(kv + NOPE_HEAD_DIM, pos, ROPE_HEAD_DIM, g_cfg.rope_base);
    std::memcpy(kvc.kv_cache.data() + (size_t) pos * HEAD_DIM, kv, HEAD_DIM * sizeof(float));

    const int seq = pos + 1;
    const float scale = 1.f / std::sqrt((float) HEAD_DIM);
    std::vector<float> o(ATT_DIM, 0.f);
    std::vector<float> sc(seq);
    for (int h = 0; h < N_HEADS; ++h) {
        const float * qh = q.data() + h * HEAD_DIM;
        for (int t = 0; t < seq; ++t) {
            const float * kt = kvc.kv_cache.data() + (size_t) t * HEAD_DIM;
            float s = 0.f;
            for (int d = 0; d < HEAD_DIM; ++d) s += qh[d] * kt[d];
            sc[t] = s * scale;
        }
        float mx = sink[h];
        for (int t = 0; t < seq; ++t) mx = std::max(mx, sc[t]);
        float sm = std::exp(sink[h] - mx);
        for (int t = 0; t < seq; ++t) { sc[t] = std::exp(sc[t] - mx); sm += sc[t]; }
        float * oh = o.data() + h * HEAD_DIM;
        for (int t = 0; t < seq; ++t) {
            float a = sc[t] / sm;
            const float * kt = kvc.kv_cache.data() + (size_t) t * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) oh[d] += a * kt[d];
        }
        rope(oh + NOPE_HEAD_DIM, pos, ROPE_HEAD_DIM, g_cfg.rope_base, true);
    }

    std::vector<float> og(O_GROUPS * O_LORA_RANK);
    for (int g = 0; g < O_GROUPS; ++g) {
        const float * o_in = o.data() + g * O_GROUP_IN;
        const float * wa   = wo_a + (size_t) g * O_LORA_RANK * O_GROUP_IN;
        float * dst        = og.data() + g * O_LORA_RANK;
        for (int r = 0; r < O_LORA_RANK; ++r) {
            const float * row = wa + (size_t) r * O_GROUP_IN;
            float s = 0.f;
            for (int d = 0; d < O_GROUP_IN; ++d) s += row[d] * o_in[d];
            dst[r] = s;
        }
    }
    matmul(og.data(), wo_b, out, O_GROUPS * O_LORA_RANK, D_MODEL);
}

// ---- Mixture-of-Experts FFN --------------------------------------------------
inline void expert_ffn(const float * x,
                       const float * gate_w, const float * up_w, const float * down_w,
                       float * out) {
    std::vector<float> gate(MOE_INTER), up(MOE_INTER), act(MOE_INTER);
    matmul(x, gate_w, gate.data(), D_MODEL, MOE_INTER);
    matmul(x, up_w,   up.data(),   D_MODEL, MOE_INTER);
    for (int i = 0; i < MOE_INTER; ++i) {
        gate[i] = std::min(gate[i], SWIGLU_LIMIT);
        up[i]   = std::min(std::max(up[i], -SWIGLU_LIMIT), SWIGLU_LIMIT);
    }
    op_swiglu(gate.data(), up.data(), act.data(), MOE_INTER);
    matmul(act.data(), down_w, out, MOE_INTER, D_MODEL);
}

inline void moe_forward(const dynllama::dynllama_layer & L, const float * x, float * delta) {
    std::fill(delta, delta + D_MODEL, 0.f);

    const float * router    = L.weight_ptr("ffn.gate.weight");
    const float * gate_bias = L.weight_ptr("ffn.gate_bias");

    float raw[N_ROUTED], orig[N_ROUTED], biased[N_ROUTED];
    matmul(x, router, raw, D_MODEL, N_ROUTED);
    for (int e = 0; e < N_ROUTED; ++e) {
        orig[e]   = sqrtsoftplus(raw[e]);
        biased[e] = orig[e] + (gate_bias ? gate_bias[e] : 0.f);
    }
    const int k = g_cfg.n_active;
    int sel[N_ROUTED];
    topk(biased, N_ROUTED, k, sel);
    float w_sum = 0.f;
    for (int i = 0; i < k; ++i) w_sum += orig[sel[i]];

    std::vector<float> ex_out(D_MODEL);
    for (int i = 0; i < k; ++i) {
        int e = sel[i];
        float w = (orig[e] / (w_sum + 1e-9f)) * ROUTE_SCALE;
        const std::string p = "ffn.experts." + std::to_string(e) + ".";
        expert_ffn(x, L.weight_ptr(p + "w1.weight"), L.weight_ptr(p + "w3.weight"),
                   L.weight_ptr(p + "w2.weight"), ex_out.data());
        for (int d = 0; d < D_MODEL; ++d) delta[d] += w * ex_out[d];
    }

    expert_ffn(x, L.weight_ptr("ffn.shared.w1.weight"), L.weight_ptr("ffn.shared.w3.weight"),
               L.weight_ptr("ffn.shared.w2.weight"), ex_out.data());
    op_add(delta, ex_out.data(), delta, D_MODEL);
}

} // namespace ds4
