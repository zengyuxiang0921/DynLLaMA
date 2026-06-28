// DeepSeek V4 -- DynLLaMA model code. Weights loaded from the GGUF.
// Entry point: dynllama_main
//
// Architecture (core port of deepseek-v4-example-code/model.py):
//   Hyper-Connections residual stream (HC_MULT parallel copies, Sinkhorn mixing)
//   V4 attention: low-rank Q, MQA latent KV, learned attention sink,
//                 grouped low-rank output projection
//   Mixture-of-Experts FFN (sqrtsoftplus gate + bias + route_scale, shared expert)
//   RMSNorm pre-norm, SwiGLU activation with swiglu_limit
//
// Weights come from the PyTorch model (model.py / train.py) converted to GGUF.
// Each decoder layer is a dynllama_layer built by name prefix "blk.{i}" from the
// host tensor accessor; the forward reads weights by key. Elementwise ops
// (add / swiglu) are JIT custom ops (ops/add.h, ops/swiglu.h).

#include "dynllama_abi.h"
#include "ops.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ds4;

// One forward pass for a single token at absolute position `pos`.
static void forward_token(model_state & m, int tok_id, int pos, float * logits_out) {
    if (tok_id < 0 || tok_id >= N_VOCAB) {
        std::fill(logits_out, logits_out + N_VOCAB, 0.f);
        return;
    }

    std::vector<float> H((size_t) HC_MULT * D_MODEL);
    for (int c = 0; c < HC_MULT; ++c)
        std::memcpy(H.data() + (size_t) c * D_MODEL,
                    m.tok_embd + (size_t) tok_id * D_MODEL,
                    D_MODEL * sizeof(float));

    std::vector<float> residual((size_t) HC_MULT * D_MODEL);
    std::vector<float> y(D_MODEL), sub(D_MODEL);
    float post[HC_MULT], comb[HC_MULT * HC_MULT];

    for (int l = 0; l < N_LAYERS; ++l) {
        const dynllama::dynllama_layer & lw = m.layers[l];

        // attention sub-block
        residual = H;
        hc_pre(H.data(), lw.weight_ptr("hc_attn.fn"), lw.weight_ptr("hc_attn.scale"),
               lw.weight_ptr("hc_attn.base"), D_MODEL, g_cfg.rms_eps, y.data(), post, comb);
        rms_norm(y.data(), lw.weight_ptr("attn_norm.weight"), D_MODEL, g_cfg.rms_eps);
        attention_forward(lw, y.data(), pos, m.kv[l], sub.data());
        hc_post(sub.data(), residual.data(), post, comb, D_MODEL, H.data());

        // FFN (MoE) sub-block
        residual = H;
        hc_pre(H.data(), lw.weight_ptr("hc_ffn.fn"), lw.weight_ptr("hc_ffn.scale"),
               lw.weight_ptr("hc_ffn.base"), D_MODEL, g_cfg.rms_eps, y.data(), post, comb);
        rms_norm(y.data(), lw.weight_ptr("ffn_norm.weight"), D_MODEL, g_cfg.rms_eps);
        moe_forward(lw, y.data(), sub.data());
        hc_post(sub.data(), residual.data(), post, comb, D_MODEL, H.data());
    }

    hc_head_reduce(H.data(), m.hc_head_fn, m.hc_head_scale, m.hc_head_base,
                   D_MODEL, g_cfg.rms_eps, y.data());
    rms_norm(y.data(), m.norm, D_MODEL, g_cfg.rms_eps);
    matmul(y.data(), m.lm_head, logits_out, D_MODEL, N_VOCAB);
}

extern "C" void dynllama_main(dynllama_run_ctx * c) {
    compile_ops(c->meta, c->ops);   // compile add/swiglu from GGUF or host table
    load_config(c->meta);
    if (!load_weights(c->tensors)) {
        std::fprintf(stderr, "[ds4] weight load failed; aborting forward\n");
        return;
    }
    model_state & m = *g_model;

    if (c->meta) {
        const char * arch = c->meta->get_str(c->meta->obj, "general.architecture", "?");
        std::fprintf(stderr, "[ds4] gguf architecture=%s  rope_base=%.1f rms_eps=%g n_active=%d\n",
                     arch, g_cfg.rope_base, g_cfg.rms_eps, g_cfg.n_active);
    }

    const int host_vocab = c->n_vocab;
    const int out_vocab  = (N_VOCAB < host_vocab) ? N_VOCAB : host_vocab;

    std::vector<float> tok_logits(N_VOCAB);
    for (int t = 0; t < c->n_tokens; ++t) {
        const int pos = c->n_past + t;
        if (pos >= MAX_SEQ) {
            std::fprintf(stderr, "[ds4] position %d exceeds MAX_SEQ=%d\n", pos, MAX_SEQ);
            return;
        }
        forward_token(m, c->tokens[t], pos, tok_logits.data());

        float * dst = c->logits + (size_t) t * host_vocab;
        std::fill(dst, dst + host_vocab, 0.f);
        std::memcpy(dst, tok_logits.data(), out_vocab * sizeof(float));
    }
}
