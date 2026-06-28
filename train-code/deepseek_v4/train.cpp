// DeepSeek V4 -- reference training recipe.
//
// Preserved beside the model code as GGUF metadata (dynllama.train_code.file.*).
// The inference host carries these keys but never compiles them; they document
// how the shipped weights were produced and let a downstream user re-train or
// fine-tune from the same recipe.
//
// V4-specific mechanics captured here:
//   - Muon optimizer for 2D weight matrices, AdamW for embeddings/norms/router
//   - Multi-Token Prediction (MTP) auxiliary loss for densified supervision
//   - aux-free MoE load balancing through a per-expert routing bias
//   - periodic Sinkhorn re-projection of the mHC residual factors (U, V)
//
// The model forward/backward (MLA + MoE + mHC + MTP) is assumed available via
// autograd; this file shows how the resulting gradients are consumed. main()
// runs a few synthetic steps so the recipe compiles and exercises every path.

#include "train_config.h"
#include "constraints.h"
#include "optimizer.h"
#include "loss.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace ds4train;

enum param_kind { PARAM_2D, PARAM_1D };   // PARAM_2D -> Muon, PARAM_1D -> AdamW

struct param {
    std::string        name;
    param_kind         kind  = PARAM_1D;
    int                rows  = 0, cols = 0;   // valid for PARAM_2D
    std::vector<float> w, g;                  // weights and gradients
    std::vector<float> m, v;                  // optimizer state
    bool               is_mhc_factor = false; // U/V need Sinkhorn re-projection
};

inline void step_param(param & p, const train_config & cfg, float lr_mul, int t) {
    if (p.kind == PARAM_2D) {
        muon_step(p.w, p.g, p.m, p.rows, p.cols,
                  cfg.lr * lr_mul, 0.95f, cfg.weight_decay);
    } else {
        adamw_step(p.w, p.g, p.m, p.v,
                   cfg.lr_adam * lr_mul, 0.9f, 0.95f, 1e-8f,
                   cfg.weight_decay, t);
    }
}

// Linear warmup then cosine decay to 10% of peak.
inline float lr_schedule(const train_config & cfg, int step) {
    if (step < cfg.warmup_steps)
        return (float) step / (float) cfg.warmup_steps;
    const float prog = (float) (step - cfg.warmup_steps) /
                       (float) std::max(1, cfg.total_steps - cfg.warmup_steps);
    return 0.1f + 0.9f * 0.5f * (1.f + std::cos(3.1415926f * prog));
}

// Global grad-norm clip across all params.
inline void clip_grads(std::vector<param> & params, float max_norm) {
    double sq = 0.0;
    for (auto & p : params) for (float gi : p.g) sq += (double) gi * gi;
    const float norm = (float) std::sqrt(sq);
    if (norm <= max_norm || norm == 0.f) return;
    const float s = max_norm / norm;
    for (auto & p : params) for (float & gi : p.g) gi *= s;
}

// One optimizer step. p.g is assumed filled by the backward pass over the
// combined loss (LM CE + MTP aux). mHC factors are re-projected onto the
// Sinkhorn manifold every cfg.mhc_resinkhorn_every steps.
inline void optimizer_step(std::vector<param> & params, std::vector<float> & moe_bias,
                           const std::vector<int> & expert_load,
                           const train_config & cfg, int step) {
    clip_grads(params, cfg.grad_clip);
    const float lr_mul = lr_schedule(cfg, step);
    for (auto & p : params) step_param(p, cfg, lr_mul, step + 1);

    moe_update_bias(moe_bias, expert_load, cfg.moe_bias_rate);

    if (cfg.mhc_resinkhorn_every > 0 && step % cfg.mhc_resinkhorn_every == 0) {
        // Factors are stored as adjacent PARAM_2D entries (..mhc_U, ..mhc_V).
        for (size_t i = 0; i + 1 < params.size(); ++i) {
            if (params[i].is_mhc_factor && params[i + 1].is_mhc_factor) {
                sinkhorn_normalize(params[i].w, params[i + 1].w,
                                   params[i].rows, params[i].cols);
            }
        }
    }
}

int main() {
    train_config cfg;
    cfg.total_steps  = 20;
    cfg.warmup_steps = 5;
    cfg.mhc_resinkhorn_every = 5;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.02f);

    std::vector<param> params;
    auto add2d = [&](const std::string & nm, int r, int c, bool mhc = false) {
        param p; p.name = nm; p.kind = PARAM_2D; p.rows = r; p.cols = c;
        p.w.resize((size_t) r * c); p.g.resize((size_t) r * c);
        p.m.assign((size_t) r * c, 0.f);
        for (auto & x : p.w) x = nd(rng);
        p.is_mhc_factor = mhc;
        params.push_back(std::move(p));
    };
    auto add1d = [&](const std::string & nm, int n) {
        param p; p.name = nm; p.kind = PARAM_1D;
        p.w.assign(n, 0.f); p.g.resize(n);
        p.m.assign(n, 0.f); p.v.assign(n, 0.f);
        params.push_back(std::move(p));
    };

    const int D = 64, NV = 128, NEXP = 8, RANK = 8, MTP = 1;
    add2d("blk.0.attn.wq", D, D);
    add2d("blk.0.ffn.w1",  D, D);
    add2d("blk.0.mhc_U",   D, RANK, true);
    add2d("blk.0.mhc_V",   D, RANK, true);
    add1d("blk.0.attn_norm", D);
    add1d("token_embd",      NV * D);

    std::vector<float> moe_bias(NEXP, 0.f);
    std::vector<float> logits(NV), dlogits(NV);
    std::vector<float> draft((size_t) MTP * NV), ddraft((size_t) MTP * NV);

    for (int step = 0; step < cfg.total_steps; ++step) {
        // forward (synthetic): main + draft logits
        for (auto & x : logits) x = nd(rng);
        for (auto & x : draft)  x = nd(rng);
        const int target  = step % NV;
        const int mtp_tgt[1] = { (step + 1) % NV };

        const float lm  = cross_entropy(logits.data(), NV, target, dlogits.data());
        const float aux = mtp_aux_loss(draft.data(), MTP, NV, mtp_tgt,
                                       cfg.mtp_weight, ddraft.data());

        // backprop dlogits/ddraft through the model -> p.g (omitted); synthetic
        // grads keep the optimizer path exercised.
        for (auto & p : params) for (auto & gi : p.g) gi = nd(rng);

        std::vector<int> load(NEXP);
        for (int e = 0; e < NEXP; ++e) load[e] = (step + e) % 5;

        optimizer_step(params, moe_bias, load, cfg, step);

        if (step % 5 == 0)
            std::printf("step %3d  lm=%.4f  mtp=%.4f  lr_mul=%.3f\n",
                        step, lm, aux, lr_schedule(cfg, step));
    }
    std::puts("training demo complete");
    return 0;
}
