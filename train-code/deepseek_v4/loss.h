#pragma once
// DeepSeek V4 training losses.
//   - language-model cross-entropy (with softmax gradient)
//   - MTP auxiliary loss: densified next-k-token supervision
//   - aux-free MoE load balancing via per-expert routing bias

#include <cmath>
#include <vector>

namespace ds4train {

// Cross-entropy for one target token. Writes the softmax gradient (p - onehot)
// into dlogits and returns the loss.
inline float cross_entropy(const float * logits, int n_vocab, int target,
                           float * dlogits) {
    float mx = logits[0];
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > mx) mx = logits[i];
    float sm = 0.f;
    for (int i = 0; i < n_vocab; ++i) sm += std::exp(logits[i] - mx);
    const float logZ = mx + std::log(sm);
    for (int i = 0; i < n_vocab; ++i) {
        const float p = std::exp(logits[i] - logZ);
        dlogits[i] = p - (i == target ? 1.f : 0.f);
    }
    return logZ - logits[target];
}

// Mean cross-entropy over the MTP draft depths, scaled by weight. dlogits holds
// the [depth x n_vocab] gradient block for the draft heads.
inline float mtp_aux_loss(const float * draft_logits, int depth, int n_vocab,
                          const int * targets, float weight, float * dlogits) {
    if (depth <= 0) return 0.f;
    float total = 0.f;
    for (int d = 0; d < depth; ++d) {
        const float * lg = draft_logits + (size_t) d * n_vocab;
        float *       dg = dlogits      + (size_t) d * n_vocab;
        total += cross_entropy(lg, n_vocab, targets[d], dg);
        const float s = weight / depth;
        for (int i = 0; i < n_vocab; ++i) dg[i] *= s;
    }
    return weight * total / depth;
}

// aux-free load balancing (DeepSeek V3/V4): no extra loss term, instead nudge a
// per-expert routing bias toward the mean load so routing self-balances.
inline void moe_update_bias(std::vector<float> & bias, const std::vector<int> & load,
                            float rate) {
    if (load.empty()) return;
    long total = 0;
    for (int c : load) total += c;
    const float avg = (float) total / load.size();
    for (size_t e = 0; e < bias.size() && e < load.size(); ++e) {
        const float err = avg - (float) load[e];   // under-loaded -> raise bias
        bias[e] += rate * (err > 0.f ? 1.f : (err < 0.f ? -1.f : 0.f));
    }
}

} // namespace ds4train
