#pragma once
// Hyper-Connections (HC) -- DeepSeek V4.
//
// Replaces the plain residual stream. The hidden state is kept as HC_MULT
// parallel copies through the whole network. Each sub-block (attn / ffn):
//   hc_pre : reduce HC_MULT copies -> 1 via learned weights (pre), and produce
//            per-copy expansion weights (post) and a Sinkhorn-balanced mixing
//            matrix (comb).
//   <sublayer runs on the single reduced state>
//   hc_post: expand 1 -> HC_MULT copies:
//            H'[j] = post[j] * sublayer_out + sum_i comb[i,j] * H_residual[i]
//
// Reference: Block.hc_pre / hc_post and kernel.hc_split_sinkhorn.

#include <algorithm>
#include <cmath>

namespace ds4 {

constexpr int   HC_MULT          = 4;        // parallel residual copies
constexpr int   HC_SINKHORN_ITERS= 20;
constexpr float HC_EPS           = 1e-6f;
constexpr int   MIX_HC           = (2 + HC_MULT) * HC_MULT;  // pre + post + comb

// Split the per-token mix vector into pre[HC_MULT], post[HC_MULT] and a
// Sinkhorn-normalised comb[HC_MULT*HC_MULT]. Mirrors hc_split_sinkhorn_kernel.
inline void hc_split_sinkhorn(const float * mixes, const float * scale, const float * base,
                              float * pre, float * post, float * comb) {
    const int hc = HC_MULT;
    for (int j = 0; j < hc; ++j)
        pre[j] = 1.f / (1.f + std::exp(-(mixes[j] * scale[0] + base[j]))) + HC_EPS;
    for (int j = 0; j < hc; ++j)
        post[j] = 2.f / (1.f + std::exp(-(mixes[j + hc] * scale[1] + base[j + hc])));
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k)
            comb[j * hc + k] = mixes[j * hc + k + hc * 2] * scale[2] + base[j * hc + k + hc * 2];

    // comb = softmax(comb, dim=row) + eps
    for (int j = 0; j < hc; ++j) {
        float mx = comb[j * hc];
        for (int k = 1; k < hc; ++k) mx = std::max(mx, comb[j * hc + k]);
        float sm = 0.f;
        for (int k = 0; k < hc; ++k) { comb[j * hc + k] = std::exp(comb[j * hc + k] - mx); sm += comb[j * hc + k]; }
        for (int k = 0; k < hc; ++k) comb[j * hc + k] = comb[j * hc + k] / sm + HC_EPS;
    }
    // comb = comb / (col_sum + eps)
    for (int k = 0; k < hc; ++k) {
        float cs = 0.f;
        for (int j = 0; j < hc; ++j) cs += comb[j * hc + k];
        for (int j = 0; j < hc; ++j) comb[j * hc + k] /= (cs + HC_EPS);
    }
    // Sinkhorn: alternate row / column normalisation
    for (int it = 0; it < HC_SINKHORN_ITERS - 1; ++it) {
        for (int j = 0; j < hc; ++j) {
            float rs = 0.f;
            for (int k = 0; k < hc; ++k) rs += comb[j * hc + k];
            for (int k = 0; k < hc; ++k) comb[j * hc + k] /= (rs + HC_EPS);
        }
        for (int k = 0; k < hc; ++k) {
            float cs = 0.f;
            for (int j = 0; j < hc; ++j) cs += comb[j * hc + k];
            for (int j = 0; j < hc; ++j) comb[j * hc + k] /= (cs + HC_EPS);
        }
    }
}

// Reduce HC_MULT copies -> single state y[D]. hc_fn is [MIX_HC x HC_MULT*D].
// Also returns post[HC_MULT] and comb[HC_MULT*HC_MULT] for the later hc_post.
inline void hc_pre(const float * H, const float * hc_fn,
                   const float * scale, const float * base,
                   int D, float eps, float * y, float * post, float * comb) {
    const int hcd = HC_MULT * D;
    float ss = 0.f;
    for (int i = 0; i < hcd; ++i) ss += H[i] * H[i];
    const float rinv = 1.f / std::sqrt(ss / hcd + eps);

    float mixes[MIX_HC];
    for (int m = 0; m < MIX_HC; ++m) {
        const float * row = hc_fn + (size_t) m * hcd;
        float s = 0.f;
        for (int i = 0; i < hcd; ++i) s += row[i] * H[i];
        mixes[m] = s * rinv;
    }
    float pre[HC_MULT];
    hc_split_sinkhorn(mixes, scale, base, pre, post, comb);
    for (int d = 0; d < D; ++d) {
        float s = 0.f;
        for (int c = 0; c < HC_MULT; ++c) s += pre[c] * H[c * D + d];
        y[d] = s;
    }
}

// Expand sublayer output back to HC_MULT copies.
// H_out[j] = post[j] * x_out + sum_i comb[i,j] * residual[i]
inline void hc_post(const float * x_out, const float * residual,
                    const float * post, const float * comb, int D, float * H_out) {
    for (int j = 0; j < HC_MULT; ++j)
        for (int d = 0; d < D; ++d) {
            float s = post[j] * x_out[d];
            for (int i = 0; i < HC_MULT; ++i) s += comb[i * HC_MULT + j] * residual[i * D + d];
            H_out[j * D + d] = s;
        }
}

// Final head reduction: sigmoid-gated weighted sum of copies (no Sinkhorn).
// hc_fn is [HC_MULT x HC_MULT*D], scale is scalar (scale[0]), base is [HC_MULT].
inline void hc_head_reduce(const float * H, const float * hc_fn,
                           const float * scale, const float * base,
                           int D, float eps, float * y) {
    const int hcd = HC_MULT * D;
    float ss = 0.f;
    for (int i = 0; i < hcd; ++i) ss += H[i] * H[i];
    const float rinv = 1.f / std::sqrt(ss / hcd + eps);

    float pre[HC_MULT];
    for (int c = 0; c < HC_MULT; ++c) {
        const float * row = hc_fn + (size_t) c * hcd;
        float s = 0.f;
        for (int i = 0; i < hcd; ++i) s += row[i] * H[i];
        pre[c] = 1.f / (1.f + std::exp(-(s * rinv * scale[0] + base[c]))) + HC_EPS;
    }
    for (int d = 0; d < D; ++d) {
        float s = 0.f;
        for (int c = 0; c < HC_MULT; ++c) s += pre[c] * H[c * D + d];
        y[d] = s;
    }
}

} // namespace ds4
