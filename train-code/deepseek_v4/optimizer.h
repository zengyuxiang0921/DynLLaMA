#pragma once
// DeepSeek V4 optimizers.
//   - Muon for 2D weight matrices: orthogonalize the momentum via a
//     Newton-Schulz quintic iteration, then step.
//   - AdamW for embeddings, norms, router and other 1D params.

#include <algorithm>
#include <cmath>
#include <vector>

namespace ds4train {

inline void adamw_step(std::vector<float> & w, const std::vector<float> & g,
                       std::vector<float> & m, std::vector<float> & v,
                       float lr, float beta1, float beta2, float eps,
                       float wd, int t) {
    const float bc1 = 1.f - std::pow(beta1, (float) t);
    const float bc2 = 1.f - std::pow(beta2, (float) t);
    for (size_t i = 0; i < w.size(); ++i) {
        m[i] = beta1 * m[i] + (1.f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.f - beta2) * g[i] * g[i];
        const float mh = m[i] / bc1;
        const float vh = v[i] / bc2;
        w[i] -= lr * (mh / (std::sqrt(vh) + eps) + wd * w[i]);
    }
}

// Newton-Schulz quintic iteration: approximates the orthogonal polar factor of
// the [rows x cols] matrix X in place.
inline void newton_schulz(std::vector<float> & X, int rows, int cols, int iters = 5) {
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    float fro = 0.f;
    for (float x : X) fro += x * x;
    fro = std::sqrt(fro) + 1e-7f;
    for (float & x : X) x /= fro;

    std::vector<float> A((size_t) rows * rows);
    std::vector<float> T1((size_t) rows * cols), T2((size_t) rows * cols);
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < rows; ++i)                 // A = X X^T
            for (int j = 0; j < rows; ++j) {
                float s = 0.f;
                for (int k = 0; k < cols; ++k) s += X[i * cols + k] * X[j * cols + k];
                A[i * rows + j] = s;
            }
        for (int i = 0; i < rows; ++i)                 // T1 = A X
            for (int k = 0; k < cols; ++k) {
                float s = 0.f;
                for (int j = 0; j < rows; ++j) s += A[i * rows + j] * X[j * cols + k];
                T1[i * cols + k] = s;
            }
        for (int i = 0; i < rows; ++i)                 // T2 = A T1
            for (int k = 0; k < cols; ++k) {
                float s = 0.f;
                for (int j = 0; j < rows; ++j) s += A[i * rows + j] * T1[j * cols + k];
                T2[i * cols + k] = s;
            }
        for (size_t idx = 0; idx < X.size(); ++idx)
            X[idx] = a * X[idx] + b * T1[idx] + c * T2[idx];
    }
}

inline void muon_step(std::vector<float> & w, const std::vector<float> & g,
                      std::vector<float> & mom, int rows, int cols,
                      float lr, float momentum, float wd) {
    for (size_t i = 0; i < w.size(); ++i) mom[i] = momentum * mom[i] + g[i];
    std::vector<float> o(mom.begin(), mom.end());
    newton_schulz(o, rows, cols);
    // rescale so the update RMS is comparable across matrix shapes
    const float scale = std::sqrt((float) std::max(rows, cols));
    for (size_t i = 0; i < w.size(); ++i)
        w[i] -= lr * (scale * o[i] + wd * w[i]);
}

} // namespace ds4train
