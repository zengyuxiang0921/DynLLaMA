#pragma once
// mHC manifold constraint -- training-side copy.
//
// The mHC residual mixes the hidden state through M = U * V^T. Left
// unconstrained, hyper-connections can blow the residual gain up by ~3000x.
// Re-projecting (U, V) onto the Sinkhorn-Knopp manifold after optimizer steps
// keeps |M|'s row/column sums near 1, bounding the gain to <=1.6x.

#include <cmath>
#include <vector>

namespace ds4train {

inline void sinkhorn_normalize(std::vector<float> & U, std::vector<float> & V,
                               int D, int rank, int iters = 4) {
    auto absM = [&](int i, int j) {
        float s = 0.f;
        for (int r = 0; r < rank; ++r) s += U[i * rank + r] * V[j * rank + r];
        return std::fabs(s);
    };
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < D; ++i) {          // balance rows of M -> scale U row i
            float rs = 0.f;
            for (int j = 0; j < D; ++j) rs += absM(i, j);
            float sc = rs > 1e-12f ? 1.f / rs : 1.f;
            for (int r = 0; r < rank; ++r) U[i * rank + r] *= sc;
        }
        for (int j = 0; j < D; ++j) {          // balance cols of M -> scale V row j
            float cs = 0.f;
            for (int i = 0; i < D; ++i) cs += absM(i, j);
            float sc = cs > 1e-12f ? 1.f / cs : 1.f;
            for (int r = 0; r < rank; ++r) V[j * rank + r] *= sc;
        }
    }
}

} // namespace ds4train
