// RMS Layer Normalization: out = x / sqrt(mean(x^2) + eps) * weight

#ifdef __CUDACC__
extern "C" __global__
void rms_norm(float * out, const float * x, const float * weight, int n, int n_embd, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        const float * x_row = x + row * n_embd;
        float * out_row = out + row * n_embd;

        float ss = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            ss += x_row[j] * x_row[j];
        }
        float scale = 1.0f / sqrtf(ss / n_embd + eps);
        for (int j = 0; j < n_embd; j++) {
            out_row[j] = x_row[j] * scale * weight[j];
        }
    }
}
#else
#include <cmath>
static inline void rms_norm(float * out, const float * x, const float * weight, int n, int n_embd, float eps) {
    for (int i = 0; i < n; i++) {
        const float * row = x + i * n_embd;
        float * orow = out + i * n_embd;

        float ss = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            ss += row[j] * row[j];
        }
        float rms = std::sqrt(ss / (float)n_embd + eps);
        float inv_rms = 1.0f / rms;

        for (int j = 0; j < n_embd; j++) {
            orow[j] = weight[j] * (row[j] * inv_rms);
        }
    }
}
#endif
