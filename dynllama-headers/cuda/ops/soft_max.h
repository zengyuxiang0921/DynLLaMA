// Softmax: out_i = exp(x_i - max) / sum(exp(x_j - max)) per row

#ifdef __CUDACC__
extern "C" __global__
void soft_max(float * out, const float * in, int n, int n_embd) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        const float * row_in = in + row * n_embd;
        float * row_out = out + row * n_embd;

        float max_val = row_in[0];
        for (int j = 1; j < n_embd; j++) {
            if (row_in[j] > max_val) max_val = row_in[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            row_out[j] = expf(row_in[j] - max_val);
            sum += row_out[j];
        }

        for (int j = 0; j < n_embd; j++) {
            row_out[j] /= sum;
        }
    }
}
#else
#include <cmath>
static inline void soft_max(float * out, const float * in, int n, int n_embd) {
    for (int i = 0; i < n; i++) {
        const float * row = in + i * n_embd;
        float * orow = out + i * n_embd;

        float max_val = row[0];
        for (int j = 1; j < n_embd; j++) {
            if (row[j] > max_val) max_val = row[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < n_embd; j++) {
            float e = std::exp(row[j] - max_val);
            orow[j] = e;
            sum += e;
        }

        for (int j = 0; j < n_embd; j++) {
            orow[j] /= sum;
        }
    }
}
#endif
