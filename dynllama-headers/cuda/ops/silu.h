// SiLU activation: x / (1 + exp(-x))

#ifdef __CUDACC__
extern "C" __global__
void silu(float * out, const float * in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] / (1.0f + expf(-in[i]));
    }
}
#else
#include <cmath>
static inline void silu(float * out, const float * in, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] / (1.0f + std::exp(-in[i]));
    }
}
#endif
