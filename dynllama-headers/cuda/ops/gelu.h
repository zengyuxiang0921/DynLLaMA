// GELU activation (tanh approximation)

#ifdef __CUDACC__
extern "C" __global__
void gelu(float * out, const float * in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}
#else
#include <cmath>
static inline void gelu(float * out, const float * in, int n) {
    for (int i = 0; i < n; i++) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}
#endif
