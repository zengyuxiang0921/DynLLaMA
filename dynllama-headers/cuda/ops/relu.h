// ReLU activation: max(0, x)

#ifdef __CUDACC__
extern "C" __global__
void relu(float * out, const float * in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}
#else
static inline void relu(float * out, const float * in, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}
#endif
