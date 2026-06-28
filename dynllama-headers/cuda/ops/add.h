// Element-wise addition: out[i] = a[i] + b[i]

#ifdef __CUDACC__
extern "C" __global__
void add_ew(float * out, const float * a, const float * b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
#else
static inline void add_ew(float * out, const float * a, const float * b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}
#endif
