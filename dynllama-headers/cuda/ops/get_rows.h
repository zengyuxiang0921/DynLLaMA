// Gather rows: out[row,:] = embd[ids[row],:]

#ifdef __CUDACC__
extern "C" __global__
void get_rows(float * out, const float * embd, const int * ids, int n, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n_embd) {
        int row = idx / n_embd;
        int col = idx % n_embd;
        int src = ids[row];
        out[row * n_embd + col] = embd[src * n_embd + col];
    }
}
#else
static inline void get_rows(float * out, const float * embd, const int * ids, int n, int n_embd) {
    for (int i = 0; i < n; i++) {
        int idx = ids[i];
        for (int j = 0; j < n_embd; j++) {
            out[i * n_embd + j] = embd[idx * n_embd + j];
        }
    }
}
#endif
