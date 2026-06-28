// Rotary Position Embedding

#ifdef __CUDACC__
extern "C" __global__
void rope(float * out, const float * in, const float * freqs, int n_tokens, int n_embd, int n_head, int n_embd_head) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * n_head * (n_embd_head / 2);
    if (idx < total) {
        int j2 = (idx % (n_embd_head / 2)) * 2;
        int h  = (idx / (n_embd_head / 2)) % n_head;
        int t  = idx / (n_head * (n_embd_head / 2));

        int pos = t * n_embd + h * n_embd_head + j2;
        float cos_v = cosf(freqs[t * n_embd_head / 2 + j2 / 2]);
        float sin_v = sinf(freqs[t * n_embd_head / 2 + j2 / 2]);

        float x0 = in[pos];
        float x1 = in[pos + 1];

        out[pos]     = x0 * cos_v - x1 * sin_v;
        out[pos + 1] = x0 * sin_v + x1 * cos_v;
    }
}
#else
#include <cmath>
static inline void rope(float * out, const float * in, const float * freqs, int n_tokens, int n_embd, int n_head, int n_embd_head) {
    for (int i = 0; i < n_tokens; i++) {
        for (int h = 0; h < n_head; h++) {
            for (int j = 0; j < n_embd_head; j += 2) {
                int idx = i * n_embd + h * n_embd_head + j;
                float cos = std::cos(freqs[i * n_embd_head / 2 + j / 2]);
                float sin = std::sin(freqs[i * n_embd_head / 2 + j / 2]);

                float x0 = in[idx];
                float x1 = in[idx + 1];

                out[idx]     = x0 * cos - x1 * sin;
                out[idx + 1] = x0 * sin + x1 * cos;
            }
        }
    }
}
#endif
