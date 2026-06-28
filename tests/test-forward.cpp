//
// Test: dequant op + ABI forward + KV-incremental == full-sequence equivalence.
// Links model-code/qwen2_5.cpp (which provides dynllama_eval).
//
// Build:
//   g++ -std=c++17 -I dynllama-headers/cpu -I dynllama-headers
//       tests/test-forward.cpp model-code/qwen2_5.cpp -o test-forward
//

#include "dynllama.h"
#include "dynllama_abi.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
static void ok(bool cond, const char * what) {
    std::printf("  %s %s\n", cond ? "OK  " : "FAIL", what);
    if (!cond) g_fail++;
}

// ---- dequant op ----
static uint16_t fp32_to_fp16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0)  return (uint16_t) sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static void test_dequant() {
    std::printf("dequantize op:\n");
    float src[32];
    for (int i = 0; i < 32; i++) src[i] = (i - 16) * 0.5f;
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) amax = std::fmax(amax, std::fabs(src[i]));
    const float d = amax / 127.0f;
    uint8_t block[2 + 32];
    uint16_t dh = fp32_to_fp16(d);
    std::memcpy(block, &dh, 2);
    for (int i = 0; i < 32; i++) {
        int q = (int) std::lround(src[i] / d);
        if (q > 127)  q = 127;
        if (q < -128) q = -128;
        ((int8_t *) (block + 2))[i] = (int8_t) q;
    }
    float out[32];
    ok(dequantize_row(DQ_Q8_0, block, out, 32), "Q8_0 supported");
    float me = 0.0f;
    for (int i = 0; i < 32; i++) me = std::fmax(me, std::fabs(out[i] - src[i]));
    ok(me < d, "Q8_0 round-trip within one quantum");
    ok(dequantize_type_size(DQ_Q4_K, 256) == 144, "Q4_K block size 144");
    ok(dequantize_type_size(DQ_Q6_K, 256) == 210, "Q6_K block size 210");
    ok(!dequantize_row(10 /* Q2_K */, block, out, 256), "Q2_K reports unsupported");
}

// ---- tiny synthetic qwen2-style model ----
struct tiny {
    static const int n_embd = 8, n_head = 2, n_head_kv = 1, n_embd_head = 4;
    static const int n_ff = 16, n_layer = 2, n_vocab = 5;
    static const int n_embd_q = n_head * n_embd_head;    // 8
    static const int n_embd_kv = n_head_kv * n_embd_head; // 4
    std::vector<float> tok_embd, out_norm, out_w;
    std::vector<std::vector<float>> aN, wq, bq, wk, bk, wv, bv, wo, fN, fU, fG, fD;
    std::vector<dynllama_layer_weights> L;

    static void fill(std::vector<float>& v, int n, float s, int seed) {
        v.resize(n);
        for (int i = 0; i < n; i++) v[i] = std::sin(0.3f * (i + 1) + seed) * s;
    }
    tiny() {
        fill(tok_embd, n_vocab * n_embd, 0.5f, 1);
        fill(out_norm, n_embd, 1.0f, 2);
        fill(out_w, n_vocab * n_embd, 0.4f, 3);
        aN.resize(n_layer); wq.resize(n_layer); bq.resize(n_layer);
        wk.resize(n_layer); bk.resize(n_layer); wv.resize(n_layer); bv.resize(n_layer);
        wo.resize(n_layer); fN.resize(n_layer); fU.resize(n_layer);
        fG.resize(n_layer); fD.resize(n_layer); L.resize(n_layer);
        for (int i = 0; i < n_layer; i++) {
            fill(aN[i], n_embd, 1.0f, 10 + i);
            fill(wq[i], n_embd_q * n_embd, 0.3f, 20 + i); fill(bq[i], n_embd_q, 0.1f, 21 + i);
            fill(wk[i], n_embd_kv * n_embd, 0.3f, 22 + i); fill(bk[i], n_embd_kv, 0.1f, 23 + i);
            fill(wv[i], n_embd_kv * n_embd, 0.3f, 24 + i); fill(bv[i], n_embd_kv, 0.1f, 25 + i);
            fill(wo[i], n_embd * n_embd_q, 0.3f, 26 + i);
            fill(fN[i], n_embd, 1.0f, 27 + i);
            fill(fU[i], n_ff * n_embd, 0.2f, 28 + i);
            fill(fG[i], n_ff * n_embd, 0.2f, 29 + i);
            fill(fD[i], n_embd * n_ff, 0.2f, 30 + i);
            dynllama_layer_weights & w = L[i];
            w.attn_norm = aN[i].data();
            w.wq = wq[i].data(); w.bq = bq[i].data();
            w.wk = wk[i].data(); w.bk = bk[i].data();
            w.wv = wv[i].data(); w.bv = bv[i].data();
            w.wo = wo[i].data(); w.wo_b = nullptr;
            w.ffn_norm = fN[i].data();
            w.ffn_up = fU[i].data(); w.ffn_gate = fG[i].data(); w.ffn_down = fD[i].data();
        }
    }
    dynllama_run_ctx ctx(const int* tokens, int nt, int n_past, float* logits, dynllama_kv_cache* kv) {
        dynllama_run_ctx c{};
        c.n_layer = n_layer; c.n_embd = n_embd; c.n_ff = n_ff;
        c.n_head = n_head; c.n_head_kv = n_head_kv; c.n_embd_head = n_embd_head;
        c.n_vocab = n_vocab; c.n_rot = n_embd_head; c.act_type = 0; c.rms_eps = 1e-5f;
        c.rope_freq_base = 10000.0f;
        c.tok_embd = tok_embd.data(); c.output_norm_w = out_norm.data(); c.output_w = out_w.data();
        c.layers = L.data(); c.kv = kv;
        c.tokens = tokens; c.n_tokens = nt; c.n_past = n_past; c.logits = logits;
        return c;
    }
};

static void test_forward_and_kv() {
    std::printf("forward + KV equivalence:\n");
    tiny m;
    const int nt = 4, nv = tiny::n_vocab;
    int tokens[nt] = {1, 3, 0, 2};

    // full-sequence pass
    std::vector<float> full(nt * nv, 0.0f);
    auto cf = m.ctx(tokens, nt, 0, full.data(), nullptr);
    dynllama_eval(&cf);

    bool finite = true;
    for (float x : full) finite = finite && std::isfinite(x);
    ok(finite, "full-sequence logits finite");

    // incremental pass via KV cache, one token at a time
    const int n_embd_kv = tiny::n_embd_kv, n_ctx = nt, nl = tiny::n_layer;
    std::vector<float> kbuf((size_t) nl * n_embd_kv * n_ctx, 0.0f);
    std::vector<float> vbuf((size_t) nl * n_embd_kv * n_ctx, 0.0f);
    dynllama_kv_cache kv{};
    kv.n_layer = nl; kv.n_embd_kv = n_embd_kv; kv.n_ctx = n_ctx;
    kv.k = kbuf.data(); kv.v = vbuf.data();

    std::vector<float> inc(nt * nv, 0.0f);
    for (int t = 0; t < nt; t++) {
        auto ci = m.ctx(&tokens[t], 1, t, &inc[t * nv], &kv);
        dynllama_eval(&ci);
    }

    double maxdiff = 0.0;
    for (int i = 0; i < nt * nv; i++)
        maxdiff = std::fmax(maxdiff, std::fabs(inc[i] - full[i]));
    std::printf("    max |incremental - full| = %.3e\n", maxdiff);
    ok(maxdiff < 1e-4, "KV incremental matches full sequence");
}

int main() {
    test_dequant();
    test_forward_and_kv();
    std::printf("\n%s (%d failure%s)\n", g_fail == 0 ? "PASSED" : "FAILED",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
