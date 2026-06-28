//
// Test: JIT-compile the embedded model code into a shared library, load
// dynllama_eval from it, and run it. Demonstrates the host JIT path.
//
// The test binary does NOT link model-code/qwen2_5.cpp - it is compiled and
// loaded at runtime, exactly as llama.cpp will do with the gguf-embedded code.
//
// Build:
//   g++ -std=c++17 -I dynllama-headers/cpu -I dynllama-headers -I src
//       tests/test-jit.cpp -o test-jit            (link -ldl on POSIX)
// Run (pass the compiler to use for JIT):
//   ./test-jit ./mingw64/bin/g++.exe
//

#include "dynllama_abi.h"
#include "dynllama/jit.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
static void ok(bool c, const char * w) { std::printf("  %s %s\n", c ? "OK  " : "FAIL", w); if (!c) g_fail++; }

// minimal synthetic qwen2-style weights (token-major, ggml weight layout)
struct mini {
    static const int n_embd = 8, n_head = 2, n_head_kv = 1, n_embd_head = 4;
    static const int n_ff = 16, n_layer = 2, n_vocab = 5;
    static const int n_embd_q = n_head * n_embd_head, n_embd_kv = n_head_kv * n_embd_head;
    std::vector<float> tok, on, ow;
    std::vector<float> aN[n_layer], wq[n_layer], bq[n_layer], wk[n_layer], bk[n_layer],
                       wv[n_layer], bv[n_layer], wo[n_layer], fN[n_layer], fU[n_layer],
                       fG[n_layer], fD[n_layer];
    dynllama_layer_weights L[n_layer];
    static void fill(std::vector<float>& v, int n, float s, int sd) {
        v.resize(n);
        for (int i = 0; i < n; i++) v[i] = std::sin(0.3f * (i + 1) + sd) * s;
    }
    mini() {
        fill(tok, n_vocab * n_embd, 0.5f, 1); fill(on, n_embd, 1.0f, 2); fill(ow, n_vocab * n_embd, 0.4f, 3);
        for (int i = 0; i < n_layer; i++) {
            fill(aN[i], n_embd, 1.0f, 10 + i);
            fill(wq[i], n_embd_q * n_embd, 0.3f, 20 + i);  fill(bq[i], n_embd_q, 0.1f, 21 + i);
            fill(wk[i], n_embd_kv * n_embd, 0.3f, 22 + i); fill(bk[i], n_embd_kv, 0.1f, 23 + i);
            fill(wv[i], n_embd_kv * n_embd, 0.3f, 24 + i); fill(bv[i], n_embd_kv, 0.1f, 25 + i);
            fill(wo[i], n_embd * n_embd_q, 0.3f, 26 + i);
            fill(fN[i], n_embd, 1.0f, 27 + i);
            fill(fU[i], n_ff * n_embd, 0.2f, 28 + i);
            fill(fG[i], n_ff * n_embd, 0.2f, 29 + i);
            fill(fD[i], n_embd * n_ff, 0.2f, 30 + i);
            L[i] = {aN[i].data(), wq[i].data(), bq[i].data(), wk[i].data(), bk[i].data(),
                    wv[i].data(), bv[i].data(), wo[i].data(), nullptr, fN[i].data(),
                    fU[i].data(), fG[i].data(), fD[i].data()};
        }
    }
    dynllama_run_ctx ctx(const int* t, int nt, int np, float* lg, dynllama_kv_cache* kv) {
        dynllama_run_ctx c{};
        c.n_layer = n_layer; c.n_embd = n_embd; c.n_ff = n_ff; c.n_head = n_head;
        c.n_head_kv = n_head_kv; c.n_embd_head = n_embd_head; c.n_vocab = n_vocab;
        c.n_rot = n_embd_head; c.act_type = 0; c.rms_eps = 1e-5f; c.rope_freq_base = 10000.0f;
        c.tok_embd = tok.data(); c.output_norm_w = on.data(); c.output_w = ow.data();
        c.layers = L; c.kv = kv; c.tokens = t; c.n_tokens = nt; c.n_past = np; c.logits = lg;
        return c;
    }
};

int main(int argc, char ** argv) {
    dynllama::jit_options opt;
    opt.compiler     = argc > 1 ? argv[1] : "g++";
    opt.include_dirs = {"dynllama-headers/cpu", "dynllama-headers"};

    const std::string lib = std::string("qwen2_5_jit") + dynllama::jit_lib_ext();

    std::printf("JIT-compiling model-code/qwen2_5.cpp with %s ...\n", opt.compiler.c_str());
    dynllama::jit_module mod;
    std::string err;
    if (!dynllama::jit_build("model-code/qwen2_5.cpp", lib, opt, mod, &err)) {
        std::printf("ERROR: %s\n", err.c_str());
        return 2;
    }
    ok(mod.ok(), "compiled + loaded dynllama_eval");

    mini m;
    const int nt = 4, nv = mini::n_vocab;
    int tokens[nt] = {1, 3, 0, 2};

    // full-sequence via JIT-loaded eval
    std::vector<float> full(nt * nv, 0.0f);
    auto cf = m.ctx(tokens, nt, 0, full.data(), nullptr);
    mod.eval(&cf);
    bool finite = true; for (float x : full) finite = finite && std::isfinite(x);
    ok(finite, "JIT full-sequence logits finite");

    // incremental via JIT-loaded eval + KV cache
    std::vector<float> kbuf(mini::n_layer * mini::n_embd_kv * nt, 0.0f);
    std::vector<float> vbuf(mini::n_layer * mini::n_embd_kv * nt, 0.0f);
    dynllama_kv_cache kv{}; kv.n_layer = mini::n_layer; kv.n_embd_kv = mini::n_embd_kv;
    kv.n_ctx = nt; kv.k = kbuf.data(); kv.v = vbuf.data();
    std::vector<float> inc(nt * nv, 0.0f);
    for (int t = 0; t < nt; t++) {
        auto ci = m.ctx(&tokens[t], 1, t, &inc[t * nv], &kv);
        mod.eval(&ci);
    }
    double md = 0.0;
    for (int i = 0; i < nt * nv; i++) md = std::fmax(md, std::fabs(inc[i] - full[i]));
    std::printf("    max |kv - full| via JIT = %.3e\n", md);
    ok(md < 1e-4, "JIT-loaded model code computes correctly (kv == full)");

    dynllama::jit_unload(mod);
    std::printf("\n%s (%d failure%s)\n", g_fail == 0 ? "PASSED" : "FAILED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
