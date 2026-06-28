//
// E2E test: compile the DeepSeek V4 elementwise ops from their custom-op bodies,
// hand them to the model code through dynllama_run_ctx::ops, and run a forward
// pass. Proves the model code uses host-compiled custom ops.
//
// Build:
//   g++ -std=c++17 -O2 -static
//       -I dynllama-headers/cpu -I dynllama-headers -I src -I model-code/deepseek_v4
//       tests/test-deepseek-ops.cpp model-code/deepseek_v4/main.cpp -o test-deepseek-ops
//
// Run:
//   ./test-deepseek-ops ./mingw64/bin/g++.exe
//

#include "custom_op.h"   // dynllama-headers/cpu/custom_op.h
#include "ops.h"         // ds4::N_VOCAB, ds4::ADD_BODY, ds4::SWIGLU_BODY, ds4::binop_fn

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

extern "C" void dynllama_main(dynllama_run_ctx * c);

int main(int argc, char ** argv) {
    const std::string compiler = argc > 1 ? argv[1] : "g++";
    dynllama::jit_options opt;
    opt.compiler = compiler;

    std::string err;
    void *add_h = nullptr, *add_fn = nullptr;
    void *swg_h = nullptr, *swg_fn = nullptr;

    // Compile the ops from the SAME body strings the model documents/embeds.
    dynllama::custom_op_spec sa; sa.name = "add";    sa.body = ds4::ADD_BODY;
    dynllama::custom_op_spec ss; ss.name = "swiglu"; ss.body = ds4::SWIGLU_BODY;
    if (!dynllama::custom_op_build(sa, opt, &add_h, &add_fn, &err)) {
        std::printf("add build failed: %s\n", err.c_str());  return 1;
    }
    if (!dynllama::custom_op_build(ss, opt, &swg_h, &swg_fn, &err)) {
        std::printf("swiglu build failed: %s\n", err.c_str()); return 1;
    }

    dynllama_op_entry entries[2] = { {"add", add_fn}, {"swiglu", swg_fn} };
    dynllama_op_table table{entries, 2};

    bool ok = true;

    // Sanity-check the compiled ops directly.
    {
        auto add = reinterpret_cast<ds4::binop_fn>(add_fn);
        auto swg = reinterpret_cast<ds4::binop_fn>(swg_fn);
        const int N = 4;
        float a[N] = {1, 2, 3, 4}, b[N] = {10, 20, 30, 40}, r[N] = {0};
        add(a, b, r, N);
        for (int i = 0; i < N; i++) if (r[i] != a[i] + b[i]) ok = false;
        swg(a, b, r, N);
        for (int i = 0; i < N; i++) {
            float want = (a[i] / (1.0f + std::exp(-a[i]))) * b[i];
            if (std::fabs(r[i] - want) > 1e-4f) ok = false;
        }
        std::printf("op sanity: %s\n", ok ? "ok" : "FAILED");
    }

    // Run the model with the op table installed.
    dynllama_run_ctx c{};
    std::vector<int> toks = {1, 2, 3};
    c.n_vocab  = ds4::N_VOCAB;
    c.tokens   = toks.data();
    c.n_tokens = (int) toks.size();
    c.n_past   = 0;
    std::vector<float> logits((size_t) c.n_tokens * c.n_vocab, 0.0f);
    c.logits   = logits.data();
    c.ops      = &table;

    std::printf("running dynllama_main with op table...\n");
    dynllama_main(&c);

    int finite = 0;
    for (float v : logits) if (std::isfinite(v)) finite++;
    std::printf("logits: %d/%zu finite\n", finite, logits.size());
    if (finite != (int) logits.size()) ok = false;

    dynllama::jit_unload_handle(add_h);
    dynllama::jit_unload_handle(swg_h);
    std::printf(ok ? "\nPASS\n" : "\nFAIL\n");
    return ok ? 0 : 1;
}
