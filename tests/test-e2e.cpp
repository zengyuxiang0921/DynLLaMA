//
// End-to-end: host-load the real Qwen2.5 gguf (dequantizing Q4_K/Q6_K/Q5_0/
// Q8_0/F32), then run the model code's dynllama_eval over a few tokens.
//
// Build:
//   g++ -std=c++17 -I dynllama-headers/cpu -I dynllama-headers -I src
//       tests/test-e2e.cpp model-code/qwen2_5.cpp -o test-e2e
// Run:
//   ./test-e2e qwen2.5-0.5b-instruct-q4_k_m.gguf
//

#include "dynllama/runtime.h"
#include "dynllama_abi.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
static void ok(bool c, const char * w) { std::printf("  %s %s\n", c ? "OK  " : "FAIL", w); if (!c) g_fail++; }

int main(int argc, char ** argv) {
    const std::string path = argc > 1 ? argv[1] : "qwen2.5-0.5b-instruct-q4_k_m.gguf";
    std::printf("Loading + dequantizing: %s\n", path.c_str());

    dynllama::host_model model;
    std::string err;
    if (!dynllama::host_load(path, model, &err)) {
        std::printf("ERROR: %s\n", err.c_str());
        return 2;
    }
    const dynllama::hparams & hp = model.hp;
    std::printf("  arch=%s n_layer=%u n_embd=%u n_head=%u/%u n_ff=%u n_vocab=%u\n",
                hp.arch.c_str(), hp.n_layer, hp.n_embd, hp.n_head, hp.n_head_kv, hp.n_ff, hp.n_vocab);

    ok(hp.n_layer == 24 && hp.n_embd == 896 && hp.n_vocab == 151936, "hparams as expected");
    ok(model.weights.size() > 0, "tensors dequantized");
    ok(model.tok_embd && model.output_w && model.layers.size() == hp.n_layer, "weights assembled");

    // a few arbitrary in-vocab token ids (no tokenizer wired up)
    std::vector<int> tokens = {9707, 2526, 11};
    const int nt = (int) tokens.size();
    const int nv = (int) hp.n_vocab;

    // full-sequence forward
    std::vector<float> logits((size_t) nt * nv, 0.0f);
    auto c = dynllama::host_make_ctx(model, tokens.data(), nt, 0, logits.data(), false);
    std::printf("Running full-sequence forward over %d tokens...\n", nt);
    dynllama_eval(&c);

    bool finite = true;
    for (float x : logits) finite = finite && std::isfinite(x);
    ok(finite, "logits finite");

    // argmax of the last token
    int best = 0; float bv = logits[(size_t)(nt - 1) * nv];
    for (int v = 1; v < nv; v++) {
        float x = logits[(size_t)(nt - 1) * nv + v];
        if (x > bv) { bv = x; best = v; }
    }
    std::printf("  last-token argmax = %d (logit %.4f)\n", best, bv);
    ok(best >= 0 && best < nv, "argmax in range");

    // KV-cache forward, incremental, must match the full pass on the last token
    dynllama::host_kv_alloc(model, 64);
    std::vector<float> inc((size_t) nt * nv, 0.0f);
    for (int t = 0; t < nt; t++) {
        auto ci = dynllama::host_make_ctx(model, &tokens[t], 1, t, &inc[(size_t) t * nv], true);
        dynllama_eval(&ci);
    }
    double maxdiff = 0.0;
    for (int v = 0; v < nv; v++)
        maxdiff = std::fmax(maxdiff, std::fabs(inc[(size_t)(nt - 1) * nv + v] - logits[(size_t)(nt - 1) * nv + v]));
    std::printf("  max |kv - full| (last token) = %.3e\n", maxdiff);
    ok(maxdiff < 1e-3, "KV cache matches full forward on real weights");

    std::printf("\n%s (%d failure%s)\n", g_fail == 0 ? "PASSED" : "FAILED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
