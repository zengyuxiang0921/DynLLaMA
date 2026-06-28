//
// Capstone: greedy text generation through the full stack
// (tokenizer -> host weight load -> JIT-free dynllama_eval with RoPE + KV cache).
//
// This is a qualitative check: coherent English continuation means RoPE,
// attention, FFN, dequant and the tokenizer are all working together.
//
// Build:
//   g++ -std=c++17 -O2 -static -I dynllama-headers/cpu -I dynllama-headers -I src
//       tests/test-generate.cpp model-code/qwen2_5.cpp -o test-generate
// Run:
//   ./test-generate qwen2.5-0.5b-instruct-q4_k_m.gguf "The capital of France is"
//

#include "dynllama/runtime.h"
#include "dynllama_abi.h"
#include "meta.h"
#include "blocks/tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

extern "C" void dynllama_eval(dynllama_run_ctx * ctx);

int main(int argc, char ** argv) {
    const std::string path   = argc > 1 ? argv[1] : "qwen2.5-0.5b-instruct-q4_k_m.gguf";
    const std::string prompt = argc > 2 ? argv[2] : "The capital of France is";
    const int n_gen = argc > 3 ? std::atoi(argv[3]) : 20;

    dynllama::gguf_meta meta;
    if (!meta.open(path)) { std::printf("open failed: %s\n", meta.error().c_str()); return 2; }
    dynllama::bpe_tokenizer tok;
    std::string err;
    if (!tok.load(meta, &err)) { std::printf("tokenizer: %s\n", err.c_str()); return 2; }

    dynllama::host_model model;
    std::printf("Loading weights...\n");
    if (!dynllama::host_load(path, model, &err)) { std::printf("load: %s\n", err.c_str()); return 2; }

    const int nv = (int) model.hp.n_vocab;
    const int n_ctx = 256;
    dynllama::host_kv_alloc(model, n_ctx);

    std::vector<int> ids = tok.encode(prompt, /*add_special=*/false);
    std::printf("prompt: \"%s\"  (%zu tokens)\n", prompt.c_str(), ids.size());

    auto argmax_last = [&](const std::vector<float> & logits, int n_tok) {
        const float * row = logits.data() + (size_t)(n_tok - 1) * nv;
        int best = 0; float bv = row[0];
        for (int v = 1; v < nv; v++) if (row[v] > bv) { bv = row[v]; best = v; }
        return best;
    };

    // process the prompt
    std::vector<float> logits((size_t) ids.size() * nv);
    auto c = dynllama::host_make_ctx(model, ids.data(), (int) ids.size(), 0, logits.data(), true);
    dynllama_eval(&c);
    int n_past = (int) ids.size();
    int next = argmax_last(logits, (int) ids.size());

    // greedy decode
    std::vector<int> out;
    std::vector<float> step_logits((size_t) nv);
    for (int i = 0; i < n_gen && n_past < n_ctx; i++) {
        if (next == tok.eos_id) break;
        out.push_back(next);
        auto cs = dynllama::host_make_ctx(model, &next, 1, n_past, step_logits.data(), true);
        dynllama_eval(&cs);
        n_past++;
        next = argmax_last(step_logits, 1);
    }

    std::printf("\n=== generation ===\n%s%s\n", prompt.c_str(), tok.decode(out).c_str());
    return 0;
}
