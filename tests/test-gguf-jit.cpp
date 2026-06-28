//
// E2E test: JIT-compile model code embedded in a GGUF, then run greedy generation.
//
// Unlike test-generate.cpp, this binary does NOT link model-code/qwen2_5.cpp.
// The model code is extracted from the GGUF dynllama.model_code.source KV,
// compiled to a shared library at runtime, and called through dynllama_eval.
//
// Workflow:
//   python embed_model_code.py model.gguf model-code/qwen2_5.cpp model-with-code.gguf
//   ./test-gguf-jit model-with-code.gguf ./mingw64/bin/g++.exe [prompt] [n_gen]
//
// Build:
//   g++ -std=c++17 -O2 -static -I dynllama-headers/cpu -I dynllama-headers -I src
//       tests/test-gguf-jit.cpp -o test-gguf-jit
//

#include "dynllama/runtime.h"
#include "dynllama/gguf_jit.h"
#include "blocks/tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::printf("Usage: %s model-with-code.gguf compiler [prompt] [n_gen]\n", argv[0]);
        return 1;
    }
    const std::string path     = argv[1];
    const std::string compiler = argv[2];
    const std::string prompt   = argc > 3 ? argv[3] : "The capital of France is";
    const int n_gen = argc > 4 ? std::atoi(argv[4]) : 20;

    // Open GGUF - reads tokenizer vocab and the embedded model-code source
    dynllama::gguf_meta meta;
    if (!meta.open(path)) {
        std::printf("open failed: %s\n", meta.error().c_str());
        return 2;
    }

    // Extract source from GGUF, write temp file, compile shared library, load
    dynllama::jit_options opt;
    opt.compiler     = compiler;
    opt.include_dirs = {"dynllama-headers/cpu", "dynllama-headers"};
    const std::string lib = std::string("dynllama_jit") + dynllama::jit_lib_ext();

    std::string err;
    dynllama::jit_module mod;
    std::printf("JIT-compiling embedded model code...\n");
    if (!dynllama::jit_from_gguf(meta, lib, opt, mod, &err)) {
        std::printf("JIT failed: %s\n", err.c_str());
        return 2;
    }
    std::printf("  dynllama_eval loaded from %s\n", lib.c_str());

    // Tokenizer (same GGUF, already open)
    dynllama::bpe_tokenizer tok;
    if (!tok.load(meta, &err)) {
        std::printf("tokenizer: %s\n", err.c_str());
        return 2;
    }

    // Dequantize all weights to float
    dynllama::host_model model;
    std::printf("Loading weights...\n");
    if (!dynllama::host_load(path, model, &err)) {
        std::printf("load: %s\n", err.c_str());
        return 2;
    }

    const int nv    = (int) model.hp.n_vocab;
    const int n_ctx = 256;
    dynllama::host_kv_alloc(model, n_ctx);

    std::vector<int> ids = tok.encode(prompt, false);
    std::printf("prompt: \"%s\"  (%zu tokens)\n", prompt.c_str(), ids.size());

    auto argmax_last = [&](const std::vector<float> & logits, int n_tok) {
        const float * row = logits.data() + (size_t)(n_tok - 1) * nv;
        int best = 0; float bv = row[0];
        for (int v = 1; v < nv; v++) if (row[v] > bv) { bv = row[v]; best = v; }
        return best;
    };

    // Prompt pass through the JIT-loaded eval
    std::vector<float> logits((size_t) ids.size() * nv);
    auto c = dynllama::host_make_ctx(model, ids.data(), (int) ids.size(), 0, logits.data(), true);
    mod.eval(&c);
    int n_past = (int) ids.size();
    int next   = argmax_last(logits, (int) ids.size());

    // Greedy decode
    std::vector<int> out;
    std::vector<float> step_logits((size_t) nv);
    for (int i = 0; i < n_gen && n_past < n_ctx; i++) {
        if (next == tok.eos_id) break;
        out.push_back(next);
        auto cs = dynllama::host_make_ctx(model, &next, 1, n_past, step_logits.data(), true);
        mod.eval(&cs);
        n_past++;
        next = argmax_last(step_logits, 1);
    }

    std::printf("\n=== generation ===\n%s%s\n", prompt.c_str(), tok.decode(out).c_str());
    dynllama::jit_unload(mod);
    return 0;
}
