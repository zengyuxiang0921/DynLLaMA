//
// Integration test: load a GGUF with embedded model code via the llama.cpp
// public API (llama_model_load_from_file + llama_context + llama_decode),
// confirm the DynLLaMA path activates and produces finite logits.
//
// This binary does NOT link model-code/qwen2_5.cpp - the model code is
// extracted from the GGUF and JIT-compiled at runtime.
//
// Build:
//   g++ -std=c++17 -O2 -static -I include -I src -I dynllama-headers/cpu
//       -I dynllama-headers -I ggml/include -I ggml/src
//       tests/test-dynllama-integration.cpp
//       <all llama + ggml sources or the built static library>
//       -o test-dynllama-integration
//
// But since this links the full llama.cpp stack, it is best built with CMake.
// A simpler demonstration: use test-gguf-jit.cpp which doesn't need llama.cpp.
//
// Run (set compiler via env var on Windows cmd):
//   set DYNLLAMA_COMPILER=./mingw64/bin/g++.exe
//   ./test-dynllama-integration qwen2.5-0.5b-with-code.gguf "The capital of France is" 20
//

#include "llama.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const std::string path   = argc > 1 ? argv[1] : "qwen2.5-0.5b-with-code.gguf";
    const std::string prompt = argc > 2 ? argv[2] : "The capital of France is";
    const int         n_gen  = argc > 3 ? std::atoi(argv[3]) : 20;

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (!model) {
        std::printf("model load failed\n");
        return 2;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 256;
    cparams.n_batch = 256;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::printf("context init failed\n");
        llama_model_free(model);
        return 2;
    }

    // tokenize prompt
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(prompt.size() + 32);
    int n = llama_tokenize(llama_model_get_vocab(model),
                           prompt.c_str(), (int) prompt.size(),
                           tokens.data(), (int) tokens.size(),
                           /*add_special=*/false, /*parse_special=*/false);
    if (n < 0) { std::printf("tokenize failed\n"); return 2; }
    tokens.resize(n);
    std::printf("prompt: \"%s\"  (%d tokens)\n", prompt.c_str(), n);

    // build batch
    llama_batch batch = llama_batch_init(256, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0]= 0;
        batch.logits[i]   = (i == n - 1) ? 1 : 0;
    }
    batch.n_tokens = n;

    if (llama_decode(ctx, batch) != 0) {
        std::printf("decode failed\n");
        return 2;
    }

    // greedy generation
    auto argmax = [&](const float * row) {
        int best = 0; float bv = row[0];
        for (int v = 1; v < n_vocab; v++) if (row[v] > bv) { bv = row[v]; best = v; }
        return best;
    };

    int next = argmax(llama_get_logits_ith(ctx, n - 1));
    int n_past = n;

    std::vector<llama_token> out;
    for (int i = 0; i < n_gen && n_past < 256; i++) {
        if (next == llama_vocab_eos(llama_model_get_vocab(model))) break;
        out.push_back(next);

        batch.token[0]     = next;
        batch.pos[0]       = n_past;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        batch.n_tokens     = 1;

        if (llama_decode(ctx, batch) != 0) { std::printf("decode step failed\n"); break; }
        next = argmax(llama_get_logits_ith(ctx, 0));
        n_past++;
    }

    // decode output tokens to string
    std::string result;
    for (llama_token tok : out) {
        char buf[256] = {};
        llama_token_to_piece(llama_model_get_vocab(model), tok, buf, sizeof(buf), 0, false);
        result += buf;
    }

    std::printf("\n=== generation ===\n%s%s\n", prompt.c_str(), result.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
