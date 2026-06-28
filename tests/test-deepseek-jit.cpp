//
// E2E test for the full embedded path:
//   - model code embedded as a whole folder (dynllama.model_code.file.*)
//   - custom ops embedded as dynllama.custom_op.*
//   - host JIT-compiles the folder, compiles the ops, builds the metadata
//     accessor, and calls dynllama_main.
//
// Prepare the GGUF:
//   python embed_model_code.py base.gguf model-code/deepseek_v4/ a.gguf
//   python embed_model_code.py op a.gguf add    "n[i] = a[i] + b[i]" b.gguf
//   python embed_model_code.py op b.gguf swiglu "n[i] = (a[i] / (1.0f + expf(-a[i]))) * b[i]" ds4.gguf
//
// Build:
//   g++ -std=c++17 -O2 -static -I dynllama-headers/cpu -I dynllama-headers -I src
//       tests/test-deepseek-jit.cpp -o test-deepseek-jit
//
// Run:
//   ./test-deepseek-jit ds4.gguf ./mingw64/bin/g++.exe
//

#include "dynllama/runtime.h"
#include "dynllama/gguf_jit.h"
#include "dynllama/custom_op.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::printf("Usage: %s model-with-code.gguf compiler\n", argv[0]);
        return 1;
    }
    const std::string path     = argv[1];
    const std::string compiler = argv[2];

    dynllama::gguf_meta meta;
    if (!meta.open(path)) { std::printf("open: %s\n", meta.error().c_str()); return 2; }

    dynllama::jit_options opt;
    opt.compiler     = compiler;
    opt.include_dirs = {"dynllama-headers/cpu", "dynllama-headers"};

    std::string err;

    // 1) JIT-compile the whole embedded model-code folder.
    dynllama::jit_module mod;
    const std::string lib = std::string("dynllama_ds4") + dynllama::jit_lib_ext();
    if (!dynllama::jit_from_gguf(meta, lib, opt, mod, &err)) {
        std::printf("model-code JIT failed: %s\n", err.c_str());
        return 2;
    }
    std::printf("model code compiled (dynllama_main loaded)\n");

    // 2) Compile every custom op embedded in the GGUF.
    dynllama::custom_op_registry reg;
    if (!dynllama::custom_op_load_all(meta, opt, reg, &err)) {
        std::printf("custom op compile failed: %s\n", err.c_str());
        return 2;
    }
    const dynllama_op_table * table = reg.view();
    std::printf("custom ops compiled: %d\n", table->n);

    // 3) Host load: dequantize weights and build the metadata accessor.
    dynllama::host_model model;
    if (!dynllama::host_load(path, model, &err)) {
        std::printf("host load: %s\n", err.c_str());
        return 2;
    }
    dynllama::host_kv_alloc(model, 64);

    // 4) Build the ctx (meta is set by host_make_ctx) and attach the op table.
    std::vector<int> toks = {1, 2, 3};
    const int nv = (int) model.hp.n_vocab;
    std::vector<float> logits((size_t) toks.size() * nv, 0.0f);
    auto c = dynllama::host_make_ctx(model, toks.data(), (int) toks.size(), 0,
                                     logits.data(), true);
    c.ops = table;

    std::printf("calling dynllama_main...\n");
    mod.eval(&c);

    int finite = 0;
    for (float v : logits) if (std::isfinite(v)) finite++;
    std::printf("logits: %d/%zu finite\n", finite, logits.size());

    const bool ok = (finite == (int) logits.size());
    dynllama::jit_unload(mod);
    std::printf(ok ? "\nPASS\n" : "\nFAIL\n");
    return ok ? 0 : 1;
}
