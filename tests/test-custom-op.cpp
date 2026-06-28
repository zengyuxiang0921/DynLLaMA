//
// E2E test: take an op body, wrap it in the standard scaffold, JIT-compile it
// into a shared library, load the symbol, and run it.
//
// Build:
//   g++ -std=c++17 -O2 -static -I dynllama-headers/cpu -I dynllama-headers -I src
//       tests/test-custom-op.cpp -o test-custom-op
//
// Run:
//   ./test-custom-op ./mingw64/bin/g++.exe
//

#include "custom_op.h"   // dynllama-headers/cpu/custom_op.h

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const std::string compiler = argc > 1 ? argv[1] : "g++";

    dynllama::jit_options opt;
    opt.compiler = compiler;

    // This body is exactly what would live in the GGUF at
    // dynllama.custom_op.my_add.body  (or the single-key dynllama.custom_ops).
    dynllama::custom_op_spec spec;
    spec.name = "my_add";
    spec.body = "n[i] = a[i] + b[i]";

    std::printf("=== generated source ===\n%s\n",
                dynllama::custom_op_source(spec).c_str());

    void * handle = nullptr;
    void * fn     = nullptr;
    std::string err;
    if (!dynllama::custom_op_build(spec, opt, &handle, &fn, &err)) {
        std::printf("build failed: %s\n", err.c_str());
        return 1;
    }
    auto add = reinterpret_cast<dynllama::custom_binop_fn>(fn);

    const int N = 8;
    std::vector<float> a(N), b(N), n(N, 0.0f);
    for (int i = 0; i < N; i++) { a[i] = (float) i; b[i] = (float) (2 * i); }

    add(a.data(), b.data(), n.data(), N);

    bool ok = true;
    std::printf("=== result ===\n");
    for (int i = 0; i < N; i++) {
        const float want = a[i] + b[i];
        std::printf("  n[%d] = %.1f (want %.1f)\n", i, n[i], want);
        if (n[i] != want) ok = false;
    }

    dynllama::jit_unload_handle(handle);

    // Optional: read a custom op straight from a GGUF (argv[2]=gguf, argv[3]=name).
    if (argc > 3) {
        const std::string gguf = argv[2];
        const std::string name = argv[3];
        dynllama::gguf_meta meta;
        if (!meta.open(gguf)) {
            std::printf("gguf open failed: %s\n", meta.error().c_str());
            return 1;
        }
        void * h2 = nullptr;
        void * f2 = nullptr;
        if (!dynllama::custom_op_from_gguf(meta, name, opt, &h2, &f2, &err)) {
            std::printf("custom_op_from_gguf failed: %s\n", err.c_str());
            return 1;
        }
        auto op = reinterpret_cast<dynllama::custom_binop_fn>(f2);
        std::vector<float> r(N, 0.0f);
        op(a.data(), b.data(), r.data(), N);
        std::printf("=== from gguf op '%s' ===\n", name.c_str());
        for (int i = 0; i < N; i++) {
            std::printf("  r[%d] = %.1f\n", i, r[i]);
            if (r[i] != a[i] + b[i]) ok = false;
        }
        dynllama::jit_unload_handle(h2);
    }

    std::printf(ok ? "\nPASS\n" : "\nFAIL\n");
    return ok ? 0 : 1;
}
