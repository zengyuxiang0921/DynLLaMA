//
// Test: auto-read Qwen2.5 hyperparameters from a real GGUF file.
//
// Build (CPU backend, project-local mingw):
//   mingw64/bin/g++.exe -std=c++17 -I dynllama-headers/cpu -I dynllama-headers
//       tests/test-qwen2_5-meta.cpp -o test-qwen2_5-meta
//
// Run:
//   ./test-qwen2_5-meta qwen2.5-0.5b-instruct-q4_k_m.gguf
//

#include "prebuilt/qwen2_5/qwen2_5.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int g_failures = 0;

template <typename A, typename B>
static void check_eq(const char * what, A got, B expected) {
    if (got == (A) expected) {
        std::printf("  OK   %-22s = %s\n", what, std::to_string(got).c_str());
    } else {
        std::printf("  FAIL %-22s = %s (expected %s)\n", what,
                    std::to_string(got).c_str(), std::to_string((A) expected).c_str());
        g_failures++;
    }
}

static void check_str(const char * what, const std::string & got, const std::string & expected) {
    if (got == expected) {
        std::printf("  OK   %-22s = %s\n", what, got.c_str());
    } else {
        std::printf("  FAIL %-22s = %s (expected %s)\n", what, got.c_str(), expected.c_str());
        g_failures++;
    }
}

static void check_near(const char * what, double got, double expected, double tol) {
    double diff = got > expected ? got - expected : expected - got;
    if (diff <= tol) {
        std::printf("  OK   %-22s = %g\n", what, got);
    } else {
        std::printf("  FAIL %-22s = %g (expected ~%g)\n", what, got, expected);
        g_failures++;
    }
}

int main(int argc, char ** argv) {
    const std::string path = argc > 1 ? argv[1] : "qwen2.5-0.5b-instruct-q4_k_m.gguf";

    std::printf("Loading: %s\n", path.c_str());

    dynllama::qwen2_5_model model;
    std::string err;
    if (!dynllama::qwen2_5_load(path, model, &err)) {
        std::printf("ERROR: %s\n", err.c_str());
        return 2;
    }

    const dynllama::hparams & hp = model.hp;

    std::printf("\nAuto-read configuration (arch = %s):\n", hp.arch.c_str());
    check_str("arch",            hp.arch, "qwen2");
    check_eq ("n_layer",         hp.n_layer,      24u);
    check_eq ("n_ctx_train",     hp.n_ctx_train,  32768u);
    check_eq ("n_embd",          hp.n_embd,       896u);
    check_eq ("n_ff",            hp.n_ff,         4864u);
    check_eq ("n_head",          hp.n_head,       14u);
    check_eq ("n_head_kv",       hp.n_head_kv,    2u);
    check_eq ("n_embd_head",     hp.n_embd_head,  64u);
    check_eq ("n_embd_k_gqa",    hp.n_embd_k_gqa, 128u);
    check_eq ("n_vocab",         hp.n_vocab,      151936u);
    check_near("rope_freq_base", hp.rope_freq_base, 1000000.0, 1.0);
    check_near("rms_eps",        hp.rms_eps,        1e-6,      1e-7);

    std::printf("\nResolved architecture flags:\n");
    check_eq ("has_qkv_bias",    model.has_qkv_bias ? 1 : 0, 1);
    check_eq ("act_type(SiLU)",  model.act_type,             0);
    check_near("kq_scale",       model.kq_scale, 1.0 / 8.0,  1e-4); // 1/sqrt(64)=1/8
    check_eq ("layers built",    (uint32_t) model.layers.size(), hp.n_layer);

    std::printf("\nSample tensor names (layer 0):\n");
    const auto & l0 = model.layers.front();
    std::printf("  %s\n  %s / %s\n  %s\n",
                l0.attn_norm.c_str(), l0.attn_q_w.c_str(), l0.attn_q_b.c_str(),
                l0.ffn_gate.c_str());

    std::printf("\n%s (%d failure%s)\n",
                g_failures == 0 ? "PASSED" : "FAILED",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
