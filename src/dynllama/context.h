//
// DynLLaMA context state: owned by llama_context when the GGUF contains
// dynllama.model_code.file.* (multi-file) or dynllama.model_code.source (legacy).
//

#pragma once

#include "dynllama/runtime.h"   // host_model, host_load, host_kv_alloc, host_make_ctx
#include "dynllama/gguf_jit.h"  // jit_from_files, jit_from_source, jit_module, jit_unload

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

// -------------------------------------------------------------------------
// compiler / include-dir discovery
// -------------------------------------------------------------------------

static inline std::string dynllama_find_compiler() {
    if (const char * e = std::getenv("DYNLLAMA_COMPILER")) {
        return std::string(e);
    }
#ifdef _WIN32
    {
        std::ifstream probe("./mingw64/bin/g++.exe");
        if (probe.good()) return "./mingw64/bin/g++.exe";
    }
    return "g++.exe";
#else
    return "g++";
#endif
}

static inline std::vector<std::string> dynllama_find_includes() {
    const char * base = std::getenv("DYNLLAMA_HEADERS");
    std::string root = base ? std::string(base) : ".";
    return {root + "/dynllama-headers/cpu", root + "/dynllama-headers"};
}

// -------------------------------------------------------------------------
// per-context state
// -------------------------------------------------------------------------

struct dynllama_context_state {
    dynllama::host_model  model;
    dynllama::jit_module  jit;
    std::vector<float>    tmp_logits;

    ~dynllama_context_state() { dynllama::jit_unload(jit); }
};

// Initialize from multiple embedded files (dynllama.model_code.file.* keys).
// `files`: vector of (relpath, content) pairs.
static inline dynllama_context_state * dynllama_ctx_init(
        const std::vector<std::pair<std::string,std::string>> & files,
        const std::string & arch,
        const std::string & path,
        int                 n_ctx,
        const char *        log_prefix) {
    auto * state = new dynllama_context_state();
    std::string err;

    if (!dynllama::host_load(path, state->model, &err)) {
        fprintf(stderr, "%s: dynllama host_load failed: %s\n", log_prefix, err.c_str());
        delete state; return nullptr;
    }
    dynllama::host_kv_alloc(state->model, n_ctx);

    dynllama::jit_options opt;
    opt.compiler     = dynllama_find_compiler();
    opt.include_dirs = dynllama_find_includes();
    const std::string lib = std::string("dynllama_ctx") + dynllama::jit_lib_ext();

    fprintf(stderr, "%s: dynllama JIT-compiling %zu file(s) with %s ...\n",
            log_prefix, files.size(), opt.compiler.c_str());
    if (!dynllama::jit_from_files(files, arch, lib, opt, state->jit, &err)) {
        fprintf(stderr, "%s: dynllama JIT failed: %s\n", log_prefix, err.c_str());
        delete state; return nullptr;
    }

    fprintf(stderr, "%s: dynllama ready (%s)\n", log_prefix, lib.c_str());
    return state;
}

// Legacy: initialize from a single source string (dynllama.model_code.source key).
static inline dynllama_context_state * dynllama_ctx_init_single(
        const std::string & src,
        const std::string & arch,
        const std::string & path,
        int                 n_ctx,
        const char *        log_prefix) {
    auto * state = new dynllama_context_state();
    std::string err;

    if (!dynllama::host_load(path, state->model, &err)) {
        fprintf(stderr, "%s: dynllama host_load failed: %s\n", log_prefix, err.c_str());
        delete state; return nullptr;
    }
    dynllama::host_kv_alloc(state->model, n_ctx);

    dynllama::jit_options opt;
    opt.compiler     = dynllama_find_compiler();
    opt.include_dirs = dynllama_find_includes();
    const std::string lib = std::string("dynllama_ctx") + dynllama::jit_lib_ext();

    fprintf(stderr, "%s: dynllama JIT-compiling (legacy single-source) with %s ...\n",
            log_prefix, opt.compiler.c_str());
    if (!dynllama::jit_from_source(src, arch, lib, opt, state->jit, &err)) {
        fprintf(stderr, "%s: dynllama JIT failed: %s\n", log_prefix, err.c_str());
        delete state; return nullptr;
    }

    fprintf(stderr, "%s: dynllama ready (%s)\n", log_prefix, lib.c_str());
    return state;
}
