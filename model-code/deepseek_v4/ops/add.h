#pragma once
// Add op: element-wise n[i] = a[i] + b[i]
//
// compile_add() reads the body from the GGUF (dynllama.custom_op.add.body),
// falls back to the hardcoded default, then JIT-compiles it into a shared
// library and stores the result in g_add.
// op_add() dispatches to the compiled version or to the builtin fallback.

#include "custom_op.h"   // dynllama-headers/cpu/custom_op.h via include path
#include "types.h"       // ds4::binop_fn (relative to this file's directory)

namespace ds4 {

static const char * ADD_BODY = "n[i] = a[i] + b[i]";

inline binop_fn  g_add        = nullptr;
inline void *    g_add_handle = nullptr;

inline void add_builtin(const float * a, const float * b, float * n, int count) {
    for (int i = 0; i < count; ++i) n[i] = a[i] + b[i];
}

// Compile the add op from the GGUF body (or the builtin default body).
// The compiler path is read from dynllama.jit.compiler in the metadata, or
// defaults to "g++".
inline bool compile_add(const dynllama_meta * meta) {
    dynllama::jit_options opt;
    if (meta) {
        const char * cc = meta->get_str(meta->obj, "dynllama.jit.compiler", "");
        if (cc && *cc) opt.compiler = cc;
    }
    dynllama::custom_op_spec s =
        dynllama::custom_op_spec_from_meta(meta, "add", ADD_BODY);
    void * h = nullptr, * fn = nullptr;
    if (!dynllama::custom_op_build(s, opt, &h, &fn)) return false;
    if (g_add_handle) dynllama::jit_unload_handle(g_add_handle);
    g_add_handle = h;
    g_add        = reinterpret_cast<binop_fn>(fn);
    return true;
}

inline void op_add(const float * a, const float * b, float * n, int count) {
    (g_add ? g_add : add_builtin)(a, b, n, count);
}

} // namespace ds4
