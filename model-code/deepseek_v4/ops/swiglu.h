#pragma once
// SwiGLU op: element-wise n[i] = (a[i] / (1 + exp(-a[i]))) * b[i]
//
// compile_swiglu() reads the body from the GGUF (dynllama.custom_op.swiglu.body),
// falls back to the hardcoded default, then JIT-compiles it.
// op_swiglu() dispatches to compiled or builtin.

#include "custom_op.h"
#include "types.h"

#include <cmath>

namespace ds4 {

static const char * SWIGLU_BODY = "n[i] = (a[i] / (1.0f + expf(-a[i]))) * b[i]";

inline binop_fn  g_swiglu        = nullptr;
inline void *    g_swiglu_handle = nullptr;

inline void swiglu_builtin(const float * a, const float * b, float * n, int count) {
    for (int i = 0; i < count; ++i)
        n[i] = (a[i] / (1.0f + std::exp(-a[i]))) * b[i];
}

inline bool compile_swiglu(const dynllama_meta * meta) {
    dynllama::jit_options opt;
    if (meta) {
        const char * cc = meta->get_str(meta->obj, "dynllama.jit.compiler", "");
        if (cc && *cc) opt.compiler = cc;
    }
    dynllama::custom_op_spec s =
        dynllama::custom_op_spec_from_meta(meta, "swiglu", SWIGLU_BODY);
    void * h = nullptr, * fn = nullptr;
    if (!dynllama::custom_op_build(s, opt, &h, &fn)) return false;
    if (g_swiglu_handle) dynllama::jit_unload_handle(g_swiglu_handle);
    g_swiglu_handle = h;
    g_swiglu        = reinterpret_cast<binop_fn>(fn);
    return true;
}

inline void op_swiglu(const float * gate, const float * up, float * n, int count) {
    (g_swiglu ? g_swiglu : swiglu_builtin)(gate, up, n, count);
}

} // namespace ds4
