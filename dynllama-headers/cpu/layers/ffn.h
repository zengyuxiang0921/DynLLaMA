//
// CPU FFN layer (token-major), SwiGLU-style:
//   out = down( act(gate . x) * (up . x) )
// Built as a dynllama_layer holding {up, gate, down} and a fused activation*gate
// custom op ("ffn_act"). If the op is compiled it is used; otherwise the forward
// falls back to the builtin activation + elementwise multiply.
//

#pragma once

#include "dynllama_layer.h"
#include <string>
#include <vector>

#include "../ops/linear.h"
#include "../ops/silu.h"
#include "../ops/gelu.h"
#include "../ops/relu.h"
#include "../ops/mul.h"

typedef void (*dynllama_binop_fn)(const float *, const float *, float *, int);

// Default fused SwiGLU body for "ffn_act": n[i] = act(a[i]) * b[i].
static inline const char * dynllama_ffn_act_body(int act_type) {
    if (act_type == DYNLLAMA_ACT_GELU)
        return "n[i] = (0.5f*a[i]*(1.0f+tanhf(0.7978845608028654f*(a[i]+0.044715f*a[i]*a[i]*a[i])))) * b[i]";
    if (act_type == DYNLLAMA_ACT_RELU)
        return "n[i] = (a[i] > 0.0f ? a[i] : 0.0f) * b[i]";
    return "n[i] = (a[i] / (1.0f + expf(-a[i]))) * b[i]";   // SiLU (default)
}

static inline void dynllama_ffn_forward(
        const dynllama::dynllama_layer & L,
        float       * out,
        const float * hidden,
        int           n_tokens,
        int           n_embd,
        int           n_ff,
        int           act_type) {

    const int ne_ff = n_ff * n_tokens;
    std::vector<float> up_buf(ne_ff);
    std::vector<float> gate_buf(ne_ff);

    linear(hidden, L.weight_ptr("up"),   nullptr, up_buf.data(),   n_tokens, n_ff, n_embd);
    linear(hidden, L.weight_ptr("gate"), nullptr, gate_buf.data(), n_tokens, n_ff, n_embd);

    if (auto fn = L.op<dynllama_binop_fn>("ffn_act")) {
        fn(gate_buf.data(), up_buf.data(), gate_buf.data(), ne_ff);   // compiled op
    } else {                                                          // builtin fallback
        if (act_type == DYNLLAMA_ACT_SILU)      silu(gate_buf.data(), gate_buf.data(), ne_ff);
        else if (act_type == DYNLLAMA_ACT_GELU) gelu(gate_buf.data(), gate_buf.data(), ne_ff);
        else                                    relu(gate_buf.data(), gate_buf.data(), ne_ff);
        mul_ew(gate_buf.data(), up_buf.data(), gate_buf.data(), ne_ff);
    }

    linear(gate_buf.data(), L.weight_ptr("down"), nullptr, out, n_tokens, n_embd, n_ff);
}

// Build an FFN layer holding {up, gate, down} and the fused activation op spec.
// act_type selects both the default op body and the builtin fallback.
static inline dynllama::dynllama_layer dynllama_make_ffn(
        const float * up, const float * gate, const float * down, int act_type) {
    dynllama::dynllama_layer L("ffn");
    L.bind_weight("up", up);
    L.bind_weight("gate", gate);
    L.bind_weight("down", down);

    dynllama::custom_op_spec spec;
    spec.name = "ffn_act";
    spec.body = dynllama_ffn_act_body(act_type);
    L.add_op(spec);   // left uncompiled: forward falls back to builtin until compile()

    L.fwd = [](const dynllama::dynllama_layer & L, dynllama::dynllama_fwd_ctx & c) {
        dynllama_ffn_forward(L, c.out, c.in, c.n_tokens, c.n_embd, c.n_ff, c.act_type);
    };
    return L;
}
