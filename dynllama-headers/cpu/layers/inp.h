//
// CPU input embedding layer: gather token embeddings via get_rows.
// Built as a dynllama_layer holding the embedding table.
//

#pragma once

#include "dynllama_layer.h"
#include "../ops/get_rows.h"

static inline void dynllama_inp_forward(
        const dynllama::dynllama_layer & L,
        float       * out,
        const int   * tokens,
        int           n_tokens,
        int           n_embd) {
    get_rows(out, L.weight_ptr("embd"), tokens, n_tokens, n_embd);
}

// Build an input-embedding layer holding the embedding table [n_embd, n_vocab].
static inline dynllama::dynllama_layer dynllama_make_inp(const float * embd_table) {
    dynllama::dynllama_layer L("inp");
    L.bind_weight("embd", embd_table);
    L.fwd = [](const dynllama::dynllama_layer & L, dynllama::dynllama_fwd_ctx & c) {
        dynllama_inp_forward(L, c.out, c.tokens, c.n_tokens, c.n_embd);
    };
    return L;
}
