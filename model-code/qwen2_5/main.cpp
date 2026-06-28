//
// Qwen2 / Qwen2.5 forward pass -- embedded in GGUF, JIT-compiled by llama.cpp.
// Entry point: dynllama_main (the single ABI symbol exported from this folder).
//
// Include-path layout (set by the JIT host at compile time):
//   -I dynllama-headers/cpu  ->  dynllama.h (ops / layers / pipeline)
//   -I dynllama-headers      ->  dynllama_abi.h, prebuilt/qwen2_5/qwen2_5.h
//

#include "dynllama.h"
#include "dynllama_abi.h"
#include "prebuilt/qwen2_5/qwen2_5.h"

extern "C" void dynllama_main(dynllama_run_ctx * c) {
    if (c->kv) {
        llm_pipeline_forward_kv(
            c->logits, c->tokens, c->tok_embd, c->layers, c->n_layer,
            c->output_norm_w, c->output_w, c->kv,
            c->n_tokens, c->n_embd, c->n_ff, c->n_head, c->n_head_kv,
            c->n_embd_head, c->n_vocab, c->n_rot, c->rope_freq_base,
            c->n_past, c->act_type, c->rms_eps);
    } else {
        llm_pipeline_forward(
            c->logits, c->tokens, c->tok_embd, c->layers, c->n_layer,
            c->output_norm_w, c->output_w,
            c->n_tokens, c->n_embd, c->n_ff, c->n_head, c->n_head_kv,
            c->n_embd_head, c->n_vocab, c->n_rot, c->rope_freq_base,
            c->act_type, c->rms_eps);
    }
}
