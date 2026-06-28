//
// Model code: Qwen2 / Qwen2.5 forward pass.
//
// This is the code embedded in the .gguf and JIT-compiled by the host
// (llama.cpp). It implements exactly one ABI symbol, dynllama_eval(), and
// ONLY runs the forward pass over buffers the host already prepared. It does
// not open files, read metadata, or dequantize - that is the host's job.
//
// It is pure glue: include the prebuilt model descriptor and the DynLLaMA
// headers, then compose the standard ops/layers/blocks.
//
// Build (CPU backend):
//   g++ -std=c++17 -I dynllama-headers/cpu -I dynllama-headers model-code/qwen2_5.cpp ...
//
//   -I dynllama-headers/cpu  -> "dynllama.h" (ops/layers/blocks/pipeline)
//   -I dynllama-headers      -> "dynllama_abi.h", "prebuilt/qwen2_5/qwen2_5.h"
//

#include "dynllama.h"                    // CPU ops, layers, blocks, pipeline
#include "dynllama_abi.h"               // dynllama_run_ctx, entry-point contract
#include "prebuilt/qwen2_5/qwen2_5.h"   // qwen2 model identity / config helpers

extern "C" void dynllama_eval(dynllama_run_ctx * c) {
    if (c->kv) {
        // incremental decode using the host-owned KV cache
        llm_pipeline_forward_kv(
            c->logits, c->tokens, c->tok_embd, c->layers, c->n_layer,
            c->output_norm_w, c->output_w, c->kv,
            c->n_tokens, c->n_embd, c->n_ff, c->n_head, c->n_head_kv,
            c->n_embd_head, c->n_vocab, c->n_rot, c->rope_freq_base,
            c->n_past, c->act_type, c->rms_eps);
    } else {
        // stateless full-sequence pass
        llm_pipeline_forward(
            c->logits, c->tokens, c->tok_embd, c->layers, c->n_layer,
            c->output_norm_w, c->output_w,
            c->n_tokens, c->n_embd, c->n_ff, c->n_head, c->n_head_kv,
            c->n_embd_head, c->n_vocab, c->n_rot, c->rope_freq_base,
            c->act_type, c->rms_eps);
    }
}
