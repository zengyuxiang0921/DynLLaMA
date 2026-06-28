//
// DynLLaMA ABI -- the stable contract between the host (llama.cpp) and the
// JIT-compiled model code.
//
// Responsibility split:
//   * HOST (llama.cpp): opens the .gguf, reads metadata, dequantizes every
//     tensor to float, allocates the KV cache, JIT-compiles the model code,
//     fills a dynllama_run_ctx, and calls dynllama_eval().
//   * MODEL CODE (embedded in gguf, JIT-compiled): implements dynllama_eval()
//     and ONLY runs the forward pass over the buffers the host prepared. It
//     never loads files or dequantizes.
//
// The host locates exactly one exported symbol in the compiled model code:
//     extern "C" void dynllama_eval(dynllama_run_ctx *);
//
// This header is the only thing both sides must agree on. Keep it stable.
//

#ifndef DYNLLAMA_ABI_H
#define DYNLLAMA_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// activation selector (matches FFN op dispatch): 0=SiLU, 1=GELU, 2=ReLU
// gating, etc. are model-specific and resolved inside the model code.

// All weights for one decoder layer, already dequantized to float by the host.
// Null pointers mean "this model does not use that weight" (e.g. no bias).
typedef struct dynllama_layer_weights {
    const float * attn_norm;
    const float * wq; const float * bq;
    const float * wk; const float * bk;
    const float * wv; const float * bv;
    const float * wo; const float * wo_b;
    const float * ffn_norm;
    const float * ffn_up;
    const float * ffn_gate;
    const float * ffn_down;
} dynllama_layer_weights;

// KV cache, allocated and owned by the host, persisting across eval calls.
// Per layer il, K/V are laid out [n_ctx, n_embd_kv] (position-major):
//   K[il][pos][d] = k[(int64_t)il * n_embd_kv * n_ctx + (int64_t)pos * n_embd_kv + d]
// n_embd_kv = n_head_kv * n_embd_head.
typedef struct dynllama_kv_cache {
    int     n_layer;
    int     n_embd_kv;
    int     n_ctx;
    float * k;          // [n_layer * n_embd_kv * n_ctx]
    float * v;          // [n_layer * n_embd_kv * n_ctx]
} dynllama_kv_cache;

// A named op the host JIT-compiled from a custom-op body in the GGUF
// (see the custom_op scaffold). `fn` is cast by the model code to the concrete
// signature it expects (e.g. void(const float*,const float*,float*,int)).
typedef struct dynllama_op_entry {
    const char * name;
    void *       fn;
} dynllama_op_entry;

typedef struct dynllama_op_table {
    const dynllama_op_entry * entries;
    int                       n;
} dynllama_op_table;

// Read-only access to every key/value in the GGUF metadata, provided by the
// host so model code can query arbitrary keys (beyond the scalars the host
// already lifted into this ctx). `obj` is an opaque host handle; pass it back
// as the first argument of each accessor.
typedef struct dynllama_meta {
    void *       obj;
    const char * (*get_str)  (void * obj, const char * key, const char * def);
    int64_t      (*get_int)  (void * obj, const char * key, int64_t def);
    double       (*get_float)(void * obj, const char * key, double def);
    int          (*has)      (void * obj, const char * key);
} dynllama_meta;

// Read-only access to the model's dequantized (float32) tensors by name. Lets
// model code fetch arbitrary weights beyond the generic per-layer fields, which
// is what name-prefix layer construction (dynllama_make_layer) builds on.
typedef struct dynllama_tensors {
    void * obj;
    // Float data for `name`, or NULL if absent; *n_out (if non-NULL) gets the
    // element count.
    const float * (*get)(void * obj, const char * name, int64_t * n_out);
    // Invoke cb(user, name) for every tensor whose name starts with `prefix`.
    void (*for_each)(void * obj, const char * prefix, void * user,
                     void (*cb)(void * user, const char * name));
} dynllama_tensors;

// Everything the model code needs for one forward pass.
typedef struct dynllama_run_ctx {
    // configuration (auto-read from gguf metadata by the host)
    int   n_layer;
    int   n_embd;
    int   n_ff;
    int   n_head;
    int   n_head_kv;
    int   n_embd_head;
    int   n_vocab;
    int   n_rot;            // rotary dims (usually n_embd_head)
    int   act_type;
    float rms_eps;
    float rope_freq_base;   // RoPE theta base (e.g. 1000000 for qwen2)

    // global weights (dequantized float, owned by host)
    const float * tok_embd;       // [n_embd, n_vocab]
    const float * output_norm_w;  // [n_embd]
    const float * output_w;       // [n_vocab, n_embd]

    // per-layer weights, n_layer entries
    const dynllama_layer_weights * layers;

    // KV cache (may be NULL to request a stateless full-sequence pass)
    dynllama_kv_cache * kv;

    // input
    const int * tokens;   // [n_tokens]
    int         n_tokens;
    int         n_past;    // number of tokens already in the KV cache

    // output (host-allocated), token-major: logits[t * n_vocab + v]
    float * logits;        // [n_tokens * n_vocab]

    // custom ops the host compiled from dynllama.custom_op.* in the GGUF.
    // NULL if the file defines none; model code should fall back to builtins.
    const dynllama_op_table * ops;

    // full GGUF metadata accessor (NULL if the host did not provide one).
    const dynllama_meta * meta;

    // tensor-by-name accessor (NULL if the host did not provide one).
    const dynllama_tensors * tensors;
} dynllama_run_ctx;

// Resolve a custom op by name, or NULL if absent. Hand-rolled compare to keep
// this header free of <string.h>.
static inline void * dynllama_find_op(const dynllama_op_table * t, const char * name) {
    if (!t) return 0;
    for (int i = 0; i < t->n; ++i) {
        const char * a = t->entries[i].name;
        const char * b = name;
        while (*a && *a == *b) { ++a; ++b; }
        if (*a == 0 && *b == 0) return t->entries[i].fn;
    }
    return 0;
}

// The single entry point every model code module must export.
// Named dynllama_main and placed in main.cpp within the model-code folder.
void dynllama_main(dynllama_run_ctx * ctx);

// Type of the entry point, for the host's dlsym/GetProcAddress.
typedef void (*dynllama_main_fn)(dynllama_run_ctx * ctx);

#define DYNLLAMA_MAIN_SYMBOL "dynllama_main"

// Deprecated aliases kept for now to avoid breaking callers incrementally.
#define dynllama_eval      dynllama_main
#define dynllama_eval_fn   dynllama_main_fn
#define DYNLLAMA_EVAL_SYMBOL DYNLLAMA_MAIN_SYMBOL

#ifdef __cplusplus
} // extern "C"
#endif

#endif // DYNLLAMA_ABI_H
