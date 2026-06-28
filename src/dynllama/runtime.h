//
// DynLLaMA host runtime (the "llama.cpp side").
//
// This is the data-preparation half of the system. It:
//   * opens a .gguf and auto-reads hyperparameters,
//   * dequantizes every tensor to float32 (using the dequantize op),
//   * assembles the per-layer / global weight pointers the ABI expects,
//   * allocates the KV cache,
//   * builds a dynllama_run_ctx and hands it to the model code's dynllama_eval.
//
// The JIT-compiled model code never does any of this - it only runs the model.
//
// Header-only for now so it builds standalone with any C++17 compiler; it will
// later fold into the llama.cpp host build.
//
// Build: -I dynllama-headers/cpu -I dynllama-headers -I src
//

#ifndef DYNLLAMA_RUNTIME_H
#define DYNLLAMA_RUNTIME_H

#include "meta.h"              // dynllama::gguf_meta, hparams (-I dynllama-headers/cpu)
#include "ops/dequantize.h"   // dequantize_row, dequantize_type_size
#include "dynllama_abi.h"     // dynllama_run_ctx, dynllama_layer_weights, kv (-I dynllama-headers)

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace dynllama {

struct host_model {
    hparams hp;
    int     act_type = 0; // SiLU for qwen2/llama

    // GGUF metadata kept alive so the ABI metadata accessor stays valid
    gguf_meta       meta;
    dynllama_meta   meta_iface{};
    dynllama_tensors tensors_iface{};

    // dequantized tensor storage, keyed by gguf tensor name
    std::unordered_map<std::string, std::vector<float>> weights;

    // assembled ABI views
    std::vector<dynllama_layer_weights> layers;
    const float * tok_embd      = nullptr;
    const float * output_norm_w = nullptr;
    const float * output_w      = nullptr;

    // KV cache storage (owned)
    std::vector<float> kv_k, kv_v;
    dynllama_kv_cache  kv{};

    const float * get(const std::string & name) const {
        auto it = weights.find(name);
        return it == weights.end() ? nullptr : it->second.data();
    }
};

// ABI metadata accessors: obj is a gguf_meta*.
static inline const char * host_meta_get_str(void * obj, const char * key, const char * def) {
    return static_cast<gguf_meta *>(obj)->get_cstr(key, def ? def : "");
}
static inline int64_t host_meta_get_int(void * obj, const char * key, int64_t def) {
    return static_cast<gguf_meta *>(obj)->get_i64(key, def);
}
static inline double host_meta_get_float(void * obj, const char * key, double def) {
    return static_cast<gguf_meta *>(obj)->get_f64(key, def);
}
static inline int host_meta_has(void * obj, const char * key) {
    return static_cast<gguf_meta *>(obj)->has(key) ? 1 : 0;
}

static inline void host_meta_bind(host_model & model) {
    model.meta_iface.obj       = &model.meta;
    model.meta_iface.get_str   = host_meta_get_str;
    model.meta_iface.get_int   = host_meta_get_int;
    model.meta_iface.get_float = host_meta_get_float;
    model.meta_iface.has       = host_meta_has;
}

// ABI tensor accessors: obj is a host_model*.
static inline const float * host_tensor_get(void * obj, const char * name, int64_t * n_out) {
    auto * m = static_cast<host_model *>(obj);
    auto it = m->weights.find(name);
    if (it == m->weights.end()) { if (n_out) *n_out = 0; return nullptr; }
    if (n_out) *n_out = (int64_t) it->second.size();
    return it->second.data();
}
static inline void host_tensor_for_each(void * obj, const char * prefix, void * user,
                                        void (*cb)(void *, const char *)) {
    auto * m = static_cast<host_model *>(obj);
    const std::string p = prefix ? prefix : "";
    for (const auto & kv : m->weights)
        if (kv.first.size() >= p.size() && kv.first.compare(0, p.size(), p) == 0)
            cb(user, kv.first.c_str());
}
static inline void host_tensors_bind(host_model & model) {
    model.tensors_iface.obj      = &model;
    model.tensors_iface.get      = host_tensor_get;
    model.tensors_iface.for_each = host_tensor_for_each;
}

// Dequantize every tensor in the file to float32.
static inline bool host_load_weights(const std::string & path,
                                     const gguf_meta & m,
                                     host_model & model,
                                     std::string * err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { if (err) *err = "cannot reopen file: " + path; return false; }

    const uint64_t base = m.data_offset();
    std::vector<uint8_t> qbuf;

    for (const auto & kvp : m.tensors()) {
        const std::string  & name = kvp.first;
        const meta_tensor  & t    = kvp.second;

        const int64_t nelem  = gguf_meta::tensor_nelements(t);
        const int64_t qbytes = dequantize_type_size((int) t.type, nelem);
        if (qbytes == 0) {
            if (err) *err = "unsupported quant type " + std::to_string(t.type) +
                            " for tensor '" + name + "'";
            return false;
        }

        qbuf.resize((size_t) qbytes);
        f.seekg((std::streamoff)(base + t.offset), std::ios::beg);
        f.read((char *) qbuf.data(), qbytes);
        if (!f) { if (err) *err = "short read for tensor '" + name + "'"; return false; }

        std::vector<float> out((size_t) nelem);
        if (!dequantize_row((int) t.type, qbuf.data(), out.data(), nelem)) {
            if (err) *err = "dequantize failed for '" + name + "'";
            return false;
        }
        model.weights[name] = std::move(out);
    }
    return true;
}

// Assemble the global + per-layer weight pointers for a llama/qwen2-style model.
static inline bool host_assemble(host_model & model, std::string * err) {
    const hparams & hp = model.hp;

    model.tok_embd      = model.get("token_embd.weight");
    model.output_norm_w = model.get("output_norm.weight");
    model.output_w      = model.get("output.weight");
    if (!model.output_w) model.output_w = model.tok_embd; // tied embeddings

    // Generic qwen/llama-style assembly. Missing tensors are left NULL (not an
    // error): models with a non-standard layout (e.g. DeepSeek V4) read their
    // weights by name through the dynllama_tensors accessor instead.
    model.layers.assign(hp.n_layer, dynllama_layer_weights{});
    for (uint32_t i = 0; i < hp.n_layer; i++) {
        const std::string p = "blk." + std::to_string(i) + ".";
        dynllama_layer_weights & w = model.layers[i];
        w.attn_norm = model.get(p + "attn_norm.weight");
        w.wq = model.get(p + "attn_q.weight");      w.bq = model.get(p + "attn_q.bias");
        w.wk = model.get(p + "attn_k.weight");      w.bk = model.get(p + "attn_k.bias");
        w.wv = model.get(p + "attn_v.weight");      w.bv = model.get(p + "attn_v.bias");
        w.wo = model.get(p + "attn_output.weight"); w.wo_b = model.get(p + "attn_output.bias");
        w.ffn_norm = model.get(p + "ffn_norm.weight");
        w.ffn_up   = model.get(p + "ffn_up.weight");
        w.ffn_gate = model.get(p + "ffn_gate.weight");
        w.ffn_down = model.get(p + "ffn_down.weight");
    }
    return true;
}

// Full host-side load: metadata -> hparams -> dequantize -> assemble.
static inline bool host_load(const std::string & path, host_model & model, std::string * err = nullptr) {
    if (!model.meta.open(path)) { if (err) *err = model.meta.error(); return false; }
    if (!load_hparams(model.meta, model.hp)) { if (err) *err = "failed to read hparams"; return false; }
    model.act_type = 0; // SiLU
    host_meta_bind(model);
    host_tensors_bind(model);
    if (!host_load_weights(path, model.meta, model, err)) return false;
    if (!host_assemble(model, err)) return false;
    return true;
}

// Allocate the KV cache for n_ctx positions.
static inline void host_kv_alloc(host_model & model, int n_ctx) {
    const hparams & hp = model.hp;
    const int n_embd_kv = hp.n_head_kv * hp.n_embd_head;
    const size_t per_layer = (size_t) n_embd_kv * n_ctx;
    model.kv_k.assign(per_layer * hp.n_layer, 0.0f);
    model.kv_v.assign(per_layer * hp.n_layer, 0.0f);
    model.kv.n_layer   = (int) hp.n_layer;
    model.kv.n_embd_kv = n_embd_kv;
    model.kv.n_ctx     = n_ctx;
    model.kv.k         = model.kv_k.data();
    model.kv.v         = model.kv_v.data();
}

// Build a run context for a batch of tokens. Pass use_kv=true and a valid
// n_past after host_kv_alloc() for incremental decoding.
static inline dynllama_run_ctx host_make_ctx(
        host_model & model,
        const int * tokens, int n_tokens, int n_past,
        float * logits, bool use_kv) {
    const hparams & hp = model.hp;
    dynllama_run_ctx c{};
    c.n_layer     = (int) hp.n_layer;
    c.n_embd      = (int) hp.n_embd;
    c.n_ff        = (int) hp.n_ff;
    c.n_head      = (int) hp.n_head;
    c.n_head_kv   = (int) hp.n_head_kv;
    c.n_embd_head = (int) hp.n_embd_head;
    c.n_vocab     = (int) hp.n_vocab;
    c.n_rot       = (int) hp.n_embd_head;   // qwen2: full-head rotary
    c.act_type    = model.act_type;
    c.rms_eps     = hp.rms_eps;
    c.rope_freq_base = hp.rope_freq_base;
    c.tok_embd      = model.tok_embd;
    c.output_norm_w = model.output_norm_w;
    c.output_w      = model.output_w;
    c.layers        = model.layers.data();
    c.kv            = use_kv ? &model.kv : nullptr;
    c.tokens        = tokens;
    c.n_tokens      = n_tokens;
    c.n_past        = n_past;
    c.logits        = logits;
    c.meta          = model.meta_iface.obj ? &model.meta_iface : nullptr;
    c.tensors       = model.tensors_iface.obj ? &model.tensors_iface : nullptr;
    return c;
}

} // namespace dynllama

#endif // DYNLLAMA_RUNTIME_H
