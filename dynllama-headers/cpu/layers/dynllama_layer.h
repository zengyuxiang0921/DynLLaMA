//
// dynllama_layer: one self-contained unit bundling
//   1. custom op specs   (custom_op_spec: body + signature, e.g. from GGUF)
//   2. compiled code      (JIT shared-lib handle + resolved function pointer)
//   3. weights            (named float buffers, owned or borrowed)
//
// Every predefined layer (inp / norm / qkv / attention / ffn) is built as a
// dynllama_layer via its make_* helper and run via its *_forward. A layer holds
// the weights its ops operate on and owns any JIT handles it compiled.
//
// Lives in dynllama-headers/cpu/ so both host and JIT-compiled model code can
// build layers at runtime. Build: -I dynllama-headers -I dynllama-headers/cpu
//

#ifndef DYNLLAMA_LAYER_H
#define DYNLLAMA_LAYER_H

#include "../custom_op.h"   // custom_op_spec, custom_op_build, custom_op_spec_from_meta

#include <string>
#include <vector>

namespace dynllama {

// A named weight tensor. Either owns its storage (`owned`) or borrows an
// external pointer (`ext`, e.g. a host-dequantized GGUF tensor). shape is
// row-major; empty shape means "flat vector".
struct layer_weight {
    std::string        name;
    std::vector<float> owned;
    const float *      ext = nullptr;
    std::vector<int>   shape;

    const float * data() const {
        return ext ? ext : (owned.empty() ? nullptr : owned.data());
    }
    size_t size() const {
        if (!shape.empty()) { size_t n = 1; for (int d : shape) n *= (size_t) d; return n; }
        return owned.size();
    }
};

// A custom op: its spec plus, once compiled, the owning JIT handle and the
// resolved symbol. fn is reinterpret_cast by the caller to the concrete
// signature (e.g. void(const float*,const float*,float*,int)).
struct layer_op {
    custom_op_spec spec;
    void *         handle = nullptr;   // JIT shared-lib handle (owned by layer)
    void *         fn     = nullptr;   // resolved op symbol
    bool compiled() const { return fn != nullptr; }
};

struct dynllama_layer;   // forward decl for the forward function pointer

// Uniform forward context: a superset of the I/O and dims every predefined
// layer needs. Each forward reads only the fields it uses. Token-major float
// buffers throughout.
struct dynllama_fwd_ctx {
    float       * out     = nullptr;   // primary output (norm/ffn/attn/inp)
    const float * in      = nullptr;   // primary input  (hidden state)
    float       * q       = nullptr;   // qkv out / attn in
    float       * k       = nullptr;
    float       * v       = nullptr;
    const int   * tokens  = nullptr;   // inp
    float       * k_cache = nullptr;   // attn: non-null selects the KV-cache path
    float       * v_cache = nullptr;
    int n_tokens = 0, n_embd = 0, n_ff = 0;
    int n_head = 0, n_head_kv = 0, n_embd_head = 0;
    int n_embd_q = 0, n_embd_kv = 0;
    int n_past = 0, n_ctx = 0, act_type = 0;
    float eps = 1e-5f;
    bool  causal = true;
};

// Standard forward signature for any layer (set by the layer's make_* helper).
typedef void (*dynllama_fwd_fn)(const dynllama_layer &, dynllama_fwd_ctx &);

struct dynllama_layer {
    std::string               name;
    std::vector<layer_op>     ops;
    std::vector<layer_weight> weights;
    dynllama_fwd_fn           fwd = nullptr;   // standard forward (set by make_*)

    dynllama_layer() = default;
    explicit dynllama_layer(std::string n) : name(std::move(n)) {}

    // Owns JIT handles: non-copyable, movable.
    dynllama_layer(const dynllama_layer &)             = delete;
    dynllama_layer & operator=(const dynllama_layer &) = delete;
    dynllama_layer(dynllama_layer && o) noexcept { steal(o); name = std::move(o.name); }
    dynllama_layer & operator=(dynllama_layer && o) noexcept {
        if (this != &o) { unload(); name = std::move(o.name); steal(o); }
        return *this;
    }
    ~dynllama_layer() { unload(); }

    // ---- weights ----
    // Borrow an external weight pointer (zero-copy). p may be null (e.g. no bias).
    layer_weight & bind_weight(const std::string & wn, const float * p,
                               std::vector<int> shape = {}) {
        weights.push_back(layer_weight{wn, {}, p, std::move(shape)});
        return weights.back();
    }
    // Own a weight buffer (copied into the layer).
    layer_weight & add_weight(const std::string & wn, std::vector<float> data,
                              std::vector<int> shape = {}) {
        weights.push_back(layer_weight{wn, std::move(data), nullptr, std::move(shape)});
        return weights.back();
    }
    layer_weight * weight(const std::string & wn) {
        for (auto & w : weights) if (w.name == wn) return &w;
        return nullptr;
    }
    const layer_weight * weight(const std::string & wn) const {
        for (auto & w : weights) if (w.name == wn) return &w;
        return nullptr;
    }
    const float * weight_ptr(const std::string & wn) const {
        const layer_weight * w = weight(wn);
        return w ? w->data() : nullptr;
    }

    // ---- ops ----
    layer_op & add_op(const custom_op_spec & spec) {
        ops.push_back(layer_op{spec, nullptr, nullptr});
        return ops.back();
    }
    layer_op & add_op_from_meta(const dynllama_meta * meta, const char * op_name,
                                const char * default_body = nullptr) {
        return add_op(custom_op_spec_from_meta(meta, op_name, default_body));
    }

    // Compile every registered op that is not compiled yet. Returns false on the
    // first failure (err set); already-compiled ops are kept. Optional: forwards
    // fall back to builtins when an op is left uncompiled.
    bool compile(const jit_options & opt, std::string * err = nullptr) {
        for (auto & op : ops) {
            if (op.compiled() || op.spec.body.empty()) continue;
            void * h = nullptr, * fn = nullptr;
            if (!custom_op_build(op.spec, opt, &h, &fn, err)) return false;
            op.handle = h;
            op.fn     = fn;
        }
        return true;
    }

    void * op_fn(const std::string & on) const {
        for (const auto & op : ops) if (op.spec.name == on) return op.fn;
        return nullptr;
    }
    template <typename Fn>
    Fn op(const std::string & on) const { return reinterpret_cast<Fn>(op_fn(on)); }

    void unload() {
        for (auto & op : ops) {
            if (op.handle) jit_unload_handle(op.handle);
            op.handle = nullptr;
            op.fn     = nullptr;
        }
    }

private:
    void steal(dynllama_layer & o) {
        ops     = std::move(o.ops);
        weights = std::move(o.weights);
        fwd     = o.fwd;
        o.ops.clear();   // moved-from must not free handles we now own
        o.fwd = nullptr;
    }
};

// Standard layer forward: dispatch through the layer's own forward adapter.
static inline void dynllama_forward(const dynllama_layer & L, dynllama_fwd_ctx & c) {
    if (L.fwd) L.fwd(L, c);
}

// Build a layer that borrows every tensor whose name starts with `prefix` + '.'.
// The bound weight key is the remainder after that prefix, so
//   dynllama_make_layer("blk.0.ffn", T)
// binds "blk.0.ffn.gate.weight" as "gate.weight", "blk.0.ffn.experts.0.w1.weight"
// as "experts.0.w1.weight", etc. -- give it a prefix and it finds the rest.
static inline dynllama_layer dynllama_make_layer(const std::string & prefix,
                                                 const dynllama_tensors * T) {
    dynllama_layer L(prefix);
    if (!T || !T->for_each || !T->get) return L;

    struct collect_ctx {
        dynllama_layer *         L;
        const dynllama_tensors * T;
        std::string              base;   // prefix + '.'
    } cx{ &L, T, prefix + "." };

    T->for_each(T->obj, prefix.c_str(), &cx, [](void * user, const char * name) {
        auto * c = static_cast<collect_ctx *>(user);
        const std::string n = name;
        // exact "<prefix>." boundary so "blk.0.ffn" does not swallow "blk.0.ffn_norm"
        if (n.size() <= c->base.size() || n.compare(0, c->base.size(), c->base) != 0)
            return;
        c->L->bind_weight(n.substr(c->base.size()), c->T->get(c->T->obj, name, nullptr));
    });
    return L;
}

} // namespace dynllama

#endif // DYNLLAMA_LAYER_H
