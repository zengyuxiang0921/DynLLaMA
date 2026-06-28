//
// DynLLaMA -- GGUF metadata reader.
//
// Dependency-free, header-only parser for the GGUF key-value metadata and
// tensor info sections. Lets model code auto-discover architecture and
// hyperparameters from a .gguf file without linking the full ggml runtime.
//
// Only the metadata + tensor-info headers are read; tensor data is never
// touched, so opening a multi-GB model is cheap.
//
// Format reference: dynllama-headers/cpu/ggml/gguf.h
//

#ifndef DYNLLAMA_META_H
#define DYNLLAMA_META_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace dynllama {

// Mirrors enum gguf_type from gguf.h
enum meta_type : uint32_t {
    META_UINT8   = 0,  META_INT8    = 1,
    META_UINT16  = 2,  META_INT16   = 3,
    META_UINT32  = 4,  META_INT32   = 5,
    META_FLOAT32 = 6,  META_BOOL    = 7,
    META_STRING  = 8,  META_ARRAY   = 9,
    META_UINT64  = 10, META_INT64   = 11,
    META_FLOAT64 = 12,
};

struct meta_value {
    meta_type   type = META_UINT32;
    uint64_t    u    = 0;   // raw integer / bool bits
    double      d    = 0.0; // float / double value
    std::string s;          // string value
};

struct meta_tensor {
    std::vector<int64_t> shape;  // ne[], fastest-moving dim first (ggml order)
    uint32_t             type = 0;
    uint64_t             offset = 0;
};

class gguf_meta {
public:
    bool open(const std::string & path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { err_ = "cannot open file: " + path; return false; }

        char magic[4];
        f.read(magic, 4);
        if (std::memcmp(magic, "GGUF", 4) != 0) { err_ = "not a GGUF file"; return false; }

        if (!rd(f, version_)) return false;
        uint64_t n_tensors = 0, n_kv = 0;
        if (!rd(f, n_tensors) || !rd(f, n_kv)) return false;

        for (uint64_t i = 0; i < n_kv; i++) {
            std::string key;
            if (!rd_str(f, key)) return false;

            uint32_t vt = 0;
            if (!rd(f, vt)) return false;

            meta_value v;
            v.type = (meta_type) vt;
            if (vt == META_ARRAY) {
                // String arrays (tokenizer vocab / merges) are stored; numeric
                // arrays are skipped (only their presence is recorded).
                if (!read_array(f, key)) return false;
                arrays_.insert(key);
                continue;
            }
            if (!rd_scalar(f, v)) return false;
            kv_[key] = std::move(v);
        }

        for (uint64_t i = 0; i < n_tensors; i++) {
            std::string name;
            if (!rd_str(f, name)) return false;
            uint32_t n_dims = 0;
            if (!rd(f, n_dims)) return false;
            meta_tensor t;
            t.shape.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; d++) {
                if (!rd(f, t.shape[d])) return false;
            }
            if (!rd(f, t.type) || !rd(f, t.offset)) return false;
            tensors_[name] = std::move(t);
        }

        // tensor data begins after the info section, aligned up to `alignment`
        alignment_ = get_u32("general.alignment", 32);
        if (alignment_ == 0) alignment_ = 32;
        const uint64_t pos = (uint64_t) f.tellg();
        data_offset_ = ((pos + alignment_ - 1) / alignment_) * alignment_;

        return true;
    }

    uint32_t version() const { return version_; }

    // byte offset of the tensor-data blob from the start of the file
    uint64_t data_offset() const { return data_offset_; }
    size_t   alignment()   const { return alignment_; }

    // total number of elements in a tensor (product of shape)
    static int64_t tensor_nelements(const meta_tensor & t) {
        int64_t n = 1;
        for (int64_t d : t.shape) n *= d;
        return n;
    }

    bool has(const std::string & key) const {
        return kv_.count(key) || arrays_.count(key);
    }

    std::string get_str(const std::string & key, const std::string & def = "") const {
        auto it = kv_.find(key);
        return (it != kv_.end() && it->second.type == META_STRING) ? it->second.s : def;
    }

    // Like get_str but returns a pointer into the stored string, stable for the
    // life of this gguf_meta. Suitable for C-callback metadata access.
    const char * get_cstr(const std::string & key, const char * def = "") const {
        auto it = kv_.find(key);
        return (it != kv_.end() && it->second.type == META_STRING) ? it->second.s.c_str() : def;
    }

    uint64_t get_u64(const std::string & key, uint64_t def = 0) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) return def;
        const meta_value & v = it->second;
        if (v.type == META_FLOAT32 || v.type == META_FLOAT64) return (uint64_t) v.d;
        return v.u;
    }

    int64_t  get_i64(const std::string & key, int64_t  def = 0) const {
        return (int64_t) get_u64(key, (uint64_t) def);
    }
    uint32_t get_u32(const std::string & key, uint32_t def = 0) const {
        return (uint32_t) get_u64(key, def);
    }
    int32_t  get_i32(const std::string & key, int32_t  def = 0) const {
        return (int32_t)  get_u64(key, (uint64_t) def);
    }
    bool     get_bool(const std::string & key, bool def = false) const {
        return get_u64(key, def ? 1 : 0) != 0;
    }

    double get_f64(const std::string & key, double def = 0.0) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) return def;
        const meta_value & v = it->second;
        if (v.type == META_FLOAT32 || v.type == META_FLOAT64) return v.d;
        return (double) v.u;
    }
    float get_f32(const std::string & key, float def = 0.0f) const {
        return (float) get_f64(key, def);
    }

    // architecture string from general.architecture
    std::string arch() const { return get_str("general.architecture"); }

    // build an arch-prefixed key, e.g. key("block_count") -> "qwen2.block_count"
    std::string key(const std::string & suffix) const {
        return arch() + "." + suffix;
    }

    // string array value (e.g. tokenizer.ggml.tokens / merges), or null
    const std::vector<std::string> * get_str_array(const std::string & key) const {
        auto it = str_arrays_.find(key);
        return it == str_arrays_.end() ? nullptr : &it->second;
    }

    const meta_tensor * tensor(const std::string & name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    const std::unordered_map<std::string, meta_tensor> & tensors() const { return tensors_; }
    const std::unordered_map<std::string, meta_value>  & kv()      const { return kv_; }

    const std::string & error() const { return err_; }

private:
    template <typename T>
    static bool rd(std::ifstream & f, T & out) {
        f.read(reinterpret_cast<char *>(&out), sizeof(T));
        return (bool) f;
    }

    static bool rd_str(std::ifstream & f, std::string & out) {
        uint64_t len = 0;
        if (!rd(f, len)) return false;
        out.resize(len);
        if (len) f.read(&out[0], (std::streamsize) len);
        return (bool) f;
    }

    static size_t type_size(uint32_t t) {
        switch (t) {
            case META_UINT8: case META_INT8: case META_BOOL:   return 1;
            case META_UINT16: case META_INT16:                 return 2;
            case META_UINT32: case META_INT32: case META_FLOAT32: return 4;
            case META_UINT64: case META_INT64: case META_FLOAT64: return 8;
            default: return 0;
        }
    }

    static bool rd_scalar(std::ifstream & f, meta_value & v) {
        if (v.type == META_STRING) {
            return rd_str(f, v.s);
        }
        switch (v.type) {
            case META_UINT8:  { uint8_t  x; if (!rd(f,x)) return false; v.u = x; break; }
            case META_INT8:   { int8_t   x; if (!rd(f,x)) return false; v.u = (uint64_t)(int64_t) x; break; }
            case META_UINT16: { uint16_t x; if (!rd(f,x)) return false; v.u = x; break; }
            case META_INT16:  { int16_t  x; if (!rd(f,x)) return false; v.u = (uint64_t)(int64_t) x; break; }
            case META_UINT32: { uint32_t x; if (!rd(f,x)) return false; v.u = x; break; }
            case META_INT32:  { int32_t  x; if (!rd(f,x)) return false; v.u = (uint64_t)(int64_t) x; break; }
            case META_BOOL:   { uint8_t  x; if (!rd(f,x)) return false; v.u = x; break; }
            case META_UINT64: { uint64_t x; if (!rd(f,x)) return false; v.u = x; break; }
            case META_INT64:  { int64_t  x; if (!rd(f,x)) return false; v.u = (uint64_t) x; break; }
            case META_FLOAT32:{ float    x; if (!rd(f,x)) return false; v.d = x; break; }
            case META_FLOAT64:{ double   x; if (!rd(f,x)) return false; v.d = x; break; }
            default: return false;
        }
        return true;
    }

    bool read_array(std::ifstream & f, const std::string & key) {
        uint32_t elem_type = 0;
        uint64_t count = 0;
        if (!rd(f, elem_type) || !rd(f, count)) return false;
        if (elem_type == META_STRING) {
            std::vector<std::string> & arr = str_arrays_[key];
            arr.resize(count);
            for (uint64_t i = 0; i < count; i++) {
                if (!rd_str(f, arr[i])) return false;
            }
        } else {
            size_t sz = type_size(elem_type);
            if (sz == 0) return false;
            f.seekg((std::streamoff)(sz * count), std::ios::cur);
        }
        return (bool) f;
    }

    uint32_t version_ = 0;
    uint64_t data_offset_ = 0;
    size_t   alignment_   = 32;
    std::unordered_map<std::string, meta_value>  kv_;
    std::unordered_map<std::string, meta_tensor> tensors_;
    std::unordered_map<std::string, std::vector<std::string>> str_arrays_;
    std::string err_;

    // tracks keys whose value was an array (skipped), so has() still reports them
    struct str_set {
        std::unordered_map<std::string, bool> m;
        void insert(const std::string & k) { m[k] = true; }
        size_t count(const std::string & k) const { return m.count(k); }
    } arrays_;
};

//
// Standard transformer hyperparameters, read from arch-prefixed metadata keys.
// Field <-> key mapping follows llama.cpp's GGUF conventions.
//
struct hparams {
    std::string arch;
    uint32_t n_layer        = 0;   // {arch}.block_count
    uint32_t n_ctx_train    = 0;   // {arch}.context_length
    uint32_t n_embd         = 0;   // {arch}.embedding_length
    uint32_t n_ff           = 0;   // {arch}.feed_forward_length
    uint32_t n_head         = 0;   // {arch}.attention.head_count
    uint32_t n_head_kv      = 0;   // {arch}.attention.head_count_kv (default n_head)
    float    rope_freq_base = 10000.0f; // {arch}.rope.freq_base
    float    rms_eps        = 1e-5f;     // {arch}.attention.layer_norm_rms_epsilon

    // derived
    uint32_t n_embd_head    = 0;   // n_embd / n_head
    uint32_t n_embd_k_gqa   = 0;   // n_embd_head * n_head_kv
    uint32_t n_vocab        = 0;   // from token_embd.weight shape, if present
};

// Auto-read the standard hyperparameters using the file's own architecture prefix.
static inline bool load_hparams(const gguf_meta & m, hparams & hp) {
    hp.arch = m.arch();
    if (hp.arch.empty()) return false;

    hp.n_layer     = m.get_u32(m.key("block_count"));
    hp.n_ctx_train = m.get_u32(m.key("context_length"));
    hp.n_embd      = m.get_u32(m.key("embedding_length"));
    hp.n_ff        = m.get_u32(m.key("feed_forward_length"));
    hp.n_head      = m.get_u32(m.key("attention.head_count"));
    hp.n_head_kv   = m.get_u32(m.key("attention.head_count_kv"), hp.n_head);
    hp.rope_freq_base = m.get_f32(m.key("rope.freq_base"), 10000.0f);
    hp.rms_eps        = m.get_f32(m.key("attention.layer_norm_rms_epsilon"), 1e-5f);

    hp.n_embd_head  = hp.n_head ? hp.n_embd / hp.n_head : 0;
    hp.n_embd_k_gqa = hp.n_embd_head * hp.n_head_kv;

    if (const meta_tensor * t = m.tensor("token_embd.weight")) {
        if (t->shape.size() >= 2) hp.n_vocab = (uint32_t) t->shape[1];
    } else if (const meta_tensor * o = m.tensor("output.weight")) {
        if (o->shape.size() >= 2) hp.n_vocab = (uint32_t) o->shape[1];
    }

    return hp.n_embd != 0 && hp.n_layer != 0 && hp.n_head != 0;
}

} // namespace dynllama

#endif // DYNLLAMA_META_H
