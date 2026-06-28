//
// DynLLaMA CPU custom ops: take an op body stored in the GGUF, wrap it in an
// extern "C" scaffold with an elementwise loop, JIT-compile it, and resolve
// it as a callable function pointer.
//
// Lives in dynllama-headers/cpu/ so it is accessible to both host code and
// JIT-compiled model code. Model code can call custom_op_build() from within
// a running DLL to compile ops at runtime from GGUF metadata.
//
// GGUF key scheme:
//   dynllama.custom_op.<name>.body         (required) op body expression
//   dynllama.custom_op.<name>.args         (optional) parameter list
//   dynllama.custom_op.<name>.ret          (optional) return type (default "void")
//   dynllama.custom_op.<name>.elementwise  (optional bool) wrap body in a loop
//   dynllama.custom_op.<name>.index        (optional) loop index name
//   dynllama.custom_op.<name>.count_arg    (optional) length arg name
//   dynllama.custom_op.<name>.includes     (optional) ';'-separated extra headers
//
// Two accessors to read specs:
//   custom_op_spec_from_meta  -- uses dynllama_meta C callbacks (works in model code)
//   custom_op_spec_from_gguf  -- uses gguf_meta C++ class (host side only)
//

#ifndef DYNLLAMA_CUSTOM_OP_H
#define DYNLLAMA_CUSTOM_OP_H

#include "dynllama_abi.h"   // dynllama_meta, dynllama_op_entry, dynllama_op_table
#include "meta.h"           // gguf_meta
#include "jit.h"            // jit_options, jit_compile, jit_load_symbol, jit_lib_ext

#include <fstream>
#include <string>
#include <vector>

namespace dynllama {

static const char * DYNLLAMA_CUSTOM_OP_PFX  = "dynllama.custom_op.";
static const char * DYNLLAMA_CUSTOM_OPS_KEY = "dynllama.custom_ops";

typedef void (*custom_binop_fn)(const float * a, const float * b, float * n, int count);

struct custom_op_spec {
    std::string name        = "dynllama_custom_op";
    std::string ret         = "void";
    std::string args        = "const float * a, const float * b, float * n, int count";
    std::string body;
    bool        elementwise = true;
    std::string index       = "i";
    std::string count_arg   = "count";
    std::vector<std::string> includes;
};

// Wrap the body in a compilable extern "C" CPU function.
static inline std::string custom_op_source(const custom_op_spec & s) {
    std::string out;
    out += "#include <cmath>\n";
    out += "#include <cstdint>\n";
    for (const auto & inc : s.includes) out += "#include " + inc + "\n";

    out += "extern \"C\" " + s.ret + " " + s.name + "(" + s.args + ") {\n";
    if (s.elementwise) {
        out += "    for (int " + s.index + " = 0; " + s.index + " < " + s.count_arg +
               "; ++" + s.index + ") {\n";
        out += "        " + s.body + ";\n";
        out += "    }\n";
    } else {
        out += "    " + s.body + "\n";
    }
    out += "}\n";
    return out;
}

static inline std::string custom_op_tmp(const std::string & name, const char * ext) {
#ifdef _WIN32
    char td[MAX_PATH];
    if (!GetTempPathA(MAX_PATH, td)) { td[0] = '.'; td[1] = '\\'; td[2] = '\0'; }
    return std::string(td) + "dynllama_op_" + name + ext;
#else
    return "/tmp/dynllama_op_" + name + ext;
#endif
}

// Generate source, compile to a shared library, and resolve the op symbol.
// On success sets *handle_out (unload with jit_unload_handle) and *fn_out.
static inline bool custom_op_build(const custom_op_spec & spec,
                                   const jit_options & opt,
                                   void ** handle_out,
                                   void ** fn_out,
                                   std::string * err = nullptr) {
    const std::string src = custom_op_source(spec);
    const std::string tmp = custom_op_tmp(spec.name, ".cpp");
    {
        std::ofstream f(tmp);
        if (!f) { if (err) *err = "cannot write temp source: " + tmp; return false; }
        f << src;
    }
    const std::string lib = custom_op_tmp(spec.name, jit_lib_ext());
    if (!jit_compile(tmp, lib, opt, err)) return false;
    void * handle = nullptr;
    void * fn = jit_load_symbol(lib, spec.name, &handle, err);
    if (!fn) return false;
    if (handle_out) *handle_out = handle;
    if (fn_out)     *fn_out     = fn;
    return true;
}

// Build a spec from the C-callback dynllama_meta interface.
// Works inside JIT-compiled model code. Falls back to default_body when the
// GGUF does not carry the key.
static inline custom_op_spec custom_op_spec_from_meta(
        const dynllama_meta * meta,
        const char * op_name,
        const char * default_body = nullptr) {
    custom_op_spec s;
    s.name = op_name;

    if (!meta) {
        s.body = default_body ? default_body : "";
        return s;
    }

    const std::string base = std::string(DYNLLAMA_CUSTOM_OP_PFX) + op_name + ".";

    auto gstr = [&](const std::string & key, const char * def = "") -> std::string {
        const char * v = meta->get_str(meta->obj, key.c_str(), def);
        return (v && *v) ? v : (def ? def : "");
    };

    const std::string body_v = gstr(base + "body");
    s.body = !body_v.empty() ? body_v : (default_body ? default_body : "");

    const std::string args_v = gstr(base + "args");
    if (!args_v.empty()) s.args = args_v;

    const std::string ret_v = gstr(base + "ret");
    if (!ret_v.empty()) s.ret = ret_v;

    const std::string idx_v = gstr(base + "index");
    if (!idx_v.empty()) s.index = idx_v;

    const std::string cnt_v = gstr(base + "count_arg");
    if (!cnt_v.empty()) s.count_arg = cnt_v;

    if (meta->has(meta->obj, (base + "elementwise").c_str()))
        s.elementwise = (meta->get_int(meta->obj, (base + "elementwise").c_str(), 1) != 0);

    const std::string inc_v = gstr(base + "includes");
    for (size_t pos = 0; pos < inc_v.size(); ) {
        size_t sep = inc_v.find(';', pos);
        if (sep == std::string::npos) sep = inc_v.size();
        std::string one = inc_v.substr(pos, sep - pos);
        if (!one.empty()) s.includes.push_back(one);
        pos = sep + 1;
    }

    return s;
}

// Build a spec from gguf_meta (host side; supports key enumeration).
static inline custom_op_spec custom_op_spec_from_gguf(const gguf_meta & m,
                                                      const std::string & name) {
    custom_op_spec s;
    s.name = name.empty() ? "dynllama_custom_op" : name;

    const std::string base = std::string(DYNLLAMA_CUSTOM_OP_PFX) + name + ".";

    std::string body = m.get_str(base + "body");
    if (body.empty() && !name.empty())
        body = m.get_str(std::string(DYNLLAMA_CUSTOM_OP_PFX) + name);
    if (body.empty())
        body = m.get_str(DYNLLAMA_CUSTOM_OPS_KEY);
    s.body = body;

    s.ret       = m.get_str(base + "ret",       s.ret);
    s.args      = m.get_str(base + "args",      s.args);
    s.index     = m.get_str(base + "index",     s.index);
    s.count_arg = m.get_str(base + "count_arg", s.count_arg);
    if (m.has(base + "elementwise")) s.elementwise = m.get_bool(base + "elementwise");

    const std::string inc = m.get_str(base + "includes");
    for (size_t pos = 0; pos < inc.size(); ) {
        size_t sep = inc.find(';', pos);
        if (sep == std::string::npos) sep = inc.size();
        std::string one = inc.substr(pos, sep - pos);
        if (!one.empty()) s.includes.push_back(one);
        pos = sep + 1;
    }

    return s;
}

static inline bool custom_op_from_gguf(const gguf_meta & m,
                                       const std::string & name,
                                       const jit_options & opt,
                                       void ** handle_out,
                                       void ** fn_out,
                                       std::string * err = nullptr) {
    custom_op_spec s = custom_op_spec_from_gguf(m, name);
    if (s.body.empty()) {
        if (err) *err = "no custom op body found for '" + name + "'";
        return false;
    }
    return custom_op_build(s, opt, handle_out, fn_out, err);
}

struct custom_op_registry {
    std::vector<void *>            handles;
    std::vector<std::string>       names;
    std::vector<dynllama_op_entry> entries;
    dynllama_op_table              table{nullptr, 0};

    const dynllama_op_table * view() {
        for (size_t i = 0; i < entries.size(); ++i) entries[i].name = names[i].c_str();
        table.entries = entries.data();
        table.n       = (int) entries.size();
        return &table;
    }

    ~custom_op_registry() {
        for (void * h : handles) jit_unload_handle(h);
    }
};

static inline std::vector<std::string> custom_op_names(const gguf_meta & m);

static inline bool custom_op_load_all(const gguf_meta & m,
                                      const jit_options & opt,
                                      custom_op_registry & reg,
                                      std::string * err = nullptr) {
    for (const auto & name : custom_op_names(m)) {
        void * h  = nullptr;
        void * fn = nullptr;
        if (!custom_op_from_gguf(m, name, opt, &h, &fn, err)) return false;
        reg.handles.push_back(h);
        reg.names.push_back(name);
        reg.entries.push_back(dynllama_op_entry{nullptr, fn});
    }
    return true;
}

static inline std::vector<std::string> custom_op_names(const gguf_meta & m) {
    std::vector<std::string> names;
    const std::string pfx = DYNLLAMA_CUSTOM_OP_PFX;
    const std::string suf = ".body";
    for (const auto & e : m.kv()) {
        const std::string & k = e.first;
        if (k.size() > pfx.size() + suf.size() &&
            k.compare(0, pfx.size(), pfx) == 0 &&
            k.compare(k.size() - suf.size(), suf.size(), suf) == 0) {
            names.push_back(k.substr(pfx.size(), k.size() - pfx.size() - suf.size()));
        }
    }
    return names;
}

} // namespace dynllama

#endif // DYNLLAMA_CUSTOM_OP_H
