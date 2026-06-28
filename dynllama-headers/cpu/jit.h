//
// DynLLaMA JIT: spawn the host compiler to build a shared library from source,
// then load and resolve symbols from it.
//
// Accessible to both host code (via -I src) and model code (via -I dynllama-headers/cpu).
// Model code can use this to JIT-compile custom ops from within a running DLL.
//
// Build: -I dynllama-headers -I dynllama-headers/cpu   (link -ldl on POSIX)
//

#ifndef DYNLLAMA_JIT_H
#define DYNLLAMA_JIT_H

#include "dynllama_abi.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace dynllama {

struct jit_options {
    std::string compiler = "g++";
    std::string std_flag = "-std=c++17";
    std::string opt_flag = "-O2";
    std::vector<std::string> include_dirs;
    bool static_runtime  = true;   // -static -static-libgcc -static-libstdc++
};

struct jit_module {
    void *           handle = nullptr;
    dynllama_eval_fn eval   = nullptr;
    bool ok() const { return handle && eval; }
};

static inline const char * jit_lib_ext() {
#ifdef _WIN32
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

static inline bool jit_compile(const std::string & src_cpp,
                               const std::string & out_lib,
                               const jit_options & opt,
                               std::string * err = nullptr) {
    std::string cmd = "\"" + opt.compiler + "\" " + opt.std_flag + " " + opt.opt_flag +
                      " -shared -fPIC";
    if (opt.static_runtime) cmd += " -static -static-libgcc -static-libstdc++";
#ifdef _WIN32
    cmd += " -Wl,--export-all-symbols";
#endif
    for (const auto & inc : opt.include_dirs) cmd += " -I \"" + inc + "\"";
    cmd += " \"" + src_cpp + "\" -o \"" + out_lib + "\"";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";
#endif
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        if (err) *err = "compile failed (rc=" + std::to_string(rc) + "): " + cmd;
        return false;
    }
    return true;
}

static inline bool jit_load(const std::string & lib_path, jit_module & m,
                             std::string * err = nullptr) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA(lib_path.c_str());
    if (!h) { if (err) *err = "LoadLibrary failed for " + lib_path; return false; }
    FARPROC sym = GetProcAddress(h, DYNLLAMA_EVAL_SYMBOL);
    if (!sym) { if (err) *err = "missing symbol " DYNLLAMA_EVAL_SYMBOL; FreeLibrary(h); return false; }
    m.handle = (void *) h;
    m.eval   = reinterpret_cast<dynllama_eval_fn>(reinterpret_cast<void *>(sym));
#else
    void * h = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { if (err) *err = std::string("dlopen failed: ") + dlerror(); return false; }
    void * sym = dlsym(h, DYNLLAMA_EVAL_SYMBOL);
    if (!sym) { if (err) *err = "missing symbol " DYNLLAMA_EVAL_SYMBOL; dlclose(h); return false; }
    m.handle = h;
    m.eval   = reinterpret_cast<dynllama_eval_fn>(sym);
#endif
    return true;
}

static inline void jit_unload(jit_module & m) {
    if (!m.handle) return;
#ifdef _WIN32
    FreeLibrary((HMODULE) m.handle);
#else
    dlclose(m.handle);
#endif
    m.handle = nullptr;
    m.eval   = nullptr;
}

static inline void * jit_load_symbol(const std::string & lib_path,
                                     const std::string & symbol,
                                     void ** handle,
                                     std::string * err = nullptr) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA(lib_path.c_str());
    if (!h) { if (err) *err = "LoadLibrary failed for " + lib_path; return nullptr; }
    FARPROC sym = GetProcAddress(h, symbol.c_str());
    if (!sym) { if (err) *err = "missing symbol " + symbol; FreeLibrary(h); return nullptr; }
    if (handle) *handle = (void *) h;
    return reinterpret_cast<void *>(sym);
#else
    void * h = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { if (err) *err = std::string("dlopen failed: ") + dlerror(); return nullptr; }
    void * sym = dlsym(h, symbol.c_str());
    if (!sym) { if (err) *err = "missing symbol " + symbol; dlclose(h); return nullptr; }
    if (handle) *handle = h;
    return sym;
#endif
}

static inline void jit_unload_handle(void * handle) {
    if (!handle) return;
#ifdef _WIN32
    FreeLibrary((HMODULE) handle);
#else
    dlclose(handle);
#endif
}

static inline bool jit_build(const std::string & src_cpp,
                              const std::string & out_lib,
                              const jit_options & opt,
                              jit_module & m,
                              std::string * err = nullptr) {
    if (!jit_compile(src_cpp, out_lib, opt, err)) return false;
    return jit_load(out_lib, m, err);
}

static inline bool jit_compile_files(const std::vector<std::string> & src_files,
                                     const std::string & out_lib,
                                     const jit_options & opt,
                                     std::string * err = nullptr) {
    std::string cmd = "\"" + opt.compiler + "\" " + opt.std_flag + " " + opt.opt_flag +
                      " -shared -fPIC";
    if (opt.static_runtime) cmd += " -static -static-libgcc -static-libstdc++";
#ifdef _WIN32
    cmd += " -Wl,--export-all-symbols";
#endif
    for (const auto & inc : opt.include_dirs) cmd += " -I \"" + inc + "\"";
    for (const auto & src : src_files)        cmd += " \"" + src + "\"";
    cmd += " -o \"" + out_lib + "\"";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";
#endif
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        if (err) *err = "compile failed (rc=" + std::to_string(rc) + "): " + cmd;
        return false;
    }
    return true;
}

static inline bool jit_from_source(const std::string & src,
                                   const std::string & stem,
                                   const std::string & out_lib,
                                   const jit_options & opt,
                                   jit_module & m,
                                   std::string * err = nullptr) {
#ifdef _WIN32
    char tmp_dir[MAX_PATH];
    if (!GetTempPathA(MAX_PATH, tmp_dir)) { tmp_dir[0]='.'; tmp_dir[1]='\\'; tmp_dir[2]='\0'; }
    const std::string tmp_cpp = std::string(tmp_dir) + "dynllama_" + stem + ".cpp";
#else
    const std::string tmp_cpp = "/tmp/dynllama_" + stem + ".cpp";
#endif
    {
        std::ofstream f(tmp_cpp);
        if (!f) { if (err) *err = "cannot write temp source: " + tmp_cpp; return false; }
        f << src;
    }
    return jit_build(tmp_cpp, out_lib, opt, m, err);
}

} // namespace dynllama

#endif // DYNLLAMA_JIT_H
