//
// DynLLaMA host: extract embedded model-code from GGUF and JIT-compile it.
// Supports two embedding formats:
//   - Multi-file: dynllama.model_code.file.<relpath> string KV entries
//     (entry point: dynllama_main exported by main.cpp)
//   - Single-file (legacy): dynllama.model_code.source string KV entry
//

#ifndef DYNLLAMA_GGUF_JIT_H
#define DYNLLAMA_GGUF_JIT_H

#include "meta.h"   // gguf_meta, META_STRING  (-I dynllama-headers/cpu)
#include "jit.h"    // jit_options, jit_module, jit_from_source, jit_compile_files

#include <cerrno>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/stat.h>
#endif

namespace dynllama {

static const char * DYNLLAMA_SRC_KEY  = "dynllama.model_code.source";
static const char * DYNLLAMA_FILE_PFX = "dynllama.model_code.file.";
// dynllama.train_code.file.* keys may also be present: training code preserved
// beside the model code. The host carries them in metadata but never compiles
// them, so they are intentionally not matched here.

static inline char jit_psep() {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

// Recursive mkdir -- ignores "already exists" error.
static inline void jit_makedirs(const std::string & path) {
#ifdef _WIN32
    if (CreateDirectoryA(path.c_str(), nullptr)) return;
    if (GetLastError() == ERROR_ALREADY_EXISTS) return;
    const size_t sep = path.rfind('\\');
    if (sep == std::string::npos || sep == 0) return;
    jit_makedirs(path.substr(0, sep));
    CreateDirectoryA(path.c_str(), nullptr);
#else
    if (::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) return;
    const size_t sep = path.rfind('/');
    if (sep == std::string::npos || sep == 0) return;
    jit_makedirs(path.substr(0, sep));
    ::mkdir(path.c_str(), 0755);
#endif
}

// Extract a set of (relpath, content) files to a temp directory, compile all
// .cpp files together into a shared library, and load dynllama_main.
// `stem` is used in the temp directory name (typically the arch string).
static inline bool jit_from_files(
        const std::vector<std::pair<std::string,std::string>> & files,
        const std::string & stem,
        const std::string & out_lib,
        const jit_options & opt,
        jit_module & mod,
        std::string * err = nullptr) {

    std::string tmp_dir;
#ifdef _WIN32
    char td[MAX_PATH];
    if (!GetTempPathA(MAX_PATH, td)) { td[0]='.'; td[1]='\\'; td[2]='\0'; }
    tmp_dir = std::string(td) + "dynllama_" + stem + "_src";
#else
    tmp_dir = "/tmp/dynllama_" + stem + "_src";
#endif
    jit_makedirs(tmp_dir);

    std::vector<std::string> cpp_paths;
    const char sep = jit_psep();

    for (const auto & kv : files) {
        // convert posix '/' separators to native
        std::string native_rel = kv.first;
        for (char & c : native_rel) if (c == '/') c = sep;

        const std::string full = tmp_dir + sep + native_rel;

        // create parent directory if needed
        const size_t last = full.rfind(sep);
        if (last != std::string::npos) jit_makedirs(full.substr(0, last));

        std::ofstream fw(full);
        if (!fw) {
            if (err) *err = "cannot write temp file: " + full;
            return false;
        }
        fw << kv.second;
        fw.close();

        // collect .cpp files for compilation
        const std::string & rel = kv.first;
        if (rel.size() >= 4 && rel.compare(rel.size()-4, 4, ".cpp") == 0) {
            cpp_paths.push_back(full);
        }
    }

    if (cpp_paths.empty()) {
        if (err) *err = "no .cpp files in embedded model code";
        return false;
    }

    // add temp dir to include paths so intra-module headers resolve
    jit_options local_opt = opt;
    local_opt.include_dirs.push_back(tmp_dir);

    if (!jit_compile_files(cpp_paths, out_lib, local_opt, err)) return false;
    return jit_load(out_lib, mod, err);
}

// Extract model-code from GGUF metadata and JIT-compile.
// Tries multi-file keys (dynllama.model_code.file.*) first;
// falls back to the legacy dynllama.model_code.source key.
static inline bool jit_from_gguf(const gguf_meta & m,
                                   const std::string & out_lib,
                                   const jit_options & opt,
                                   jit_module & mod,
                                   std::string * err = nullptr) {
    const std::string pfx = DYNLLAMA_FILE_PFX;

    std::vector<std::pair<std::string,std::string>> files;
    for (const auto & entry : m.kv()) {
        if (entry.second.type != META_STRING) continue;
        if (entry.first.size() > pfx.size() &&
            entry.first.compare(0, pfx.size(), pfx) == 0) {
            files.push_back({entry.first.substr(pfx.size()), entry.second.s});
        }
    }

    if (!files.empty()) {
        return jit_from_files(files, m.arch(), out_lib, opt, mod, err);
    }

    // legacy single-source fallback
    const std::string src = m.get_str(DYNLLAMA_SRC_KEY);
    if (!src.empty()) {
        return jit_from_source(src, m.arch(), out_lib, opt, mod, err);
    }

    if (err) *err = "no dynllama.model_code.file.* or dynllama.model_code.source in GGUF";
    return false;
}

} // namespace dynllama

#endif // DYNLLAMA_GGUF_JIT_H
