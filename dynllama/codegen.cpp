#include "codegen.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

// gguf C API
#include "../dynllama-headers/cpu/ggml/gguf.h"

namespace dynllama {

// Math function names excluded from scalar-parameter detection.
static const std::set<std::string> k_math = {
    "exp", "expf", "log", "logf", "sqrt", "sqrtf",
    "sin", "sinf", "cos", "cosf", "tan", "tanf",
    "abs", "fabsf", "fabs", "pow", "powf",
    "max", "min", "fmax", "fmin", "fmaxf", "fminf",
    "tanh", "tanhf", "atan", "atanf", "atan2", "atan2f",
    "ceil", "ceilf", "floor", "floorf",
};

bool parse_custom_op(const std::string & name, const std::string & expr, custom_op & op) {
    op.name = name;
    op.expr = expr;

    // Output variable: identifier[i] = (not ==)
    std::regex lhs_re(R"((\w+)\s*\[i\]\s*=(?!=))");
    std::smatch m;
    if (!std::regex_search(expr, m, lhs_re)) {
        fprintf(stderr, "dynllama codegen: '%s': no 'var[i] =' found in: %s\n",
                name.c_str(), expr.c_str());
        return false;
    }
    op.out_var = m[1].str();

    // RHS: everything after the first bare '='
    auto eq = std::string::npos;
    for (size_t i = 0; i < expr.size(); ++i) {
        if (expr[i] == '=' && (i == 0 || (expr[i-1] != '!' && expr[i-1] != '<'
                                           && expr[i-1] != '>' && expr[i-1] != '='))
                            && (i + 1 >= expr.size() || expr[i+1] != '=')) {
            eq = i;
            break;
        }
    }
    if (eq == std::string::npos) {
        fprintf(stderr, "dynllama codegen: '%s': cannot find RHS\n", name.c_str());
        return false;
    }
    std::string rhs = expr.substr(eq + 1);

    // Input arrays: var[i] on RHS
    std::regex arr_re(R"((\w+)\s*\[i\])");
    std::set<std::string> seen_in;
    auto it = std::sregex_iterator(rhs.begin(), rhs.end(), arr_re);
    for (; it != std::sregex_iterator(); ++it) {
        std::string v = (*it)[1].str();
        if (v != op.out_var && !seen_in.count(v)) {
            op.in_vars.push_back(v);
            seen_in.insert(v);
        }
    }

    // Scalar parameters: bare identifiers not in array set, not 'i', not math
    std::set<std::string> all_arr(seen_in);
    all_arr.insert(op.out_var);
    std::regex id_re(R"(\b([a-zA-Z_]\w*)\b)");
    std::set<std::string> seen_sc;
    auto it2 = std::sregex_iterator(rhs.begin(), rhs.end(), id_re);
    for (; it2 != std::sregex_iterator(); ++it2) {
        std::string v = (*it2)[1].str();
        if (v == "i" || all_arr.count(v) || k_math.count(v) || seen_sc.count(v)) {
            continue;
        }
        op.scalar_vars.push_back(v);
        seen_sc.insert(v);
    }

    return true;
}

static std::string build_params(const custom_op & op) {
    std::ostringstream ss;
    ss << "float * " << op.out_var;
    for (auto & v : op.in_vars)     ss << ", const float * " << v;
    for (auto & v : op.scalar_vars) ss << ", float " << v;
    ss << ", int n";
    return ss.str();
}

static std::string build_args(const custom_op & op) {
    std::ostringstream ss;
    ss << op.out_var;
    for (auto & v : op.in_vars)     ss << ", " << v;
    for (auto & v : op.scalar_vars) ss << ", " << v;
    ss << ", n";
    return ss.str();
}

std::string gen_op_header(const custom_op & op) {
    std::string p = build_params(op);
    std::string a = build_args(op);
    std::ostringstream ss;

    ss << "// custom op: " << op.name << "\n"
       << "// expr: "      << op.expr << "\n"
       << "\n"
       << "#ifdef __CUDACC__\n"
       << "extern \"C\" __global__\n"
       << "void " << op.name << "(" << p << ") {\n"
       << "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
       << "    if (i < n) {\n"
       << "        " << op.expr << ";\n"
       << "    }\n"
       << "}\n"
       << "\n"
       << "static inline void " << op.name << "_launch(" << p << ") {\n"
       << "    dim3 block(256);\n"
       << "    dim3 grid((n + 255) / 256);\n"
       << "    " << op.name << "<<<grid, block>>>(" << a << ");\n"
       << "}\n"
       << "\n"
       << "#else\n"
       << "#include <cmath>\n"
       << "static inline void " << op.name << "(" << p << ") {\n"
       << "    for (int i = 0; i < n; i++) {\n"
       << "        " << op.expr << ";\n"
       << "    }\n"
       << "}\n"
       << "\n"
       << "static inline void " << op.name << "_launch(" << p << ") {\n"
       << "    " << op.name << "(" << a << ");\n"
       << "}\n"
       << "\n"
       << "#endif\n";

    return ss.str();
}

std::vector<std::string> gen_custom_ops(const gguf_context * ctx, const std::string & out_dir) {
    const std::string prefix = "dynllama.custom_ops.";
    std::filesystem::create_directories(out_dir + "/custom_ops");

    std::vector<std::string> generated;
    int64_t n_kv = gguf_get_n_kv(ctx);

    for (int64_t i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(ctx, i);
        std::string skey(key);
        if (skey.substr(0, prefix.size()) != prefix) {
            continue;
        }
        if (gguf_get_kv_type(ctx, i) != GGUF_TYPE_STRING) {
            continue;
        }

        std::string op_name = skey.substr(prefix.size());
        std::string expr    = gguf_get_val_str(ctx, i);

        custom_op op;
        if (!parse_custom_op(op_name, expr, op)) {
            continue;
        }

        std::string header_path = out_dir + "/custom_ops/" + op_name + ".h";
        std::ofstream f(header_path);
        if (!f) {
            fprintf(stderr, "dynllama codegen: failed to write %s\n", header_path.c_str());
            continue;
        }
        f << gen_op_header(op);
        generated.push_back(op_name);
    }

    // Write bundle header
    std::ofstream bundle(out_dir + "/custom_ops_all.h");
    bundle << "// auto-generated - do not edit\n"
           << "#pragma once\n\n";
    for (auto & name : generated) {
        bundle << "#include \"custom_ops/" << name << ".h\"\n";
    }

    return generated;
}

} // namespace dynllama
