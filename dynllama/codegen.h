#pragma once

#include <string>
#include <vector>

struct gguf_context;

namespace dynllama {

struct custom_op {
    std::string name;
    std::string expr;
    std::string out_var;
    std::vector<std::string> in_vars;
    std::vector<std::string> scalar_vars;
};

// Parse a single element-wise expression body.
// Returns false and writes a message to stderr on parse failure.
bool parse_custom_op(const std::string & name, const std::string & expr, custom_op & out);

// Generate the text of a self-contained .h file for one op.
// Pattern: #ifdef __CUDACC__ kernel / #else inline fallback, plus <name>_launch().
std::string gen_op_header(const custom_op & op);

// Read all dynllama.custom_ops.* string keys from ctx,
// write per-op headers into out_dir/custom_ops/<name>.h,
// and write out_dir/custom_ops_all.h that includes them all.
// Returns the list of successfully generated op names.
std::vector<std::string> gen_custom_ops(const gguf_context * ctx, const std::string & out_dir);

} // namespace dynllama
