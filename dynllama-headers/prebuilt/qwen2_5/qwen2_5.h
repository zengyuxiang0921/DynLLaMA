//
// Prebuilt model code: Qwen2 / Qwen2.5 dense decoder.
//
// Auto-configures the whole model from GGUF metadata - no hand-edited
// hyperparameters. Drop a Qwen2.5 .gguf in and qwen2_5_load() reads the
// architecture, layer count, dims, head config, rope and norm epsilon
// straight from the file.
//
// Include resolution: compile model code with `-I dynllama-headers/cpu`,
// so "meta.h" and "dynllama.h" resolve to the CPU backend headers.
//
// Qwen2 specifics vs. plain LLaMA:
//   - separate Q/K/V projections, each WITH bias (attn_{q,k,v}.bias)
//   - SwiGLU FFN (SiLU activation, separate gate/up/down)
//   - RMSNorm, rotary position embedding (NEOX style)
//

#ifndef DYNLLAMA_PREBUILT_QWEN2_5_H
#define DYNLLAMA_PREBUILT_QWEN2_5_H

#include "meta.h"
#include "blocks/tokenizer.h"

#include <cmath>
#include <string>
#include <vector>

namespace dynllama {

// Per-layer tensor names for the Qwen2 architecture.
struct qwen2_5_layer_tensors {
    std::string attn_norm;   // blk.{i}.attn_norm.weight
    std::string attn_q_w,  attn_q_b;
    std::string attn_k_w,  attn_k_b;
    std::string attn_v_w,  attn_v_b;
    std::string attn_out;    // blk.{i}.attn_output.weight
    std::string ffn_norm;    // blk.{i}.ffn_norm.weight
    std::string ffn_gate, ffn_up, ffn_down;
};

struct qwen2_5_model {
    hparams hp;

    // architecture flags resolved from the config
    bool  has_qkv_bias = true;     // Qwen2 uses Q/K/V bias
    int   act_type     = 0;        // 0 = SiLU (matches dynllama-headers ffn act_type)
    int   n_rot        = 0;        // rotary dims (= n_embd_head for qwen2)
    float kq_scale     = 0.0f;     // 1 / sqrt(n_embd_head)

    // special tokens
    int  bos_id = -1, eos_id = -1, pad_id = -1;

    // tokenizer (byte-level BPE)
    bpe_tokenizer tokenizer;

    // global tensor names
    std::string tok_embd   = "token_embd.weight";
    std::string output_norm = "output_norm.weight";
    std::string output_w    = "output.weight";  // may be absent -> tied to tok_embd

    std::vector<qwen2_5_layer_tensors> layers;
};

// Build the per-layer tensor name table from the layer index.
static inline qwen2_5_layer_tensors qwen2_5_layer_names(int i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    qwen2_5_layer_tensors t;
    t.attn_norm = p + "attn_norm.weight";
    t.attn_q_w  = p + "attn_q.weight";   t.attn_q_b = p + "attn_q.bias";
    t.attn_k_w  = p + "attn_k.weight";   t.attn_k_b = p + "attn_k.bias";
    t.attn_v_w  = p + "attn_v.weight";   t.attn_v_b = p + "attn_v.bias";
    t.attn_out  = p + "attn_output.weight";
    t.ffn_norm  = p + "ffn_norm.weight";
    t.ffn_gate  = p + "ffn_gate.weight";
    t.ffn_up    = p + "ffn_up.weight";
    t.ffn_down  = p + "ffn_down.weight";
    return t;
}

//
// Load and auto-configure a Qwen2/Qwen2.5 model from a GGUF file.
// Returns false (and sets *err) when the file is missing, not GGUF,
// not a qwen2 architecture, or missing required hyperparameters.
//
static inline bool qwen2_5_load(const std::string & path, qwen2_5_model & model,
                                std::string * err = nullptr) {
    gguf_meta m;
    if (!m.open(path)) {
        if (err) *err = m.error();
        return false;
    }

    const std::string arch = m.arch();
    if (arch != "qwen2" && arch != "qwen2.5" && arch != "qwen2vl") {
        if (err) *err = "unsupported architecture for qwen2_5 prebuilt: '" + arch + "'";
        return false;
    }

    if (!load_hparams(m, model.hp)) {
        if (err) *err = "failed to read required hyperparameters";
        return false;
    }

    model.has_qkv_bias = m.tensor("blk.0.attn_q.bias") != nullptr;
    model.act_type     = 0; // SiLU / SwiGLU
    model.n_rot        = (int) model.hp.n_embd_head; // qwen2: full-head rotary
    model.kq_scale     = model.hp.n_embd_head
                       ? 1.0f / std::sqrt((float) model.hp.n_embd_head) : 0.0f;

    // special tokens
    model.bos_id = (int) m.get_i32("tokenizer.ggml.bos_token_id", -1);
    model.eos_id = (int) m.get_i32("tokenizer.ggml.eos_token_id", -1);
    model.pad_id = (int) m.get_i32("tokenizer.ggml.padding_token_id", -1);

    // byte-level BPE tokenizer (non-fatal: a model may ship without one)
    model.tokenizer.load(m, nullptr);

    // output.weight is optional; when absent the model ties to token_embd
    if (m.tensor("output.weight") == nullptr) {
        model.output_w = model.tok_embd;
    }

    model.layers.clear();
    model.layers.reserve(model.hp.n_layer);
    for (uint32_t i = 0; i < model.hp.n_layer; i++) {
        model.layers.push_back(qwen2_5_layer_names((int) i));
    }

    return true;
}

} // namespace dynllama

#endif // DYNLLAMA_PREBUILT_QWEN2_5_H
