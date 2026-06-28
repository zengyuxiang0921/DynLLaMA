# Prebuilt model code

Ready-made DynLLaMA model code for specific architectures. Each subdirectory
auto-configures itself from a GGUF file's metadata - no hand-edited
hyperparameters - via the metadata reader in `dynllama-headers/cpu/meta.h`.

These headers are consumed by the embedded model code in `model-code/*.cpp`,
which `#include`s a prebuilt descriptor plus the DynLLaMA headers and composes
the standard ops/layers/blocks into a forward pass.

## Available

| Directory   | Architecture | gguf `general.architecture` |
| ----------- | ------------ | --------------------------- |
| `qwen2_5/`  | Qwen2 / Qwen2.5 dense | `qwen2`, `qwen2.5`, `qwen2vl` |

## Include conventions

```
-I dynllama-headers/cpu     -> "dynllama.h", "meta.h", ops/layers/blocks
-I dynllama-headers         -> "prebuilt/qwen2_5/qwen2_5.h"
```

The CPU backend is pure float-pointer and **ggml-free**: `dynllama.h` pulls in
the ops (`silu`, `mul_mat`, `rms_norm`, `dequantize`, ...), the layers
(`inp`, `norm`, `qkv`, `attention`, `ffn`) and the blocks (`transformer`,
`pipeline`).

## Metadata reader

`dynllama-headers/cpu/meta.h` is a dependency-free, header-only GGUF reader.
It parses only the key-value metadata and tensor-info headers (never tensor
data), so opening a multi-GB model is cheap.

```cpp
#include "meta.h"   // -I dynllama-headers/cpu

dynllama::gguf_meta m;
m.open("model.gguf");

dynllama::hparams hp;
dynllama::load_hparams(m, hp);   // reads {arch}.block_count, embedding_length, ...
// hp.n_embd, hp.n_layer, hp.n_head, hp.n_head_kv, hp.n_ff,
// hp.rope_freq_base, hp.rms_eps, hp.n_embd_head, hp.n_vocab
```

## qwen2_5

```cpp
#include "prebuilt/qwen2_5/qwen2_5.h"   // -I dynllama-headers (+ -I dynllama-headers/cpu)

dynllama::qwen2_5_model model;
std::string err;
if (!dynllama::qwen2_5_load("qwen2.5-0.5b-instruct-q4_k_m.gguf", model, &err)) {
    // err describes why (missing file, wrong arch, missing hparams)
}
// model.hp           - auto-read hyperparameters
// model.has_qkv_bias - Qwen2 uses Q/K/V bias (detected from tensors)
// model.kq_scale     - 1/sqrt(n_embd_head)
// model.layers[i]    - per-layer tensor names (blk.{i}.attn_q.weight, ...)
```

`model-code/qwen2_5.cpp` then turns this config into a forward pass via
`qwen2_5_forward(model, weights, tok_embd, output_norm_w, output_w, tokens, ...)`.

## Tests

```sh
# metadata auto-config against a real gguf
mingw64/bin/g++.exe -std=c++17 -I dynllama-headers/cpu -I dynllama-headers \
    tests/test-qwen2_5-meta.cpp -o test-qwen2_5-meta
./test-qwen2_5-meta qwen2.5-0.5b-instruct-q4_k_m.gguf

# dequantize op + float forward (synthetic weights), links model-code/qwen2_5.cpp
mingw64/bin/g++.exe -std=c++17 -I dynllama-headers/cpu -I dynllama-headers \
    tests/test-forward.cpp model-code/qwen2_5.cpp -o test-forward
./test-forward
```

## Host / model-code split

The system has two halves, with `dynllama-headers/dynllama_abi.h` as the contract:

- **Host (llama.cpp side)** - `src/dynllama/runtime.h`: opens the gguf,
  auto-reads hparams, dequantizes every tensor to float, assembles the
  per-layer/global weight pointers, allocates the KV cache, fills a
  `dynllama_run_ctx`. `src/dynllama/jit.h` compiles the embedded model code to
  a shared library and resolves the single `dynllama_eval` symbol.
- **Model code (embedded, JIT-compiled)** - `model-code/qwen2_5.cpp`:
  implements `dynllama_eval(ctx)` and ONLY runs the forward pass over the
  buffers the host prepared. No file IO, no dequantization.

## Status

- **Tokenizer** (`blocks/tokenizer.h`): byte-level BPE (gpt2/qwen2), loaded
  from gguf metadata. encode() matches the authoritative Qwen2 tokenizer
  exactly on the test samples; decode(encode(x)) == x for any input.
- **Dequantization** (`ops/dequantize.h`): F32, F16, Q8_0, Q4_0/1, Q5_0/1,
  **Q4_K, Q6_K** (the last two validated bit-exact vs gguf-py). Loads the
  sample `Q4_K_M` file end-to-end. Other K-quants / IQ-quants pending.
- **RoPE** (`ops/rope.h` `rope_neox`): NEOX rotary applied to Q/K in the
  transformer block with absolute positions (KV-cache aware).
- **KV cache**: incremental decode via `llm_pipeline_forward_kv`; verified to
  match the full-sequence pass exactly (`tests/test-forward.cpp`,
  `tests/test-e2e.cpp`).
- **JIT**: `tests/test-jit.cpp` compiles `model-code/qwen2_5.cpp` to a DLL,
  loads `dynllama_eval`, and runs it.
- **End-to-end generation**: `tests/test-generate.cpp` tokenizes a prompt,
  loads weights, and greedily decodes - produces coherent text
  ("The capital of France is Paris.").

## Still pending

- **gguf-embedded source extraction.** `jit.h` compiles a `.cpp` on disk; the
  step that writes the gguf `dynllama.model_code.*` keys out to a temp `.cpp`
  still needs wiring inside llama.cpp.
- **Sampling beyond greedy / chat template.** The demo uses argmax; temperature
  / top-p sampling and the chat template are not wired up.
- **Pre-tokenizer edge cases.** The Qwen2 pre-tokenizer is matched for common
  text; exotic Unicode category boundaries are approximated (round-trip always
  holds).
