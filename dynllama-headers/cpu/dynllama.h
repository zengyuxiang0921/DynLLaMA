// DynLLaMA -- master include header (CPU backend, ggml-free).
// One #include to get all float-pointer ops, layers, and blocks.
// Build with: g++ -I dynllama-headers/cpu ...

#ifndef DYNLLAMA_CPU_H
#define DYNLLAMA_CPU_H

// metadata
#include "meta.h"

// ops
#include "ops/silu.h"
#include "ops/gelu.h"
#include "ops/relu.h"
#include "ops/add.h"
#include "ops/mul.h"
#include "ops/mul_mat.h"
#include "ops/linear.h"
#include "ops/soft_max.h"
#include "ops/rms_norm.h"
#include "ops/rope.h"
#include "ops/get_rows.h"
#include "ops/dequantize.h"

// layers
#include "layers/types.h"
#include "layers/dynllama_layer.h"
#include "layers/inp.h"
#include "layers/norm.h"
#include "layers/qkv.h"
#include "layers/attention.h"
#include "layers/ffn.h"

// blocks
#include "blocks/transformer.h"
#include "blocks/repeat.h"
#include "blocks/pipeline.h"

#endif // DYNLLAMA_CPU_H
