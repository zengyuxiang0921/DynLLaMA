// DynLLaMA -- master include header (CUDA backend).
// One #include to get all ops, layers, and blocks.
// Build with: nvcc -I dynllama-headers/cuda/ ...

#ifndef DYNLLAMA_CUDA_H
#define DYNLLAMA_CUDA_H

#include "ops/silu.h"
#include "ops/gelu.h"
#include "ops/relu.h"
#include "ops/add.h"
#include "ops/mul.h"
#include "ops/mul_mat.h"
#include "ops/soft_max.h"
#include "ops/rms_norm.h"
#include "ops/rope.h"
#include "ops/get_rows.h"

#include "layers/types.h"
#include "layers/norm.h"
#include "layers/ffn.h"
#include "layers/attention.h"
#include "layers/qkv.h"
#include "layers/inp.h"
#include "layers/moe-ffn.h"

#include "blocks/transformer.h"
#include "blocks/repeat.h"
#include "blocks/pipeline.h"

#endif
