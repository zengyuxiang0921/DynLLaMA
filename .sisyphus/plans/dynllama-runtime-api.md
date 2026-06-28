# Plan: dynllama.h — Master Include Header

## Current State

`cpu/dynllama.h` is an 8-line skeleton. The user should be able to just `#include "dynllama.h"` and have access to all ops, layers, and blocks.

## Design

```
// 用户唯一需要做的事：
#include "dynllama.h"     ← 聚合所有 ops + layers + blocks

// 编译时指好算子路径：
//   g++ -I dynllama-headers/cpu/   ...  ← CPU 算子
//   nvcc -I dynllama-headers/cuda/ ...  ← CUDA 算子
```

dynllama.h = 聚合头文件，按层 include 所有内容：

```
dynllama.h
  ├── types.h                    ← 公共类型（ggml_tensor, llm_ffn_op_type 等）
  ├── ops/
  │   ├── silu.h
  │   ├── mul_mat.h
  │   ├── rms_norm.h
  │   ├── soft_max.h
  │   ├── gelu.h / relu.h / add.h / mul.h / rope.h / get_rows.h
  ├── layers/
  │   ├── ffn.h, attention.h, norm.h, qkv.h, inp.h, moe-ffn.h
  ├── blocks/
  │   ├── transformer.h, repeat.h, pipeline.h
```

用户写模型代码时：
```c
#include "dynllama.h"

void my_ffn(float * out, const float * x,
            const float * w1, const float * w2) {
    // 直接调 ops
    mul_mat(w1, x, tmp1, ...);
    silu(tmp1, tmp1, ...);
    mul_mat(w2, tmp1, out, ...);
}

void my_pipeline(float * logits, const int * tokens, ...) {
    // 直接调 layers/blocks
    llm_inp_embd(hidden, tok_embd, tokens, ...);
    llm_block_transformer(hidden, hidden, ...);
    mul_mat(output_w, hidden, logits, ...);
}
```

## Tasks

### Task 1 — cpu/dynllama.h ✅

Replace `dynllama-headers/cpu/dynllama.h` with:

```c
// DynLLaMA — master include header (CPU backend).
// One #include to get all ops, layers, and blocks.
// Build with: g++ -I dynllama-headers/cpu/ ...

#ifndef DYNLLAMA_CPU_H
#define DYNLLAMA_CPU_H

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
#include "layers/rwkv.h"

#include "blocks/transformer.h"
#include "blocks/repeat.h"
#include "blocks/pipeline.h"

#endif
```

### Task 2 — cuda/dynllama.h ✅

Create `dynllama-headers/cuda/dynllama.h`:

```c
// DynLLaMA — master include header (CUDA backend).
// One #include to get all ops, layers, and blocks.
// Build with: nvcc -I dynllama-headers/cuda/ ...

#ifndef DYNLLAMA_CUDA_H
#define DYNLLAMA_CUDA_H

#include "ops/silu.h"       // extern "C" __global__ + CPU fallback
#include "ops/gelu.h"
#include "ops/relu.h"
#include "ops/add.h"
#include "ops/mul.h"
#include "ops/mul_mat.h"
#include "ops/soft_max.h"
#include "ops/rms_norm.h"
#include "ops/rope.h"
#include "ops/get_rows.h"

#include "layers/types.h"   // ggml types for the overloads
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
```

## Design: JIT Compilation Pipeline (create_ops / run_ops)

用户代码在 GGUF 中以数学片段存储，需要在运行时根据编译环境拼接成完整 kernel 并通过对应JIT编译执行。

### 模板 → 拼接 → 编译 流程

```
GGUF 中存的用户代码（数学部分）：
  out[i] = in[i] / (1.0f + expf(-in[i]));

create_ops() 拼接成完整 CUDA kernel：
  extern "C" __global__
  void fn(float * out, const float * in, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) {
          out[i] = in[i] / (1.0f + expf(-in[i]));
      }
  }

run_ops(code) → NVRTC 编译 → nvrtcCompileProgram()
                        → nvrtcGetPTX() → cuModuleLoadDataEx()
                        → cuModuleGetFunction() → cuLaunchKernel()
```

### create_ops()

```c
char * dynllama_create_ops(
    struct gguf_context * gguf,
    const char          * path,
    const char          * args);  // 如 "float*,float*,int"
```

拼接步骤：
1. 从 GGUF metadata 读用户代码：`gguf_find_key(ctx, path)`
2. 解析 `args`，生成函数签名：`extern "C" __global__ void fn(PARAMS)`
3. 注入线程索引模板：`int i = blockIdx.x * blockDim.x + threadIdx.x;`
4. 注入边界检查：`if (i < n) {`
5. 粘贴用户代码 → 闭合 `}}`

模板索引类型：

| 类型 | 注入代码 |
|---|---|
| `1d` | `int i = blockIdx.x * blockDim.x + threadIdx.x;` |
| `2d` | `int row = blockIdx.y * blockDim.y + threadIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;` |
| `row` | `int row = blockIdx.x;` |

### run_ops()

```c
typedef void (*dynllama_kernel_t)(void ** args);
dynllama_kernel_t dynllama_run_ops(const char * code);
```

内部：`nvrtcCreateProgram` → `nvrtcCompileProgram` → `nvrtcGetPTX` → `cuModuleLoadDataEx` → `cuModuleGetFunction`

### kernel 缓存

```c
// dynllama-headers/cuda/kernel_cache.h
typedef struct {
    CUmodule   module;
    CUfunction kernel;
} dynllama_cached_kernel_t;

dynllama_cached_kernel_t dynllama_get_or_compile(
    struct gguf_context * gguf, const char * ops_path);
```

首次调用时 compile 并缓存，后续直接从缓存查。

```
