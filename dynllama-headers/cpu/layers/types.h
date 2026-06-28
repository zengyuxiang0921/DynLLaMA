//
// Shared selectors for the CPU (ggml-free) layer/block API.
// The float-pointer layers take plain ints; these names document the values.
//

#ifndef DYNLLAMA_TYPES_H
#define DYNLLAMA_TYPES_H

// activation selector for FFN / MoE (matches op dispatch order)
enum dynllama_act {
    DYNLLAMA_ACT_SILU = 0,
    DYNLLAMA_ACT_GELU = 1,
    DYNLLAMA_ACT_RELU = 2,
};

// MoE gating selector
enum dynllama_gating {
    DYNLLAMA_GATING_SOFTMAX = 0,
    DYNLLAMA_GATING_SIGMOID = 1,
};

#endif // DYNLLAMA_TYPES_H
