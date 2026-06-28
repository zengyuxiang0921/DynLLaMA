#pragma once
// DeepSeek V4 training hyperparameters (demo values; a real run is far larger).

namespace ds4train {

struct train_config {
    int   micro_batch  = 4;
    int   seq_len      = 512;
    int   grad_accum   = 8;
    float lr           = 2.4e-3f;   // Muon peak LR for 2D weight matrices
    float lr_adam      = 6e-4f;     // AdamW LR for embeddings / norms / 1D params
    float weight_decay = 0.1f;
    float grad_clip    = 1.0f;
    int   warmup_steps = 2000;
    int   total_steps  = 100000;
    // MTP auxiliary loss weight (DeepSeek densified supervision).
    float mtp_weight   = 0.3f;
    // aux-free MoE load balancing: per-expert routing-bias update step.
    float moe_bias_rate = 1e-3f;
    // mHC manifold constraint: re-run Sinkhorn on (U, V) every N steps.
    int   mhc_resinkhorn_every = 50;
};

} // namespace ds4train
