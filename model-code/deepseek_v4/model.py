"""
DeepSeek V4 (4B-class) -- PyTorch reference.

A clean, runnable PyTorch port of the C++ DynLLaMA core model-code in this
folder (main.cpp / ops.h / ops/hc.h). It mirrors the same architecture:

  - Hyper-Connections residual stream (hc_mult parallel copies, Sinkhorn mixing)
  - V4 hybrid attention: low-rank Q, MQA latent KV, per-head q RMSNorm, learned
    attention sink, grouped low-rank output projection, and per-layer sliding
    window + learned KV compression (Compressor) + sparse top-k selection
    (Indexer), chosen per layer by compress_ratios
  - Mixture-of-Experts FFN: sqrtsoftplus gate + aux-free bias + route_scale,
    one shared expert, SwiGLU with swiglu_limit
  - RMSNorm pre-norm, final Hyper-Connection head reduction

The hybrid attention is a dense, differentiable, prefill-only port: it uses
masked dense math instead of the reference's sparse kernels and omits FP8/FP4
quantization, the decode-time KV cache, the overlap-window Compressor variant,
and MTP. ModelArgs.tiny() sets compress_ratios all 0 and window >= max_seq_len,
which reduces hybrid attention to plain dense causal attention (so the C++/GGUF
port stays bit-identical); the 4B-class default enables the full hybrid mix.

Run a tiny CPU smoke test:  python model.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    # ---- 4B-class defaults (mirrors the C++ "real" comments, scaled down) ----
    vocab_size: int = 129280
    dim: int = 2048
    n_layers: int = 28
    # attention (MQA latent + low-rank Q/O)
    n_heads: int = 16
    head_dim: int = 128
    rope_head_dim: int = 64        # trailing RoPE dims of each head
    q_lora_rank: int = 512
    o_groups: int = 8
    o_lora_rank: int = 512
    # hybrid attention: sliding window + per-layer KV compression / sparse selection
    window_size: int = 128
    # per-layer compression ratio (0 = local-only, 128 = coarse compressed KV,
    # 4 = fine compressed KV selected by the Indexer). Indexed by layer (modulo).
    compress_ratios: tuple = (0, 0, 4, 128, 4, 128, 4, 0)
    compress_rope_theta: float = 40000.0
    # YaRN (applied to compressed layers only; original_seq_len = 0 disables)
    original_seq_len: int = 0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    # Indexer (sparse top-k over compressed KV, used by ratio == 4 layers)
    index_n_heads: int = 16
    index_head_dim: int = 128
    index_topk: int = 512
    # MoE
    n_routed_experts: int = 64
    n_activated_experts: int = 6
    n_shared_experts: int = 1
    moe_inter_dim: int = 1024
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # misc
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_seq_len: int = 4096

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    def compress_ratio_for(self, layer_id: int) -> int:
        r = self.compress_ratios
        return r[layer_id] if layer_id < len(r) else r[layer_id % len(r)]

    @classmethod
    def tiny(cls) -> "ModelArgs":
        # Matches the C++ demo constants for quick CPU testing. compress_ratios
        # all 0 + window >= max_seq_len makes hybrid attention reduce to plain
        # dense causal attention, keeping the C++/GGUF port bit-identical.
        return cls(
            vocab_size=1024, dim=256, n_layers=4,
            n_heads=8, head_dim=64, rope_head_dim=16,
            q_lora_rank=64, o_groups=4, o_lora_rank=64,
            n_routed_experts=8, n_activated_experts=2, n_shared_experts=1,
            moe_inter_dim=128, hc_mult=4, max_seq_len=512,
            window_size=512, compress_ratios=(0, 0, 0, 0),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dt))


# ---- rotary embedding (consecutive-pair / interleaved) with optional YaRN -----
def build_rope_cache(seqlen: int, rope_dim: int, base: float, device, dtype=torch.float32,
                     original_seq_len: int = 0, factor: float = 1.0,
                     beta_fast: int = 32, beta_slow: int = 1):
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))
    if original_seq_len > 0:
        # YaRN: blend interpolated and original frequencies with a linear ramp.
        def corr_dim(num_rot):
            return rope_dim * math.log(original_seq_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr_dim(beta_fast)), 0)
        high = min(math.ceil(corr_dim(beta_slow)), rope_dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(rope_dim // 2, device=device, dtype=torch.float32) - low)
                / (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
    t = torch.arange(seqlen, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)         # [S, rope_dim/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False):
    # x: [B, S, *mid, rope_dim]; cos/sin: [S, rope_dim/2]
    S = x.shape[1]
    rd2 = x.shape[-1] // 2
    n_mid = x.ndim - 3                       # dims between S and the rope pairs
    shape = (1, S) + (1,) * n_mid + (rd2,)
    c = cos.view(shape)
    s = sin.view(shape)
    if inverse:
        s = -s
    xe = x[..., 0::2]
    xo = x[..., 1::2]
    re = xe * c - xo * s
    ro = xe * s + xo * c
    return torch.stack((re, ro), dim=-1).flatten(-2)


# ---- Hyper-Connections ------------------------------------------------------
def hc_split_sinkhorn(mixes, scale, base, hc, iters, eps):
    # mixes: [..., mix_hc]; returns pre[...,hc], post[...,hc], comb[...,hc,hc]
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]
    comb = comb.unflatten(-1, (hc, hc))               # [..., hc, hc]

    comb = comb.softmax(dim=-1) + eps                  # row softmax
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class HyperConn(nn.Module):
    """One sub-block's HC weights (hc_pre reduce + hc_post expand)."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc = args.hc_mult
        self.iters = args.hc_sinkhorn_iters
        self.eps = args.hc_eps
        self.norm_eps = args.norm_eps
        hc_dim = args.hc_mult * args.dim
        self.fn = nn.Parameter(torch.randn(args.mix_hc, hc_dim) * 0.02)
        self.base = nn.Parameter(torch.zeros(args.mix_hc))
        self.scale = nn.Parameter(torch.ones(3))

    def pre(self, H: torch.Tensor):
        # H: [B, S, hc, dim] -> reduced y: [B, S, dim], plus post/comb for expand
        x = H.flatten(-2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, self.scale, self.base, self.hc, self.iters, self.eps)
        y = torch.sum(pre.unsqueeze(-1) * H, dim=2)
        return y.to(H.dtype), post, comb

    @staticmethod
    def post(x_out, residual, post, comb):
        # new[..,j,:] = post[..,j]*x_out + sum_i comb[..,i,j]*residual[..,i,:]
        expand = torch.einsum("bsij,bsid->bsjd", comb, residual)
        return post.unsqueeze(-1) * x_out.unsqueeze(-2) + expand


# ---- KV compression + sparse selection --------------------------------------
def _rope_split(x, cos, sin, rd, inverse=False):
    """Apply RoPE to the trailing rd dims of x, leaving the rest untouched."""
    return torch.cat([x[..., :-rd], apply_rope(x[..., -rd:], cos, sin, inverse)], dim=-1)


class Compressor(nn.Module):
    """Learned gated pooling that compresses KV over `ratio` consecutive tokens.

    Each compressed token is a softmax-gated sum of `ratio` source tokens (with a
    learned per-slot absolute-position embedding `ape`), then RMSNorm'd and rotated
    at the block's first source position. Dense, prefill-only port of the reference
    Compressor (no overlap windows, no decode-time state, no quantization).
    """
    def __init__(self, dim: int, head_dim: int, rope_head_dim: int, ratio: int, eps: float):
        super().__init__()
        self.ratio = ratio
        self.rd = rope_head_dim
        self.wkv = nn.Linear(dim, head_dim, bias=False)
        self.wgate = nn.Linear(dim, head_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(ratio, head_dim))
        self.norm = RMSNorm(head_dim, eps)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        n = S // self.ratio
        if n == 0:
            return None
        cut = n * self.ratio
        kv = self.wkv(x[:, :cut]).unflatten(1, (n, self.ratio))        # [B, n, ratio, d]
        score = self.wgate(x[:, :cut]).unflatten(1, (n, self.ratio)) + self.ape
        ckv = (kv * score.softmax(dim=2)).sum(dim=2)                   # [B, n, d]
        ckv = self.norm(ckv)
        pos = torch.arange(n, device=x.device) * self.ratio           # block first positions
        return _rope_split(ckv, cos[pos], sin[pos], self.rd)


class Indexer(nn.Module):
    """Learned scorer that selects the top-k compressed KV positions per query.

    Builds its own compressed KV (separate Compressor), scores each query against
    every compressed token with a multi-head ReLU dot product, and returns a
    boolean keep-mask over compressed positions. Dense port of the reference
    Indexer (no Hadamard rotation, no FP4 quant).

    Note: top-k selection is non-differentiable, so these parameters receive no
    gradient from the LM loss (same as the reference). Training the indexer needs
    a separate index-distillation objective; under plain cross-entropy it stays
    at its initialization.
    """
    def __init__(self, args: ModelArgs, ratio: int):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.hd = args.index_head_dim
        self.rd = args.rope_head_dim
        self.topk = args.index_topk
        self.ratio = ratio
        self.scale = self.hd ** -0.5
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.hd, bias=False)
        self.weights_proj = nn.Linear(args.dim, self.n_heads, bias=False)
        self.compressor = Compressor(args.dim, self.hd, args.rope_head_dim, ratio, args.norm_eps)

    def forward(self, x, qr, cos, sin):
        B, S, _ = x.shape
        ckv = self.compressor(x, cos, sin)                            # [B, n, hd]
        if ckv is None:
            return None
        n = ckv.shape[1]
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.hd))      # [B, S, nh, hd]
        q = _rope_split(q, cos[:S], sin[:S], self.rd)
        w = self.weights_proj(x) * (self.scale * self.n_heads ** -0.5)  # [B, S, nh]
        score = torch.einsum("bshd,bcd->bshc", q, ckv).relu()        # [B, S, nh, n]
        score = (score * w.unsqueeze(-1)).sum(dim=2)                  # [B, S, n]
        # causal: block c is visible to query s once it is fully in the past
        c = torch.arange(n, device=x.device).unsqueeze(0)
        s = torch.arange(S, device=x.device).unsqueeze(1)
        valid = (c + 1) * self.ratio <= s + 1                        # [S, n]
        score = score.masked_fill(~valid, float("-inf"))
        k = min(self.topk, n)
        topk = score.topk(k, dim=-1).indices                        # [B, S, k]
        mask = torch.zeros(B, S, n, dtype=torch.bool, device=x.device)
        mask.scatter_(-1, topk, True)
        return mask & valid                                         # drop -inf picks


# ---- V4 hybrid attention ----------------------------------------------------
class Attention(nn.Module):
    """Hybrid MQA latent attention: sliding-window local attention plus, on
    compressed layers, attention over learned-compressed KV (optionally sparse
    via the Indexer). Reduces to plain dense causal attention when ratio == 0 and
    window_size >= seq_len. Low-rank Q, per-head q RMSNorm, learned attention
    sink, grouped low-rank O projection. Prefill/dense; no decode KV cache."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rd = args.rope_head_dim
        self.eps = args.norm_eps
        self.scale = args.head_dim ** -0.5
        self.window = args.window_size
        self.ratio = args.compress_ratio_for(layer_id)

        self.wq_a = nn.Linear(args.dim, args.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.wq_b = nn.Linear(args.q_lora_rank, args.n_heads * args.head_dim, bias=False)
        self.wkv = nn.Linear(args.dim, args.head_dim, bias=False)     # MQA: single KV head
        self.kv_norm = RMSNorm(args.head_dim, args.norm_eps)
        self.attn_sink = nn.Parameter(torch.zeros(args.n_heads))

        att_dim = args.n_heads * args.head_dim
        self.o_groups = args.o_groups
        self.group_in = att_dim // args.o_groups
        self.wo_a = nn.Parameter(torch.randn(args.o_groups, args.o_lora_rank, self.group_in) * 0.02)
        self.wo_b = nn.Linear(args.o_groups * args.o_lora_rank, args.dim, bias=False)

        if self.ratio:
            self.compressor = Compressor(args.dim, args.head_dim, args.rope_head_dim,
                                         self.ratio, args.norm_eps)
            # ratio == 4 layers use fine compressed KV gated by a learned Indexer.
            self.indexer = Indexer(args, self.ratio) if self.ratio == 4 else None

        # Per-layer RoPE cache: compressed layers use compress_rope_theta + YaRN.
        base = args.compress_rope_theta if self.ratio else args.rope_theta
        osl = args.original_seq_len if self.ratio else 0
        cos, sin = build_rope_cache(args.max_seq_len, args.rope_head_dim, base, "cpu",
                                    torch.float32, osl, args.rope_factor,
                                    args.beta_fast, args.beta_slow)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, S, _ = x.shape
        cos, sin = self.cos[:S], self.sin[:S]
        rd = self.rd

        qr = self.q_norm(self.wq_a(x))                               # shared with Indexer
        q = self.wq_b(qr).view(B, S, self.n_heads, self.head_dim)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps).to(q.dtype)
        q = _rope_split(q, cos, sin, rd)

        kv = self.kv_norm(self.wkv(x))                               # [B, S, head_dim]
        kv = _rope_split(kv, cos, sin, rd)

        # local sliding-window causal scores
        scores_l = torch.einsum("bihd,bjd->bhij", q, kv) * self.scale  # [B, H, S, S]
        i = torch.arange(S, device=x.device).unsqueeze(1)
        j = torch.arange(S, device=x.device).unsqueeze(0)
        local_mask = (j <= i) & (j > i - self.window)
        scores_l = scores_l.masked_fill(~local_mask, float("-inf"))

        # compressed (global) scores on compressed layers
        ckv = None
        scores_c = None
        if self.ratio:
            ckv = self.compressor(x, cos, sin)                      # [B, n, head_dim]
            if ckv is not None:
                n = ckv.shape[1]
                scores_c = torch.einsum("bihd,bcd->bhic", q, ckv) * self.scale  # [B, H, S, n]
                c = torch.arange(n, device=x.device).unsqueeze(0)
                cvalid = (c + 1) * self.ratio <= i + 1             # [S, n]
                if self.indexer is not None:
                    sel = self.indexer(x, qr, cos, sin)            # [B, S, n]
                    cmask = cvalid.unsqueeze(0) & sel              # [B, S, n]
                    scores_c = scores_c.masked_fill(~cmask.unsqueeze(1), float("-inf"))
                else:
                    scores_c = scores_c.masked_fill(~cvalid.view(1, 1, S, n), float("-inf"))

        # joint sink-softmax over the union of local + compressed positions
        scores = scores_l if scores_c is None else torch.cat([scores_l, scores_c], dim=-1)
        sink = self.attn_sink.view(1, -1, 1, 1)
        m = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
        e = torch.exp(scores - m)
        denom = e.sum(dim=-1, keepdim=True) + torch.exp(sink - m)
        attn = e / denom

        o = torch.einsum("bhij,bjd->bihd", attn[..., :S], kv)       # [B, S, H, head_dim]
        if scores_c is not None:
            o = o + torch.einsum("bhic,bcd->bihd", attn[..., S:], ckv)
        o = _rope_split(o, cos, sin, rd, inverse=True)

        # grouped low-rank output projection
        o = o.reshape(B, S, self.o_groups, self.group_in)
        og = torch.einsum("bsgd,grd->bsgr", o, self.wo_a).reshape(B, S, -1)
        return self.wo_b(og)


# ---- MoE --------------------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, dim, inter, limit):
        super().__init__()
        self.w1 = nn.Linear(dim, inter, bias=False)   # gate
        self.w3 = nn.Linear(dim, inter, bias=False)   # up
        self.w2 = nn.Linear(inter, dim, bias=False)   # down
        self.limit = limit

    def forward(self, x):
        gate = self.w1(x)
        up = self.w3(x)
        if self.limit > 0:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(-self.limit, self.limit)
        return self.w2(F.silu(gate) * up)


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed = args.n_routed_experts
        self.topk = args.n_activated_experts
        self.route_scale = args.route_scale
        self.gate = nn.Linear(args.dim, args.n_routed_experts, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(args.n_routed_experts))   # aux-free balancing
        self.experts = nn.ModuleList(
            [Expert(args.dim, args.moe_inter_dim, args.swiglu_limit) for _ in range(args.n_routed_experts)])
        assert args.n_shared_experts == 1
        self.shared = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)

    def forward(self, x):
        B, S, _ = x.shape
        xf = x.reshape(-1, self.dim)
        scores = F.softplus(self.gate(xf).float()).sqrt()              # sqrtsoftplus
        idx = (scores + self.gate_bias).topk(self.topk, dim=-1).indices
        w = scores.gather(1, idx)
        w = (w / w.sum(-1, keepdim=True) * self.route_scale).to(x.dtype)

        y = torch.zeros_like(xf)
        for e in range(self.n_routed):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            y.index_add_(0, tok, self.experts[e](xf[tok]) * w[tok, slot, None])
        y = y + self.shared(xf)
        return y.view(B, S, self.dim)


# ---- block + model ----------------------------------------------------------
class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(args)
        self.hc_attn = HyperConn(args)
        self.hc_ffn = HyperConn(args)

    def forward(self, H):
        residual = H
        y, post, comb = self.hc_attn.pre(H)
        y = self.attn(self.attn_norm(y))
        H = HyperConn.post(y, residual, post, comb)

        residual = H
        y, post, comb = self.hc_ffn.pre(H)
        y = self.ffn(self.ffn_norm(y))
        H = HyperConn.post(y, residual, post, comb)
        return H


class HeadReduce(nn.Module):
    """Final Hyper-Connection reduction (sigmoid-gated, no Sinkhorn)."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc = args.hc_mult
        self.eps = args.hc_eps
        self.norm_eps = args.norm_eps
        self.fn = nn.Parameter(torch.randn(args.hc_mult, args.hc_mult * args.dim) * 0.02)
        self.base = nn.Parameter(torch.zeros(args.hc_mult))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, H):
        x = H.flatten(-2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.fn) * rsqrt
        pre = torch.sigmoid(mixes * self.scale + self.base) + self.eps
        return torch.sum(pre.unsqueeze(-1) * H, dim=2).to(H.dtype)


class DeepSeekV4(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hc_mult = args.hc_mult
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        # named "blk" so state_dict keys are blk.{i}.* (llama.cpp GGUF convention)
        self.blk = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.head_reduce = HeadReduce(args)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)                              # [B, S, dim]
        H = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)    # HC copies [B, S, hc, dim]
        for layer in self.blk:                              # each Attention owns its RoPE
            H = layer(H)
        y = self.head_reduce(H)
        return self.lm_head(self.norm(y))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    torch.manual_seed(42)
    args = ModelArgs.tiny()
    model = DeepSeekV4(args).eval()
    print(f"tiny config: {args.n_layers} layers, dim={args.dim}, "
          f"{args.n_routed_experts} experts (top-{args.n_activated_experts}), "
          f"params={model.num_params()/1e6:.1f}M")

    x = torch.randint(0, args.vocab_size, (2, 16))
    with torch.no_grad():
        logits = model(x)
    print("logits:", tuple(logits.shape), "finite:", bool(torch.isfinite(logits).all()))

    # exercise the full hybrid attention mix (local + compressed + indexer) on a
    # small but enabled config so every branch runs.
    hy = ModelArgs(
        vocab_size=512, dim=128, n_layers=4, n_heads=4, head_dim=32, rope_head_dim=8,
        q_lora_rank=32, o_groups=2, o_lora_rank=32, n_routed_experts=4,
        n_activated_experts=2, moe_inter_dim=64, hc_mult=4, max_seq_len=128,
        window_size=8, compress_ratios=(0, 4, 128, 4), index_n_heads=4,
        index_head_dim=32, index_topk=4, original_seq_len=64,
    )
    hm = DeepSeekV4(hy).eval()
    xh = torch.randint(0, hy.vocab_size, (2, 64))
    with torch.no_grad():
        lh = hm(xh)
    print(f"hybrid config: layers={hy.n_layers} ratios={hy.compress_ratios} "
          f"window={hy.window_size}  params={hm.num_params()/1e6:.2f}M")
    print("hybrid logits:", tuple(lh.shape), "finite:", bool(torch.isfinite(lh).all()))

    full = ModelArgs()
    print(f"4B-class config: {full.n_layers} layers, dim={full.dim}, "
          f"{full.n_routed_experts} experts (top-{full.n_activated_experts}), "
          f"vocab={full.vocab_size}")
