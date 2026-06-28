"""
DeepSeek V4 (4B-class) -- PyTorch reference.

A clean, runnable PyTorch port of the C++ DynLLaMA core model-code in this
folder (main.cpp / ops.h / ops/hc.h). It mirrors the same architecture:

  - Hyper-Connections residual stream (hc_mult parallel copies, Sinkhorn mixing)
  - V4 attention: low-rank Q, MQA latent KV, per-head q RMSNorm, learned
    attention sink, grouped low-rank output projection
  - Mixture-of-Experts FFN: sqrtsoftplus gate + aux-free bias + route_scale,
    one shared expert, SwiGLU with swiglu_limit
  - RMSNorm pre-norm, final Hyper-Connection head reduction

Like the C++ core port it omits FP8/FP4 quantization, KV compression
(Compressor/Indexer) and MTP -- attention is dense/causal over the sequence.

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

    @classmethod
    def tiny(cls) -> "ModelArgs":
        # Matches the C++ demo constants for quick CPU testing.
        return cls(
            vocab_size=1024, dim=256, n_layers=4,
            n_heads=8, head_dim=64, rope_head_dim=16,
            q_lora_rank=64, o_groups=4, o_lora_rank=64,
            n_routed_experts=8, n_activated_experts=2, n_shared_experts=1,
            moe_inter_dim=128, hc_mult=4, max_seq_len=512,
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


# ---- rotary embedding (consecutive-pair / interleaved, plain theta) ----------
def build_rope_cache(seqlen: int, rope_dim: int, base: float, device, dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))
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


# ---- V4 attention -----------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rd = args.rope_head_dim
        self.eps = args.norm_eps
        self.scale = args.head_dim ** -0.5

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

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, S, self.n_heads, self.head_dim)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps).to(q.dtype)
        q = torch.cat([q[..., :-self.rd], apply_rope(q[..., -self.rd:], cos, sin)], dim=-1)

        kv = self.kv_norm(self.wkv(x))                                # [B, S, head_dim]
        kv = torch.cat([kv[..., :-self.rd], apply_rope(kv[..., -self.rd:], cos, sin)], dim=-1)

        # MQA attention: every query head attends the single shared kv stream
        scores = torch.einsum("bihd,bjd->bhij", q, kv) * self.scale   # [B, H, S, S]
        causal = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal
        # softmax with a learned per-head attention sink in the denominator
        m = torch.maximum(scores.amax(dim=-1, keepdim=True),
                          self.attn_sink.view(1, -1, 1, 1))
        e = torch.exp(scores - m)
        denom = e.sum(dim=-1, keepdim=True) + torch.exp(self.attn_sink.view(1, -1, 1, 1) - m)
        attn = e / denom
        o = torch.einsum("bhij,bjd->bihd", attn, kv)                  # [B, S, H, head_dim]
        o = torch.cat([o[..., :-self.rd], apply_rope(o[..., -self.rd:], cos, sin, inverse=True)], dim=-1)

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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attn = Attention(args)
        self.ffn = MoE(args)
        self.hc_attn = HyperConn(args)
        self.hc_ffn = HyperConn(args)

    def forward(self, H, cos, sin):
        residual = H
        y, post, comb = self.hc_attn.pre(H)
        y = self.attn(self.attn_norm(y), cos, sin)
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
        self.blk = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.head_reduce = HeadReduce(args)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, S = tokens.shape
        cos, sin = build_rope_cache(S, self.args.rope_head_dim, self.args.rope_theta,
                                    tokens.device, self.embed.weight.dtype)
        h = self.embed(tokens)                              # [B, S, dim]
        H = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)    # HC copies [B, S, hc, dim]
        for layer in self.blk:
            H = layer(H, cos, sin)
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

    full = ModelArgs()
    print(f"4B-class config: {full.n_layers} layers, dim={full.dim}, "
          f"{full.n_routed_experts} experts (top-{full.n_activated_experts}), "
          f"vocab={full.vocab_size}")
