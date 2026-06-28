"""
Training script for the PyTorch DeepSeek V4 (model.py).

Trains a next-token language model and saves a checkpoint whose state_dict keys
are already in llama.cpp blk.{i}.* form (the model names its decoder stack "blk"),
so it converts straight to GGUF:

    python train.py --out dsv4.pt --steps 300
    python ../../convert_torch_to_gguf.py dsv4.pt dsv4.gguf --arch deepseek2

The default config matches the C++ model-code demo constants (ModelArgs.tiny),
so the produced GGUF loads directly into model-code/deepseek_v4 (main.cpp). Use
--config 4b for the full 4B-class shape (not CPU-runnable).

Demo task: a learnable "+1" sequence (x_{t+1} = (x_t + step) % vocab) so the loss
visibly drops, exercising the whole architecture (HC / MQA attn / MoE) end to end.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from model import DeepSeekV4, ModelArgs


def make_batch(bsz: int, seqlen: int, vocab: int, step: int, device) -> torch.Tensor:
    start = torch.randint(0, vocab, (bsz, 1), device=device)
    offs = torch.arange(seqlen, device=device).unsqueeze(0) * step
    return (start + offs) % vocab            # [bsz, seqlen]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PyTorch DeepSeek V4")
    ap.add_argument("--out", default="dsv4.pt", help="checkpoint output path")
    ap.add_argument("--config", choices=("tiny", "4b"), default="tiny")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--task-step", type=int, default=1, help="+k sequence stride")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = ModelArgs.tiny() if args.config == "tiny" else ModelArgs()
    device = torch.device(args.device)

    model = DeepSeekV4(cfg).to(device).train()
    print(f"config={args.config}  params={model.num_params()/1e6:.1f}M  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    for step in range(1, args.steps + 1):
        seq = make_batch(args.batch, args.seqlen + 1, cfg.vocab_size, args.task_step, device)
        inp, tgt = seq[:, :-1], seq[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"step {step:4d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}")

    # Save weights only (state_dict). Keys are blk.{i}.* -> convert to GGUF directly.
    torch.save(model.state_dict(), args.out)
    print(f"saved checkpoint -> {args.out}")
    print("convert with: python convert_torch_to_gguf.py "
          f"{args.out} dsv4.gguf --arch deepseek2")


if __name__ == "__main__":
    main()
