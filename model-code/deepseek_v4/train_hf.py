"""
Streaming HuggingFace-datasets training for the PyTorch DeepSeek V4 (model.py).

Streams a text dataset (no full download -- examples are pulled on the fly via
datasets streaming mode), tokenizes, packs tokens into fixed-length blocks, and
trains the next-token LM. Saves a state_dict whose keys are blk.{i}.* so it
converts straight to GGUF:

    pip install datasets                  # byte-level mode (default, no tokenizer)
    pip install datasets transformers     # only needed for --tokenizer <hf-name>

    python train_hf.py --out dsv4.pt --steps 2000
    python ../../convert_torch_to_gguf.py dsv4.pt dsv4.gguf --arch deepseek2

Tokenization modes:
  byte-level (default): each UTF-8 byte is a token (vocab=256). No tokenizer
    download; keeps the model small. The C++ tiny demo can run the result after
    setting N_VOCAB=256 in ops.h.
  --tokenizer <name>: use a HuggingFace tokenizer; vocab_size is taken from it.

The architecture comes from model.py's ModelArgs (tiny by default, --config 4b
for the full shape). vocab_size is overridden to match the tokenizer, so for a
real C++ run the C++ demo constants must be bumped to match.

This differs from train.py, which trains a synthetic +1 task with no downloads
(used for the C++ end-to-end verification). Use this file for real text data.
"""

from __future__ import annotations

import argparse
import itertools

import torch
import torch.nn.functional as F

from model import DeepSeekV4, ModelArgs


def build_encoder(tokenizer_name: str | None):
    """Return (encode_fn, vocab_size). encode_fn maps str -> list[int]."""
    if tokenizer_name is None:
        # Byte-level: UTF-8 bytes, vocab fixed at 256, no external dependency.
        def encode(text: str):
            return list(text.encode("utf-8"))
        return encode, 256

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "--tokenizer needs 'transformers' (pip install transformers)") from e

    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode(text: str):
        return tok.encode(text, add_special_tokens=False)

    return encode, len(tok)


def open_stream(name: str, config: str | None, split: str, shuffle_buffer: int, seed: int):
    """Open a streaming IterableDataset (nothing is written to disk)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("this script needs 'datasets' (pip install datasets)") from e

    ds = load_dataset(name, config, split=split, streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return ds


def token_generator(ds, encode_fn, text_column: str):
    """Yield a flat, infinite stream of token ids (re-streams the split on exhaust)."""
    while True:
        produced = False
        for ex in ds:
            text = ex.get(text_column)
            if not text:
                continue
            produced = True
            for t in encode_fn(text):
                yield t
        if not produced:
            raise SystemExit(
                f"no text found in column '{text_column}'; set --text-column")


def batch_generator(token_gen, batch: int, seqlen: int, device):
    """Pack the token stream into [batch, seqlen+1] blocks (last col = next target)."""
    need = batch * (seqlen + 1)
    buf: list[int] = []
    for t in token_gen:
        buf.append(t)
        if len(buf) >= need:
            chunk, buf = buf[:need], buf[need:]
            yield torch.tensor(chunk, dtype=torch.long, device=device).view(batch, seqlen + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream-train PyTorch DeepSeek V4 on HF datasets")
    # data / streaming
    ap.add_argument("--dataset", default="roneneldan/TinyStories",
                    help="HuggingFace dataset id (streamed, not downloaded whole)")
    ap.add_argument("--dataset-config", default=None, help="dataset config/name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--shuffle-buffer", type=int, default=10000,
                    help="streaming shuffle buffer (0 to disable)")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer name; omit for byte-level (vocab=256)")
    # model / optim
    ap.add_argument("--out", default="dsv4.pt", help="checkpoint output path")
    ap.add_argument("--config", choices=("tiny", "4b"), default="tiny")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=0,
                    help="checkpoint every N steps (0 = only at end)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    encode_fn, vocab = build_encoder(args.tokenizer)
    cfg = ModelArgs.tiny() if args.config == "tiny" else ModelArgs()
    cfg.vocab_size = vocab                     # match the tokenizer
    if args.seqlen > cfg.max_seq_len:
        cfg.max_seq_len = args.seqlen

    model = DeepSeekV4(cfg).to(device).train()
    print(f"config={args.config}  vocab={vocab}  params={model.num_params()/1e6:.1f}M  device={device}")
    print(f"streaming dataset={args.dataset} config={args.dataset_config} split={args.split}")

    ds = open_stream(args.dataset, args.dataset_config, args.split, args.shuffle_buffer, args.seed)
    batches = batch_generator(
        token_generator(ds, encode_fn, args.text_column), args.batch, args.seqlen, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(prog * 3.1415926535)).item())

    for step in range(1, args.steps + 1):
        seq = next(batches)
        inp, tgt = seq[:, :-1], seq[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))

        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {opt.param_groups[0]['lr']:.2e}")

        if args.save_every and step % args.save_every == 0 and step != args.steps:
            torch.save(model.state_dict(), args.out)
            print(f"  checkpoint -> {args.out}")

    torch.save(model.state_dict(), args.out)
    print(f"saved checkpoint -> {args.out}")
    print("convert with: python convert_torch_to_gguf.py "
          f"{args.out} dsv4.gguf --arch deepseek2")


if __name__ == "__main__":
    main()
