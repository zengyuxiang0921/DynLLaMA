"""
Inference / text generation for the PyTorch DeepSeek V4 (model.py).

Loads a checkpoint saved by train.py / train_hf.py and autoregressively generates
a continuation. Tokenization mirrors the trainers:

  byte-level (default): vocab=256, output decoded from UTF-8 bytes.
  --tokenizer <name>:    HuggingFace tokenizer (decode via the tokenizer).
  --chat:                wrap the prompt with a chat template (assistant turn).

  python generate.py --ckpt dsv4.pt --prompt "Once upon a time"
  python generate.py --ckpt dsv4.pt --tokenizer gpt2 --chat --prompt "Hello"

The model has no KV cache (reference implementation), so each step re-runs the
forward over the (truncated) context -- fine for short generations / testing.

The architecture preset (--config tiny|4b) must match how the checkpoint was
trained; vocab_size and n_layers are read back from the checkpoint and dim is
validated against the preset.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from model import DeepSeekV4, ModelArgs


def build_codec(tokenizer_name: str | None):
    """Return (encode, decode, vocab_size, tokenizer). encode: str->ids, decode: ids->str."""
    if tokenizer_name is None:
        def encode(text: str):
            return list(text.encode("utf-8"))

        def decode(ids):
            return bytes(int(i) for i in ids).decode("utf-8", errors="replace")
        return encode, decode, 256, None

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "--tokenizer needs 'transformers' (pip install transformers)") from e
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode(text: str):
        return tok.encode(text, add_special_tokens=False)

    def decode(ids):
        return tok.decode(ids, skip_special_tokens=True)
    return encode, decode, len(tok), tok


def render_prompt(prompt, tokenizer, chat, chat_template, system):
    """Apply a chat template (assistant generation prompt) when --chat is set."""
    if not chat:
        return prompt
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    if tokenizer is not None and chat_template:
        tokenizer.chat_template = chat_template
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if chat_template:
        from jinja2 import Template
        return Template(chat_template).render(messages=msgs, add_generation_prompt=True)
    # minimal fallback (byte-level / templateless tokenizers)
    return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in msgs) + "<|assistant|>\n"


def sample_next(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    """Sample one token id from final-position logits [vocab]."""
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum - sp <= top_p          # always keeps the top-1
        sp = sp * keep
        sp = sp / sp.sum()
        return int(si[torch.multinomial(sp, 1)])
    return int(torch.multinomial(probs, 1))


@torch.no_grad()
def generate(model, ids, max_new, temperature, top_k, top_p, eos_id, max_seq, device):
    out = list(ids)
    for _ in range(max_new):
        ctx = out[-max_seq:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        nxt = sample_next(logits, temperature, top_k, top_p)
        out.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
    return out


def load_model(ckpt_path, config, device):
    """Build the model from the preset, sized to the checkpoint, and load weights."""
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    vocab, dim = state["embed.weight"].shape
    n_layers = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blk."))

    cfg = ModelArgs.tiny() if config == "tiny" else ModelArgs()
    if cfg.dim != dim:
        raise SystemExit(
            f"checkpoint dim={dim} != --config {config} dim={cfg.dim}; pick the matching preset")
    cfg.vocab_size = vocab
    cfg.n_layers = n_layers

    model = DeepSeekV4(cfg).to(device).eval()
    model.load_state_dict(state)
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate text with PyTorch DeepSeek V4")
    ap.add_argument("--ckpt", default="dsv4.pt", help="checkpoint from train.py / train_hf.py")
    ap.add_argument("--config", choices=("tiny", "4b"), default="tiny",
                    help="architecture preset the checkpoint was trained with")
    ap.add_argument("--prompt", default="", help="prompt text")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer name; omit for byte-level")
    # chat
    ap.add_argument("--chat", action="store_true", help="wrap prompt with a chat template")
    ap.add_argument("--chat-template", default=None, help="custom Jinja chat template")
    ap.add_argument("--system", default=None, help="optional system message (with --chat)")
    # sampling
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=40, help="0 = disabled")
    ap.add_argument("--top-p", type=float, default=0.95, help="1 = disabled")
    ap.add_argument("--no-eos", action="store_true", help="ignore the tokenizer EOS token")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(args.device)

    encode, decode, vocab, tokenizer = build_codec(args.tokenizer)
    model, cfg = load_model(args.ckpt, args.config, device)
    if cfg.vocab_size != vocab:
        raise SystemExit(
            f"checkpoint vocab={cfg.vocab_size} != tokenizer vocab={vocab}; "
            "use the same --tokenizer (or byte-level) as in training")
    print(f"loaded {args.ckpt}  config={args.config}  layers={cfg.n_layers}  "
          f"vocab={vocab}  params={model.num_params()/1e6:.1f}M  device={device}")

    text = render_prompt(args.prompt, tokenizer, args.chat, args.chat_template, args.system)
    ids = encode(text)
    if not ids:
        ids = [0]                                  # need at least one token to start

    eos_id = None
    if not args.no_eos and tokenizer is not None:
        eos_id = getattr(tokenizer, "eos_token_id", None)

    out = generate(model, ids, args.max_new_tokens, args.temperature,
                   args.top_k, args.top_p, eos_id, cfg.max_seq_len, device)
    completion = decode(out[len(ids):])

    print("--- prompt ---")
    print(text, end="" if text.endswith("\n") else "\n")
    print("--- completion ---")
    print(completion)


if __name__ == "__main__":
    main()
