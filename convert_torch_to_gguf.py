#!/usr/bin/env python3
"""
convert_torch_to_gguf.py - Dump all weights from a PyTorch checkpoint into a GGUF file.

Usage:
    python convert_torch_to_gguf.py model.pt output.gguf [--arch NAME]
           [--dtype f32|f16|keep] [--name NAME] [--state-dict-key KEY]

Accepts a state_dict (.pt/.pth/.bin), a pickled nn.Module, or a checkpoint dict
that wraps the weights under a key such as 'state_dict' or 'model'. Every tensor
is written as a GGUF tensor under its original parameter name.

  --dtype f32   cast all float tensors to float32 (default, lossless)
  --dtype f16   cast all float tensors to float16 (half the size)
  --dtype keep  keep each float tensor's dtype (bfloat16 still -> float32,
                numpy has no native bfloat16)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent / 'gguf-py'))

import gguf  # noqa: E402

logger = logging.getLogger("convert-torch")


def load_checkpoint(path: Path):
    """Load a torch checkpoint, preferring the safe weights_only path and falling
    back to a full unpickle (needed for pickled nn.Module objects)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("weights_only load failed (%s); retrying with "
                       "weights_only=False -- only do this for files you trust", e)
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(obj, key: str | None):
    """Return a {name: tensor} dict from whatever torch.load produced."""
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        if key:
            if key not in obj:
                raise SystemExit(f"ERROR: key {key!r} not in checkpoint; "
                                 f"top-level keys: {list(obj)[:10]}")
            return obj[key]
        for k in ("state_dict", "model", "module"):
            if isinstance(obj.get(k), dict):
                return obj[k]
        return obj
    raise SystemExit(f"ERROR: unsupported checkpoint type {type(obj)}")


def to_numpy(t: torch.Tensor, dtype: str) -> np.ndarray:
    """Convert a torch tensor to a numpy array of a GGUF-supported dtype."""
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:                       # no native numpy bf16
        t = t.to(torch.float16 if dtype == "f16" else torch.float32)
    if t.is_floating_point():
        if dtype == "f16":
            t = t.to(torch.float16)
        elif dtype == "f32":
            t = t.to(torch.float32)
        # keep: leave float16/float32/float64 as-is
    elif t.dtype == torch.bool:
        t = t.to(torch.int8)
    elif t.dtype in (torch.uint8, torch.uint16, torch.uint32):
        t = t.to(torch.int32)
    return np.ascontiguousarray(t.numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump a torch model's weights into a GGUF file")
    ap.add_argument("input", type=Path, help="torch checkpoint (.pt/.pth/.bin)")
    ap.add_argument("output", type=Path, help="output .gguf path")
    ap.add_argument("--arch", default="torch", help="general.architecture value (default: torch)")
    ap.add_argument("--name", default=None, help="general.name value")
    ap.add_argument("--dtype", choices=("f32", "f16", "keep"), default="f32",
                    help="float tensor output dtype (default: f32)")
    ap.add_argument("--state-dict-key", default=None,
                    help="pull the weights from this top-level checkpoint key")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")

    if not args.input.is_file():
        raise SystemExit(f"ERROR: {args.input} not found")

    sd = extract_state_dict(load_checkpoint(args.input), args.state_dict_key)

    writer = gguf.GGUFWriter(args.output, args.arch)
    if args.name:
        writer.add_name(args.name)

    ftype = {"f16": gguf.LlamaFileType.MOSTLY_F16,
             "f32": gguf.LlamaFileType.ALL_F32}.get(args.dtype)
    if ftype is not None:
        writer.add_file_type(ftype)

    n = 0
    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            logger.warning("skip %r: not a tensor (%s)", name, type(tensor).__name__)
            continue
        arr = to_numpy(tensor, args.dtype)
        writer.add_tensor(name, arr)
        logger.debug("  +%s  %s  %s", name, list(arr.shape), arr.dtype)
        n += 1

    if n == 0:
        raise SystemExit("ERROR: no tensors found in checkpoint")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    logger.info("Written: %s  (%d tensor(s), dtype=%s)", args.output, n, args.dtype)


if __name__ == "__main__":
    main()
