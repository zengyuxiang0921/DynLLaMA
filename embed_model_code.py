#!/usr/bin/env python3
"""
embed_model_code.py - Embed model code into a GGUF file.

Two modes:
  Folder mode (new):
    python embed_model_code.py input.gguf model-code/qwen2_5/ output.gguf
    Each file in the folder is embedded as dynllama.model_code.file.<relpath>.
    Entry point: dynllama_main in main.cpp.

  Single-file mode (legacy):
    python embed_model_code.py input.gguf model_code.cpp output.gguf
    Embeds as dynllama.model_code.source.

  Training-code mode:
    python embed_model_code.py train input.gguf train-code/qwen2_5/ output.gguf
    Each file is embedded as dynllama.train_code.file.<relpath>. Preserved as
    metadata beside the model code; the inference host never compiles it.
    Run after the model-code embed to keep both in the same GGUF.

Existing keys under the prefix being written (dynllama.model_code.* or
dynllama.train_code.*) are removed before writing new ones, so it is safe to
re-run either mode on an already-patched GGUF.
"""

import sys
import struct
import shutil
import io
import pathlib

GGUF_MAGIC       = b'GGUF'
DYNLLAMA_PFX     = 'dynllama.model_code.'
DYNLLAMA_SRC_KEY = 'dynllama.model_code.source'
DYNLLAMA_FILE_PFX= 'dynllama.model_code.file.'
# Training code rides along beside the model code. It is preserved as metadata
# only -- the inference host never compiles dynllama.train_code.* keys.
DYNLLAMA_TRAIN_PFX     = 'dynllama.train_code.'
DYNLLAMA_TRAIN_FILE_PFX= 'dynllama.train_code.file.'

_SCALAR_SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}


def _rd_u32(f):
    return struct.unpack('<I', f.read(4))[0]

def _rd_u64(f):
    return struct.unpack('<Q', f.read(8))[0]

def _rd_str(f):
    n = _rd_u64(f)
    return f.read(n).decode('utf-8', errors='replace')

def _skip_str(f):
    f.read(_rd_u64(f))

def _skip_scalar(f, vtype):
    sz = _SCALAR_SIZES.get(vtype)
    if sz is None:
        raise ValueError(f"unknown scalar type {vtype}")
    f.read(sz)

def _skip_array(f):
    et    = _rd_u32(f)
    count = _rd_u64(f)
    if et == 8:
        for _ in range(count): _skip_str(f)
    elif et in _SCALAR_SIZES:
        f.read(_SCALAR_SIZES[et] * count)
    else:
        raise ValueError(f"unknown array element type {et}")


def _parse_kv_entries(path):
    """Parse GGUF KV section entry-by-entry.
    Returns dict with kv_entries list of (key, raw_bytes), tensor_info_bytes,
    version, n_tensors, alignment, orig_data_offset.
    """
    entries = []
    with open(path, 'rb') as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        version   = _rd_u32(f)
        n_tensors = _rd_u64(f)
        n_kv      = _rd_u64(f)

        alignment = 32
        for _ in range(n_kv):
            entry_start = f.tell()
            key   = _rd_str(f)
            vtype = _rd_u32(f)
            if vtype == 9:
                _skip_array(f)
            elif vtype == 8:
                _skip_str(f)
            else:
                val_pos = f.tell()
                _skip_scalar(f, vtype)
                if key == 'general.alignment' and vtype == 4:
                    saved = f.tell()
                    f.seek(val_pos)
                    a = _rd_u32(f)
                    if a > 0: alignment = a
                    f.seek(saved)
            entry_end = f.tell()
            f.seek(entry_start)
            entries.append((key, f.read(entry_end - entry_start)))

        kv_end = f.tell()

        tensor_info_start = kv_end
        for _ in range(n_tensors):
            _skip_str(f)
            n_dims = _rd_u32(f)
            f.read(8 * n_dims)  # shape (int64 each)
            f.read(4)           # quant type
            f.read(8)           # data blob offset
        tensor_info_end = f.tell()

        orig_data_offset = ((tensor_info_end + alignment - 1) // alignment) * alignment

        f.seek(tensor_info_start)
        ti_bytes = f.read(tensor_info_end - tensor_info_start)

    return dict(
        version          = version,
        n_tensors        = n_tensors,
        kv_entries       = entries,
        tensor_info_bytes= ti_bytes,
        alignment        = alignment,
        orig_data_offset = orig_data_offset,
    )


def _make_str_kv(key, value):
    kb = key.encode('utf-8')
    vb = value.encode('utf-8')
    buf = io.BytesIO()
    buf.write(struct.pack('<Q', len(kb))); buf.write(kb)
    buf.write(struct.pack('<I', 8))        # GGUF_TYPE_STRING = 8
    buf.write(struct.pack('<Q', len(vb))); buf.write(vb)
    return buf.getvalue()


def _write_gguf(in_path, info, new_entries, out_path, strip_prefixes=(DYNLLAMA_PFX,)):
    """Write output GGUF: filtered original KVs + new_entries + tensor data."""
    filtered = [(k, raw) for k, raw in info['kv_entries']
                if not any(k.startswith(p) for p in strip_prefixes)]
    total_kv  = len(filtered) + len(new_entries)
    alignment = info['alignment']

    with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
        fout.write(GGUF_MAGIC)
        fout.write(struct.pack('<I', info['version']))
        fout.write(struct.pack('<Q', info['n_tensors']))
        fout.write(struct.pack('<Q', total_kv))

        for _, raw in filtered:   fout.write(raw)
        for _, raw in new_entries: fout.write(raw)

        fout.write(info['tensor_info_bytes'])

        pos = fout.tell()
        pad = (alignment - pos % alignment) % alignment
        fout.write(b'\x00' * pad)

        fin.seek(info['orig_data_offset'])
        shutil.copyfileobj(fin, fout)


# Source extensions embedded by embed_folder; everything else (IDE files,
# build artifacts, hidden dirs) is skipped.
SOURCE_EXTS = {'.cpp', '.cxx', '.cc', '.c', '.h', '.hpp', '.hh', '.cu', '.cuh'}


def _embed_folder_prefixed(in_path, folder_path, out_path, file_pfx, strip_pfx):
    """Embed every source file in a folder as <file_pfx><relpath>, replacing any
    existing keys under strip_pfx."""
    folder = pathlib.Path(folder_path)
    if not folder.is_dir():
        raise SystemExit(f"ERROR: {folder_path} is not a directory")

    info = _parse_kv_entries(in_path)

    new_entries = []
    for fpath in sorted(folder.rglob('*')):
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(folder)
        if any(part.startswith('.') for part in rel.parts):
            continue                          # skip hidden dirs/files (.idea, .git)
        if fpath.suffix.lower() not in SOURCE_EXTS:
            continue                          # skip non-source files
        relpath = rel.as_posix()
        content = fpath.read_text(encoding='utf-8')
        key     = file_pfx + relpath
        new_entries.append((key, _make_str_kv(key, content)))
        print(f"  +{key!r}  ({len(content)} chars)")

    if not new_entries:
        raise SystemExit("ERROR: no source files found in folder")

    _write_gguf(in_path, info, new_entries, out_path, strip_prefixes=(strip_pfx,))
    print(f"Written: {out_path}  ({len(new_entries)} file(s) embedded)")


def embed_folder(in_path, folder_path, out_path):
    """Embed source files in a folder as dynllama.model_code.file.<relpath>."""
    _embed_folder_prefixed(in_path, folder_path, out_path,
                           DYNLLAMA_FILE_PFX, DYNLLAMA_PFX)


def embed_train_folder(in_path, folder_path, out_path):
    """Embed training source files as dynllama.train_code.file.<relpath>.
    Uses a separate prefix so it survives alongside the model code and is never
    picked up by the inference JIT."""
    _embed_folder_prefixed(in_path, folder_path, out_path,
                           DYNLLAMA_TRAIN_FILE_PFX, DYNLLAMA_TRAIN_PFX)


def embed_single(in_path, cpp_path, out_path):
    """Legacy: embed a single .cpp as dynllama.model_code.source."""
    source  = pathlib.Path(cpp_path).read_text(encoding='utf-8')
    info    = _parse_kv_entries(in_path)
    new_entries = [(DYNLLAMA_SRC_KEY, _make_str_kv(DYNLLAMA_SRC_KEY, source))]
    _write_gguf(in_path, info, new_entries, out_path)
    print(f"Written: {out_path}")
    print(f"  +{DYNLLAMA_SRC_KEY!r}  ({len(source)} chars from {cpp_path})")


def embed_custom_op(in_path, name, body, out_path, attrs=None):
    """Embed a custom op: body at dynllama.custom_op.<name>.body, plus optional
    attrs ({'args':..., 'ret':..., 'backend':...}) at .<attr>.
    Any existing dynllama.custom_op.<name>.* keys are replaced."""
    info = _parse_kv_entries(in_path)
    base = f'dynllama.custom_op.{name}.'

    new_entries = [(base + 'body', _make_str_kv(base + 'body', body))]
    for k, v in (attrs or {}).items():
        new_entries.append((base + k, _make_str_kv(base + k, str(v))))

    _write_gguf(in_path, info, new_entries, out_path, strip_prefixes=(base,))
    print(f"Written: {out_path}")
    print(f"  +{base + 'body'!r}  ({len(body)} chars)")
    for k in (attrs or {}):
        print(f"  +{base + k!r} = {attrs[k]!r}")


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) >= 2 and argv[1] == 'op':
        # python embed_model_code.py op input.gguf <name> "<body>" output.gguf [k=v ...]
        if len(argv) < 6:
            print(f"Usage: {argv[0]} op input.gguf <name> \"<body>\" output.gguf [args=... ret=... backend=...]")
            sys.exit(1)
        in_p, name, body, out_p = argv[2], argv[3], argv[4], argv[5]
        attrs = {}
        for kv in argv[6:]:
            k, _, v = kv.partition('=')
            attrs[k] = v
        embed_custom_op(in_p, name, body, out_p, attrs)
    elif len(argv) == 5 and argv[1] == 'train':
        # python embed_model_code.py train input.gguf train-code/<arch>/ output.gguf
        embed_train_folder(argv[2], argv[3], argv[4])
    elif len(argv) == 4:
        in_p, src_p, out_p = argv[1], argv[2], argv[3]
        if pathlib.Path(src_p).is_dir():
            embed_folder(in_p, src_p, out_p)
        else:
            embed_single(in_p, src_p, out_p)
    else:
        print(f"Usage: {argv[0]} input.gguf <folder/ | model_code.cpp> output.gguf")
        print(f"       {argv[0]} train input.gguf <train-folder/> output.gguf")
        print(f"       {argv[0]} op input.gguf <name> \"<body>\" output.gguf [args=... ret=... backend=...]")
        sys.exit(1)
