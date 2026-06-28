#!/usr/bin/env python3
from __future__ import annotations

import logging
import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent / 'gguf-py'))

import gguf  # noqa: E402
from dynllama import codegen  # noqa: E402

logger = logging.getLogger("convert-gguf")


def load_custom_ops(custom_ops_file: Path) -> list[codegen.CustomOp]:
    """Read a custom-ops definition file and return parsed CustomOp objects.

    File format - one op per line:
        <name>: <expression>

    Example:
        my_add:   n[i] = a[i] + b[i]
        my_scale: n[i] = a[i] * scale
        my_swish: n[i] = a[i] / (1.0f + expf(-a[i]))

    Lines starting with '#' and blank lines are ignored.
    """
    if not custom_ops_file.is_file():
        logger.warning(f'Custom ops file not found: {custom_ops_file}, skipping')
        return []

    ops: list[codegen.CustomOp] = []
    for lineno, raw in enumerate(custom_ops_file.read_text(encoding='utf-8').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if ':' not in line:
            logger.warning(f'{custom_ops_file}:{lineno}: expected "<name>: <expr>", got {line!r}')
            continue
        name, _, expr = line.partition(':')
        name = name.strip()
        expr = expr.strip()
        if not name or not expr:
            logger.warning(f'{custom_ops_file}:{lineno}: empty name or expression, skipping')
            continue
        try:
            op = codegen.parse(name, expr)
        except ValueError as e:
            logger.error(f'{custom_ops_file}:{lineno}: {e}')
            sys.exit(1)
        ops.append(op)
        logger.info(f'  Parsed custom op: {name!r} out={op.out_var!r} '
                    f'in={op.in_vars} scalars={op.scalar_vars}')
    return ops


def embed_custom_ops(writer: gguf.GGUFWriter, ops: list[codegen.CustomOp]) -> None:
    """Write each op expression as a dynllama.custom_ops.<name> string key."""
    for op in ops:
        key = f'dynllama.custom_ops.{op.name}'
        writer.add_string(key, op.expr)
        logger.info(f'  Embedded {key!r} = {op.expr!r}')


def load_model_code(model_code_dir: Path) -> dict[str, str]:
    """Scan model-code directory recursively for C++ files and return
    a dict mapping 'dynllama.<relative_path_no_ext>' -> file content."""
    if not model_code_dir.is_dir():
        logger.warning(f'Model code directory not found: {model_code_dir}, skipping')
        return {}

    code_entries: dict[str, str] = {}
    extensions = {'.cu', '.cpp', '.c', '.h', '.cuh'}

    for fpath in sorted(model_code_dir.rglob('*')):
        if not fpath.is_file() or fpath.suffix not in extensions:
            continue
        rel = fpath.relative_to(model_code_dir)
        key = 'dynllama.' + str(rel.with_suffix('')).replace(os.sep, '/')
        content = fpath.read_text(encoding='utf-8')
        code_entries[key] = content
        logger.info(f'  Loaded model code: {rel} -> {key}')

    return code_entries


def copy_with_dynllama_metadata(
    reader: gguf.GGUFReader,
    writer: gguf.GGUFWriter,
    source_code_value: str,
    model_code_dir: Path,
    custom_ops: list[codegen.CustomOp] | None = None,
) -> None:
    # Copy all existing metadata fields, skipping GGUF internals and architecture
    for field in reader.fields.values():
        if field.name.startswith('GGUF.'):
            logger.debug(f'Suppressing {field.name}')
            continue

        if field.name == gguf.Keys.General.ARCHITECTURE:
            logger.debug('Suppressing general.architecture (will be set to dynllama)')
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = field.contents()

        # Skip source_code and dynllama.* if they already exist (we will add fresh ones)
        if field.name == 'source_code' or field.name.startswith('dynllama.'):
            logger.debug(f'Replacing existing {field.name}')
            continue

        if val is not None:
            writer.add_key_value(field.name, val, val_type, sub_type=sub_type)
            logger.debug(f'Copying {field.name}')

    # Add the new source_code entry
    logger.info(f'Adding source_code: {source_code_value}')
    writer.add_string('source_code', source_code_value)

    # Embed custom op expressions
    if custom_ops:
        logger.info(f'Embedding {len(custom_ops)} custom op(s):')
        embed_custom_ops(writer, custom_ops)

    # Embed model code from model-code/ directory
    model_code = load_model_code(model_code_dir)
    if model_code:
        logger.info(f'Embedding {len(model_code)} model code entries:')
        for key, content in model_code.items():
            # Strip leading blank lines for cleaner metadata
            content = content.lstrip('\n')
            writer.add_string(key, content)
            logger.info(f'  {key} ({len(content)} bytes)')

    # Copy tensor info
    total_bytes = 0
    for tensor in reader.tensors:
        total_bytes += tensor.n_bytes
        writer.add_tensor_info(
            tensor.name, tensor.data.shape,
            tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type,
        )

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        bar.update(tensor.n_bytes)

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a GGUF file for dynllama: set architecture to dynllama and add source_code metadata",
    )
    parser.add_argument("model", type=Path, help="GGUF format model filename")
    parser.add_argument("source_code", type=str, help='Source code URL or identifier, e.g. "https://github.com/org/repo"')
    parser.add_argument("--output", "-o", type=Path, help="Output GGUF filename (default: overwrite input)")
    parser.add_argument("--model-code", type=Path, default=Path("./model-code"), help="Directory containing C++ model code to embed (default: ./model-code)")
    parser.add_argument("--custom-ops", type=Path, default=None, metavar="FILE",
                        help="File of custom op definitions to embed as dynllama.custom_ops.* keys. "
                             "Format: one 'name: expr' per line, e.g.  my_add: n[i] = a[i] + b[i]")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")

    args = parser.parse_args(None if len(sys.argv) > 2 else ["--help"])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    input_path: Path = args.model
    if not input_path.is_file():
        logger.error(f'Input file not found: {input_path}')
        sys.exit(1)

    output_path: Path = args.output if args.output is not None else input_path

    if input_path == output_path:
        logger.warning('*** Warning *** Warning *** Warning **')
        logger.warning('* Modifying a GGUF file in-place requires a full rewrite.')
        logger.warning('* The input file will be replaced. Proceed at your own risk.')
        logger.warning('* Enter exactly YES if you are positive you want to proceed:')
        response = input('YES, I am sure> ')
        if response != 'YES':
            logger.info("You didn't enter YES. Okay then, see ya!")
            sys.exit(0)

    logger.info(f'* Loading: {input_path}')
    reader = gguf.GGUFReader(str(input_path), 'r')

    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()

    logger.info(f'* Architecture: {arch} -> dynllama')

    # Write to a temporary file first for safety when overwriting
    if input_path == output_path:
        tmp_path = input_path.with_suffix('.gguf.tmp')
    else:
        tmp_path = output_path

    logger.info(f'* Writing: {tmp_path}')
    writer = gguf.GGUFWriter(str(tmp_path), arch='dynllama', endianess=reader.endianess)

    alignment_field = reader.fields.get(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None:
        alignment = alignment_field.contents()
        logger.debug(f'Setting custom alignment: {alignment}')
        writer.data_alignment = alignment

    custom_ops = load_custom_ops(args.custom_ops) if args.custom_ops else []
    copy_with_dynllama_metadata(reader, writer, args.source_code, args.model_code, custom_ops)

    # Replace input with output if in-place
    if input_path == output_path and tmp_path != output_path:
        tmp_path.replace(output_path)
        logger.info(f'* Replaced {output_path}')

    logger.info('* Done.')


if __name__ == '__main__':
    main()
