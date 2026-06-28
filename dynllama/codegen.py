"""
Expression parser and header generator for dynllama custom ops.

GGUF key convention:  dynllama.custom_ops.<name>
Value:                a single C expression using index variable 'i', e.g.
                          n[i] = a[i] + b[i]
                          n[i] = a[i] * scale
                          n[i] = a[i] / (1.0f + expf(-a[i]))

Generated header matches the pattern in dynllama-headers/cuda/ops/*.h:
  - #ifdef __CUDACC__ block with extern "C" __global__ kernel
  - #else block with static inline CPU fallback
  - Additional <name>_launch() wrapper for target-agnostic call sites
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

_MATH_FUNCS: set[str] = {
    'exp', 'expf', 'log', 'logf', 'sqrt', 'sqrtf',
    'sin', 'sinf', 'cos', 'cosf', 'tan', 'tanf',
    'abs', 'fabsf', 'fabs', 'pow', 'powf',
    'max', 'min', 'fmax', 'fmin', 'fmaxf', 'fminf',
    'tanh', 'tanhf', 'atan', 'atanf', 'atan2', 'atan2f',
    'ceil', 'ceilf', 'floor', 'floorf',
}


@dataclass
class CustomOp:
    name:        str
    expr:        str
    out_var:     str             = ''
    in_vars:     list[str]      = field(default_factory=list)
    scalar_vars: list[str]      = field(default_factory=list)


def parse(name: str, expr: str) -> CustomOp:
    """Parse an element-wise expression body into a CustomOp descriptor.

    Raises ValueError when the expression does not match the expected form.
    """
    op   = CustomOp(name=name, expr=expr.strip())
    expr = op.expr

    # --- output variable: the identifier before [i] on the LHS of = ---
    m = re.search(r'(\w+)\s*\[i\]\s*=(?!=)', expr)
    if not m:
        raise ValueError(f"'{name}': no 'var[i] =' pattern found in {expr!r}")
    op.out_var = m.group(1)

    # --- RHS starts after the first bare '=' (not '==') ---
    rhs_start = re.search(r'(?<![=!<>])=(?!=)', expr)
    if not rhs_start:
        raise ValueError(f"'{name}': cannot locate RHS in {expr!r}")
    rhs = expr[rhs_start.end():]

    # --- input arrays: var[i] on RHS (preserving first-occurrence order) ---
    seen_in: dict[str, None] = {}
    for m2 in re.finditer(r'(\w+)\s*\[i\]', rhs):
        v = m2.group(1)
        if v != op.out_var:
            seen_in.setdefault(v)
    op.in_vars = list(seen_in)

    # --- scalar parameters: bare identifiers not in array set, not 'i', not math ---
    array_vars = {op.out_var} | set(op.in_vars)
    seen_sc: dict[str, None] = {}
    for m3 in re.finditer(r'\b([a-zA-Z_]\w*)\b', rhs):
        v = m3.group(1)
        if v in ('i',) or v in array_vars or v in _MATH_FUNCS:
            continue
        seen_sc.setdefault(v)
    op.scalar_vars = list(seen_sc)

    return op


def _params(op: CustomOp) -> str:
    parts  = [f'float * {op.out_var}']
    parts += [f'const float * {v}' for v in op.in_vars]
    parts += [f'float {v}'         for v in op.scalar_vars]
    parts.append('int n')
    return ', '.join(parts)


def _args(op: CustomOp) -> str:
    parts  = [op.out_var]
    parts += op.in_vars
    parts += op.scalar_vars
    parts.append('n')
    return ', '.join(parts)


def gen_header(op: CustomOp) -> str:
    """Return the text of a self-contained .h file for the op.

    Pattern mirrors dynllama-headers/cuda/ops/*.h: no #pragma once,
    #ifdef __CUDACC__ for GPU kernel, #else for CPU inline fallback.
    A <name>_launch() wrapper is added so callers need not write <<<>>> syntax.
    """
    p = _params(op)
    a = _args(op)
    lines = [
        f'// custom op: {op.name}',
        f'// expr: {op.expr}',
        '',
        '#ifdef __CUDACC__',
        'extern "C" __global__',
        f'void {op.name}({p}) {{',
        '    int i = blockIdx.x * blockDim.x + threadIdx.x;',
        '    if (i < n) {',
        f'        {op.expr};',
        '    }',
        '}',
        '',
        f'static inline void {op.name}_launch({p}) {{',
        '    dim3 block(256);',
        '    dim3 grid((n + 255) / 256);',
        f'    {op.name}<<<grid, block>>>({a});',
        '}',
        '',
        '#else',
        '#include <cmath>',
        f'static inline void {op.name}({p}) {{',
        '    for (int i = 0; i < n; i++) {',
        f'        {op.expr};',
        '    }',
        '}',
        '',
        f'static inline void {op.name}_launch({p}) {{',
        f'    {op.name}({a});',
        '}',
        '',
        '#endif',
        '',
    ]
    return '\n'.join(lines)


def gen_bundle_header(op_names: list[str]) -> str:
    """Return custom_ops_all.h that #includes every generated op header."""
    lines = ['// auto-generated - do not edit', '#pragma once', '']
    for name in op_names:
        lines.append(f'#include "custom_ops/{name}.h"')
    lines.append('')
    return '\n'.join(lines)


def write_custom_ops(ops: list[CustomOp], out_dir: Path) -> None:
    """Write per-op headers and the bundle header into out_dir."""
    ops_dir = out_dir / 'custom_ops'
    ops_dir.mkdir(parents=True, exist_ok=True)

    for op in ops:
        (ops_dir / f'{op.name}.h').write_text(gen_header(op), encoding='utf-8')

    (out_dir / 'custom_ops_all.h').write_text(
        gen_bundle_header([op.name for op in ops]), encoding='utf-8')
