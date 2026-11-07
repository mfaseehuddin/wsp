"""Direct weight files (`.wts`): weights written as literal numbers, no code.

An agent writes every nonzero weight by hand. The format is data only: a
config line, `group` headers that carry the reasoning, and assignments of
literal values to tensor positions. Nothing is computed.

    # comments start with '#'
    config d_model=64 n_layers=2 n_heads=2 d_head=16 d_mlp=128 pos=learned bias=true
    group lex.file | mechanism: file-name tokens light channel 3 | why: copy head keys on it
    embed @budget,@plan,@notes 3 = 1
    layers.0.wq 0 3 0:4 = 2 2 2 2
    b_unembed @<end> = 0.5

Assignment: `TENSOR AXIS AXIS ... = VALUE [VALUE ...]`, one AXIS per tensor
dimension. An AXIS is `*` (all), an int, a range `a:b` or `a:b:s` (stop
exclusive), `@token` (that token's id), or a comma list of ints / @tokens.
One value fills the whole selection; several values must match its size
(row-major). Later assignments overwrite earlier ones. Every assignment
belongs to the current `group`, so provenance is automatic.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from . import model, prov
from . import tokenizer as T

MAX_BYTES = 3_000_000
_NUM = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_FLAGS = {
    "d_model": int, "n_layers": int, "n_heads": int, "d_head": int, "d_mlp": int,
    "pos": str, "norm": bool, "bias": bool, "tie": bool,
}  # fmt: skip


class WtsError(ValueError):
    pass


def _bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise WtsError(f"bad boolean {s!r}")


def _config(fields: list[str], ln: int) -> model.Config:
    kw = {}
    for f in fields:
        if "=" not in f:
            raise WtsError(f"line {ln}: config needs key=value, got {f!r}")
        k, v = f.split("=", 1)
        if k not in _FLAGS:
            raise WtsError(f"line {ln}: unknown config key {k!r}")
        typ = _FLAGS[k]
        kw["tie_embeddings" if k == "tie" else k] = (
            _bool(v) if typ is bool else typ(v)
        )
    cfg = model.Config(**kw)
    cfg.validate()
    return cfg


def _tok(s: str, ln: int) -> int:
    ids = T.token_to_id()
    if s not in ids:
        raise WtsError(f"line {ln}: @{s} is not a vocab token")
    return ids[s]


def _axis(s: str, size: int, ln: int):
    if s == "*":
        return slice(None)
    if ":" in s and not s.startswith("@"):
        parts = s.split(":")
        if len(parts) not in (2, 3) or not all(p == "" or p.lstrip("-").isdigit() for p in parts):
            raise WtsError(f"line {ln}: bad range {s!r}")
        a, b, *c = [int(p) if p else None for p in parts]
        return slice(a, b, c[0] if c else None)
    out = []
    for item in s.split(","):
        if item.startswith("@"):
            out.append(_tok(item[1:], ln))
        elif item.lstrip("-").isdigit():
            out.append(int(item))
        else:
            raise WtsError(f"line {ln}: bad index {item!r}")
        if not -size <= out[-1] < size:
            raise WtsError(f"line {ln}: index {out[-1]} out of range 0..{size - 1}")
    return out[0] if len(out) == 1 else out


def parse(path: str | Path):
    """-> (Config, params float64, groups). Raises WtsError with the line number."""
    text = Path(path).read_text()
    if len(text.encode()) > MAX_BYTES:
        raise WtsError("weight file over 3 MB")
    cfg, params, groups, cur = None, None, [], None
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head = line.split()[0]
        if head == "config":
            if cfg is not None:
                raise WtsError(f"line {ln}: second config line")
            cfg = _config(line.split()[1:], ln)
            params = {
                k: torch.zeros(s, dtype=model.DTYPE)
                for k, s in model.param_shapes(cfg).items()
            }
            continue
        if head == "group":
            parts = [p.strip() for p in line[5:].split("|")]
            if len(parts) != 3 or not all(parts):
                raise WtsError(f"line {ln}: use 'group NAME | mechanism: … | why: …'")
            cur = parts
            continue
        if cfg is None:
            raise WtsError(f"line {ln}: the config line must come first")
        if cur is None:
            raise WtsError(f"line {ln}: assignment before any group header")
        if "=" not in line:
            raise WtsError(f"line {ln}: expected 'TENSOR AXES = VALUES'")
        lhs, rhs = line.split("=", 1)
        lhs, vals = lhs.split(), rhs.split()
        name, axes = lhs[0], lhs[1:]
        if name not in params:
            raise WtsError(f"line {ln}: unknown tensor {name!r}")
        t = params[name]
        if len(axes) != t.ndim:
            raise WtsError(f"line {ln}: {name} has {t.ndim} axes {tuple(t.shape)}, got {len(axes)}")
        if not vals or not all(_NUM.match(v) for v in vals):
            raise WtsError(f"line {ln}: values must be literal numbers (no expressions)")
        idx = tuple(_axis(a, n, ln) for a, n in zip(axes, t.shape, strict=True))
        try:
            sel = t[idx]
            if len(vals) == 1:
                t[idx] = float(vals[0])
            elif len(vals) == sel.numel():
                t[idx] = torch.tensor([float(v) for v in vals], dtype=t.dtype).reshape(sel.shape)
            else:
                raise WtsError(f"line {ln}: {len(vals)} values for {sel.numel()} positions")
        except (IndexError, RuntimeError) as e:
            raise WtsError(f"line {ln}: {e}") from e
        mech, why = cur[1], cur[2]
        groups.append(prov.group(cur[0], name, idx, mechanism=mech, why=why))
    if cfg is None:
        raise WtsError("no config line")
    if not groups:
        groups.append(prov.group("empty", "embed", (0, 0), mechanism="none", why="none"))
    return cfg, params, groups
