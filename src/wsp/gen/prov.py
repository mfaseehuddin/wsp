"""Provenance: every weight must belong to a named, explained group.

In `weights.py`, `build()` returns `(cfg, params, groups)`, where `groups` is a
list made with `group(...)`:

    from wsp.gen.prov import group, S
    groups = [
        group("lexicon.file_names", "embed", S[file_ids, 0:24],
              mechanism="one-hot 'is a file name' features",
              why="each file name token gets +1 on its own column so a copy head can find it"),
        ...
    ]

`S[...]` builds an index exactly as you would write it on the tensor
(rows/cols as ints, slices, lists or LongTensors). A group may cover any
part of one tensor. Groups may overlap.

The harness reports **coverage**: the share of nonzero parameters that fall
inside at least one group. Uncovered nonzero weights are listed by tensor.
`wsp-gen explain` prints every group with its mechanism and reason, and flags
group names that don't appear in EXPLANATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class _Slicer:
    def __getitem__(self, idx):
        return idx


S = _Slicer()


@dataclass
class Group:
    name: str
    tensor: str
    index: Any
    mechanism: str
    why: str


def group(
    name: str, tensor: str, index: Any = ..., *, mechanism: str, why: str
) -> Group:
    if not (mechanism.strip() and why.strip()):
        raise ValueError(f"group {name!r}: mechanism and why must be non-empty")
    return Group(name, tensor, index, mechanism, why)


def _index_str(idx: Any) -> str:
    def one(i):
        if i is Ellipsis:
            return "..."
        if isinstance(i, slice):
            return (
                f"{'' if i.start is None else i.start}:{'' if i.stop is None else i.stop}"
                + (f":{i.step}" if i.step else "")
            )
        if isinstance(i, torch.Tensor):
            i = i.flatten().tolist()
        if isinstance(i, (list, tuple, range)):
            i = list(i)
            return (
                f"[{','.join(map(str, i[:6]))}{',…' if len(i) > 6 else ''}] ({len(i)})"
            )
        return str(i)

    return ", ".join(one(i) for i in idx) if isinstance(idx, tuple) else one(idx)


def coverage(params: dict[str, torch.Tensor], groups: list[Group]) -> dict:
    masks = {k: torch.zeros(t.shape, dtype=torch.bool) for k, t in params.items()}
    rows, errors = [], []
    for g in groups:
        if not isinstance(g, Group):
            errors.append("groups must be made with prov.group(...)")
            continue
        if g.tensor not in params:
            errors.append(f"{g.name}: unknown tensor {g.tensor!r}")
            continue
        sel = torch.zeros(params[g.tensor].shape, dtype=torch.bool)
        try:
            sel[g.index] = True
        except (IndexError, TypeError, RuntimeError) as e:
            errors.append(f"{g.name}: bad index on {g.tensor}: {e}")
            continue
        masks[g.tensor] |= sel
        t = params[g.tensor]
        rows.append(
            {
                "name": g.name,
                "tensor": g.tensor,
                "index": _index_str(g.index),
                "params": int(sel.sum()),
                "nonzero": int(((t != 0) & sel).sum()),
                "mechanism": g.mechanism,
                "why": g.why,
            }
        )
    nz = {k: t != 0 for k, t in params.items()}
    total = int(sum(m.sum() for m in nz.values()))
    covered = int(sum((nz[k] & masks[k]).sum() for k in params))
    uncovered = {k: int((nz[k] & ~masks[k]).sum()) for k in params}
    return {
        "coverage": round(covered / total, 4) if total else 1.0,
        "nonzero": total,
        "covered_nonzero": covered,
        "uncovered_by_tensor": {k: n for k, n in uncovered.items() if n},
        "groups": rows,
        "errors": errors,
    }
