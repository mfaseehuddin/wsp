"""Custom architectures: an agent may supply its own forward function.

`weights.py` returns `(CustomArch("path/to/arch.py", "one-line justification"), params)`.
`arch.py` defines

    def forward(params: dict[str, Tensor], tokens: LongTensor[B, T]) -> Tensor[B, T, V]

and may define `trace(params, tokens) -> dict` (optional, used by `probe --trace`).

The forward pass must be a neural network, not a program in disguise. So
`arch.py` is checked statically, and loading fails if it breaks a rule:
- imports: only torch, torch.nn.functional and math;
- no Python branching or boolean logic on values: no if, while, ternary,
  and/or, assert or try (use torch.where, masks, relu, softmax instead);
- loops only as `for i in range(...)`, e.g. over layers;
- no constants hidden in code: no torch.tensor / as_tensor / from_numpy / load,
  no list, dict or set literal with more than 8 elements, at most 60 numeric literals;
- no .item(), .tolist(), .numpy(), and no open/exec/eval/getattr/print/__import__.
Every learned number must be in `params`, and all of `params` counts toward size.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

import torch

ALLOWED_IMPORTS = {"torch", "torch.nn.functional", "math"}
BANNED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "vars",
    "input",
    "print",
    "breakpoint",
}
BANNED_ATTRS = {
    "item",
    "tolist",
    "numpy",
    "tensor",
    "as_tensor",
    "from_numpy",
    "load",
    "save",
    "data_ptr",
    "__dict__",
    "storage",
}
BANNED_NODES = (
    ast.If,
    ast.While,
    ast.IfExp,
    ast.BoolOp,
    ast.Assert,
    ast.Try,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
    ast.With,
    ast.AsyncFunctionDef,
    ast.Await,
)
MAX_PARAMS = 5_000_000


class ArchError(ValueError):
    pass


def validate_source(src: str) -> None:
    tree = ast.parse(src)
    nums = 0
    for node in ast.walk(tree):
        if isinstance(node, BANNED_NODES):
            raise ArchError(
                f"line {node.lineno}: {type(node).__name__} is not allowed in arch.py"
            )
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    raise ArchError(f"import {a.name} not allowed")
        if isinstance(node, ast.ImportFrom):
            full = {f"{node.module}.{a.name}" for a in node.names} | {node.module or ""}
            if not full & ALLOWED_IMPORTS or node.module not in (
                "torch",
                "torch.nn",
                "math",
            ):
                raise ArchError(f"from {node.module} import ... not allowed")
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise ArchError(f"line {node.lineno}: {node.id} not allowed")
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            raise ArchError(f"line {node.lineno}: .{node.attr} not allowed")
        if isinstance(node, ast.For) and not (
            isinstance(node.iter, ast.Call)
            and getattr(node.iter.func, "id", None) == "range"
        ):
            raise ArchError(
                f"line {node.lineno}: only `for _ in range(...)` loops are allowed"
            )
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and len(node.elts) > 8:
            raise ArchError(f"line {node.lineno}: literal with more than 8 elements")
        if isinstance(node, ast.Dict) and len(node.keys) > 8:
            raise ArchError(
                f"line {node.lineno}: dict literal with more than 8 entries"
            )
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            nums += 1
    if nums > 60:
        raise ArchError(
            f"{nums} numeric literals in arch.py (max 60); put numbers in params"
        )
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "forward" not in names:
        raise ArchError("arch.py must define forward(params, tokens)")


class CustomArch:
    blocked = False  # set by the harness while build() runs

    def __init__(self, arch_path: str | Path, justification: str):
        self.path = Path(arch_path)
        self.justification = justification
        src = self.path.read_text()
        validate_source(src)
        self.sha = hashlib.sha256(src.encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(f"arch_{self.sha}", self.path)
        self._mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._mod)

    def forward(self, params, tokens, trace: bool = False):
        if CustomArch.blocked:
            raise RuntimeError(
                "forward passes are not allowed inside build(); use the harness"
            )
        with torch.no_grad():
            logits = self._mod.forward(params, tokens)
            if trace:
                tr = (
                    self._mod.trace(params, tokens)
                    if hasattr(self._mod, "trace")
                    else {}
                )
                return logits, tr
        return logits

    def to_dict(self) -> dict:
        return {
            "custom": str(self.path),
            "arch_sha": self.sha,
            "justification": self.justification,
        }

    def __eq__(self, other) -> bool:
        return isinstance(other, CustomArch) and self.sha == other.sha

    def __hash__(self) -> int:
        return hash(self.sha)


def check_params(params: dict) -> dict[str, torch.Tensor]:
    out = {}
    for k, t in params.items():
        if not isinstance(k, str):
            raise ArchError("param names must be strings")
        t = torch.as_tensor(t).to(torch.float64)
        if not torch.isfinite(t).all():
            raise ArchError(f"{k}: non-finite values")
        out[k] = t
    if sum(t.numel() for t in out.values()) > MAX_PARAMS:
        raise ArchError(f"more than {MAX_PARAMS:,} parameters")
    return out
