"""Closed API for construction programs. Hidden evaluation never runs them.

This is defense in depth, not a general Python sandbox. The builder itself has
no execution tools. The coordinator runs approved source in a timed worker,
before any hidden data exists, and exports only validated numeric arrays.
"""

import ast
import builtins
import math

import torch

BUILTINS = {
    n: getattr(builtins, n)
    for n in (
        "range",
        "enumerate",
        "zip",
        "sorted",
        "min",
        "max",
        "sum",
        "len",
        "abs",
        "int",
        "float",
        "str",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "round",
        "reversed",
        "any",
        "all",
        "isinstance",
        "ord",
        "chr",
    )
}
ATTRS = {
    "zeros",
    "ones",
    "full",
    "tensor",
    "arange",
    "eye",
    "zeros_like",
    "ones_like",
    "stack",
    "cat",
    "outer",
    "sin",
    "cos",
    "tan",
    "sqrt",
    "exp",
    "log",
    "log2",
    "log10",
    "pow",
    "clamp",
    "where",
    "minimum",
    "maximum",
    "float32",
    "float64",
    "long",
    "int64",
    "bool",
    "Tensor",
    "pi",
    "e",
    "inf",
    "shape",
    "ndim",
    "numel",
    "dtype",
    "T",
    "transpose",
    "permute",
    "reshape",
    "view",
    "flatten",
    "unsqueeze",
    "squeeze",
    "expand",
    "repeat",
    "clone",
    "to",
    "float",
    "double",
    "contiguous",
    "copy_",
    "fill_",
    "zero_",
    "add_",
    "mul_",
    "scatter_",
    "scatter_add_",
    "index_add_",
    "index_fill_",
    "masked_fill_",
    "diagonal",
    "triu",
    "tril",
    "abs",
    "square",
    "reciprocal",
    "sum",
    "mean",
    "max",
    "min",
    "clamp_",
    "item",
    "get",
    "items",
    "keys",
    "values",
    "setdefault",
    "update",
    "append",
    "extend",
    "insert",
    "index",
    "count",
    "split",
    "join",
    "lower",
    "upper",
    "strip",
    "startswith",
    "endswith",
    "bit_length",
    "ceil",
    "floor",
    "factorial",
    "comb",
    "gcd",
    "isfinite",
    "prod",
    # Seeded randomness (deterministic given the seed) for dense latent codes.
    "Generator",
    "manual_seed",
    "randn",
    "rand",
    "randint",
    "randperm",
    "normal_",
    "uniform_",
    "sign",
    "topk",
    "argmax",
    "norm",
    "matmul",
    "einsum",
    "softmax",
    "relu",
}
RANDOM = {"randn", "rand", "randint", "randperm", "normal_", "uniform_"}
# item is harmless during construction (no prompts are supplied to build).
ALLOWED_NODES = {
    "Module",
    "Import",
    "alias",
    "FunctionDef",
    "arguments",
    "arg",
    "Return",
    "Assign",
    "AnnAssign",
    "AugAssign",
    "Expr",
    "For",
    "If",
    "Break",
    "Continue",
    "Pass",
    "ListComp",
    "SetComp",
    "DictComp",
    "GeneratorExp",
    "comprehension",
    "Name",
    "Load",
    "Store",
    "Constant",
    "List",
    "Tuple",
    "Dict",
    "Set",
    "Subscript",
    "Slice",
    "Attribute",
    "Call",
    "keyword",
    "BinOp",
    "UnaryOp",
    "BoolOp",
    "Compare",
    "IfExp",
    "JoinedStr",
    "FormattedValue",
    "Add",
    "Sub",
    "Mult",
    "Div",
    "FloorDiv",
    "Mod",
    "Pow",
    "MatMult",
    "LShift",
    "RShift",
    "BitOr",
    "BitAnd",
    "BitXor",
    "USub",
    "UAdd",
    "Not",
    "Invert",
    "And",
    "Or",
    "Eq",
    "NotEq",
    "Lt",
    "LtE",
    "Gt",
    "GtE",
    "In",
    "NotIn",
    "Is",
    "IsNot",
}


def validate_source(source):
    if len(source.encode()) > 1_000_000:
        raise ValueError("construction source exceeds 1MB")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if type(node).__name__ not in ALLOWED_NODES:
            raise ValueError(
                f"line {getattr(node, 'lineno', 0)}: {type(node).__name__} is forbidden"
            )
        if isinstance(node, ast.Import) and (
            node not in tree.body
            or any(a.name not in {"torch", "math"} or a.asname for a in node.names)
        ):
            raise ValueError("only top-level import torch / import math")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names forbidden")
        if isinstance(node, ast.Attribute) and (
            node.attr not in ATTRS or isinstance(node.ctx, ast.Store)
        ):
            raise ValueError(f"attribute {node.attr} forbidden")
        if isinstance(node, ast.FunctionDef) and (
            node.decorator_list or node.name.startswith("__")
        ):
            raise ValueError("decorators and dunder functions forbidden")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (str, bytes))
            and len(node.value) > 100_000
        ):
            raise ValueError("oversized literal")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in RANDOM
            and not any(k.arg == "generator" for k in node.keywords)
        ):
            raise ValueError(
                f"line {node.lineno}: {node.func.attr} needs generator= "
                "(torch.Generator().manual_seed(n)) so builds are deterministic"
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            in {
                "open",
                "eval",
                "exec",
                "compile",
                "getattr",
                "setattr",
                "vars",
                "globals",
                "locals",
                "input",
                "print",
                "__import__",
            }
        ):
            raise ValueError(f"call {node.func.id} forbidden")
    tree.body = [n for n in tree.body if not isinstance(n, ast.Import)]
    ast.fix_missing_locations(tree)
    return tree


def construct(source):
    tree = validate_source(source)
    namespace = {"__builtins__": BUILTINS, "torch": torch, "math": math}
    exec(compile(tree, "<sanitized-construction>", "exec"), namespace)  # noqa: S102 -- validated closed API, no hidden data
    if not callable(namespace.get("build")):
        raise TypeError("build() missing")
    with torch.no_grad():
        return namespace["build"]()


def api_markdown():
    return (
        "# Exact construction allowlist\n\nOnly import torch and math. "
        "No other imports, dunder names, decorators, or attribute assignment.\n\n"
        "Builtins: " + ", ".join(sorted(BUILTINS)) + "\n\n"
        "Attributes/methods: " + ", ".join(sorted(ATTRS)) + "\n\n"
        "AST node kinds: " + ", ".join(sorted(ALLOWED_NODES)) + "\n"
    )
