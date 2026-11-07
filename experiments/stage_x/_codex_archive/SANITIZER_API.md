# Exact construction allowlist

Only import torch and math. No other imports, dunder names, decorators, or attribute assignment.

Builtins: abs, all, any, bool, dict, enumerate, float, int, isinstance, len, list, max, min, range, reversed, round, set, sorted, str, sum, tuple, zip

Attributes/methods: T, Tensor, abs, add_, append, arange, bit_length, bool, cat, ceil, clamp, clamp_, clone, comb, contiguous, copy_, cos, count, diagonal, double, dtype, e, endswith, exp, expand, extend, eye, factorial, fill_, flatten, float, float32, float64, floor, full, gcd, get, index, index_add_, index_fill_, inf, insert, int64, isfinite, item, items, join, keys, log, log10, log2, long, lower, masked_fill_, max, maximum, mean, min, minimum, mul_, ndim, numel, ones, ones_like, outer, permute, pi, pow, prod, reciprocal, repeat, reshape, scatter_, scatter_add_, setdefault, shape, sin, split, sqrt, square, squeeze, stack, startswith, strip, sum, tan, tensor, to, transpose, tril, triu, unsqueeze, update, upper, values, view, where, zero_, zeros, zeros_like

AST node kinds: Add, And, AnnAssign, Assign, Attribute, AugAssign, BinOp, BitAnd, BitOr, BitXor, BoolOp, Break, Call, Compare, Constant, Continue, Dict, DictComp, Div, Eq, Expr, FloorDiv, For, FormattedValue, FunctionDef, GeneratorExp, Gt, GtE, If, IfExp, Import, In, Invert, Is, IsNot, JoinedStr, LShift, List, ListComp, Load, Lt, LtE, MatMult, Mod, Module, Mult, Name, Not, NotEq, NotIn, Or, Pass, Pow, RShift, Return, Set, SetComp, Slice, Store, Sub, Subscript, Tuple, UAdd, USub, UnaryOp, alias, arg, arguments, comprehension, keyword
