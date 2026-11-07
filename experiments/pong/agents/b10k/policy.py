"""Constructed (never trained) ReLU policy for wsp Pong -- agent b10k.

Idea in one line: the crossing height is fold(U) with
    U = y + (vy/vx) (1 - x) = y + 1.5 * b * (1 - x) / a          (a = vx_n, b = vy_n)
and, because a > 0, the sign of fold(U) - p equals the sign of
    a * fold(U) - a * p = a * fold(V / a) - a * p,     V = a*y + 1.5*b - 1.5*b*x.
`a * fold(V / a)` is the *perspective* of the triangle wave `fold`, which is an
exact sum of ReLUs  sum_k c_k ReLU(V - k a).  So the only things that are not
exactly piecewise-linear are the three products a*y, b*x and a*p, each built
from the polarisation identity  u*v = ((u+v)^2 - (u-v)^2) / 4  with the squares
written as piecewise-linear chord interpolants (ReLU hinges on a uniform knot grid).

Layer 1 (64 ReLU):  a, b+, b-, p pass-through; hinge grids for (a+y)^2, (a-y)^2,
                    (b+x)^2, (b-x)^2.          -> V and a*.., exactly linear in h1
Layer 2 (64 ReLU):  9 perspective-fold hinges ReLU(V - k a), k=-4..4; a pass-through;
                    hinge grids for (a+p)^2, (a-p)^2.
Output (3 logits):  e = a*fold(U) - a*p   (signed, scaled paddle error)
                    down = -e - DEADBAND*a,  stay = 0,  up = e - DEADBAND*a

See EXPLANATION.md for derivations and error bounds.  Every unit carries a
label in `UNIT_LABELS` (filled by build()) for provenance.
"""

from __future__ import annotations

import torch
from torch import nn

DTYPE = torch.float64

# ---- physics constants restated from src/wsp/pong/game.py -------------------
VX_MAX = 0.04  # vx = VX_MAX * vx_n
VY_MAX = 0.06  # vy = VY_MAX * vy_n
PADDLE_SPEED = 0.03

# U = y + SLOPE * b * (1 - x) / a : dy per unit dx is (VY_MAX*b)/(VX_MAX*a)
SLOPE = VY_MAX / VX_MAX  # = 1.5

# ---- tunable, interpretable constants ----------------------------------------
# Dead band (in screen heights) around the predicted crossing where the paddle
# stays.  Half a paddle step makes the paddle settle within +-0.015 of target.
DEADBAND = PADDLE_SPEED / 2
# Number of hinge segments per squared variable in layer 1 / layer 2.
# Layer 1 has 64 - 4 pass-throughs = 60 hinges over a total range length of 9
# (1.5 + 1.5 + 3 + 3), i.e. a common knot spacing h1 = 0.15.
SEG_AY = 10  # (a +- y) ranges have length 1.5 -> h = 0.15
SEG_BX = 20  # (b +- x) ranges have length 3.0 -> h = 0.15
# Layer 2 has 64 - 9 fold hinges - 1 pass-through = 54 hinges, 27 per square.
SEG_AP = 27  # (a +- p) ranges have length 1.5 -> h = 0.0556

# Fold kinks: U lies in [-3, 4] exactly (|1.5 b (1-x)/a| <= 3, y in [0,1]);
# one extra integer kink on each side gives a margin of 1 for approximation error.
FOLD_K_MIN, FOLD_K_MAX = -4, 4

# Input indices of the observation [ball_x, ball_y, vx_n, vy_n, paddle_y].
IX, IY, IA, IB, IP = range(5)

# Exact ranges of the inputs: x in [0,1), y in [0,1], a in [0.5,1], b in [-1,1], p in [0,1].
RANGE_A_PLUS_Y = (0.5, 2.0)
RANGE_A_MINUS_Y = (-0.5, 1.0)
RANGE_B_PLUS_X = (-1.0, 2.0)
RANGE_B_MINUS_X = (-2.0, 1.0)
RANGE_A_PLUS_P = (0.5, 2.0)
RANGE_A_MINUS_P = (-0.5, 1.0)

UNIT_LABELS: dict[str, list[str]] = {}


class Lin:
    """A linear expression  vec . h + const  over the units h of one layer."""

    def __init__(self, n: int, vec=None, const: float = 0.0):
        self.vec = torch.zeros(n, dtype=DTYPE) if vec is None else vec
        self.const = float(const)

    def __add__(self, o: "Lin") -> "Lin":
        return Lin(len(self.vec), self.vec + o.vec, self.const + o.const)

    def __sub__(self, o: "Lin") -> "Lin":
        return Lin(len(self.vec), self.vec - o.vec, self.const - o.const)

    def __mul__(self, c: float) -> "Lin":
        return Lin(len(self.vec), self.vec * c, self.const * c)

    __rmul__ = __mul__


class LayerBuilder:
    """Collects ReLU units  ReLU(w . inp + bias)  of one hidden layer, with labels."""

    def __init__(self, n_in: int, width: int):
        self.n_in, self.width = n_in, width
        self.W = torch.zeros(width, n_in, dtype=DTYPE)
        self.b = torch.zeros(width, dtype=DTYPE)
        self.labels: list[str] = []

    def unit(self, expr: Lin, shift: float, label: str) -> int:
        """Add unit ReLU(expr - shift); return its index."""
        i = len(self.labels)
        assert i < self.width, "layer full"
        self.W[i] = expr.vec
        self.b[i] = expr.const - shift
        self.labels.append(label)
        return i

    def out(self, terms: dict[int, float], const: float = 0.0) -> Lin:
        """Linear expression over this layer's outputs."""
        v = torch.zeros(self.width, dtype=DTYPE)
        for i, c in terms.items():
            v[i] += c
        return Lin(self.width, v, const)

    def square(self, expr: Lin, lo: float, hi: float, m: int, name: str) -> Lin:
        """Chord interpolant of expr^2 on [lo, hi] with m uniform segments.

        S(s) = lo^2 + (2 lo + h) ReLU(s - lo) + sum_{j=1}^{m-1} 2h ReLU(s - t_j),
        t_j = lo + j h.  It equals s^2 at every knot and overestimates it by
        (s - t_j)(t_{j+1} - s) <= h^2/4 in between.
        """
        h = (hi - lo) / m
        terms = {}
        for j in range(m):
            t = lo + j * h
            i = self.unit(expr, t, f"{name}: hinge at {t:+.4f} (h={h:.4f})")
            # first hinge carries the first chord slope, later ones the slope jump 2h
            terms[i] = (2 * lo + h) if j == 0 else 2 * h
        return self.out(terms, lo * lo)

    def linear(self) -> nn.Linear:
        lin = nn.Linear(self.n_in, self.width).to(DTYPE)
        with torch.no_grad():
            lin.weight.copy_(self.W)
            lin.bias.copy_(self.b)
        return lin


def make(
    deadband: float = DEADBAND,
    seg_ay: int = SEG_AY,
    seg_bx: int = SEG_BX,
    seg_ap: int = SEG_AP,
) -> nn.Sequential:
    # ---------------- inputs as linear expressions ----------------
    def inp(i: int) -> Lin:
        v = torch.zeros(5, dtype=DTYPE)
        v[i] = 1.0
        return Lin(5, v)

    x, y, a, b, p = (inp(i) for i in (IX, IY, IA, IB, IP))

    # ---------------- hidden layer 1 ----------------
    L1 = LayerBuilder(5, 64)
    i_a = L1.unit(a, 0.0, "pass a = vx_n (a >= 0.5 > 0)")
    i_bp = L1.unit(b, 0.0, "pass b+ = ReLU(vy_n)")
    i_bm = L1.unit(-1.0 * b, 0.0, "pass b- = ReLU(-vy_n)")
    i_p = L1.unit(p, 0.0, "pass p = paddle_y (p >= 0)")
    sq_apy = L1.square(a + y, *RANGE_A_PLUS_Y, seg_ay, "(a+y)^2")
    sq_amy = L1.square(a - y, *RANGE_A_MINUS_Y, seg_ay, "(a-y)^2")
    sq_bpx = L1.square(b + x, *RANGE_B_PLUS_X, seg_bx, "(b+x)^2")
    sq_bmx = L1.square(b - x, *RANGE_B_MINUS_X, seg_bx, "(b-x)^2")
    assert len(L1.labels) <= 64

    a1 = L1.out({i_a: 1.0})
    b1 = L1.out({i_bp: 1.0, i_bm: -1.0})
    p1 = L1.out({i_p: 1.0})
    ay1 = (sq_apy - sq_amy) * 0.25  # polarisation: a*y
    bx1 = (sq_bpx - sq_bmx) * 0.25  # polarisation: b*x
    # V = a * U = a*y + SLOPE*b*(1-x)
    V1 = ay1 + SLOPE * b1 - SLOPE * bx1

    # ---------------- hidden layer 2 ----------------
    L2 = LayerBuilder(64, 64)
    fold_terms = {}
    for k in range(FOLD_K_MIN, FOLD_K_MAX + 1):
        i = L2.unit(V1 - k * a1, 0.0, f"perspective fold hinge ReLU(V - ({k})a) = a*ReLU(U - ({k}))")
        # fold(u) = ReLU(u+4) - 2ReLU(u+3) + 2ReLU(u+2) - ... for u >= -4
        fold_terms[i] = 1.0 if k == FOLD_K_MIN else 2.0 * (-1) ** (k - FOLD_K_MIN)
    i_a2 = L2.unit(a1, 0.0, "pass a")
    sq_apP = L2.square(a1 + p1, *RANGE_A_PLUS_P, seg_ap, "(a+p)^2")
    sq_amP = L2.square(a1 - p1, *RANGE_A_MINUS_P, seg_ap, "(a-p)^2")
    assert len(L2.labels) <= 64

    a_fold = L2.out(fold_terms)  # = a * fold(U)
    ap2 = (sq_apP - sq_amP) * 0.25  # = a * p
    a2 = L2.out({i_a2: 1.0})
    e = a_fold - ap2  # = a * (fold(U) - p): positive => target above paddle

    # ---------------- output layer ----------------
    down = (-1.0 * e) - deadband * a2
    up = e - deadband * a2
    out = nn.Linear(64, 3).to(DTYPE)
    with torch.no_grad():
        out.weight.zero_()
        out.bias.zero_()
        out.weight[0], out.bias[0] = down.vec, down.const  # down
        # row 1 (stay) is identically 0
        out.weight[2], out.bias[2] = up.vec, up.const  # up

    UNIT_LABELS.clear()
    UNIT_LABELS["hidden1"] = L1.labels
    UNIT_LABELS["hidden2"] = L2.labels
    UNIT_LABELS["output"] = [
        "down = -a(fold(U)-p) - DEADBAND*a",
        "stay = 0",
        "up = a(fold(U)-p) - DEADBAND*a",
    ]

    # pad unused units (if any) are all-zero rows -> ReLU(0) = 0, no effect
    model = nn.Sequential(L1.linear(), nn.ReLU(), L2.linear(), nn.ReLU(), out)
    model.requires_grad_(False)  # constructed weights; no autograd anywhere
    return model


def build() -> nn.Sequential:
    return make()
