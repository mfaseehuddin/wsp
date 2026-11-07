"""Constructed (untrained) Pong policy for agent b50k.

The network computes, in closed form, where the ball will cross x = 1 and
steers the paddle toward it:

    a      = 1 - ball_x                       (distance still to travel)
    D      = vy * a / vx = 1.5 * vy_n * a / vx_n   (vertical travel, unfolded)
    u      = ball_y + D                       (unfolded crossing height)
    target = fold(u)                          (triangle wave: wall bounces)
    e      = target - paddle_y
    logits = [-e, EPS, +e]                    -> down / stay / up

Layer plan (5 -> 64 -> 62 -> 10 -> 3):
  hidden 1: y, p pass-through; 1/vx_n (piecewise linear); squares of
            s1 = vy_n/sqrt2 + sqrt2*(0.5-x) and d1 = vy_n/sqrt2 - sqrt2*(0.5-x)
            so that (s1^2 - d1^2)/4 = vy_n*(0.5-x).
  hidden 2: y, p pass-through; squares of s2 = c/sqrt3 + sqrt3*w',
            d2 = c/sqrt3 - sqrt3*w' with c = 1.5*vy_n*a, w' = 1/vx_n - 1.5,
            so that (s2^2 - d2^2)/4 = c*w'.  D = c*w' + 1.5*c.
  hidden 3: fold(u) as a sum of ReLU hinges at the integers; p pass-through.
  output  : e = fold(u) - p compared against the dead band EPS.

Every weight is produced by one of the named helpers below; see
EXPLANATION.md for the derivations. No training or data fitting is used.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# ---------------- physics constants (from src/wsp/pong/game.py) -------------
VX_MAX = 0.04
VY_MAX = 0.06
VY_OVER_VX = VY_MAX / VX_MAX  # 1.5: converts vy_n / vx_n to vy / vx
VXN_MIN, VXN_MAX = 0.5, 1.0  # vx_n = vx / VX_MAX ranges over [0.5, 1]
W_CENTER = 1.5  # midpoint of 1/vx_n in [1, 2]

# ---------------- approximation resolution (interpretable knobs) ------------
K_INV = 14  # pieces of the piecewise-linear 1/vx_n (knots uniform in 1/sqrt(v))
M1 = 12  # knots per side for the squares in hidden layer 1
M2 = 15  # knots per side for the squares in hidden layer 2
R1 = math.sqrt(2.0)  # |s1|, |d1| <= 1/sqrt2 + sqrt2*0.5 = sqrt2
R2 = 1.75  # |s2|, |d2| <= 1.5/sqrt3 + sqrt3*0.5 = sqrt3 = 1.732, plus margin
LAMBDA1 = 1.0 / math.sqrt(2.0)  # balances |vy_n|<=1 against |0.5-x|<=0.5
LAMBDA2 = 1.0 / math.sqrt(3.0)  # balances |c|<=1.5 against |w'|<=0.5

# ---------------- controller --------------------------------------------------
EPS = 0.015  # dead band: stay if |target - paddle| <= EPS (> half a paddle step)
TARGET_SHIFT = 0.064  # diagnostic only: deliberately aim this far above the target
FOLD_LO, FOLD_HI = -4, 5  # hinges of the triangle wave at integers in [-4, 4]

X, Y, VXN, VYN, P = range(5)  # observation indices


class Layer:
    """Accumulates ReLU units; each unit is relu(w . prev + b)."""

    def __init__(self, n_in: int):
        self.n_in = n_in
        self.rows: list[torch.Tensor] = []
        self.bias: list[float] = []
        self.names: list[str] = []

    def unit(self, w: torch.Tensor, b: float, name: str) -> int:
        self.rows.append(w.clone())
        self.bias.append(float(b))
        self.names.append(name)
        return len(self.rows) - 1

    def linear(self) -> nn.Linear:
        lin = nn.Linear(self.n_in, len(self.rows), dtype=torch.float64)
        with torch.no_grad():
            lin.weight.copy_(torch.stack(self.rows))
            lin.bias.copy_(torch.tensor(self.bias, dtype=torch.float64))
        return lin


class Form:
    """A linear form over a layer's outputs: value = w . h + b."""

    def __init__(self, n: int, w=None, b: float = 0.0):
        self.w = torch.zeros(n, dtype=torch.float64) if w is None else w
        self.b = b

    def __add__(self, o):
        return Form(len(self.w), self.w + o.w, self.b + o.b)

    def __sub__(self, o):
        return Form(len(self.w), self.w - o.w, self.b - o.b)

    def __mul__(self, k: float):
        return Form(len(self.w), self.w * k, self.b * k)

    __rmul__ = __mul__

    def plus(self, k: float):
        return Form(len(self.w), self.w.clone(), self.b + k)


def basis(n: int, i: int) -> Form:
    f = Form(n)
    f.w[i] = 1.0
    return f


def add_square(layer: Layer, z: Form, m: int, radius: float, tag: str):
    """Units for z^2 on [-radius, radius]: chord interpolation, knots spaced h.

    z^2 ~= h*|z| + sum_{i=1}^{m-1} 2h*relu(|z| - i h),  |z| split into relu(+z),
    relu(-z). Returns (index, coefficient) pairs for the square readout and for
    the exact linear readout z = relu(z) - relu(-z).
    """
    h = radius / m
    sq, lin = [], []
    for i in range(m):
        t = i * h
        coef = h if i == 0 else 2.0 * h
        for sign in (+1.0, -1.0):
            k = layer.unit(sign * z.w, sign * z.b - t, f"{tag}:sq{'+' if sign > 0 else '-'}{i}")
            sq.append((k, coef))
            if i == 0:
                lin.append((k, sign))
    return sq, lin


def readout(n: int, pairs, scale: float = 1.0) -> Form:
    f = Form(n)
    for k, c in pairs:
        f.w[k] += c * scale
    return f


def inv_knots(k: int) -> list[float]:
    """Knots for 1/v on [0.5, 1], uniform in 1/sqrt(v) (equalises chord error)."""
    lo, hi = 1.0 / math.sqrt(VXN_MAX), 1.0 / math.sqrt(VXN_MIN)
    return [1.0 / (hi - j * (hi - lo) / k) ** 2 for j in range(k + 1)]


def build() -> nn.Sequential:
    n0 = 5

    # ---------------- hidden layer 1 ----------------
    L1 = Layer(n0)
    iy1 = L1.unit(basis(n0, Y).w, 0.0, "pass:y")
    ip1 = L1.unit(basis(n0, P).w, 0.0, "pass:p")

    # 1/vx_n as a chord interpolant, shifted down by half its max error
    v = inv_knots(K_INV)
    inv_err = ((1.0 / math.sqrt(VXN_MIN) - 1.0) / K_INV) ** 2
    inv_units = []
    prev_slope = 0.0
    for j in range(K_INV):
        slope = -1.0 / (v[j] * v[j + 1])
        k = L1.unit(basis(n0, VXN).w, -v[j], f"inv:knot{j}@{v[j]:.4f}")
        inv_units.append((k, slope - prev_slope))
        prev_slope = slope

    # vy_n * (0.5 - x) via difference of squares
    a_c = (basis(n0, X) * -1.0).plus(0.5)  # 0.5 - x = a - 0.5
    s1 = basis(n0, VYN) * LAMBDA1 + a_c * (1.0 / LAMBDA1)
    d1 = basis(n0, VYN) * LAMBDA1 - a_c * (1.0 / LAMBDA1)
    sq_s1, lin_s1 = add_square(L1, s1, M1, R1, "s1")
    sq_d1, lin_d1 = add_square(L1, d1, M1, R1, "d1")

    n1 = len(L1.rows)
    y_1 = readout(n1, [(iy1, 1.0)])
    p_1 = readout(n1, [(ip1, 1.0)])
    prod1 = readout(n1, sq_s1, 0.25) - readout(n1, sq_d1, 0.25)  # vy_n*(0.5-x)
    vyn_1 = (readout(n1, lin_s1) + readout(n1, lin_d1)) * (1.0 / (2 * LAMBDA1))
    c_1 = (prod1 + vyn_1 * 0.5) * VY_OVER_VX  # 1.5 * vy_n * (1 - x)
    w_1 = readout(n1, inv_units).plus(1.0 / VXN_MIN - inv_err / 2 - W_CENTER)  # 1/vx_n - 1.5

    # ---------------- hidden layer 2 ----------------
    L2 = Layer(n1)
    iy2 = L2.unit(y_1.w, y_1.b, "pass:y")
    ip2 = L2.unit(p_1.w, p_1.b, "pass:p")
    s2 = c_1 * LAMBDA2 + w_1 * (1.0 / LAMBDA2)
    d2 = c_1 * LAMBDA2 - w_1 * (1.0 / LAMBDA2)
    sq_s2, lin_s2 = add_square(L2, s2, M2, R2, "s2")
    sq_d2, lin_d2 = add_square(L2, d2, M2, R2, "d2")

    n2 = len(L2.rows)
    c_2 = (readout(n2, lin_s2) + readout(n2, lin_d2)) * (1.0 / (2 * LAMBDA2))
    cw_2 = readout(n2, sq_s2, 0.25) - readout(n2, sq_d2, 0.25)  # c * (1/vx_n - 1.5)
    D_2 = cw_2 + c_2 * W_CENTER  # vy * (1 - x) / vx
    u_2 = readout(n2, [(iy2, 1.0)]) + D_2  # unfolded crossing height
    p_2 = readout(n2, [(ip2, 1.0)])

    # ---------------- hidden layer 3: fold ----------------
    L3 = Layer(n2)
    fold_terms = []
    for t in range(FOLD_LO, FOLD_HI):
        k = L3.unit(u_2.w, u_2.b - t, f"fold:hinge@{t}")
        # slope of fold on [t, t+1] is +1 if t even else -1; base slope from 0
        coef = 1.0 if t == FOLD_LO else (2.0 if t % 2 == 0 else -2.0)
        fold_terms.append((k, coef))
    ip3 = L3.unit(p_2.w, p_2.b, "pass:p")
    n3 = len(L3.rows)
    # fold(FOLD_LO) = 0 because FOLD_LO is even
    target = readout(n3, fold_terms).plus(TARGET_SHIFT)
    e = target - readout(n3, [(ip3, 1.0)])

    # ---------------- output ----------------
    out = nn.Linear(n3, 3, dtype=torch.float64)
    with torch.no_grad():
        out.weight.copy_(torch.stack([-e.w, torch.zeros(n3, dtype=torch.float64), e.w]))
        out.bias.copy_(torch.tensor([-e.b, EPS, e.b], dtype=torch.float64))

    model = nn.Sequential(L1.linear(), nn.ReLU(), L2.linear(), nn.ReLU(), L3.linear(), nn.ReLU(), out)
    model.unit_names = [L1.names, L2.names, L3.names]  # provenance
    return model
