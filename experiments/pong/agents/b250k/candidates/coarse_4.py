"""[candidate: N_RECIP=4 N_SQ1=4 N_SQ2=4]

Constructed (untrained) Pong policy for agent b250k.

The network computes, in closed form, where the ball will cross x = 1 and
steers the paddle towards it:

    crossing = fold(u),   u = y + (VY_MAX / VX_MAX) * vyn * (1 - x) / vxn

where fold is the reflection off both walls (a triangle wave with period 2).
This is exact physics: the current folded state (y, signed vy) continued in a
straight line for the remaining (1 - x) / vx steps gives the same crossing
height as the original launch (see EXPLANATION.md, section 0).

Layer plan (every unit is ReLU(z - knot) of some named affine quantity z):

  L1 (64): y and p pass-through; r(v) ~ 1/vxn (piecewise linear, PWL);
           squares of z = 2(1-x) +/- vyn, whose difference / 8 is
           P = vyn * (1 - x)                      (polarisation identity)
  L2 (64): y and p pass-through; squares of z = 2 r +/- P, whose
           difference / 8 is M = P * r = vyn (1 - x) / vxn
  L3 (10): p pass-through; fold(u) as an exact PWL with knots at the integers,
           u = y + 1.5 M
  out (3): e = fold(u) - p;  logits = [-e, DEADBAND, +e]   (down, stay, up)

Every weight is produced by `build()` from the named constants below; nothing
is trained or fitted. `build()` also attaches `model.provenance`, a per-layer
list naming what each hidden unit computes.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# ---------------------------------------------------------------- physics
# Copied from src/wsp/pong/game.py (the observation is normalised by these).
VX_MAX = 0.04  # vxn = vx / VX_MAX, in [0.5, 1]
VY_MAX = 0.06  # vyn = vy / VY_MAX, in [-1, 1]
VXN_MIN = 0.02 / VX_MAX  # = 0.5
PADDLE_SPEED = 0.03
SLOPE_GAIN = VY_MAX / VX_MAX  # dy/dx of the ball = SLOPE_GAIN * vyn / vxn = 1.5 * ...

# ---------------------------------------------------------------- design constants
# Segment counts of the three piecewise-linear (PWL) approximations.
# Layer 1 has 64 units = 2 pass-through + N_RECIP + 2 * N_SQ1.
N_RECIP = 4  # segments for r(v) ~ 1/v on [0.5, 1]
N_SQ1 = 4  # segments for each square in the bilinear product vyn * (1 - x)
N_SQ2 = 4  # segments for each square in the product P * r (layer 2: 2 + 2*31 = 64)
# Scale constants in the polarisation identity a*b = ((c a + b)^2 - (c a - b)^2) / (4 c).
# The PWL error of z^2 is h^2/4 with h = range / N, and range = c + 2 for both
# products, so the error per unit product is proportional to (c + 2)^2 / c,
# which is minimised at c = 2.
C1 = 2.0  # for P = (1 - x) * vyn: z = C1 (1 - x) +/- vyn, z in [-1, C1 + 1]
C2 = 2.0  # for M = r * P:        z = C2 r +/- P,       z in [C2 - 1, 2 C2 + 1]
# Safety margin on the layer-2 square domain: its input already carries the
# (small) layer-1 approximation errors, so the domain is widened slightly.
SQ2_MARGIN = 0.02
# Knot placement for 1/v: equal-error spacing is uniform in s = v^(-1/2)
# (see EXPLANATION.md); RECIP_SHIFT = 1 subtracts half of the per-segment
# interpolation error so the error is centred around 0 (minimax offset).
RECIP_SHIFT = 1.0
# fold() domain: u lies in [0 - 3, 1 + 3] because |SLOPE_GAIN * vyn / vxn| <= 3.
FOLD_LO, FOLD_HI = -3.25, 4.25
# Dead band of the controller: stay if |target - paddle| <= DEADBAND.
# PADDLE_SPEED / 2 is the smallest band that cannot oscillate: a move from just
# outside the band lands at most PADDLE_SPEED / 2 on the other side.
DEADBAND = PADDLE_SPEED / 2
# Stress-test / ablation switches. The final policy uses AIM_OFFSET = 0 and
# WALLS = "reflect"; METHOD.md records the runs that changed them.
AIM_OFFSET = 0.0  # deliberate aiming error added to the target (closed-loop slack test)
WALLS = "reflect"  # "reflect" = true physics (fold); "clamp" = ablation without bounces

N_IN = 5
IX, IY, IVX, IVY, IP = range(N_IN)  # observation layout (game.OBS_NAMES)


# ---------------------------------------------------------------- helpers
def pwl_terms(f, knots):
    """PWL interpolant of f through `knots`, written as ReLU terms.

    Returns (const, [(knot, coef), ...]) such that, for z >= knots[0],
        f_hat(z) = const + sum_i coef_i * ReLU(z - knot_i),
    with f_hat(knot_j) = f(knot_j) for every knot. coef_0 is the first slope
    and coef_i (i > 0) is the change of slope at knot_i. The last knot only
    closes the domain and needs no unit.
    """
    vals = [f(k) for k in knots]
    slopes = [
        (vals[i + 1] - vals[i]) / (knots[i + 1] - knots[i])
        for i in range(len(knots) - 1)
    ]
    coefs = [slopes[0]] + [slopes[i] - slopes[i - 1] for i in range(1, len(slopes))]
    return vals[0], list(zip(knots[:-1], coefs))


def uniform_knots(lo, hi, n):
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def recip_knots(n):
    """Knots on [0.5, 1] uniform in s = v^(-1/2) (equal PWL error for 1/v)."""
    s_lo, s_hi = VXN_MIN**-0.5, 1.0
    return sorted((s_lo + (s_hi - s_lo) * i / n) ** -2 for i in range(n + 1))


class Expr:
    """Affine expression  w . h + b  over the units h of the previous layer."""

    def __init__(self, n, w=None, b=0.0):
        self.w = torch.zeros(n, dtype=torch.float64) if w is None else w
        self.b = float(b)

    @staticmethod
    def unit(n, i, scale=1.0):
        e = Expr(n)
        e.w[i] = scale
        return e

    def __add__(self, o):
        if isinstance(o, Expr):
            return Expr(len(self.w), self.w + o.w, self.b + o.b)
        return Expr(len(self.w), self.w.clone(), self.b + o)

    __radd__ = __add__

    def __mul__(self, s):
        return Expr(len(self.w), self.w * s, self.b * s)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-o)


class Layer:
    """Collects hidden units ReLU(z - knot) with a provenance label each."""

    def __init__(self, n_prev):
        self.n_prev = n_prev
        self.rows, self.bias, self.labels = [], [], []

    def relu(self, z: Expr, label: str) -> int:
        self.rows.append(z.w.clone())
        self.bias.append(z.b)
        self.labels.append(label)
        return len(self.rows) - 1

    def pwl(self, z: Expr, f, knots, name: str):
        """Add units for f(z); return a placeholder readout (const, [(unit, coef)])."""
        const, terms = pwl_terms(f, knots)
        out = []
        for k, c in terms:
            i = self.relu(z - k, f"{name}: ReLU(z - {k:+.5f}), slope change {c:+.5f}")
            out.append((i, c))
        return const, out

    def linear(self) -> nn.Linear:
        lin = nn.Linear(self.n_prev, len(self.rows), dtype=torch.float64)
        with torch.no_grad():
            lin.weight.copy_(torch.stack(self.rows))
            lin.bias.copy_(torch.tensor(self.bias, dtype=torch.float64))
        return lin


def readout(n, const, terms) -> Expr:
    """Affine expression over a finished layer from a pwl() placeholder."""
    e = Expr(n, b=const)
    for i, c in terms:
        e.w[i] += c
    return e


def square(z):
    return z * z


def fold(u):
    m = math.fmod(u, 2.0)
    m = m + 2.0 if m < 0 else m
    return m if m <= 1.0 else 2.0 - m


# ---------------------------------------------------------------- build
def build() -> nn.Sequential:
    inp = lambda i, s=1.0: Expr.unit(N_IN, i, s)

    # ---- Layer 1: pass-through, 1/vxn, squares for P = (1 - x) vyn
    L1 = Layer(N_IN)
    y1 = L1.relu(inp(IY), "y pass-through (y >= 0)")
    p1 = L1.relu(inp(IP), "paddle pass-through (p >= 0)")
    rk = recip_knots(N_RECIP)
    r_c, r_t = L1.pwl(inp(IVX), lambda v: 1.0 / v, rk, "r=1/vxn")
    one_minus_x = inp(IX, -1.0) + 1.0
    zp = C1 * one_minus_x + inp(IVY)
    zm = C1 * one_minus_x - inp(IVY)
    sq1_knots = uniform_knots(-1.0, C1 + 1.0, N_SQ1)
    sp_c, sp_t = L1.pwl(zp, square, sq1_knots, "(2(1-x)+vyn)^2")
    sm_c, sm_t = L1.pwl(zm, square, sq1_knots, "(2(1-x)-vyn)^2")
    n1 = len(L1.rows)

    # Readouts of layer 1 (affine in its outputs).
    # Minimax offset for r: with knots uniform in s = v^-1/2 the max
    # interpolation error of 1/v on every segment is ds^2 (EXPLANATION.md).
    ds = (VXN_MIN**-0.5 - 1.0) / N_RECIP
    r_hat = readout(n1, r_c, r_t) - RECIP_SHIFT * ds**2 / 2
    P_hat = (readout(n1, sp_c, sp_t) - readout(n1, sm_c, sm_t)) * (1.0 / (4 * C1))
    y_hat1 = Expr.unit(n1, y1)
    p_hat1 = Expr.unit(n1, p1)

    # ---- Layer 2: pass-through, squares for M = r * P
    L2 = Layer(n1)
    y2 = L2.relu(y_hat1, "y pass-through")
    p2 = L2.relu(p_hat1, "paddle pass-through")
    z2p = C2 * r_hat + P_hat
    z2m = C2 * r_hat - P_hat
    sq2_knots = uniform_knots(C2 - 1.0 - SQ2_MARGIN, 2 * C2 + 1.0 + SQ2_MARGIN, N_SQ2)
    qp_c, qp_t = L2.pwl(z2p, square, sq2_knots, "(2r+P)^2")
    qm_c, qm_t = L2.pwl(z2m, square, sq2_knots, "(2r-P)^2")
    n2 = len(L2.rows)
    M_hat = (readout(n2, qp_c, qp_t) - readout(n2, qm_c, qm_t)) * (1.0 / (4 * C2))
    u_hat = Expr.unit(n2, y2) + SLOPE_GAIN * M_hat  # unfolded crossing height
    p_hat2 = Expr.unit(n2, p2)

    # ---- Layer 3: fold(u) with knots at every integer in the domain (exact)
    L3 = Layer(n2)
    p3 = L3.relu(p_hat2, "paddle pass-through")
    fold_knots = (
        [FOLD_LO] + list(range(math.ceil(FOLD_LO), math.floor(FOLD_HI) + 1)) + [FOLD_HI]
    )
    wall = fold if WALLS == "reflect" else (lambda u: min(max(u, 0.0), 1.0))
    f_c, f_t = L3.pwl(u_hat, wall, [float(k) for k in fold_knots], f"{WALLS}(u)")
    n3 = len(L3.rows)
    target = readout(n3, f_c, f_t) + AIM_OFFSET  # predicted crossing height in [0, 1]
    err = target - Expr.unit(n3, p3)  # e = target - paddle

    # ---- Output: [down, stay, up] = [-e, DEADBAND, e]
    out = nn.Linear(n3, 3, dtype=torch.float64)
    with torch.no_grad():
        out.weight.copy_(
            torch.stack([(-err).w, torch.zeros(n3, dtype=torch.float64), err.w])
        )
        out.bias.copy_(torch.tensor([(-err).b, DEADBAND, err.b], dtype=torch.float64))

    model = nn.Sequential(
        L1.linear(), nn.ReLU(), L2.linear(), nn.ReLU(), L3.linear(), nn.ReLU(), out
    )
    model.provenance = {
        "layer1": L1.labels,
        "layer2": L2.labels,
        "layer3": L3.labels,
        "output": [
            "down = -(target - p)",
            f"stay = DEADBAND = {DEADBAND}",
            "up = target - p",
        ],
    }
    return model
