"""Agent b0: a constructed (untrained) ReLU network that plays Pong.

Budget: 0 forward passes. Every weight below comes from a formula derived
from the physics in src/wsp/pong/game.py; see EXPLANATION.md for the
derivations and METHOD.md for the hand-checked worked examples.

The network computes, from one observation [x, y, vx_n, vy_n, p],

    crossing = fold( y + (VY_MAX/VX_MAX) * vy_n * (1 - x) * (1 / vx_n) )

which is the *exact* height where the ball will cross x = 1 (bounces
included), and then moves the paddle toward it.

Layer plan (5 -> 62 -> 64 -> 10 -> 3):

  H1 (62)  Y1, P1 passthroughs; S_j, Q_j  hinge features of s = vy_n + m and
           q = m - vy_n (m = 1 - x) for the product a = vy_n*m = (s^2 - q^2)/4;
           V_i = ReLU(vx_n - v_i) speed thresholds where 1/vx_n crosses a cell edge.
  H2 (64)  Y2, P2 passthroughs; G+_i, G-_i = ReLU(+-a - GATE_GAIN * V_i): the
           positive / negative part of a, passed only if 1/vx_n > t_i.
           Summing c_i (G+_i - G-_i) gives a * r_hat with r_hat a 31-cell
           midpoint staircase of r = 1/vx_n.
  H3 (10)  F_b = ReLU(z - b), b = -4..4, with z = y + 1.5 * a * r_hat; P3.
  Out (3)  T = fold(z) = sum phi_b F_b;  logits [P - T, DEADBAND, T - P].
"""

from __future__ import annotations

import torch
from torch import nn

from wsp.pong.game import PADDLE_SPEED, VX_MAX, VY_MAX

DT = torch.float64

# ---- observation layout (game.OBS_NAMES) ----
IN_X, IN_Y, IN_VXN, IN_VYN, IN_P = range(5)

# ---- physics-derived constants ----
# Remaining unfolded displacement = vy * (1 - x) / vx
#   = (VY_MAX * vy_n) * (1 - x) / (VX_MAX * vx_n) = D_SCALE * vy_n * m * r.
D_SCALE = VY_MAX / VX_MAX  # = 1.5

# ---- product a = vy_n * m via (s^2 - q^2) / 4, s = vy_n + m, q = m - vy_n ----
# s and q both lie in [-1, 2] (vy_n in [-1, 1], m in (0, 1]).
SQ_LO, SQ_HI = -1.0, 2.0
SQ_H = 0.2  # knot spacing; 2 / SQ_H is an integer so the knots are symmetric about 0
SQ_J = round((SQ_HI - SQ_LO) / SQ_H)  # 15 intervals -> 15 units per square
SQ_KNOTS = [
    SQ_LO + j * SQ_H for j in range(SQ_J)
]  # k_0 .. k_14 (k_15 = 2 needs no unit)
# Piecewise-linear interpolant of s^2 on knots k_j:
#   f(s) = k_0^2 + (2 k_0 + h) ReLU(s - k_0) + 2 h * sum_{j>=1} ReLU(s - k_j)
SQ_W0 = 2.0 * SQ_LO + SQ_H  # slope of the first chord = -1.8
SQ_WJ = 2.0 * SQ_H  # slope increment at each interior knot = 0.4

# ---- staircase of r = 1 / vx_n over [1, 2] ----
R_CELLS = 31
R_DELTA = 1.0 / R_CELLS
# Cell edges t_i = 1 + i*R_DELTA (i = 1..30). "r > t_i" <=> "vx_n < v_i = 1/t_i".
R_EDGES = [1.0 + i * R_DELTA for i in range(1, R_CELLS)]
VX_THRESH = [1.0 / t for t in R_EDGES]
# r_hat = (1 + R_DELTA/2) + R_DELTA * #{i : r > t_i}  (= midpoint of r's cell)
GATE_COEF = [1.0 + R_DELTA / 2] + [R_DELTA] * (
    R_CELLS - 1
)  # c_0 (always on), c_1..c_30
GATE_GAIN = 1.0e4  # closes a gate once vx_n exceeds v_i by |a| / GATE_GAIN <= ~1e-4

# ---- fold (triangle wave, period 2) on z in [-4, 5] ----
FOLD_BREAKS = list(range(-4, 5))  # -4..4
# fold(z) = ReLU(z+4) - 2ReLU(z+3) + 2ReLU(z+2) - ... + 2ReLU(z-4)   for z >= -4
FOLD_COEF = [1.0] + [(-2.0 if k % 2 == 1 else 2.0) for k in range(1, len(FOLD_BREAKS))]

# ---- controller ----
DEADBAND = PADDLE_SPEED / 2  # stay if |T - p| <= 0.015 (half a paddle step)

# ---- unit index maps (provenance) ----
# H1
H1_Y, H1_P = 0, 1
H1_S = list(range(2, 2 + SQ_J))  # 2..16
H1_Q = list(range(2 + SQ_J, 2 + 2 * SQ_J))  # 17..31
H1_V = list(range(2 + 2 * SQ_J, 2 + 2 * SQ_J + len(VX_THRESH)))  # 32..61, gates i=1..30
H1_N = 2 + 2 * SQ_J + len(VX_THRESH)  # 62
# H2
H2_Y, H2_P = 0, 1
H2_GP = list(range(2, 2 + R_CELLS))  # 2..32  G+_0..G+_30
H2_GN = list(range(2 + R_CELLS, 2 + 2 * R_CELLS))  # 33..63 G-_0..G-_30
H2_N = 2 + 2 * R_CELLS  # 64
# H3
H3_F = list(range(len(FOLD_BREAKS)))  # 0..8
H3_P = len(FOLD_BREAKS)  # 9
H3_N = len(FOLD_BREAKS) + 1  # 10
# Output
OUT_DOWN, OUT_STAY, OUT_UP = 0, 1, 2

UNIT_NAMES = {
    "h1": ["Y1", "P1"]
    + [f"S{j}(s-{k:+.1f})" for j, k in enumerate(SQ_KNOTS)]
    + [f"Q{j}(q-{k:+.1f})" for j, k in enumerate(SQ_KNOTS)]
    + [f"V{i}(vx_n-{v:.5f})" for i, v in enumerate(VX_THRESH, start=1)],
    "h2": ["Y2", "P2"]
    + [f"G+{i}" for i in range(R_CELLS)]
    + [f"G-{i}" for i in range(R_CELLS)],
    "h3": [f"F(z{-b:+d})" for b in FOLD_BREAKS] + ["P3"],
    "out": ["down", "stay", "up"],
}


def _layer1() -> tuple[torch.Tensor, torch.Tensor]:
    W = torch.zeros(H1_N, 5, dtype=DT)
    b = torch.zeros(H1_N, dtype=DT)
    W[H1_Y, IN_Y] = 1.0  # y >= 0, so ReLU(y) = y
    W[H1_P, IN_P] = 1.0  # p >= 0
    for j, k in enumerate(SQ_KNOTS):
        # S_j = ReLU(s - k_j),  s = vy_n + 1 - x
        W[H1_S[j], IN_VYN] = 1.0
        W[H1_S[j], IN_X] = -1.0
        b[H1_S[j]] = 1.0 - k
        # Q_j = ReLU(q - k_j),  q = 1 - x - vy_n
        W[H1_Q[j], IN_VYN] = -1.0
        W[H1_Q[j], IN_X] = -1.0
        b[H1_Q[j]] = 1.0 - k
    for unit, v in zip(H1_V, VX_THRESH, strict=True):
        # V_i = ReLU(vx_n - v_i): 0 while the gate should be open (r >= t_i)
        W[unit, IN_VXN] = 1.0
        b[unit] = -v
    return W, b


def _product_readout() -> torch.Tensor:
    """Row vector over H1 with  a_hat = alpha . H1  ~=  vy_n * (1 - x)."""
    alpha = torch.zeros(H1_N, dtype=DT)
    # a = (f(s) - f(q)) / 4 ; the k_0^2 constants cancel.
    alpha[H1_S[0]] = SQ_W0 / 4
    alpha[H1_Q[0]] = -SQ_W0 / 4
    for j in range(1, SQ_J):
        alpha[H1_S[j]] = SQ_WJ / 4
        alpha[H1_Q[j]] = -SQ_WJ / 4
    return alpha


def _layer2() -> tuple[torch.Tensor, torch.Tensor]:
    W = torch.zeros(H2_N, H1_N, dtype=DT)
    b = torch.zeros(H2_N, dtype=DT)
    W[H2_Y, H1_Y] = 1.0
    W[H2_P, H1_P] = 1.0
    alpha = _product_readout()
    for i in range(R_CELLS):
        W[H2_GP[i]] = alpha  # +a
        W[H2_GN[i]] = -alpha  # -a
        if i >= 1:  # gate 0 is always open (r >= 1 always)
            W[H2_GP[i], H1_V[i - 1]] = -GATE_GAIN
            W[H2_GN[i], H1_V[i - 1]] = -GATE_GAIN
    return W, b


def _layer3() -> tuple[torch.Tensor, torch.Tensor]:
    W = torch.zeros(H3_N, H2_N, dtype=DT)
    b = torch.zeros(H3_N, dtype=DT)
    # z = Y2 + D_SCALE * sum_i c_i (G+_i - G-_i)  ~=  y + 1.5 * vy_n * m / vx_n
    zrow = torch.zeros(H2_N, dtype=DT)
    zrow[H2_Y] = 1.0
    for i, c in enumerate(GATE_COEF):
        zrow[H2_GP[i]] = D_SCALE * c
        zrow[H2_GN[i]] = -D_SCALE * c
    for unit, brk in zip(H3_F, FOLD_BREAKS, strict=True):
        W[unit] = zrow
        b[unit] = -float(brk)  # F_b = ReLU(z - b)
    W[H3_P, H2_P] = 1.0
    return W, b


def _output() -> tuple[torch.Tensor, torch.Tensor]:
    W = torch.zeros(3, H3_N, dtype=DT)
    b = torch.zeros(3, dtype=DT)
    for unit, c in zip(H3_F, FOLD_COEF, strict=True):
        W[OUT_DOWN, unit] = -c  # down = P - T
        W[OUT_UP, unit] = c  # up   = T - P
    W[OUT_DOWN, H3_P] = 1.0
    W[OUT_UP, H3_P] = -1.0
    b[OUT_STAY] = DEADBAND  # stay wins iff |T - P| < DEADBAND
    return W, b


def build() -> nn.Sequential:
    specs = [_layer1(), _layer2(), _layer3(), _output()]
    layers: list[nn.Module] = []
    for n, (W, b) in enumerate(specs):
        lin = nn.Linear(W.shape[1], W.shape[0], dtype=DT)
        lin.weight.requires_grad_(False)
        lin.bias.requires_grad_(False)
        lin.weight.copy_(W)
        lin.bias.copy_(b)
        layers.append(lin)
        if n < len(specs) - 1:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    model.unit_names = UNIT_NAMES  # provenance for every hidden/output unit
    return model
