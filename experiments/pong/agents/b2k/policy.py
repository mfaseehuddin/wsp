"""Constructed (untrained) Pong policy for agent b2k.

The network computes, in closed form, where the ball will cross x = 1 and
moves the paddle toward that point.

    a      = 1 - ball_x                       remaining horizontal distance
    c      = ball_vy_n  = vy / VY_MAX         in [-1, 1]
    v      = ball_vx_n  = vx / VX_MAX         in [0.5, 1]
    d      = vy * a / vx = K * c * a / v      vertical displacement until impact,
                                              K = VY_MAX / VX_MAX = 1.5
    u      = ball_y + d                       unfolded crossing height, in [-3, 4]
    t      = fold(u)                          predicted crossing height (triangle wave)
    logits = [p - t - THETA, 0, t - p - THETA]

Hidden layer 1 (62 units): ball_y and paddle_y pass-throughs, a piecewise-linear
    (PWL) 1/v, and PWL squares of r1 = a + c and r2 = a - c, so that the
    product P1 = c*a = (r1^2 - r2^2)/4 can be read out linearly.
Hidden layer 2 (62 units): pass-throughs, PWL squares of z3 = b + P1 and
    z4 = b - P1, so that P1*b = c*a/v = (z3^2 - z4^2)/4 can be read out linearly.
Hidden layer 3 (10 units): paddle pass-through and hinges ReLU(u - k) at the
    integers k = -4..4, from which the triangle wave fold(u) is exact.
Output: argmax of [down, stay, up] with a dead band of half a paddle step.

Every weight below is a formula of the named constants. See EXPLANATION.md.
"""

from __future__ import annotations

import torch
from torch import nn

DT = torch.float64

# ---- physics constants (copied from src/wsp/pong/game.py) ----
VX_MAX = 0.04
VY_MAX = 0.06
PADDLE_SPEED = 0.03
V_MIN, V_MAX = 0.5, 1.0  # range of ball_vx_n = vx / VX_MAX
K_SLOPE = VY_MAX / VX_MAX  # d = K_SLOPE * c * a / v

# ---- observation indices ----
IX, IY, IV, IC, IP = 0, 1, 2, 3, 4  # ball_x, ball_y, vx_n, vy_n, paddle_y

# ---- design constants ----
NB = 12  # segments of the PWL 1/v on [0.5, 1] (geometric knots)
SQ1_LO, SQ1_HI, SQ1_H = -1.0, 2.0, 1.0 / 8.0  # grid for r1, r2 in [-1, 2]
SQ2_LO, SQ2_HI, SQ2_H = 0.0, 3.0, 0.1  # grid for z3, z4 in [0, 3]
FOLD_K = list(range(-4, 5))  # integer knots of the triangle wave on u >= -4
THETA = PADDLE_SPEED / 2.0  # dead band: move only if that reduces |t - p|


def recip_knots() -> list[float]:
    """Geometric knots v_j = 0.5 * 2^(j/NB), j = 0..NB (equal relative error)."""
    return [V_MIN * (V_MAX / V_MIN) ** (j / NB) for j in range(NB + 1)]


def grid(lo: float, hi: float, h: float) -> list[float]:
    n = round((hi - lo) / h)
    return [lo + j * h for j in range(n + 1)]


def pwl_hinge_coeffs(knots: list[float], f) -> tuple[float, list[float]]:
    """Interpolate f at the knots with hinges ReLU(z - knot_j), j = 0..n-1.

    Returns (f(knot_0), [coef_j]) so that on [knot_0, knot_n]
        f(z) ~= f(knot_0) + sum_j coef_j * ReLU(z - knot_j),
    with coef_0 = slope of the first segment and coef_j = change of slope at
    knot_j. This is exact at every knot and linear in between.
    """
    slopes = [
        (f(knots[j + 1]) - f(knots[j])) / (knots[j + 1] - knots[j])
        for j in range(len(knots) - 1)
    ]
    coefs = [slopes[0]] + [slopes[j] - slopes[j - 1] for j in range(1, len(slopes))]
    return f(knots[0]), coefs


def build() -> nn.Sequential:
    vk = recip_knots()  # NB + 1 knots
    b0, b_coef = pwl_hinge_coeffs(vk, lambda v: 1.0 / v)  # b(v) ~ 1/v
    g1 = grid(SQ1_LO, SQ1_HI, SQ1_H)
    s1_0, s1_coef = pwl_hinge_coeffs(g1, lambda z: z * z)
    g2 = grid(SQ2_LO, SQ2_HI, SQ2_H)
    s2_0, s2_coef = pwl_hinge_coeffs(g2, lambda z: z * z)
    n1, n2 = len(s1_coef), len(s2_coef)

    # ------------------------------------------------------------ layer 1
    # unit layout
    L1_Y, L1_P = 0, 1
    L1_B = list(range(2, 2 + NB))  # ReLU(v - v_j)
    L1_R1 = list(range(L1_B[-1] + 1, L1_B[-1] + 1 + n1))  # ReLU(a + c - z_j)
    L1_R2 = list(range(L1_R1[-1] + 1, L1_R1[-1] + 1 + n1))  # ReLU(a - c - z_j)
    H1 = L1_R2[-1] + 1
    W1 = torch.zeros(H1, 5, dtype=DT)
    c1 = torch.zeros(H1, dtype=DT)
    W1[L1_Y, IY] = 1.0
    W1[L1_P, IP] = 1.0
    for j, u in enumerate(L1_B):
        W1[u, IV] = 1.0
        c1[u] = -vk[j]
    for j, (u1, u2) in enumerate(zip(L1_R1, L1_R2)):
        # r1 = a + c = 1 - x + c ; r2 = a - c = 1 - x - c
        W1[u1, IX], W1[u1, IC], c1[u1] = -1.0, 1.0, 1.0 - g1[j]
        W1[u2, IX], W1[u2, IC], c1[u2] = -1.0, -1.0, 1.0 - g1[j]

    # Linear read-outs of layer 1 (rows over the H1 units, plus constants):
    #   b  = b0 + sum_j b_coef_j * B_j                         ~ 1/v
    #   P1 = (sq(r1) - sq(r2)) / 4 = sum_j s1_coef_j (R1_j - R2_j) / 4   ~ c*a
    read_b = torch.zeros(H1, dtype=DT)
    for j, u in enumerate(L1_B):
        read_b[u] = b_coef[j]
    read_P1 = torch.zeros(H1, dtype=DT)
    for j, (u1, u2) in enumerate(zip(L1_R1, L1_R2)):
        read_P1[u1] = s1_coef[j] / 4.0
        read_P1[u2] = -s1_coef[j] / 4.0
    # (the constant s1_0 cancels in the difference of the two squares)

    # ------------------------------------------------------------ layer 2
    L2_Y, L2_P = 0, 1
    L2_Z3 = list(range(2, 2 + n2))  # ReLU(b + P1 - w_k)
    L2_Z4 = list(range(L2_Z3[-1] + 1, L2_Z3[-1] + 1 + n2))  # ReLU(b - P1 - w_k)
    H2 = L2_Z4[-1] + 1
    W2 = torch.zeros(H2, H1, dtype=DT)
    c2 = torch.zeros(H2, dtype=DT)
    W2[L2_Y, L1_Y] = 1.0
    W2[L2_P, L1_P] = 1.0
    for k, (u3, u4) in enumerate(zip(L2_Z3, L2_Z4)):
        W2[u3] = read_b + read_P1
        c2[u3] = b0 - g2[k]
        W2[u4] = read_b - read_P1
        c2[u4] = b0 - g2[k]

    # Linear read-out of layer 2:
    #   u = y + K * (sq2(z3) - sq2(z4)) / 4                    ~ y + K c a / v
    read_u = torch.zeros(H2, dtype=DT)
    read_u[L2_Y] = 1.0
    for k, (u3, u4) in enumerate(zip(L2_Z3, L2_Z4)):
        read_u[u3] = K_SLOPE * s2_coef[k] / 4.0
        read_u[u4] = -K_SLOPE * s2_coef[k] / 4.0
    # (the constant s2_0 cancels)

    # ------------------------------------------------------------ layer 3
    L3_P = 0
    L3_F = list(range(1, 1 + len(FOLD_K)))  # ReLU(u - k)
    H3 = L3_F[-1] + 1
    W3 = torch.zeros(H3, H2, dtype=DT)
    c3 = torch.zeros(H3, dtype=DT)
    W3[L3_P, L2_P] = 1.0
    for i, k in enumerate(FOLD_K):
        W3[L3_F[i]] = read_u
        c3[L3_F[i]] = -float(k)

    # Triangle wave on u >= -4: fold(-4) = 0, slope +1 on [-4, -3], and the
    # slope flips at every integer: change -2 at odd k, +2 at even k.
    read_t = torch.zeros(H3, dtype=DT)
    for i, k in enumerate(FOLD_K):
        read_t[L3_F[i]] = 1.0 if i == 0 else (2.0 if k % 2 == 0 else -2.0)

    # ------------------------------------------------------------ output
    W4 = torch.zeros(3, H3, dtype=DT)
    c4 = torch.zeros(3, dtype=DT)
    p_row = torch.zeros(H3, dtype=DT)
    p_row[L3_P] = 1.0
    W4[0] = p_row - read_t  # down:  p - t - THETA
    c4[0] = -THETA
    # stay: 0
    W4[2] = read_t - p_row  # up:    t - p - THETA
    c4[2] = -THETA

    layers = []
    for W, c in ((W1, c1), (W2, c2), (W3, c3), (W4, c4)):
        lin = nn.Linear(W.shape[1], W.shape[0], dtype=DT)
        with torch.no_grad():
            lin.weight.copy_(W)
            lin.bias.copy_(c)
        layers += [lin, nn.ReLU()]
    return nn.Sequential(*layers[:-1])
