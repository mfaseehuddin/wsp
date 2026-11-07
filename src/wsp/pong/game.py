"""Single-player Pong physics, vectorized over a batch of independent rallies.

Geometry (all in float64, unit square):
- The field is x in [0, 1], y in [0, 1].
- Each rally launches the ball from the left wall (x = 0) at a random height,
  moving right. It reflects off the top (y = 1) and bottom (y = 0) walls.
- The agent's paddle sits at x = 1 and moves vertically. Its centre starts at
  y = 0.5 and is clamped to [0, 1].
- The rally ends on the first step where the ball reaches x >= 1. At that moment
  the ball's crossing height is compared with the paddle centre.

The paddle is slower than the ball's fastest vertical speed, so following the
ball's current height is not enough. A good policy has to predict where the
ball will cross x = 1, bounces included.

The ball's path never depends on the paddle, so the number of steps in every
rally (and therefore the number of policy forward passes) is known before the
rally is played.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

DTYPE = torch.float64

# Physics constants
VX_MIN, VX_MAX = 0.02, 0.04  # horizontal speed per step (rally lasts 25..50 steps)
VY_MAX = 0.06  # |vertical speed| per step, uniform in [-VY_MAX, VY_MAX]
PADDLE_SPEED = 0.03  # paddle moves at most this far per step
PADDLE_HALF = 0.08  # paddle half-height; hit if |ball_y - paddle_y| <= PADDLE_HALF
PADDLE_START = 0.5

# Actions: index of the argmax of the policy's 3 logits.
# torch.argmax returns the LOWEST index on ties.
ACTION_DOWN, ACTION_STAY, ACTION_UP = 0, 1, 2
ACTION_DELTA = torch.tensor([-PADDLE_SPEED, 0.0, PADDLE_SPEED], dtype=DTYPE)

OBS_NAMES = ("ball_x", "ball_y", "ball_vx_n", "ball_vy_n", "paddle_y")
"""Observation layout given to the policy at every step.

ball_x, ball_y, paddle_y are raw positions in [0, 1].
ball_vx_n = vx / VX_MAX, in [0.5, 1].
ball_vy_n = vy / VY_MAX, in [-1, 1].
"""


def fold(y: torch.Tensor) -> torch.Tensor:
    """Map an unfolded height onto [0, 1] (reflection off both walls).

    This is a triangle wave with period 2.
    """
    m = torch.remainder(y, 2.0)
    return torch.where(m <= 1.0, m, 2.0 - m)


@dataclass(frozen=True)
class Rallies:
    """Initial conditions for a batch of rallies."""

    y0: torch.Tensor
    vx: torch.Tensor
    vy: torch.Tensor

    @property
    def n(self) -> int:
        return self.y0.shape[0]

    def steps(self) -> torch.Tensor:
        """Policy decisions (= forward passes) each rally takes.

        The ball is at x = k*vx before decision k. The rally ends after the
        first advance with x >= 1, so the number of decisions is ceil(1/vx).
        """
        return torch.ceil(1.0 / self.vx - 1e-12).to(torch.int64)

    def total_forward_passes(self) -> int:
        return int(self.steps().sum())

    def crossing_y(self) -> torch.Tensor:
        """Exact height at which the ball crosses x = 1."""
        return fold(self.y0 + self.vy / self.vx)


def sample_rallies(seed: int, n: int) -> Rallies:
    g = torch.Generator().manual_seed(seed)
    u = torch.rand((n, 3), generator=g, dtype=DTYPE)
    return Rallies(
        y0=u[:, 0],
        vx=VX_MIN + (VX_MAX - VX_MIN) * u[:, 1],
        vy=VY_MAX * (2.0 * u[:, 2] - 1.0),
    )


@dataclass
class RallyResult:
    loss: torch.Tensor  # per rally: max(0, |paddle - crossing| - PADDLE_HALF)
    hit: torch.Tensor  # per rally: bool
    forward_passes: int

    @property
    def mean_loss(self) -> float:
        return float(self.loss.mean())

    @property
    def hit_rate(self) -> float:
        return float(self.hit.to(DTYPE).mean())


def observe(
    r: Rallies, k: int, paddle: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor:
    x = k * r.vx[idx]
    u = r.y0[idx] + k * r.vy[idx]  # unfolded height
    y = fold(u)
    # Current vertical velocity: the sign flips on each wall bounce.
    vy = torch.where(torch.remainder(u, 2.0) <= 1.0, r.vy[idx], -r.vy[idx])
    return torch.stack([x, y, r.vx[idx] / VX_MAX, vy / VY_MAX, paddle[idx]], dim=1)


def play(r: Rallies, act) -> RallyResult:
    """Play every rally to completion.

    `act` maps an (m, 5) float64 observation batch to m action indices. It is
    only called for rallies still in play, so the total rows passed to it
    equal `r.total_forward_passes()`.
    """
    steps = r.steps()
    paddle = torch.full((r.n,), PADDLE_START, dtype=DTYPE)
    passes = 0
    for k in range(int(steps.max())):
        idx = torch.nonzero(steps > k).squeeze(1)
        if idx.numel() == 0:
            break
        obs = observe(r, k, paddle, idx)
        a = act(obs)
        passes += idx.numel()
        paddle[idx] = torch.clamp(paddle[idx] + ACTION_DELTA[a], 0.0, 1.0)
    err = (paddle - r.crossing_y()).abs()
    loss = torch.clamp(err - PADDLE_HALF, min=0.0)
    return RallyResult(loss=loss, hit=err <= PADDLE_HALF, forward_passes=passes)


def max_reachable_ok() -> bool:
    """Every crossing is reachable from PADDLE_START in the slowest-case time."""
    min_steps = math.ceil(1.0 / VX_MAX)
    return PADDLE_SPEED * min_steps + PADDLE_HALF >= 0.5
