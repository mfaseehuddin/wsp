"""Non-neural reference policies, used only to calibrate the task.

They are not networks and are never counted against an agent's budget.
"""

from __future__ import annotations

import torch

from .game import (
    ACTION_DOWN,
    ACTION_STAY,
    ACTION_UP,
    PADDLE_SPEED,
    VX_MAX,
    VY_MAX,
    fold,
)


def _toward(target: torch.Tensor, paddle: torch.Tensor) -> torch.Tensor:
    d = target - paddle
    a = torch.full(d.shape, ACTION_STAY, dtype=torch.int64)
    a[d > PADDLE_SPEED / 2] = ACTION_UP
    a[d < -PADDLE_SPEED / 2] = ACTION_DOWN
    return a


def stay(obs: torch.Tensor) -> torch.Tensor:
    return torch.full((obs.shape[0],), ACTION_STAY, dtype=torch.int64)


def track(obs: torch.Tensor) -> torch.Tensor:
    """Move toward the ball's current height."""
    return _toward(obs[:, 1], obs[:, 4])


def oracle(obs: torch.Tensor) -> torch.Tensor:
    """Move toward the exact crossing height (bounces included)."""
    x, y, vx, vy = obs[:, 0], obs[:, 1], obs[:, 2] * VX_MAX, obs[:, 3] * VY_MAX
    target = fold(y + vy * (1.0 - x) / vx)
    return _toward(target, obs[:, 4])


BASELINES = {"stay": stay, "track": track, "oracle": oracle}
