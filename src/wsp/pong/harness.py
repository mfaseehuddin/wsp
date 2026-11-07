"""Budgeted evaluation harness for generated Pong policies.

Every forward pass of an agent's network (one observation row = one pass) must
go through this harness, which charges it to that agent's ledger and refuses
any request that would go over budget. The cost of an evaluation is known
before it runs (see `Rallies.total_forward_passes`), so a refused request costs
nothing.

Policy contract: a Python file defining `build() -> torch.nn.Sequential` made
only of `nn.Linear` and `nn.ReLU` layers, alternating, with input width 5
(see `game.OBS_NAMES`), output width 3 (logits for down/stay/up), at most 3
hidden layers of width <= 64. The action is `argmax(logits)` (ties go to the
lowest index). Weights are cast to float64 and run on CPU so the results are exact.

CLI (run with `uv run wsp-pong ...`):
  init   --agent A --budget N               (admin) create a ledger
  status --agent A                          remaining budget
  eval   --agent A --policy P --rallies N [--offset K] [--note TEXT]
  probe  --agent A --policy P --obs JSON [--note TEXT]
  test   --agent A --policy P               (admin, WSP_ADMIN=1) held-out score, uncharged
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

from . import game

ROOT = Path(os.environ.get("WSP_PONG_ROOT", "experiments/pong"))
DEV_SEED, DEV_N = 1000, 20000
TEST_SEED, TEST_N = 2000, 5000
MAX_HIDDEN, MAX_WIDTH = 3, 64


class BudgetExceeded(RuntimeError):
    pass


class PolicyError(ValueError):
    pass


# ---------- ledger ----------


def agent_dir(agent: str) -> Path:
    return ROOT / "agents" / agent


def _ledger_path(agent: str) -> Path:
    return agent_dir(agent) / ".ledger.json"


def init_ledger(agent: str, budget: int) -> None:
    p = _ledger_path(agent)
    if p.exists():
        raise FileExistsError(f"ledger already exists: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"budget": budget, "used": 0}))


def read_ledger(agent: str) -> dict:
    p = _ledger_path(agent)
    if not p.exists():
        raise FileNotFoundError(
            f"no ledger for agent {agent!r}; ask the admin to init it"
        )
    return json.loads(p.read_text())


def _charge(agent: str, cost: int) -> dict:
    led = read_ledger(agent)
    remaining = led["budget"] - led["used"]
    if cost > remaining:
        raise BudgetExceeded(
            f"request costs {cost} forward passes but only {remaining} remain "
            f"(budget {led['budget']}). Nothing was charged."
        )
    led["used"] += cost
    _ledger_path(agent).write_text(json.dumps(led))
    return led


def _log(agent: str, record: dict) -> None:
    with open(agent_dir(agent) / "log.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------- policy loading ----------


def load_policy(path: str | Path) -> tuple[nn.Sequential, str]:
    path = Path(path)
    src = path.read_bytes()
    spec = importlib.util.spec_from_file_location(f"policy_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)

    # Forbid forward passes while building: they would not be counted.
    real_linear = torch.nn.functional.linear

    def _blocked(*a, **k):
        raise PolicyError(
            "forward passes are not allowed inside build(); use the harness"
        )

    torch.nn.functional.linear = _blocked
    try:
        spec.loader.exec_module(mod)
        if not hasattr(mod, "build"):
            raise PolicyError(f"{path} defines no build()")
        model = mod.build()
    finally:
        torch.nn.functional.linear = real_linear

    validate(model)
    return model.to(dtype=game.DTYPE, device="cpu").eval(), hashlib.sha256(
        src
    ).hexdigest()[:12]


def validate(model) -> None:
    if not isinstance(model, nn.Sequential):
        raise PolicyError("build() must return torch.nn.Sequential")
    layers = list(model)
    for i, layer in enumerate(layers):
        want = nn.Linear if i % 2 == 0 else nn.ReLU
        if type(layer) is not want:
            raise PolicyError(
                f"layer {i} must be {want.__name__}, got {type(layer).__name__}"
            )
    if not layers or not isinstance(layers[-1], nn.Linear):
        raise PolicyError("last layer must be nn.Linear")
    linears = layers[0::2]
    if linears[0].in_features != 5 or linears[-1].out_features != 3:
        raise PolicyError("input width must be 5 and output width 3")
    if len(linears) - 1 > MAX_HIDDEN:
        raise PolicyError(f"at most {MAX_HIDDEN} hidden layers")
    for a, b in itertools.pairwise(linears):
        if a.out_features != b.in_features:
            raise PolicyError("layer widths do not chain")
        if a.out_features > MAX_WIDTH:
            raise PolicyError(f"hidden width must be <= {MAX_WIDTH}")
    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise PolicyError("non-finite weights")


def _actor(model: nn.Sequential):
    @torch.no_grad()
    def act(obs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(model(obs), dim=1)

    return act


# ---------- operations ----------


def evaluate(
    agent: str, policy: str, rallies: int, offset: int = 0, note: str = ""
) -> dict:
    if rallies < 1 or offset < 0 or offset + rallies > DEV_N:
        raise ValueError(f"need 1 <= rallies and offset + rallies <= {DEV_N}")
    model, sha = load_policy(policy)
    r_all = game.sample_rallies(DEV_SEED, DEV_N)
    sl = slice(offset, offset + rallies)
    r = game.Rallies(r_all.y0[sl], r_all.vx[sl], r_all.vy[sl])
    cost = r.total_forward_passes()
    led = _charge(agent, cost)
    res = game.play(r, _actor(model))
    assert res.forward_passes == cost
    out = {
        "ts": time.time(),
        "kind": "eval",
        "policy": str(policy),
        "sha": sha,
        "rallies": rallies,
        "offset": offset,
        "cost": cost,
        "used": led["used"],
        "budget": led["budget"],
        "mean_loss": res.mean_loss,
        "hit_rate": res.hit_rate,
        "note": note,
    }
    _log(agent, out)
    return out


def probe(agent: str, policy: str, obs: list[list[float]], note: str = "") -> dict:
    x = torch.tensor(obs, dtype=game.DTYPE)
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("obs must be a list of 5-element rows")
    model, sha = load_policy(policy)
    led = _charge(agent, x.shape[0])
    with torch.no_grad():
        logits = model(x)
    out = {
        "ts": time.time(),
        "kind": "probe",
        "policy": str(policy),
        "sha": sha,
        "cost": x.shape[0],
        "used": led["used"],
        "budget": led["budget"],
        "note": note,
        "logits": logits.tolist(),
        "actions": torch.argmax(logits, dim=1).tolist(),
    }
    _log(agent, {k: v for k, v in out.items() if k != "logits"})
    return out


def test_score(policy: str, act=None) -> dict:
    """Held-out score. Admin only; never charged, never shown to agents."""
    r = game.sample_rallies(TEST_SEED, TEST_N)
    if act is None:
        model, _ = load_policy(policy)
        act = _actor(model)
    res = game.play(r, act)
    return {"mean_loss": res.mean_loss, "hit_rate": res.hit_rate, "rallies": TEST_N}


# ---------- CLI ----------


def _require_admin() -> None:
    if os.environ.get("WSP_ADMIN") != "1":
        sys.exit("admin command: set WSP_ADMIN=1 (agents must not use this)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="wsp-pong")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init")
    s.add_argument("--agent", required=True)
    s.add_argument("--budget", type=int, required=True)
    s = sub.add_parser("status")
    s.add_argument("--agent", required=True)
    s = sub.add_parser("eval")
    s.add_argument("--agent", required=True)
    s.add_argument("--policy", required=True)
    s.add_argument("--rallies", type=int, required=True)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--note", default="")
    s = sub.add_parser("probe")
    s.add_argument("--agent", required=True)
    s.add_argument("--policy", required=True)
    s.add_argument(
        "--obs",
        required=True,
        help="JSON list of [ball_x, ball_y, vx_n, vy_n, paddle_y]",
    )
    s.add_argument("--note", default="")
    s = sub.add_parser("test")
    s.add_argument("--policy", required=True)
    a = ap.parse_args(argv)

    try:
        if a.cmd == "init":
            _require_admin()
            init_ledger(a.agent, a.budget)
            out = read_ledger(a.agent)
        elif a.cmd == "status":
            led = read_ledger(a.agent)
            out = {**led, "remaining": led["budget"] - led["used"]}
        elif a.cmd == "eval":
            out = evaluate(a.agent, a.policy, a.rallies, a.offset, a.note)
        elif a.cmd == "probe":
            out = probe(a.agent, a.policy, json.loads(a.obs), a.note)
        else:
            _require_admin()
            out = test_score(a.policy)
    except (
        BudgetExceeded,
        PolicyError,
        FileNotFoundError,
        FileExistsError,
        ValueError,
    ) as e:
        sys.exit(f"error: {e}")
    print(json.dumps(out, indent=2))
