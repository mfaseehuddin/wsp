import json

import pytest
import torch
from torch import nn

from wsp.pong import baselines, game, harness


def test_observed_velocity_matches_step_by_step_reflection():
    r = game.sample_rallies(7, 200)
    y, vy = r.y0.clone(), r.vy.clone()
    idx = torch.arange(r.n)
    for k in range(60):
        obs = game.observe(r, k, torch.zeros(r.n, dtype=game.DTYPE), idx)
        assert torch.allclose(obs[:, 1], y, atol=1e-9)
        assert torch.allclose(obs[:, 3] * game.VY_MAX, vy, atol=1e-12)
        y = y + vy
        lo, hi = y < 0, y > 1
        y = torch.where(lo, -y, torch.where(hi, 2 - y, y))
        vy = torch.where(lo | hi, -vy, vy)


def test_forward_pass_count_known_in_advance():
    r = game.sample_rallies(3, 300)
    calls = []
    res = game.play(r, lambda o: (calls.append(o.shape[0]), baselines.stay(o))[1])
    assert res.forward_passes == sum(calls) == r.total_forward_passes()


def test_oracle_is_perfect_and_tracking_is_not():
    r = game.sample_rallies(11, 2000)
    assert game.play(r, baselines.oracle).hit_rate == 1.0
    assert game.play(r, baselines.track).hit_rate < 0.8


def _write_policy(tmp_path, body: str):
    p = tmp_path / "policy.py"
    p.write_text("import torch\nfrom torch import nn\n" + body)
    return str(p)


STAY_POLICY = """
def build():
    m = nn.Sequential(nn.Linear(5, 3))
    with torch.no_grad():
        m[0].weight.zero_(); m[0].bias.copy_(torch.tensor([0., 1., 0.]))
    return m
"""


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "ROOT", tmp_path / "exp")
    return tmp_path


def test_budget_refuses_without_charging(root):
    harness.init_ledger("a", 100)
    pol = _write_policy(root, STAY_POLICY)
    with pytest.raises(harness.BudgetExceeded):
        harness.evaluate("a", pol, rallies=10)
    assert harness.read_ledger("a")["used"] == 0
    out = harness.evaluate("a", pol, rallies=2)
    assert harness.read_ledger("a")["used"] == out["cost"] > 0


def test_probe_charges_one_pass_per_row(root):
    harness.init_ledger("a", 5)
    pol = _write_policy(root, STAY_POLICY)
    out = harness.probe("a", pol, [[0, 0.5, 1, 0, 0.5]] * 3)
    assert out["cost"] == 3 and out["actions"] == [1, 1, 1]
    log = [
        json.loads(line)
        for line in (harness.agent_dir("a") / "log.jsonl").read_text().splitlines()
    ]
    assert log[-1]["used"] == 3


def test_forward_pass_inside_build_is_blocked(root):
    pol = _write_policy(
        root,
        "def build():\n    m = nn.Sequential(nn.Linear(5, 3))\n    m(torch.zeros(1, 5))\n    return m\n",
    )
    with pytest.raises(harness.PolicyError):
        harness.load_policy(pol)


@pytest.mark.parametrize(
    "model",
    [
        nn.Sequential(nn.Linear(5, 3), nn.ReLU()),
        nn.Sequential(nn.Linear(5, 65), nn.ReLU(), nn.Linear(65, 3)),
        nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3)),
        nn.Sequential(nn.Linear(4, 3)),
    ],
)
def test_validate_rejects_bad_architectures(model):
    with pytest.raises(harness.PolicyError):
        harness.validate(model)
