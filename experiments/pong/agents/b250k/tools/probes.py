"""Generate probe observations and analyse probe outputs (no forward passes here).

Usage:
  uv run python tools/probes.py gen <set> <out.json>      # writes obs list
  uv run python tools/probes.py analyse <obs.json> <harness_stdout.json>

The reference target is the exact physics, not a fitted model:
  target = fold(y + 1.5 * vyn * (1 - x) / vxn).
"""

import json
import sys
from pathlib import Path

import numpy as np


def fold(u):
    m = np.mod(u, 2.0)
    return np.where(m <= 1.0, m, 2.0 - m)


def target(o):
    o = np.asarray(o, dtype=float)
    x, y, vx, vy = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
    return fold(y + 1.5 * vy * (1 - x) / vx)


def gen(name):
    if name.startswith("rand"):  # rand<N>_<seed>: uniform over the state box
        n, seed = map(int, name[4:].split("_"))
        g = np.random.default_rng(seed)
        o = np.stack(
            [
                g.uniform(0, 1, n),
                g.uniform(0, 1, n),
                g.uniform(0.5, 1, n),
                g.uniform(-1, 1, n),
                g.uniform(0, 1, n),
            ],
            1,
        )
        return o.tolist()
    if name == "corners":  # extremes of every input
        rows = []
        for x in (0.0, 0.5, 0.999):
            for y in (0.0, 0.5, 1.0):
                for vx in (0.5, 0.75, 1.0):
                    for vy in (-1.0, 0.0, 1.0):
                        rows.append([x, y, vx, vy, 0.0])
        return rows
    if name.startswith(
        "slow"
    ):  # slow<N>_<seed>: vxn near 0.5, |vyn| near 1, x small (largest |u - y|)
        n, seed = map(int, name[4:].split("_"))
        g = np.random.default_rng(seed)
        o = np.stack(
            [
                g.uniform(0, 0.2, n),
                g.uniform(0, 1, n),
                g.uniform(0.5, 0.6, n),
                g.choice([-1, 1], n) * g.uniform(0.8, 1, n),
                np.zeros(n),
            ],
            1,
        )
        return o.tolist()
    raise SystemExit(f"unknown set {name}")


def analyse(obs_path, out_path):
    obs = np.asarray(json.loads(Path(obs_path).read_text()))
    out = json.loads(Path(out_path).read_text())
    lg = np.asarray(out["logits"])
    p = obs[:, 4]
    pred = lg[:, 2] + p  # up logit = target_hat - p
    t = target(obs)
    err = pred - t
    # consistency checks of the output layer
    sym = np.abs(lg[:, 0] + lg[:, 2]).max()
    stay = np.abs(lg[:, 1] - lg[0, 1]).max()
    a = np.abs(err)
    i = int(a.argmax())
    res = {
        "n": len(t),
        "mean_abs": float(a.mean()),
        "rms": float(np.sqrt((err**2).mean())),
        "p99_abs": float(np.quantile(a, 0.99)),
        "max_abs": float(a.max()),
        "mean_signed": float(err.mean()),
        "worst_obs": obs[i].tolist(),
        "worst_pred": float(pred[i]),
        "worst_true": float(t[i]),
        "down+up_max": float(sym),
        "stay_const_dev": float(stay),
        "stay": float(lg[0, 1]),
        "frac_err_gt_0.02": float((a > 0.02).mean()),
    }
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        Path(sys.argv[3]).write_text(json.dumps(gen(sys.argv[2])))
    else:
        analyse(sys.argv[2], sys.argv[3])
