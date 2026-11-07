"""Score the bot agents (hidden test, float64 and float32) and plot.

Admin script: WSP_ADMIN=1 uv run python experiments/bot/report.py
Writes experiments/bot/results_stage<N>.json and experiments/bot/bot_agents_stage<N>.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from wsp.bot import harness as H
from wsp.bot import kb
from wsp.bot import model as M

ROOT = Path("experiments/bot")
AGENTS = ["c1", "c10", "c50", "c100", "c1000", "x500"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"


def score(agent: str, st: int) -> dict:
    f = ROOT / "agents" / agent / "weights.py"
    cfg, p, _ = H.load(f)
    r = H.cmd_test(str(f), "test", st)
    convs = kb.conversations("test", st)
    M.DTYPE = torch.float32
    try:
        ok32 = sum(
            H.run_conversation(cfg, {k: v.float() for k, v in p.items()}, c)["ok"]
            for c in convs
        )
    finally:
        M.DTYPE = torch.float64
    led = H.read_ledger(agent)
    return {
        "params": r["params"],
        "nonzero": r["nonzero"],
        "conv_exact": r["conv_exact"],
        "turn_exact": r["turn_exact"],
        "conv_exact_f32": ok32 / len(convs),
        "max_abs_weight": max(float(t.abs().max()) for t in p.values()),
        "runs_used": led["used"],
        "runs_budget": led["budget"],
        "custom": not isinstance(cfg, M.Config),
        "per_kind": r["per_kind"],
    }


def plot(res: dict, path: Path, st: int) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
        }
    )
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE)
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for a, c in zip(AGENTS, COLORS, strict=True):
        r = res[a]
        lab = f"{a}{' (Sonnet, custom arch)' if r['custom'] else ''}"
        for ax, x in ((a1, r["runs_used"]), (a2, r["params"])):
            ax.plot(
                [x, x],
                [r["conv_exact_f32"], r["conv_exact"]],
                color=c,
                linewidth=2,
                zorder=2,
            )
            ax.scatter(
                [x],
                [r["conv_exact"]],
                s=90,
                color=c,
                edgecolors=SURFACE,
                linewidths=2,
                zorder=3,
                label=lab if ax is a1 else None,
            )
            ax.scatter([x], [r["conv_exact_f32"]], s=50, marker="v", color=c, zorder=3)
            ax.annotate(
                f"{a} {r['conv_exact']:.0%}",
                (x, r["conv_exact"]),
                xytext=(7, 4),
                textcoords="offset points",
                fontsize=8,
                color=INK2,
            )
    a1.scatter([], [], marker="v", color=INK2, label="same model run in float32")
    a1.set_xscale("log")
    a1.set_xlabel("runs used (log)")
    a1.set_ylabel("hidden test: whole conversations exactly right")
    a1.set_title(f"Accuracy vs runs spent (stage {st})", loc="left", fontsize=11)
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    a2.set_xscale("log")
    a2.set_xlabel("parameters (log)")
    a2.set_title("Accuracy vs model size", loc="left", fontsize=11)
    for ax in (a1, a2):
        ax.set_ylim(0.55, 0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    st = H.stage()
    usage = json.loads((ROOT / "agent_usage.json").read_text())
    res = {a: {**score(a, st), "llm_tokens": usage.get(a)} for a in AGENTS}
    (ROOT / f"results_stage{st}.json").write_text(json.dumps(res, indent=2))
    plot(res, ROOT / f"bot_agents_stage{st}.png", st)
    for a in AGENTS:
        r = res[a]
        print(
            f"{a:6} runs {r['runs_used']:4} params {r['params']:>9,} test {r['conv_exact']:.3f} "
            f"f32 {r['conv_exact_f32']:.3f} max|w| {r['max_abs_weight']:.2g} tokens {r['llm_tokens']}"
        )
