"""Collect agent logs, score final policies on the held-out test set, and plot.

Admin script (uncharged test scoring). Run from the repo root:
    WSP_ADMIN=1 uv run python experiments/pong/report.py
Writes experiments/pong/results.json and experiments/pong/forward_passes_vs_loss.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wsp.pong import baselines, harness

ROOT = Path("experiments/pong")
AGENTS = ["b0", "b2k", "b10k", "b50k", "b250k"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # dataviz slots 1-5
SPEND = {  # what a forward pass was spent on
    "confirm": ("eval of the final design", "#6250d6"),
    "explore": ("eval of a deliberately changed design", "#008300"),
    "probe": ("probe of individual units", "#a3a29c"),
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def collect() -> dict:
    out = {"agents": {}, "baselines": {}}
    for name, fn in baselines.BASELINES.items():
        out["baselines"][name] = harness.test_score("", act=fn)
    for a in AGENTS:
        d = ROOT / "agents" / a
        led = harness.read_ledger(a)
        log = d / "log.jsonl"
        calls = (
            [json.loads(x) for x in log.read_text().splitlines()]
            if log.exists()
            else []
        )
        spend = dict.fromkeys(SPEND, 0)
        curve = []
        for c in calls:
            final = (
                Path(c["policy"]).name == "policy.py" and Path(c["policy"]).parent == d
            )
            kind = (
                "probe" if c["kind"] == "probe" else ("confirm" if final else "explore")
            )
            spend[kind] += c["cost"]
            if c["kind"] == "eval":
                curve.append(
                    {
                        "used": c["used"],
                        "loss": c["mean_loss"],
                        "final": final,
                        "policy": Path(c["policy"]).name,
                        "rallies": c["rallies"],
                        "offset": c["offset"],
                        "note": c["note"],
                    }
                )
        policy = d / "policy.py"
        out["agents"][a] = {
            "budget": led["budget"],
            "used": led["used"],
            "spend": spend,
            "curve": curve,
            "test": harness.test_score(str(policy)) if policy.exists() else None,
        }
    return out


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot(res: dict, path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
        }
    )
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 5.6),
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    _style(ax1)
    _style(ax2)
    bl = res["baselines"]

    # Panel 1: dev loss of every evaluated network vs cumulative passes spent
    for label, y in [
        ("paddle stays still", bl["stay"]["mean_loss"]),
        ("follow the ball (no prediction)", bl["track"]["mean_loss"]),
    ]:
        ax1.axhline(y, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax1.annotate(
            label,
            (0.01, y),
            xycoords=("axes fraction", "data"),
            ha="left",
            va="bottom",
            fontsize=8,
            color=INK2,
        )
    for a, col in zip(AGENTS, COLORS, strict=True):
        r = res["agents"][a]
        lab = f"{a}  (budget {r['budget']:,})"
        if not r["curve"]:  # b0: never ran; plot its held-out score at 0 passes
            ax1.scatter(
                [0],
                [r["test"]["mean_loss"]],
                marker="D",
                s=70,
                color=col,
                edgecolors=SURFACE,
                linewidths=2,
                zorder=5,
                label=lab + ", never ran",
            )
            continue
        fin = [p for p in r["curve"] if p["final"]]
        alt = [p for p in r["curve"] if not p["final"]]
        ax1.plot(
            [p["used"] for p in fin],
            [p["loss"] for p in fin],
            color=col,
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=4,
            label=lab,
        )
        ax1.scatter(
            [p["used"] for p in alt],
            [p["loss"] for p in alt],
            s=64,
            color=col,
            marker="X",
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=3,
        )
    ax1.scatter(
        [],
        [],
        marker="X",
        s=64,
        color=INK2,
        label="deliberately changed design (ablation / stress test)",
    )
    ax1.set_xscale("symlog", linthresh=1000, linscale=0.6)
    ax1.set_xlim(-200, 3.2e5)
    ax1.set_xticks([0, 1e3, 1e4, 1e5], ["0", "1k", "10k", "100k"])
    ax1.xaxis.set_minor_locator(plt.NullLocator())
    for p in res["agents"]["b250k"]["curve"]:  # name the two worst ablations
        txt = {
            "abl_follow.py": "b250k: no anticipation",
            "abl_clamp.py": "b250k: no wall bounces",
        }
        if p["policy"] in txt:
            ax1.annotate(
                txt[p["policy"]],
                (p["used"], p["loss"]),
                xytext=(-8, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8,
                color=INK2,
            )
    ax1.set_yscale("symlog", linthresh=1e-3, linscale=0.8)
    ax1.set_ylim(-1e-4, 0.3)
    ax1.set_xlabel("cumulative forward passes spent by the agent")
    ax1.set_ylabel("mean loss on the dev rallies evaluated")
    ax1.set_title(
        "Every final network scored 0 the first time it was run",
        loc="left",
        fontsize=11,
    )
    ax1.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0, 0.42))

    # Panel 2: what the budget was spent on (share of passes used)
    ys = list(range(len(AGENTS)))[::-1]
    for y, a in zip(ys, AGENTS, strict=True):
        r = res["agents"][a]
        left = 0.0
        for k, (lab, col) in SPEND.items():
            w = r["spend"][k] / r["used"] if r["used"] else 0.0
            if w:
                ax2.barh(
                    y,
                    w,
                    left=left,
                    color=col,
                    height=0.6,
                    edgecolor=SURFACE,
                    linewidth=2,
                    label=lab if a == "b250k" else None,
                )
            left += w
        t = r["test"]
        ax2.annotate(
            f"{r['used']:,} of {r['budget']:,} passes   test: {t['hit_rate']:.0%} hit, "
            f"loss {t['mean_loss']:.3g}",
            (0, y + 0.34),
            fontsize=8,
            color=INK2,
        )
    ax2.set_yticks(ys, AGENTS)
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_xlabel("share of the passes the agent used")
    ax2.set_title("What the forward passes were spent on", loc="left", fontsize=11)
    ax2.legend(
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
    )
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    res = collect()
    (ROOT / "results.json").write_text(json.dumps(res, indent=2))
    plot(res, ROOT / "forward_passes_vs_loss.png")
    for a in AGENTS:
        r = res["agents"][a]
        print(a, r["used"], "/", r["budget"], r["spend"], "test:", r["test"])
