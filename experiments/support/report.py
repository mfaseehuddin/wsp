"""Score the support agents on the hidden test set and plot.

Admin script. Run from the repo root:
    WSP_ADMIN=1 uv run python experiments/support/report.py
Reads experiments/support/agent_usage.json ({agent: llm_tokens}) if present.
Writes experiments/support/results.json and experiments/support/support_agents.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wsp.support import harness

ROOT = Path("experiments/support")
AGENTS = ["s0", "s10", "s50"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"


def collect() -> dict:
    usage_f = ROOT / "agent_usage.json"
    usage = json.loads(usage_f.read_text()) if usage_f.exists() else {}
    out = {}
    for a in AGENTS:
        d = ROOT / "agents" / a
        led = harness.read_ledger(a)
        w = d / "weights.py"
        r = {"budget": led["budget"], "used": led["used"], "llm_tokens": usage.get(a)}
        if w.exists():
            try:
                r.update(harness.cmd_test(str(w)))
            except Exception as e:  # noqa: BLE001 - report broken submissions, do not crash
                r["error"] = f"{type(e).__name__}: {e}"
        out[a] = r
    return out


def _style(ax):
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
        }
    )
    ok = [a for a in AGENTS if "params" in res[a]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    for ax in axes:
        _style(ax)
    ax1, ax2, ax3 = axes
    for a, c in zip(AGENTS, COLORS, strict=True):
        if a not in ok:
            continue
        r = res[a]
        ax1.scatter(
            r["params"],
            r["test_exact"],
            s=90,
            color=c,
            edgecolors=SURFACE,
            linewidths=2,
            zorder=3,
            label=f"{a} (budget {r['budget']})",
        )
        ax1.annotate(
            f"{a}: {r['params']:,} params, {r['test_exact']:.1%}",
            (r["params"], r["test_exact"]),
            xytext=(0, -18 if a == "s10" else 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK2,
        )
    ax1.set_xscale("log")
    ax1.set_xlim(4e3, 3e4)
    ax1.set_xticks([5e3, 1e4, 2e4], ["5k", "10k", "20k"])
    ax1.xaxis.set_minor_locator(plt.NullLocator())
    ax1.set_ylim(-0.03, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_xlabel("parameters (log scale)")
    ax1.set_ylabel("hidden test: exact replies")
    ax1.set_title("Accuracy vs size", loc="left", fontsize=11)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")

    xs = range(len(AGENTS))
    labels = [f"{a}\nbudget {res[a]['budget']}" for a in AGENTS]
    for i, a in enumerate(AGENTS):
        r = res[a]
        if "dev_exact" not in r:
            ax2.annotate(
                "FAILED\nno model submitted\n(stopped after 18 min)",
                (i, 0.5),
                ha="center",
                va="center",
                fontsize=8,
                color=INK2,
            )
            ax1.annotate(
                f"{a}: failed, no model",
                (0.02, 0.04 + 0.06 * i),
                xycoords="axes fraction",
                fontsize=8,
                color=INK2,
            )
        for j, (k, lab) in enumerate([("dev_exact", "dev"), ("test_exact", "test")]):
            if k in r:
                ax2.bar(
                    i + (j - 0.5) * 0.36,
                    r[k],
                    width=0.34,
                    color=COLORS[i],
                    alpha=0.45 if k == "dev_exact" else 1.0,
                    edgecolor=SURFACE,
                    linewidth=2,
                )
                ax2.annotate(
                    f"{lab} {r[k]:.0%}",
                    (i + (j - 0.5) * 0.36, r[k]),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
    ax2.set_xticks(list(xs), labels)
    ax2.set_ylim(0, 1.1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_title("Exact replies: dev (light) vs hidden test", loc="left", fontsize=11)

    toks = [res[a]["llm_tokens"] or 0 for a in AGENTS]
    ax3.bar(xs, toks, color=COLORS, width=0.55)
    for i, t in enumerate(toks):
        ax3.annotate(
            f"{t:,}" if t else "not reported\n(stopped)",
            (i, t),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )
    ax3.set_xticks(list(xs), labels)
    ax3.set_ylabel("agent LLM tokens used")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax3.set_title("Agent token cost to generate the model", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    res = collect()
    (ROOT / "results.json").write_text(json.dumps(res, indent=2))
    plot(res, ROOT / "support_agents.png")
    for a in AGENTS:
        r = res[a]
        print(
            a,
            {
                k: r.get(k)
                for k in (
                    "params",
                    "nonzero",
                    "dev_exact",
                    "test_exact",
                    "used",
                    "llm_tokens",
                    "error",
                )
            },
        )
