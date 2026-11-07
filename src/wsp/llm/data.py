"""Task data for the bionic-circuit experiment: arithmetic problems, prompt formats, guard corpus."""

from __future__ import annotations

import random
import textwrap as _tw
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STDLIB = Path(_tw.__file__).parent

SHOTS = {
    "fewshot": "12+35=47\n481+207=688\n9036+1185=10221\n",
    "zeroshot": "",
    # hidden-only formats
    "fewshot_alt": "7+8=15\n60+44=104\n315+986=1301\n2048+512=2560\n77031+9406=86437\n",
    "longctx": "Here are some sums, each one computed exactly.\n33+19=52\n1500+275=1775\n",
}
DEV_FORMATS = ("fewshot", "zeroshot")
HIDDEN_FORMATS = ("fewshot", "zeroshot", "fewshot_alt", "longctx")
OPS = {"add": ("+", lambda a, b: a + b), "mul": ("*", lambda a, b: a * b)}
MUL_SHOTS = {"fewshot": "3*4=12\n12*11=132\n25*16=400\n"}


def problems(op: str, d1: int, d2: int, n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(f"{op}-{d1}-{d2}-{seed}")
    lo = lambda d: 0 if d == 1 else 10 ** (d - 1)  # noqa: E731
    return [(rng.randrange(lo(d1), 10**d1), rng.randrange(lo(d2), 10**d2)) for _ in range(n)]


def prompt(op: str, a: int, b: int, fmt: str) -> str:
    sym = OPS[op][0]
    shots = (MUL_SHOTS.get(fmt, "") if op == "mul" and fmt != "zeroshot" else SHOTS[fmt])
    return f"{shots}{a}{sym}{b}="


def answer(op: str, a: int, b: int) -> str:
    return str(OPS[op][1](a, b))


# ---- guard corpus: ordinary text the graft must leave (almost) untouched

_PROSE = [
    "The meeting moved from March 3 to March 10, and 42 people signed up; the room holds 60.",
    "Version 3.12 shipped in October 2023. It was about 5% faster on the benchmark suite than 3.11.",
    "She paid $14.99 for the book and $3.50 for shipping, so the receipt showed two lines.",
    "Call 555-0142 between 9 and 5. The office is at 221 Baker Street, floor 2.",
    "In 1969, Apollo 11 landed on the Moon; the crew of 3 returned on July 24.",
    "for i in range(10):\n    total = total + i * 2\nprint(total)\n",
    "x = a+b\ny = x-1\nif y == 0:\n    return a+b+1\n",
    "The recipe needs 2 cups of flour, 3 eggs and 250 ml of milk. Bake for 35 minutes at 180 C.",
    "Chapter 7 begins on page 112. The appendix lists 1,024 entries and 16 tables.",
    "C++ and C# are different languages; a+b=c is an equation, not an assignment.",
    "The train left at 08:15 and arrived at 11:40, a trip of 3 hours and 25 minutes.",
    "Population grew from 8,400 in 1990 to 12,750 in 2020, an increase of about 52 percent.",
]


def guard_texts(split: str) -> list[str]:
    if split == "dev":
        files, prose = ["textwrap.py", "fractions.py"], _PROSE[::2]
        doc = (REPO / "README.md").read_text()
    else:
        files, prose = ["heapq.py", "statistics.py", "calendar.py"], _PROSE[1::2]
        doc = (REPO / "docs/research/weight-space-programming-survey.md").read_text()
    out = []
    for f in files:
        src = (STDLIB / f).read_text()
        out += [src[i : i + 1500] for i in range(0, min(len(src), 9000), 1500)]
    out += [doc[i : i + 1500] for i in range(0, min(len(doc), 9000), 1500)]
    out += prose
    return out
