"""Write a candidate policy = policy.py with some named constants overridden.

Usage: uv run python tools/make_candidate.py <out.py> NAME=value [NAME=value ...]
Only top-level `NAME = ...` constant lines are replaced, so each candidate
differs from policy.py by interpretable constants only.
"""

import re
import sys
from pathlib import Path

src = (Path(__file__).parent.parent / "policy.py").read_text()
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    pat = re.compile(rf"^{k} = [^#\n]*", re.MULTILINE)
    if not pat.search(src):
        raise SystemExit(f"no constant {k}")
    src = pat.sub(f"{k} = {v}  ", src, count=1)
src = src.replace(
    '"""Constructed', f'"""[candidate: {" ".join(sys.argv[2:])}]\n\nConstructed', 1
)
Path(sys.argv[1]).write_text(src)
