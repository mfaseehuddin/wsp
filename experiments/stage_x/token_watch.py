"""Admin monitor: print one line when an agent crosses its token limit or limit + grace.

Usage: uv run python experiments/stage_x/token_watch.py x500 x2000
"""

import sys
import time

from wsp.stage_x import lab

GRACE = 100_000
fired: set[tuple[str, str]] = set()

while True:
    for a in sys.argv[1:]:
        try:
            s = lab.token_status(a)
        except (FileNotFoundError, ValueError, OSError):
            continue
        used, lim = s["tokens_used"], s["token_budget"]
        for name, threshold in (("LIMIT", lim), ("HARD-STOP", lim + GRACE)):
            if used >= threshold and (a, name) not in fired:
                fired.add((a, name))
                print(f"{a} crossed {name}: {used:,} tokens (limit {lim:,})", flush=True)
    time.sleep(20)
