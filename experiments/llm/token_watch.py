"""Admin monitor: print one line when an agent crosses its token limit or limit + grace.

Usage: uv run python experiments/llm/token_watch.py o60 s60
"""

import sys
import time

from wsp.llm import harness

GRACE = 100_000
fired: set[tuple[str, str, int]] = set()

while True:
    for a in sys.argv[1:]:
        try:
            led = harness.read_ledger(a)
            s = harness.token_status(a)
        except (FileNotFoundError, ValueError, OSError):
            continue
        used, lim, base = s["tokens_used"], s["token_budget"], led.get("token_base", 0)
        for name, threshold in (("LIMIT", lim), ("HARD-STOP", lim + GRACE)):
            key = (a, name, base)
            if used >= threshold and key not in fired:
                fired.add(key)
                print(f"{a} crossed {name}: {used:,} tokens used this round (limit {lim:,})", flush=True)
    time.sleep(20)
