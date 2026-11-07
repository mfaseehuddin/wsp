"""Admin monitor: print one line when an agent crosses its token limit or limit + grace.

Usage: uv run python experiments/bot/token_watch.py c1 c10 c50 c100 c1000
"""

import sys
import time

from wsp.bot import harness

GRACE = 100_000
fired: set[tuple[str, str, int]] = set()

while True:
    for a in sys.argv[1:]:
        try:
            led = harness.read_ledger(a)
            s = harness.token_status(a)
        except (FileNotFoundError, ValueError, OSError):
            continue
        used, lim, base = s["tokens_used_this_round"], s["token_budget"], led.get("token_base", 0)
        for name, threshold in (("LIMIT", lim), ("HARD-STOP", lim + GRACE)):
            key = (a, name, base)
            if used >= threshold and key not in fired:
                fired.add(key)
                print(f"{a} crossed {name}: {used:,} tokens used this round (limit {lim:,})", flush=True)
    time.sleep(20)
