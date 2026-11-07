"""Admin monitor: print one line when an agent crosses its token limit or limit + grace.

Usage: uv run python experiments/ncmdeep/token_watch.py ncd60
"""

import sys
import time

from wsp.ncmchat import deep as _d, harness
harness.ROOT = harness.REPO / "experiments/ncmdeep"

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
