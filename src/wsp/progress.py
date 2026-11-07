"""Live progress logs for long-running harness commands.

Every command that runs a model writes timestamped lines to <experiment>/logs/<cmd>_<agent>_<time>.log (and to
stderr), so a human can `tail -f` it to see the phase, how far through it is, partial scores, and what it is waiting
on. `latest.log` in the same folder always points at the newest log.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_log: Path | None = None
_t0 = time.time()


def start(root: Path, cmd: str, agent: str | None) -> Path:
    global _log, _t0
    d = Path(root) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    _log = d / f"{cmd}_{agent or 'all'}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    _t0 = time.time()
    latest = d / "latest.log"
    try:
        latest.unlink(missing_ok=True)
        latest.symlink_to(_log.name)
    except OSError:
        pass
    say(f"start: {cmd} agent={agent} pid={os.getpid()} argv={' '.join(sys.argv[1:])}")
    return _log


def say(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} +{time.time() - _t0:7.1f}s  {msg}"
    print(line, file=sys.stderr, flush=True)
    if _log is not None:
        with open(_log, "a") as f:
            f.write(line + "\n")
