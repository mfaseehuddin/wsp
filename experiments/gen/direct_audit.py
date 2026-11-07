"""Admin: flag Bash commands in a direct-weights agent transcript that aren't allowed.

Allowed: `uv run wsp-gen …` and simple viewing (grep/head/sed -n/wc/cat/ls/tail) of the
agent's own files. Anything else (python, scripts, loops, awk, heredocs) is flagged,
as are reads of vocab.txt / src/ / other agents.
Usage: uv run python experiments/gen/direct_audit.py d100 d250
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OK = re.compile(r"^\s*(cd [^;&|]+&&\s*)?(uv run wsp-gen\b|grep\b|head\b|tail\b|wc\b|cat\b|ls\b|sed -n\b)")
BAD = re.compile(r"python|\bfor\b|\bwhile\b|awk|perl|<<|node\b|\bbc\b|vocab|src/|hidden|fresh|agents/(?!%s\b)")

for a in sys.argv[1:]:
    t = Path((ROOT / "agents" / a / ".transcript").read_text().strip())
    flagged = []
    for line in t.read_text().splitlines():
        d = json.loads(line)
        m = d.get("message")
        if d.get("type") != "assistant" or not isinstance(m, dict):
            continue
        for c in m.get("content") or []:
            if not isinstance(c, dict) or c.get("type") != "tool_use":
                continue
            inp = c.get("input", {})
            cmd = inp.get("command") or ""
            path = inp.get("file_path") or ""
            if c["name"] == "Bash":
                parts = [p.strip() for p in re.split(r"\|\||&&|;|\|", cmd) if p.strip()]
                parts = [p for p in parts if not p.startswith("cd ")]
                weights = re.search(r"\.wts|model\.", cmd)
                code = re.search(r"python|\bfor\b|\bwhile\b|awk|perl|<<|node\b", cmd)
                if re.search(r"vocab|src/|hidden|fresh|agents/(?!%s\b)" % a, cmd):
                    flagged.append(("FORBIDDEN-READ", cmd[:200]))
                elif code and weights:
                    flagged.append(("WEIGHT-CODE", cmd[:200]))
                elif code:
                    flagged.append(("code-on-data (tolerated)", cmd[:120]))
                elif not all(OK.match(p) or p.startswith(("head", "tail", "grep", "wc", "sed -n", "cat", "ls", "uv run wsp-gen")) for p in parts):
                    flagged.append(("other", cmd[:200]))
            elif path and f"agents/{a}/" not in path and not path.endswith(("DIRECT.md", "DIRECT_CHEATSHEET.md")):
                flagged.append((c["name"], path))
    print(a, "flags:", len(flagged))
    for f in flagged[:40]:
        print("  ", f)
