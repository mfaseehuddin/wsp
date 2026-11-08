"""Admin audit for wsp-pure: list every shell command an agent ran that is not a wsp-pure call or a plain file view.

Usage: uv run python experiments/pure/audit.py pw100
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OK = re.compile(r"^\s*(cd [^;&]+(&&|;)\s*)?((uv run wsp-pure|cat|head|tail|grep|ls|wc|sed -n|echo)\b[^;&|]*(\s*(\||&&|;)\s*(cat|head|tail|grep|ls|wc|sed -n|uv run wsp-pure)\b[^;&|]*)*)$")
for a in sys.argv[1:]:
    tf = Path((ROOT / "agents" / a / ".transcript").read_text().strip())
    bad = []
    for line in tf.read_text().splitlines():
        d = json.loads(line)
        c = (d.get("message") or {}).get("content")
        if d.get("type") == "assistant" and isinstance(c, list):
            for x in c:
                if x.get("type") == "tool_use" and x.get("name") == "Bash":
                    cmd = x["input"].get("command", "")
                    for part in re.split(r"\n", cmd):
                        if part.strip() and not OK.match(part) and re.search(r"python|\.py\b|uv run (?!wsp-pure)|node|perl|ruby", part):
                            bad.append(part[:200])
    print(f"{a}: {len(bad)} flagged commands")
    for b in bad[:20]:
        print("  -", b)
