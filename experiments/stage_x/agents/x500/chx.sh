#!/bin/bash
cd /Users/mfaseehuddin/mfaseehuddin/development/wsp
uv run wsp-x chat --agent x500 --turns "$1" 2>&1 | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'turns' not in d: print(str(d)[:600])
else:
  for t in d['turns']: print(repr(t['user'])[:50],'->',repr(t['reply'])[:250],[ (x['req'],x['result']) for x in t['tools']][:2],t['error'])"
