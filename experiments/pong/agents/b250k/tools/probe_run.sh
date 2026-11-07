#!/bin/zsh
# probe_run.sh <policy.py> <probe-set-name> <tag> <note>  -> runs/out_<tag>_<set>.json + analysis
cd /Users/mfaseehuddin/mfaseehuddin/development/wsp
A=experiments/pong/agents/b250k
[ -f $A/runs/obs_$2.json ] || uv run python $A/tools/probes.py gen $2 $A/runs/obs_$2.json
uv run wsp-pong probe --agent b250k --policy $1 --obs "$(cat $A/runs/obs_$2.json)" --note "$4" > $A/runs/out_$3_$2.json || exit 1
uv run python $A/tools/probes.py analyse $A/runs/obs_$2.json $A/runs/out_$3_$2.json > $A/runs/ana_$3_$2.json
python3 -c "import json;d=json.load(open('$A/runs/ana_$3_$2.json'));print('$3 $2', 'max=%.5f p99=%.5f rms=%.5f' % (d['max_abs'],d['p99_abs'],d['rms']))"
