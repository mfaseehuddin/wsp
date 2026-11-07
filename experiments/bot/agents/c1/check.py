"""Static check of the script table (symbolic; no forward pass)."""
import itertools, weights as W
from wsp.bot import kb
units = W.script_units()
vals = {"$ID": ["4049"], "@ID": ["4049"], "$TOPIC": list(kb.FAQ), "$TEAMH": kb.TEAMS, "$TEAMT": kb.TEAMS,
        "$SVC": kb.SERVICES, "$ST": kb.STATES + kb.STATUSES, "@ST": kb.STATES + kb.STATUSES, "$USR": ["jsmith"]}
def feats(w):
    return {("w", w)} | {("c", c) for c in W.class_of(w)}
bad = 0
for intent, seqs in W.templates().items():
    us = [u for u in units if u[0] == intent]
    for s in seqs:
        s = [x for x in s if x != "#R"]
        slots = sorted({x for x in s if x in vals})
        for combo in itertools.product(*[vals[k] for k in slots]):
            sub = dict(zip(slots, combo))
            if "$TEAMH" in sub and "$TOPIC" in sub and sub["$TEAMH"] != kb.FAQ[sub["$TOPIC"]][0]: continue
            if "$ST" in sub and "@ST" in sub and sub["$ST"] != sub["@ST"]: continue
            c = [sub.get(x, x) for x in s]
            harness = set()
            for i, x in enumerate(s):
                if x == "</tool>":
                    j = i + 1
                    while s[j] != "</result>": harness.add(j); j += 1
                    harness.add(j)
            for i in range(len(s) - 1):
                if i + 1 in harness: continue
                cf, pf = feats(c[i]), feats(c[i - 1]) if i else set()
                fired = {u[3] for u in us if u[1] in cf and (u[2] is None or u[2] in pf)}
                fired = {sub.get(o, o) for o in fired}
                if fired != {c[i + 1]}:
                    bad += 1
                    if bad < 15: print(intent, c[i - 1:i + 1], "->", c[i + 1], "got", fired)
print("bad", bad, "units", len(units))
