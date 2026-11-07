"""Symbolic check of the design tables (no tensors, no forward pass).
1) every script position fires exactly the intended unit;
2) the hand-written keyword votes pick the intended script/topic on dev turns."""
import importlib.util, sys, random
from wsp.bot import kb, tokenizer as T
spec = importlib.util.spec_from_file_location("w", __file__.replace("check_design.py", "weights.py"))
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
scr = W.scripts(); code, C, G = W.colouring(scr); tab = W.units(scr, code)
print("C", C, "G", G, "units", len(tab))
assert len(set(code.values())) == len(code)
bad = 0
def fire(F):
    out = set()
    for k, a in tab.items():
        if all((c[1] not in F) if isinstance(c, tuple) and c[0] == "not" else (c in F) for c in k):
            out.add(a)
    return out
rng = random.Random(0)
for it, topic, s in scr:
    for trial in range(40):
        fill = {f"{{{c}}}": rng.choice(m) for c, m in W.CLASSES.items()}
        seq = ["<bot>", *[fill.get(w, w) for w in s]]
        inside = False
        for i in range(len(seq) - 1):
            if seq[i] == "<result>": inside = True
            if seq[i] == "</result>": inside = False
            if inside or seq[i] == "</tool>": continue
            stale_k = rng.choice(W.INTENTS) if i < 2 else it
            tt = topic if (topic and i >= 3) else rng.choice(W.TOPICS)
            F = {("K", stale_k), ("T", tt)} | W.feats(seq[i], code) | {("p", f) for f in W.feats(seq[i - 1], code)}
            got = fire(F)
            nxt = s[i] if i < len(s) else None
            if i == 1: want = {("suppress", "T")}
            elif i == 2 and it in ("faq", "page"): want = {("suppress", "K")}
            elif i == 2: want = {("out", "</think>")}
            else: want = {("allow", nxt[1:-1]) if nxt.startswith("{") else ("out", nxt)}
            if i == 0: want = {("out", "<think>")}
            if got != want:
                bad += 1
                if bad < 10: print("BAD", it, topic, i, seq[i], "want", want, "got", got)
print("unit check failures:", bad)
# keyword votes on dev turns
kw = W.keyword_votes(); K = W.K_WORD; inv = {v: k for k, v in K.items()}; tinv = {v: k for k, v in W.T_WORD.items()}
def decide(user):
    toks = T.split(user)[:40]
    sc = {}
    for w in toks:
        for tgt, x in kw.get(w, []): sc[tgt] = sc.get(tgt, 0) + x
    ks = {k: sc.get(v, 0) for k, v in K.items()}; ks["unknown"] += 0.005 * (len(toks) + 3)
    k = max(ks, key=ks.get)
    ts = {t: sc.get(v, 0) for t, v in W.T_WORD.items()}
    return k, max(ts, key=ts.get)
flow = {"create_ticket": "esc", "lookup_ticket": "status", "service_status": "service", "page_team": "page", "reset_password": "reset"}
social = {v: k for k, v in kb.SOCIAL.items()}
nbad = 0
for c in kb.conversations("dev", 0):
    for t in c["turns"]:
        if t["tools"]: want = flow[t["tools"][0][0]]; wt = None
        elif t["reply"] in social: want = social[t["reply"]]; wt = None
        elif t["reply"].startswith("your ticket number"): want = "recall"; wt = None
        else: want = "faq"; wt = [x for x, (_, r) in kb.FAQ.items() if r == t["reply"]][0]
        if want == "page": wt = [x for x in kb.FAQ if kb.FAQ[x][0] == t["tools"][0][1]]
        k, tp = decide(t["user"])
        ok = k == want and (wt is None or tp == wt or (isinstance(wt, list) and tp in wt))
        if not ok:
            nbad += 1; print("VOTE", repr(t["user"]), "want", want, wt, "got", k, tp)
print("vote failures:", nbad)
