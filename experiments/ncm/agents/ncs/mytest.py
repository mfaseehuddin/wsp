import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import weights as W
from wsp.ncm import model, tokenizer as T
cfg, P, _ = W.build()
def run(q):
    ids = T.encode(q)[:32]
    out = model.generate(cfg, P, ids)
    toks = out["tokens"]; r = []
    for t in toks:
        if t == 1: break
        r.append(T.vocab()[t] if hasattr(T,'vocab') else t)
    return " ".join(r)
def expect(intent, name=None, ticket=None):
    r = W.REPLIES[intent]
    if name: r = f"hi {name} , " + r
    if ticket: r += " i added this to ticket " + " ".join(ticket) + " ."
    return r
tests = [
 ("is there a gym nearby", "unknown",None,None),
 ("what is the best laptop to buy", "unknown",None,None),
 ("can you help me with my homework", "unknown",None,None),
 ("how do i bake bread", "unknown",None,None),
 ("what is the capital of france", "unknown",None,None),
 ("where is the nearest coffee shop", "unknown",None,None),
 ("what time is my meeting", "unknown",None,None),
 ("tell me a joke please", "unknown",None,None),
 ("hi , i am nina . what is the weather", "unknown","nina",None),
 ("hi there", "hello",None,None),
 ("hello there , anyone here ?", "hello",None,None),
 ("hey can you help me", "hello",None,None),
 ("no space left on my laptop", "disk",None,None),
 ("pc takes forever to start", "slow",None,None),
 ("my account is locked", "locked",None,None),
 ("monitor shows no signal", "monitor",None,None),
 ("external screen not detected", "monitor",None,None),
 ("that worked thanks", "thanks",None,None),
 ("how do i connect two screens", "monitor",None,None),
 ("how do i get on the vpn from home", "vpn",None,None),
 ("can you install excel for me", "software",None,None),
 ("it says my user is locked", "locked",None,None),
 ("no internet on the office wifi", "wifi",None,None),
 ("i can't connect to the wifi", "wifi",None,None),
 ("i got a new phone and need mfa", "mfa",None,None),
 ("two factor login fails", "mfa",None,None),
 ("i did not get the authenticator code", "mfa",None,None),
 ("print job stuck", "printer",None,None),
 ("i am not getting emails", "email",None,None),
]
bad = 0
for q, i, n, t in tests:
    r = run(q); e = expect(i, n, t)
    if T.encode(r) != T.encode(e):
        bad += 1; print("FAIL", q, "\n  got ", r[:110], "\n  want", e[:110])
print(len(tests) - bad, "/", len(tests))
