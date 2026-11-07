import importlib.util, torch, sys
from wsp.ncmchat.harness import tok, ids
D="experiments/ncmchat/agents/nco60"
spec=importlib.util.spec_from_file_location("w",f"{D}/weights.py"); W=importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
cfg,P,_=W.build(); T=tok(); f=W.f
I=ids(sys.argv[1])
for t in I:
    e=P["tok_emb"][t]
    print(f"{T.decode([t])!r:14s}"+" ".join(f"{k}={e[f[k]].item():.2f}" for k in ["W","STOP","BND","QM","GREET","HARM"]))
