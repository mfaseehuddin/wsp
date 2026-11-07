import importlib.util, torch, sys
from pathlib import Path
from wsp.ncmchat import model
from wsp.ncmchat.harness import load, tok, ids
D="experiments/ncmchat/agents/nco60"
spec=importlib.util.spec_from_file_location("w",f"{D}/weights.py"); W=importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
cfg,P,_=load(Path(f"{D}/weights.py")); T=tok()
I=ids("<|im_start|>user\n"+sys.argv[1])
x=P["tok_emb"][torch.tensor(I)]; pos=torch.arange(len(I)); c=model.Cache(cfg)
FE=sys.argv[2].split(",")
for i in range(int(sys.argv[3])):
    x,k,v=model._layer(cfg,P,i,x,pos,c.k[i],c.v[i],0,None)
for j,t in enumerate(I): print(f"{T.decode([t])!r:12s}"+" ".join(f"{k}={x[j,W.f[k]].item():.2f}" for k in FE))
E=P["tok_emb"][:,W.L["ID"]]
for j,t in enumerate(I):
    r=x[j,W.L["RSID"]]; b=(E@r).topk(2)
    print(repr(T.decode([t])), "RSID->", [(T.decode([i]),round(v,2)) for v,i in zip(b.values.tolist(),b.indices.tolist())])
