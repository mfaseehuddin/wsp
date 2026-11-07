import importlib.util, torch
from pathlib import Path
from wsp.ncmchat import model
from wsp.ncmchat.harness import load, tok, ids
D="experiments/ncmchat/agents/nco60"
spec=importlib.util.spec_from_file_location("w",f"{D}/weights.py"); W=importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
cfg,P,_=load(Path(f"{D}/weights.py")); T=tok()
I=ids("<|im_start|>user\nHi, I'm Teodra. What is it?<|im_end|>\n<|im_start|>assistant\n")
x=P["tok_emb"][torch.tensor(I)]; pos=torch.arange(len(I))
c=model.Cache(cfg)
for i in range(cfg.n_layers):
    x,k,v=model._layer(cfg,P,i,x,pos,c.k[i],c.v[i],0,None)
    if i==1:
        md=x[:,W.f["MOD0"]:W.f["MOD0"]+32]; tg=x[:,W.f["TG0"]:W.f["TG0"]+32]
        for j,t in enumerate(I): print(repr(T.decode([t])), "CS",round(x[j,W.f["CS"]].item(),4),"CT",round(x[j,W.f["CT"]].item(),3),"mod",md[j].argmax().item(),round(md[j].max().item(),2),"tg",tg[j].topk(2).indices.tolist(),"F",round(x[j,W.f["FIRST"]].item(),2),"G1",round(x[j,W.f["G1"]].item(),2),"RS",x[j,W.f["RS"]].item())
