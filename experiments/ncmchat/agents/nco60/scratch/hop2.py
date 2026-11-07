import importlib.util, torch, math
from pathlib import Path
from wsp.ncmchat import model
from wsp.ncmchat.harness import load, tok, ids
D="experiments/ncmchat/agents/nco60"
spec=importlib.util.spec_from_file_location("w",f"{D}/weights.py"); W=importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
cfg,P,_=load(Path(f"{D}/weights.py")); L=W.L
c=model.Cache(cfg)
model.extend(cfg,P,c,ids("<|im_start|>user\nIn my world, pelins are purple birds. What color are pelins?<|im_end|>\n<|im_start|>assistant\n"))
z=torch.zeros(64,cfg.D); z[:,W.R+W.M["ANSWER"]]=1
x=z@P["w_in"]+P["lvl_emb"][1]+P["lvl_pos"][:64]; pos=torch.full((64,),len(c))
print({k:round(x[0,v].norm().item(),3) for k,v in L.items()})
for i in range(5):
    x,_,_=model._layer(cfg,P,i,x,pos,c.k[i],c.v[i],None,None)
    print(i,{k:round(x[0,v].norm().item(),3) for k,v in L.items()})
