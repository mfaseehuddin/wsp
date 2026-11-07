import importlib.util, torch, math
from pathlib import Path
from wsp.ncmchat import model
from wsp.ncmchat.harness import load, tok, ids, special
D="experiments/ncmchat/agents/nco60"
spec=importlib.util.spec_from_file_location("w",f"{D}/weights.py"); W=importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
cfg,P,_=load(Path(f"{D}/weights.py")); T=tok(); f=W.f
c=model.Cache(cfg)
model.extend(cfg,P,c,ids("<|im_start|>user\nIn my fictional world, pelins are purple birds that sleep at noon. What color are pelins?<|im_end|>\n<|im_start|>assistant\n"))
z=torch.zeros(64,cfg.D); z[:,W.R+W.M["ANSWER"]]=1
x=z@P["w_in"]+P["lvl_emb"][1]+P["lvl_pos"][:64]; pos=torch.full((64,),len(c))
for i in range(5): x,_,_=model._layer(cfg,P,i,x,pos,c.k[i],c.v[i],None,None)
A=x[0,W.L["A"]]; print("A norm",A.norm().item(), "best tok", T.decode([(P["tok_emb"][:,W.L["ID"]]@A).argmax().item()]))
q=W.rope(torch.einsum("td,de->te",x[:1],P["L5.wq"][0])[None]+P["L5.bq"][0][None,None],pos[:1],cfg.rope_theta)[0] if False else None
h=1
qq=(x[:1]@P["L5.wq"][h]+P["L5.bq"][h])[None]
qq=model.rope(qq,pos[:1],cfg.rope_theta)[0,0]
K=c.k[5][h]
s=(K@qq)/math.sqrt(cfg.d_head)
for j in s.topk(4).indices.tolist(): print(T.decode([c.ids[j]]), s[j].item())
# sink score
kl=(x[:1]@P["L5.wk"][h]+P["L5.bk"][h])[None]; kl=model.rope(kl,pos[:1],cfg.rope_theta)[0,0]
print("sink", (kl@qq).item()/math.sqrt(cfg.d_head))
print({k:round(x[0,f[k]].item(),2) for k in ["SINK","LVL","ONE","G1","G2","G3","PRE","W","STOP"]})
prod=(qq*kl)/math.sqrt(cfg.d_head)
nz=prod.abs()>0.5
print([(i,round(prod[i].item(),1)) for i in torch.nonzero(nz).flatten().tolist()])
