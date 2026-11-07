import importlib.util, torch, math, sys, json
from pathlib import Path
from wsp.ncmchat import model
from wsp.ncmchat.harness import load, tok, ids, special
D="experiments/ncmchat/agents/nco60"
spec=importlib.util.spec_from_file_location("w",f"{D}/weights.py"); W=importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
cfg,P,_=load(Path(f"{D}/weights.py")); T=tok(); f=W.f; L=W.L
turns=json.loads(sys.argv[1]); LAYER=int(sys.argv[2]); HEAD=int(sys.argv[3])
text="".join(f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"+("OK.<|im_end|>\n" if i<len(turns)-1 else "") for i,t in enumerate(turns))
I=ids(text); c=model.Cache(cfg)
xs=P["tok_emb"][torch.tensor(I)]; pos=torch.arange(len(I)); X=[]
for i in range(cfg.n_layers):
    X.append(xs); xs,k,v=model._layer(cfg,P,i,xs,pos,c.k[i],c.v[i],0,None); c.k[i]=k; c.v[i]=v
c.ids=I
z=torch.zeros(64,cfg.D); z[:,W.R+W.M[sys.argv[5] if len(sys.argv)>5 else "ANSWER"]]=1
x=model.down(z,1)@P["w_in"]+P["lvl_emb"][1]+P["lvl_pos"][:1]; p1=torch.full((1,),len(I))
for i in range(LAYER): x,_,_=model._layer(cfg,P,i,x,p1,c.k[i],c.v[i],None,None)
pre=f"L{LAYER}."; h=HEAD
q=x@P[pre+"wq"][h]+P[pre+"bq"][h]; q=model.rope(q[None],p1,cfg.rope_theta)[0,0]
Xp=X[LAYER]; kk=Xp@P[pre+"wk"][h]+P[pre+"bk"][h]; kk=model.rope(kk[None],pos,cfg.rope_theta)[0]
# group key dims by source slot
srcs={}
for name,sl in list(L.items()):
    if name=="F": continue
    srcs[name]=list(range(sl.start,sl.stop))
for nm in W.FN: srcs[nm]=[f[nm]]
contrib={}
for name,rows in srcs.items():
    Wk=P[pre+"wk"][h][rows]
    dims=(Wk.abs().sum(0)>0).nonzero().flatten()
    if len(dims)==0: continue
    contrib[name]=dims
tot=(kk@q)/math.sqrt(cfg.d_head)
top=tot.topk(int(sys.argv[4]) if len(sys.argv)>4 else 8).indices.tolist()
for j in top:
    parts={n:round(((kk[j,d]*q[d]).sum()/math.sqrt(cfg.d_head)).item(),1) for n,d in contrib.items()}
    parts={n:v for n,v in parts.items() if abs(v)>0.4}
    print(f"{T.decode([I[j]])!r:12s} tot={tot[j].item():.1f}",parts)
