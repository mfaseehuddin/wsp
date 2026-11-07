import sys, json, importlib.util, torch
from pathlib import Path
from wsp.ncmchat import model
from wsp.ncmchat.harness import load, Chat, tok
D="experiments/ncmchat/agents/nco60"
spec=importlib.util.spec_from_file_location("w",f"{D}/weights.py"); W=importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
cfg,P,_=load(Path(f"{D}/weights.py"))
T=tok()
def level0_feats(ch):
    c=ch.cache; end=len(c)
    x=P["lvl_emb"][0]+P["lvl_pos"][:1]
    pos=torch.full((1,),end)
    for i in range(cfg.n_layers):
        x,_,_=model._layer(cfg,P,i,x,pos,c.k[i],c.v[i],None,None)
    fe={k:round(x[0,W.f[k]].item(),2) for k in ["PG","PB","PT","PH","PQ","PN","PC","CT","CS"]}
    logits=(x@P["head0"]+P["head_b0"])[0]
    return fe, W.MODES[logits.argmax().item()]
convs=json.load(open(sys.argv[1])) if len(sys.argv)>1 else [["Hello!","Hi, I'm Teodra.","What is the capital of France?"]]
for conv in convs:
    ch=Chat(cfg,P); print("=====")
    for t in conv:
        r=ch.turn(t)
        # features for the turn just answered: recompute on cache truncated? approximate: show reply only
        print("U:",t,"\n  A:",repr(r["reply"]))
