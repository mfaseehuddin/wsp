import sys, json, importlib.util, torch
from pathlib import Path
from wsp.ncmchat import model
from wsp.ncmchat.harness import load, Chat, tok, ids, special
D="experiments/ncmchat/agents/nco60"
spec=importlib.util.spec_from_file_location("w",f"{D}/weights.py"); W=importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
cfg,P,_=load(Path(f"{D}/weights.py")); T=tok()
FE=sys.argv[2].split(",") if len(sys.argv)>2 else ["PQ","PN","PC","PM","PY","PI"]
def run_level(c,l,z,n):
    x=model.down(z,n)@P["w_in"]+P["lvl_emb"][l]+P["lvl_pos"][:n]
    pos=torch.full((n,),len(c)); att=[]
    for i in range(cfg.n_layers):
        x,_,_=model._layer(cfg,P,i,x,pos,c.k[i],c.v[i],None,att)
    return x,att
def topatt(c,att,layer,head,row=0,k=3):
    a=att[layer][head,row]; t=a[:len(c)].topk(k)
    return [(T.decode([c.ids[i]]),round(v,2)) for v,i in zip(t.values.tolist(),t.indices.tolist())]+[("LV",round(a[len(c):].sum().item(),2))]
for conv in json.load(open(sys.argv[1])):
    c=model.Cache(cfg); print("=====")
    for t in conv:
        model.extend(cfg,P,c,ids("<|im_start|>user\n"+t+"<|im_end|>\n<|im_start|>assistant\n"))
        z=torch.zeros(64,cfg.D)
        x,att=run_level(c,0,z,1)
        mode=(x@P["head0"]+P["head_b0"]).argmax(-1)
        print("U:",t,"| mode",W.MODES[mode.item()],{k:round(x[0,W.f[k]].item(),2) for k in FE})
        print("   CAND",topatt(c,att,4,0),"NAME",topatt(c,att,3,1),"MATCH",topatt(c,att,3,0))
        print("   QSENT",topatt(c,att,2,0,k=6),"PQ",topatt(c,att,2,5,k=4),"ROLE",topatt(c,att,2,6,k=2))
        z=z+model.up(P["code0"][mode],1)
        x,att=run_level(c,1,z,64); print("   L1 PC/PCB", x[5,W.f["PC"]].item(), x[5,W.f["PCB"]].item(), "CAND1", topatt(c,att,4,0,row=5))
        codes=(x@P["head1"]+P["head_b1"]).argmax(-1).tolist()
        r=[];
        for cc in codes:
            if cc==special("<|im_end|>"): break
            r.append(cc)
        print("   A:",repr(T.decode(r)), "hops",[topatt(c,att,5,hh,row=5,k=1) for hh in range(4)])
        model.extend(cfg,P,c,r+[special("<|im_end|>")]+ids("\n"))
