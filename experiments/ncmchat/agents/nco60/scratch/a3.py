import torch
from safetensors import safe_open
from wsp.ncmchat.harness import tok, ids
QW="models/Qwen2.5-0.5B/model.safetensors"
with safe_open(QW,"pt") as f: E=f.get_tensor("model.embed_tokens.weight").float()
T=tok()
En=E/E.norm(dim=1,keepdim=True)
txt=" Teodra and Zorbin visited Vantoria with Biscotto, Oskarine, Quillo and Darnet. My dog sleeps at noon; pelins are purple glimps. The mango notebook belongs to Nilo near Kofi, Ines, Bram, Mira, Omar and Tess yesterday photosynthesis lantern"
I=ids(txt); lab=torch.tensor([T.decode([i]).startswith(" ") for i in I])
for name,pairs in [("cw",[(" cat","cat"),(" house","house"),(" water","water")]),("mix",[(" day","day"),(" form","form"),(" light","light")])]:
    d=sum(En[ids(a)[0]]-En[ids(b)[0]] for a,b in pairs); d=d/d.norm(); pr=En[I]@d
    ws=pr[lab]; cs=pr[~lab]
    print(name, "start min/mean", ws.min().item(), ws.mean().item(), "cont max/mean", cs.max().item(), cs.mean().item())
    bad=[(T.decode([i]),round(p,3)) for i,p,l in zip(I,pr.tolist(),lab.tolist()) if (l and p<0.0) or (not l and p>0.0)]
    print(bad)
mu0=En[:100000].mean(0)
U,S,V=torch.pca_lowrank(En[:100000]-mu0,q=4,center=False)
pc1=(En[I]-mu0)@V[:,0]
d=sum(En[ids(a)[0]]-En[ids(b)[0]] for a,b in [(" cat","cat"),(" house","house"),(" water","water")]); d=d/d.norm(); cw=En[I]@d
txt2=" I'm Rosa and he is Kwame. They gave it to Liesel who owns the Fenwick Shop, then went to Ulaanbaatar for Christmas"
I2=ids(txt2); pc2=(En[I2]-mu0)@V[:,0]; cw2=En[I2]@d
for II,P,C in [(I,pc1,cw),(I2,pc2,cw2)]:
  print(" ".join(f"{T.decode([i])!r}:{p:.2f}/{c:.2f}" for i,p,c in zip(II,P.tolist(),C.tolist())))
