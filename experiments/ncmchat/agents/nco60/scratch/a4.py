import torch
from safetensors import safe_open
from wsp.ncmchat.harness import tok, ids
QW="models/Qwen2.5-0.5B/model.safetensors"
with safe_open(QW,"pt") as f: E=f.get_tensor("model.embed_tokens.weight").float()
T=tok(); En=E/E.norm(dim=1,keepdim=True)
mu0=En[:100000].mean(0); U,S,V=torch.pca_lowrank(En[:100000]-mu0,q=4,center=False)
def dirn(pairs):
    d=sum(En[ids(a)[0]]-En[ids(b)[0]] for a,b in pairs); return d/d.norm()
dcap=dirn([(" Apple"," apple"),(" River"," river"),(" Stone"," stone")])
dsp=dirn([(" cat","cat"),(" house","house"),(" water","water")])
for txt in [" Teodra and Zorbin visited Vantoria with Biscotto, Oskarine, Quillo and Darnet. My dog sleeps at noon; pelins are purple glimps. The mango notebook belongs to Nilo near Kofi, Ines, Bram, Mira, Omar and Tess yesterday",
 " I'm Rosa and he is Kwame. They gave it to Liesel who owns the Fenwick Shop, then went to Ulaanbaatar for Christmas. Hi, I'm Teodra. Hello there! My teammate's name is Oskarine"]:
  I=ids(txt); c=En[I]@dcap; p=(En[I]-mu0)@V[:,0]; s=En[I]@dsp
  print(" ".join(f"{T.decode([i])!r}:{a:.2f}" for i,a in zip(I,c.tolist())))
print("quant",(En@dcap).quantile(torch.tensor([.5,.9,.99])))
