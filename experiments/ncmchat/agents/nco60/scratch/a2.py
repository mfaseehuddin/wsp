import torch
from safetensors import safe_open
from wsp.ncmchat.harness import tok, ids
QW="models/Qwen2.5-0.5B/model.safetensors"
with safe_open(QW,"pt") as f: E=f.get_tensor("model.embed_tokens.weight").float()
T=tok()
En=E/E.norm(dim=1,keepdim=True); mu=En[:100000].mean(0)
pm=(En@mu/mu.norm())
U,S,V=torch.pca_lowrank(En[:100000]-En[:100000].mean(0),q=8,center=False)
P=(En-En[:100000].mean(0))@V
words=" the is of and a to in my What Who How When Where Why color green sleep capital France Paris Zorbin Teodra mango noon notebook owns called dog four years old Lena Daro purple birds world fictional".split()
for w in words:
    i=ids(" "+w)[0]
    print(f"{w:10s} {i:6d} mean={pm[i]:.3f} pcs="+" ".join(f"{x:+.2f}" for x in P[i,:6].tolist()))
# word-start direction from 3 pairs
pairs=[(" an","an"),(" in","in"),(" the","the")]
d=sum(En[ids(a)[0]]-En[ids(b)[0]] for a,b in pairs); d=d/d.norm()
pr=En@d
for w in [" Te","od","ra"," Z","or","bin"," Vant","oria"," cat","ing","ed"," mango",".","?"," Bisc"]:
    i=ids(w); print(repr(w), i, [round(pr[j].item(),3) for j in i])
print(pr.quantile(torch.tensor([.1,.3,.5,.7,.9])))
