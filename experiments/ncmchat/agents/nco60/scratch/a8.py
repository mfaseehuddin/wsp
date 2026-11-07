import torch
from safetensors import safe_open
from wsp.ncmchat.harness import tok, ids
QW="models/Qwen2.5-0.5B/model.safetensors"
with safe_open(QW,"pt") as f: E=f.get_tensor("model.embed_tokens.weight").float()
T=tok(); En=E/E.norm(dim=1,keepdim=True); mu0=En[:100000].mean(0); C=En-mu0
torch.manual_seed(0); _,_,Vp=torch.pca_lowrank(C[:100000],q=4,center=False); pc1=C@Vp[:,0]
if pc1[ids(" the")[0]]<0: pc1=-pc1
L=C[[ids(" "+c)[0] for c in "BKM"]].mean(0); Fd=C[[ids(w)[0] for w in [" and"," the"," of"]]].mean(0)
d=Fd-L; d=d/d.norm(); fs=C@d
ws=" am is are was were be been I you he she it we they me my your his her our their this that these those what who how when where why which do does did have has had will would can could should not no yes very just also so but if then than as at by for from in on of to with about into over after before up down out Hi Hello dog called old years four name fruit pear mango notebook red green Lena Teodra Biscotto capital Paris color sleep noon purple birds today doing make bomb Thanks bye favorite current".split()
out=[]
for w in ws:
    i=ids(" "+w)[0]; out.append(f"{w}:{fs[i]:.2f}/{pc1[i]:.2f}")
print(" ".join(out))
