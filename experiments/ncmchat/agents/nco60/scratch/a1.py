import torch
from safetensors import safe_open
from wsp.ncmchat.harness import tok, ids
QW="models/Qwen2.5-0.5B/model.safetensors"
with safe_open(QW,"pt") as f: E=f.get_tensor("model.embed_tokens.weight").float()
n=E.norm(dim=1)
T=tok()
def show(w): 
    i=ids(w); return [(T.decode([j]),j,round(n[j].item(),3)) for j in i]
for w in [" the"," is"," of"," and"," a"," capital"," France"," Paris"," purple"," Teodra"," Zorbin"," noon"," sleep",".","?",","," What"," Who"," color"," green"," mango","Hello"," hello"]:
    print(w, show(w))
print("norm quantiles", n.quantile(torch.tensor([.01,.1,.5,.9,.99])))
En=E/n[:,None]
mu=En.mean(0); Ec=En-mu; Ec=Ec/Ec.norm(dim=1,keepdim=True)
def nn(vec,k=12,excl=()):
    s=Ec@ (vec/vec.norm()); s[list(excl)]=-9
    t=s.topk(k).indices.tolist(); return [T.decode([j]) for j in t]
def v(w): return Ec[ids(w)].mean(0)
for q in [[" bees"," make"],[" spider"," legs"],[" capital"," France"],[" grass"," color"],[" whale"," mammal"," fish"],[" sky"," color"],[" opposite"," cold"]]:
    vec=sum(v(w) for w in q); ex=[i for w in q for i in ids(w)]
    print(q, nn(vec,excl=ex))
