import torch
from safetensors import safe_open
from wsp.ncmchat.harness import ids
with safe_open("models/Qwen2.5-0.5B/model.safetensors","pt") as f: E=f.get_tensor("model.embed_tokens.weight").float()
En=E/E.norm(dim=1,keepdim=True); C=En-En[:100000].mean(0); Cn=C/C.norm(dim=1,keepdim=True)
A=[ids(w)[0] for w in ["4","1","7"]]; P=[ids(w)[0] for w in [".",",","?"]]
for w in ["0","2","3","5","6","8","9",".",",","!",":","-"," ","a","x","pm"," the","(","%"]:
    i=ids(w)[0]; print(repr(w), round(max((Cn[i]@Cn[a]).item() for a in A),2), round((Cn[i]@(sum(Cn[p] for p in P)/3)).item(),2), end=" | ")
