import torch
from safetensors import safe_open
from wsp.ncmchat.harness import ids
QW="models/Qwen2.5-0.5B/model.safetensors"
with safe_open(QW,"pt") as f: E=f.get_tensor("model.embed_tokens.weight").float()
En=E/E.norm(dim=1,keepdim=True); mu0=En[:100000].mean(0); C=En-mu0; Cn=C/C.norm(dim=1,keepdim=True)
A=[Cn[ids(w)[0]] for w in ["Write","Tell","Give"]]
cen=sum(A); cen=cen/cen.norm()
for w in "Make Book Reply Reverse Compute Multiply Summarize Rewrite Explain List Name Say Describe Translate Count Please My Lena Zorbin He All In Correction What How Hello Kofi Mira The I".split():
    i=ids(w)[0]; print(w, round(max((Cn[i]@a).item() for a in A),2), round((Cn[i]@cen).item(),2), end=" | ")
