import torch
from safetensors import safe_open
from wsp.ncmchat.harness import tok, ids
QW="models/Qwen2.5-0.5B/model.safetensors"
with safe_open(QW,"pt") as f: E=f.get_tensor("model.embed_tokens.weight").float()
T=tok(); En=E/E.norm(dim=1,keepdim=True); mu0=En[:100000].mean(0); C=En-mu0; C=C/C.norm(dim=1,keepdim=True)
L=sum(C[ids(w)[0]] for w in [" B"," K"," M"]); L=L/L.norm()
txt=" Teodra and Zorbin visited Vantoria with Biscotto, Oskarine, Quillo and Darnet. My dog sleeps at noon; I'm Rosa and he is Kwame. They gave it to Liesel who owns the Fenwick Shop, then went to Ulaanbaatar for Christmas. Hi, I'm Teodra. A cat and the Elk of an Ice"
I=ids(txt); c=C[I]@L
print(" ".join(f"{T.decode([i])!r}:{a:.2f}" for i,a in zip(I,c.tolist())))
