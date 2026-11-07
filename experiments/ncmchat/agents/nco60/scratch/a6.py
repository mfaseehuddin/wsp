import torch
from safetensors import safe_open
from wsp.ncmchat.harness import tok, ids
QW="models/Qwen2.5-0.5B/model.safetensors"
with safe_open(QW,"pt") as f: E=f.get_tensor("model.embed_tokens.weight").float()
n=E.norm(dim=1); En=E/n[:,None]; mu0=En[:100000].mean(0); C=En-mu0
Ln=[ids(" "+c)[0] for c in "BKM"]; L=C[Ln].mean(0)
F=[ids(w)[0] for w in [" and"," the"," of"]]; Fd=C[F].mean(0)
d=L-Fd; d=d/d.norm()
let=[ids(" "+c)[0] for c in "ABCDEFGHIJLNOPRSTUVWZ"]
fun=[ids(w)[0] for w in " is, to, at, with, for, in, a, an, I, he, it, who, then, my, are, was, on, by, as, or, that, this, we, you, they".split(",")]
con=[ids(w)[0] for w in [" Te"," Lies"," Kw"," Fen"," Rosa"," dog"," Qu"," My"," Hi"," Elk"]]
for nm,g in [("letters",let),("func",fun),("content",con)]:
    p=(C[g]@d); print(nm, [round(x,2) for x in p.tolist()], "norm",[round(x,2) for x in n[g].tolist()][:8])
