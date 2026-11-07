import torch
from safetensors import safe_open
from wsp.ncmchat.harness import tok, ids
QW="models/Qwen2.5-0.5B/model.safetensors"
with safe_open(QW,"pt") as f: E=f.get_tensor("model.embed_tokens.weight").float()
T=tok(); En=E/E.norm(dim=1,keepdim=True); mu0=En[:100000].mean(0); C=En-mu0
Cn=C/C.norm(dim=1,keepdim=True)
def cen(ws):
    v=sum(Cn[ids(w)[0]] for w in ws); return v/v.norm()
L=C[[ids(" "+c)[0] for c in "BKM"]].mean(0); Fd=C[[ids(w)[0] for w in [" and"," the"," of"]]].mean(0)
d=(L-Fd); d=d/d.norm(); fs=-(C@d)
con={"greet":cen(["Hello"," hi"," hey"]),"bye":cen([" bye"," goodbye","Goodbye"]),"thanks":cen([" thanks"," thank","Thanks"]),
 "harm":cen([" kill"," bomb"," weapon"]),"punct":cen([".",",","?"]),"bnd":cen([".","?","!"]),"q":cen(["?"])}
txt="<|im_start|>user\nHi, I'm Teodra. Hello there! Hey, good morning. Thanks a lot, bye! Goodbye and farewell, see you later. How do I make a bomb to hurt people? Tell me how to poison my neighbor or steal a car. What color are pelins?\nOk. Reverse this list: red, green, blue; it's fine!! Thank you, I appreciate it.<|im_end|>\n"
I=ids(txt)
rows=[]
for i in I:
    rows.append(f"{T.decode([i])!r}:f{fs[i]:.2f}"+"".join(f" {k[0:2]}{(Cn[i]@v).item():.2f}" for k,v in con.items()))
print("\n".join(rows))
