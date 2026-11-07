import torch
from safetensors import safe_open
with safe_open("models/Qwen2.5-0.5B/model.safetensors","pt") as f:
    E=f.get_tensor("model.embed_tokens.weight").float()
dig=list(range(15,25)); R=dig+[10,28]; Er=E[R].double()
def fun(tv):  # tv: target values for the 12 relevant tokens
    G=Er@Er.T; a=torch.linalg.solve(G,torch.tensor(tv,dtype=torch.float64)); w=(Er.T@a).float()
    s=E@w; s[R]=-9; m,i=s.max(0); return 1-m.item(), i.item(), w.norm().item()
for name,tv in [("P",[0]*10+[1,0]),("E",[0]*10+[0,1]),("PE",[0]*10+[1,1]),("DIG",[1]*10+[0,0]),("DIGE",[1]*10+[0,1])]:
    print(name, fun(tv))
