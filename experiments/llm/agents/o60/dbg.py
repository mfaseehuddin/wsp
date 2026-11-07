import numpy as np, torch, sys
z=np.load("reads/"+sys.argv[1]+".npz"); d=torch.load("dbg_basis.pt"); B=d["B"].numpy(); nm=d["names"]; sg=d["sig"]
ti=int(sys.argv[2]) if len(sys.argv)>2 else 0
ids=z["input_ids"][ti]; T=z["lengths"][ti]
def q(l,name,pos): return float(z[f"resid_L{l}"][ti,pos]@B[:,nm[name]]/sg[name])
pl=[i for i in range(T) if ids[i]==10][-1]; eq=[i for i in range(T) if ids[i]==28][-1]
f=lambda v: " ".join(f"{x:.2f}" for x in v)
print("plus",pl,"eq",eq)
print("L1 flags DIG/VAL/E/P @plus-1,plus,eq:",[f([q(1,n,p) for n in ["DIG","VAL","E","P"]]) for p in (pl-1,pl,eq)])
print("L2 FD1..9 @plus",f([q(2,f"FD{k}",pl) for k in range(1,10)]),"FV",f([q(2,f"FV{k}",pl) for k in range(1,10)]))
print("L2 ZO@plus",q(2,"ZO",pl),"VLD@plus",f([q(2,f"VLD{p}",pl) for p in range(12)]))
print("L2 MM@plus",f([q(2,f"MM{p}",pl) for p in range(12)]), "ZS",f([q(2,f"ZS{p}",pl) for p in range(12)]))
print("L2 MM@eq",f([q(2,f"MM{p}",eq) for p in range(12)]),"PB",q(2,"PB",eq))
print("L3 AVm",f([q(3,f"AVm{p}",pl) for p in range(12)]),"AMm",f([q(3,f"AMm{p}",pl) for p in range(12)]))
print("L4 AV@eq",f([q(4,f"AV{p}",eq) for p in range(12)]),"AM",f([q(4,f"AM{p}",eq) for p in range(12)]))
print("L4 CP",f([q(4,f"CP{p}",eq) for p in range(1,13)]),"V",q(4,"V",eq),"GL",f([q(4,f"GL{p}",eq) for p in range(1,13)]))
print("L5 SC",f([q(5,f"SC{p}",eq) for p in range(13)]),"NN",q(5,"NN",eq))
print("L6 XS",f([q(6,f"XS{p}",eq) for p in range(13)]),"XN",q(6,"XN",eq),"XV",q(6,"XV",eq))
for p in range(eq,min(T,eq+4)):
    print("pos",p,"R,K",f([q(2,"R",p),q(2,"K",p)]),"QN,QV",f([q(7,"QN",p),q(7,"QV",p)]),"F",f([q(7,f"F{k}",p) for k in range(-1,13)]),"Y",round(q(8,"Y",p),2),"top",z["top5_ids"][ti,p,:3].tolist())
print("QS@eq",f([q(7,f"QS{p}",eq) for p in range(13)]))
