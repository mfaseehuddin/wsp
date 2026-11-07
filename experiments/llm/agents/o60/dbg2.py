import numpy as np, torch, sys, re
z=np.load("experiments/llm/agents/o60/reads/"+sys.argv[1]+".npz"); d=torch.load("experiments/llm/agents/o60/dbg_basis.pt"); lab=d["labels"]
ti=int(sys.argv[2]); layers=[int(x) for x in sys.argv[3].split(",")]; pat=sys.argv[4] if len(sys.argv)>4 else ""
ids=z["input_ids"][ti]; T=z["lengths"][ti]
pl=[i for i in range(T) if ids[i]==10][-1]; eq=[i for i in range(T) if ids[i]==28][-1]
for l in layers:
    A=z[f"mlp_L{l}"][ti].astype(np.float64)
    # combine neuron pairs: print per label (first neuron of each step pair is relu(y-lo))
    for pos,tag in [(pl,"+"),(eq,"=")]+[(eq+k,f"a{k}") for k in range(1,10)]:
        out=[]
        for n,lb in enumerate(lab[l]):
            if pat and not re.search(pat,lb): continue
            v=A[pos,n]
            if abs(v)>0.02: out.append(f"{n}:{lb.split('|')[0]}={v:.2f}")
        print(f"L{l} {tag}@{pos}:", " ".join(out)[:900])
