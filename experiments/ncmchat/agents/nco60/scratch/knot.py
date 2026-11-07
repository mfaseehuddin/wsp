import importlib.util, torch
D="experiments/ncmchat/agents/nco60"
spec=importlib.util.spec_from_file_location("w",f"{D}/weights.py"); W=importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
cfg,P,_=W.build()
f=W.f
for N in [1,2,5,50,300,700,1000,1090]:
    x=torch.zeros(1,cfg.d_model); x[0,f["ONE"]]=1; x[0,f["CS"]]=W.KS/(W.KS+N-1)
    y=torch.relu(x@P["L1.w1"]+P["L1.b1"])@P["L1.w2"]+P["L1.b2"]
    md=y[0,f["MOD0"]:f["MOD0"]+32]; print(N, N%32, md.argmax().item(), round(md.max().item(),3), round((md.abs().sum()-md.max()).item(),3))
for N in [0,1,5,30,64,100]:
    x=torch.zeros(1,cfg.d_model); x[0,f["ONE"]]=1; x[0,f["CT"]]=W.KT/(W.KT+N)
    y=torch.relu(x@P["L1.w1"]+P["L1.b1"])@P["L1.w2"]+P["L1.b2"]
    print("T",N,(y[0,f["ET0"]:f["ET0"]+6]-W.enc(N).float()).abs().max().item())
