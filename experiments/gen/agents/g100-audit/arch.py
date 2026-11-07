import math
import torch
import torch.nn.functional as F


def forward(params, tokens):
    # GenGPT-attn: a causal transformer whose token code (12 bits of the id) and position
    # code (sin/cos at 10 octave frequencies) are fixed functions, everything else is in params.
    dev = tokens.device
    dt = params["wt"].dtype
    B, T = tokens.shape
    pw = 2.0 ** torch.arange(12, device=dev)
    bits = torch.div(tokens.unsqueeze(-1).to(dt), pw.to(dt), rounding_mode="floor") % 2
    code = bits * 2 - 1
    t = torch.arange(T, device=dev).to(dt)
    fr = (2.0 ** torch.arange(10, device=dev)).to(dt) * (2 * math.pi / 1024)
    ang = t.unsqueeze(-1) * fr
    pos = torch.cat([torch.cos(ang), torch.sin(ang), (t / 1024).unsqueeze(-1)], -1)
    x = params["tab"][tokens] @ params["wt"] + code @ params["wc"] + pos @ params["wp"]
    mask = torch.tril(torch.ones(T, T, device=dev, dtype=dt)) - 1
    for l in range(6):
        nh = params[f"L{l}.nh"].shape[0]
        dx = 0
        for i in range(nh):
            p = f"L{l}.h{i}."
            q = x @ params[p + "wq"] + params[p + "bq"]
            k = x @ params[p + "wk"]
            v = x @ params[p + "wv"]
            a = torch.softmax(q @ k.transpose(-1, -2) + mask * 1e9, -1)
            dx = dx + (a @ v) @ params[p + "wo"]
        x = x + dx
        h = F.relu(x @ params[f"L{l}.w_in"] + params[f"L{l}.b_in"])
        x = x + h @ params[f"L{l}.w_out"]
    ids = torch.arange(4096, device=dev).to(dt).unsqueeze(-1)
    cb = (torch.div(ids, pw.to(dt), rounding_mode="floor") % 2) * 2 - 1
    return (x @ params["wu"]) @ cb.transpose(0, 1)
