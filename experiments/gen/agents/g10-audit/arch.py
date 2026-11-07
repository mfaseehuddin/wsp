"""GenGPT custom architecture: a transformer whose attention heads have extra
structural knobs (all numbers live in params).  Per head hp = (A, shift, R, local, ssum):
  score = q.k - A*(t-s-shift-dyn)^2 + R*s - local*big*|seg_t-seg_s|   (causal)
  A>0: relative-position head (shift = fixed lag, dyn = lag read from a channel via dvec)
  R : recency (+) or earliest (-) preference;  local=1: only keys of the same turn (turn = count of <user>)
  ssum=1: the head is a counting head (uniform sum over same-turn past tokens, no softmax).
Then a ReLU MLP per layer.  Output: 12 'out' channels -> 12-bit token code -> 4096 logits."""
import torch
import torch.nn.functional as F


def _attn(p, i, x, seg):
    pre = "l%d." % i
    T = x.shape[1]
    q = torch.einsum("btd,hde->bhte", x, p[pre + "wq"])
    k = torch.einsum("btd,hde->bhte", x, p[pre + "wk"])
    v = torch.einsum("btd,hde->bhte", x, p[pre + "wv"])
    hp = p[pre + "hp"]
    A, sh, R, loc, ssum = [hp[:, j][None, :, None, None] for j in range(5)]
    dq = torch.einsum("btd,hd->bht", x, p[pre + "dvec"])[..., None]
    idx = torch.arange(T, device=x.device).to(x.dtype)
    tt, ss = idx[:, None], idx[None, :]
    rel = tt - ss - (sh + dq)
    dseg = (seg[:, :, None] - seg[:, None, :]).abs()[:, None]
    causal = torch.tril(torch.ones(T, T, dtype=x.dtype, device=x.device))
    big = p["big"]
    score = torch.einsum("bhte,bhse->bhts", q, k) - A * rel * rel + R * ss
    score = score - loc * big * dseg + (causal - 1) * big
    soft = torch.softmax(score, dim=-1)
    sumw = causal * F.relu(1 - dseg)
    w = ssum * sumw + (1 - ssum) * soft
    o = torch.einsum("bhts,bhse->bhte", w, v)
    return torch.einsum("bhte,hed->btd", o, p[pre + "wo"])


def forward(params, tokens):
    p = params
    x = p["embed"][tokens] @ p["embproj"]
    idx = torch.arange(tokens.shape[1], device=x.device).to(x.dtype)
    x = x + idx[None, :, None] * p["pos_dir"]
    seg = torch.cumsum(x @ p["seg_in"], dim=1)
    for i in range(6):
        x = x + _attn(p, i, x, seg)
        x = x + F.relu(x @ p["l%d.w_in" % i] + p["l%d.b_in" % i]) @ p["l%d.w_out" % i]
    return (x @ p["out_proj"]) @ p["code_table"]


def trace(params, tokens):
    p = params
    x = p["embed"][tokens] @ p["embproj"]
    idx = torch.arange(tokens.shape[1], device=x.device).to(x.dtype)
    x = x + idx[None, :, None] * p["pos_dir"]
    seg = torch.cumsum(x @ p["seg_in"], dim=1)
    out = {}
    for i in range(6):
        x = x + _attn(p, i, x, seg)
        x = x + F.relu(x @ p["l%d.w_in" % i] + p["l%d.b_in" % i]) @ p["l%d.w_out" % i]
        r = torch.round(x[0, -1])
        for c in range(12):
            out["L%d_c%02d" % (i, c)] = r[c * 40 : c * 40 + 40]
    return out
