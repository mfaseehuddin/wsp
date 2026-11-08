"""A standard decoder-only transformer whose every weight an agent writes (wsp-pure, family "transformer").

    x_t = tok_emb[token_t] (+ pos_emb[t] if pos == "learned")
    per layer i:  x += Attn_i(N1 x);  x += MLP_i(N2 x)
    logits = Nf(x) @ U,   U = tok_emb^T if tied else unembed [d_model, vocab]
N = RMSNorm (gain) | LayerNorm (gain, bias) | identity, per cfg.norm.  eps 1e-5.
Attn: causal; n_heads query heads, n_kv_heads key/value heads (query head h uses kv head h * n_kv // n_heads);
      q = x Wq + bq, k = x Wk + bk, v = x Wv + bv (per head, [d_model, d_head] each); RoPE(theta) on q, k if
      pos == "rope" (rotate-half: dim i pairs with i + d_head/2); softmax(q.k / sqrt(d_head)); out = sum_h o_h Wo_h + bo.
MLP: act(x W1 + b1) W2 + b2 with act in relu | gelu | silu | tanh;  "swiglu": (silu(x Wg) * (x W1 + b1)) W2 + b2.
Generation: greedy, KV-cached, one token at a time. float32.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

DTYPE = torch.float32


@dataclass(frozen=True)
class Config:
    vocab: int
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int
    n_kv_heads: int = 0  # 0 = same as n_heads
    d_mlp: int = 0  # 0 = no MLP
    act: str = "relu"  # relu | gelu | silu | tanh | swiglu
    norm: str = "rms"  # rms | layer | none
    pos: str = "rope"  # rope | learned | none
    rope_theta: float = 10000.0
    max_ctx: int = 8192
    tied: bool = True

    @property
    def kv(self) -> int:
        return self.n_kv_heads or self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


def shapes(c: Config) -> dict[str, tuple[int, ...]]:
    s = {"tok_emb": (c.vocab, c.d_model)}
    if c.pos == "learned":
        s["pos_emb"] = (c.max_ctx, c.d_model)
    for i in range(c.n_layers):
        p = f"L{i}."
        s |= {p + "wq": (c.n_heads, c.d_model, c.d_head), p + "bq": (c.n_heads, c.d_head),
              p + "wk": (c.kv, c.d_model, c.d_head), p + "bk": (c.kv, c.d_head),
              p + "wv": (c.kv, c.d_model, c.d_head), p + "bv": (c.kv, c.d_head),
              p + "wo": (c.n_heads, c.d_head, c.d_model), p + "bo": (c.d_model,)}
        if c.d_mlp:
            s |= {p + "w1": (c.d_model, c.d_mlp), p + "b1": (c.d_mlp,), p + "w2": (c.d_mlp, c.d_model), p + "b2": (c.d_model,)}
            if c.act == "swiglu":
                s[p + "wg"] = (c.d_model, c.d_mlp)
        if c.norm != "none":
            s |= {p + "n1_g": (c.d_model,), p + "n2_g": (c.d_model,)}
            if c.norm == "layer":
                s |= {p + "n1_b": (c.d_model,), p + "n2_b": (c.d_model,)}
    if c.norm != "none":
        s["nf_g"] = (c.d_model,)
        if c.norm == "layer":
            s["nf_b"] = (c.d_model,)
    if not c.tied:
        s["unembed"] = (c.d_model, c.vocab)
    return s


def validate(c: Config, P: dict) -> None:
    assert c.act in ("relu", "gelu", "silu", "tanh", "swiglu") and c.norm in ("rms", "layer", "none")
    assert c.pos in ("rope", "learned", "none") and c.d_head % 2 == 0 and c.n_heads % c.kv == 0
    want = shapes(c)
    missing, extra = set(want) - set(P), set(P) - set(want)
    assert not missing and not extra, f"missing {sorted(missing)[:5]} extra {sorted(extra)[:5]}"
    for k, shp in want.items():
        assert tuple(P[k].shape) == shp, f"{k}: {tuple(P[k].shape)} != {shp}"


def _norm(c, P, x, name):
    if c.norm == "none":
        return x
    if c.norm == "rms":
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * P[name + "_g"]
    mu, var = x.mean(-1, keepdim=True), x.var(-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + 1e-5) * P[name + "_g"] + P[name + "_b"]


def _rope(x, pos, theta):
    dh = x.shape[-1]
    inv = theta ** (-torch.arange(0, dh // 2, dtype=torch.float64) * 2 / dh)
    ang = (pos.double()[:, None] * inv[None]).to(x.dtype)
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    return x * cos + torch.cat([-x[..., dh // 2 :], x[..., : dh // 2]], -1) * sin


_ACT = {"relu": torch.relu, "gelu": torch.nn.functional.gelu, "silu": torch.nn.functional.silu, "tanh": torch.tanh}


class Cache:
    def __init__(self, c: Config):
        self.k = [torch.zeros(c.kv, 0, c.d_head) for _ in range(c.n_layers)]
        self.v = [torch.zeros(c.kv, 0, c.d_head) for _ in range(c.n_layers)]
        self.n = 0


@torch.no_grad()
def forward(c: Config, P: dict, cache: Cache, ids: list[int], trace: list | None = None) -> torch.Tensor:
    """Append tokens to the cache; return logits [len(ids), vocab]."""
    m, n = cache.n, len(ids)
    assert m + n <= c.max_ctx, "context too long"
    pos = torch.arange(m, m + n)
    x = P["tok_emb"][torch.tensor(ids, dtype=torch.long)]
    if c.pos == "learned":
        x = x + P["pos_emb"][pos]
    rep = c.n_heads // c.kv
    for i in range(c.n_layers):
        p = f"L{i}."
        h = _norm(c, P, x, p + "n1")
        q = torch.einsum("td,hde->hte", h, P[p + "wq"]) + P[p + "bq"][:, None]
        k = torch.einsum("td,hde->hte", h, P[p + "wk"]) + P[p + "bk"][:, None]
        v = torch.einsum("td,hde->hte", h, P[p + "wv"]) + P[p + "bv"][:, None]
        if c.pos == "rope":
            q, k = _rope(q, pos, c.rope_theta), _rope(k, pos, c.rope_theta)
        cache.k[i], cache.v[i] = torch.cat([cache.k[i], k], 1), torch.cat([cache.v[i], v], 1)
        K, V = cache.k[i].repeat_interleave(rep, 0), cache.v[i].repeat_interleave(rep, 0)
        s = q @ K.transpose(1, 2) / math.sqrt(c.d_head)
        s = s.masked_fill(torch.arange(m + n)[None, :] > (torch.arange(n)[:, None] + m), float("-inf"))
        a = torch.softmax(s, -1)
        if trace is not None:
            trace.append(a)
        x = x + torch.einsum("hte,hed->td", a @ V, P[p + "wo"]) + P[p + "bo"]
        if c.d_mlp:
            h = _norm(c, P, x, p + "n2")
            u = h @ P[p + "w1"] + P[p + "b1"]
            u = torch.nn.functional.silu(h @ P[p + "wg"]) * u if c.act == "swiglu" else _ACT[c.act](u)
            x = x + u @ P[p + "w2"] + P[p + "b2"]
    cache.n += n
    x = _norm(c, P, x, "nf")
    return x @ (P["tok_emb"].T if c.tied else P["unembed"])


@torch.no_grad()
def generate(c: Config, P: dict, cache: Cache, prompt_ids: list[int], stop: int, max_new: int = 128,
             trace: bool = False) -> dict:
    logits = forward(c, P, cache, prompt_ids)[-1]
    out, steps = [], []
    for _ in range(max_new):
        t = int(logits.argmax())
        if trace:
            top = logits.topk(5)
            steps.append({"top5": top.indices.tolist(), "logit": [round(v, 2) for v in top.values.tolist()]})
        if t == stop:
            break
        out.append(t)
        logits = forward(c, P, cache, [t])[-1]
    return {"tokens": out, "stopped": len(out) < max_new, "steps": steps}
