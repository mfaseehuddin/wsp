"""The fixed decoder-only transformer that every agent must fill with weights.

Agents choose a `Config` and supply every parameter tensor. They cannot change
this forward function, so the model is an ordinary transformer, not a
disguised lookup table.

Forward (pre-norm residual stream, all float64):
    x = embed[tok] (+ pos[t] if cfg.pos == "learned")
    per layer:  x = x + Attn(LN1(x));  x = x + MLP(LN2(x))   (MLP skipped if d_mlp == 0)
    logits = LNf(x) @ U (+ b_unembed),  U = embed.T if tie_embeddings else unembed
LN is the identity unless cfg.norm is True (LayerNorm, eps 1e-5, with gain and bias).
Attention: causal softmax(q.k / sqrt(d_head)) per head, then summed through wo.
MLP: ReLU(x @ w_in + b_in) @ w_out + b_out.
Biases (bq, bk, bv, bo, b_in, b_out, b_unembed) exist only if cfg.bias is True.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from . import tokenizer  # GenGPT vocabulary (4,096 tokens)

DTYPE = torch.float64


@dataclass(frozen=True)
class Config:
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int
    d_mlp: int = 0
    pos: str = "learned"  # "learned" | "none"
    tie_embeddings: bool = False
    norm: bool = False
    bias: bool = True

    def validate(self) -> None:
        if (
            min(self.d_model, self.n_layers, self.n_heads, self.d_head) < 1
            or self.d_mlp < 0
        ):
            raise ValueError("sizes must be positive (d_mlp may be 0)")
        if self.pos not in ("learned", "none"):
            raise ValueError("pos must be 'learned' or 'none'")
        if self.n_layers > 8 or self.d_model > 2048:
            raise ValueError("n_layers <= 8 and d_model <= 2048")

    def to_dict(self) -> dict:
        return asdict(self)


def param_shapes(cfg: Config) -> dict[str, tuple[int, ...]]:
    V, L, d, h, e, m = (
        tokenizer.vocab_size(),
        tokenizer.MAX_LEN,
        cfg.d_model,
        cfg.n_heads,
        cfg.d_head,
        cfg.d_mlp,
    )
    s: dict[str, tuple[int, ...]] = {"embed": (V, d)}
    if cfg.pos == "learned":
        s["pos"] = (L, d)
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        if cfg.norm:
            s[p + "ln1.g"], s[p + "ln1.b"] = (d,), (d,)
        for w in ("wq", "wk", "wv"):
            s[p + w] = (h, d, e)
        s[p + "wo"] = (h, e, d)
        if cfg.bias:
            for b in ("bq", "bk", "bv"):
                s[p + b] = (h, e)
            s[p + "bo"] = (d,)
        if m:
            if cfg.norm:
                s[p + "ln2.g"], s[p + "ln2.b"] = (d,), (d,)
            s[p + "w_in"], s[p + "w_out"] = (d, m), (m, d)
            if cfg.bias:
                s[p + "b_in"], s[p + "b_out"] = (m,), (d,)
    if cfg.norm:
        s["lnf.g"], s["lnf.b"] = (d,), (d,)
    if not cfg.tie_embeddings:
        s["unembed"] = (d, V)
    if cfg.bias:
        s["b_unembed"] = (V,)
    return s


def check_params(
    cfg: Config, params: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    cfg.validate()
    want = param_shapes(cfg)
    if set(params) != set(want):
        missing, extra = (
            sorted(set(want) - set(params)),
            sorted(set(params) - set(want)),
        )
        raise ValueError(f"parameter names mismatch; missing={missing} extra={extra}")
    out = {}
    for k, shape in want.items():
        t = torch.as_tensor(params[k]).to(DTYPE)
        if tuple(t.shape) != shape:
            raise ValueError(f"{k}: shape {tuple(t.shape)} != {shape}")
        if not torch.isfinite(t).all():
            raise ValueError(f"{k}: non-finite values")
        out[k] = t
    return out


def n_params(params: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() for t in params.values())


def _ln(x, params, name, cfg):
    if not cfg.norm:
        return x
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-5) * params[name + ".g"] + params[name + ".b"]


@torch.no_grad()
def forward(
    cfg: Config,
    params: dict[str, torch.Tensor],
    tokens: torch.Tensor,
    trace: bool = False,
):
    """tokens: (B, T) int64, right-padded. Returns logits (B, T, V) [, trace dict]."""
    T = tokens.shape[1]
    x = params["embed"][tokens]
    if cfg.pos == "learned":
        x = x + params["pos"][:T]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    tr: dict = {"attn": [], "resid": [x.clone()]}
    for i in range(cfg.n_layers):
        p = f"layers.{i}."
        a = _ln(x, params, p + "ln1", cfg)
        q = torch.einsum("btd,hde->bhte", a, params[p + "wq"])
        k = torch.einsum("btd,hde->bhte", a, params[p + "wk"])
        v = torch.einsum("btd,hde->bhte", a, params[p + "wv"])
        if cfg.bias:
            q = q + params[p + "bq"][None, :, None, :]
            k = k + params[p + "bk"][None, :, None, :]
            v = v + params[p + "bv"][None, :, None, :]
        scores = q @ k.transpose(-1, -2) / math.sqrt(cfg.d_head)
        att = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        z = att @ v
        x = x + torch.einsum("bhte,hed->btd", z, params[p + "wo"])
        if cfg.bias:
            x = x + params[p + "bo"]
        if trace:
            tr["attn"].append(att.clone())
        if cfg.d_mlp:
            mm = _ln(x, params, p + "ln2", cfg)
            hdn = torch.relu(
                mm @ params[p + "w_in"] + (params[p + "b_in"] if cfg.bias else 0)
            )
            x = x + hdn @ params[p + "w_out"] + (params[p + "b_out"] if cfg.bias else 0)
        if trace:
            tr["resid"].append(x.clone())
    f = _ln(x, params, "lnf", cfg)
    U = params["embed"].T if cfg.tie_embeddings else params["unembed"]
    logits = f @ U + (params["b_unembed"] if cfg.bias else 0)
    return (logits, tr) if trace else logits


def unembed_matrix(cfg: Config, params: dict[str, torch.Tensor]) -> torch.Tensor:
    return params["embed"].T if cfg.tie_embeddings else params["unembed"]
