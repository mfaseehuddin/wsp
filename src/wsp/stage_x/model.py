"""Standard decoder with matching full-sequence and KV-cached inference."""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    d_head: int = 8
    d_mlp: int = 128
    max_len: int = 16384
    pos: str = "sinusoidal"
    norm: bool = False
    bias: bool = True
    tie_embeddings: bool = False

    def validate(self):
        for key, lo, hi in (
            ("d_model", 1, 1024),
            ("n_layers", 1, 12),
            ("n_heads", 1, 16),
            ("d_head", 1, 128),
            ("d_mlp", 0, 4096),
            ("max_len", 1, 16384),
        ):
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"invalid {key}: {value}")
        if self.pos not in {"none", "learned", "sinusoidal"}:
            raise ValueError("invalid positions")
        cache_bytes = 2 * self.n_layers * self.n_heads * self.d_head * self.max_len * 4
        if cache_bytes > 512 * 1024 * 1024:
            raise ValueError("float32 KV cache would exceed 512 MiB at full context")
        if any(
            type(getattr(self, k)) is not bool
            for k in ("norm", "bias", "tie_embeddings")
        ):
            raise ValueError("flags must be booleans")

    def to_dict(self):
        return asdict(self)


def param_shapes(cfg, vocab_size):
    cfg.validate()
    d, h, e, m = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_mlp
    shapes = {"embed": (vocab_size, d)}
    if cfg.pos == "learned":
        shapes["pos"] = (cfg.max_len, d)
    for i in range(cfg.n_layers):
        s = f"layers.{i}."
        for name in ("wq", "wk", "wv"):
            shapes[s + name] = (h, d, e)
        shapes[s + "wo"] = (h, e, d)
        if cfg.bias:
            for name in ("bq", "bk", "bv"):
                shapes[s + name] = (h, e)
            shapes[s + "bo"] = (d,)
        if cfg.norm:
            shapes[s + "ln1.g"] = shapes[s + "ln1.b"] = (d,)
        if m:
            shapes[s + "w_in"], shapes[s + "w_out"] = (d, m), (m, d)
            if cfg.bias:
                shapes[s + "b_in"], shapes[s + "b_out"] = (m,), (d,)
            if cfg.norm:
                shapes[s + "ln2.g"] = shapes[s + "ln2.b"] = (d,)
    if cfg.norm:
        shapes["lnf.g"] = shapes["lnf.b"] = (d,)
    if not cfg.tie_embeddings:
        shapes["unembed"] = (d, vocab_size)
    if cfg.bias:
        shapes["b_unembed"] = (vocab_size,)
    if sum(math.prod(s) for s in shapes.values()) > 20_000_000:
        raise ValueError("over 20M parameters")
    return shapes


def check_params(cfg, params, vocab_size):
    shapes = param_shapes(cfg, vocab_size)
    if set(shapes) != set(params):
        raise ValueError(
            f"parameter mismatch: missing={set(shapes) - set(params)}, extra={set(params) - set(shapes)}"
        )
    for name, shape in shapes.items():
        t = params[name]
        if (
            not isinstance(t, torch.Tensor)
            or t.dtype != torch.float32
            or tuple(t.shape) != shape
        ):
            raise ValueError(f"{name}: expected float32 {shape}")
        if not torch.isfinite(t).all() or (t.abs() > 60_000).any():
            raise ValueError(f"{name}: nonfinite or unsafe magnitude")
    return params


def _ln(x, p, prefix, enabled):
    if not enabled:
        return x
    return F.layer_norm(x, (x.shape[-1],), p[prefix + ".g"], p[prefix + ".b"], 1e-5)


@torch.no_grad()
def forward(cfg, p, tokens, cache=None):
    """tokens [B,N]. Cache is layer (K,V) pairs; returns logits and new cache.

    Chunked prefill uses explicit offset-aware causal masking. Single-token
    decode attends over all cached keys without allocating a T by T matrix.
    """
    offset = cache[0][0].shape[2] if cache else 0
    count = tokens.shape[1]
    if count < 1 or offset + count > cfg.max_len:
        raise ValueError("context full")
    x = p["embed"][tokens]
    pos = torch.arange(offset, offset + count, device=x.device)
    if cfg.pos == "learned":
        x = x + p["pos"][offset : offset + count]
    elif cfg.pos == "sinusoidal":
        freq = torch.exp(
            torch.arange(0, cfg.d_model, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / cfg.d_model)
        )
        angle = pos.to(x.dtype)[:, None] * freq
        pe = torch.zeros(count, cfg.d_model, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = angle.sin()
        pe[:, 1::2] = angle[:, : cfg.d_model // 2].cos()
        x = x + pe
    new_cache = []
    mask = torch.arange(offset + count, device=x.device)[None, :] <= pos[:, None]
    for i in range(cfg.n_layers):
        s = f"layers.{i}."
        a = _ln(x, p, s + "ln1", cfg.norm)
        q, k, v = [
            torch.einsum("btd,hde->bhte", a, p[s + name]) for name in ("wq", "wk", "wv")
        ]
        if cfg.bias:
            q, k, v = [
                z + p[s + name][None, :, None, :]
                for z, name in zip((q, k, v), ("bq", "bk", "bv"))
            ]
        if cache:
            k = torch.cat((cache[i][0], k), dim=2)
            v = torch.cat((cache[i][1], v), dim=2)
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + torch.einsum("bhte,hed->btd", z, p[s + "wo"])
        if cfg.bias:
            x = x + p[s + "bo"]
        if cfg.d_mlp:
            a = _ln(x, p, s + "ln2", cfg.norm)
            hidden = F.relu(a @ p[s + "w_in"] + (p[s + "b_in"] if cfg.bias else 0))
            x = x + hidden @ p[s + "w_out"] + (p[s + "b_out"] if cfg.bias else 0)
        new_cache.append((k, v))
    x = _ln(x, p, "lnf", cfg.norm)
    u = p["embed"].T if cfg.tie_embeddings else p["unembed"]
    return x @ u + (p["b_unembed"] if cfg.bias else 0), new_cache


@torch.no_grad()
def trace(cfg, p, tokens, top=3):
    """Debug view of one full-sequence pass (no cache), for the last position.

    Returns the logits plus, per layer, each head's top attended positions and the
    logit-lens top tokens of the residual stream after that layer. Mirrors
    `forward` exactly; tests check the logits agree.
    """
    count = tokens.shape[1]
    x = p["embed"][tokens]
    if cfg.pos == "learned":
        x = x + p["pos"][:count]
    elif cfg.pos == "sinusoidal":
        freq = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=x.dtype)
            * (-math.log(10000.0) / cfg.d_model)
        )
        angle = torch.arange(count, dtype=x.dtype)[:, None] * freq
        pe = torch.zeros(count, cfg.d_model, dtype=x.dtype)
        pe[:, 0::2] = angle.sin()
        pe[:, 1::2] = angle[:, : cfg.d_model // 2].cos()
        x = x + pe
    u = p["embed"].T if cfg.tie_embeddings else p["unembed"]
    ub = p["b_unembed"] if cfg.bias else 0
    causal = torch.ones(count, count, dtype=torch.bool).tril()
    layers = []
    for i in range(cfg.n_layers):
        s = f"layers.{i}."
        a = _ln(x, p, s + "ln1", cfg.norm)
        q, k, v = [
            torch.einsum("btd,hde->bhte", a, p[s + name]) for name in ("wq", "wk", "wv")
        ]
        if cfg.bias:
            q, k, v = [
                z + p[s + name][None, :, None, :]
                for z, name in zip((q, k, v), ("bq", "bk", "bv"))
            ]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.d_head)
        att = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        x = x + torch.einsum("bhte,hed->btd", att @ v, p[s + "wo"])
        if cfg.bias:
            x = x + p[s + "bo"]
        if cfg.d_mlp:
            a = _ln(x, p, s + "ln2", cfg.norm)
            hidden = F.relu(a @ p[s + "w_in"] + (p[s + "b_in"] if cfg.bias else 0))
            x = x + hidden @ p[s + "w_out"] + (p[s + "b_out"] if cfg.bias else 0)
        last = att[0, :, -1]
        w, j = last.topk(min(top, count), dim=-1)
        lens = (_ln(x, p, "lnf", cfg.norm)[0, -1] @ u + ub).topk(top)
        layers.append(
            {
                "heads": [
                    [(int(jj), round(float(ww), 4)) for ww, jj in zip(w[h], j[h])]
                    for h in range(cfg.n_heads)
                ],
                "lens": [(int(t), round(float(v_), 3)) for v_, t in zip(*lens)],
            }
        )
    logits = _ln(x, p, "lnf", cfg.norm) @ u + ub
    return logits, layers
