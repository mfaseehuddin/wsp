"""NCM chat model: a multi-layer Next-Concept Model over a cached conversation (Track D, chat round).

Generation scheme from Kumbong et al., "Next-Concept Modeling" (arXiv 2610.14162), extended to multi-turn chat:

  * The conversation so far (Qwen chat format, Qwen2.5 tokenizer) is a CAUSAL prefix. Its keys/values are
    computed once per token and kept in a KV cache across levels and across turns.
  * A reply of T = 64 positions is built in K levels of discrete codes, coarse to fine (n_1 < ... < n_K = 64).
    Each level is ONE forward pass over n_l query rows that attend to the whole cached prefix and to each other
    (bidirectionally within the level). All codes of a level are predicted in parallel.
  * The codes' vectors are broadcast to 64 positions and ADDED to a latent z [64, D]; the next level's query rows
    are built from z mean-pooled to its length. After the last level a per-position decoder maps z to tokens.
    The reply ends at the first <|im_end|>; reply + <|im_end|> + "\\n" then join the cached prefix.

Positions: prefix token i has RoPE position i. All query rows of a level sit at RoPE position len(prefix)
(just after the conversation), distinguished by lvl_pos[j] (learned, added to the input). So a query head with a
RoPE-structured q/k measures DISTANCE BACK from the end of the conversation.

Rows:
  prefix row  i : tok_emb[token_i]
  level row j, l: down(z, n_l)[j] @ w_in + lvl_emb[l] + lvl_pos[j]
Layer (pre-RMSNorm if cfg.norm, gain only):  x += Attn(N1 x);  x += MLP(N2 x)
  Attn: per head h, q = x Wq_h + bq_h, k = x Wk_h + bk_h, v = x Wv_h + bv_h; RoPE(theta) on q, k
        (rotate-half: dim i pairs with i + d_head/2, angle = pos * theta**(-2i/d_head));
        softmax(q.k / sqrt(d_head)) ; output summed through Wo_h, + bo.
  MLP: ReLU(x W1 + b1) W2 + b2   (skipped if d_mlp == 0)
codes_l = argmax(h_level @ head_l + head_b_l);  z += up(code_l[codes_l])
dec(z) = ReLU(z dec_w1 + dec_b1) dec_w2 + dec_b2 if dec_hidden else z dec_w2 + dec_b2
float32.

Round 2 ("deep") options, all off by default:
  qwen_embed=True      prefix rows start from Qwen2.5-0.5B's embedding (896-d, read from disk, not in params)
  qwen_layers=(i, ...) run these VERBATIM Qwen decoder layers, in this order, over the prefix (causal, KV-cached);
                       see qwen.py. They only see the conversation, never the level rows.
  qwen_final_norm      apply Qwen's final RMSNorm after them
  then prefix rows = qwen_residual @ pre_proj [896, d_model] and continue through the agent's own layers.
  level_meanings       one stated meaning per level (required by the deep round, K >= 4).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

DTYPE = torch.float32
T_MAX = 64


@dataclass(frozen=True)
class Config:
    vocab: int
    D: int
    levels: tuple[int, ...]
    codebooks: tuple[int, ...]
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int
    d_mlp: int = 0
    dec_hidden: int = 0
    rope_theta: float = 10000.0
    norm: bool = False
    qwen_embed: bool = False
    qwen_layers: tuple[int, ...] = ()
    qwen_final_norm: bool = False
    level_meanings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def shapes(cfg: Config) -> dict[str, tuple[int, ...]]:
    K, dm, H, dh = len(cfg.levels), cfg.d_model, cfg.n_heads, cfg.d_head
    s = {"lvl_pos": (T_MAX, dm), "lvl_emb": (K, dm), "w_in": (cfg.D, dm)}
    s |= {"pre_proj": (896, dm)} if cfg.qwen_embed else {"tok_emb": (cfg.vocab, dm)}
    for i in range(cfg.n_layers):
        p = f"L{i}."
        s |= {p + "wq": (H, dm, dh), p + "wk": (H, dm, dh), p + "wv": (H, dm, dh), p + "wo": (H, dh, dm),
              p + "bq": (H, dh), p + "bk": (H, dh), p + "bv": (H, dh), p + "bo": (dm,)}
        if cfg.d_mlp:
            s |= {p + "w1": (dm, cfg.d_mlp), p + "b1": (cfg.d_mlp,), p + "w2": (cfg.d_mlp, dm), p + "b2": (dm,)}
        if cfg.norm:
            s |= {p + "n1": (dm,), p + "n2": (dm,)}
    for l, V in enumerate(cfg.codebooks):
        s |= {f"head{l}": (dm, V), f"head_b{l}": (V,), f"code{l}": (V, cfg.D)}
    if cfg.dec_hidden:
        s |= {"dec_w1": (cfg.D, cfg.dec_hidden), "dec_b1": (cfg.dec_hidden,), "dec_w2": (cfg.dec_hidden, cfg.vocab)}
    else:
        s |= {"dec_w2": (cfg.D, cfg.vocab)}
    s["dec_b2"] = (cfg.vocab,)
    return s


def validate(cfg: Config, P: dict) -> None:
    assert len(cfg.levels) == len(cfg.codebooks) >= 1
    assert list(cfg.levels) == sorted(set(cfg.levels)) and cfg.levels[-1] == T_MAX, f"levels must increase to {T_MAX}"
    assert cfg.d_head % 2 == 0
    assert cfg.qwen_embed or not cfg.qwen_layers, "qwen_layers need qwen_embed=True"
    assert all(0 <= i < 24 for i in cfg.qwen_layers), "Qwen2.5-0.5B has layers 0..23"
    assert not cfg.level_meanings or len(cfg.level_meanings) == len(cfg.levels), "one meaning per level"
    want = shapes(cfg)
    missing, extra = set(want) - set(P), set(P) - set(want)
    assert not missing and not extra, f"missing {sorted(missing)[:5]} extra {sorted(extra)[:5]}"
    for k, shp in want.items():
        assert tuple(P[k].shape) == shp, f"{k}: {tuple(P[k].shape)} != {shp}"


def _rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """x [H, n, dh], pos [n]."""
    dh = x.shape[-1]
    inv = theta ** (-torch.arange(0, dh // 2, dtype=torch.float64) * 2 / dh)
    ang = (pos.to(torch.float64)[:, None] * inv[None]).to(x.dtype)
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    x1, x2 = x[..., : dh // 2], x[..., dh // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Cache:
    """Per-layer RoPE'd keys and values of every prefix token seen so far."""

    def __init__(self, cfg: Config):
        H, dh = cfg.n_heads, cfg.d_head
        self.k = [torch.zeros(H, 0, dh, dtype=DTYPE) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(H, 0, dh, dtype=DTYPE) for _ in range(cfg.n_layers)]
        self.ids: list[int] = []
        self.qk = [torch.zeros(2, 0, 64) for _ in cfg.qwen_layers]  # verbatim Qwen layers' cache
        self.qv = [torch.zeros(2, 0, 64) for _ in cfg.qwen_layers]

    def __len__(self) -> int:
        return len(self.ids)


def _layer(cfg, P, i, x, pos, ck, cv, causal_offset: int | None, trace):
    """x [n, dm] new rows at positions pos; ck/cv cached [H, m, dh]. causal_offset: if set, new rows are prefix
    rows (row r may see cached rows and new rows <= r); if None, they are level rows (see everything)."""
    p = f"L{i}."
    h = _rms(x, P[p + "n1"]) if cfg.norm else x
    q = rope(torch.einsum("td,hde->hte", h, P[p + "wq"]) + P[p + "bq"][:, None], pos, cfg.rope_theta)
    k = rope(torch.einsum("td,hde->hte", h, P[p + "wk"]) + P[p + "bk"][:, None], pos, cfg.rope_theta)
    v = torch.einsum("td,hde->hte", h, P[p + "wv"]) + P[p + "bv"][:, None]
    K, V = torch.cat([ck, k], 1), torch.cat([cv, v], 1)
    s = q @ K.transpose(1, 2) / math.sqrt(cfg.d_head)
    if causal_offset is not None:
        n, m = x.shape[0], ck.shape[1]
        mask = torch.arange(m + n)[None, :] > (torch.arange(n)[:, None] + m)
        s = s.masked_fill(mask, float("-inf"))
    a = torch.softmax(s, -1)
    if trace is not None:
        trace.append(a)
    x = x + torch.einsum("hte,hed->td", a @ V, P[p + "wo"]) + P[p + "bo"]
    if cfg.d_mlp:
        h = _rms(x, P[p + "n2"]) if cfg.norm else x
        x = x + torch.relu(h @ P[p + "w1"] + P[p + "b1"]) @ P[p + "w2"] + P[p + "b2"]
    return x, k, v


@torch.no_grad()
def extend(cfg: Config, P: dict, cache: Cache, ids: list[int]) -> None:
    """Append prefix tokens to the cache (causal)."""
    if not ids:
        return
    m = len(cache)
    pos = torch.arange(m, m + len(ids))
    if cfg.qwen_embed:
        from . import qwen

        x = qwen.embed()[torch.tensor(ids, dtype=torch.long)]
        for j, li in enumerate(cfg.qwen_layers):
            x, k, v = qwen.block(li, x, pos, cache.qk[j], cache.qv[j])
            cache.qk[j], cache.qv[j] = torch.cat([cache.qk[j], k], 1), torch.cat([cache.qv[j], v], 1)
        if cfg.qwen_final_norm:
            x = qwen.rms(x, qwen.final_norm())
        x = x @ P["pre_proj"]
    else:
        x = P["tok_emb"][torch.tensor(ids, dtype=torch.long)]
    for i in range(cfg.n_layers):
        x, k, v = _layer(cfg, P, i, x, pos, cache.k[i], cache.v[i], m, None)
        cache.k[i] = torch.cat([cache.k[i], k], 1)
        cache.v[i] = torch.cat([cache.v[i], v], 1)
    cache.ids += list(ids)


def _map(n):
    return torch.arange(T_MAX) * n // T_MAX


def down(z, n):
    m = _map(n)
    out = torch.zeros(n, z.shape[1], dtype=DTYPE).index_add_(0, m, z)
    return out / torch.bincount(m, minlength=n).clamp(min=1)[:, None].to(DTYPE)


def up(c, n):
    return c[_map(n)]


def decode_latent(cfg, P, z):
    h = torch.relu(z @ P["dec_w1"] + P["dec_b1"]) if cfg.dec_hidden else z
    return h @ P["dec_w2"] + P["dec_b2"]


@torch.no_grad()
def reply(cfg: Config, P: dict, cache: Cache, trace: bool = False) -> dict:
    """Generate one reply from the cached conversation (which must end with the assistant header)."""
    z = torch.zeros(T_MAX, cfg.D, dtype=DTYPE)
    end = len(cache)
    out = {"codes": [], "levels": []}
    for l, n in enumerate(cfg.levels):
        x = down(z, n) @ P["w_in"] + P["lvl_emb"][l] + P["lvl_pos"][:n]
        pos = torch.full((n,), end)
        att = [] if trace else None
        for i in range(cfg.n_layers):
            x, _, _ = _layer(cfg, P, i, x, pos, cache.k[i], cache.v[i], None, att)
        codes = (x @ P[f"head{l}"] + P[f"head_b{l}"]).argmax(-1)
        z = z + up(P[f"code{l}"][codes], n)
        out["codes"].append(codes.tolist())
        if trace:
            out["levels"].append({"z": z.clone(), "attn": att})
    out["tokens"] = decode_latent(cfg, P, z).argmax(-1).tolist()
    return out
