"""A minimal Next-Concept Model (NCM): fixed forward pass, agents supply every weight.

Follows the generation scheme of Kumbong et al., "Next-Concept Modeling" (arXiv 2610.14162): a reply of T
positions is represented by K levels of discrete codes, coarse to fine. Level l has n_l positions
(n_1 < ... < n_K = T) and its own codebook C_l [V_l, D]. Generation needs ONE forward pass per level; all
positions of a level are predicted in parallel.

    z = 0                                                   # [T, D] multiscale residual latent
    for l = 1..K:
        q   = down(z, n_l)                                  # [n_l, D] mean-pool of the latent so far
        rows= [ tok_emb[u] + pre_pos[i]       for each user token u_i ]          (prefix, raw tokens)
            + [ q_j @ w_in + lvl_emb[l] + lvl_pos[j]  for j < n_l ]              (queries of level l)
        h   = transformer(rows)                             # bidirectional attention over all rows
        c_l = argmax( h[level rows] @ head_l + head_b_l )   # [n_l] codes, in parallel
        z  += up(C_l[c_l], T)                               # nearest-neighbour broadcast to T positions
    reply token t = argmax( dec(z_t) )                      # per position; stop at the first <eos>

down: position j of level l averages the latent over the positions t with floor(t * n_l / T) == j.
up:   position t takes code floor(t * n_l / T).
transformer layer (pre-norm if cfg.norm else no norm):  x += Attn(LN1 x);  x += MLP(LN2 x)
  Attn: softmax(q.k / sqrt(d_head)) per head, no mask (one example at a time), summed through wo.
  MLP: ReLU(x @ w1 + b1) @ w2 + b2  (skipped if d_mlp == 0).
dec(z): ReLU(z @ dec_w1 + dec_b1) @ dec_w2 + dec_b2 if dec_hidden > 0, else z @ dec_w2 + dec_b2.
All float64.

Simplifications vs the paper (documented, deliberate): fixed nearest/mean up/down sampling instead of learned
convolutions; no VQ encoder (it only matters for training); a per-position MLP decoder instead of a
transformer decoder.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

DTYPE = torch.float64
T_MAX = 48  # reply positions (pad with <eos>)


@dataclass(frozen=True)
class Config:
    vocab: int
    D: int                       # latent width
    levels: tuple[int, ...]      # n_1 < ... < n_K, last == T_MAX
    codebooks: tuple[int, ...]   # V_l per level
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int
    d_mlp: int = 0
    dec_hidden: int = 0
    norm: bool = False
    max_prefix: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


def shapes(cfg: Config) -> dict[str, tuple[int, ...]]:
    K, dm, H, dh = len(cfg.levels), cfg.d_model, cfg.n_heads, cfg.d_head
    s = {
        "tok_emb": (cfg.vocab, dm), "pre_pos": (cfg.max_prefix, dm), "lvl_pos": (T_MAX, dm),
        "lvl_emb": (K, dm), "w_in": (cfg.D, dm),
    }
    for i in range(cfg.n_layers):
        p = f"L{i}."
        s |= {p + "wq": (H, dm, dh), p + "wk": (H, dm, dh), p + "wv": (H, dm, dh), p + "wo": (H, dh, dm),
              p + "bq": (H, dh), p + "bk": (H, dh), p + "bv": (H, dh), p + "bo": (dm,)}
        if cfg.d_mlp:
            s |= {p + "w1": (dm, cfg.d_mlp), p + "b1": (cfg.d_mlp,), p + "w2": (cfg.d_mlp, dm), p + "b2": (dm,)}
        if cfg.norm:
            s |= {p + "ln1_g": (dm,), p + "ln1_b": (dm,), p + "ln2_g": (dm,), p + "ln2_b": (dm,)}
    for l, V in enumerate(cfg.codebooks):
        s |= {f"head{l}": (dm, V), f"head_b{l}": (V,), f"code{l}": (V, cfg.D)}
    if cfg.dec_hidden:
        s |= {"dec_w1": (cfg.D, cfg.dec_hidden), "dec_b1": (cfg.dec_hidden,), "dec_w2": (cfg.dec_hidden, cfg.vocab)}
    else:
        s |= {"dec_w2": (cfg.D, cfg.vocab)}
    s["dec_b2"] = (cfg.vocab,)
    return s


def validate(cfg: Config, params: dict) -> None:
    assert len(cfg.levels) == len(cfg.codebooks) >= 1, "one codebook size per level"
    assert list(cfg.levels) == sorted(set(cfg.levels)) and cfg.levels[-1] == T_MAX, f"levels must increase to {T_MAX}"
    want = shapes(cfg)
    missing, extra = set(want) - set(params), set(params) - set(want)
    assert not missing and not extra, f"missing {sorted(missing)[:5]} extra {sorted(extra)[:5]}"
    for k, shp in want.items():
        assert tuple(params[k].shape) == shp, f"{k}: {tuple(params[k].shape)} != {shp}"


def _ln(x, g, b):
    mu, var = x.mean(-1, keepdim=True), x.var(-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + 1e-5) * g + b


def transformer(cfg: Config, P: dict, x: torch.Tensor, trace: list | None = None) -> torch.Tensor:
    for i in range(cfg.n_layers):
        p = f"L{i}."
        h = _ln(x, P[p + "ln1_g"], P[p + "ln1_b"]) if cfg.norm else x
        q = torch.einsum("td,hde->hte", h, P[p + "wq"]) + P[p + "bq"][:, None]
        k = torch.einsum("td,hde->hte", h, P[p + "wk"]) + P[p + "bk"][:, None]
        v = torch.einsum("td,hde->hte", h, P[p + "wv"]) + P[p + "bv"][:, None]
        a = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(cfg.d_head), -1)
        x = x + torch.einsum("hte,hed->td", a @ v, P[p + "wo"]) + P[p + "bo"]
        if trace is not None:
            trace.append(a)
        if cfg.d_mlp:
            h = _ln(x, P[p + "ln2_g"], P[p + "ln2_b"]) if cfg.norm else x
            x = x + torch.relu(h @ P[p + "w1"] + P[p + "b1"]) @ P[p + "w2"] + P[p + "b2"]
    return x


def _map(n: int) -> torch.Tensor:
    return torch.arange(T_MAX) * n // T_MAX  # position t -> index in a level of length n


def down(z: torch.Tensor, n: int) -> torch.Tensor:
    m = _map(n)
    out = torch.zeros(n, z.shape[1], dtype=DTYPE).index_add_(0, m, z)
    return out / torch.bincount(m, minlength=n).clamp(min=1)[:, None].to(DTYPE)


def up(c: torch.Tensor, n: int) -> torch.Tensor:
    return c[_map(n)]


def decode_latent(cfg: Config, P: dict, z: torch.Tensor) -> torch.Tensor:
    h = torch.relu(z @ P["dec_w1"] + P["dec_b1"]) if cfg.dec_hidden else z
    return h @ P["dec_w2"] + P["dec_b2"]


@torch.no_grad()
def generate(cfg: Config, P: dict, prefix_ids: list[int], trace: bool = False) -> dict:
    prefix_ids = prefix_ids[: cfg.max_prefix]
    npre = len(prefix_ids)
    pre = P["tok_emb"][torch.tensor(prefix_ids, dtype=torch.long)] + P["pre_pos"][:npre] if npre else \
        torch.zeros(0, cfg.d_model, dtype=DTYPE)
    z = torch.zeros(T_MAX, cfg.D, dtype=DTYPE)
    out = {"codes": [], "levels": []}
    for l, n in enumerate(cfg.levels):
        q = down(z, n) @ P["w_in"] + P["lvl_emb"][l] + P["lvl_pos"][:n]
        att = [] if trace else None
        h = transformer(cfg, P, torch.cat([pre, q]), att)[npre:]
        codes = (h @ P[f"head{l}"] + P[f"head_b{l}"]).argmax(-1)
        z = z + up(P[f"code{l}"][codes], n)
        out["codes"].append(codes.tolist())
        if trace:
            out["levels"].append({"z": z.clone(), "attn": att, "h": h})
    out["tokens"] = decode_latent(cfg, P, z).argmax(-1).tolist()
    out["z"] = z
    return out
