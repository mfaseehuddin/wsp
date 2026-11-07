"""Verbatim Qwen2.5-0.5B decoder layers, usable as a frozen, KV-cached conversation encoder inside the NCM.

Implements exactly the HF Qwen2DecoderLayer forward (checked against transformers in tests/test_ncmchat_qwen.py):
    h = RMSNorm(x; input_layernorm, eps 1e-6)
    q, k, v = h Wq^T + bq, h Wk^T + bk, h Wv^T + bv   (14 query heads, 2 kv heads, head 64; q head h uses kv h // 7)
    RoPE theta 1e6 (rotate-half) on q, k;  causal softmax(q.k / 8);  x += attn Wo^T
    h = RMSNorm(x; post_attention_layernorm);  x += down( silu(gate h) * up h )
Weights are read from models/Qwen2.5-0.5B/model.safetensors; they are never written into the agent's params.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

QW = Path(__file__).resolve().parents[3] / "models/Qwen2.5-0.5B/model.safetensors"
HID, NQ, NKV, HD, THETA, EPS = 896, 14, 2, 64, 1_000_000.0, 1e-6


def tensor(name: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(QW, "pt") as f:
        return f.get_tensor(name).float()


@lru_cache(None)
def embed() -> torch.Tensor:
    return tensor("model.embed_tokens.weight")


@lru_cache(None)
def final_norm() -> torch.Tensor:
    return tensor("model.norm.weight")


@lru_cache(None)
def layer(i: int) -> dict[str, torch.Tensor]:
    p = f"model.layers.{i}."
    names = ["self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight", "self_attn.k_proj.bias",
             "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.o_proj.weight", "mlp.gate_proj.weight",
             "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"]
    return {n: tensor(p + n) for n in names}


def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def _rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:  # x [H, n, 64]
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
    ang = pos.float()[:, None] * inv[None]
    cos, sin = torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)
    x1, x2 = x[..., : HD // 2], x[..., HD // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


@torch.no_grad()
def block(i: int, x: torch.Tensor, pos: torch.Tensor, ck: torch.Tensor, cv: torch.Tensor):
    """New rows x [n, 896] at positions pos, cached k/v [2, m, 64]. Returns (x_out, k_new, v_new)."""
    W = layer(i)
    n, m = x.shape[0], ck.shape[1]
    h = rms(x, W["input_layernorm.weight"])
    q = (h @ W["self_attn.q_proj.weight"].T + W["self_attn.q_proj.bias"]).view(n, NQ, HD).transpose(0, 1)
    k = (h @ W["self_attn.k_proj.weight"].T + W["self_attn.k_proj.bias"]).view(n, NKV, HD).transpose(0, 1)
    v = (h @ W["self_attn.v_proj.weight"].T + W["self_attn.v_proj.bias"]).view(n, NKV, HD).transpose(0, 1)
    q, k = _rope(q, pos), _rope(k, pos)
    K, V = torch.cat([ck, k], 1), torch.cat([cv, v], 1)
    K, V = K.repeat_interleave(NQ // NKV, 0), V.repeat_interleave(NQ // NKV, 0)
    s = q @ K.transpose(1, 2) / HD**0.5
    mask = torch.arange(m + n)[None, :] > (torch.arange(n)[:, None] + m)
    a = torch.softmax(s.masked_fill(mask, float("-inf")), -1)
    o = (a @ V).transpose(0, 1).reshape(n, NQ * HD)
    x = x + o @ W["self_attn.o_proj.weight"].T
    h = rms(x, W["post_attention_layernorm.weight"])
    x = x + (torch.nn.functional.silu(h @ W["mlp.gate_proj.weight"].T) * (h @ W["mlp.up_proj.weight"].T)) @ W["mlp.down_proj.weight"].T
    return x, k, v
