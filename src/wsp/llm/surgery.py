"""Surgery helpers: change the shape of a Qwen2 HF model so that it stays a stock checkpoint.

All helpers add *inert* capacity (zero output weights), so the model computes exactly the same function
until you write non-zero values. The result always reloads with plain
`AutoModelForCausalLM.from_pretrained` (no custom code).

    widen_mlp(model, extra)      -> slice of new neuron indices (in EVERY layer's MLP)
    add_head_group(model)        -> (q_heads, kv_head): a new KV head + its group of query heads in EVERY layer
    insert_layer(model, at)      -> the new (identity) decoder layer at index `at`
    digit_ids(tok)               -> token ids of '0'..'9'

Qwen2 shares one intermediate_size / head count across layers, so widening is global.
Unused new neurons and heads are zero in other layers: they cost parameters but change nothing.
"""

from __future__ import annotations

import torch
from torch import nn


def _resize_linear(lin: nn.Linear, out_f: int | None = None, in_f: int | None = None) -> nn.Linear:
    """New Linear with zero-padded extra rows/cols, keeping existing values."""
    o, i = lin.out_features, lin.in_features
    out_f, in_f = out_f or o, in_f or i
    new = nn.Linear(in_f, out_f, bias=lin.bias is not None, dtype=lin.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:o, :i] = lin.weight
        if lin.bias is not None:
            new.bias.zero_()
            new.bias[:o] = lin.bias
    if hasattr(lin, "_wsp_base"):  # keep the harness's provenance tag
        new._wsp_base = lin._wsp_base
    return new


@torch.no_grad()
def widen_mlp(model, extra: int) -> slice:
    cfg = model.config
    old = cfg.intermediate_size
    for layer in model.model.layers:
        mlp = layer.mlp
        mlp.gate_proj = _resize_linear(mlp.gate_proj, out_f=old + extra)
        mlp.up_proj = _resize_linear(mlp.up_proj, out_f=old + extra)
        mlp.down_proj = _resize_linear(mlp.down_proj, in_f=old + extra)
        mlp.intermediate_size = old + extra
    cfg.intermediate_size = old + extra
    return slice(old, old + extra)


@torch.no_grad()
def add_head_group(model) -> tuple[list[int], int]:
    """Add one KV head and its full group of query heads to every layer.

    Qwen2.5-0.5B has 14 q heads / 2 kv heads (7 per group); this makes 21 / 3. The existing heads keep
    their kv head (q head h uses kv h // 7), so nothing changes until the new o_proj columns are non-zero.
    """
    cfg = model.config
    nq, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    g = nq // nkv
    for layer in model.model.layers:
        at = layer.self_attn
        at.q_proj = _resize_linear(at.q_proj, out_f=(nq + g) * hd)
        at.k_proj = _resize_linear(at.k_proj, out_f=(nkv + 1) * hd)
        at.v_proj = _resize_linear(at.v_proj, out_f=(nkv + 1) * hd)
        at.o_proj = _resize_linear(at.o_proj, in_f=(nq + g) * hd)
        at.head_dim = hd
        at.num_key_value_groups = g
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = nq + g, nkv + 1, hd
    return list(range(nq, nq + g)), nkv


@torch.no_grad()
def insert_layer(model, at: int):
    """Insert an identity decoder layer at index `at` (0 = right after the embeddings)."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    cfg = model.config
    layers = list(model.model.layers)
    n = len(layers) + 1
    cfg.num_hidden_layers = n
    cfg.max_window_layers = n
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = ["full_attention"] * n
    new = Qwen2DecoderLayer(cfg, at).to(next(model.parameters()).dtype)
    for name, p in new.named_parameters():
        p.copy_(torch.ones_like(p) if "layernorm" in name else torch.zeros_like(p))
    layers.insert(at, new)
    model.model.layers = nn.ModuleList(layers)
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i
    return new


def digit_ids(tok) -> list[int]:
    return [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
