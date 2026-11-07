"""KV-cache correctness for the NCM chat model: incremental == from-scratch."""

import torch

from wsp.ncmchat import model


def _rand(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(*s, generator=g) * 0.2 for k, s in model.shapes(cfg).items()}


def test_cache_incremental_matches_full():
    cfg = model.Config(vocab=50, D=8, levels=(2, 16, 64), codebooks=(5, 7, 50), d_model=16, n_layers=3,
                       n_heads=2, d_head=8, d_mlp=12, norm=True)
    P = _rand(cfg)
    seq = list(torch.randint(0, 50, (23,), generator=torch.Generator().manual_seed(1)).tolist())
    a, b = model.Cache(cfg), model.Cache(cfg)
    model.extend(cfg, P, a, seq)
    for i in range(0, 23, 5):
        model.extend(cfg, P, b, seq[i : i + 5])
    for i in range(cfg.n_layers):
        assert torch.allclose(a.k[i], b.k[i], atol=1e-5) and torch.allclose(a.v[i], b.v[i], atol=1e-5)
    assert model.reply(cfg, P, a)["tokens"] == model.reply(cfg, P, b)["tokens"]


def test_rope_relative():
    q, k = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
    s1 = (model.rope(q, torch.tensor([10]), 1e4) * model.rope(k, torch.tensor([3]), 1e4)).sum()
    s2 = (model.rope(q, torch.tensor([107]), 1e4) * model.rope(k, torch.tensor([100]), 1e4)).sum()
    assert torch.allclose(s1, s2, atol=1e-4)
