"""KV-cached generation in the wsp-pure transformer equals recomputing from scratch."""

import torch

from wsp.pure import transformer as TF


def test_cache_matches_full():
    for act, norm, pos, kv in [("relu", "rms", "rope", 0), ("swiglu", "layer", "learned", 2), ("gelu", "none", "none", 1)]:
        c = TF.Config(vocab=40, d_model=16, n_layers=2, n_heads=4, d_head=8, n_kv_heads=kv, d_mlp=24, act=act,
                      norm=norm, pos=pos, max_ctx=64, tied=(pos != "learned"))
        g = torch.Generator().manual_seed(0)
        P = {k: torch.randn(*s, generator=g) * 0.3 for k, s in TF.shapes(c).items()}
        TF.validate(c, P)
        seq = torch.randint(0, 40, (20,), generator=g).tolist()
        full = TF.forward(c, P, TF.Cache(c), seq)
        ca, parts = TF.Cache(c), []
        for i in range(0, 20, 7):
            parts.append(TF.forward(c, P, ca, seq[i : i + 7]))
        assert torch.allclose(full, torch.cat(parts), atol=1e-4), (act, norm, pos)
