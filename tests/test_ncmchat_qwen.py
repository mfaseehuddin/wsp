"""The verbatim Qwen layers in wsp.ncmchat.qwen must reproduce HF transformers' hidden states (KV-cached, chunked)."""

import pytest
import torch

from wsp.ncmchat import qwen

pytestmark = pytest.mark.skipif(not qwen.QW.exists(), reason="Qwen2.5-0.5B not downloaded")


def test_matches_hf_hidden_states():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = qwen.QW.parent
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok("The capital of France is Paris. Two plus two equals", return_tensors="pt").input_ids
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        hs = hf(ids, output_hidden_states=True).hidden_states  # [0]=embeddings, [l]=after layer l, [24]=normed
    del hf
    n = ids.shape[1]
    x_all, cache = [], [(torch.zeros(2, 0, 64), torch.zeros(2, 0, 64)) for _ in range(24)]
    for a, b in [(0, 5), (5, n)]:  # two chunks: exercises the KV cache
        x = qwen.embed()[ids[0, a:b]]
        pos = torch.arange(a, b)
        for i in range(24):
            x, k, v = qwen.block(i, x, pos, *cache[i])
            cache[i] = (torch.cat([cache[i][0], k], 1), torch.cat([cache[i][1], v], 1))
            if i == 11:
                mid = x
        x_all.append((mid, x))
    mid = torch.cat([m for m, _ in x_all])
    last = qwen.rms(torch.cat([x for _, x in x_all]), qwen.final_norm())
    assert torch.allclose(mid, hs[12][0], atol=2e-3, rtol=1e-3), (mid - hs[12][0]).abs().max()
    assert torch.allclose(last, hs[24][0], atol=2e-3, rtol=1e-3), (last - hs[24][0]).abs().max()
