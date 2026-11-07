import pytest
import torch

from wsp.support import data, harness, model, tokenizer

THANKS_WEIGHTS = """
import torch
from wsp.support import data, model, tokenizer as tk

def build():
    V = tk.vocab_size()
    cfg = model.Config(d_model=V, n_layers=1, n_heads=1, d_head=1, pos="none", bias=False)
    p = {k: torch.zeros(s, dtype=torch.float64) for k, s in model.param_shapes(cfg).items()}
    p["embed"] = torch.eye(V, dtype=torch.float64)
    seq = [tk.tid(tk.SEP)] + tk.encode_words(data.REPLIES["thanks"]) + [tk.tid(tk.EOS)]
    for a, b in zip(seq, seq[1:]):
        p["unembed"][a, b] = 1.0
    return cfg, p
"""


@pytest.fixture
def weights(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "ROOT", tmp_path / "exp")
    f = tmp_path / "weights.py"
    f.write_text(THANKS_WEIGHTS)
    return str(f)


def test_tokenizer_round_trip_and_replies_in_vocab():
    for r in data.REPLIES.values():
        ids = tokenizer.encode_words(r)
        assert tokenizer.tid(tokenizer.UNK) not in ids
        assert tokenizer.decode(ids) == " ".join(tokenizer.split(r))
    assert len(data._test_set()) == 112


def test_teacher_forcing_matches_greedy_generation(weights):
    cfg, params, _ = harness.load(weights)
    dev = data.dev_set()
    res = harness.teacher_forced(cfg, params, dev)
    for it, r in zip(dev, res, strict=True):
        assert r["exact"] == (it["intent"] == "thanks")
        gen = harness.generate(cfg, params, it["query"])
        assert (gen == " ".join(tokenizer.split(data.REPLIES[it["intent"]]))) == r[
            "exact"
        ]


def test_eval_charges_one_pass_per_query(weights):
    harness.init_ledger("a", 3)
    out = harness.cmd_eval("a", weights, [48, 49, 0], "")
    assert out["exact"] == 2 and harness.read_ledger("a")["used"] == 3
    with pytest.raises(harness.BudgetExceeded):
        harness.cmd_probe("a", weights, "thanks", "", True, "")
    assert harness.read_ledger("a")["used"] == 3


def test_build_cannot_run_forward(tmp_path):
    f = tmp_path / "w.py"
    f.write_text(
        THANKS_WEIGHTS.replace(
            "    return cfg, p",
            "    model.forward(cfg, p, torch.zeros(1, 1, dtype=torch.long))\n    return cfg, p",
        )
    )
    with pytest.raises(RuntimeError):
        harness.load(str(f))


def test_shape_and_name_checks():
    cfg = model.Config(d_model=4, n_layers=1, n_heads=1, d_head=2)
    p = {k: torch.zeros(s) for k, s in model.param_shapes(cfg).items()}
    model.check_params(cfg, p)
    p["embed"] = torch.zeros(3, 4)
    with pytest.raises(ValueError):
        model.check_params(cfg, p)
    del p["embed"]
    with pytest.raises(ValueError):
        model.check_params(cfg, p)
