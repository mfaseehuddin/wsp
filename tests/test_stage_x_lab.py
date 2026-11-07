import json

import pytest
import torch

from wsp.stage_x import harness, lab, model

TINY = """import torch


def build():
    cfg = dict(d_model=16, n_layers=1, n_heads=1, d_head=8, d_mlp=0, max_len=256,
               pos="none", norm=False, bias=True, tie_embeddings=False)
    V = 273
    p = {"embed": torch.zeros(V, 16), "unembed": torch.zeros(16, V), "b_unembed": torch.zeros(V)}
    for k, s in (("wq", (1, 16, 8)), ("wk", (1, 16, 8)), ("wv", (1, 16, 8)), ("wo", (1, 8, 16)),
                 ("bq", (1, 8)), ("bk", (1, 8)), ("bv", (1, 8)), ("bo", (16,))):
        p["layers.0." + k] = torch.zeros(s)
    p["b_unembed"][10] = 5.0
    g = [dict(name="stop.end", tensor="b_unembed", index=[[10]], mechanism="END", why="5>0")]
    return cfg, p, g, ["hello"]
"""


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    (tmp_path / "dev.json").write_text(
        json.dumps(
            [
                {
                    "id": "a",
                    "category": "c",
                    "turns": [{"user": "hi", "check": {"any": ["x", ""]}}],
                },
                {
                    "id": "b",
                    "category": "c",
                    "turns": [{"user": "yo"}, {"user": "again"}],
                },
            ]
        )
    )
    d = tmp_path / "agents" / "t"
    d.mkdir(parents=True)
    lab._write_ledger(
        "t", {"budget": 3, "used": 0, "token_budget": 10, "token_base": 0}
    )
    (d / "weights.py").write_text(TINY)
    return tmp_path


def test_eval_charges_per_case_and_refuses_over_budget(root):
    out = lab.cmd_eval("t", None, "0,1", "n", 8)
    assert out["turns_executed"] == 3 and out["mechanical_passes"] == 1
    assert lab.read_ledger("t")["used"] == 2
    with pytest.raises(lab.BudgetExceeded):
        lab.cmd_eval("t", None, "0,1", "n", 8)
    assert lab.read_ledger("t")["used"] == 2


def test_build_cache_keeps_only_latest(root):
    a = lab.built("t")
    (root / "agents/t/weights.py").write_text(TINY.replace("5.0", "6.0"))
    b = lab.built("t")
    assert a != b and not a.exists() and b.exists()


def test_reuse_report_flags_identical_and_stock_text():
    stock = "this is a fixed sentence that never changes at all"
    res = [
        {"id": str(i), "turns": [{"user": f"q{i}", "reply": stock}]} for i in range(3)
    ]
    r = lab.reuse_report(res)
    assert r["identical_reply_texts"] == 1 and r["stock_spans"] > 0
    varied = [
        {
            "id": str(i),
            "turns": [{"user": f"name {n}", "reply": f"hello {n} nice to meet you"}],
        }
        for i, n in enumerate(["ana", "bo", "cy"])
    ]
    assert lab.reuse_report(varied)["identical_reply_texts"] == 0


def test_long_literals():
    assert lab.long_literals('x = "one two three"\ny = "a b c d e f g h"') == [
        "a b c d e f g h"
    ]


def test_trace_matches_forward():
    torch.manual_seed(0)
    cfg = model.Config(
        d_model=8, n_layers=2, n_heads=2, d_head=4, d_mlp=6, max_len=32, norm=True
    )
    p = {k: torch.randn(s) for k, s in model.param_shapes(cfg, 300).items()}
    x = torch.randint(0, 300, (1, 9))
    full = model.forward(cfg, p, x)[0][0, -1]
    traced, layers = model.trace(cfg, p, x)
    assert torch.allclose(full, traced[0, -1], atol=1e-5) and len(layers) == 2


def test_new_check_kinds(tmp_path):
    src = tmp_path / "w.py"
    src.write_text(TINY)
    art = tmp_path / "art"
    harness.build(src, art)
    cases = tmp_path / "c.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "category": "c",
                    "turns": [
                        {"user": "hi", "check": {"absent": ["zzz"], "tools": []}},
                        {"user": "hi", "check": {"any": ["zzz"]}},
                    ],
                }
            ]
        )
    )
    rep = harness.evaluate(art, cases, 8)
    assert [t["mechanical_pass"] for t in rep["results"][0]["turns"]] == [True, False]
