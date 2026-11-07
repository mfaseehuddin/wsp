import itertools

import pytest
import torch

from wsp.bot import harness, kb, model
from wsp.bot import tokenizer as T

BIGRAM = """
import torch
from wsp.bot import model, tokenizer as T

CHAIN = {CHAIN}

def build():
    V = T.vocab_size()
    cfg = model.Config(d_model=V, n_layers=1, n_heads=1, d_head=1, pos="none", bias=False,
                       tie_embeddings=False)
    p = {{k: torch.zeros(s, dtype=torch.float64) for k, s in model.param_shapes(cfg).items()}}
    p["embed"] = torch.eye(V, dtype=torch.float64)
    for a, b in CHAIN:
        p["unembed"][T.tid(a), T.tid(b)] = 1.0
    return cfg, p
"""


def _model(tmp_path, seq):
    f = tmp_path / "weights.py"
    f.write_text(BIGRAM.format(CHAIN=list(itertools.pairwise(seq))))
    return harness.load(str(f))


def test_vocab_is_2000_and_replies_known():
    assert T.vocab_size() == 2000
    for st in range(4):
        for _, r in kb.faq(st).values():
            assert T.tid(T.UNK) not in T.encode(r)


def test_conversations_are_deterministic():
    assert kb.conversations("dev", 0) == kb.conversations("dev", 0)
    assert len(kb.conversations("test", 1)) == 160


def test_unknown_bigram_passes_first_turn_only(tmp_path):
    seq = ["<bot>", *T.split(kb.SOCIAL["unknown"]), "<end>"]
    cfg, p, _ = _model(tmp_path, seq)
    for c in kb.conversations("dev", 0)[:24]:
        r = harness.run_conversation(cfg, p, c)
        assert r["turns_ok"] == (
            1 if c["turns"][0]["reply"] == kb.SOCIAL["unknown"] else 0
        )


def test_tool_call_is_executed_and_result_appended(tmp_path):
    seq = [
        "<bot>",
        "<think>",
        "vpn",
        "</think>",
        "<tool>",
        "service_status",
        "email",
        "</tool>",
    ]
    cfg, p, _ = _model(tmp_path, seq)
    ids = [T.tid("<bos>"), *harness.user_ids("is email down ?")]
    db = kb.MockDB(5)
    out = harness.bot_turn(cfg, p, ids, db)
    assert out["think"] == ["vpn"] and out["tools"] == [["service_status", "email"]]
    tail = T.decode(ids[-4:-1])  # then <pad>: the bigram has no successor
    assert tail == f"<result> {db.state['email']} </result>"


def test_runs_are_charged_and_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "ROOT", tmp_path / "exp")
    harness._write_ledger(
        "a", {"budget": 2, "used": 0, "token_budget": 1, "token_base": 0}
    )
    seq = ["<bot>", *T.split(kb.SOCIAL["hello"]), "<end>"]
    f = tmp_path / "weights.py"
    f.write_text(BIGRAM.format(CHAIN=list(itertools.pairwise(seq))))
    out = harness.cmd_eval("a", str(f), [0, 1], "")
    assert out["runs_left"] == 0
    with pytest.raises(harness.BudgetExceeded):
        harness.cmd_probe("a", str(f), "<bos> <user> hi <bot>", False, "")


def test_build_cannot_run_forward(tmp_path):
    f = tmp_path / "w.py"
    f.write_text(
        BIGRAM.format(CHAIN=[]).replace(
            "    return cfg, p",
            "    model.forward(cfg, p, torch.zeros(1, 1, dtype=torch.long))\n    return cfg, p",
        )
    )
    with pytest.raises(RuntimeError):
        harness.load(str(f))


def test_shapes_checked():
    cfg = model.Config(d_model=4, n_layers=1, n_heads=1, d_head=2)
    p = {k: torch.zeros(s) for k, s in model.param_shapes(cfg).items()}
    assert p["embed"].shape == (2000, 4) and p["pos"].shape == (256, 4)
    model.check_params(cfg, p)


ARCH = """
import torch


def forward(params, tokens):
    return params["E"][tokens] @ params["U"]
"""

CUSTOM_WEIGHTS = """
import torch
from wsp.bot import tokenizer as T
from wsp.bot.custom import CustomArch

def build():
    V = T.vocab_size()
    seq = ["<bot>", *T.split("sorry , i can only help with it problems ."), "<end>"]
    d = len(seq)
    E, U = torch.zeros(V, d, dtype=torch.float64), torch.zeros(d, V, dtype=torch.float64)
    for j, (a, b) in enumerate(zip(seq, seq[1:])):
        E[T.tid(a), j] = 1.0
        U[j, T.tid(b)] = 1.0
    return CustomArch(__file__.replace("weights.py", "arch.py"), "bigram"), {"E": E, "U": U}
"""


def test_custom_arch_runs_through_harness(tmp_path):
    (tmp_path / "arch.py").write_text(ARCH)
    (tmp_path / "weights.py").write_text(CUSTOM_WEIGHTS)
    cfg, p, _ = harness.load(str(tmp_path / "weights.py"))
    assert sum(t.numel() for t in p.values()) == 2 * 2000 * 12
    c = kb.conversations("dev", 0)[9]
    assert harness.run_conversation(cfg, p, c)["turns_ok"] == 1


@pytest.mark.parametrize(
    "body",
    [
        "def forward(params, tokens):\n    if tokens[0, -1] == 3:\n        return 1\n",
        "def forward(params, tokens):\n    return torch.tensor([1, 2, 3])\n",
        "import os\ndef forward(params, tokens):\n    return 0\n",
        "def forward(params, tokens):\n    return tokens.tolist()\n",
        "def forward(params, tokens):\n    for t in tokens:\n        pass\n",
        "T = {1: 2, 3: 4, 5: 6, 7: 8, 9: 1, 2: 3, 4: 5, 6: 7, 8: 9}\ndef forward(params, tokens):\n    return 0\n",
    ],
)
def test_custom_arch_rejects_disguised_programs(body):
    from wsp.bot import custom

    with pytest.raises(custom.ArchError):
        custom.validate_source(body)


def test_unknown_user_reset_result_arrives_as_unk(tmp_path):
    seq = ["<bot>", "<tool>", "reset_password", "uberg", "</tool>"]
    cfg, p, _ = _model(tmp_path, seq)
    ids = [T.tid("<bos>"), *harness.user_ids("reset password for user uberg")]
    out = harness.bot_turn(cfg, p, ids, kb.MockDB(1))
    assert out["tools"] == [["reset_password", "uberg"]]
    assert T.decode(ids[-4:-1]) == "<result> <unk> </result>"
