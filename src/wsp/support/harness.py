"""Budgeted harness for generated IT-support transformers.

One forward pass = one sequence through the model (any length up to MAX_LEN).
A teacher-forced pass over `<bos> query <sep> reply <eos>` checks every reply
token at once. Greedy generation reproduces the reply exactly iff every one of
those predictions is right, so one dev query costs exactly 1 pass to check.

Agent file contract (`weights.py`): `build() -> (Config, dict[str, Tensor])`,
with exactly the tensors named by `model.param_shapes(cfg)`.

CLI (`uv run wsp-support ...`):
  free (no forward pass):
    check    --model F                 validate shapes, count parameters
    tokenize --text T                  show token ids for a text
    status   --agent A                 remaining budget
  charged:
    eval  --agent A --model F --ids 0,5,9 [--note N]              1 pass per id
    probe --agent A --model F --text Q [--reply PREFIX] [--trace] [--note N]   1 pass
  admin (WSP_ADMIN=1):
    init --agent A --budget N;  test --model F;  chat --model F --text Q
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

from . import data, model, tokenizer

ROOT = Path(os.environ.get("WSP_SUPPORT_ROOT", "experiments/support"))


class BudgetExceeded(RuntimeError):
    pass


# ---------- ledger ----------


def agent_dir(agent: str) -> Path:
    return ROOT / "agents" / agent


def _ledger(agent: str) -> Path:
    return agent_dir(agent) / ".ledger.json"


def init_ledger(agent: str, budget: int) -> None:
    p = _ledger(agent)
    if p.exists():
        raise FileExistsError(f"ledger exists: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"budget": budget, "used": 0}))


def read_ledger(agent: str) -> dict:
    p = _ledger(agent)
    if not p.exists():
        raise FileNotFoundError(f"no ledger for {agent!r}")
    return json.loads(p.read_text())


def _charge(agent: str, cost: int) -> dict:
    led = read_ledger(agent)
    if cost > led["budget"] - led["used"]:
        raise BudgetExceeded(
            f"costs {cost} passes, {led['budget'] - led['used']} remain. Nothing charged."
        )
    led["used"] += cost
    _ledger(agent).write_text(json.dumps(led))
    return led


def _log(agent: str, rec: dict) -> None:
    with open(agent_dir(agent) / "log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------- loading ----------


def load(path: str | Path) -> tuple[model.Config, dict, str]:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"weights_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    real = model.forward

    def _blocked(*a, **k):
        raise RuntimeError(
            "forward passes are not allowed inside build(); use the harness"
        )

    model.forward = _blocked
    try:
        spec.loader.exec_module(mod)
        cfg, params = mod.build()
    finally:
        model.forward = real
    if not isinstance(cfg, model.Config):
        raise TypeError("build() must return (wsp.support.model.Config, params)")
    params = model.check_params(cfg, params)
    return cfg, params, hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ---------- scoring ----------


def _batch(seqs: list[list[int]]) -> torch.Tensor:
    T = max(map(len, seqs))
    out = torch.full((len(seqs), T), tokenizer.tid(tokenizer.PAD), dtype=torch.int64)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


def teacher_forced(cfg, params, items: list[dict]) -> list[dict]:
    """Score each item with one forward pass. Returns per-item results."""
    seqs, starts = [], []
    for it in items:
        seqs.append(tokenizer.example_ids(it["query"], data.REPLIES[it["intent"]]))
        starts.append(len(tokenizer.prompt_ids(it["query"])))
    logits = model.forward(cfg, params, _batch(seqs))
    v = tokenizer.vocab()
    res = []
    for b, (s, p0) in enumerate(zip(seqs, starts, strict=True)):
        gold = torch.tensor(s[p0:])
        pred = logits[b, p0 - 1 : len(s) - 1].argmax(-1)
        ok = pred == gold
        r = {"exact": bool(ok.all()), "tok_acc": round(float(ok.double().mean()), 4)}
        if not r["exact"]:
            j = int((~ok).nonzero()[0])
            r["first_error"] = f"reply pos {j}: want {v[gold[j]]!r} got {v[pred[j]]!r}"
        res.append(r)
    return res


@torch.no_grad()
def generate(cfg, params, query: str, max_new: int = 30) -> str:
    ids = tokenizer.prompt_ids(query)
    out = []
    for _ in range(max_new):
        nxt = int(model.forward(cfg, params, torch.tensor([ids]))[0, -1].argmax())
        if nxt == tokenizer.tid(tokenizer.EOS) or len(ids) >= tokenizer.MAX_LEN:
            break
        ids.append(nxt)
        out.append(nxt)
    return tokenizer.decode(out)


# ---------- commands ----------


def cmd_check(path: str) -> dict:
    cfg, params, sha = load(path)
    groups: dict[str, int] = {}
    for k, t in params.items():
        g = (
            k.split(".")[0]
            if not k.startswith("layers.")
            else ".".join(k.split(".")[:2])
        )
        groups[g] = groups.get(g, 0) + t.numel()
    return {
        "config": cfg.to_dict(),
        "vocab_size": tokenizer.vocab_size(),
        "params": model.n_params(params),
        "nonzero": int(sum((t != 0).sum() for t in params.values())),
        "by_group": groups,
        "sha": sha,
    }


def cmd_eval(agent: str, path: str, ids: list[int], note: str) -> dict:
    dev = data.dev_set()
    if not ids or any(i < 0 or i >= len(dev) for i in ids):
        raise ValueError(f"ids must be in 0..{len(dev) - 1}")
    cfg, params, sha = load(path)
    led = _charge(agent, len(ids))
    res = teacher_forced(cfg, params, [dev[i] for i in ids])
    rows = [{"id": i, **r} for i, r in zip(ids, res, strict=True)]
    out = {
        "cost": len(ids),
        "used": led["used"],
        "budget": led["budget"],
        "exact": sum(r["exact"] for r in rows),
        "of": len(rows),
        "results": rows,
    }
    _log(
        agent,
        {
            "ts": time.time(),
            "kind": "eval",
            "model": path,
            "sha": sha,
            "ids": ids,
            "cost": len(ids),
            "used": led["used"],
            "exact": out["exact"],
            "note": note,
        },
    )
    return out


def cmd_probe(
    agent: str, path: str, text: str, reply: str, do_trace: bool, note: str
) -> dict:
    cfg, params, sha = load(path)
    seq = tokenizer.prompt_ids(text) + tokenizer.encode_words(reply)
    if len(seq) > tokenizer.MAX_LEN:
        raise ValueError("sequence too long")
    led = _charge(agent, 1)
    v = tokenizer.vocab()
    logits, tr = model.forward(cfg, params, torch.tensor([seq]), trace=True)

    def top(vec, k=5):
        val, idx = vec.topk(k)
        return [
            f"{v[i]}:{x:.3g}" for x, i in zip(val.tolist(), idx.tolist(), strict=True)
        ]

    out = {
        "tokens": [v[i] for i in seq],
        "next_top5": top(logits[0, -1]),
        "cost": 1,
        "used": led["used"],
        "budget": led["budget"],
    }
    if do_trace:
        U = model.unembed_matrix(cfg, params)
        lens = []
        for li, x in enumerate(tr["resid"]):
            f = model._ln(x, params, "lnf", cfg)
            lens.append(
                {
                    "after": "embed" if li == 0 else f"layer{li - 1}",
                    "top3": top(f[0, -1] @ U, 3),
                }
            )
        heads = []
        for li, att in enumerate(tr["attn"]):
            for h in range(cfg.n_heads):
                w, j = att[0, h, -1].topk(min(3, len(seq)))
                heads.append(
                    {
                        "layer": li,
                        "head": h,
                        "attends": [
                            f"{v[seq[jj]]}@{jj}:{ww:.2f}"
                            for ww, jj in zip(w.tolist(), j.tolist(), strict=True)
                        ],
                    }
                )
        out["logit_lens"] = lens
        out["attention_last_pos"] = heads
    _log(
        agent,
        {
            "ts": time.time(),
            "kind": "probe",
            "model": path,
            "sha": sha,
            "text": text,
            "reply": reply,
            "cost": 1,
            "used": led["used"],
            "note": note,
        },
    )
    return out


def cmd_test(path: str) -> dict:
    cfg, params, _ = load(path)
    items = data._test_set()
    res = teacher_forced(cfg, params, items)
    dev_res = teacher_forced(cfg, params, data.dev_set())
    per: dict[str, list[int]] = {}
    for it, r in zip(items, res, strict=True):
        per.setdefault(it["intent"], []).append(int(r["exact"]))
    return {
        "params": model.n_params(params),
        "nonzero": int(sum((t != 0).sum() for t in params.values())),
        "config": cfg.to_dict(),
        "test_exact": sum(r["exact"] for r in res) / len(res),
        "test_tok_acc": sum(r["tok_acc"] for r in res) / len(res),
        "dev_exact": sum(r["exact"] for r in dev_res) / len(dev_res),
        "per_intent": {k: f"{sum(x)}/{len(x)}" for k, x in per.items()},
        "failures": [
            {"query": it["query"], "intent": it["intent"], **r}
            for it, r in zip(items, res, strict=True)
            if not r["exact"]
        ],
    }


def _admin() -> None:
    if os.environ.get("WSP_ADMIN") != "1":
        sys.exit("admin command: agents must not use this")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="wsp-support")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, args in {
        "init": ["agent", "budget"],
        "status": ["agent"],
        "check": ["model"],
        "tokenize": ["text"],
        "eval": ["agent", "model", "ids"],
        "probe": ["agent", "model", "text"],
        "test": ["model"],
        "chat": ["model", "text"],
    }.items():
        s = sub.add_parser(name)
        for a in args:
            s.add_argument(f"--{a}", required=True, type=int if a == "budget" else str)
        if name in ("eval", "probe"):
            s.add_argument("--note", default="")
        if name == "probe":
            s.add_argument("--reply", default="")
            s.add_argument("--trace", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "init":
            _admin()
            init_ledger(a.agent, a.budget)
            out = read_ledger(a.agent)
        elif a.cmd == "status":
            led = read_ledger(a.agent)
            out = {**led, "remaining": led["budget"] - led["used"]}
        elif a.cmd == "check":
            out = cmd_check(a.model)
        elif a.cmd == "tokenize":
            ids = tokenizer.encode_words(a.text)
            out = {"ids": ids, "tokens": [tokenizer.vocab()[i] for i in ids]}
        elif a.cmd == "eval":
            out = cmd_eval(a.agent, a.model, [int(x) for x in a.ids.split(",")], a.note)
        elif a.cmd == "probe":
            out = cmd_probe(a.agent, a.model, a.text, a.reply, a.trace, a.note)
        elif a.cmd == "test":
            _admin()
            out = cmd_test(a.model)
        else:
            _admin()
            cfg, params, _ = load(a.model)
            out = {"query": a.text, "reply": generate(cfg, params, a.text)}
    except (
        BudgetExceeded,
        FileNotFoundError,
        FileExistsError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as e:
        sys.exit(f"error: {e}")
    print(json.dumps(out, separators=(",", ":")))
