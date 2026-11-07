"""Budgeted harness for generated IT-support chatbots with tool calls.

A "run" is the unit of budget: one full conversation simulated end to end
(`eval`, 1 run per conversation id), one single-position forward pass
(`probe`), or one generated bot turn (`turn`).

How a bot turn is generated (greedy):
  the model emits tokens after <bot> until it emits <end>.
  - <think> ... </think> is free scratch text (at most THINK_CAP tokens); it is
    never scored but stays in the context, so it can carry state to later turns.
  - <tool> name arg </tool>: when the model emits </tool>, the harness runs the
    mock tool and appends <result> ... </result>. The model then continues.
  - every other non-special token is part of the visible reply.
A turn is correct iff its tool calls (in order) and its visible reply tokens
exactly match the gold turn. A conversation is correct iff every turn is.
Later turns see the model's own earlier turns (including its think text).

Agent file contract (`weights.py`): `build() -> (Config, dict[str, Tensor])`,
using exactly the tensors in `model.param_shapes(cfg)`.

CLI (`uv run wsp-bot ...`), free:
  stage | kb | dev [--ids ..] | check --model F | tokenize --text T
  status --agent A | tokens --agent A
charged (runs):
  eval  --agent A --model F --ids 0,5 [--note N]          1 run per conversation
  probe --agent A --model F --context "<bos> <user> hi <bot>" [--trace] [--note N]   1 run
  turn  --agent A --model F --context "..." [--seed S] [--note N]                    1 run
admin (WSP_ADMIN=1):
  init --agent A --budget N --tokens T | round --agent A --stage S --tokens T
  set-stage --stage S | test --model F [--split test|field] | chat --model F
  diff --old F --new G
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

from . import custom, kb, model
from . import tokenizer as T

ROOT = Path(os.environ.get("WSP_BOT_ROOT", "experiments/bot"))
THINK_CAP, TURN_CAP, USER_CAP = 12, 60, 40


class BudgetExceeded(RuntimeError):
    pass


# ---------------------------------------------------------------- stage, ledger, tokens


def stage() -> int:
    f = ROOT / "STAGE"
    return int(f.read_text()) if f.exists() else 0


def agent_dir(agent: str) -> Path:
    return ROOT / "agents" / agent


def _ledger_path(agent: str) -> Path:
    return agent_dir(agent) / ".ledger.json"


def read_ledger(agent: str) -> dict:
    p = _ledger_path(agent)
    if not p.exists():
        raise FileNotFoundError(f"no ledger for {agent!r}")
    return json.loads(p.read_text())


def _write_ledger(agent: str, led: dict) -> None:
    _ledger_path(agent).parent.mkdir(parents=True, exist_ok=True)
    _ledger_path(agent).write_text(json.dumps(led))


def _charge(agent: str, cost: int) -> dict:
    led = read_ledger(agent)
    if cost > led["budget"] - led["used"]:
        raise BudgetExceeded(
            f"costs {cost} runs, {led['budget'] - led['used']} remain. Nothing charged."
        )
    led["used"] += cost
    _write_ledger(agent, led)
    return led


def _log(agent: str, rec: dict) -> None:
    with open(agent_dir(agent) / "log.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), "stage": stage(), **rec}) + "\n")


def context_tokens(transcript: str | Path) -> int:
    """Agent context size: last API call's input (incl. cache) + output tokens."""
    last = 0
    with open(transcript) as f:
        for line in f:
            d = json.loads(line)
            m = d.get("message")
            if d.get("type") == "assistant" and isinstance(m, dict) and m.get("usage"):
                u = m["usage"]
                n = (
                    u.get("input_tokens", 0)
                    + u.get("cache_read_input_tokens", 0)
                    + u.get("cache_creation_input_tokens", 0)
                    + u.get("output_tokens", 0)
                )
                last = n or last  # failed API calls report 0; keep the last real size
    return last


def token_status(agent: str) -> dict:
    led = read_ledger(agent)
    tf = agent_dir(agent) / ".transcript"
    now = context_tokens(tf.read_text().strip()) if tf.exists() else 0
    used = max(0, now - led.get("token_base", 0))
    lim = led["token_budget"]
    return {
        "tokens_used_this_round": used,
        "token_budget": lim,
        "remaining": lim - used,
        "rule": "at the limit you will be told to stop; finish within 100k more",
    }


# ---------------------------------------------------------------- loading


def load(path: str | Path) -> tuple[model.Config, dict, str]:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(
        f"bot_weights_{path.parent.name}", path
    )
    mod = importlib.util.module_from_spec(spec)
    real = model.forward

    def _blocked(*a, **k):
        raise RuntimeError(
            "forward passes are not allowed inside build(); use the harness"
        )

    model.forward = _blocked
    custom.CustomArch.blocked = True
    try:
        spec.loader.exec_module(mod)
        cfg, params = mod.build()
    finally:
        model.forward = real
        custom.CustomArch.blocked = False
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    if isinstance(cfg, custom.CustomArch):
        return cfg, custom.check_params(params), sha + "+" + cfg.sha
    if not isinstance(cfg, model.Config):
        raise TypeError("build() must return (Config or CustomArch, params)")
    return cfg, model.check_params(cfg, params), sha


def forward(cfg, params, tokens: torch.Tensor, trace: bool = False):
    """Dispatch to the fixed transformer or to an agent's custom architecture."""
    if isinstance(cfg, custom.CustomArch):
        return cfg.forward(params, tokens, trace=trace)
    return model.forward(cfg, params, tokens, trace=trace)


# ---------------------------------------------------------------- generation


def _next(cfg, params, ids: list[int]) -> int:
    return int(forward(cfg, params, torch.tensor([ids]))[0, -1].argmax())


def bot_turn(cfg, params, ids: list[int], db: kb.MockDB) -> dict:
    """Generate one bot turn in place (ids is extended). Returns what was produced."""
    v, specials = T.vocab(), set(T.SPECIALS)
    out = {"think": [], "tools": [], "reply": [], "error": None}
    mode, buf, n = "free", [], 0
    while True:
        if len(ids) >= T.MAX_LEN:
            out["error"] = "context full"
            return out
        tok = v[_next(cfg, params, ids)]
        ids.append(T.tid(tok))
        n += 1
        if n > TURN_CAP:
            out["error"] = f"turn longer than {TURN_CAP} tokens"
            return out
        if mode == "think":
            if tok == T.THINK_END:
                mode = "free"
            elif len(out["think"]) >= THINK_CAP:
                out["error"] = f"think longer than {THINK_CAP} tokens"
                return out
            else:
                out["think"].append(tok)
        elif mode == "tool":
            if tok == T.TOOL_END:
                res = db.call(buf[0], buf[1:]) if buf else ["error"]
                # out-of-vocabulary result words (e.g. "unknown") arrive as <unk>
                m, unk = T.token_to_id(), T.tid(T.UNK)
                ids.extend(m.get(x, unk) for x in [T.RESULT, *res, T.RESULT_END])
                out["tools"].append(buf)
                mode, buf = "free", []
            elif tok in specials:
                out["error"] = f"special token {tok} inside a tool call"
                return out
            else:
                buf.append(tok)
        elif tok == T.THINK:
            mode = "think"
        elif tok == T.TOOL:
            mode = "tool"
        elif tok == T.END:
            return out
        elif tok in specials:
            out["error"] = f"unexpected {tok}"
            return out
        else:
            out["reply"].append(tok)


def user_ids(text: str) -> list[int]:
    return [T.tid(T.USER), *T.encode(text)[:USER_CAP], T.tid(T.BOT)]


@torch.no_grad()
def run_conversation(cfg, params, conv: dict) -> dict:
    db = kb.MockDB(conv["seed"])
    ids = [T.tid(T.BOS)]
    for k, turn in enumerate(conv["turns"]):
        ids += user_ids(turn["user"])
        got = bot_turn(cfg, params, ids, db)
        want_reply = T.split(turn["reply"])
        if got["error"] or got["tools"] != turn["tools"] or got["reply"] != want_reply:
            return {
                "ok": False,
                "turns_ok": k,
                "turns": len(conv["turns"]),
                "fail": {
                    "turn": k,
                    "user": turn["user"],
                    "want": {"tools": turn["tools"], "reply": turn["reply"]},
                    "got": {
                        "think": " ".join(got["think"]),
                        "tools": got["tools"],
                        "reply": " ".join(got["reply"]),
                        "error": got["error"],
                    },
                },
            }
    return {"ok": True, "turns_ok": len(conv["turns"]), "turns": len(conv["turns"])}


# ---------------------------------------------------------------- commands


def cmd_check(path: str) -> dict:
    cfg, params, sha = load(path)
    groups: dict[str, int] = {}
    for k, t in params.items():
        g = ".".join(k.split(".")[:2]) if k.startswith("layers.") else k.split(".")[0]
        groups[g] = groups.get(g, 0) + t.numel()
    return {
        "config": cfg.to_dict(),
        "vocab_size": T.vocab_size(),
        "params": model.n_params(params),
        "nonzero": int(sum((t != 0).sum() for t in params.values())),
        "by_group": groups,
        "sha": sha,
    }


def cmd_eval(agent: str, path: str, ids: list[int], note: str) -> dict:
    dev = kb.conversations("dev", stage())
    if not ids or any(i < 0 or i >= len(dev) for i in ids):
        raise ValueError(f"ids must be in 0..{len(dev) - 1}")
    cfg, params, sha = load(path)
    led = _charge(agent, len(ids))
    rows = [{"id": i, **run_conversation(cfg, params, dev[i])} for i in ids]
    _log(
        agent,
        {
            "kind": "eval",
            "model": path,
            "sha": sha,
            "ids": ids,
            "cost": len(ids),
            "used": led["used"],
            "ok": sum(r["ok"] for r in rows),
            "note": note,
        },
    )
    return {
        "cost": len(ids),
        "runs_left": led["budget"] - led["used"],
        "ok": sum(r["ok"] for r in rows),
        "of": len(rows),
        "results": rows,
    }


def _context(text: str) -> list[int]:
    ids = T.encode(text)
    if not ids or len(ids) > T.MAX_LEN:
        raise ValueError("context must be 1..256 tokens")
    return ids


def cmd_probe(agent, path, text, do_trace, note) -> dict:
    cfg, params, sha = load(path)
    ids = _context(text)
    led = _charge(agent, 1)
    v = T.vocab()
    logits, tr = forward(cfg, params, torch.tensor([ids]), trace=True)

    def top(vec, k=5):
        val, idx = vec.topk(k)
        return [
            f"{v[i]}:{x:.3g}" for x, i in zip(val.tolist(), idx.tolist(), strict=True)
        ]

    out = {"next_top5": top(logits[0, -1]), "runs_left": led["budget"] - led["used"]}
    if do_trace and isinstance(cfg, custom.CustomArch):
        out["trace"] = {k: str(v)[:400] for k, v in tr.items()}  # arch-defined trace
    elif do_trace:
        U = model.unembed_matrix(cfg, params)
        out["logit_lens"] = [
            top(model._ln(x, params, "lnf", cfg)[0, -1] @ U, 3) for x in tr["resid"]
        ]
        out["heads"] = []
        for li, att in enumerate(tr["attn"]):
            for h in range(cfg.n_heads):
                w, j = att[0, h, -1].topk(min(3, len(ids)))
                out["heads"].append(
                    f"L{li}H{h}: "
                    + " ".join(
                        f"{v[ids[jj]]}@{jj}:{ww:.2f}"
                        for ww, jj in zip(w.tolist(), j.tolist(), strict=True)
                    )
                )
    _log(
        agent,
        {
            "kind": "probe",
            "model": path,
            "sha": sha,
            "cost": 1,
            "used": led["used"],
            "context": text[-120:],
            "note": note,
        },
    )
    return out


@torch.no_grad()
def cmd_turn(agent, path, text, seed, note) -> dict:
    cfg, params, sha = load(path)
    ids = _context(text)
    led = _charge(agent, 1)
    got = bot_turn(cfg, params, ids, kb.MockDB(seed))
    _log(
        agent,
        {
            "kind": "turn",
            "model": path,
            "sha": sha,
            "cost": 1,
            "used": led["used"],
            "context": text[-120:],
            "note": note,
        },
    )
    return {
        "think": " ".join(got["think"]),
        "tools": got["tools"],
        "reply": " ".join(got["reply"]),
        "error": got["error"],
        "runs_left": led["budget"] - led["used"],
    }


KINDS = [
    "faq",
    "hello-faq-thanks",
    "faq-escalate",
    "escalate-recall",
    "ticket-status",
    "service-status",
    "urgent-page",
    "reset-user",
    "topic-switch-bye",
    "unknown-faq",
    "switch-escalate",
    "hello-service-thanks-bye",
]


def cmd_test(path: str, split: str = "test", st: int | None = None) -> dict:
    st = stage() if st is None else st
    cfg, params, _ = load(path)
    convs = kb.conversations(split, st)
    res = [run_conversation(cfg, params, c) for c in convs]
    per: dict[str, list[int]] = {}
    for c, r in zip(convs, res, strict=True):
        per.setdefault(KINDS[c["id"] % 12], []).append(int(r["ok"]))
    tool_turns = [
        (t, r)
        for c, r in zip(convs, res, strict=True)
        for k, t in enumerate(c["turns"])
        if t["tools"] and k <= r["turns_ok"]
    ]
    return {
        "stage": st,
        "split": split,
        "params": model.n_params(params),
        "nonzero": int(sum((t != 0).sum() for t in params.values())),
        "conv_exact": sum(r["ok"] for r in res) / len(res),
        "turn_exact": sum(r["turns_ok"] for r in res) / sum(r["turns"] for r in res),
        "tool_turns_reached": len(tool_turns),
        "per_kind": {k: f"{sum(x)}/{len(x)}" for k, x in per.items()},
        "failures": [
            {"id": c["id"], **r["fail"]}
            for c, r in zip(convs, res, strict=True)
            if not r["ok"]
        ],
    }


def cmd_diff(old: str, new: str) -> dict:
    (c1, p1, _), (c2, p2, _) = load(old), load(new)
    if c1 != c2:
        return {
            "config_changed": True,
            "old": c1.to_dict(),
            "new": c2.to_dict(),
            "params_old": model.n_params(p1),
            "params_new": model.n_params(p2),
        }
    ch = {k: int((p1[k] != p2[k]).sum()) for k in p1}
    return {
        "config_changed": False,
        "changed": sum(ch.values()),
        "of": model.n_params(p1),
        "by_tensor": {k: n for k, n in ch.items() if n},
    }


def chat(path: str) -> None:
    cfg, params, _ = load(path)
    ids, db = [T.tid(T.BOS)], kb.MockDB(int(time.time()))
    print("chat with the bot (empty line to quit)")
    while line := input("you> ").strip():
        ids += user_ids(line)
        got = bot_turn(cfg, params, ids, db)
        if got["think"]:
            print(f"  [think] {' '.join(got['think'])}")
        for t in got["tools"]:
            print(f"  [tool]  {' '.join(t)}")
        print(
            "bot>",
            " ".join(got["reply"])
            + (f"   (error: {got['error']})" if got["error"] else ""),
        )


def _admin() -> None:
    if os.environ.get("WSP_ADMIN") != "1":
        sys.exit("admin command: agents must not use this")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="wsp-bot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    spec = {
        "stage": [],
        "kb": [],
        "dev": [("--ids", str, "")],
        "check": [("--model", str, None)],
        "tokenize": [("--text", str, None)],
        "status": [("--agent", str, None)],
        "tokens": [("--agent", str, None)],
        "eval": [
            ("--agent", str, None),
            ("--model", str, None),
            ("--ids", str, None),
            ("--note", str, ""),
        ],
        "probe": [
            ("--agent", str, None),
            ("--model", str, None),
            ("--context", str, None),
            ("--note", str, ""),
        ],
        "turn": [
            ("--agent", str, None),
            ("--model", str, None),
            ("--context", str, None),
            ("--seed", int, 0),
            ("--note", str, ""),
        ],
        "init": [
            ("--agent", str, None),
            ("--budget", int, None),
            ("--tokens", int, None),
        ],
        "round": [
            ("--agent", str, None),
            ("--stage", int, None),
            ("--tokens", int, None),
        ],
        "set-stage": [("--stage", int, None)],
        "test": [("--model", str, None), ("--split", str, "test")],
        "chat": [("--model", str, None)],
        "diff": [("--old", str, None), ("--new", str, None)],
    }
    for name, args in spec.items():
        s = sub.add_parser(name)
        for flag, typ, default in args:
            s.add_argument(flag, type=typ, default=default, required=default is None)
        if name == "probe":
            s.add_argument("--trace", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "stage":
            out = {"stage": stage()}
        elif a.cmd == "kb":
            st = stage()
            out = {
                "stage": st,
                "faq": {k: {"team": t, "reply": r} for k, (t, r) in kb.faq(st).items()},
                "social": kb.SOCIAL,
                "flows": kb.FLOWS,
                "tools": kb.TOOLS,
                "services": kb.SERVICES,
                "states": kb.STATES,
                "statuses": kb.STATUSES,
                "teams": kb.TEAMS,
                "ticket_ids": "4000..4099",
                "users_in_db": kb.USERS,
            }
        elif a.cmd == "dev":
            convs = kb.conversations("dev", stage())
            ids = [int(x) for x in a.ids.split(",")] if a.ids else range(len(convs))
            out = [
                {
                    "id": convs[i]["id"],
                    "kind": KINDS[i % 12],
                    "turns": convs[i]["turns"],
                }
                for i in ids
            ]
        elif a.cmd == "check":
            out = cmd_check(a.model)
        elif a.cmd == "tokenize":
            ids = T.encode(a.text)
            out = {"ids": ids, "tokens": [T.vocab()[i] for i in ids]}
        elif a.cmd == "status":
            led = read_ledger(a.agent)
            out = {
                "stage": stage(),
                "runs_budget": led["budget"],
                "runs_used": led["used"],
                "runs_left": led["budget"] - led["used"],
            }
        elif a.cmd == "tokens":
            out = token_status(a.agent)
        elif a.cmd == "eval":
            out = cmd_eval(a.agent, a.model, [int(x) for x in a.ids.split(",")], a.note)
        elif a.cmd == "probe":
            out = cmd_probe(a.agent, a.model, a.context, a.trace, a.note)
        elif a.cmd == "turn":
            out = cmd_turn(a.agent, a.model, a.context, a.seed, a.note)
        else:
            _admin()
            if a.cmd == "init":
                if _ledger_path(a.agent).exists():
                    raise FileExistsError("ledger exists")
                _write_ledger(
                    a.agent,
                    {
                        "budget": a.budget,
                        "used": 0,
                        "token_budget": a.tokens,
                        "token_base": 0,
                    },
                )
                out = read_ledger(a.agent)
            elif a.cmd == "round":
                led = read_ledger(a.agent)
                tf = agent_dir(a.agent) / ".transcript"
                led.update(
                    used=0,
                    token_budget=a.tokens,
                    token_base=context_tokens(tf.read_text().strip())
                    if tf.exists()
                    else 0,
                )
                _write_ledger(a.agent, led)
                out = led
            elif a.cmd == "set-stage":
                (ROOT / "STAGE").write_text(str(a.stage))
                out = {"stage": a.stage}
            elif a.cmd == "test":
                out = cmd_test(a.model, a.split)
            elif a.cmd == "diff":
                out = cmd_diff(a.old, a.new)
            else:
                chat(a.model)
                return
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
