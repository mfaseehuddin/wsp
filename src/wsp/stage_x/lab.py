"""Agent-facing Stage X CLI (`wsp-x`): budgeted build/eval/chat/probe for builder agents.

A "run" is the unit of budget: one dev conversation (`eval`), one chat turn
(`chat`), or one probe. Free: spec facts, dev listing, check, tokenize, status.
Admin commands (init, test) need WSP_ADMIN=1; builder agents never use them.

Builds always go through the sanitized subprocess in `harness.build`, so the
builder's Python never runs in this process and inference sees only arrays.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch

from wsp.bot.harness import context_tokens

from . import artifact, harness, model
from . import tokenizer as T

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("WSP_X_ROOT", REPO / "experiments/stage_x"))
TEXT_UNIT_WORDS = (
    6  # longest stored text unit allowed (see TASK.md, "No canned answers")
)


class BudgetExceeded(RuntimeError):
    pass


# ---------------------------------------------------------------- ledger


def agent_dir(agent: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", agent):
        raise ValueError("bad agent name")
    return ROOT / "agents" / agent


def read_ledger(agent: str) -> dict:
    p = agent_dir(agent) / ".ledger.json"
    if not p.exists():
        raise FileNotFoundError(f"no ledger for {agent!r}")
    return json.loads(p.read_text())


def _write_ledger(agent: str, led: dict) -> None:
    (agent_dir(agent) / ".ledger.json").write_text(json.dumps(led))


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
        f.write(json.dumps({"ts": time.time(), **rec}) + "\n")


def token_status(agent: str) -> dict:
    led = read_ledger(agent)
    tf = agent_dir(agent) / ".transcript"
    now = context_tokens(tf.read_text().strip()) if tf.exists() else 0
    used = max(0, now - led.get("token_base", 0))
    lim = led["token_budget"]
    return {
        "tokens_used": used,
        "token_budget": lim,
        "remaining": lim - used,
        "rule": "at the limit you will be told to stop; finish within 100k more",
    }


# ---------------------------------------------------------------- building


def _source(agent: str, path: str | None) -> Path:
    p = Path(path) if path else agent_dir(agent) / "weights.py"
    return p.resolve()


def built(agent: str, path: str | None = None) -> Path:
    """Build (or reuse) the artifact for the current weights.py; keep only the latest."""
    src = _source(agent, path)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    cache = agent_dir(agent) / ".artifacts"
    out = cache / sha
    if not (out / "weights.npz").exists():
        if cache.exists():
            shutil.rmtree(cache)
        cache.mkdir(parents=True)
        try:
            harness.build(src, out)
        except Exception:
            shutil.rmtree(out, ignore_errors=True)
            raise
    return out


def long_literals(source: str, limit: int = TEXT_UNIT_WORDS) -> list[str]:
    """String literals longer than `limit` words: candidate canned text units."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value.split()) > limit
        ):
            hits.append(node.value[:80])
    return hits


# ---------------------------------------------------------------- canned-answer metrics


def _words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", s.lower())


def reuse_report(results: list[dict], span: int = 8) -> dict:
    """How much reply text is stock text rather than composed from the input.

    - identical_replies: replies of >= 5 words given verbatim to different user
      messages (in different cases).
    - stock_spans: `span`-word sequences not present in the user's own text that
      recur in replies of >= 3 different cases.
    Both should stay near zero for a model that composes its answers.
    """
    by_text = defaultdict(set)
    span_cases = defaultdict(set)
    n_replies = 0
    for r in results:
        for t in r["turns"]:
            reply = t["reply"].strip()
            w = _words(reply)
            if len(w) >= 5:
                n_replies += 1
                by_text[" ".join(w)].add((r["id"], t["user"]))
            user = " ".join(_words(t["user"]))
            for i in range(len(w) - span + 1):
                s = " ".join(w[i : i + span])
                if s not in user:
                    span_cases[s].add(r["id"])
    identical = {
        k: len(v)
        for k, v in by_text.items()
        if len(v) >= 2 and len({c for c, _ in v}) >= 2
    }
    stock = sorted(
        ((k, len(v)) for k, v in span_cases.items() if len(v) >= 3),
        key=lambda kv: -kv[1],
    )
    return {
        "replies_5plus_words": n_replies,
        "identical_reply_texts": len(identical),
        "identical_examples": sorted(identical, key=lambda k: -identical[k])[:5],
        "stock_spans": len(stock),
        "stock_span_examples": [k for k, _ in stock[:5]],
    }


# ---------------------------------------------------------------- commands


def _cases(path: Path, ids: str | None) -> list[dict]:
    cases = json.loads(path.read_text())
    if ids:
        want = [int(i) for i in ids.split(",")]
        cases = [cases[i] for i in want]
    return cases


def _short(turn: dict, limit: int = 600) -> dict:
    out = {
        "user": turn["user"][:200],
        "reply": turn["reply"][:limit] + ("…" if len(turn["reply"]) > limit else ""),
        "tokens": turn["generated_tokens"],
        "pass": turn["mechanical_pass"],
    }
    for k in ("error", "think"):
        if turn.get(k):
            out[k] = turn[k][:200]
    if turn["events"]:
        out["tools"] = [
            {"req": e["request"], "violation": e["violation"]} for e in turn["events"]
        ]
    if turn.get("rubric"):
        out["rubric"] = turn["rubric"]
    return out


def cmd_eval(agent, model_path, ids, note, max_new_tokens):
    cases = _cases(ROOT / "dev.json", ids)
    art = built(agent, model_path)
    _charge(agent, len(cases))
    tmp = agent_dir(agent) / ".eval_cases.json"
    tmp.write_text(json.dumps(cases))
    t0 = time.monotonic()
    rep = harness.evaluate(art, tmp, max_new_tokens)
    tmp.unlink()
    out = {
        "mechanical_passes": rep["mechanical_passes"],
        "mechanical_failures": rep["mechanical_failures"],
        "turns_executed": rep["turns_executed"],
        "safety_violations": rep["safety_violations"],
        "seconds": round(time.monotonic() - t0, 1),
        "reuse": reuse_report(rep["results"]),
        "cases": [
            {
                "id": r["id"],
                "category": r["category"],
                "turns": [_short(t) for t in r["turns"]],
            }
            for r in rep["results"]
        ],
    }
    _log(
        agent,
        {
            "cmd": "eval",
            "ids": ids,
            "note": note,
            **{
                k: out[k]
                for k in (
                    "mechanical_passes",
                    "mechanical_failures",
                    "safety_violations",
                )
            },
        },
    )
    return out


def cmd_chat(agent, model_path, turns, files, note, max_new_tokens):
    turns = json.loads(turns)
    if not isinstance(turns, list) or not all(isinstance(t, str) for t in turns):
        raise ValueError("--turns must be a JSON list of strings")
    art = built(agent, model_path)
    _charge(agent, len(turns))
    cfg, params, tok, _ = artifact.load(art)
    chat = harness.Conversation(cfg, params, tok, json.loads(files) if files else None)
    got = []
    try:
        for text in turns:
            r = chat.turn(text, (), max_new_tokens)
            got.append(
                {
                    "user": text[:200],
                    "reply": r["reply"][:1500],
                    "think": r["think"],
                    "tokens": r["generated_tokens"],
                    "error": r["error"],
                    "tools": [
                        {
                            "req": e["request"],
                            "result": e["result"],
                            "violation": e["violation"],
                        }
                        for e in r["events"]
                    ],
                    "context_tokens": r["context_tokens"],
                    "seconds": r["seconds"],
                }
            )
            if r["error"] and len(chat.ids) < cfg.max_len:
                chat.append([T.END])
    finally:
        chat.sandbox.close()
    _log(agent, {"cmd": "chat", "turns": len(turns), "note": note})
    return {"turns": got}


def _name(tok, i):
    if i < T.BYTE0:
        names = [
            "PAD",
            "BOS",
            "USER",
            "BOT",
            "THINK",
            "THINK_END",
            "TOOL",
            "TOOL_END",
            "RESULT",
            "RESULT_END",
            "END",
            "UNK",
            "MEMORY",
            "MEMORY_END",
            "R14",
            "R15",
        ]
        return f"<{names[i]}>"
    return tok.decode([i])


def cmd_probe(agent, model_path, turns, prefix, trace, note):
    """Top-5 next tokens after the given user turns and a partial bot reply."""
    turns = json.loads(turns)
    art = built(agent, model_path)
    _charge(agent, 1)
    cfg, params, tok, _ = artifact.load(art)
    ids = [T.BOS]
    for i, text in enumerate(turns):
        ids += [T.USER] + tok.encode(text) + [T.BOT]
        if i < len(turns) - 1:
            raise ValueError("probe takes exactly one user turn; use chat for history")
    ids += tok.encode(prefix or "")
    x = torch.tensor([ids])
    if trace:
        logits, layers = model.trace(cfg, params, x)
        logits = logits[0, -1]
    else:
        logits = model.forward(cfg, params, x)[0][0, -1]
        layers = None
    v, j = logits.topk(5)
    out = {
        "context_tokens": [_name(tok, i) for i in ids][-40:],
        "position_of_last": len(ids) - 1,
        "top5": [(_name(tok, int(t)), round(float(s), 3)) for s, t in zip(v, j)],
    }
    if layers:
        for L in layers:
            L["lens"] = [(_name(tok, t), s) for t, s in L["lens"]]
        out["layers"] = layers
    _log(agent, {"cmd": "probe", "note": note})
    return out


def cmd_check(agent, model_path):
    src = _source(agent, model_path)
    art = built(agent, model_path)
    cfg, params, tok, rep = artifact.load(art)
    meta = json.loads((art / "model.json").read_text())
    doc = agent_dir(agent) / "EXPLANATION.md"
    text = doc.read_text() if doc.exists() else ""
    missing = [g["name"] for g in meta["groups"] if g["name"] not in text]
    return {
        **rep,
        "config": cfg.to_dict(),
        "vocab_size": tok.vocab_size,
        "words": len(tok.words),
        "kv_cache_mib_full_context": round(
            2 * cfg.n_layers * cfg.n_heads * cfg.d_head * cfg.max_len * 4 / 2**20, 1
        ),
        "max_abs_weight": round(max(float(t.abs().max()) for t in params.values()), 3),
        "long_text_literals": long_literals(src.read_text())[:10],
        "groups_missing_from_EXPLANATION": missing[:30],
    }


def cmd_test(model_path, cases, out, max_new_tokens):
    art = Path(model_path)
    if art.suffix == ".py":
        art = Path(out).with_suffix("") / "artifact"
        harness.build(model_path, art)
    rep = harness.evaluate(art, cases, max_new_tokens)
    rep["reuse"] = reuse_report(rep["results"])
    Path(out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return {k: v for k, v in rep.items() if k != "results"}


def main():
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser(prog="wsp-x")
    sub = ap.add_subparsers(dest="cmd", required=True)
    A = lambda p: p.add_argument("--agent", required=True)
    M = lambda p: p.add_argument("--model", default=None)

    p = sub.add_parser("status")
    A(p)
    p = sub.add_parser("tokens")
    A(p)
    p = sub.add_parser("dev")
    p.add_argument("--ids")
    p = sub.add_parser("check")
    A(p)
    M(p)
    p = sub.add_parser("tokenize")
    A(p)
    M(p)
    p.add_argument("--text", required=True)
    p = sub.add_parser("eval")
    A(p)
    M(p)
    p.add_argument("--ids", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p = sub.add_parser("chat")
    A(p)
    M(p)
    p.add_argument("--turns", required=True)
    p.add_argument("--files")
    p.add_argument("--note", default="")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p = sub.add_parser("probe")
    A(p)
    M(p)
    p.add_argument("--turns", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--note", default="")
    p = sub.add_parser("init")
    A(p)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--tokens", type=int, required=True)
    p = sub.add_parser("test")
    M(p)
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    a = ap.parse_args()

    if a.cmd in {"init", "test"} and os.environ.get("WSP_ADMIN") != "1":
        raise SystemExit("admin command: builder agents must not use it")
    try:
        if a.cmd == "init":
            d = agent_dir(a.agent)
            d.mkdir(parents=True, exist_ok=True)
            _write_ledger(
                a.agent,
                {
                    "budget": a.budget,
                    "used": 0,
                    "token_budget": a.tokens,
                    "token_base": 0,
                },
            )
            out = {"ok": True, "dir": str(d)}
        elif a.cmd == "status":
            led = read_ledger(a.agent)
            out = {
                "runs_budget": led["budget"],
                "runs_used": led["used"],
                "runs_left": led["budget"] - led["used"],
                **token_status(a.agent),
            }
        elif a.cmd == "tokens":
            out = token_status(a.agent)
        elif a.cmd == "dev":
            allc = json.loads((ROOT / "dev.json").read_text())
            want = [int(i) for i in a.ids.split(",")] if a.ids else range(len(allc))
            out = [{"index": i, **allc[i]} for i in want]
        elif a.cmd == "check":
            out = cmd_check(a.agent, a.model)
        elif a.cmd == "tokenize":
            _, _, tok, _ = artifact.load(built(a.agent, a.model))
            ids = tok.encode(a.text)
            out = {
                "ids": ids,
                "pieces": [tok.decode([i]) for i in ids],
                "bytes": len(a.text.encode()),
                "tokens": len(ids),
            }
        elif a.cmd == "eval":
            out = cmd_eval(a.agent, a.model, a.ids, a.note, a.max_new_tokens)
        elif a.cmd == "chat":
            out = cmd_chat(a.agent, a.model, a.turns, a.files, a.note, a.max_new_tokens)
        elif a.cmd == "probe":
            out = cmd_probe(a.agent, a.model, a.turns, a.prefix, a.trace, a.note)
        elif a.cmd == "test":
            out = cmd_test(a.model, a.cases, a.out, a.max_new_tokens)
        if a.cmd not in {"init", "test", "dev"} and hasattr(a, "agent"):
            led = read_ledger(a.agent)
            if isinstance(out, dict):
                out["runs_left"] = led["budget"] - led["used"]
                ts = token_status(a.agent)
                out["tokens_used"] = ts["tokens_used"]
    except (BudgetExceeded, FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        out = {"error": str(e)[-3000:]}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
