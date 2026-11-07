"""Budgeted harness for the hand-written NCM help-desk bot (Track D).

Agent contract: experiments/ncm/agents/<a>/weights.py defines

    def build() -> (cfg: wsp.ncm.model.Config, params: dict[str, Tensor], groups: list[dict])

groups = provenance: [{"name", "tensors", "mechanism", "why"}, ...]. No training, gradients or fitting.
Budget: one "run" = one `eval` (any number of dev examples) or one `probe`. check/spec/dev/tokenize are free.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from . import data, model
from . import tokenizer as T

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("WSP_NCM_ROOT", REPO / "experiments/ncm"))


class BudgetExceeded(RuntimeError):
    pass


def adir(agent: str) -> Path:
    return ROOT / "agents" / agent


def read_ledger(agent: str) -> dict:
    p = adir(agent) / ".ledger.json"
    if not p.exists():
        raise FileNotFoundError(f"no ledger for {agent!r}")
    return json.loads(p.read_text())


def _charge(agent: str) -> int:
    led = read_ledger(agent)
    if led["used"] >= led["budget"]:
        raise BudgetExceeded(f"all {led['budget']} runs used")
    led["used"] += 1
    (adir(agent) / ".ledger.json").write_text(json.dumps(led))
    return led["used"]


def _log(agent: str, rec: dict) -> None:
    with open(adir(agent) / "log.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), **rec}) + "\n")


def token_status(agent: str) -> dict:
    from wsp.bot.harness import context_tokens

    led = read_ledger(agent)
    tf = adir(agent) / ".transcript"
    now = context_tokens(tf.read_text().strip()) if tf.exists() else 0
    used = max(0, now - led.get("token_base", 0))
    return {"tokens_used": used, "token_budget": led["token_budget"], "remaining": led["token_budget"] - used}


# ---------------------------------------------------------------- model loading / running


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"ncm_weights_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    orig = torch.Tensor.backward

    def blocked(*a, **k):
        raise RuntimeError("no gradients: weights must be written, not trained")

    torch.Tensor.backward = blocked
    try:
        with torch.no_grad():
            spec.loader.exec_module(mod)
            cfg, params, groups = mod.build()
    finally:
        torch.Tensor.backward = orig
        sys.path.pop(0)
    params = {k: torch.as_tensor(v, dtype=model.DTYPE) for k, v in params.items()}
    model.validate(cfg, params)
    return cfg, params, groups or []


def reply_tokens(cfg, P, query: str, trace: bool = False) -> tuple[list[int], dict]:
    out = model.generate(cfg, P, T.encode(query)[: T.MAX_QUERY], trace=trace)
    eos = T.index()[T.EOS]
    toks = out["tokens"]
    return (toks[: toks.index(eos)] if eos in toks else toks), out


def _kind(ex: dict) -> str:
    return {(0, 0): "plain", (1, 0): "name", (0, 1): "ticket", (1, 1): "both"}[(bool(ex["name"]), bool(ex["ticket"]))]


def run_set(cfg, P, exs: list[dict]) -> list[dict]:
    res = []
    for ex in exs:
        got, _ = reply_tokens(cfg, P, ex["query"])
        want = T.encode(ex["reply"])
        core = T.encode(data.REPLIES[ex["intent"]])
        g = T.decode(got)
        res.append({**ex, "got": g, "ok": got == want, "kind": _kind(ex),
                    "intent_ok": " ".join(T.vocab()[i] for i in core) in g,
                    "name_ok": (ex["name"] is None) == (not g.startswith("hi ")) and (not ex["name"] or g.startswith(f"hi {ex['name']} ,")),
                    "ticket_ok": (ex["ticket"] is None and "ticket" not in g) or (bool(ex["ticket"]) and g.endswith("ticket " + " ".join(ex["ticket"]) + " ."))})
    return res


def summarize(res: list[dict], n_fail: int = 6) -> dict:
    acc = lambda rs, k="ok": round(sum(r[k] for r in rs) / max(1, len(rs)), 4)  # noqa: E731
    kinds = Counter(r["kind"] for r in res)
    return {
        "exact": acc(res), "n": len(res),
        "by_kind": {k: acc([r for r in res if r["kind"] == k]) for k in kinds},
        "intent_part_ok": acc(res, "intent_ok"), "name_slot_ok": acc(res, "name_ok"), "ticket_slot_ok": acc(res, "ticket_ok"),
        "failures": [{"id": r["id"], "query": r["query"], "want": r["reply"], "got": r["got"]} for r in res if not r["ok"]][:n_fail],
    }


# ---------------------------------------------------------------- commands


def cmd_check(agent: str) -> dict:
    cfg, P, groups = load(adir(agent) / "weights.py")
    n = sum(v.numel() for v in P.values())
    nz = sum(int((v != 0).sum()) for v in P.values())
    return {"ok": True, "config": cfg.to_dict(), "params": n, "nonzero": nz, "provenance_groups": len(groups),
            "forward_passes_per_reply": len(cfg.levels)}


def cmd_eval(agent: str, ids: str, note: str) -> dict:
    exs = data.dev_set()
    if ids:
        want = {int(x) for x in ids.split(",")}
        exs = [e for e in exs if e["id"] in want]
    cfg, P, _ = load(adir(agent) / "weights.py")
    k = _charge(agent)
    res = run_set(cfg, P, exs)
    out = {"run": k, **summarize(res)}
    _log(agent, {"cmd": "eval", "note": note, "n": len(exs), "exact": out["exact"]})
    return out


def cmd_probe(agent: str, text: str, note: str) -> dict:
    cfg, P, _ = load(adir(agent) / "weights.py")
    k = _charge(agent)
    got, out = reply_tokens(cfg, P, text, trace=True)
    eos = T.index()[T.EOS]
    lv = []
    for l, L in enumerate(out["levels"]):
        toks = model.decode_latent(cfg, P, L["z"]).argmax(-1).tolist()
        toks = toks[: toks.index(eos)] if eos in toks else toks
        att = L["attn"][-1].mean(0)[:, : len(T.encode(text)[: cfg.max_prefix])] if L["attn"] else None
        lv.append({"level": l, "n": cfg.levels[l], "codes": out["codes"][l],
                   "decoded_so_far": T.decode(toks),
                   "last_layer_top_prefix_attention": att.argmax(-1).tolist() if att is not None and att.numel() else None})
    _log(agent, {"cmd": "probe", "note": note, "text": text})
    return {"run": k, "tokens": [T.vocab()[i] for i in T.encode(text)], "reply": T.decode(got), "levels": lv}


def cmd_test(agent: str, out_path: str) -> dict:
    p = ROOT / "frozen" / f"{agent}.py" if (ROOT / "frozen" / f"{agent}.py").exists() else adir(agent) / "weights.py"
    cfg, P, _ = load(p)
    res = run_set(cfg, P, data._hidden_set())
    by_intent = {i: round(sum(r["ok"] for r in res if r["intent"] == i) / max(1, sum(r["intent"] == i for r in res)), 3)
                 for i in data.INTENTS}
    out = {"agent": agent, **summarize(res, 12), "by_intent": by_intent,
           "params": sum(v.numel() for v in P.values()), "nonzero": sum(int((v != 0).sum()) for v in P.values()),
           "levels": list(cfg.levels)}
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
    return out


def chat(agent: str) -> None:
    cfg, P, _ = load(adir(agent) / "weights.py")
    print("NCM help-desk bot. Ctrl-D to quit.")
    for line in sys.stdin:
        got, out = reply_tokens(cfg, P, line.strip(), trace=True)
        print("bot:", T.decode(got))


def _admin() -> None:
    if os.environ.get("WSP_ADMIN") != "1":
        sys.exit("admin command: agents must not use this")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="wsp-ncm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    A, N = ("--agent", str, None), ("--note", str, "")
    spec = {
        "spec": [], "dev": [("--ids", str, "")], "tokenize": [("--text", str, None)],
        "status": [A], "tokens": [A], "check": [A],
        "eval": [A, ("--ids", str, ""), N], "probe": [A, ("--text", str, None), N],
        "init": [A, ("--budget", int, None), ("--tokens", int, None)],
        "test": [A, ("--out", str, "")], "chat": [A],
    }
    for name, args in spec.items():
        s = sub.add_parser(name)
        for flag, typ, default in args:
            s.add_argument(flag, type=typ, default=default, required=default is None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "spec":
            ix = T.index()
            out = {"replies": data.REPLIES, "reply_format": "[hi <name> , ] REPLIES[intent] [ i added this to ticket d d d d .]",
                   "dev_name_templates": data.DEV_NAME_TPL, "dev_ticket_templates": data.DEV_TICKET_TPL,
                   "T_MAX": model.T_MAX, "max_query_tokens": T.MAX_QUERY, "vocab_size": len(T.vocab()),
                   "special_ids": {s: ix[s] for s in T.SPECIALS},
                   "note": "names and tickets in the hidden test differ from dev; hidden queries are new phrasings"}
        elif a.cmd == "dev":
            exs = data.dev_set()
            if a.ids:
                want = {int(x) for x in a.ids.split(",")}
                exs = [e for e in exs if e["id"] in want]
            out = exs
        elif a.cmd == "tokenize":
            ids = T.encode(a.text)
            out = {"ids": ids, "tokens": [T.vocab()[i] for i in ids]}
        elif a.cmd == "status":
            led = read_ledger(a.agent)
            out = {"runs_budget": led["budget"], "runs_used": led["used"], "runs_left": led["budget"] - led["used"]}
        elif a.cmd == "tokens":
            out = token_status(a.agent)
        elif a.cmd == "check":
            out = cmd_check(a.agent)
        elif a.cmd == "eval":
            out = cmd_eval(a.agent, a.ids, a.note)
        elif a.cmd == "probe":
            out = cmd_probe(a.agent, a.text, a.note)
        elif a.cmd == "init":
            _admin()
            adir(a.agent).mkdir(parents=True, exist_ok=True)
            (adir(a.agent) / ".ledger.json").write_text(json.dumps({"budget": a.budget, "used": 0, "token_budget": a.tokens, "token_base": 0}))
            out = read_ledger(a.agent)
        elif a.cmd == "test":
            _admin()
            out = cmd_test(a.agent, a.out)
        elif a.cmd == "chat":
            chat(a.agent)
            return
    except (BudgetExceeded, FileNotFoundError, AssertionError) as e:
        out = {"error": str(e)}
    print(json.dumps(out, indent=1))
