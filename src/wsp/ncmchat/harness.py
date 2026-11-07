"""Budgeted harness for the NCM chat agent (Track D, chat round). CLI: wsp-ncmchat.

Agent contract: experiments/ncmchat/agents/<a>/weights.py defines build() -> (cfg, params, groups).
Budget: one run = one `eval` (any dev cases), one `chat` (a conversation you write), or one `probe`.
Free: spec, dev, tokenize, check, status, tokens.

Dev cases: experiments/ncmchat/dev.json (Stage X dev without tool cases). Mechanical checks:
contains / any / absent / exact on the reply (case-insensitive substring). Rubric turns are reviewed by a judge.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import torch

from wsp import progress

from . import model

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("WSP_NCMCHAT_ROOT", REPO / "experiments/ncmchat"))
QWEN = REPO / "models/Qwen2.5-0.5B"
PARAM_CAP = 300_000_000
TEXT_UNIT_WORDS = 6
MIN_LEVELS = 1  # the deep round (wsp-ncmdeep) sets 4 and requires a stated meaning per level


class BudgetExceeded(RuntimeError):
    pass


@lru_cache(None)
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN)


def ids(text: str) -> list[int]:
    return tok()(text, add_special_tokens=False).input_ids


def special(name: str) -> int:
    return tok().convert_tokens_to_ids(name)


def user_turn_ids(text: str) -> list[int]:
    return ids(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")


# ---------------------------------------------------------------- ledger


def adir(a: str) -> Path:
    return ROOT / "agents" / a


def read_ledger(a: str) -> dict:
    p = adir(a) / ".ledger.json"
    if not p.exists():
        raise FileNotFoundError(f"no ledger for {a!r}")
    return json.loads(p.read_text())


def _charge(a: str) -> int:
    led = read_ledger(a)
    if led["used"] >= led["budget"]:
        raise BudgetExceeded(f"all {led['budget']} runs used")
    led["used"] += 1
    (adir(a) / ".ledger.json").write_text(json.dumps(led))
    return led["used"]


def _log(a: str, rec: dict) -> None:
    with open(adir(a) / "log.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), **rec}) + "\n")


def token_status(a: str) -> dict:
    from wsp.bot.harness import context_tokens

    led = read_ledger(a)
    tf = adir(a) / ".transcript"
    now = context_tokens(tf.read_text().strip()) if tf.exists() else 0
    used = max(0, now - led.get("token_base", 0))
    return {"tokens_used": used, "token_budget": led["token_budget"], "remaining": led["token_budget"] - used}


# ---------------------------------------------------------------- model


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"ncmchat_w_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    orig = torch.Tensor.backward

    def blocked(*a, **k):
        raise RuntimeError("no gradients: weights must be written, not trained")

    torch.Tensor.backward = blocked
    try:
        with torch.no_grad():
            spec.loader.exec_module(mod)
            cfg, P, groups = mod.build()
    finally:
        torch.Tensor.backward = orig
        sys.path.pop(0)
    P = {k: torch.as_tensor(v).to(model.DTYPE) for k, v in P.items()}
    model.validate(cfg, P)
    if MIN_LEVELS > 1:
        assert len(cfg.levels) >= MIN_LEVELS, f"this round needs K >= {MIN_LEVELS} levels"
        assert len(cfg.level_meanings) == len(cfg.levels) and all(m.strip() for m in cfg.level_meanings), \
            "give every level a stated meaning in cfg.level_meanings"
    n = sum(v.numel() for v in P.values())
    assert n <= PARAM_CAP, f"{n:,} params > cap {PARAM_CAP:,}"
    return cfg, P, groups or []


class Chat:
    def __init__(self, cfg, P):
        self.cfg, self.P, self.cache = cfg, P, model.Cache(cfg)
        self.end = special("<|im_end|>")

    def turn(self, text: str, trace: bool = False) -> dict:
        model.extend(self.cfg, self.P, self.cache, user_turn_ids(text))
        self.prefix_len = len(self.cache)
        out = model.reply(self.cfg, self.P, self.cache, trace)
        toks = out["tokens"]
        toks = toks[: toks.index(self.end)] if self.end in toks else toks
        model.extend(self.cfg, self.P, self.cache, toks + [self.end] + ids("\n"))
        return {"reply": tok().decode(toks), "out": out}


def check_turn(reply: str, rule: dict) -> bool | None:
    if not rule or set(rule) <= {"no_tools"}:
        return None
    r, c = reply.lower(), []
    if "exact" in rule:
        c.append(reply.strip() == rule["exact"])
    if "contains" in rule:
        c += [s.lower() in r for s in rule["contains"]]
    if "any" in rule:
        c.append(any(s.lower() in r for s in rule["any"]))
    if "absent" in rule:
        c += [s.lower() not in r for s in rule["absent"]]
    return all(c)


def run_cases(cfg, P, cases: list[dict]) -> list[dict]:
    res = []
    for j, case in enumerate(cases):
        done = [t["mechanical"] for r in res for t in r["turns"] if t["mechanical"] is not None]
        progress.say(f"case {j + 1}/{len(cases)} {case['id']} ({len(case['turns'])} turns); mechanical so far {sum(done)}/{len(done)}")
        ch, turns = Chat(cfg, P), []
        for t in case["turns"]:
            got = ch.turn(t["user"])["reply"]
            turns.append({"user": t["user"], "reply": got, "rubric": t.get("rubric", ""),
                          "mechanical": check_turn(got, t.get("check", {}))})
        res.append({"id": case["id"], "category": case["category"], "turns": turns})
    return res


def summarize(res: list[dict], full: bool = True) -> dict:
    mech = [t["mechanical"] for r in res for t in r["turns"] if t["mechanical"] is not None]
    out = {"mechanical_pass": sum(mech), "mechanical_checked": len(mech),
           "turns": sum(len(r["turns"]) for r in res)}
    if full:
        out["cases"] = [{"id": r["id"], "category": r["category"],
                         "turns": [{"user": t["user"][:120], "reply": t["reply"][:300], "pass": t["mechanical"]}
                                   for t in r["turns"][:8]] + ([{"note": f"{len(r['turns']) - 8} more turns"}] if len(r["turns"]) > 8 else [])}
                        for r in res]
    return out


def long_literals(src: str) -> list[str]:
    lits = re.findall(r'"([^"\n]{20,})"|\'([^\'\n]{20,})\'', src)
    return [s for pair in lits for s in pair if s and len(s.split()) > TEXT_UNIT_WORDS][:10]


# ---------------------------------------------------------------- commands


def _cases(ids_arg: str) -> list[dict]:
    cases = json.loads((ROOT / "dev.json").read_text())
    if ids_arg:
        want = set(ids_arg.split(","))
        cases = [c for c in cases if c["id"] in want]
    return cases


def cmd_check(a: str) -> dict:
    t0 = time.time()
    cfg, P, groups = load(adir(a) / "weights.py")
    src = "".join(p.read_text() for p in adir(a).glob("*.py"))
    return {"ok": True, "config": cfg.to_dict(), "params": sum(v.numel() for v in P.values()),
            "nonzero": sum(int((v != 0).sum()) for v in P.values()), "provenance_groups": len(groups),
            "forward_passes_per_reply": len(cfg.levels), "build_seconds": round(time.time() - t0, 1),
            "reused_qwen_layers": list(cfg.qwen_layers), "reused_qwen_params": 14_912_384 * len(cfg.qwen_layers)
            + (136_134_656 if cfg.qwen_embed else 0), "level_meanings": list(cfg.level_meanings),
            "long_text_literals (> 6 words, not allowed)": long_literals(src)}


def cmd_eval(a, ids_arg, note):
    cfg, P, _ = load(adir(a) / "weights.py")
    k = _charge(a)
    res = run_cases(cfg, P, _cases(ids_arg))
    out = {"run": k, **summarize(res)}
    _log(a, {"cmd": "eval", "ids": ids_arg, "note": note, "pass": out["mechanical_pass"], "of": out["mechanical_checked"]})
    return out


def cmd_chat(a, turns: list[str], note):
    cfg, P, _ = load(adir(a) / "weights.py")
    k = _charge(a)
    ch = Chat(cfg, P)
    conv = [{"user": t, "bot": ch.turn(t)["reply"]} for t in turns[:12]]
    _log(a, {"cmd": "chat", "note": note, "n": len(conv)})
    return {"run": k, "conversation": conv}


def cmd_probe(a, turns: list[str], note):
    cfg, P, _ = load(adir(a) / "weights.py")
    k = _charge(a)
    ch = Chat(cfg, P)
    for t in turns[:-1]:
        ch.turn(t)
    got = ch.turn(turns[-1], trace=True)
    end = special("<|im_end|>")
    levels = []
    for l, L in enumerate(got["out"]["levels"]):
        tk = model.decode_latent(cfg, P, L["z"]).argmax(-1).tolist()
        tk = tk[: tk.index(end)] if end in tk else tk
        a_last = L["attn"][-1].mean(0)  # [n_l, prefix+n_l]
        top = a_last[:, : ch.prefix_len].argmax(-1).tolist()
        levels.append({"level": l, "n": cfg.levels[l], "codes": got["out"]["codes"][l][:64],
                       "decoded_so_far": tok().decode(tk),
                       "last_layer_top_prefix_token": [tok().decode([ch.cache.ids[i]]) for i in top[:16]]})
    _log(a, {"cmd": "probe", "note": note})
    return {"run": k, "reply": got["reply"], "levels": levels}


def cmd_test(a, cases_path, out_path):
    cfg, P, _ = load(adir(a) / "weights.py")
    res = run_cases(cfg, P, json.loads(Path(cases_path).read_text()))
    out = {"agent": a, **summarize(res), "params": sum(v.numel() for v in P.values())}
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
    return out


def interactive(a):
    cfg, P, _ = load(adir(a) / "weights.py")
    ch = Chat(cfg, P)
    print("NCM chat. Ctrl-D to quit.")
    for line in sys.stdin:
        print("bot:", ch.turn(line.strip())["reply"])


def _admin():
    if os.environ.get("WSP_ADMIN") != "1":
        sys.exit("admin command: agents must not use this")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsp-ncmchat")
    sub = ap.add_subparsers(dest="cmd", required=True)
    A, N = ("--agent", str, None), ("--note", str, "")
    spec = {"spec": [], "dev": [("--ids", str, "")], "tokenize": [("--text", str, None)],
            "status": [A], "tokens": [A], "check": [A], "eval": [A, ("--ids", str, ""), N],
            "chat": [A, ("--turns", str, None), N], "probe": [A, ("--turns", str, None), N],
            "init": [A, ("--budget", int, None), ("--tokens", int, None)],
            "test": [A, ("--cases", str, None), ("--out", str, "")], "talk": [A]}
    for name, args in spec.items():
        s = sub.add_parser(name)
        for flag, typ, default in args:
            s.add_argument(flag, type=typ, default=default, required=default is None)
    a = ap.parse_args(argv)
    if a.cmd in ("eval", "probe", "test", "chat"):
        progress.say(f"log: {progress.start(ROOT, a.cmd, getattr(a, 'agent', None))}")
    turns = lambda s: json.loads(Path(s).read_text()) if Path(s).exists() else [s]
    try:
        if a.cmd == "spec":
            out = {"T_MAX": model.T_MAX, "vocab (config.vocab = Qwen embedding rows)": 151936, "param_cap": PARAM_CAP,
                   "chat_format": "<|im_start|>user\\n{msg}<|im_end|>\\n<|im_start|>assistant\\n{reply}<|im_end|>\\n",
                   "special_ids": {s: special(s) for s in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")},
                   "qwen_weights": str(QWEN / "model.safetensors"),
                   "dev_cases": len(_cases("")), "text_unit_limit_words": TEXT_UNIT_WORDS}
        elif a.cmd == "dev":
            out = _cases(a.ids)
        elif a.cmd == "tokenize":
            t = ids(a.text)
            out = {"ids": t, "pieces": [tok().decode([i]) for i in t]}
        elif a.cmd == "status":
            led = read_ledger(a.agent)
            out = {"runs_budget": led["budget"], "runs_used": led["used"], "runs_left": led["budget"] - led["used"]}
        elif a.cmd == "tokens":
            out = token_status(a.agent)
        elif a.cmd == "check":
            out = cmd_check(a.agent)
        elif a.cmd == "eval":
            out = cmd_eval(a.agent, a.ids, a.note)
        elif a.cmd == "chat":
            out = cmd_chat(a.agent, turns(a.turns), a.note)
        elif a.cmd == "probe":
            out = cmd_probe(a.agent, turns(a.turns), a.note)
        elif a.cmd == "init":
            _admin()
            adir(a.agent).mkdir(parents=True, exist_ok=True)
            (adir(a.agent) / ".ledger.json").write_text(json.dumps({"budget": a.budget, "used": 0, "token_budget": a.tokens, "token_base": 0}))
            out = read_ledger(a.agent)
        elif a.cmd == "test":
            _admin()
            out = cmd_test(a.agent, a.cases, a.out)
        elif a.cmd == "talk":
            interactive(a.agent)
            return
    except (BudgetExceeded, FileNotFoundError, AssertionError) as e:
        out = {"error": str(e)}
    progress.say("done: " + json.dumps(out)[:300])
    print(json.dumps(out, indent=1))
