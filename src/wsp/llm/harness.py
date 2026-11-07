"""Budgeted harness for grafting hand-derived circuits into a pretrained LLM (Qwen2.5-0.5B).

Agent contract: experiments/llm/agents/<a>/graft.py defines

    def graft(model, tok) -> list[dict]

which edits the HF Qwen2 model IN PLACE (any weight values; new neurons / heads / layers via
wsp.llm.surgery) and returns provenance groups: [{"name", "tensors", "mechanism", "why"}, ...].
The model must not be run inside graft() (module calls are blocked) and nothing may be trained.

The deliverable is a STOCK checkpoint: `build` saves it and reloads it with plain
AutoModelForCausalLM.from_pretrained. Every eval/read/gen command runs that reloaded checkpoint.

Budget: one "pass" = one eval, gen or read call. build, guard, status and tokens are free.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from wsp import progress

from . import data

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("WSP_LLM_ROOT", REPO / "experiments/llm"))
BASE = REPO / "models/Qwen2.5-0.5B"
torch.set_grad_enabled(False)


class BudgetExceeded(RuntimeError):
    pass


# ---------------------------------------------------------------- ledger / tokens


def adir(agent: str) -> Path:
    return ROOT / "agents" / agent


def read_ledger(agent: str) -> dict:
    p = adir(agent) / ".ledger.json"
    if not p.exists():
        raise FileNotFoundError(f"no ledger for {agent!r}")
    return json.loads(p.read_text())


def _write_ledger(agent: str, led: dict) -> None:
    (adir(agent) / ".ledger.json").write_text(json.dumps(led))


def _charge(agent: str) -> int:
    led = read_ledger(agent)
    if led["used"] >= led["budget"]:
        raise BudgetExceeded(f"all {led['budget']} passes used")
    led["used"] += 1
    _write_ledger(agent, led)
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


@contextlib.contextmanager
def _model_lock():
    """One model in RAM at a time (8 GB machine, several agents)."""
    ROOT.mkdir(parents=True, exist_ok=True)
    holder = ROOT / ".model.lock.holder"
    with open(ROOT / ".model.lock", "a") as f:
        waited = 0
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if waited % 30 == 0:
                    who = holder.read_text().strip() if holder.exists() else "unknown"
                    progress.say(f"waiting for model lock ({waited}s) held by: {who}")
                time.sleep(5)
                waited += 5
        holder.write_text(f"pid {os.getpid()}: wsp-llm {' '.join(sys.argv[1:])} (since {time.strftime('%H:%M:%S')})")
        progress.say("model lock acquired")
        try:
            yield
        finally:
            holder.unlink(missing_ok=True)
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------- models


def tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    return tok


def load(path: Path, eager: bool = False):
    from transformers import AutoModelForCausalLM

    kw = {"attn_implementation": "eager"} if eager else {}
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, **kw).eval()


def model_path(agent: str, base: bool) -> Path:
    if base:
        return BASE
    p = adir(agent) / "build"
    if not (p / "config.json").exists():
        sys.exit("no build yet: run `wsp-llm build --agent <you>` first")
    return p


@contextlib.contextmanager
def _no_forward():
    orig_call, orig_bw = torch.nn.Module.__call__, torch.Tensor.backward

    def blocked(*a, **k):
        raise RuntimeError("graft() must not run the model or any module (use the forward-pass tools)")

    torch.nn.Module.__call__ = blocked
    torch.Tensor.backward = blocked
    try:
        yield
    finally:
        torch.nn.Module.__call__, torch.Tensor.backward = orig_call, orig_bw


def cmd_build(agent: str) -> dict:
    from safetensors import safe_open

    gp = adir(agent) / "graft.py"
    spec = importlib.util.spec_from_file_location(f"graft_{agent}", gp)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(gp.parent))
    spec.loader.exec_module(mod)
    tok = tokenizer()
    t0 = time.time()
    with _model_lock():
        m = load(BASE)
        for name, sub in m.named_modules():  # tag modules so the diff survives renames / resizes
            sub._wsp_base = name
        with _no_forward():
            groups = mod.graft(m, tok) or []
        diff, new_params = {}, 0
        with safe_open(BASE / "model.safetensors", "pt") as f:
            base_keys, seen = set(f.keys()), set()
            for name, sub in m.named_modules():
                for leaf, p in sub.named_parameters(recurse=False):
                    if id(p) in seen:  # tied lm_head = embed_tokens
                        continue
                    seen.add(id(p))
                    full = f"{name}.{leaf}" if name else leaf
                    bname = f"{getattr(sub, '_wsp_base', None)}.{leaf}"
                    if bname not in base_keys:
                        nz = int((p != 0).sum())
                        new_params += p.numel()
                        if nz:
                            diff[full] = {"new": list(p.shape), "nonzero": nz}
                        continue
                    b = f.get_tensor(bname).float()
                    region = p[tuple(slice(0, s) for s in b.shape)]
                    changed = int((region != b).sum())
                    extra = int((p != 0).sum()) - int((region != 0).sum())
                    new_params += p.numel() - b.numel()
                    if changed or extra:
                        rec = {"changed": changed, "added_nonzero": extra}
                        if full != bname:
                            rec["base_name"] = bname
                        diff[full] = rec
        out = adir(agent) / "build"
        m.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        del m
        m2 = load(out)  # stock reload check
        n = sum(p.numel() for p in m2.parameters())
        del m2
    (adir(agent) / "build_report.json").write_text(json.dumps({"groups": groups, "diff": diff}, indent=1))
    rep = {
        "ok": True,
        "params": n,
        "added_params": new_params,
        "tensors_touched": len(diff),
        "weights_changed_or_added_nonzero": sum(v.get("changed", 0) + v.get("added_nonzero", v.get("nonzero", 0)) for v in diff.values()),
        "provenance_groups": len(groups),
        "seconds": round(time.time() - t0, 1),
        "report": str(adir(agent) / "build_report.json"),
    }
    _log(agent, {"cmd": "build", **rep})
    return rep


# ---------------------------------------------------------------- generation / eval


def _generate(m, tok, prompts: list[str], max_new: int, bs: int = 64) -> list[str]:
    outs = []
    for i in range(0, len(prompts), bs):
        t0 = time.time()
        enc = tok(prompts[i : i + bs], return_tensors="pt", padding=True)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
        progress.say(f"generate: {len(outs)}/{len(prompts)} prompts done ({time.time() - t0:.1f}s for this batch)")
    return outs


def _parse(text: str) -> str:
    s = text.lstrip(" ")
    k = 0
    while k < len(s) and s[k].isdigit():
        k += 1
    return s[:k]


def run_arith(m, tok, items: list[dict]) -> list[dict]:
    """items: {op,a,b,fmt}. Adds got/want/ok."""
    prompts = [data.prompt(it["op"], it["a"], it["b"], it["fmt"]) for it in items]
    want = [data.answer(it["op"], it["a"], it["b"]) for it in items]
    max_new = max(len(w) for w in want) + 2
    got = [_parse(t) for t in _generate(m, tok, prompts, max_new)]
    progress.say(f"arith: {sum(g == w for g, w in zip(got, want))}/{len(want)} exact")
    return [{**it, "want": w, "got": g, "ok": g == w} for it, w, g in zip(items, want, got)]


def _summ(res: list[dict]) -> dict:
    pos = {}
    for r in res:
        if not r["ok"]:
            w, g = r["want"], r["got"]
            k = next((i for i in range(min(len(w), len(g))) if w[i] != g[i]), min(len(w), len(g)))
            pos[k] = pos.get(k, 0) + 1
    return {
        "acc": round(sum(r["ok"] for r in res) / max(1, len(res)), 4),
        "n": len(res),
        "first_wrong_digit_index": dict(sorted(pos.items())),
        "failures": [f"{r['a']}{data.OPS[r['op']][0]}{r['b']}= want {r['want']} got {r['got']}" for r in res if not r["ok"]][:6],
    }


def cmd_eval(agent, base, op, d1, d2, n, fmt, seed, problems_arg, note) -> dict:
    n = min(n, 256)
    if problems_arg:
        sym = data.OPS[op][0]
        pairs = [tuple(int(x) for x in p.split(sym)) for p in problems_arg.split(",")][:256]
    else:
        pairs = data.problems(op, d1, d2 or d1, n, seed)
    if fmt not in data.DEV_FORMATS:
        sys.exit(f"--format must be one of {data.DEV_FORMATS}")
    k = _charge(agent)
    tok = tokenizer()
    with _model_lock():
        m = load(model_path(agent, base))
        res = run_arith(m, tok, [{"op": op, "a": a, "b": b, "fmt": fmt} for a, b in pairs])
    runs = adir(agent) / "runs"
    runs.mkdir(exist_ok=True)
    with open(runs / f"eval_{k}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in res)
    out = {"pass": k, "model": "base" if base else "edited", **_summ(res), "file": str(runs / f"eval_{k}.jsonl")}
    _log(agent, {"cmd": "eval", "note": note, "args": [op, d1, d2, n, fmt, seed], "acc": out["acc"], "base": base})
    return out


def cmd_gen(agent, base, prompts: list[str], max_new: int, note) -> dict:
    prompts, max_new = prompts[:32], min(max_new, 64)
    k = _charge(agent)
    tok = tokenizer()
    with _model_lock():
        m = load(model_path(agent, base))
        outs = _generate(m, tok, prompts, max_new)
    _log(agent, {"cmd": "gen", "note": note, "base": base, "n": len(prompts)})
    return {"pass": k, "outputs": [{"prompt": p[-80:], "out": o} for p, o in zip(prompts, outs)]}


def cmd_read(agent, base, texts: list[str], layers, mlp_layers, attn_layers, name, note) -> dict:
    texts, mlp_layers = texts[:64], mlp_layers[:4]
    if attn_layers:
        texts, attn_layers = texts[:8], attn_layers[:4]
    k = _charge(agent)
    tok = tokenizer()
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    ids, am = enc.input_ids, enc.attention_mask
    nxt = torch.cat([ids[:, 1:], ids[:, :1]], 1)
    chunks: dict[str, list] = {}
    add = lambda key, v: chunks.setdefault(key, []).append(v)
    with _model_lock():
        m = load(model_path(agent, base), eager=bool(attn_layers))
        cur = {}
        hooks = [
            m.model.layers[li].mlp.down_proj.register_forward_pre_hook(
                lambda _m, inp, li=li: cur.__setitem__(f"mlp_L{li}", inp[0].half().numpy())
            )
            for li in mlp_layers
        ]
        for i in range(0, len(texts), 2):  # small chunks: full-vocab logits are 0.6 GB per text
            o = m(input_ids=ids[i : i + 2], attention_mask=am[i : i + 2],
                  output_hidden_states=True, output_attentions=bool(attn_layers))
            lp = torch.log_softmax(o.logits, -1)
            top = lp.topk(5, -1)
            add("top5_ids", top.indices.numpy())
            add("top5_logprob", top.values.half().numpy())
            add("next_logprob", lp.gather(-1, nxt[i : i + 2, :, None])[..., 0].numpy())
            for li in layers:
                add(f"resid_L{li}", o.hidden_states[li].half().numpy())
            for li in attn_layers:
                add(f"attn_L{li}", o.attentions[li].half().numpy())
            for key, v in cur.items():
                add(key, v)
            del o, lp
        for h in hooks:
            h.remove()
    arrs = {"input_ids": ids.numpy(), "lengths": am.sum(1).numpy(),
            **{key: np.concatenate(v) for key, v in chunks.items()}}
    rd = adir(agent) / "reads"
    rd.mkdir(exist_ok=True)
    np.savez(rd / f"{name}.npz", **arrs)
    _log(agent, {"cmd": "read", "note": note, "base": base, "n": len(texts), "name": name})
    return {"pass": k, "file": str(rd / f"{name}.npz"), "arrays": {key: list(v.shape) for key, v in arrs.items()},
            "note": "right-padded; resid_L{l} = hidden state after layer l (0 = embeddings); mlp_L{l} = down_proj input; "
                    "next_logprob[t] = log p(token t+1 | prefix)"}


def _guard_stats(m, tok, texts: list[str]) -> dict:
    nll, top1 = [], []
    for j, t in enumerate(texts):  # one text at a time: full-vocab logits are large
        if j % 5 == 0:
            progress.say(f"guard: text {j}/{len(texts)}")
        ids = tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids
        lp = torch.log_softmax(m(input_ids=ids).logits[0, :-1], -1)
        nll += (-lp.gather(-1, ids[0, 1:, None])[:, 0]).tolist()
        top1 += lp.argmax(-1).tolist()
    return {"nll": nll, "top1": top1}


def guard(m_path: Path, split: str) -> dict:
    tok = tokenizer()
    texts = data.guard_texts(split)
    import hashlib

    key = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:12]
    cache = ROOT / ".cache" / f"guard_base_{split}_{key}.json"  # keyed by the exact texts
    with _model_lock():
        if not cache.exists():
            progress.say(f"guard: computing base-model reference for split={split} (cached afterwards)")
            cache.parent.mkdir(exist_ok=True)
            cache.write_text(json.dumps(_guard_stats(load(BASE), tok, texts)))
        b = json.loads(cache.read_text())
        e = _guard_stats(load(m_path), tok, texts)
    bn, en = np.mean(b["nll"]), np.mean(e["nll"])
    agree = float(np.mean([x == y for x, y in zip(b["top1"], e["top1"])]))
    return {"tokens": len(b["nll"]), "base_ppl": round(float(np.exp(bn)), 4), "edited_ppl": round(float(np.exp(en)), 4),
            "ppl_ratio": round(float(np.exp(en - bn)), 4), "top1_agreement": round(agree, 4)}


def cmd_test(agent: str, out_path: str | None) -> dict:
    items = []
    for fmt in data.HIDDEN_FORMATS:
        for d in range(1, 13):
            items += [{"op": "add", "a": a, "b": b, "fmt": fmt, "d": f"{d}x{d}"} for a, b in data.problems("add", d, d, 15, 9001)]
        for d1, d2 in [(12, 3), (9, 5), (4, 11), (7, 10), (1, 12), (6, 2)]:
            items += [{"op": "add", "a": a, "b": b, "fmt": fmt, "d": f"{d1}x{d2}"} for a, b in data.problems("add", d1, d2, 10, 9001)]
    tok = tokenizer()
    p = BASE if agent == "base" else adir(agent) / "build"
    progress.say(f"test: phase 1/2 = {len(items)} hidden addition problems")
    with _model_lock():
        res = run_arith(load(p), tok, items)
    progress.say(f"test: phase 1 done, add_acc={_summ(res)['acc']}; phase 2/2 = hidden guard")
    by = lambda key: {k: _summ([r for r in res if r[key] == k])["acc"] for k in dict.fromkeys(r[key] for r in res)}
    out = {"agent": agent, "add_acc": _summ(res)["acc"], "n": len(res), "by_format": by("fmt"), "by_digits": by("d"),
           "guard_hidden": guard(p, "hidden")}
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- CLI


def _admin() -> None:
    if os.environ.get("WSP_ADMIN") != "1":
        sys.exit("admin command: agents must not use this")


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="wsp-llm")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, *args):
        s = sub.add_parser(name)
        for flag, typ, default in args:
            s.add_argument(flag, type=typ, default=default, required=default is None)
        if name in ("eval", "gen", "read"):
            s.add_argument("--base", action="store_true", help="run the unedited model (still costs a pass)")
        return s

    A, N = ("--agent", str, None), ("--note", str, "")
    p("status", A)
    p("tokens", A)
    p("build", A)
    p("guard", A)
    p("eval", A, ("--op", str, "add"), ("--digits", int, 8), ("--digits2", int, 0), ("--n", int, 64),
      ("--format", str, "fewshot"), ("--seed", int, 0), ("--problems", str, ""), N)
    p("gen", A, ("--prompts", str, None), ("--max-new", int, 32), N)
    p("read", A, ("--texts", str, None), ("--layers", str, ""), ("--mlp", str, ""), ("--attn", str, ""),
      ("--name", str, None), N)
    p("init", A, ("--budget", int, None), ("--tokens", int, None))
    p("test", A, ("--out", str, ""))
    a = ap.parse_args(argv)
    if a.cmd not in ("status", "tokens", "init"):
        progress.say(f"log: {progress.start(ROOT, a.cmd, getattr(a, 'agent', None))}")
    try:
        if a.cmd == "status":
            led = read_ledger(a.agent)
            out = {"passes_budget": led["budget"], "passes_used": led["used"], "passes_left": led["budget"] - led["used"]}
        elif a.cmd == "tokens":
            out = token_status(a.agent)
        elif a.cmd == "build":
            out = cmd_build(a.agent)
        elif a.cmd == "guard":
            out = guard(model_path(a.agent, False), "dev")
            _log(a.agent, {"cmd": "guard", **out})
        elif a.cmd == "eval":
            out = cmd_eval(a.agent, a.base, a.op, a.digits, a.digits2, a.n, a.format, a.seed, a.problems, a.note)
        elif a.cmd == "gen":
            prompts = json.loads(Path(a.prompts).read_text()) if Path(a.prompts).exists() else [a.prompts]
            out = cmd_gen(a.agent, a.base, prompts, a.max_new, a.note)
        elif a.cmd == "read":
            texts = json.loads(Path(a.texts).read_text())
            out = cmd_read(a.agent, a.base, texts, _ints(a.layers), _ints(a.mlp), _ints(a.attn), a.name, a.note)
        elif a.cmd == "init":
            _admin()
            adir(a.agent).mkdir(parents=True, exist_ok=True)
            _write_ledger(a.agent, {"budget": a.budget, "used": 0, "token_budget": a.tokens, "token_base": 0})
            out = read_ledger(a.agent)
        elif a.cmd == "test":
            _admin()
            out = cmd_test(a.agent, a.out or None)
    except (BudgetExceeded, FileNotFoundError) as e:
        out = {"error": str(e)}
    if a.cmd not in ("status", "tokens", "init"):
        progress.say("done: " + json.dumps(out)[:400])
    print(json.dumps(out, indent=1))
