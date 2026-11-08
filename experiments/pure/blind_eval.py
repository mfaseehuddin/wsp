"""Blind evaluation of the chat models on experiments/pure/hidden.json (admin only).

    uv run python experiments/pure/blind_eval.py run      # run every model, save raw replies (live log in blind/)
    uv run python experiments/pure/blind_eval.py pack     # anonymised + shuffled judge file, key kept separately
    uv run python experiments/pure/blind_eval.py score    # merge the judge's verdicts with the key -> results

Models: the four frozen pw100 checkpoints, nco60 (ncmchat) and ncd60 (ncmdeep).
"""

import json
import random
import sys
from pathlib import Path

from wsp import progress

ROOT = Path(__file__).parent
REPO = ROOT.parents[1]
OUT = ROOT / "blind"
HIDDEN = ROOT / "hidden.json"
MODELS = [f"pw100_p{p:03d}" for p in (10, 30, 50, 100)] + ["nco60", "ncd60"]


def run_model(name, cases):
    if name.startswith("pw100"):
        from wsp.pure import harness as H

        family, cfg, P, _, _ = H.build("pw100", ROOT / "frozen" / f"{name}.py")
        res = H.run_cases(family, cfg, P, cases)
        return [{"id": r["id"], "turns": [{"user": t["user"], "reply": t["reply"], "mech": t["pass"]} for t in r["turns"]]} for r in res]
    from wsp.ncmchat import harness as C

    if name == "ncd60":
        C.ROOT = REPO / "experiments/ncmdeep"
    else:
        C.ROOT = REPO / "experiments/ncmchat"
    cfg, P, _ = C.load(C.adir(name) / "weights.py")
    out = []
    for j, case in enumerate(cases):
        progress.say(f"{name}: case {j + 1}/{len(cases)} {case['id']}")
        ch = C.Chat(cfg, P)
        turns = []
        for t in case["turns"]:
            rep = ch.turn(t["user"])["reply"]
            turns.append({"user": t["user"], "reply": rep, "mech": C.check_turn(rep, t.get("check", {}))})
        out.append({"id": case["id"], "turns": turns})
    return out


def cmd_run(only=None):
    OUT.mkdir(exist_ok=True)
    progress.start(OUT, "run", "all")
    cases = json.loads(HIDDEN.read_text())
    for m in only or MODELS:
        f = OUT / f"raw_{m}.json"
        if f.exists():
            progress.say(f"{m}: already done")
            continue
        progress.say(f"{m}: running {len(cases)} conversations")
        res = run_model(m, cases)
        f.write_text(json.dumps(res, indent=1))
        mech = [t["mech"] for c in res for t in c["turns"] if t["mech"] is not None]
        progress.say(f"{m}: done, mechanical {sum(mech)}/{len(mech)}")
        import gc

        gc.collect()


def cmd_pack(seed=20261108):
    cases = {c["id"]: c for c in json.loads(HIDDEN.read_text())}
    items, key = [], {}
    for m in MODELS:
        for conv in json.loads((OUT / f"raw_{m}.json").read_text()):
            spec = cases[conv["id"]]
            for i, t in enumerate(conv["turns"]):
                ctx = [f"U: {x['user']}\nB: {x['reply'][:200]}" for x in conv["turns"][max(0, i - 3) : i]]
                items.append({"user": t["user"], "reply": t["reply"], "rubric": spec["turns"][i]["rubric"],
                              "recent_turns": ctx, "_k": (m, conv["id"], i)})
    uniq = {}  # identical (context, reply, rubric) -> judged once; the verdict is shared
    for it in items:
        k = it.pop("_k")
        sig = json.dumps([it["user"], it["reply"], it["rubric"], it["recent_turns"]])
        uniq.setdefault(sig, (it, []))[1].append(k)
    items = [it for it, _ in uniq.values()]
    keys = [ks for _, ks in uniq.values()]
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    items = [items[i] for i in order]
    keys = [keys[i] for i in order]
    for n, (it, ks) in enumerate(zip(items, keys)):
        iid = f"item{n:04d}"
        key[iid] = ks
        it["item"] = iid
    (OUT / "judge_items.json").write_text(json.dumps(items, indent=1))
    (OUT / "KEY_do_not_give_to_judge.json").write_text(json.dumps(key))
    print(len(items), "items")


def cmd_score():
    key = json.loads((OUT / "KEY_do_not_give_to_judge.json").read_text())
    verdicts = json.loads((OUT / "verdicts.json").read_text())  # {item: {"ok": 0|1, "why": "..."}}
    cases = {c["id"]: c for c in json.loads(HIDDEN.read_text())}
    agg = {m: {"ok": 0, "n": 0, "by_cat": {}} for m in MODELS}
    for iid, ks in key.items():
        v = verdicts.get(iid)
        if v is None:
            continue
        for m, cid, _i in ks:
            _add(agg, cases, m, cid, v)
    return _finish(agg)


def _add(agg, cases, m, cid, v):
    cat = cases[cid]["category"]
    a = agg[m]
    a["ok"] += int(v["ok"])
    a["n"] += 1
    c = a["by_cat"].setdefault(cat, [0, 0])
    c[0] += int(v["ok"])
    c[1] += 1


def _finish(agg):
    for m in MODELS:
        raw = json.loads((OUT / f"raw_{m}.json").read_text())
        mech = [t["mech"] for c in raw for t in c["turns"] if t["mech"] is not None]
        agg[m]["mechanical"] = [sum(mech), len(mech)]
        agg[m]["judged_acceptable"] = round(agg[m]["ok"] / max(1, agg[m]["n"]), 4)
    (OUT / "results.json").write_text(json.dumps(agg, indent=1))
    for m in MODELS:
        a = agg[m]
        print(f"{m:12s} judged {a['ok']}/{a['n']} = {a['judged_acceptable']:.1%}   mechanical {a['mechanical'][0]}/{a['mechanical'][1]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"run": lambda: cmd_run(sys.argv[2:] or None), "pack": cmd_pack, "score": cmd_score}[cmd]()
