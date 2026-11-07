"""Writes EXPLANATION.md sections from the group registry (run: uv run python gen_docs.py)."""
import importlib.util
spec = importlib.util.spec_from_file_location("w", "experiments/gen/agents/g100/weights.py")
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
_, p, groups = w.build()
seen = {}
for g in groups:
    seen.setdefault(g.name, g)
head = open("experiments/gen/agents/g100/EXPL_HEAD.md").read()
out = [head, "\n## Group sections (one per group name)\n"]
for n, g in seen.items():
    where = "tensors " + ", ".join(sorted({x.tensor.split('.h')[0] + ('.h*' if '.h' in x.tensor else '') for x in groups if x.name == n}))
    out.append(f"### {n}\n- Computes: {g.mechanism}.\n- Why these values: {g.why}.\n- Lives in: {where}. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).\n")
open("experiments/gen/agents/g100/EXPLANATION.md", "w").write("\n".join(out))
