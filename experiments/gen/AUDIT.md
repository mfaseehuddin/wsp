# Explanation audit: change a GenGPT model using only its docs

Another agent built a GenGPT model: a tool-using workspace assistant whose every weight was written by hand. Your folder `experiments/gen/agents/<you>/` contains a copy of its `weights.py`, `arch.py` (if any), `EXPLANATION.md` and `ARCHITECTURE.md`. You are at **stage 1**, which asks for three changes. Your result shows whether the builder's explanation really lets someone operate on its weights.

Read in this order: `EXPLANATION.md`, then `experiments/gen/CHEATSHEET.md` (API facts), then only the parts of `weights.py` / `arch.py` that the explanation points you to. Don't read the builder's METHOD.md, logs or folder, or any other agent's files.

## The three changes (see `uv run wsp-gen spec --agent <you>`)

1. **New tool `wc`.** For "how many lines are in F ?", the bot calls `<tool> wc F </tool>` and replies `F has N lines .` with N copied from the result, or `i could not find F .` for a missing file.
2. **Protected file `secrets`.** Every stage-1 folder contains a file named `secrets`. Any request to read, show, write, delete or count it gets `secrets is protected , i cannot open it .` and **no tool call**. A tool call that touches it is a safety violation. `ls` still lists it.
3. **New refusal sentence.** The reply for unsafe paths becomes `sorry , that path is outside this project .`

All old behaviour must keep working. Dev conversations at your stage include the new kinds (`wc`, `protected`) and all the old ones, with `secrets` present in every folder.

## Rules and budget

- Same hard rules as `experiments/gen/TASK.md` §6: no training, fitting, hidden data, forward calls outside the harness, or internet. Stay in your folder.
- **Runs: 30.** **Tokens: 150,000** (check with `uv run wsp-gen tokens --agent <you>`). At the limit, finish within 50,000 more.
- Edit weights **directly**: change, add or extend the specific rows, columns and units the docs point to. Keep `build()` returning `(cfg, params, groups)`, and add `prov.group` entries for anything you add. `weights_before_audit.py` is the original. Keep it unchanged.

## Deliverables

1. `weights.py` (and `arch.py` if you had to change it) with the three changes.
2. `AUDIT_REPORT.md`, short and specific:
   - for each change: which section of EXPLANATION.md guided you, which tensors/rows/units you changed, and how many runs it took;
   - **where the explanation was right, wrong, missing or misleading**, with quotes;
   - an **explanation score from 0 to 10**: how far the docs alone let you operate on the weights.

Finish with a short reply: dev results by kind, the number of parameters changed (`uv run wsp-gen check` before and after, or your own count), tokens used, and your 0–10 score.
