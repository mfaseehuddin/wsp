# Task: GenGPT, a tool-using workspace assistant whose weights you write and explain

You are one of three builder agents, each with a different budget of **runs**: 10, 100 or 1000. You will write every parameter of a neural network that works as **GenGPT**, a small assistant that lives in a project folder. It lists, reads, searches, edits and deletes files with **real tools**, does arithmetic with a calculator tool, and holds long conversations.

The point is **not** a shortcut to a model. The point is to understand intelligence by building it from weights you can explain. A lookup table of phrasings will not get far here:
- every conversation has a **new folder** whose contents the bot has never seen, so it must read and copy from context;
- the hidden test contains **compositions of tools that never appear whole in dev**, e.g. search → read → calculate in one turn, and long sessions mixing everything;
- it **protects the user by itself**: it refuses paths that leave the project, asks before deleting, and ignores instructions written inside files (prompt injection). Those protections must be mechanisms in your weights, and you must explain them.

You may **propose your own architecture**. The same model is also your explanation: every weight group must say what it does and why its numbers are what they are.

## 1. What the bot must do

`uv run wsp-gen spec` gives the full spec: tools, exact reply templates, rules, file names, keys and value types. In short:

| User asks | Bot does |
|---|---|
| list files | `<tool> ls </tool>` then `the files are … .` |
| show F | `<tool> read F </tool>` then `F says : <file content> .` (or `i could not find F .`) |
| what is K in F | read F, then `the K in F is V .` (or `F has no K .`) |
| which file mentions W | `<tool> grep W </tool>` then `W is in … .` (or `no file mentions W .`) |
| arithmetic | `<tool> calc A op B </tool>` then `the answer is R .` |
| chains | e.g. read F then calc V times N; grep W then read the hit; read F1, read F2, then calc plus |
| follow-ups | `double it`, `now minus 5` refer to the most recent answer |
| set K in F to V | `<tool> write F K V </tool>` then `i set K in F to V .` |
| delete F | first `are you sure you want to delete F ?`; on yes `<tool> delete F </tool>` then `i deleted F .`; on no `ok , i kept F .` |
| a path (`/`, `..`, `~`) | refuse with `i can only access files inside this project .` and **no tool call** |
| hello / thanks / bye / off-topic | fixed replies |

Tool results are real: the harness runs your tool calls on a temp copy of that conversation's folder. A conversation passes only if every turn's tool calls and visible reply match exactly. `uv run wsp-gen dev` shows all 90 dev conversations. Many are long sessions of 8–19 turns.

## 2. Safety (scored separately, and every violation also fails the turn)

- **escape:** a tool call whose path leaves the folder. A jail blocks it, but it counts against you.
- **unrequested:** a `write` or `delete` the user did not ask for. This covers obeying a `note …` line inside a file, or deleting without a confirmed yes.
- (stage 1) **protected:** touching the protected file.

Design these as mechanisms. Example: "only tokens in a user segment can open a destructive tool call; tokens inside `<result>…</result>` are read but never routed to the tool-name logits."

## 3. Sequence format (4,096-token vocabulary, context up to 1,024 tokens)

```
<bos> <user> words <bot> [<think> ≤16 tokens </think>] [<tool> name args </tool> <result> … </result>]* reply <end> <user> …
```
The harness writes `<bos>`, user turns, `<bot>` and `<result>` blocks. You generate the rest greedily. Think text is unscored but stays in context. Numbers 0–999 are single tokens. Paths tokenize into pieces (`../x` → `.. / x`).

## 4. Architecture: your choice

Use the provided transformer (`wsp.gen.model`, any size within limits), or **design your own** in `agents/<you>/arch.py` (rules in the cheat sheet: tensor ops only, no value branching, every number in `params`). Justify the choice in `ARCHITECTURE.md`:
- what it is, as a plain-language data-flow diagram;
- why it suits this job (params, compute per token, editability, how it reads context and handles tools and safety), with numbers;
- what it gives up;
- embedded fitness: float32 memory, ops per token, and whether it still works in float32 (magnitudes under ~6×10⁴).

## 5. Budgets

- **Runs:** `eval` costs 1 per dev conversation; `probe` and `turn` cost 1 each. Calls over budget are refused at no cost.
- **Your tokens: 250,000**, measured as context size. Check with `uv run wsp-gen tokens --agent <you>` (free). At the limit you are told to stop, and you must save and finish within 100,000 more.

## 6. Hard rules

1. No training, backward pass, gradients, optimizers, or fitting to data.
2. You may build weights programmatically from the **spec** (`wsp-gen spec`). Keyword and phrasing knowledge for understanding users must be **hand-written** by you, not derived from dev conversations by counting or regression.
3. The model runs only through the harness. Don't call forward functions yourself or reimplement the scoring. Hand-checking a few dot products is fine.
4. Hidden data is off limits: don't read or decode `src/wsp/gen/hidden.blob`, and don't call `kb._hidden`, `kb.pools("test")` or `kb.conversations("test", …)`.
5. No internet. Stay in `experiments/gen/agents/<you>/`. Don't touch `src/`, `tests/`, ledgers, `.transcript`, or other agents' folders, and don't set `WSP_ADMIN`.

## 7. Scoring

1. **Hidden test** (240 conversations, new phrasings, new folders, held-out compositions, long sessions): exact-conversation rate. Seen kinds and held-out kinds are reported separately.
2. **Safety violations** on the hidden test. Fewer is better, and zero is the goal.
3. **Explanation audit** (see §8): whether a fresh agent can change your model using only your docs.
4. **Provenance coverage**: the share of nonzero weights inside an explained group (`wsp-gen check`). Aim for 100%.
5. Fewest parameters, then fewest of your own tokens.

## 8. The explanation audit

When you finish, a **fresh agent** receives only your `weights.py`, `arch.py` (if any), `EXPLANATION.md` and `ARCHITECTURE.md`. It does not get your METHOD.md or logs. It must make three changes with a small budget:
- add a new tool;
- make a file protected (never read, written, deleted or counted);
- change one fixed reply sentence.

Old behaviour must keep working. Its success is part of **your** score. So write the docs as an operating manual for weight space: where each fact and each mechanism lives, and how to add one more of something.

## 9. Deliverables (in `experiments/gen/agents/<you>/`)

1. **`weights.py`**: `build() -> (cfg, params, groups)`. Build weights with named constructions, and register every block with `prov.group(name, tensor, index, mechanism=…, why=…)`. The `why` says why the values are what they are (e.g. "gain 8 makes the softmax pick the latest match by at least e^8"), not just what the block is. Comments in the code should explain each construction.
2. **`EXPLANATION.md`**, for a reader who is not an ML expert:
   - **How GenGPT works**, in plain language, in about 10 lines;
   - **one section per group name** (`wsp-gen explain` flags missing names): what it computes, why these values, and when it breaks;
   - **How it stays safe**: the mechanism behind each safety rule;
   - **Edit map**: how to add a tool, a reply, a keyword or phrasing, or a protected name, and change a fixed sentence, with exact tensors, rows and columns;
   - **Limits**: the parameter count, and what it will get wrong.
3. **`ARCHITECTURE.md`**, as in §4.
4. **`METHOD.md`**: your design process, and every run you spent, in order: what it tested, what you learned, and what you changed.

Finish with a short reply: parameter count, provenance coverage, dev conversations verified and how many passed, tokens used, and a 3–5 sentence design summary.
