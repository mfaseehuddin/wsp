# Task: GenGPT by direct weight output (no code)

You are one of two builder agents, `d100` and `d250` (your number is your **run** budget). You will build **GenGPT**: a small interactive assistant that lives in a project folder. It lists, reads and searches files, edits and deletes them, does arithmetic with a calculator tool, remembers what the conversation is about, and protects the user. Earlier agents built it by writing Python that computed the weights. **You will write the weights themselves, as literal numbers, in a `.wts` file. No code generates them.**

The question this experiment asks: can an LLM "speak weights" directly, and build a network that **generalises to messages it has never seen**?

## 1. Hard rules

1. **No code produces your weights.** You write `experiments/gen/agents/<you>/model.wts` by hand with the Write and Edit tools. You may not write or run scripts, Python, `python -c`, awk/sed programs, shell loops or any other code that computes, generates or transforms weights or weight files. Bash is only for `uv run wsp-gen …` commands and for simple viewing of your own files (`grep`, `head`, `sed -n`, `wc`). Transcripts are audited, and a violation disqualifies you.
2. **Use the fixed transformer** (`wsp.gen.model`). There is no custom architecture (`arch.py`).
3. No training, gradients or fitting. No internet.
4. **Your understanding of users must come from your own language knowledge.** Use `wsp-gen spec` and `wsp-gen dev` for the task's facts. **Never scan, list, grep or read the vocabulary file (`vocab.txt`, `T.vocab()`) or `src/`.** You may look up whether a specific word you thought of exists with `wsp-gen tokenize --text "word"`. Don't read `hidden.blob`, the test or fresh splits, or other agents' folders.
5. Stay in `experiments/gen/agents/<you>/`. Don't touch ledgers or `.transcript`, and don't set `WSP_ADMIN`.

## 2. What the bot must do (stage 2)

`uv run wsp-gen spec --agent <you>` gives the tools, exact reply templates, files, keys, values and rules. `uv run wsp-gen dev --agent <you>` gives the 108 dev conversations. In short: list / show / look up a key / find which file mentions a word / arithmetic with follow-ups ("double it") / chains (find then look up, read then calculate) / set a value / delete with confirmation / refuse paths outside the project / ignore instructions written inside files / **references to earlier turns** ("what is its owner ?", "show it") / greetings, thanks, bye, off-topic. Many dev conversations are long sessions.

## 3. Unseen messages: the main score

You are scored mainly on a **blind test of phrasings you have never seen**: users ask for the same things in other words, with other folders, in held-out combinations. A second blind set will be written after you stop. Dev is only for checking mechanisms. **A model that memorises dev phrasings will fail.** Earlier builders capped out at about 80% because their keyword lists matched dev wording. The best model so far (87% on the blind set) generalised from **the shape of the message**, e.g.:
- a file name plus a key with no other verb means a lookup;
- a person's name plus a key with no file means "find the file that mentions the name, then look up the key";
- a key with no file refers to the file in focus.

Cover **many synonyms from your own knowledge** for each intent ("delete / remove / erase / get rid of / trash", "show / open / display / print / cat / what's in"). Make unknown words harmless: an unrecognised word should not break a message whose structure is clear.

## 4. Budgets: be smart

- **Runs:** `eval` costs 1 per dev conversation, `probe` and `turn` cost 1 each. `check`, `explain`, `spec`, `dev` and `tokenize` are free. Over-budget calls are refused at no cost.
- **Tokens: 300,000**, measured as your context size (`uv run wsp-gen tokens --agent <you>`, free). At the limit you'll be told to stop; finish within 100k more.
- Writing numbers costs output tokens, so **design before you write**:
  - choose a compact residual layout;
  - use ranges, `*`, comma lists and `@token` indices so one line sets many weights;
  - keep most weights zero;
  - write the file in sections, and patch it with Edit rather than rewriting it;
  - use `check` (free) after every edit.
- Spend runs on **batches of diverse dev conversations**, not one at a time, and test with your own unusual phrasings via `turn`.

## 5. Deliverables (in `experiments/gen/agents/<you>/`)

1. **`model.wts`**: the weights. Every `group` header carries a **short reason**: what the block does and why its numbers are what they are (e.g. "gain 10 ⇒ the latest file name wins the softmax by e^10").
2. **`EXPLANATION.md`** (about 1 page): how the bot works in plain language, the residual layout (which channels mean what), how unseen phrasings are handled, how safety works, and how to add a synonym, an intent or a tool.
3. **`METHOD.md`**: your design steps and each batch of runs: what it tested, what you learned, what you changed.

Save a valid `model.wts` early. Whatever is on disk when you stop is what gets scored. Finish with a short reply: params, nonzero, dev conversations run and passed, tokens used, and a 3–5 sentence design summary.
