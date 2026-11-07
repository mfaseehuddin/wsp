# wsp — weight space programming

**Writing neural-network weights by understanding them, not by training them.**

**Live results: <https://mfaseehuddin.github.io/wsp/>** · Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md)

This repo is meant as a starting point. It holds harnesses, agent prompts, results and ideas for anyone who wants to
point an LLM agent (Claude Code, Codex, …) at a network's weights and have it **explain** them, **write** them from
scratch, or **graft new circuits** into them. Every experiment here was run by LLM agents working through a CLI
harness, with token budgets, blind hidden tests and audits.

## Credit

This work started from, and models its agent prompts on, the following post:

> **Rikiya Takehi\*, Jayden Teoh\*, Alex Zhang, Omar Khattab** (MIT CSAIL).
> *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* November 2026.
> Blog: <https://rikiyat.github.io/blog/weight-space-programming/> · Logs: <https://github.com/RikiyaT/lm_understand-logs>

From their work we took the framing (an LLM reasons about what each part of a network computes, then edits weights to
match, with no optimizer), the forward-pass budget as the main experimental knob, and the protocol of their prompts:
diagnose first, localize causally, measure with paired comparisons, explain every step, and no gradients, training or
simulating the network outside the tool. Their logs are CC BY 4.0. Please cite them if you build on this.

`docs/research/weight-space-programming-survey.md` surveys the wider prior art (RASP, Tracr, ALTA, CNN
constructions, interpretability).

## Quickstart

```bash
uv sync                 # Python 3.12 + PyTorch (CPU, CUDA or Apple MPS)
uv run pytest           # harness tests
uv run wsp-gen --help   # e.g. the GenGPT harness
```

## What's inside

```
src/wsp/<experiment>/      harness + reference task + fixed model forward pass (one package per experiment)
experiments/<experiment>/  TASK.md (agent prompt), CHEATSHEET.md, results, per-agent work in agents/<name>/
experiments/_report/       an HTML report of the runs so far
.claude/agents/            wsp-builder: a lean agent type (Bash/Read/Write/Edit only) for running experiments
```

| Experiment | CLI | What agents do | Headline result |
|---|---|---|---|
| `pong` | `wsp-pong` | Write a small ReLU policy that plays Pong, with 0 to 2,000 forward passes | The 0-pass agent's policy hits every ball in 5,000 test rallies |
| `support` | `wsp-support` | Write a tiny transformer for an IT help desk (14 intents), with 0, 10 or 50 passes | 97% exact replies on the test set with 8.6k parameters |
| `bot` | `wsp-bot` | A multi-turn, tool-calling chatbot; run budgets of 1 to 1,000 | 64–83% exact conversations on a hidden test |
| `gen` | `wsp-gen` | **GenGPT**: a workspace assistant with real sandboxed tools; safety (path escape, delete confirmation, prompt injection) lives in the weights. Includes explanation **audits** (a fresh agent must extend the model using only the builder's docs) and a **direct** variant where weights are written as literal numbers (`.wts`), with no code | 79–81% hidden test, 0 safety violations; audits fixed new tasks with 0 regressions; direct no-code weights reached 45–60% on a blind split |
| `llm` *(running)* | `wsp-llm` | **Track A, bionic circuits:** graft an exact adder into **Qwen2.5-0.5B** with inert-capacity surgery (new neurons, heads, layers). The result must stay a stock HF checkpoint and leave ordinary text untouched (perplexity guard) | The unedited model gets 27.4% on the hidden test (0% from 9 digits); agent results pending |
| `stage_x` | `wsp-x` | A general chat model in a fixed standard decoder transformer, up to 20M params, no canned answers | An honest negative result: rule machines and retrieve-and-copy, not general chat |

`experiments/gen/RESULTS.md` and `experiments/_report/index.html` have the details. Large weight files (`*.npz` over
GitHub's comfortable size, `.artifacts/`) are not committed. Rebuild them from the agents' `weights.py` files.

## Running your own agent experiment

The pattern every experiment here follows, which you can copy:

1. **A task prompt** (`TASK.md`): the goal, the fixed architecture or the freedom to choose one, the hard rules (no
   gradient descent, fitting or training; no reading hidden data), the budget, and what to deliver: weights plus an
   explanation of every weight group.
2. **A cheat sheet** (`CHEATSHEET.md`) with only the commands and formats the agent needs. Every file an agent reads
   stays in its context, and context size is its token budget.
3. **A harness CLI** that agents use for everything: build, run a forward pass (counted against the budget), score on
   dev, chat. Admin commands (hidden test, freezing) sit behind `WSP_ADMIN=1`.
4. **Budgets as the independent variable**: forward passes or runs, plus a token cap watched by `token_watch.py`.
5. **Blind scoring**: freeze the weights first. An independent agent writes the hidden test afterwards.
6. **Audit the explanation**: a fresh agent gets only the builder's docs and must extend the model. If it can't, the
   explanation was not good enough.

To run one with Claude Code, spawn a `wsp-builder` agent and point it at `experiments/<name>/CHEATSHEET.md`.

## Now: editing real LLMs

Three tracks on small pretrained models (details in [`docs/ROADMAP.md`](docs/ROADMAP.md)):

- **A · Bionic circuits** *(running)*: hand-derived circuits grafted into the LLM, which the residual stream calls on
  mid-forward-pass.
- **B · Transplant learned memory**: find how a pretrained model stores facts, write new ones in its own format with
  closed-form edits, and cut out a small, cheap local support bot.
- **C · Latent function calling**: a port inside the forward pass (trigger → read arguments → tool → write the
  result back as a vector), instead of emitting text and parsing it.

`src/wsp/llm/surgery.py` holds reusable helpers that add inert capacity to a Qwen2 checkpoint (`insert_layer`,
`widen_mlp`, `add_head_group`). The model stays loadable by stock `transformers`, and it computes exactly the same
function until you write values.

## Ideas to try

- **More bionic circuits**: comparison, counting characters (the "strawberry" problem), unit conversion, dates.
- **Repair a failure mechanism** in a small LLM (loops, refusals, a specific factual error), in the style of the
  post's Qwen3.5-0.8B math edit: diagnose, localize causally, edit, then a paired evaluation.
- **Read weights blind**: which task or dataset was this network trained on? Where are its induction heads?
- **Explanation audits** for any constructed model: can a second agent extend it using only the first agent's docs?
- **Speak weights directly**: no code, only literal numbers. How far does it generalise?
