# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack and commands

Python 3.12 and PyTorch, managed with **uv** (src layout: `src/wsp/`). Development machine is an Apple M1, so GPU work runs on the **MPS** backend.

```bash
uv sync                                   # install deps into .venv
uv run pytest                             # all tests
uv run pytest tests/test_foo.py::test_bar # single test
uv run ruff check . && uv run ruff format .
uv add <pkg> / uv add --dev <pkg>         # never pip install directly
```

MPS notes:
- Select the device with `torch.device("mps" if torch.backends.mps.is_available() else "cpu")`. Code must also run on CPU.
- MPS has **no float64**. Build weights and check exactness in float64 on CPU, then cast to float32 for MPS. Don't expect bit-exact results on MPS. Verification should use tolerances there, or compare against the CPU float64 result.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` lets ops that MPS doesn't support fall back to CPU.

## What wsp is

**wsp (weight space programming)** is about *understanding weights well enough to write them*. The core questions, for any network (MLP, CNN, Transformer):

- **Why** are the weights in layer N the way they are?
- **What mechanism** do they implement, and when does that mechanism apply?
- **How** can that understanding be turned around to **generate** weights directly for a task, without training?

Explaining weights and generating weights are the same theory used in two directions. A mechanism is only understood once we can derive its weights from first principles, and a generated network is only acceptable once every weight has an explanation. Goals:

1. **Explain.** For each layer and weight group, give the mechanism it performs and the reason for its values (e.g. "these layer-2 rows are oriented edge detectors built from layer-1 features; the bias sets the activation threshold").
2. **Generate.** Write working CNN and Transformer weights directly from those derived mechanisms. This skips training, which is limited by the compute available on the device.
3. **LLM Programming Language (later).** An LLM turns natural-language intent into a spec of mechanisms, which is then built into weights.

Explanations are the primary output, not a by-product. Code should make them easy to produce and inspect.

## Prior art

`docs/research/weight-space-programming-survey.md` surveys the formal work (RASP, Tracr, ALTA, CNN constructions, interpretability). Reuse existing tools where they serve the goal, e.g. Tracr-compiled models as ground-truth examples whose mechanisms are known. Tracr itself compiles programs without explaining *why* the resulting weights work, so it is a reference and not the center of the project.

## Intended architecture (conceptual)

The project is organized as a pipeline. Keep these layers separate so each one can be inspected and tested on its own:

```
user intent (natural language)
   → LLM front-end: intent → structured task/program spec
   → weight-space compiler: spec → layer-by-layer weight construction
        ├─ CNN constructors (e.g. hand-derived conv kernels / filters)
        └─ Transformer constructors (e.g. attention patterns + MLPs that implement explicit operations)
   → runnable model (standard framework tensors)
   → verification: does the generated network actually perform the task?
```

Principles for code in this repo:
- **Traceability.** Generated weights should be traceable back to the spec element that produced them, e.g. through named components or metadata. Don't emit opaque tensors with no provenance.
- **Verification as a first-class step.** Every constructor needs an executable check that the generated network computes what the spec says, compared against a reference implementation of the task.
- **Training is optional.** Gradient-based fine-tuning may be used to compare against or refine constructed weights. It must not be the main way a network gets its weights.

## Running agent experiments

The experiments live in `experiments/<name>/`. Each has a `TASK.md` that gives agents the rules, an agent-facing harness CLI (`wsp-pong`, `wsp-support`, `wsp-bot`), and per-agent folders under `agents/`. The admin commands need `WSP_ADMIN=1`, and agents must never use them.

- **Spawn experiment agents with the `wsp-builder` agent type** (`.claude/agents/wsp-builder.md`). It has only Bash, Read, Write and Edit, so its baseline context is much smaller than general-purpose.
- **Point agents to `experiments/bot/CHEATSHEET.md`** rather than the source files. A file an agent reads stays in its context for every later call, and context size is the token budget.
- **Track tokens and enforce limits:** write each agent's transcript path to `agents/<a>/.transcript` after spawning it. Agents check `wsp-bot tokens --agent <a>`, and `experiments/bot/token_watch.py` (run under Monitor) flags when an agent reaches its limit and its limit plus 100k.
- **5-hour usage limits:** running many agents in parallel hits them quickly, so stagger agents. A stopped agent keeps its saved files and can be resumed with SendMessage.
