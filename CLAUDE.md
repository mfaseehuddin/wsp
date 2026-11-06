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

## Prior art: reuse before building

The user wants to avoid reinventing existing work. RASP and Tracr (compiling RASP programs into transformer weights) are the starting point. `docs/research/weight-space-programming-survey.md` surveys the formal work. Read it before designing a new spec language or compiler, and prefer reusing or porting existing tools (e.g. converting Tracr output into PyTorch modules) over rewriting them.

## What wsp is

**wsp (weight space programming)** builds neural networks by *generating their weights programmatically* rather than training them. The goals are:

1. **Constructed, not trained.** Produce working CNNs and Transformers whose weights are written directly to perform a specific task. This skips gradient-based training, which is limited by the compute available on hand or on the device.
2. **Open the black box.** Every weight should be explainable: why it has its value and what computation it implements. Interpretability is a primary output, not a side effect.
3. **LLM Programming Language.** An LLM reads the user's intent in natural language and turns it into a network specification that is then compiled into weights. The result is a network that comes out "pretrained" for the job.

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
