# Roadmap: from hand-built toys to editing real LLMs

The experiments in `experiments/` build small networks from scratch. The next phase works on **small pretrained
LLMs** (starting with Qwen2.5-0.5B). It follows the protocol of Takehi et al., *Sparks of Weight-Space Programming*
(2026): diagnose, localize causally, edit, and verify with paired measurements.

## Track A — Bionic circuits: exact computation grafted into an LLM

Write circuits by hand, as extra neurons or inserted layers. While the signal propagates, the residual stream calls
on a circuit, which computes and writes its result back.

- A0: tooling (`wsp.llm`): load, run, logit lens, activation reads, ablations, and grafts with provenance, saved as
  a standard HF checkpoint.
- A1: baseline. Where does Qwen2.5-0.5B fail at arithmetic (digits, carries, operation)?
- A2: first graft. Exact multi-digit addition as inserted layers. They are identity on other text and checked on
  ordinary text by a perplexity guard.
- A3: other circuits (comparison, counting characters, unit conversion).

## Track B — Extract and transplant learned mechanisms

Hand-inserting facts worked in the toy models. A pretrained model has already learned a good way to store and
retrieve facts (MLP key→value memories, attribute-extraction heads), so we can reuse it.

- B1: localize where facts live (subject → MLP layers → attribute heads), using causal tracing with forward passes only.
- B2: describe the storage format: the keys (subject directions) and values (attribute directions).
- B3: write new facts in that format, by direct rank-one edits computed in closed form. No optimizer.
- B4: transplant. Cut the minimum set of layers or heads needed for a narrow domain into a small model: a support
  bot that runs locally with little or no API cost.

## Track C — Latent function calling

The model calls a tool from inside its forward pass, rather than emitting text that has to be parsed.

- A **port** at layer L has three parts:
  - a trigger direction that tells it when to fire;
  - an argument read, either from residual directions or by attention over the context;
  - a result write, which turns the tool's return value into a residual vector (e.g. the embedding or unembedding
    direction of the answer tokens).
- Pure computations (Track A) can live entirely in weights. Real tools (databases, APIs) are external functions
  behind the port interface. The port's read and write sides are still weights we derive and explain.

## Later

- GitHub Pages with interactive demos (in-browser models, weight explorers).
