# Task: graft an exact adder into a pretrained LLM (bionic circuits, round 1)

*The protocol follows Takehi, Teoh, Zhang & Khattab, "Sparks of Weight-Space Programming" (2026): diagnose,
localize causally, edit, and verify with paired measurements. No training.*

You are one of several agents. Each of you designs your own system.

## The model

**Qwen2.5-0.5B** (base, not instruct), read-only at `models/Qwen2.5-0.5B/`:
- 24 decoder layers, width 896.
- Attention: 14 query heads and 2 key/value heads (GQA: query head h uses kv head h // 7), head size 64, biases on
  q/k/v, RoPE with θ = 1e6 (rotate-half convention: dimension i pairs with i + 32).
- SwiGLU MLP of width 4,864 (`down(silu(gate x) * up x)`), pre-RMSNorm.
- Input embedding and output layer are **tied**. Vocabulary 151,936.
- **Every digit is its own token**: `482913+305127=` is 14 tokens.

## The goal

The model fails at long addition. Unedited, with a 3-shot prompt and greedy decoding:

| digits | 2 | 3 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| exact `a+b=` | 98% | 100% | 85% | 50% | 5% | 0% |

When it fails, the wrong digit is usually off by one, in the middle or lower places.

**Build circuits into the weights** so that, as the signal moves through the network, the circuit computes the
sum and the model writes the exact answer, MSB first, after `=`. Target: exact addition of numbers up to
**12 digits**, of equal or unequal length. On all other text the model must behave as before: the circuit must
stay silent when it is not needed.

How you do this is up to you. Some possibilities:
- repair the model's own addition mechanism (find what goes wrong and fix it);
- read the model's own digit and position features and add new neurons or heads on top of them;
- write a fresh circuit from the embeddings (new inserted layers, new heads, new neurons) that does the place-value
  alignment and the carries;
- a mix of these.

Find out what the network already does before you decide.

## How you will be measured (hidden, never shown to you)

- **Addition accuracy:** `a+b=` with a, b of 1–12 digits, equal and unequal lengths. Prompt formats: the two dev
  formats (`fewshot`, `zeroshot`) plus two unseen ones (other few-shot lines before the problem, different
  context lengths). Always the `a+b=` form, with no spaces. Greedy decoding. Exact match of the digits generated
  after `=`.
- **Guard:** perplexity and top-1 agreement against the unedited model on unseen ordinary text: Python code
  (which contains `a+b`), prose with numbers, documentation. You are expected to keep the perplexity ratio ≤ 1.01
  and top-1 agreement ≥ 0.99. A graft that wins accuracy by breaking language is a failure.
- **Explanation:** a reader of your `EXPLANATION.md` should be able to check that every non-zero weight you wrote
  does what you say.

## What you have

1. **The weights.** Analyse them freely in your own Python: load tensors with `safetensors.safe_open` (one tensor
   at a time; the machine has 8 GB of RAM shared with other agents), and use numpy, torch or scipy.
2. **Surgery helpers** (`wsp.llm.surgery`). They add *inert* capacity, so the model stays a stock checkpoint:
   - `insert_layer`: a whole new layer;
   - `widen_mlp`: new MLP neurons, in every layer at once;
   - `add_head_group`: one new KV head plus 7 query heads, in every layer.
   The build check reloads your model with plain `AutoModelForCausalLM.from_pretrained`. Custom code, hooks and
   `trust_remote_code` are not possible.
3. **The `wsp-llm` harness** with a budget of **forward passes** (see `CHEATSHEET.md`):
   - `eval`: up to 256 addition problems, with accuracy and the failures;
   - `gen`: free-form generation for up to 32 prompts;
   - `read`: teacher-forced reading of up to 64 texts, returning residual streams, MLP activations, attention maps
     and top-5 predictions.
   `build` and `guard` (dev perplexity and top-1 check) are free.

## How to work

1. **Diagnose first.** How does the model add today? Where are the operand digits and their place values in the
   residual stream, and which heads align the digits? Where does it go wrong?
2. **Localize causally.** That a component correlates with a result is not enough. Show that editing it changes the
   behaviour you predicted.
3. **Measure with enough power.** Accuracy over n problems has a standard error of about 50/√n points. Use as many
   problems per pass as the pass allows, and use fresh seeds for confirmation.
4. **Check the guard** after every build (it is free). Check RoPE carefully: inserted attention sees rotated q/k.
5. **Save a working `graft.py` early** and improve it. Whatever builds when you stop is what gets scored.

## Explain every step

- Before every forward pass, append to `NOTES.md` in 2–5 sentences: what the last result showed, what you will
  test now, and what result would confirm or refute it.
- `graft()` returns provenance groups (`name`, `tensors`, `mechanism`, `why`).
- `EXPLANATION.md` is the final document. For each group, state:
  - what it computes;
  - why its numbers are what they are;
  - when it fires;
  - how it stays silent elsewhere;
  - its known failure modes.
  Also include the path that led to the design.

## Rules

- **No training:** no gradient descent, backward passes, autograd, optimizers, or iterative fitting of weights to a
  loss. Closed-form computations (means, differences, PCA, pseudo-inverses, a one-shot least-squares solve) are
  allowed if you explain them.
- **Don't run the model** except through `wsp-llm`. Don't implement or simulate its forward pass on inputs in your
  own code (`graft()` cannot call modules). Analysing weights and the `read` outputs is free.
- Don't read `src/wsp/llm/` (except `surgery.py`), other agents' folders, or `experiments/llm/results_*`. No
  internet. Never use `WSP_ADMIN`.
- Work only inside `experiments/llm/agents/<you>/`.

## When you finish

You stop when your passes or tokens run out, or when you are confident. Run a final `wsp-llm build` and
`wsp-llm guard`. Your final reply should be short:
- what you built;
- dev accuracy by digit count;
- guard numbers;
- passes and tokens used;
- the one thing that would most improve it.
