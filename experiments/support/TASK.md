# Task: write the weights of a tiny IT-support transformer

You are one of several agents, each with a different budget of **forward passes**: 0, 10 or 50. Write every parameter of a standard transformer that works as a basic IT help-desk bot. Use as few parameters as you can, and explain every one of them.

## What the bot must do

`src/wsp/support/data.py` defines 14 intents (password, locked, vpn, wifi, email, printer, software, slow, mfa, disk, monitor, hello, thanks, unknown). Each intent has exactly one canonical reply.
- Given a user message, the model must generate that intent's reply token by token with greedy decoding, followed by `<eos>`.
- Anything outside IT support (weather, lunch, jokes, ...) gets the `unknown` reply.

The rule for "correct" is strict: every token of the reply and the final `<eos>` must be right.

- **Dev set:** the 56 queries in `data.DEV`, which you may read.
- **Hidden test set:** 112 different, unseen phrasings of the same 14 intents. It includes synonyms, paraphrases and out-of-domain questions, and it is what you are scored on.

## Tokenizer and sequence format

`src/wsp/support/tokenizer.py` is a word-level tokenizer with a fixed vocabulary of 470 tokens. Unknown words map to `<unk>`. The sequence format is:

```
<bos> user tokens (max 24) <sep> reply tokens <eos>
```

The model must predict the first reply token at `<sep>`, each following token at the token before it, and `<eos>` after the last one. The sequence length is at most 64.

## The model (fixed; you supply only the parameters)

`src/wsp/support/model.py` defines a standard pre-norm decoder:

- token embedding, plus optional learned positions
- n layers, each with multi-head causal softmax attention and an optional ReLU MLP
- optional LayerNorm
- unembedding, which can be tied to the embedding

You choose a `Config` (d_model, n_layers, n_heads, d_head, d_mlp, pos, tie_embeddings, norm, bias) and must provide exactly the tensors named in `model.param_shapes(cfg)`. You cannot change the forward function.

**Every parameter counts toward your size, zeros included.** Embedding rows cost vocab × d_model, so think hard about d_model.

## Objective (in priority order)

1. **Hidden test exact-reply accuracy.** Correct, robust intent detection is what makes the bot useful.
2. **Fewest parameters.** Among models within about 5 percentage points of each other on accuracy, the smaller one wins.
3. **Your own token efficiency.** Your LLM token usage is measured and reported, so work economically:
   - generate weights from compact code (formulas, loops, hand-written keyword lists), never write out tensors;
   - never print large tensors or long outputs;
   - read only what you need and keep notes concise.

## Harness and checking scripts

Run everything with `uv run wsp-support ...` from the repo root.

These commands are **free** and run no forward pass:
- `check --model F`: validates names and shapes and prints the parameter count by group and the number of nonzero parameters.
- `tokenize --text "..."`: shows the token ids for a text.
- `status --agent <you>`: shows your remaining budget.
- Reading the vocabulary from Python is also free: `from wsp.support import tokenizer; tokenizer.vocab()`, `tokenizer.tid(w)`.

These commands are **charged**:
- `eval --agent <you> --model F --ids 0,5,9 --note "why"` costs 1 pass per dev query. It runs teacher-forced and returns, per query, whether the reply was exactly right, the token accuracy, and the first wrong token.
- `probe --agent <you> --model F --text "query" [--reply "partial reply"] [--trace] --note "why"` costs 1 pass. It returns the top-5 predicted next tokens at the last position. With `--trace` it also returns the top-3 tokens each head attends to and the logit-lens top-3 after each layer.

The harness logs every call to `experiments/support/agents/<you>/log.jsonl`, and refuses any request over your budget without charging you.

## Hard rules

1. **No training, no backward pass, no gradients, no optimizers.**
2. **No fitting to data.** Don't derive weights programmatically from dataset statistics (word counts, regression, clustering). You may read the dev queries and the vocabulary, and hand-write the keyword lists and choices your design needs, as a human engineer would.
3. **Forward passes only through the harness.** Don't call `model.forward`, the harness scoring functions, or your own reimplementation of the model on token sequences. You may hand-check single quantities on a few tokens, e.g. the attention score of head 0 between `<sep>` and `vpn`, as long as you're not emulating whole-sequence outputs.
4. **The hidden test set is off limits.** Don't read `src/wsp/support/test_blob.txt`, and don't call `data._test_set` or `data._test_vocab_words`.
5. **No internet.** Don't use WebSearch or WebFetch, and don't download anything.
6. **Stay in your own directory.** Work only in `experiments/support/agents/<you>/`. Don't read other agents' directories, don't edit `src/`, `tests/`, ledgers or this file, and don't set `WSP_ADMIN`.

## Deliverables (in `experiments/support/agents/<you>/`)

1. `weights.py`: `build() -> (Config, dict[str, torch.Tensor])`, every tensor built from named, commented constructions.
2. `EXPLANATION.md`: for **every** parameter tensor, what mechanism it implements, why its values are what they are, and when it fails. Include the parameter count and why each dimension of the config is needed.
3. `METHOD.md`: your design reasoning, and every forward pass in order: what it tested, what it showed, and what you changed. With a budget of 0, say how you checked correctness without running the model.

Finish with a short reply giving your parameter count, the dev queries you verified and how many passed, and a 3–5 sentence summary of the design.
