# Task: write the weights of a Next-Concept Model help-desk bot (Track D, round 1)

You are one of several builder agents. You will write **every weight** of a small **Next-Concept Model (NCM)**: a
text generator that writes its reply **coarse to fine, one level at a time, with all positions of a level in
parallel**. NCM was introduced by Kumbong et al. (Hazy Research), "Next-Concept Modeling: Fast, Parallel
Coarse-to-Fine Text Generation" (arXiv 2610.14162). Nobody has written an NCM by hand before. No training.

The question: does a coarse-to-fine generator make a bot easier to **write and explain** than a token-by-token
transformer? Earlier agents hand-wrote autoregressive transformers for the same help desk and reached 97–98% on
the canned-reply version.

## 1. The bot

An IT help-desk bot (`uv run wsp-ncm spec`) with 14 intents, each with a fixed core reply, plus **two slots copied
from the user's message**:

    reply = [ "hi <name> , " ]  +  REPLIES[intent]  +  [ " i added this to ticket d d d d ." ]

- **The name part** appears only when the user says who they are ("this is priya ,", "i'm tom ."). The bot copies
  the name.
- **The ticket part** appears only when the user mentions a ticket ("re ticket 4821 ."). The bot copies its 4 digits
  (each digit is a token).
- **Off-topic messages** get the `unknown` reply.

Exact match is required: every token of the reply, then `<eos>`.

## 2. The architecture is fixed (`src/wsp/ncm/model.py`; you may read that file)

- A reply is T = 48 positions, padded with `<eos>`. It is built from K levels of discrete codes. Level l has n_l
  positions (n_1 < … < n_K = 48) and a codebook `code{l}` [V_l, D].
- One forward pass per level:
  1. A bidirectional transformer reads `[the user's raw tokens ; n_l query rows]`.
  2. The query rows are built from the **latent so far**, mean-pooled down to n_l positions.
  3. The transformer predicts all n_l codes at once.
  4. The chosen code vectors are broadcast back up to 48 positions and **added** to the latent.
- After the last level, a per-position decoder turns the latent into the reply tokens.

So the coarse levels can say *what* the reply is, such as the intent or whether there's a name or ticket, and the
fine levels say *exactly which tokens go where*, such as the copied name and digits. **How you divide the work
across levels is your design.** Fewer levels mean fewer forward passes per reply.

## 3. How you will be measured (hidden)

- **Exact replies on a hidden set (main score).** It contains new phrasings of every intent that you have never
  seen, names and ticket numbers that are not in dev, and one unseen template each for introducing a name and
  mentioning a ticket. Your bot must generalise; dev is only a sample.
- **Separately:** intent part correct, name slot correct, ticket slot correct.
- **Size:** parameters, nonzero weights, and K (forward passes per reply).
- **Explanation:** `EXPLANATION.md` must say **what each level's codes mean**, i.e. what the reply "looks like"
  after each level (use `probe`, which shows the reply decoded after every level), and what every weight group
  does and why its numbers are what they are.

## 4. Rules

1. **No training:** no gradient descent, autograd, optimizers or iterative fitting. Closed-form computations
   (means, pseudo-inverses, one-shot least squares) are allowed if explained. Generating weights with code
   (formulas, loops, tables) is fine.
2. **Your understanding of users must come from your own language knowledge.** Use `spec` and `dev` for the
   task's facts. **Never read, list or scan the vocabulary (`src/wsp/ncm/vocab.txt`, `T.vocab()`, `T.index()`
   over all words).** You may check whether a specific word you thought of exists with
   `uv run wsp-ncm tokenize --text "word"`, or `T.encode("word")` for words you name yourself.
3. Don't read `src/wsp/ncm/` except `model.py`, other agents' folders, or `src/wsp/support/` (test data lives
   there). No internet. Never use `WSP_ADMIN`.
4. Work only in `experiments/ncm/agents/<you>/`.
5. **Local runs** (clarified during round 1): you may run `model.generate` yourself on messages **you write**,
   the way a developer tests. Evaluating on the **dev set** goes through `wsp-ncm eval`, which counts against
   your runs. Don't loop over `wsp-ncm dev` examples locally.

## 5. Deliverables

- `weights.py` with `build() -> (cfg, params, groups)`; see `CHEATSHEET.md`. Save a valid version early and
  overwrite it as you improve it.
- `EXPLANATION.md`:
  - the level-by-level story;
  - every weight group: mechanism and why its values;
  - how the bot generalises;
  - known failure modes;
  - the path that led to your design.
- `NOTES.md`: 1–3 lines per run, covering what you tested and what you learned.

Final reply, short: your design in 3 sentences, K and the sizes of the levels, dev exact, params / nonzero, runs and
tokens used, and the biggest remaining weakness.
