*Protocol adapted from Takehi, Teoh, Zhang & Khattab, "Sparks of Weight-Space Programming" (2026), Experiments 5,
6 and 8. The one addition is a folder of example models.*

TASK. Write, from scratch, a neural network that a person can **chat with**, open-ended, as well as possible. It
should:
- hold a multi-turn conversation;
- greet;
- remember what it is told (names, facts, corrections);
- answer simple questions, including general knowledge you can put into it;
- follow simple instructions;
- say when it does not know;
- decline harmful requests.

All of its weights start at zero; you set every one. There is no training: no data, no loss, no gradients, no
fine-tuning, and no pretrained weights. Every weight comes from your own reasoning, written by code from your own
design. Groups of weights may be set together by one formula or rule; you don't have to justify each number
separately, but every group needs a reason.

THE CONVERSATION. Text is tokenized with the Qwen2.5 BPE tokenizer (embedding rows 151,936). A conversation is a
sequence of turns in this format:
`<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n` …
The model sees the whole conversation so far. Its reply ends when it produces `<|im_end|>`, or after 128 tokens.
The reply is then added to the conversation, and the next user message follows.

THE NETWORK. You write ONE file, `experiments/pure/agents/<you>/model.py`. It says how the system should run your
model and generates all its weights: `build() -> (family, cfg, params, groups)`. Choose one family:
- **"transformer"**: a standard decoder-only transformer (`src/wsp/pure/transformer.py`):
  - token embedding;
  - n layers of causal multi-head attention (optionally grouped-query; RoPE, learned or no positions) and an MLP
    (relu / gelu / silu / tanh / swiglu);
  - RMSNorm, LayerNorm or none;
  - tied or separate unembedding.
  Replies are generated greedily, one token at a time, with a KV cache.
- **"ncm"**: a Next-Concept Model (`src/wsp/ncmchat/model.py`, Kumbong et al. 2026):
  - the conversation is a KV-cached causal prefix;
  - the reply (64 positions) is built in K levels of codes, coarse to fine, one forward pass per level;
  - all positions of a level are predicted in parallel.
  Qwen options must stay off.

Read the family's source file once: it states the exact forward pass. You choose the size (cap: 300M parameters)
and every value.

HOW IT WILL BE MEASURED. When you finish, your file is frozen. An independent agent then writes a new hidden set of
conversations, with unseen topics, names, facts, instructions and harmful requests, plus a rubric for every turn. A
blind judge grades every reply of your model against its rubric. The main score is the share of turns judged
acceptable. You never see that test. Earlier hand-built chat models for comparison, with their dev mechanical
scores: 7/33 and 4/33.

WHAT YOU HAVE.
1. **Example models** in `experiments/pure/examples/` (start with its README.md). These are hand-written models
   from earlier experiments (chat, help desk, an exact adder) with their explanations, as references for
   techniques. They are not solutions.
2. **Pass tools**, with a budget of 100 passes in total. Each use of a pass tool counts as one pass, and a request
   that fails (e.g. your model.py doesn't build) is not counted. Every pass builds your current model.py.
   - `dev`: runs the practice conversations (40 cases, 115 turns, 33 with mechanical checks; see
     `wsp-pure dev-list`). Returns the mechanical score and the failed checks. Every reply goes to
     `runs/dev_<k>.txt` in your folder.
   - `chat`: runs up to 8 conversations of up to 12 turns that **you** write (a JSON file). Returns every reply.
   - `probe`: runs one conversation and shows the internals of its last reply:
     - transformer: the top-5 tokens at each reply step, and every head's top-3 attended positions at the point
       where the first reply token is chosen;
     - ncm: the codes per level, and the reply decoded after each level.
   - Free and not counted: `check` (builds your file, validates shapes, counts parameters), `spec`, `dev-list`,
     `examples`, `tokenize` (ids of text you write), `status`, `tokens`.
   **The session continues until every pass is used.** Use the passes: measure, diagnose, revise, measure again.
3. Your shell, for exactly two things: running `uv run wsp-pure …` commands, and viewing files in your folder and
   the example folder (`cat`, `head`, `grep`, `ls`, `wc`). You write `model.py` and your notes with the Write and
   Edit tools.

RULES. Not allowed:
- (a) Gradient descent or any other training: no backward passes, autograd or optimizers, and no fitting of weights
  to data or labels (no regression or least squares on examples).
- (b) Running anything yourself: no `python`, no `uv run python`, no scripts, no notebooks. **Your network runs only
  inside the pass tools.** Code that generates weights lives in `model.py` and runs only when the system builds
  it.
- (c) Pretrained weights of any model, or text generated by another model.
- (d) Internet access.
- (e) Reading harness source other than the two family files, `src/wsp/stage_x/`, or other agents' folders.

Allowed: writing the weights by code from your own design (formulas, loops, tables of knowledge you write
yourself, linear algebra on the weights themselves), and any analysis of the pass results.

HOW TO WORK.
1. **Diagnose first.** Look at what fails in the practice conversations, and why.
2. **Localize.** Use `probe` to see which layer, head or level does what. Change one thing, and confirm it changed
   the behaviour you predicted.
3. **Generalize.** The hidden test uses conversations you have never seen. Practice cases are a sample, not the
   target. Write your own `chat` conversations to test phrasings, names and facts that aren't in dev.
4. **Save a working model.py early**, and keep it working.

EXPLAIN EVERY STEP. Before every pass, append to `NOTES.md` in plain English (2–5 sentences):
- what the last result showed;
- what you conclude;
- what you will change or test now, and why;
- what outcome would confirm or refute your hypothesis.
`EXPLANATION.md` is the final document. It covers:
- the architecture;
- every weight group: what it computes and why its values are what they are;
- where the model's knowledge lives and how much it holds;
- what it can and can't do;
- the path that led to the design.
These are read by the experimenters.

HOW TO FINISH. When every pass is used, `model.py` builds, and `EXPLANATION.md` is written, reply without running
anything. Describe the design, the last dev score, and an honest list of what the model can and can't do in
conversation.
