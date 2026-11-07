# Task: an open-ended chat model, written as a multi-layer Next-Concept Model (Track D, chat round)

You are agent **nco60**. You will write every weight of a **Next-Concept Model (NCM)** that a person can **chat
with**. The model:
- holds a multi-turn conversation;
- greets and remembers what it is told (names, facts, corrections);
- answers simple questions;
- follows simple instructions;
- admits when it doesn't know;
- declines harmful requests.

No training. The model generates each reply coarse to fine: one forward pass per level, every position of a level
in parallel (Kumbong et al., "Next-Concept Modeling", arXiv 2610.14162).

The earlier hand-built 20M-parameter chat models (Stage X, standard transformers) became rule bots and echo bots.
Your advantages:
- **a real pretrained model's weights as material**, i.e. weight space you can study and reuse;
- **multi-layer attention** over a **KV-cached conversation**;
- **coarse-to-fine** generation.

The question is how far a chat model gets when every mechanism is *derived* from understanding weight space.

## 1. The architecture is fixed (`src/wsp/ncmchat/model.py`; read it)

- **The conversation is a causal prefix** in Qwen chat format, using the Qwen2.5 tokenizer (`<|im_start|>user …`).
  It sits in a **KV cache** across levels and turns. RoPE positions: token i is at position i.
- **A reply is 64 positions.** It is built from K levels of codes (n_1 < … < n_K = 64). In each level:
  1. The query rows sit at position `len(prefix)`, so RoPE measures distance back from the end of the
     conversation.
  2. They attend through `n_layers` attention + MLP layers to the whole conversation and to each other.
  3. They pick codes in parallel.
  4. The code vectors are added into a latent [64, D].
- **A per-position decoder** maps the latent to tokens. The reply ends at `<|im_end|>`.
- **Size:** the embedding has 151,936 rows (`config.vocab = 151936`). Cap: 300M parameters, float32.

## 2. Hard rules: strict weight generation from weight-space understanding

1. **No training or fitting of any kind:**
   - no gradients, autograd or optimizers;
   - no regression or least squares against targets, not even closed-form;
   - no k-means or any other iterative fitting.
   **Allowed:** *analysis* of existing weights, i.e. dot products, norms, SVD/PCA, nearest neighbours, projections.
   Generating weights with code (formulas, loops) is fine.
2. **No lookup lexicons.** No hand-typed lists that map words or phrases to meanings, intents, categories or
   replies.
   - Language understanding must come from **structure you derive from weight space**, e.g. the geometry of
     Qwen's embeddings, or from mechanisms (attention, copying, position).
   - You may use **anchor words only to name a direction** (at most 3 per concept). Coverage must come from the
     geometry, not from listing synonyms.
3. **No canned answers.**
   - No text literal in your code longer than 6 words; `check` flags them.
   - Knowledge may be stored as short units, e.g. (subject, relation, object).
   - Replies must be composed.
4. **Qwen2.5-0.5B is material, not a crutch.**
   - Read any tensor of `models/Qwen2.5-0.5B/model.safetensors` (via `safetensors.safe_open`, one tensor at a time).
   - Reuse tensors or subspaces when you can explain what each contributes in your network.
   - Don't transplant whole Qwen layers.
   - **Never run Qwen**, or any of its layers, on inputs: no forward passes and no text or data generated with it.
5. Don't read harness source other than `model.py`, other agents' folders, or `experiments/stage_x/` (except
   that `wsp-ncmchat dev` shows you its dev cases). No internet. Never use `WSP_ADMIN`.
6. **Local runs.** You may run `Chat` / `model.reply` on conversations **you write**. Dev evaluation goes
   through `wsp-ncmchat eval`, which counts against your runs.

## 3. How you will be measured

- **The main score is blind.** After you stop, your weights are frozen. An independent agent then writes a new
  hidden chat test (unseen topics, names, facts and instructions), and a blind judge grades every reply against
  its rubric.
- **The dev mechanical checks** (`eval`, 33 checked turns in 40 cases) are a guide, not the score.
- **Explanation:** `EXPLANATION.md` must cover:
  - what each level means (`probe` decodes the reply after every level);
  - what each layer and head does;
  - which Qwen structure you used and why it carries meaning;
  - capacity limits;
  - failure modes.

## 4. Deliverables

- `weights.py`: `build() -> (cfg, params, groups)`. Save a valid version early and keep improving it.
- `EXPLANATION.md`
- `NOTES.md`: 1–3 lines per run.

Final reply, short:
- your design in 4 sentences;
- K, layers and heads;
- dev mechanical score;
- params / nonzero;
- runs and tokens used;
- an honest list of what it can and can't do in conversation.
