# wsp-ncm cheat sheet

Read `experiments/ncm/TASK.md` once. `<a>` is your name and your folder is `experiments/ncm/agents/<a>/`. Run
everything from the repo root with `uv run`.

## Deliverable: `agents/<a>/weights.py`

```python
import torch
from wsp.ncm import model, tokenizer as T

def build():
    V = len(T.vocab())                  # vocab size (only the SIZE; don't scan the words)
    ix = T.encode                       # T.encode("hi tom") -> ids of words you name
    cfg = model.Config(vocab=V, D=64, levels=(1, 8, 48), codebooks=(16, 64, 512),
                       d_model=128, n_layers=2, n_heads=4, d_head=32, d_mlp=256, dec_hidden=0, norm=False)
    P = {k: torch.zeros(*s, dtype=torch.float64) for k, s in model.shapes(cfg).items()}
    ...                                  # write values
    groups = [{"name": "intent concept", "tensors": ["head0", "code0"], "mechanism": "...", "why": "..."}]
    return cfg, P, groups
```

Parameter names and shapes come from `model.shapes(cfg)` (H = n_heads, dm = d_model):
- `tok_emb [V,dm]`, `pre_pos [32,dm]` (prefix positions), `lvl_pos [48,dm]`, `lvl_emb [K,dm]`, `w_in [D,dm]`
- per layer i: `L{i}.wq/wk/wv [H,dm,dh]`, `L{i}.wo [H,dh,dm]`, `L{i}.bq/bk/bv [H,dh]`, `L{i}.bo [dm]`
  - if `d_mlp`: `L{i}.w1 [dm,d_mlp]`, `b1`, `w2 [d_mlp,dm]`, `b2`
  - if `norm`: `ln1_g/b`, `ln2_g/b`
- per level l: `head{l} [dm,V_l]`, `head_b{l} [V_l]`, `code{l} [V_l,D]`
- decoder: `dec_w1 [D,dec_hidden]`, `dec_b1`, `dec_w2 [dec_hidden or D, V]`, `dec_b2 [V]`

Facts you'll need:
- **Level positions.** Position t of the reply belongs to index `t * n_l // 48` of level l. `up` broadcasts codes by
  this map; `down` averages the latent over each group.
- **Attention.** It is bidirectional and has no mask: level rows see all user tokens and all rows of their own
  level. Rows of earlier levels are not rows; they only reach later levels through the latent.
- **Level 0 queries.** On level 0 the latent is zero, so its queries are just `lvl_emb[0] + lvl_pos[:n_0]`.
- **Ids.** `<pad>`=0, `<eos>`=1, `<unk>`=2. The user's message is truncated to 32 tokens. Numbers are split into
  digits. Words the model doesn't know are `<unk>`.
- **Output.** The reply is the decoded tokens up to the first `<eos>`.

## Commands

| command | cost | what |
|---|---|---|
| `uv run wsp-ncm spec` | free | replies, reply format, dev templates, sizes, special ids |
| `uv run wsp-ncm dev [--ids 0,5]` | free | the 224 dev examples (query, reply, intent, name, ticket) |
| `uv run wsp-ncm tokenize --text "..."` | free | ids/tokens of text **you** wrote |
| `uv run wsp-ncm check --agent <a>` | free | builds weights.py, validates shapes, params / nonzero |
| `uv run wsp-ncm eval --agent <a> [--ids ...] [--note "..."]` | 1 run | exact on dev (all by default), by kind, slot accuracies, failures |
| `uv run wsp-ncm probe --agent <a> --text "..."` | 1 run | per level: codes, **reply decoded after that level**, top prefix attention |
| `uv run wsp-ncm status --agent <a>` / `tokens --agent <a>` | free | runs left / tokens used |

An eval of all of dev takes about 2 s. Write any message you like into `probe` to test generalisation. Phrasings
you make up yourself are your best guide to the hidden test.
