# wsp-pure cheat sheet

Read `TASK.md` once. Your folder is `experiments/pure/agents/<a>/`. Run commands from the repo root.

## model.py

```python
import torch
from wsp.pure import transformer as TF          # family "transformer"
# from wsp.ncmchat import model as NCM          # family "ncm"
from wsp.pure.harness import ids, special        # ids("text you write"), special("<|im_end|>")

def build():
    cfg = TF.Config(vocab=151936, d_model=256, n_layers=4, n_heads=8, d_head=32, n_kv_heads=0, d_mlp=1024,
                    act="relu", norm="rms", pos="rope", rope_theta=10000.0, max_ctx=8192, tied=True)
    P = {k: torch.zeros(*s) for k, s in TF.shapes(cfg).items()}
    ...                                          # set values by your reasoned code
    groups = [{"name": "...", "tensors": ["L0.wq"], "mechanism": "...", "why": "..."}]
    return "transformer", cfg, P, groups
```

Shapes:
- **Transformer:** `tok_emb [V,d]`, `L{i}.wq [H,d,dh]`, `bq [H,dh]`, `wk/wv [KV,d,dh]`, `bk/bv [KV,dh]`,
  `wo [H,dh,d]`, `bo [d]`, `w1 [d,m]`, `b1`, `w2 [m,d]`, `b2`, `wg [d,m]` (swiglu only), `n1_g/n2_g` (+`_b` for
  LayerNorm), `nf_g`, `unembed [d,V]` if not tied, `pos_emb [max_ctx,d]` if learned. Linear maps are `x @ W`.
- **NCM:** see `NCM.shapes(cfg)`.

`check` is free. Use it after every edit: it catches shape and build errors without costing a pass.

## Commands (`uv run wsp-pure …`)

| command | cost | |
|---|---|---|
| `spec`, `dev-list [--ids a,b]`, `examples`, `tokenize --text "..."` | free | facts, practice cases, example index, token ids |
| `check --agent <a>` | free | build + validate; params, nonzero, build seconds |
| `dev --agent <a> [--ids ...] [--note "..."]` | 1 pass | practice cases; failed checks; full replies in `runs/dev_<k>.txt` |
| `chat --agent <a> --convs convs.json` | 1 pass | your conversations: `[["turn1","turn2"], ["..."]]`, ≤ 8 × 12 |
| `probe --agent <a> --turns turns.json` | 1 pass | internals of the last reply; details in `runs/probe_<k>.txt` |
| `status --agent <a>`, `tokens --agent <a>` | free | passes left; your tokens used |

Builds are cached per version of model.py. Long commands log live to `experiments/pure/logs/latest.log`.
