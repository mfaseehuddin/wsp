# wsp-ncmchat cheat sheet

Read `experiments/ncmchat/TASK.md` once, then `src/wsp/ncmchat/model.py`. Your folder is
`experiments/ncmchat/agents/<a>/`. Run everything from the repo root with `uv run`.

## Deliverable: `agents/<a>/weights.py`

```python
import torch
from safetensors import safe_open
from wsp.ncmchat import model

QW = "models/Qwen2.5-0.5B/model.safetensors"
def qwen(name):                                   # read-only material, one tensor at a time
    with safe_open(QW, "pt") as f:
        return f.get_tensor(name).float()
# e.g. E = qwen("model.embed_tokens.weight")   # [151936, 896], tied in/out embedding of Qwen2.5-0.5B

def build():
    cfg = model.Config(vocab=151936, D=128, levels=(1, 8, 64), codebooks=(64, 512, 151936),
                       d_model=256, n_layers=4, n_heads=8, d_head=32, d_mlp=512, dec_hidden=0,
                       rope_theta=10000.0, norm=True)
    P = {k: torch.zeros(*s) for k, s in model.shapes(cfg).items()}
    ...
    return cfg, P, [{"name": "...", "tensors": [...], "mechanism": "...", "why": "..."}]
```

- **Names and shapes:** `model.shapes(cfg)`.
- **Linear layers** are `x @ W`, with W stored as `[in, out]`.
- **Attention weights** are per head: `wq [H, d_model, d_head]`, `wo [H, d_head, d_model]`.
- **Parameter sizes:** the codebook and decoder for 151,936 tokens are large (151,936 × D each).
  `dec_w2 [D or dec_hidden, 151936]`. Keep the total ≤ 300M.
- **Qwen tensor names:** `model.embed_tokens.weight`, `model.norm.weight`,
  `model.layers.{0..23}.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input_layernorm,
  post_attention_layernorm}.{weight,bias}`. Qwen's nn.Linear stores `[out, in]`.
- **Tokenizer:**
  ```python
  from wsp.ncmchat.harness import tok, ids, special
  ids("text")
  special("<|im_end|>")
  tok().decode([...])
  ```
  `ids` gives the ids of text **you** wrote. Do not scan the vocabulary.

## Local testing (allowed, free) on conversations you write

```python
from wsp.ncmchat.harness import load, Chat
cfg, P, _ = load(Path("experiments/ncmchat/agents/<a>/weights.py"))
ch = Chat(cfg, P); print(ch.turn("Hi, I'm Rosa.")["reply"]); print(ch.turn("What's my name?")["reply"])
```

## Commands

| command | cost | what |
|---|---|---|
| `uv run wsp-ncmchat spec` | free | sizes, chat format, special ids, Qwen path |
| `uv run wsp-ncmchat dev [--ids greet,coref]` | free | the 40 dev cases (turns, rubrics, checks) |
| `uv run wsp-ncmchat tokenize --text "..."` | free | ids and pieces of text you wrote |
| `uv run wsp-ncmchat check --agent <a>` | free | builds, validates, params / nonzero, flags text literals over 6 words |
| `uv run wsp-ncmchat eval --agent <a> [--ids ...] [--note ...]` | 1 run | dev cases: mechanical pass / checked, every reply |
| `uv run wsp-ncmchat chat --agent <a> --turns turns.json` | 1 run | a conversation of ≤ 12 turns you write (JSON list, or one string) |
| `uv run wsp-ncmchat probe --agent <a> --turns turns.json` | 1 run | last turn: codes per level, **reply decoded after each level**, most-attended prefix tokens |
| `uv run wsp-ncmchat status --agent <a>` / `tokens --agent <a>` | free | runs left / tokens used |

The 64-turn dev session is `long_session_64`.

**Memory:** the machine has 8 GB of RAM, shared with other experiments. Load Qwen tensors one at a time, and
`del` them when you're done.
