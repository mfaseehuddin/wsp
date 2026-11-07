# wsp-ncmdeep cheat sheet

Read `experiments/ncmdeep/TASK.md` once, then `src/wsp/ncmchat/model.py` and `src/wsp/ncmchat/qwen.py`. Your folder is
`experiments/ncmdeep/agents/<a>/`. Run everything from the repo root with `uv run`.

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
    cfg = model.Config(vocab=151936, D=128, levels=(1, 4, 16, 64), codebooks=(32, 256, 1024, 151936),
                       d_model=256, n_layers=4, n_heads=8, d_head=32, d_mlp=512, dec_hidden=0,
                       rope_theta=10000.0, norm=True,
                       qwen_embed=True, qwen_layers=tuple(range(12)), qwen_final_norm=False,   # reuse Qwen 0..11 verbatim
                       level_meanings=("dialogue act", "content", "phrase skeleton", "tokens"))
    # with qwen_embed=True there is no tok_emb; prefix rows = qwen_residual @ P["pre_proj"] [896, d_model]
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

Reused Qwen layers cost about 60 MB of RAM each (fp32): all 24 layers plus the embedding come to about 2 GB.
Prefix processing is cached, so each turn only runs the new tokens through them.

## Local testing (allowed, free) on conversations you write

```python
from wsp.ncmchat.harness import load, Chat
cfg, P, _ = load(Path("experiments/ncmdeep/agents/<a>/weights.py"))
ch = Chat(cfg, P); print(ch.turn("Hi, I'm Rosa.")["reply"]); print(ch.turn("What's my name?")["reply"])
```

## Commands

| command | cost | what |
|---|---|---|
| `uv run wsp-ncmdeep spec` | free | sizes, chat format, special ids, Qwen path |
| `uv run wsp-ncmdeep dev [--ids greet,coref]` | free | the 40 dev cases (turns, rubrics, checks) |
| `uv run wsp-ncmdeep tokenize --text "..."` | free | ids and pieces of text you wrote |
| `uv run wsp-ncmdeep check --agent <a>` | free | builds, validates, params / nonzero, flags text literals over 6 words |
| `uv run wsp-ncmdeep eval --agent <a> [--ids ...] [--note ...]` | 1 run | dev cases: mechanical pass / checked, every reply |
| `uv run wsp-ncmdeep chat --agent <a> --turns turns.json` | 1 run | a conversation of ≤ 12 turns you write (JSON list, or one string) |
| `uv run wsp-ncmdeep probe --agent <a> --turns turns.json` | 1 run | last turn: codes per level, **reply decoded after each level**, most-attended prefix tokens |
| `uv run wsp-ncmdeep status --agent <a>` / `tokens --agent <a>` | free | runs left / tokens used |

The 64-turn dev session is `long_session_64`.

**Memory:** the machine has 8 GB of RAM, shared with other experiments. Load Qwen tensors one at a time, and
`del` them when you're done.
