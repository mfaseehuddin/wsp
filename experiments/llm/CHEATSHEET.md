# wsp-llm cheat sheet

Read `experiments/llm/TASK.md` once. `<a>` is your agent name and your folder is `experiments/llm/agents/<a>/`.
Run everything from the repo root with `uv run`.

## Deliverable: `agents/<a>/graft.py`

```python
from wsp.llm import surgery
import torch

def graft(model, tok):            # HF Qwen2ForCausalLM (float32, CPU) + tokenizer; edit IN PLACE
    L = model.model.layers         # L[i].self_attn.{q,k,v,o}_proj, L[i].mlp.{gate,up,down}_proj,
                                   # L[i].input_layernorm / post_attention_layernorm; model.model.embed_tokens
    new = surgery.insert_layer(model, 24)      # identity layer after the last block (index 0..24)
    sl = surgery.widen_mlp(model, 64)          # new neurons sl.start..sl.stop-1, in every layer's MLP
    q_heads, kv = surgery.add_head_group(model)  # 7 new q heads (14..20) sharing new kv head 2
    with torch.no_grad():
        L[3].mlp.down_proj.weight[:, sl] = ...   # write values
    return [{"name": "...", "tensors": ["model.layers.3.mlp.down_proj.weight[:, 4864:4928]"],
             "mechanism": "...", "why": "..."}]
```

- Weights are nn.Linear: `weight` is `[out, in]`, and the layer computes `x @ W.T + b`.
- q head h uses rows `h*64:(h+1)*64` of `q_proj`; kv head j uses rows `j*64:(j+1)*64` of `k_proj` / `v_proj`; the
  output of head h is in `o_proj.weight[:, h*64:(h+1)*64]`.
- RoPE (θ=1e6, head_dim 64): pair (i, i+32) rotates by angle `pos * 1e6**(-2i/64)`, i=0..31. The score between q at
  position m and k at position n depends only on m−n.
- `surgery.digit_ids(tok)` gives the token ids of '0'..'9'.
- Calling modules inside `graft()` raises. Graft is pure weight writing.

## Weights for analysis (free)

```python
from safetensors import safe_open
with safe_open("models/Qwen2.5-0.5B/model.safetensors", "pt") as f:
    W = f.get_tensor("model.layers.3.mlp.down_proj.weight").float()   # bf16 on disk
# names: model.embed_tokens.weight, model.norm.weight, model.layers.{i}.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input_layernorm, post_attention_layernorm}.{weight,bias}
```

## Commands

| command | cost | what |
|---|---|---|
| `uv run wsp-llm build --agent <a>` | free | run graft.py, save `build/`, stock reload check; diff vs base in `build_report.json` |
| `uv run wsp-llm guard --agent <a>` | free | dev guard: ppl_ratio and top1_agreement vs base (~30 s) |
| `uv run wsp-llm eval --agent <a> --digits 8 [--digits2 5] --n 256 [--format fewshot\|zeroshot] [--seed 3] [--base] [--note "..."]` | 1 pass | addition accuracy, first-wrong-digit histogram, sample failures; all rows in `runs/eval_<k>.jsonl` |
| `uv run wsp-llm eval --agent <a> --problems "123+45,99999+1" ...` | 1 pass | your own problems (≤256) |
| `uv run wsp-llm gen --agent <a> --prompts prompts.json [--max-new 32] [--base]` | 1 pass | greedy generation for ≤32 prompts (a JSON list, or one literal string) |
| `uv run wsp-llm read --agent <a> --texts texts.json --name NAME [--layers 0,8,24] [--mlp 5,20] [--attn 3] [--base]` | 1 pass | teacher-forced; saves `reads/NAME.npz` |
| `uv run wsp-llm status --agent <a>` / `tokens --agent <a>` | free | passes left / tokens used |

- `--base` runs the unedited model and still costs a pass.
- Every command except status and tokens uses the latest **build**, so rebuild after editing graft.py.

`read` npz arrays (right-padded, at most 64 texts × 512 tokens; with `--attn`, at most 8 texts and 4 layers):
- `input_ids [n,T]`, `lengths [n]`
- `resid_L{l} [n,T,896]`: hidden state after layer l (0 = embeddings, 24 = last block, before the final norm)
- `mlp_L{l} [n,T,I]`: down_proj input (SiLU(gate)·up), at most 4 layers
- `attn_L{l} [n,H,T,T]`
- `top5_ids`, `top5_logprob [n,T,5]`: predictions for the next token
- `next_logprob [n,T]`: log p(token t+1)

Prompt formats: `fewshot` = `"12+35=47\n481+207=688\n9036+1185=10221\n" + "a+b="`, `zeroshot` = `"a+b="`.
The answer is parsed as the digits generated right after `=`.

## Machine

8 GB RAM, shared. The harness runs one model at a time (a lock), so a command may wait for another agent.
Load single tensors, not the whole model, in your own scripts. Don't print big arrays.
