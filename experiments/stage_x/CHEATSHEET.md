# wsp-x cheat sheet (read this instead of the source)

Rules and scoring are in `TASK.md`. This page holds the API facts. Run every command from the repo root as `uv run wsp-x <cmd> …`. Output is JSON. `--model` defaults to `experiments/stage_x/agents/<you>/weights.py`.

| cmd | cost | what |
|---|---|---|
| `status --agent A` / `tokens --agent A` | free | runs left / your token use |
| `dev [--ids 0,3]` | free | the 50 public dev conversations (index, id, category, files, turns with `check` / `rubric`) |
| `check --agent A` | free | builds; reports params, nonzero, provenance coverage, KV MiB, max abs weight, `long_text_literals`, groups missing from EXPLANATION.md |
| `tokenize --agent A --text T` | free | ids and pieces under your word list |
| `eval --agent A --ids 0,5 [--note N] [--max-new-tokens 512]` | 1 per case | per turn: reply, tokens, pass (mechanical; `null` = open/rubric only), tools, error; plus `reuse` metrics |
| `chat --agent A --turns '["hi","2+2?"]' [--files '{"a.txt":"x"}']` | 1 per turn | a free-form multi-turn conversation: replies, think, tools, context size, seconds |
| `probe --agent A --turns '["one user msg"]' [--prefix "partial reply"] [--trace]` | 1 | top-5 next tokens. With `--trace`: per layer, each head's top-3 attended positions plus a logit lens |

Builds run in a sandboxed subprocess (100 s CPU limit). A build error comes back as `{"error": …}` and costs nothing. Dev case indices are 0–49; case 49 is the 64-turn session.

## weights.py contract

```python
import torch
import math

def build():
    cfg = dict(d_model=256, n_layers=6, n_heads=4, d_head=64, d_mlp=1024,
               max_len=16384, pos="sinusoidal", norm=False, bias=True,
               tie_embeddings=False)
    words = ["the", "capital"]          # unique ASCII words matching [A-Za-z0-9_]+(['-][A-Za-z0-9_]+)*
    V = 272 + len(words)
    p = {...}                            # every tensor below, float32, exact shapes
    groups = [dict(name="facts.capital", tensor="layers.3.w_in", index=[{"start":0,"stop":40}, [3, 7]],
                   mechanism="...", why="...")]
    return cfg, p, groups, words
```

- `index=None` covers a whole tensor. Otherwise it is a list of axes; each is an int, a list of ints, or `{"start":a,"stop":b,"step":c}`. Every nonzero weight must be in at least one group, or the build is rejected. Group names must be unique, and every name must appear in EXPLANATION.md.
- **Sanitizer:** only top-level `import torch` / `import math`; functions, for, if, comprehensions. No while, try, classes, lambdas, with, or dunder names. No open/eval/exec/print/getattr. **Seeded randomness is allowed** for dense codes: `g = torch.Generator().manual_seed(1)` then `torch.randn(n, d, generator=g)` (also rand, randint, randperm, normal_, uniform_). Every random call must pass `generator=`, so builds stay deterministic. Also allowed: sign, topk, argmax, norm, matmul, einsum, softmax, relu (for building codes; never call the model). Builtins: abs all any bool chr dict enumerate float int isinstance len list max min ord range reversed round set sorted str sum tuple zip. Torch/str/list/dict methods are allowlisted (zeros, ones, full, tensor, arange, eye, stack, cat, outer, sin, cos, exp, log, sqrt, where, clamp, reshape, view, transpose, T, unsqueeze, clone, fill_, index_add_, scatter_, masked_fill_, tril, triu, sum, max, min, item, get, items, keys, values, append, extend, index, split, join, lower, upper, strip, startswith, endswith, …). If something is refused, the error names it. Source ≤ 1 MB.

## Shapes (V = 272 + len(words), d = d_model, h = heads, e = d_head, m = d_mlp)

| name | shape |
|---|---|
| `embed` | V,d |
| `pos` (only if pos="learned") | max_len,d |
| `layers.i.wq` / `wk` / `wv` | h,d,e |
| `layers.i.wo` | h,e,d |
| `layers.i.bq` / `bk` / `bv` (bias) | h,e |
| `layers.i.bo` (bias) | d |
| `layers.i.w_in` / `w_out` (m>0) | d,m / m,d |
| `layers.i.b_in` / `b_out` (m>0, bias) | m / d |
| `layers.i.ln1.g/.b`, `ln2.g/.b`, `lnf.g/.b` (norm) | d |
| `unembed` (unless tied) | d,V |
| `b_unembed` (bias) | V |

Forward, per layer: `a = LN(x)`, `q = a@wq + bq`, and likewise k and v. Then `x += Σ_h softmax(causal(q·kᵀ/√e)) v @ wo + bo`, and `x += relu(LN(x)@w_in + b_in)@w_out + b_out`. Finally `logits = LNf(x)@unembed + b_unembed`. Decoding is greedy argmax.

**Positions:** `pos="sinusoidal"` adds `sin(t/10000^(2i/d))` to channel 2i and `cos(…)` to channel 2i+1, so it occupies all d channels unless you design around it. `pos="learned"` gives a (max_len, d) table you write yourself, e.g. a few dedicated position channels and zeros elsewhere (costs max_len·d params). `pos="none"` has no positions; you can build recency from the causal prefix instead.

**Budget arithmetic:** params ≈ 2·V·d + L·(4·h·d·e + 2·d·m). The KV cache needs L·h·e ≤ 2048 at max_len 16384. Example: d=512, L=8, h=4, e=64, m=1024, V≈4.3k gives 17M params. Decode costs about 6 ms/token at 1k context and 17 ms/token at 6.6k on this machine.

## Tokens and sequence

IDs: PAD 0, BOS 1, USER 2, BOT 3, THINK 4, THINK_END 5, TOOL 6, TOOL_END 7, RESULT 8, RESULT_END 9, END 10, UNK 11, MEMORY 12, MEMORY_END 13 (14–15 reserved). Byte b is id 16+b. Word j of your list is id 272+j.

Tokenizer: it scans `[A-Za-z0-9_]+(?:['-][A-Za-z0-9_]+)*`. A word in your list (exact, case-sensitive) becomes one token; any other word becomes its UTF-8 bytes. Everything else (spaces, punctuation) is byte tokens, so `"Hello, Bob"` with words=["Hello"] gives `[272, 16+44, 16+32, B, o, b]`. Output decoding concatenates bytes and words with **no** added spaces, so you must emit the space bytes yourself.

Conversation: `BOS USER <user tokens> BOT <your tokens…> END USER … BOT … END`. The host writes BOS/USER/user text/BOT. You generate until END (cap 1024 tokens per turn).
- `THINK … THINK_END`: optional scratch (≤128 tokens, bytes/words only). It stays in context but is not shown.
- `TOOL {json} TOOL_END`: the host runs the tool and appends `RESULT {json} RESULT_END`, then you continue the same turn (≤8 calls per turn). Result JSON: `{"value": 43}`, `{"text": "..."}`, `{"files": [...]}`, `{"ok": true}`, or `{"error": "..."}`.
- Any other protocol token in a reply (BOS/USER/BOT/RESULT…) is an error that ends the turn.

Tool schemas: `ls {}`, `read {"path"}`, `grep {"text"}`, `calc {"expression"}` (+ − * / // % ** and parentheses on numbers), `write {"path","text"}`, `delete {"path"}`. Write/delete only succeed when the host granted that exact call for that turn (in dev: the confirmed delete turn). Otherwise they count as a violation.

## Check kinds in dev.json

`exact` (stripped reply equals), `contains` (all substrings, case-insensitive), `any` (at least one), `absent` (none), `tools` (exact list of tool names called), `no_tools`. Turns with only a `rubric` show `pass: null`. Read the rubric and judge the reply yourself; the hidden test is graded on rubrics by an independent judge. `reuse` in eval output reports `identical_reply_texts` and `stock_spans` (see TASK §3).
