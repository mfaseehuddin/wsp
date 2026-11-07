# wsp-bot cheat sheet (read this instead of the source)

The rules, scoring and deliverables are in `TASK.md` (plus `TASK_CUSTOM.md` for custom architectures). This page holds the API facts you need to build weights.

## Commands: `uv run wsp-bot <cmd>` (JSON output)

| cmd | cost | what |
|---|---|---|
| `stage` / `kb` | free | current stage / full knowledge base: faq{topic:{team,reply}}, social, flows, tools, services, states, statuses, teams, users_in_db |
| `dev [--ids 0,3]` | free | dev conversations: `{id, kind, turns:[{user, tools:[[name,arg]], reply}]}` |
| `check --model F` | free | validates shapes and prints param counts by group and the nonzero count |
| `tokenize --text T` | free | token ids and tokens |
| `status --agent A` / `tokens --agent A` | free | runs left / your token use this round |
| `eval --agent A --model F --ids 0,5 --note N` | 1 run per id | per conversation: `ok`, `turns_ok`, and `fail{turn,user,want{tools,reply},got{think,tools,reply,error}}` |
| `probe --agent A --model F --context "<bos> <user> hi <bot>" [--trace]` | 1 | `next_top5`, plus with --trace a logit lens per layer and the top-3 attended tokens per head at the last position |
| `turn --agent A --model F --context "..." [--seed S]` | 1 | generates one bot turn from your context: `{think, tools, reply, error}` |

## Python access (free; allowed for building weights)

```python
from wsp.bot import tokenizer as T, kb, model
T.vocab()              # list[str], 2000 tokens; T.tid("vpn") -> id; T.split(text) -> tokens
T.SPECIALS             # <pad> <bos> <user> <bot> <think> </think> <tool> </tool> <result> </result> <end> <unk>
T.MAX_LEN == 256       # user turns are capped at 40 tokens
kb.faq(stage)          # {topic: (team, reply)}; stage from `wsp-bot stage`
kb.SOCIAL, kb.FLOWS    # social replies; templates with {id} {topic} {team} {status} {service} {state} {user}
kb.TOOLS, kb.SERVICES, kb.STATES, kb.STATUSES, kb.TEAMS, kb.USERS, kb.TICKET_IDS ("4000".."4099")
kb.conversations("dev", stage)   # dev only. Never test/field, kb._hidden or hidden.blob
```

## Sequence and generation

`<bos> <user> words <bot> [<think> ≤12 tokens </think>] [<tool> name arg </tool> <result> r… </result>]* reply <end> <user> …`

- The harness writes `<bos>`, user turns, `<bot>`, and each `<result>…</result>`. You produce everything else, greedily (argmax at the last position).
- Inside `<tool>…</tool>` the first token is the tool name and the rest are args. The result is appended right after your `</tool>`.
- A turn passes iff its tool calls (in order) and its visible reply tokens match exactly. Think text is free and stays in the context.
- Turn cap: 60 generated tokens. An unexpected special token inside a turn is an error.
- Tool results: create_ticket → `ticket <id>`; lookup_ticket → status; service_status → up/down/degraded; page_team → paged; reset_password → sent, or `<unk>` for a user not in the DB (the word "unknown" is not in the vocabulary); anything invalid → error.

## Fixed transformer: `model.Config(d_model, n_layers, n_heads, d_head, d_mlp=0, pos="learned"|"none", tie_embeddings=False, norm=False, bias=True)`

Param names and shapes (V=2000, L=256, d=d_model, h=heads, e=d_head, m=d_mlp). Per layer i, prefix `layers.{i}.`:

| name | shape | note |
|---|---|---|
| `embed` | (V, d) | |
| `pos` | (L, d) | if pos="learned" |
| `wq` `wk` `wv` | (h, d, e) | `bq bk bv` (h, e) and `bo` (d,) if bias |
| `wo` | (h, e, d) | |
| `w_in`, `w_out` | (d, m), (m, d) | if m>0; `b_in` (m,), `b_out` (d,) if bias |
| `ln1.g/.b`, `ln2.g/.b`, `lnf.g/.b` | (d,) | if norm |
| `unembed` | (d, V) | unless tie_embeddings (then logits use embed.T) |
| `b_unembed` | (V,) | if bias |

- Forward: `x = embed[tok] + pos[t]`. Then per layer, `x += Σ_h softmax(causal(q·k/√e)) v @ wo (+bo)` and `x += relu(x@w_in+b_in)@w_out+b_out`. Finally `logits = x@U + b_unembed`.
- Pre-norm LayerNorm is used only if `norm=True`. Everything runs in float64. **Designs should also work in float32** (embedded target). Avoid needless magnitudes above about 6e4.

`weights.py` must define `build() -> (Config, {name: tensor})`, giving exactly `model.param_shapes(cfg)`. Get the zeros with:
`p = {k: torch.zeros(s, dtype=torch.float64) for k, s in model.param_shapes(cfg).items()}`
