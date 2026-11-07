# Task: hand-write the weights of a tool-calling IT support chatbot

You are one of five agents. Each has a different budget of **runs**: 1, 10, 50, 100 or 1000. Your job is to write every parameter of a standard transformer that works as an interactive IT help-desk chatbot. The bot must:

- hold a multi-turn conversation, remembering earlier turns;
- think in a private scratch area;
- call mock tools and use their results in its reply.

No training is allowed. Look for a **novel** design. You are scored on how well the bot works, how small it is, and how few of your own tokens you spend.

Later, you will run 3 **learning rounds**. In each one, the same you gets new knowledge or bug reports and must update the weights **by editing them directly**, without retraining. Design for that from the start: know where every fact lives in weight space.

## 1. What the bot must do

Run `uv run wsp-bot kb` to see the knowledge base. In summary:

- **FAQ topics (22):** each has an owning team and a canonical reply.
- **Social turns:** hello, thanks, bye, and unknown for off-topic messages.
- **Flows** that use tools or context:

| Kind | Example user turn | What the bot must output |
|---|---|---|
| FAQ | "vpn is not connecting" | the FAQ reply |
| escalate | "it is still not working" (after a FAQ turn) | `<tool> create_ticket <topic of the most recent FAQ turn> </tool>`, then "i opened ticket {id} about {topic} . the {team} team will contact you today ." |
| recall | "what is my ticket number ?" | "your ticket number is {id} .", using the id from earlier in the chat |
| status | "any update on ticket 4012 ?" | `<tool> lookup_ticket 4012 </tool>`, then "ticket 4012 is {status} ." |
| service | "is email down ?" | `<tool> service_status email </tool>`, then "email is {state} right now ." |
| urgent | "urgent : vpn keeps disconnecting" | `<tool> page_team <team of that topic> </tool>`, then "i paged the {team} team . they will call you within 15 minutes ." |
| reset | "please reset the password for jsmith" | `<tool> reset_password jsmith </tool>`, then "a reset link was sent to jsmith ." if the result is `sent`, or "i could not find the user jsmith ." if it is `unknown` |

The 12 conversation shapes that combine these are listed in `harness.KINDS`. Run `uv run wsp-bot dev` to see all 70 dev conversations. Tool results come from a mock database (`kb.MockDB`) and differ per conversation, so the bot must **copy** ids, states and names from the context. It cannot memorise them.

## 2. Sequence format (`src/wsp/bot/tokenizer.py`, 2,000-word vocabulary)

```
<bos> <user> words <bot> [<think> up to 12 tokens </think>] [<tool> name arg </tool> <result> … </result>]* reply words <end> <user> …
```

- The harness writes `<bos>`, each user turn, `<bot>`, and each `<result> … </result>`.
- Your model writes the rest of each bot turn with greedy decoding: optional think text, tool calls, the reply, then `<end>`.
- **Think text is never scored, but it stays in the context**, so it can carry state such as the current topic or a ticket id to later turns.
- Only the tool calls and the visible reply are scored. They must match exactly. A conversation passes only if every turn passes.
- Maximum length: 256 tokens. User turns are capped at 40 tokens.

## 3. The model (fixed; you supply only the parameters)

`src/wsp/bot/model.py` is a standard pre-norm decoder:

- token embedding, plus optional learned positions
- multi-head causal softmax attention, with an optional ReLU MLP
- optional LayerNorm
- unembedding, which can be tied to the embedding
- `d_model ≤ 2048`, `n_layers ≤ 8`

You choose a `Config` and supply exactly the tensors in `model.param_shapes(cfg)`. **Every parameter counts, zeros included.** The embedding alone is 2,000 × d_model.

## 4. Budgets

**Runs.** One `eval` of one dev conversation costs 1 run. So does one `probe` (top-5 next tokens, with an optional attention and logit-lens trace) and one `turn` (generate one bot turn from a context you write). Requests over budget are refused at no cost.

**Your own tokens: 250,000 for the build.** This is measured as your context size, which grows with everything you read and write. Check it any time for free with `uv run wsp-bot tokens --agent <you>`.
- When you reach the limit, the coordinator will tell you to stop. You must then finish, saving your current best weights and the docs, within **100,000 more tokens**, or you will be stopped.
- Save working files early and often, so whatever is on disk is what gets scored.
- Don't read big files you don't need. Generate weights with compact code, not by writing them out. Never print large outputs.

## 5. Commands (`uv run wsp-bot …`)

**Free:**
- `kb`, `dev [--ids 0,3]`, `stage`
- `check --model F`: validates shapes and prints the parameter counts
- `tokenize --text "…"`
- `status --agent A`: runs left
- `tokens --agent A`: your token use

**Charged:**
- `eval --agent A --model F --ids 0,5 --note "why"`: per conversation, whether it passed, plus the first failing turn with the gold and actual tool calls, reply and think text
- `probe --agent A --model F --context "<bos> <user> hi <bot>" [--trace] --note "why"`
- `turn --agent A --model F --context "…" [--seed S] --note "why"`

## 6. Hard rules

1. **No training, backward pass, gradients or optimizers.**
2. **No fitting to data.** You may build weights programmatically from the **knowledge base** (replies, teams, tools, flows): it is the spec. Keyword and phrasing lists for understanding user messages must be hand-written by you. Don't derive them from dev conversations by counting or regression.
3. **The model runs only through the harness.** Don't call `model.forward`, the harness's run or scoring functions, or a reimplementation of them. Hand-checking a few individual dot products is fine.
4. **The hidden data is off limits.** Don't read or decode `src/wsp/bot/hidden.blob`, and don't call `kb._hidden`, or `kb.pools` / `kb.conversations` for any split other than `"dev"`. Use the current stage only (`wsp-bot stage`).
5. **No internet.**
6. **Stay in `experiments/bot/agents/<you>/`.** Don't touch `src/`, `tests/`, ledgers, `.transcript` files or other agents' directories, and don't set `WSP_ADMIN`.

## 7. Scoring

1. Exact-conversation rate on a **hidden test set** of 160 conversations, with new phrasings, at the current stage.
2. Fewest parameters.
3. Fewest of your own tokens.

After each learning round, the coordinator also reports:
- the new hidden-test score;
- how many parameters you changed (`diff` against your previous version);
- whether old behaviour still works.

## 8. Deliverables (in `experiments/bot/agents/<you>/`)

1. `weights.py`: `build() -> (Config, params)`, made from named constructions with comments.
2. `EXPLANATION.md`, **written for a human reader who is not an ML expert**:
   - **How the bot works**, in plain language, in about 10 lines. Start here.
   - **The parts**, one short section per component: what it does, which tensors and rows or units implement it, and why the numbers are what they are. Put the maths after the plain explanation.
   - **Where knowledge lives (edit map):** for each kind of fact (a FAQ reply, a topic keyword, a team mapping, the help-desk number, a tool name), give the exact tensors, rows and columns to change. This is your guide for the learning rounds.
   - **Limits:** the parameter count, and what the bot will get wrong.
3. `METHOD.md`: your design process, and every run you spent, in order: what it tested, what you learned, and what you changed.

Finish with a short reply: your parameter count, the dev conversations you verified and how many passed, the tokens you used, and a 3–5 sentence design summary.
