# x500: how the bot works

## How the bot works (plain language)
1. Every word is just a number (its id). A small **dictionary table** (the "lexicon") says what each id means: "this word is evidence for the VPN topic", "this id is a ticket number", "this is the start of a bot turn".
2. At the start of its turn the bot adds up the evidence from the words of the user's last message. The best-scoring option wins. That option is written down in the private think area, e.g. `<think> vpn </think>` or `<think> ticket </think>` (ticket = "escalate").
3. The think word is the **script name**. The bot then reads that script out, one step at a time. A **program table** has one row per output word: "script vpn, step 3, say 'check'".
4. Some steps say "copy", not a fixed word: copy the latest ticket number, service name, user name, topic or result word from the conversation. Five small **copy heads** (look-back lookups) find these.
5. Tool scripts have two parts: the tool call (`<tool> lookup_ticket 4071 </tool>`) and, after the harness inserts the result, the reply ("ticket 4071 is waiting .").
6. The think word stays in the context, so later turns (escalate) find the topic of the last FAQ turn there. The ticket number sits in the tool result, so recall just copies it.
No training was used. Every number was written by hand or generated from the knowledge base (KB).

## The parts
- **Lexicon** (`lex_lo, lex_hi, lex_dim, lex_w`, 386 rows). A row means "a token with id in [lo,hi] adds `w` to feature `dim`". Dims 0-31 are keyword evidence per script (22 topics, hello, thanks, bye, other, escalate, recall, status, service, urgent, reset). Dims 32-49 are flags: turn markers, plan id, team id, "is topic word", "is ticket id", "is service", "is state/status/sent/unknown", "is user name", "what to say next". Maths: `box = relu(1-relu(lo-x)-relu(x-hi))`, features = box times weights.
- **Counters** (no params). Cummax and cumsum find the last `<bot>`, `<user>`, `</think>`, `</result>`, giving the step number `k` and flags "think closed", "result arrived".
- **Think phase.** `plan_bias`, `plan_id`, `beta`, `topic_mask`. A softmax over the evidence picks the plan token. After `urgent`, the same vote restricted to topics (via `topic_mask`) names the topic, so the team can be looked up.
- **Copy heads** (`head_req, att, slot_grid`). Hard attention (huge score for the matching flag, plus a small bonus for newer positions) fetches the newest topic (inside think text), ticket id, service, state/status word, user name. Team = the TM feature of the newest topic word.
- **Program memory** (`rom_plan, rom_after, rom_k, rom_res, rom_slot, rom_tok`, 532 rows). A row fires when plan, "after result", and step all match (product of triangular bumps). It votes for the literal token `rom_tok`, or, if `rom_slot` is 1-6, for the copied value (1 topic, 2 team, 3 id, 4 service, 5 state/status, 6 user). `rom_res` makes the two reset replies depend on the result word (`sent` or `<unk>`).

## Where knowledge lives (edit map)
- FAQ reply word: row(s) in `rom_*` with `rom_plan` = topic index+1 and `rom_k` = word position; change `rom_tok` (a vocabulary id). A longer reply needs more rows (append to all six rom_ tensors).
- Topic keyword: add/change a lexicon row (`lex_lo=lex_hi=word id`, `lex_dim`=topic index, `lex_w`=strength).
- Topic to team: lexicon row with `lex_dim`=42 (TM) for the topic word, `lex_w` = team token id.
- New topic: new plan slot (needs the 32-wide evidence dims widened), a plan token in `plan_id`, rows in lexicon and rom.
- Social replies, flow reply templates: rom rows with the matching `rom_plan` (hello 23, thanks 24, bye 25, other 26, escalate 27, recall 28, status 29, service 30, urgent 31, reset 32); `rom_after`=1 marks the post-tool reply.
- Tool name: rom rows for that script at `rom_after`=0, `rom_k`=1 (`rom_tok` = tool name id).
- Help-desk number 4357: the reply tokens of `locked` and `mfa` (one `rom_tok` each).
- Service / state / status / user name lists: lexicon rows with `lex_dim` 45 / 46 / 46 / 47.
- Unknown-message threshold: `plan_bias[25]` = 0.5.

## Limits
4,897 parameters (3,281 non-zero). Dev: 70/70 conversations. Weak points: phrasing with no keyword in my lists falls to "other"; "vpn is down" is treated as a status query; ambiguous words (e.g. "log in") are unmapped; a topic word used in an urgent turn becomes the "latest topic" for later escalations; only the first 256 tokens fit.
