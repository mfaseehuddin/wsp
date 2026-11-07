# c50: a help-desk bot whose weights are written by hand

## How the bot works (plain language)

1. Every word gets a fixed "ID card" (a list of numbers) plus a few labels, such as "this is a ticket number", "this word means VPN (strength 7)" or "this word means *urgent*".
2. The first layer works out three facts at each position: which word came just before it, which user message it belongs to, and whether it is inside a bot reply.
3. The second layer reads only the **latest user message** and asks: "what is the most important cue here?" Cues are ranked: urgent > outage > ticket number > "my ticket number" > a user name > "still not working" > a topic word > weak complaints > bye > thanks > hello. Nothing found means "unknown".
4. A second reader finds the **most recent topic word** the user said in any turn. That is how "it is still not working" knows which topic to open a ticket about.
5. Every reply is stored as a chain of words. A small table of "neurons" says, for each reply, "if this is the chosen reply and the last two words were X Y, the next word is Z". The bot types the reply one word at a time with that table.
6. Values that change per chat (ticket ids, user names, service names, states, statuses) are not stored. When the table reaches such a slot, a **copy reader** finds the most recent word of that kind in the chat and repeats it.
7. Tool calls are ordinary words in a reply chain (`<tool> service_status SVC </tool>`). The harness then adds the `<result>`, and the chain carries on from what the result says (for example `sent` versus `<unk>` for a password reset).
8. The bot never uses the think area. Everything it needs to remember is still in the chat, and the readers find it there.

## The parts

Residual stream width is 216. The dims are grouped into named blocks; `S[name]` in `weights.py` has the indices.

| block | dims | written by | meaning |
|---|---|---|---|
| IN | 0-47 | embed | word ID card (48-d random unit code) |
| OUT | 48-95 | embed, MLPs, copy head | "what to say next". The tied unembedding reads it |
| PREV | 96-119 | L0 head 0 | ID card of the previous word (first 24 dims of its code) |
| PFL | 120-126 | L0 head 0 | class labels of the previous word |
| FL | 127-133 | embed | class labels: ID, USER, SVC, STATE, STATUS, TOPIC-name, TEAM |
| ISUSER/ISBOT/ISBOS | 134-136 | embed | marks `<user>`, `<bot>`, `<bos>` |
| ISKW, KWT, TPV | 137, 150, 151-172 | embed | is-topic-word, strength/8, which of 22 topics |
| KWF, FLV | 138, 139-149 | embed | cue priority/110, which of 11 cue channels |
| P, P2 | 173, 174 | pos table | position p and p² |
| U | 175-176 | L0 head 1 | (p, p²) of the latest `<user>`, i.e. which turn |
| INB | 177 | L0 head 2 | 1 if inside a bot turn |
| FLOW | 178-188 | L1 head 0 | winning cue channel of the current message |
| TL | 189-210 | L1 head 1 | latest topic (22 one-hot) |
| GATE | 211-215 | L0 MLP | "copy a word of class c now" |

**Embedding (`embed`, tied with the unembedding).** Every word the bot can see while answering, or can say, has a random 48-d unit code, written into both IN and OUT. Other words have no code. Labels are 0/1, and scores are divided down so they are at most 1. This keeps them from adding noise to the output scores.

**Positions (`pos`).** Only two columns are used: p and p². With them an attention head can score "how far is key j from position t" as a quadratic, −μ(j−t)², which turns attention into an exact pointer.

**L0 head 0, previous word.** Score −10·(j−(t−1))², so it attends only to position t−1. It copies that word's 24-d prev-code and class labels into PREV/PFL.
**L0 head 1, which turn.** Score 5000·[is `<user>`] + 10·j, so it picks the latest `<user>`. It copies (p, p²) into U.
**L0 head 2, inside a bot turn.** It works the same way, but over `<user>` and `<bot>`. It copies "was it `<bot>`" into INB.

**L0 MLP (282 units).**
- Units 0-10 are **copy triggers**, one for each place a reply needs a copied word. For example, (current word `lookup_ticket`, previous word `<tool>`) sets GATE[ID].
- The other units are **literal reply units** for word pairs that occur in only one place across all replies, so they need no decision check.
- Each unit is `ReLU(10·(match_cur−1) + 10·(match_prev−1) + 1)`. It fires (value 1) only when both words match exactly. The threshold needs a match above 0.9, but different codes overlap by at most 0.60, so look-alike words cannot trigger it. Its output adds 50 × (code of next word) to OUT.

**L1 head 0, flow (what kind of request).** The score adds three parts:
- `110·KWF`, the cue priority (gaps of 10 between ranks);
- `−5·(u_j−u_t)²`, a penalty for any other user message, at least 180;
- `−5000·INB`, which ignores the bot's own words.

The value is the cue's channel, so FLOW is close to a one-hot of the strongest cue. Topic words are a channel too (priority 50). Social and weak-complaint cues win only when no topic word is present.

**L1 head 1, latest topic.** The score is `3000·ISKW + 10·u_j + 48·KWT − 5000·INB`: topic words only, latest user turn first, then the strongest word in that turn. TL is that topic's one-hot.

**L1 heads 2-3, copy (24 dims each).** The score is `3000·Σ_c GATE_c·FL_c(j) + 5·j + 2000·(1−ΣGATE)·[j is <bos>]`. When a gate is on, the head attends to the latest word of that class and writes 50 × its code into OUT. When no gate is on, it attends to `<bos>`, whose code is zero, so it writes nothing.

**L1 MLP (282 units), reply table with a decision check.** The unit for (reply r, step k) has pre-activation `10·(D_r−1) + 10·(match_cur−1) + 10·(match_prev−1) + 1`. D_r is 1 only when r is the chosen reply:
- FAQ τ: `TL_τ + FLOW_topic − 1`
- hello / thanks / bye: `FLOW_x`
- unknown: `1 − ΣFLOW`
- escalate: `FLOW_esc + FLOW_escweak`
- the others: `FLOW_r`

Slot units for TOPIC and TEAM add `+TL_τ − 1` (or the sum over that team's topics), so the unit picks the right topic or team word.

## Where knowledge lives (edit map)

All edits go in `weights.py`. Rebuild with `build()`. Row r of `embed` is token id `T.tid(word)`.

- **A FAQ reply** comes from `kb.faq(stage)` and is rebuilt automatically. It lives in `layers.{0,1}.w_in[:, u]`, `b_in[u]` and `w_out[u, 48:96]` for the units of that template (see `split_units()`). To change one word, change the reply text. Only the units at and next to that step change: their match columns and the w_out row (50 × new word code). A new word also needs a code row: `embed[row, 0:96]`.
- **A topic keyword** lives in `TOPIC_WORDS[topic][word] = strength` → `embed[row(word)]` at cols 137 (ISKW = 1), 150 (strength/8), 151+topic_index (TPV = 1), plus 138 (KWF = 50/110) and 139+6 (FLV "topic" channel). That is 5 numbers in one row.
- **A flow cue word** (urgent, outage, still, thanks, …) lives in `FLOW_WORDS` → `embed[row]` cols 138 (priority/110) and 139+channel.
- **A new FAQ topic** needs one new TPV and TL dim: grow the `("TPV", 22)` and `("TL", 22)` blocks. Then add its keywords, its reply units (automatic) and its slot units for TOPIC (escalate). Its team comes from the KB.
- **The team of a topic** comes from `kb.faq` → only the TEAM slot units' decision columns: `layers.1.w_in[189+topic_index, u]` for the 5 escalate and 5 urgent team units (×2 steps).
- **The help-desk number (4357)** is a literal word in the locked/mfa replies. Change the reply text, which changes the w_out row of the unit before it and the match column of the unit after it (plus a code row if the number is a new word).
- **A tool name** (`create_ticket`, …) is a literal in the flow templates in `templates()`. Its units are the one emitting it (`<tool>` → name, w_out) and the one after it, or the copy trigger in `layers.0` units 0-10.
- **Ticket ids, user names, services, states and statuses** are class labels `embed[row, 127:132]` (from `CLASS_TOKENS`). A new user name needs that label, a code and the "reset" cue (col 138 = 70/110, FLV channel 4).

## Limits

- **Parameters: 956,324** (embed 432,000; pos 55,296; each layer 233,514; b_unembed 2,000). About 103k are non-zero.
- Understanding is keyword-based. A message with no known word becomes "unknown", and a misleading word can win (for example "slowed down" reads as an outage).
- "help", "working", "again", "fails", "same", "nothing" count as complaints only when no topic word is present. So "can you help me ?" with no topic opens a ticket.
- Escalation uses the most recent topic word the user said, even if that turn was an urgent or status turn.
- Words outside the vocabulary are `<unk>`, so a user name that tokenizes to `<unk>` cannot be copied.
- If two different reply words would follow the same two words inside one reply, `units()` raises an error. None do today.
