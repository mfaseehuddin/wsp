# c1000: how the hand-written help-desk bot works

## How the bot works (plain language)

1. Every word the bot can say has a fixed spot on a **dial**, like an hour hand on a clock face. Its angle is the word's number in the vocabulary × 360°/2000. To say a word, the bot turns an output dial to that word's angle.
2. When a user message arrives, the bot first writes a private note, `<think> MODE TOPIC </think>`, using two made-up code words, for example `abiotic aglow` = "FAQ about VPN".
3. MODE and TOPIC are chosen by a **keyword vote**. Every word of the latest user message votes, for example "vpn" votes for VPN and "urgent" votes for paging. The word with the most votes is written.
4. After the note, the rest of the turn follows a **fixed script** taken from the knowledge base. A small "next word" rule fires for each step. It checks the mode, the topic, the previous word and the current word, then turns the dial to the next word.
5. When the script needs a value it cannot know in advance (a ticket id, a user name, a tool result), a rule opens a **copy gate**. A copy head then finds the most recent such value in the conversation and copies its dial angle.
6. The notes stay in the conversation, so a later "it still does not work" can look up the topic in the last note and open a ticket about it.

## The parts

The model is a standard transformer: 2 layers, 6 heads of size 12, d_model 70, 369 MLP units per layer, learned positions, no LayerNorm, untied unembedding. The residual stream is split into named slots (`LAYOUT` / `D` in `weights.py`):

| slot | dims | written by | meaning |
|---|---|---|---|
| pcos, psin, plin | 0-2 | `pos` | position as an angle (π/256 per step) and as a ramp t/256 |
| CUR | 3-4 | `embed` | dial code of the current token (cos, sin of 2π·id/2000) |
| f_user f_bot f_bos f_mode f_topic f_id f_uname f_rv f_gate | 5-13 | `embed` | yes/no flags: token is `<user>` / `<bot>` / `<bos>` / mode code word / topic code word / ticket id / user name / tool-result value / "read the vote here" |
| INT | 14-24 | `embed` | intent votes (11 modes) |
| TOP | 25-48 | `embed` | topic votes (22 FAQ topics + internet + phone) |
| s_topic s_mode s_xp s_nsvc | 49-52 | `embed` | "forbid" signals that switch off unsuitable choices |
| U, in_user | 53-56 | layer-0 heads 1, 2 | where the current user turn starts; whether we are inside a user turn |
| PREV, p_id p_uname p_rv | 56-60 | layer-0 head 0 | the previous token's dial code and class |
| MODE, TOPC | 61-64 | layer-0 heads 3, 4 | dial code of the latest mode and topic code word |
| OUT | 65-66 | MLPs + copy heads | the output dial (what to say next) |
| g_id g_uname g_rv | 67-69 | MLP gate units | copy gates |

**Dial (embed CUR, unembed OUT).** Token *y* reads `S_OUT·code(y)` from the OUT dims, so the logit is `1e9·cos(angle difference)`. The word at exactly the dial angle wins. Its nearest neighbour is ~4,900 logits behind.

**Layer-0 attention (lookups).**
- *Head 0, previous token.* The query is the position angle rotated back one step, scaled by 4e5. The key is the position angle. The score `4e5·cos(π/256·(t−1−s))` peaks only at s = t−1. This head copies the previous token's CUR and its class flags into PREV / p_*.
- *Head 1, start of turn.* Score = `1e5·f_user + 7680·plin`, which picks the most recent `<user>`. It copies that position's angle into U.
- *Head 2, inside a user turn.* It finds the most recent `<user>`/`<bot>`/`<bos>` and copies f_user into in_user.
- *Heads 3 and 4, memory.* Each finds the most recent mode code word (f_mode) or topic code word (f_topic) and copies its CUR into MODE or TOPC.

**Layer-1 heads 0-2 (keyword bag).** The score is `4e5·cos(U_t − U_s)` (same turn start), −1000 for tokens outside a user turn, and a `<bos>` escape. That gives the average of the votes (INT, TOP) over the current user turn. The `<bos>` escape is +8e5 unless f_gate is set, so the bag only switches on at `<think>` and at the faq, urgent and service code words. Elsewhere the head looks at `<bos>`, whose value is zero. Unembed rows INT→mode code word and TOP→topic code word (×10) turn the vote into an argmax. The forbid signals add −1e4: `<think>` forbids topic words, mode words forbid mode words, faq/urgent forbid internet/phone, and service forbids non-service topics. Small `b_unembed` values break ties (1e-3 × priority for modes, 1e-4 for topics).

**MLP units (next-word rules), `w_in`/`b_in`/`w_out`.** One unit per script step, with key (mode, topic, previous word, current word). Unneeded parts are left out (`_units` picks the least specific unambiguous key). Each tested part adds `S_CODE·code·(slot)` with S_CODE = 10/δ ≈ 2e6, or 2·flag for classes like "any ticket id". The bias is −(sum − 1), so the unit outputs exactly 1 when everything matches and ≤ 0 otherwise. `w_out` writes the next word's code into OUT, or a 1 into a copy gate. Gate units sit in layer 0, the other units fill layer 0 and then layer 1. There are 738 units in all: 9 gates, 31 for the think section, and 698 script steps.

**Layer-1 heads 3-5 (copy).** Score = `1e5·gate·flag + 7680·plin + 1e5·(1−gate)·f_bos`. With the gate open, the head takes the most recent ticket id / user name / result value. With it closed, it takes `<bos>` (zero). The value is the source token's CUR, written into OUT.

## Where knowledge lives (edit map)

All of these are generated in `weights.py`. Edit the named source and rebuild. Dial angles are tied to vocabulary ids, so no other word moves.

| fact | source in weights.py | tensors touched |
|---|---|---|
| FAQ reply text | `kb.FAQ[topic][1]` → `_scripts()` | the units for that topic's steps: `layers.{0,1}.w_in/b_in/w_out` columns whose key has `TOPC = code(TOPIC_CW[topic])`; new words also get `embed[w, CUR]` and `unembed[OUT, w]` |
| topic keyword | `TOPIC_KW[topic]` | `embed[word, TOP+i]` and `embed[word, INT+0]` (faq intent); i = index in `TOPICS_X` |
| intent keyword | `INTENT_KW[mode]` | `embed[word, INT+MODES.index(mode)]` |
| topic → team | `TEAM` (from `kb.FAQ`) | the units keyed (esc/urgent, TOPC=topic, …) whose `w_out` points at the team word |
| help-desk number 4357 | inside the FAQ reply text | the two units whose `w_out` = code("4357") (locked, mfa); retarget them to code("4400") |
| tool name | `_scripts()` (e.g. "create_ticket") | the unit (mode, cur=`<tool>`) → tool word, and the unit (mode, prev=`<tool>`, cur=tool) |
| new FAQ topic | add to `kb`/`TOPICS`, give it a code word in `_CW`, add `TOPIC_KW` | a new `TOP` dim is needed (d_model +1), or reuse the internet/phone layout pattern |
| flow templates | `kb.FLOWS` → `_scripts()` | units for that mode |
| user names / ids / statuses | `kb.USERS`, `NON_USERS`, `TICKET_IDS`, `STATUSES`, `STATES` | `embed[w, f_uname/f_id/f_rv]`, plus vote for reset/status |

Units are placed in `_all_units` order. To keep the parameter diff small in a learning round, append new units at the end instead of reordering. Changed units are edited in place.

## Limits

- **Parameters: 445,010** (embedding 140k, unembedding 140k, positions 18k, 2 × 72.5k layers). Only ~9.4k are non-zero.
- Understanding is a bag of hand-written keywords. Unseen synonyms fall back to "unknown". Mixed messages such as "printer still not working" (topic word plus escalation word) can pick the wrong intent. Negation and word order are ignored.
- Escalation always uses the most recent note's topic (including urgent/service turns).
- With no ticket in the conversation, "what is my ticket number" has nothing to copy, and the turn loops until the length cap.
- The service name must be a literal service word or a listed synonym (outlook → email).
