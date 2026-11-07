# c100: a help-desk bot whose weights are written by hand

## How the bot works (plain language)

1. Every word has a small **name tag** (a random code), plus a few marks: "this is a ticket number", "this is a user name", and **votes**, such as "vpn" voting for the VPN topic.
2. When the user finishes typing, the bot **averages the votes** of the words in that message only. The topic or action with the most votes wins. A message with no votes falls back to "unknown".
3. The bot writes the winner as one secret code word inside `<think> … </think>`, for example `alumina` = VPN. That code word is its **sticky note**. Later steps, and later turns, look back at it to know what they are doing.
4. Each kind of answer is a fixed **script**. The bot counts how many steps it is past the `<bot>` marker, and the pair (sticky note, step number) picks the next word of the script.
5. Some words in a script are **blanks**: a ticket id, a user name, a service state. To fill one, the bot looks back for the most recent word with the right mark and copies it. Ticket ids and states come from the tool result. Topics and teams come from the latest FAQ sticky note.
6. Tool calls are part of the script, so `<tool> page_team desktop </tool>` is just more script words plus a blank.
7. After `reset_password` the tool result itself (`sent` or `<unk>`) acts as a new sticky note, which switches to the "link sent" or "could not find" script.

Size: **742,256 parameters**, 2 layers. All 70 dev conversations pass.

## The parts

### 1. Embedding: word tags, votes and marks (`embed`, 2000 × 153)
- **CODE, columns 0–47:** a random unit vector for each word, seed 100. The output layer is tied to the embedding, so the same vector is also how the bot *says* that word. Register words and `<bos>` have CODE = 0.
- **VOTE, columns 48–81:** one column per intent, in the order of the 22 FAQ topics in KB order, then hello, thanks, bye, unknown, escalate, recall, status, service, urgent, reset, reset_sent, reset_unknown. A keyword's entry is 0.1 × its hand-written weight from `KW` in `weights.py`. Weights are 2 for a core word, 1 for a supporting word and 3 for an action word ("still", "down", "urgent"). User names add 0.5 to *reset*, ticket ids add 0.5 to *status*, and `<user>` adds 0.04 to *unknown*, which is the fallback. Because action words weigh more than topic words, "urgent: vpn …" becomes urgent and "is email down" becomes a service check.
- **Register words** (`REG` in `weights.py`, odd dictionary words such as `alumina`) have a 1 in their own VOTE column, and the marks `f_areg` (columns 120) and, for FAQ topics, `f_treg` (121). FAQ registers also carry their team as a one-hot in TEAM (columns 140–144).
- **Marks (columns 116–126):** `<bos>`, `<user>`, `<bot>`, `<think>`, ticket id, status word, state word, user name, service word. Service words and synonyms ("outlook" → email) have a one-hot in SVC (columns 145–152) that says which service they mean.
- **Position (`pos`):** column 127 (P1) = t. Nothing else is stored there.

### 2. Layer 0 attention: "where am I?" (3 heads)
Each head finds the **most recent** token with a given mark. The score is 10000·mark + 20·position + 7000·[is `<bos>`], so the newest marked token wins by at least e^-20, and `<bos>` is the fallback when no token has the mark.
- Head 0 copies the position of the latest `<user>` into SU (column 128).
- Head 1 copies the position of the latest `<bot>` into SB (column 129).
- Head 2 copies the VOTE one-hot of the latest register word into I (columns 82–115). This is the **current intent**.

### 3. Scripts as "ramps" (MLP layer 0, units 0–304, and MLP layer 1, units 0–320)
Step number d = t − SB. For intent i and step j, one ReLU unit computes R_j = relu(d − j + 1 + 1000·(I_i − 1)). It is zero unless the intent is i. The difference R_k − 2R_{k+1} + R_{k+2} is exactly 1 when d = k and 0 otherwise. So unit j stores the **second difference** of the script's output vectors: v_j − 2v_{j−1} + v_{j−2}. Each v_k is 40 × CODE(word), or a copy gate for a blank. This needs one unit per script word, and no lookup table of step numbers.
- Layer 0 holds the fixed `<bot>`→`<think>` units 0–2, FAQs password to printer, all social replies and every intent that uses tools. Script outputs from layer 0 also set the `dirty` mark (column 130), so a copy never reads from a position whose word tag is mixed with an output.
- Layer 1 holds the other FAQs (software to restore).
- Script steps come from `turn_positions()`: the harness-written `<result> … </result>` slots are skipped. FAQ scripts start at step 3, after `<think> REG </think>`.

### 4. Layer 1 attention (3 heads)
- **Vote head (head 0):** score = 10·SU(key), so only tokens in the current user turn are kept, with equal weight. It averages their VOTE columns.
- **Copy-ID head (head 1):** gate columns 131–136 are opened by the ramp units at a blank. The score is 10000·gate·mark + 20·position − 10000·dirty, with a `<bos>` fallback. It copies 40 × the word tag of the newest ticket id, status, state or name. For a topic blank it copies from the newest FAQ register, mapping its VOTE one-hot to CODE(topic word).
- **Copy-payload head (head 2):** the same, for team (FAQ register TEAM → CODE(team word)) and service (SVC → CODE(canonical service)).

### 5. Choosing the intent (MLP layer 1, units 321–353)
- Units 321–351 fire only on the `<think>` token: relu(1e5·vote_a + 1e6·f_think − 1e6). They write into VOTE_a, and because the output is tied, that raises the logit of register word a. The largest vote wins.
- Unit 352: after an FAQ register, say `</think>`. It is switched off if the message had an "urgent" vote.
- Unit 353: in that urgent case, write the urgent register instead. Urgent turns therefore think `<think> TOPIC URGENT </think>`, so the topic, and with it the team, is remembered.

## Where knowledge lives (edit map)

| Fact | What to change |
|---|---|
| A FAQ reply's text | Edit the KB or the reply string, then rebuild. Only that topic's ramp units change (see the unit ranges below). To edit weights by hand: unit j's `w_out` CODE part = 40·(code(w_j) − 2code(w_{j−1}) + code(w_{j−2})). |
| A new FAQ topic | Add it to `INTENTS` with a new register word, its VOTE column and an I column. This grows d_model by 2, or you can reuse spare columns. Add its keywords to `KW`, its team to the register's TEAM, and its ramp units. |
| A topic keyword | `embed[token, 48 + topic_index]` = 0.1 × weight (`KW` dict). |
| Topic → team | `embed[REG[topic], 140 + team_index]` (TEAMS order: network, desktop, accounts, messaging, security). |
| Help-desk number 4357 | It appears inside the *locked* and *mfa* replies, so it lives in those topics' ramp `w_out` rows: layer-0 units 21–42 (locked) and layer-1 units 41–64 (mfa). Change the reply string and rebuild. |
| A tool name, e.g. `page_team` | The ramp unit for that script step. urgent is layer-0 units 251–275, step 5 = `page_team`. |
| Service synonyms | `SVC_SYN` → `embed[word, 126]` (mark) and `embed[word, 145 + k]`. |
| Social / flow reply text | hello 125–142, thanks 143–155, bye 156–166, unknown 167–180, escalate 181–207, recall 208–217, status 218–233, service 234–250, reset 276–282, reset_sent 283–293, reset_unknown 294–304 (all layer-0 units). |
| Users / ticket ids / statuses | The marks `f_name` / `f_tid` / `f_status` (columns 125 / 122 / 123) on those tokens. |

Layer-0 FAQ units: password 3–20, locked 21–42, vpn 43–60, wifi 61–79, email 80–103, printer 104–124. Layer-1 FAQ units: software 0–18, slow 19–40, mfa 41–64, disk 65–83, monitor 84–101, audio 102–120, camera 121–140, teams 141–159, onedrive 160–179, access 180–197, browser 198–215, battery 216–239, bluetooth 240–257, phishing 258–279, newhire 280–299, restore 300–320.

## Limits
- **742,256 parameters.** Most are the 2000 × 153 tied embedding (306k) and the two layers (198k each). About 130k are nonzero.
- Understanding is a bag of keywords. New phrasings that use none of the keywords, or that mix topics ("my outlook is suspicious"), can pick the wrong topic or fall back to "unknown". Negations are not understood.
- Words outside the 2,000-word vocabulary become `<unk>`, so a user name that is not in the vocabulary cannot be copied.
- Recall only finds the newest *clean* ticket id. Several tickets in one chat would return the latest one.
- "Help" is weak evidence for escalating. "can you help with vpn" is answered as a VPN FAQ, which is the intended behaviour.
