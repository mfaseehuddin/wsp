# c1: a help-desk bot whose weights are written by hand

## How the bot works (plain language)

1. Each word in the vocabulary is given a fixed card (its embedding). The card lists the word's identity, what kind of thing it is (a ticket id, a user name, a service, a team, a topic name), and, if it is a keyword, which topic or request it points to and how strongly.
2. Layer 0 gets the bearings. One head looks at the previous word. One finds where the current turn starts and whether we are on the user's side or the bot's side. Two more remember the latest ticket id and the latest user name or service name seen so far.
3. Layer 1 decides what the user wants. One head picks the user's strongest "action" word in the current turn (a ticket id means status, a user name means reset, "urgent", "down"/"outage"). Another picks the strongest topic or social word ("vpn", "headphones", "thanks", "still"...). A third head finds the topic the bot noted privately in its last `<think>`. A fourth remembers the latest state or status that a tool returned.
4. The layer-1 MLP is a set of scripts, with one unit per line of a script: "if the intent is X, the current word is Y (and the previous word is Z), say W next". Slots such as {id}, {team} or {user} are filled by units that also check which value the copy heads found.
5. In each FAQ answer the bot first writes `<think> topic </think>`. That note stays in the chat, so a later "it still doesn't work" can open a ticket about the right topic.
6. Nothing is trained. Every number comes from `weights.py`, which builds it from the knowledge base plus keyword lists written by hand.

## The parts

Every residual dimension has a name. `weights.build.space.idx` maps each name to its column. Config: d_model 576, 2 layers, 4 heads, d_head 74, d_mlp 836, learned positions, tied embeddings (the output layer reuses the embedding), no LayerNorm, biases on.

**Embedding `embed[token, :]` (also the output layer).**
- `W:<word>`: a one-hot dimension for each of the ~250 words the bot can say or must recognise in its scripts (plus 30 spare). The same column is the output direction for that word.
- Ticket ids use `tens0-9` and `units0-9` (4049 means tens4 + units9). User names use `U:<name>`.
- Class flags: `isID`, `isUSR`, `isSVC`, `isST` (state or status), `isTopic`, `isTeam`, `isUserTag` (`<user>`), `isBotTag` (`<bot>`).
- Keyword fields: `vote:<class>` is a one-hot over 25 topic slots and 5 pseudo-classes (recall, escalate, hello, thanks, bye). `team:<team>` is the owning team. `sT` is the strength divided by 10. Action keywords use `fvote:<flow>` (status, reset, urgent, service) and `sF`.
- Strengths. Strong synonyms (e.g. "suspicious") are 6.0 to 6.7, topic names 5.0 to 5.7 and synonyms 3.9 to 4.6, each topic offset by 0.03 × its index so that two topics never tie. After that come recall 3.6, thanks 3.5, bye 3.45, escalate 3.4, hello 3.3 and "ticket" 3.2. In the action head, ids are 5.0, user names 4.8, urgent words 4.6 and down/outage 4.4.

**Positions `pos[t]`**: `pos1 = t/256` and `pos2 = (t/256)²`.

**Layer 0 attention** (each head copies dimensions with value/output weights equal to 1):
- Head 0, previous token. The score is 60·((t−1)² − (s−(t−1))²), built from pos1 and pos2. It peaks at s = t−1, and every other position loses at least 60 nats. The head copies the identity of the ~74 words and classes that the scripts need as the "previous word" into `PV:*`.
- Head 1, segment. The score is 1e5·[token is `<user>` or `<bot>`] + 100·position, so the latest tag wins by 100 nats per position. The head writes `seg` (the tag's pos1) and `botside` (1 if that tag is `<bot>`).
- Head 2, latest ticket id. The score is 1e5·isID + 100·position, and the head copies the digits to `IDc_*`.
- Head 3, latest user name or service. Same scoring on isUSR + isSVC, copied to `USc:*` and `USsvc:*`.

**Layer 1 attention:**
- Head 0 (F, action) and head 1 (T, topic/social). Key = 1e5·seg − 1e6·botside + 4000·strength. Tokens from earlier turns lose at least ~3900 nats. Bot-written tokens are invisible. Within the current user turn the strongest keyword wins, and the 0.03 gap between strengths is 12 nats. With no keyword the output is 0, which means "unknown". F writes `Fout:*`. T writes `Tout:*` and `Tteam:*`.
- Head 2 (TH, think topic). The score is 1e5·(isTopic + previous-is-`<think>`) + 100·position, so it finds the latest note in a think block. It writes `THout:*` and `THteam:*`.
- Head 3 (ST): the latest state or status word, written to `STc:*`.

**Layer 1 MLP (the scripts).** Each script lists the tokens after `<bot>`. For example, escalate is `<tool> create_ticket {topic} </tool>`, then the harness adds the result, then `i opened ticket {id} about {topic} . the {team} team ...`. Each unit is ReLU(sum of matching features − (count − 1)), so it fires (value 1) only when all of its features hold.
- The features are the intent, the current word, optionally the previous word, and, for slots, the copied value. The intent feature is `Fout:f` for actions. For topic and social intents it is `Tout:c` − 2·ΣFout, so an action always overrides. "Unknown" is 1 − 2·ΣFout − 2·ΣTout.
- The previous word is used only when the current word alone is ambiguous within that script (for example "." in a two-sentence reply).
- Each unit writes +20 to the output column of the next word. Ticket ids get +20 for tens and +20 for units, so the right id scores 40 and any near miss scores 20. Leftover overlap from the embedding is at most about 5, so the right token wins by more than 14.
- The layer-0 MLP is all zeros. It exists only because d_mlp is shared between layers.

## Where knowledge lives (edit map)

`S = weights.build.space.idx` gives the column for each named dimension. `id(w)` is the token id.

| Fact | Where to edit |
|---|---|
| FAQ reply text for topic t | MLP units of intent t (`layers.1.w_in/b_in/w_out` columns whose `Tout:t` weight is 1). The easy way is to change the reply in the template and rebuild: `templates()` generates the units, and the unit count only changes if the reply length changes. A changed word only needs `w_out[unit, S['W:old']]` moved to `S['W:new']`, and the next unit's `w_in` row moved the same way. |
| A topic keyword w | `embed[id(w), S['vote:t']] = 1`, `embed[id(w), S['team:<team>']] = 1` and `embed[id(w), S['sT']] = strength/10` (in code, the `SYN` dict). |
| Team that owns topic t | `embed[id(t), S['team:*']]` and the same field on every synonym of t (TH and T copy it). |
| New FAQ topic (stage 1) | Use a spare slot `spare0..2`. Rename it in `tclasses`, set the keyword rows, and add script units for its reply. New reply words go into the spare `W:spare*` columns: set `embed[id(word), S['W:spareK']] = 1`. |
| Help desk number 4357 → 4400 | Put the output on `W:4400` in the two units that now write `w_out[j, S['W:4357']]` (locked and mfa replies, after "on"). Also move the `w_in[S['W:4357'], j']` of the units that follow it to `W:4400`, which is a spare column if 4400 has none. In code, just edit the reply text. |
| Link expiry 15 → 30 minutes | Password-reply unit after "in": `w_out` moves from `W:15` to `W:30`, and the next unit's input moves from `W:15` to `W:30`. |
| Tool name | `W:create_ticket` etc. These are the output columns of the `<tool>` units of each flow. |
| Action words (urgent, down...) | `embed[id(w), S['fvote:<flow>']]` and `S['sF']` (the `FLOW_KW` dict). |
| Social / escalate / recall words | `embed[id(w), S['vote:<class>']]` and `S['sT']` (the `PSEUDO_KW` dict). |

## Limits

- **4,597,320 parameters** (8,721 of them nonzero). The embedding is 1.15M, each layer 1.65M, and about 0.96M of that is the unused layer-0 MLP.
- **Single-keyword understanding.** The bot does not read sentences. One keyword decides, so it fails when a phrasing has no known keyword, or when a stronger keyword from another topic is present ("teams meeting has no sound" goes to teams, not audio).
- Social words lose to any topic word. Escalate words ("still", "help", "tried") would trigger a ticket in a turn that has no topic word.
- **Reset needs a known user name.** It works only if the name is one of the 28 user tokens in the knowledge base.
- **Recall and status use the latest ticket id** in the chat, and escalation uses the latest `<think>` topic.
- **Possible harness problem.** The tool result `unknown` is not in the vocabulary (`T.tid('unknown')` raises an error), so conversations that reset an unknown user may crash in the harness. The bot keys that branch on `<unk>`, in case the harness maps the word to it.
