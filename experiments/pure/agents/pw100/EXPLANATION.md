# pw100: a hand-written "rule program" chat transformer

Family `transformer`: 11 layers, 16 heads of 64, d_model 1216, d_mlp 1664, ReLU, **no normalisation**, learned
positions (max_ctx 6144), tied embeddings. About 292M parameters, ~22M nonzero. Nothing is trained or fitted.
Every weight comes from a formula or a hand-written table in `model.py`. Last dev score: 24/33 mechanical checks.

## Core idea
The network runs a small program. It classifies the latest user turn into a *plan*, then emits that plan's reply
token by token. Some plans are fixed templates, and some include a copied span or a computed number. Three choices
make exact construction possible:
* `norm="none"`: the residual stream stays linear, so integer quantities (positions, counts, digits) stay exact.
* Learned positions: `pos_emb[p]` holds CONST=1, POS=p (a raw integer), BIASOUT=1, a 20-d sinusoid code (for
  exact offset heads) and a random unit 32-d code RPOS[p] (sentence identity).
* Tied output: the 128-d random unit identity code E[v] in `tok_emb` is also the output code. A plan writes
  30·E[token] into the same subspace, and that token wins by about 16 nats. `<|im_end|>` also gets a +4 bias
  (BIASOUT), so when nothing is written the reply simply ends.

## Residual subspaces (named, allocated in `build()`)
Token features (`tok_emb`): identity code; role-marker flags; one 0/1 dim per *concept* (≈340: intent words,
operators, entities); a keyword flag; digit flag and value×0.01; sentence-boundary / statement-closer / colon /
newline / terminator flags; span-trigger flags; "common lowercase word" flag; content flag; question-word flag;
and a 2-hot hash of the token over 58 buckets (word matching). Computed dims: turn fields, concept presences,
intent conditions, arithmetic registers, sentence fields, selections, copy gates.

## Layer by layer
**L0 attention.** Three recency heads (score 40000·marker + 4·POS) find the latest `user` / `assistant` marker:
U_POS (start of the current user turn), A_POS (start of the reply), ROLE (+1 user, -1 assistant). Offset heads,
with score 25/gap·Σcos(ω(p_q−s−p_k)) peaking exactly at p−s, read the previous 1–3 tokens: digit flags/values,
trigger-bigram flags, the previous identity code, and "previous token is content". The latest-boundary head gives
SB fields (boundary position, its RPOS code, colon and newline flags). Other heads track the latest terminator and
INQ (whether the latest boundary-or-question-word is a question word).
**L0 MLP.** Running value of a digit run (`d + 10·d1 + 100·d2 + 1000·d3`, gated by digit flags). Span triggers
(user tokens only): I'm, called, named, word, "name is", "call me", "I am". Two- and three-piece keywords fire on
their last piece through an exact bigram neuron relu(E[a]·prev + E[b]·own + 5·ROLE − 6.6).
**L1 attention.** Keyword-count heads: keys of the current user turn score 4·U_POS + 20·ROLE. The `user` marker
is a sink (+14) and keyword tokens get +9, so each keyword has weight e^-5 and the output ×e^5 is a **count** per
concept. Self-attending heads (20·POS) subtract the query token's own concept features. Offset heads copy the
previous token's value, its sentence fields (so each token knows SIDX, the boundary before its sentence, and that
sentence's code) and its word hash. Another head gives the latest trigger position.
**L1 MLP.** Exact presence clamp01(10·count−5.5) per concept. Q flag (question mark or question word). Number-end
tokens and their values. FT, the first terminator after a trigger (a name span ends there). BADN, a common
lowercase word right after a trigger ("I'm fine" is not a name). o² for the offset o = POS−SIDX−1.
**L2 attention.** Arithmetic operand heads (first/last number of the turn, the latest earlier number for "that
answer"). Number count. Word votes: each content word finds its latest earlier user occurrence. The key is the
token after the occurrence (PHK = hash of the previous token), the gain is 1e4 so an exact 2-hot match always beats
a partial collision, and recency is 0.5/token. Keys in question sentences are excluded. A FOUND head against a
threshold sink (the `user` marker at 0.75 of an exact match) confirms the vote. Other heads compute the per-sentence
question-word fraction and the number fraction, and count FT/BADN tokens.
**L2 MLP.** Intent conditions a_i = clamp01(Σ required presences − n + 1 − 3·forbidden), all exactly 0/1.
FACT_ANY/ENT_ANY flags. Operator one-hot by priority (× ÷ − + bigger smaller). The vote is kept only if it was
found, comes from a user token and is word-initial.
**L3–L4 attention.** Selections read at reply rows: H_NAME (latest FT of the turn → span start offset and
length); H_ECHO (the sentence after ':' in the turn); the vote sum of the turn. At L4, H_SENT picks the user
statement closer whose sentence code best matches the vote sum (gain 150, a threshold sink at 75, recency
0.01/token, and a fixed random tie-breaker so ties do not blend). L3 also fetches the previous answer.
**L3–L5 MLP.** Slot-plan conditions (name, echo, recall). Arithmetic: effective operands, sum, |difference|,
min/max, sign, floor division (99 steps), product = Σ_j relu(max − 2e4·relu(2j−1−2·min)) (exact for min ≤ 60),
the operator-selected result, and its digits by steep half-integer step counts. Plan winners.
**L6–L7 MLP.** INSLOT: whether this reply step is inside the copied span (from t = k − prefix length and the span
length). The copy query features are gated by INSLOT.
**L8 attention: copy heads.** Score = 250·(selected code · key sentence code) + 20·(2·TGT·o − o²) + 30·ROLE. The
quadratic peaks exactly at offset TGT, and neighbours lose 20 nats. When the step is not in the slot, the gated
query is zero and an `<|im_start|>` sink (+50, identity code 0) wins.
**L9 MLP.** Substitutions on copied tokens only: a pronoun swap for recall (my→your, I→you, 'm→'re, me→you) and
polite rewording (Give→give, "."→"?").
**L5–L10 MLP: template banks.** For a template s_0..s_{L−1}, L+2 ReLUs relu(k − m + MG·(gate−1) − MG·(1+ROLE))
with outputs 30·(E[s_{m+1}] − 2E[s_m] + E[s_{m−1}]) form the exact piecewise-linear interpolant of the token codes
over the reply step k = POS − A_POS − 1. The gate is the plan's own condition minus every higher-priority
condition, so it is exactly 1 for the winner and ≤ 0 for all others. The arithmetic answer uses the same trick
over the digit value, gated by a digit-place selector.

## Plans (priority order)
Harm refusal (word list, plus a lock+break rule for burglary) → echo/copy instructions (exact word, named story, summarize, polite, repeat) → arithmetic (bigger,
smaller, general) → recall of an earlier statement ("You told me that …") → "my …?" not told → ambiguous "it" →
unknowable private things → poem / story / tips / generic polite / generic summary → **facts** (two-concept,
e.g. YELLOW+fruit, LARGEST+planet) → **entity cards** (question naming a known entity) → self-intro, how-are-you,
empathy for "I'm feeling sad/stressed…", abilities, thanks/bye, name greeting ("I'm X", "my name is X",
"call me X"; the earliest name span of the turn; common lowercase/feeling words after "I'm" are not names), "Nice to meet you, X!", greeting, encouragement, joke, correction/remember
acks → statement ack → "I don't know" fallback.

## Knowledge
About 230 entity cards: 50 countries with capital/region/language, cities, continents, planets, about 50 animals,
nature/science, body, food, days and months, units of time, shapes, grammar words, famous people/inventions, and
about 40 adjectives with synonym and opposite. There are about 20 two-concept facts and about 30 intents. Each
card is one or two dense sentences, so one card answers several question types (e.g. legs, class and diet).
Arithmetic covers integers up to 4 digits per operand and results below 100000 (products need min ≤ 60).

## What it can and cannot do
Can: greet (including by name), keep a multi-turn conversation of any length (every turn is classified freshly
from the latest user message), answer covered facts, do + − × ÷ / bigger / smaller including "multiply that answer
by 3", recall earlier statements by word overlap (corrections win by recency), say when it does not know, decline
harmful requests, copy an exact word, write a story around a given name, rewrite politely, echo a summary.
Cannot: reason (ordering puzzles, syllogisms, precedence `25 − 4·3` gives 75), change case, reverse lists, count
words, compose novel text beyond templates, combine two recalled facts into one answer, or match inflected forms
(park/parked), or apply corrections that share no word with the original ("Actually, it's green" after "My bike
is blue" still recalls blue). Topics outside the cards fall back to "I don't know". Recall echoes a whole sentence rather than
answering briefly.

## Path
1. Stage 1: turn heads + keyword counts + priority banks → 10/33 on the first pass.
2. Arithmetic module. Fixed leaked digit-value logits (feature ×0.01) and float32 precision (steep half-integer
   steps; exact "operator off" flags).
3. Spans: name triggers, echo after ':', sentence copy with exact offset addressing.
4. Recall. Hashed sentence overlap was too noisy: the sink token carried a content hash, the reply rows matched
   themselves, and collisions caused false recalls. It was replaced by exact word votes with recency and a FOUND
   threshold → no false recalls in the 64-turn session.
5. Generalisation tests on my own unseen conversations (names like Xochitl/Arjun/Selma, corrections, harmful
   variants, weather) led to small rules: call-me names, earliest span, more not-a-name words, live-data
   "can't know", extra cards, burglary refusal, empathy. Passes 53-99 were mostly an unchanged-model stability
   sweep, which in hindsight wasted passes (greedy decoding makes the replies identical).
Each fix was found with `probe` (attention at the first reply token) plus reading the weights.
