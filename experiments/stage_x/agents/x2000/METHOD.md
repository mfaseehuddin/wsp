# METHOD (x2000)

Design first on paper (about 100k tokens of reasoning): pointer-free copying by induction, sentence retrieval by word signatures, frames chosen by MLP rules over gathered flags. Built weights.py in one pass, then debugged with `chat`/`probe --trace` plus a small local script that ran layers 0-6 of the built tensors for a few positions (debug only).

## Run batches
- check builds (free) x ~8: sanitizer fixes (no assert/pop/any/islower/starred).
- probe/chat singles (~40 runs): found (1) recency flag swamped by recency term (fixed with huge key scale), (2) BOS sink had position-dependent kernel noise (fixed by zeroing position-0 code), (3) frame registers fired at every position (gated by BOT flag), (4) tq dilution (carried from the previous token), (5) stop registers shared across heads (split), (6) bigram cross-terms in the copy key (orthogonal permutation Q), (7) ack literal fired at any space (added punctuation context).
- eval on 41 dev conversations (1 batch): 19/42 then 20/42 (after retrieval threshold 100 and protocol-token copy penalty) mechanical passes, 0 safety violations, reuse: 1 identical reply (fixed by shortening the unknown reply to 6 words).

## Not done
Calc tool, file tools, ordering/syllogism reasoning, instruction formats, greetings variety. Ran out of token budget before these.

## Known defects at stop
Copy loops ("txtx...") when the copied sentence has no final punctuation and repeated letter pairs; retrieval across many earlier turns returns unrelated sentences (code cross-talk noise); calc/tool use absent; hand-copied acknowledgements echo the user's own sentence in first person.
