# ARCHITECTURE: a table-driven script interpreter (custom `arch.py`)

## What it is (data flow)
```
tokens -> lexicon lookup (lex.id -> lex.feat: 33 yes/no features per token)
       -> pointers: last <user>, last <bot>, last decision point (<bot> or </result>) via cummax of positions
       -> bag-of-cues of the CURRENT user turn (masked matrix MU) -> indicators -> AND-rules -> intent (softmax of priorities)
       -> slots: n-th flagged token of the user turn (hard attention with rank == n)      \
       -> V1/V2: key-at-line-start lookup in result number r (hard attention), HIT, ANS, PREVF, RES0 |  value vector sv (18)
       -> script: (intent, calls done c, step k) -> literal token or sv entry (call.*), or reply template (rt.map, tmpl.*)
       -> optional copy of the newest result (pointer = start + (k - k0)) -> target token id
       -> logits = -gain * min(|vocab_id - target|, cap)
```
No hidden state and no KV cache: everything is recomputed from the token ids, which is exactly how the harness runs the model.

## Why it suits this job
- Every folder is new and values must be copied from context; copying is explicit hard attention (a masked sum), exact, with no
  training noise. Arithmetic is delegated to the calc tool, so no number circuit is needed.
- 38,440 parameters (7,098 non-zero) against a 5M cap; a 4096x d transformer with embeddings alone would need hundreds of thousands.
- Editability: one concept = one table (intent rules `flow.*`, scripts `call.*`, sentences `tmpl.*`, word lists `lex.*`),
  each generated from a short list in `weights.py`. A new tool, a protected name, or a new sentence is a list edit (Edit map).
- Safety is structural: intent and arguments only read the user turn (mask `MU`), `unsafe` has no call row, `delete` only exists in
  the script that needs a pending confirm plus a yes.
- Compute per token: dominated by a few (T x T) masked sums for T <= 1024 (~12 matrices of 1M floats, a few ms on CPU);
  everything else is (T x 20..200) matrix products.

## What it gives up
- It is not learned and is not a generic language model: only phrasings whose cue words are in the word lists are understood
  (unknown phrasing -> `unknown` reply). Multi-token values and calls with more than 3 args need table extensions.
- Intent rules are an explicit decision list; two unusual cues in one turn can collide (priorities resolve it deterministically).
- The custom forward is more code than a stock transformer (about 130 lines), and is not a standard layer stack.

## Embedded fitness
- float32 memory: 38,440 x 4 B = 154 KB of parameters (28 KB if only the 7,098 non-zero values were stored).
- Ops per token: O(T^2 * 12) multiply-adds for the masks (<= 12M at T=1024) plus ~0.2M for tables; no exponentials except one 20-way softmax.
- Float32 works (this is what the harness runs): largest magnitudes are positions/ids (<= 4095), logits within +-120, softmax logits
  <= 2000 (priority 100 x gain 20, saturating safely). Nothing approaches 6e4. Exact comparisons use `|a-b| < 0.5` on integer-valued floats.
