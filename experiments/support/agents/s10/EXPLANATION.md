# s10: what every parameter does

**Config:** d_model=11, n_layers=1, n_heads=2, d_head=2, d_mlp=91, pos=learned, tie_embeddings=True, norm=False, bias=True.
**Parameter count: 8647** (embed 5170, pos 704, layer 2303, b_unembed 470). 1751 of them are nonzero.

## Residual dimensions (why d_model = 11)
| dims | name | content |
|---|---|---|
| 0,1 | A | identity and output code. The j-th of the 126 reply tokens (incl. `<sep>`, `<eos>`) is the unit vector at angle 2πj/126. All other tokens are 0. |
| 2,3 | B | default successor. This is the A-code of the token's most common next token across the 14 replies (a bigram table kept in the embedding). |
| 4 | K | keyword strength = tier + 0.05·priority(intent). It is the key of the intent head. |
| 5,6 | I | intent vote, the unit vector at angle 2πi/14. The intent head writes the winning vote back into these dims. |
| 7,8 | P | position, (cos ωt, sin ωt) with ω=2π/64 |
| 9,10 | H | A-code of the previous token, written by head 1 |

Every group is needed. A is needed for tied output and for cur identity. B is the cheap bigram store. K and I do intent detection. P and H give the previous token, which separates the repeated tokens inside a reply (".", ",", "the", "and", "it", "corp"). I is reused as the intent-result slot. This is safe because no token at a place where the gates read the intent has a vote of its own.

## Tensors
- **embed (470×11)**: holds A, B, K and I as above. Keyword lists are hand-written per intent in `KW`. The tiers are 6 for specific nouns (vpn, outlook, toner…), 5 for strong cues, 4 for weak cues, 3 for off-topic words (weather, lunch, pizza…; these vote *unknown*) and 2 for greetings and thanks. `<bos>` = 1 and votes unknown (the sink), and every other token = 0. The 0.05·priority offset breaks ties inside a tier, so two different intents never tie. Because embeddings are tied, this table is also the unembedding: logit[n] = x·E[n]. The A part dominates through the R=1e6 scale. The other dims add less than about 50 to each logit, while the gap between the target and the nearest code is R(1−cos(2π/126)) ≈ 1243.
- **pos (64×11)**: only P is nonzero. It is the rotating 2-D clock that head 1 uses.
- **wq/wk/wv/wo, bq/bk/bv/bo** (head 0, intent): the query is constant (`bq[0,0]=600√2`) and `wk` reads K. So score = 600·strength, which gives hard attention to the strongest keyword. Ties between tiers or intents are at least 30 logits apart. If the query has no keyword, `<bos>` wins and the intent is unknown. `wv`/`wo` copy I→I. (Head 1, previous token): `wq` = c·Rot(ω) on P and `wk` = c·I on P, with c² = 1e4·√2. That gives score = 1e4·cos(ω(t−1−s)), so position t−1 beats its neighbours by at least 48. `wv`/`wo` copy A→H. bk, bv and bo are zero; they exist only because bias=True.
- **w_in, b_in, w_out, b_out (91 neurons)**: the neuron names are in `build.neurons`.
  - 2 *transfer* neurons: relu(B_j+2)·R → A_j, and `b_out[A] = −2R` removes the offset. This copies the bigram successor code into the output exactly and linearly.
  - 14 *sep* neurons (cur==`<sep>` AND intent==i) write R·code(first reply token of i).
  - 75 *correction* neurons (cur==c AND prev==p, and also intent==i for the 15 contexts that are ambiguous across intents) write R·(code(true next) − code(B(c))). Every gate term is GH(A·a_c−1), GH(H·a_p−1) or GI(J·c_i−1). Each term is 0 on a match and at most −12.4 or −19.8 on a mismatch. With b_in carrying the +1, a neuron outputs exactly 1 or 0.
- **b_unembed (470)**: all zero. It exists only because bias=True, and bias=True costs less than an extra constant dimension.

## When it fails
- **Intent errors.** The query has no listed keyword even though it is in-domain, so it becomes unknown. Or a stronger keyword of another intent is present (e.g. "wifi password" goes to password: both are tier 6 and password has the higher tie-break priority). Off-topic words in tier 3 beat greetings but lose to any IT cue.
- **Contamination.** At reply positions the intent head also sees earlier reply tokens. This matters only at the 15 intent-gated contexts, and the static check in METHOD.md found no cross-intent keyword before any of them.
- **Truncation.** Queries over 24 tokens are truncated. If the keyword is in the cut part, the query becomes unknown.
