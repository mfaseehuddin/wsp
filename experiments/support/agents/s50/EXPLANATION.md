# s50: explanation of every parameter

**Config:** d_model 11, 1 layer, 2 heads, d_head 4, d_mlp 245, learned pos, tied embeddings, no LayerNorm, bias on.
**Parameters: 12,377** (embed 5,170 · pos 704 · layer 6,033 · b_unembed 470). Of these, 2,878 are nonzero.

## Residual layout (why d_model = 11)
| dims | name | content |
|---|---|---|
| 0-4 | ID | identity code of every token the bot can emit, plus `<sep>`: 126 unit vectors from {-1,0,1}^5 with 3, 2 or 1 nonzeros, so max cosine is 0.82. 5 dims is the smallest size where this ±1 pattern family has ≥126 members. |
| 5-8 | CODE | intent codes: 14 unit vectors from {-1,0,1}^4 with 2 nonzeros (pairwise cos ∈ {0, ±½, −1}). Embedding rows hold `code_X − code_unknown`, and head 0 writes its evidence here ×A. |
| 9 | STR | keyword strength: strong 1.0, weak 0.5, out-of-domain 0.75, neutral 0 |
| 10 | P | position t from `pos`. Head 1 subtracts the `<sep>` position, which leaves the reply step n. |

## Tensors
- **embed (470×11), tied with unembed.** Reply tokens get an ID code. Hand-listed keywords get CODE+STR (lists in `weights.py: KW, OOD`). All other rows are zero. Because unembedding is tied, the logit of token w is ID(w)·S·code(next): the emitted token gets cos 1 and every other token ≤ 0.82.
- **pos (64×11).** Only column P is nonzero, with pos[t,P] = t.
- **Head 0 (intent).** bq[0,0]=√4 makes the query constant. wk[STR]=120 and wk[P]=−1.6 give score = 120·strength − 1.6·t. Strong keywords therefore beat weak ones, weak beat neutral, and when strengths tie the **earliest** keyword wins. That ordering also stops reply words (e.g. "password" inside the wifi reply) from overriding the query. wv copies CODE, and bv = code_unknown, so neutral tokens vote "unknown" and a query with no keyword attends only to neutrals and comes out unknown. wo writes the result to CODE ×A (A=2000), so the token's own CODE (scale 1) changes the intent match by at most 1e-3.
- **Head 1 (sep locator).** Key = 100·(ID·code_sep): `<sep>` scores 100 and every other token ≤ 82, so attention on `<sep>` is ≈1. wv reads P, and wo writes −1 into P. After the layer, P = n = t − t_sep: 0 at `<sep>` and k at the k-th reply token. Its unused d_head dims are zero.
- **bo, b_out, bk, b_unembed.** Zero. They are required by `bias=True`. Bias is needed for bq/bv/b_in, and turning it on costs less than a constant dimension would.
- **MLP (w_in 11×245, b_in, w_out 245×11).** There is one neuron per (intent X, k), with k = −1 … len_X−2: ReLU(n − k − 50·(1 − code_X·CODE/A)). For the detected intent the gate term is ≈0, so these are ramps in n. For any other intent code_X·I ≤ ½, which subtracts ≥24, and every ramp stays off for n ≤ 22. w_out row = 2e5 · (second difference of the reply's ID codes), i.e. v₀ for k=−1, v₁−2v₀ for k=0, v_{k+1}−2v_k+v_{k−1} otherwise. The sum is the piecewise-linear interpolation equal to 2e5·ID(reply_X[n]), so step n emits reply token n and the final step emits `<eos>`. The 245 neurons are the 14 reply lengths summed.

## When it fails
- No keyword of the right intent is listed (paraphrases that use unlisted words), so the reply falls to **unknown**.
- Two keywords of equal strength but different intents sit 1–3 tokens apart. Both gates partially close and the output is garbage (no reply). Further apart, the earlier one wins.
- An IT word inside an off-topic question (e.g. "is the coffee machine screen broken") is classified as IT, unless an OOD word with strength 0.75 comes early enough.
- Precision limit: the intent weight must be ≥0.999. With the current margins this holds except in the ties above.
