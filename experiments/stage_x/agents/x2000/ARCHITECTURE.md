# ARCHITECTURE (x2000)

Config: d_model 448, 7 layers, 3 heads, d_head 88, d_mlp 1024, learned positions (16384 x 448), no norm, bias on. Params 17.6M (about 377k nonzero; the position table and unused matrices are zeros). KV cache 231 MiB (7*3*88*16384*8 bytes). Max |weight| 9381 (key scales for the 2e6-gain recency heads), float32 safe.

## Residual layout (448 channels)
C 0-63 token code | Vb 64-127 previous-token context | O 128-191 output code | Tb 192-255 word signature / question bag | P 256-271 position code | TL 272 t/16 | F 273-288 flags (bos user bot bnd wordc content digit op qm sep dot excl lexw space) | R 289-447 registers: SPi, SPex, SPw, SPq, RC, RCt, SPqt (16 each), then scalar regs (word features, frame flags f_/g_ fast, s_/u_ slow, nomatch, stop_q, stop_a, relation regs).

## Layer plan
- L0: h0 previous token (Vb), h1 latest boundary (SPi), h2 latest word separator (SPw).
- L1: h0 carry SPi->SPex and Vb += 0.7 Q Vb(t-1); h1 word signature (Tb).
- L2: h0 carry Tb, SPq, pcont, qm; h1 second Vb carry (3-token context).
- L3: h0 question bag, h1 intent flags, h2 entity code; MLP: fast frames + fact hits.
- L4: h0 retrieval (RC, nomatch), h1 fetch fast frames; MLP: slow frames.
- L5: h0 start copy (QA), h1 fetch slow frames/RC/SPq, h2 start copy (ACK).
- L6: h0 induction copy (QA), h1 induction copy (ACK); MLP: text units, END, fact values.

## Composes / does not
Composes: frames x copy x fact table. Does not: arithmetic, tools, ordering/syllogisms, instruction formats, long replies. Code coherence (64-d, about 0.3 max cross product) limits copy reliability when letter n-grams repeat.
