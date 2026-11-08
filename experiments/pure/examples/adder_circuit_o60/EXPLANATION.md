# o60: a fresh exact adder grafted under Qwen2.5-0.5B

## Path to the design
- Weights first (no passes). The model already lines up digits for short sums. Its failures (off-by-one in the
  middle places) are carry-chain failures, and a carry chain needs lookahead over all lower places. A new circuit
  that does the lookahead exactly is easier to verify than a patch on an internal mechanism we only partly know.
- So the graft is a self-contained calculator in **10 new layers inserted before original layer 0** (indices 0-9;
  original layers are 10-33). It reads clean token embeddings, computes the sum, and at each answer position it
  writes ("parks") a vector `LAM*v_d` with rms 1e4. Every original layer RMS-normalises its input, so its output is
  bounded and cannot outweigh the park. After the final norm and the tied unembedding, the park alone sets the
  argmax. `v_d` is `E_d*g_final` minus its least-squares projection on competitor rows (the other digits, '\n',
  and the top-20 violators, found in two rounds). Margins are 19-33 logits for every digit and for '\n'.
- Debugging went in this order: (1) the harness stores read arrays in fp16, so I decoded MLP activations, not tiny
  residual directions. (2) Float32 cancellation in gates with huge constants: the error grew like ~2e-6*M, so all
  scales were chosen to keep M <= 5.4e3. (3) The rounding of the constant vector C*u0 leaked into every read row,
  so the constant now sits on the exact axis 221, and every read and write direction has an exact 0 on that axis.
  (4) Gates built from stored flags drifted with RMS, so they were replaced by token-detecting functionals that
  are exact at any RMS.

## Shared conventions (they make every number checkable)
- `B`: 320 orthonormal directions orthogonal to the embeddings of the 12 *relevant* tokens ('0'-'9', '+', '=')
  and to axis 221. They are the lowest-variance directions of the vocabulary embedding covariance inside that
  complement. Each quantity `q` owns one column `u_q` and is stored as `sigma_q * value * u_q`. At relevant tokens
  the embedding contributes exactly 0 to a read, because the directions are orthogonal to those embeddings.
- Constant `C = 1000` on axis 221. Layer 0 adds it and layer 9 removes it, each with a bias-only head
  (`v_proj.bias[0]=1`, `o_proj[:,0]=+/-C e_221`). The attention weights sum to 1, so the output is exactly
  that constant. Every inserted RMSNorm therefore divides by about C/sqrt(896). A read row `u_q*rmsC/sigma_q`
  returns the value in units, and the "one" row `e_221*rmsC/C` returns 1.
- Neuron = `silu(G*y)*one/G = relu(y)` with G = 200. `step(y>=n)` is two neurons,
  `2relu(y-n+0.75)-2relu(y-n+0.25)`. For integer-valued y it gives exactly 0 or 1.
- **Silence gate**: each computing neuron adds `M*(<w,e_x>-1)` to its pre-activation. `w` is the min-norm vector
  in span(relevant embeddings, axis 221 zeroed) with `<w,e_r> = 1+e_r[221]/C` on its target tokens and 0 on the
  other relevant tokens. On target tokens the term is exactly 0, and on any other token it is `<= -M*margin`.
  The margins are P 0.36, E 0.35, P/E 0.32, DIG 0.42, DIG/E 0.19 (worst token over the whole vocabulary).
  M = 2*(worst-case read of the neuron's inputs on a non-relevant token)/margin + 50. So a neuron can fire only
  on its token type. At non-relevant positions, every MLP output is exactly 0.

## Groups (layer: what / why the numbers / when it fires / how it stays silent / failure modes)
1. **normaliser C** (L0 attn, L9 attn): see above. It is always on and removed exactly, so its only trace is
   float rounding of e[221], about 1e-4.
2. **token flags** (L0 MLP, 24 neurons): for each relevant token r, `z=(<ê_r,e_x>-O_r)/(S_r-O_r)`.
   S_r is r's own dot and O_r is the best other token in the whole vocabulary, so r gets z=1 and every other
   token gets z<=0. A ramp on z writes DIG, VAL=d, P or E at scale 1. L8 recomputes the same neurons with negated
   outputs, which cancels the flags exactly. Failure mode: none seen. All detector margins are >= 0.035 in dot
   units.
3. **offset fetch** (L1 attn). Heads 0-12 (d = 1..13) use constant q/k biases. The score is
   `40*sum_{i<6} cos(w_i(D-d)) + 480 cos(w_15 D)`, with `q=(8A cos dw, -8A sin dw)` per RoPE pair and
   `k=(1,0)`, using the rotate-half convention `score = q1 cos(Dw) - q2 sin(Dw)`. The peak at D=d beats every
   other D <= 4000 by >= 32. The pass-5 attention map shows weight 1.000 at offset d. Values copy
   (DIG, VAL, E, P) of the token d back (sigma 3e-3). If offset d does not exist (start of text) the head
   attends elsewhere. Only the zero-shot A operand hits this, and group 6 masks it.
   Head 13 (Z): content key `30*(1-DIG-P)` with value P. When only digits precede '+' it returns exactly
   1/(la+1). Otherwise it attends to non-digit tokens and returns about 0.
4. **anchor logic** (L1 MLP; at '+'/'=' only):
   - `VLD_p = [sum_{d<=p+1} FD_d >= p+1]`: the run of digits reaches place p.
   - `ZS_p = [ZO >= (1/(p+1)+1/(p+2))/2]`, at '+' only: the zero-shot length test.
   - `PB = sum_d [FP_d + sum_{x<d} FD_x >= d]`, at '=' only: the first non-digit before B is '+'.
   **answer position** (at digits): `h_d = [FE_d + sum_{x<d} FD_x >= d]` writes R += 1 and K += d.
   K is the distance to the '=' just before the answer.
5. **zero-shot mask / digits** (L2, at '+'/'='):
   - `MM_p = relu(FV_{p+1} + 12 VLD_p - 12 - 12 ZS_p)` is the digit at place p, or 0.
   - `AVm_p = [VLD_p - ZS_p >= 1]`.
   Both use scale 0.1 and fire only at the two anchors.
6. **copy A** (L3 attn): the '=' attends to the most recent '+'. The key is `5000*P` (RoPE pair 31, which is
   effectively unrotated) plus the recency term `-3250 sin(w_15 D)`. That term falls 5 per position, so the
   nearest '+' wins by >= 25. A '+' anywhere in the last ~2000 tokens beats every non-'+'. It copies MM and
   AVm (24 values) into AM and AV (sigma 1e-3).
7. **generate/kill** (L3 MLP, at '='): with `x_j = AM_j + MM_j`, the neurons compute `g_j = [x_j >= 10]` and
   `k_j = [x_j <= 8]`. They write `CP_p += (g_j - k_j)*2^j` for every p > j.
   `V = [PB + AV0 >= 2]` marks a valid "digits+digits=". `GL_p = [L >= p]` is the OR of the two validity bits.
8. **carries** (L4, at '='): `c_p = [CP_p >= 1]`. The sign of `sum_{j<p}(g_j-k_j)2^j` is the sign of the highest
   place that is not a 9-sum, which is exact carry lookahead. Values stay below 4096, so float32 is exact. The
   neurons write `SC_p += c_p`, `SC_{p-1} -= 10 c_p` and `CC_p = c_p`. As a result the residual holds
   `s_p = AM_p + MM_p + SC_p = a_p + b_p + c_p - 10c_{p+1}` as a linear read.
9. **export sums** (L5, at '=' and only if V): `XS_p = relu(s_p + 30V - 30)` and `XN = n = L + c_L`, where the
   top carry is `[CC_p + GL_p - GL_{p+1} + V >= 3]`. The scale is 0.1, so a copy head reading them can never
   amplify embedding contamination. L8 recomputes these neurons negated, which cancels them.
10. **copy sums** (L6 attn): answer positions attend to the most recent '=' (same key design as group 6, on E).
    They copy XS, XN and XV into QS, QN and QV.
11. **place index** (L6 MLP, at digits/'='): `F_p = [n - 1 - K >= p]` for p = -1..12. The gate
    `20*(R + E + QV - 2)` requires a valid problem and an answer position.
12. **select** (L7): `Y = sum_p silu(G(F_p - F_{p+1} - 0.5))*QS_p*2/G`. When F_p - F_{p+1} = 1 the SwiGLU product
    passes `QS_p` exactly. Otherwise it gives 0.
13. **park** (L8): `[F_0 >= 1]` writes `LAM v_0`, `[Y >= d]` writes `LAM(v_d - v_{d-1})`, and
    `[F_{-1} - F_0 >= 1]` (one step past the last digit) writes `LAM v_newline`.
    **unflag** (L8): cancels groups 2, 5 and 9.

## When it is silent / known failure modes
- At non-relevant tokens every MLP neuron is gated off. Attention scratch remains at scale <= 3e-3 per unit.
  With the parks removed, guard equals the unedited build.
- It fires on any "digits+digits=" in ordinary text: it writes the true sum, then forces '\n'. If the text continues
  differently, each such event costs about 20-30 nats.
- Wrong on purpose: negative numbers, decimals, and multi-term sums ("1+2+3=") are read as the last two operands.
  So are operands with more than 12 digits.
- The recency keys wrap after about 2000 tokens of context.

## Verification (dev, greedy, exact match)
| test | acc |
|---|---|
| 12-digit fewshot (n=256) | 100% |
| 6-digit fewshot, seed 5 (n=256) | 100% |
| 3-digit + 11-digit zero-shot, seed 6 (n=256) | 100% |
| zero-shot, own set: lengths 1-12 each, unequal, plus edge cases 0+0, 9..9+1, 1+9..9, 999999999999+999999999999 (n=256) | 100% |
| 32 prompts in 9 unseen styles (prose/code prefixes, other few-shot lines, 700-token prefix) | 32/32 |
| 8-digit fewshot, before the last precision fix | 98.4% |

Guard (dev): ppl_ratio 1.1267, top1 0.743. The harness's **unedited** build (graft returns without touching the
model) already scores 1.1257 / 0.7476 on this machine, so the graft itself costs about +0.001 ppl ratio and
-0.005 top-1. With the park neurons removed, guard equals the unedited build exactly.
