# s60: explicit adder circuit in 13+4 inserted layers

## Path to the design
No repair of Qwen's own mechanism was attempted (not diagnosed). Instead a fresh, fully explicit circuit is written at the
embedding level in inserted layers, using scratch channels in the residual stream, and then un-computed so that every token
that is not part of an addition `A+B=` leaves the stack exactly as it entered (guard 1.0/1.0).

## Mechanism (all weights are closed-form)
- **Constant C=4096 on dim c0** (layer 0 attention, value bias, softmax weights sum to 1). It makes the RMS of the residual
  constant (~C/sqrt(896)) so all scratch channels are exactly linear. It is removed by the last layer (bias -C).
- **Scratch channels**: ~660 residual dims chosen where the 13 relevant tokens have ~0 embedding. Each neuron subtracts the '=' token's own
  baseline in its bias (calibration). Neurons are silu(K x)/K ~ relu(x) (K=30).
- **L0 classifier**: one neuron per digit, '+', '=': ridge LDA direction with u.e=1 and ~0 on the other 12 relevant embeddings.
  Writes valD (d+1), isD, isP, isE.
- **L1 banks**: 14 heads with constant RoPE-phase queries attend to offset d (peak margin ~16 logits from 18 frequency pairs,
  verified numerically for offsets <800). Values give "is '+'" (lb = digits of B), "non-digit" (la = digits of A, measured from '+'),
  "is '='" (distance i to nearest '='). MLP converts to first-hit one-hots.
- **L2/L3 gathers**: b_k read at offset k+1; a_k at offset lb+2+k via query = sum_d onehot_d*phase(d+1+k) (linear in the one-hot);
  la copied from '+' to '=' by a head at offset lb+1. Saturating-step cleanup of the gate flags; effective digits are
  relu(d - 40 + 20*valid flags) = digit if place exists else 0.
- **L4-L6**: G=[a+b>=10], P=[a+b=9]; carry c_k = OR_j (G_j AND P_{j+1..k-1}) with AND neurons relu(sum-(m-1));
  sum digit one-hot via triangle neurons on t=a+b+c mod 10; answer length n=M+c_M; trigger requires isEq, digits at both operands.
- **L7**: A[i][x] = s_{n-1-i}[x] AND n (MSB-first), plus a newline marker at i=n.
- **L8**: head i attends to offset i (answer position p_eq+i is offset i from '='), reads block A_i; gated by the first-hit one-hot
  of the nearest '=' so earlier lines' '=' cannot interfere. MLP writes ANS.
- **R1-R4 mirrors**: the same neurons and heads with negated output writes, in reverse order (MLP-only parts merged), erase every scratch channel and C.
- **Conversion** (inside R3): ANS_x -> U_x (min-norm solution of "logit of token x +B, others 0" under the final norm), ANS erased. U is
  written after all reads, so it never disturbs scratch.

## When it fires / silence / failure modes
Fires only where a '=' token is preceded by digits, a '+' and digits (and at the following answer positions). Elsewhere every
channel is zero and the mirrors cancel (measured residual ~3e-4 on code/prose). Failure modes: zero-shot prompts have no non-digit before A, so la
(measured from '+') is unknown (handled only approximately); operands >12 digits; non-digit separators between digits and '+';
false positives of the LDA detectors on exotic tokens; boost scale (GBOOST) tuned on dev only.
