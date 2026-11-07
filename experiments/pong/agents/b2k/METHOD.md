# b2k method: how the 2,000 forward passes were spent

## Before any forward pass (cost 0)

1. I read `game.py` and derived the exact crossing formula `t* = fold(y + 1.5·c·(1−x)/v)` (see EXPLANATION.md).
2. I chose the architecture to fit the 3-hidden-layer limit. Layer 1 gives `c·a` by polarization plus a PWL `1/v`.
   Layer 2 gives `(c·a)·(1/v)` by polarization. Layer 3 gives an exact fold.
   I picked the knot spacings (1/v: 12 geometric segments; squares: h = 1/8 and h = 0.1) from analytic chord-error bounds.
   I then worked out the error budget (≤ 0.0065 in t) and the reachability argument showing every rally should be a hit.
3. I wrote `policy.py`. I instantiated `build()` once to confirm the layer shapes (62, 62, 10). This ran no forward pass.
   For that structural check I called `wsp.pong.harness.validate` on the built model. It inspects only types and shapes
   and runs no forward pass, but it is a harness function and I note it here for full disclosure.
   I did not call `game.play`, `harness.evaluate` or the network outside the CLI at any point.

## Harness calls, in order (also in `log.jsonl`)

| # | call | cost | cumulative | what I was testing | result | change made |
|---|---|---|---|---|---|---|
| 1 | `probe`, 48 rows | 48 | 48 | Unit-level correctness of the whole predictor. 8 hand-picked extremes (`x = 0`, `v = 0.5`, `|c| = 1` → `|d| = 3`, wall-adjacent, late-rally) plus 40 Halton points over the input box. Because `up = t − p − θ`, each row gives back the network's internal target `t = up + p + θ`. I compared it with the reference formula computed independently in `scratch/probe_rows.py` (task formula, not the network). | Max \|t − t*\| = 0.0028 (at a = 0.97), median 0.00053. Extremes were exact. Rows with a < 0.15 had error ≤ 0.0008. Consistency `down + up = −2θ` held to 3·10⁻¹⁷. | None. The errors sit inside the analytic bound and shrink as a → 0, as derived. |
| 2 | `eval`, 25 rallies, offset 0 | 910 | 958 | First end-to-end test with the controller and dead band θ = 0.015. | mean loss **0.0**, hit rate **1.0** | None |
| 3 | `eval`, 27 rallies, offset 10000 | 953 | 1911 | Confirmation on a disjoint block of the dev set. It is the most informative use of the remaining budget, since there was no error left to tune away. | mean loss **0.0**, hit rate **1.0** | None |
| 4 | `probe`, 9 rows | 9 | 1920 | (a) Dead-band boundary. With `c = 0` the network's t equals y exactly (P1 = 0 and z3 = z4, so both multiplication errors cancel), so I set `p = y ± 0.0149` and `y ± 0.0151`. Expected actions: stay, stay, down, stay, up. (b) Late wall-adjacent rows where `u` crosses 0 or 1. (c) The midpoint of the widest 1/v segment (v = 0.5147, the worst case for the reciprocal). | (a) Actions matched exactly (1, 1, 0, 1, 2), and the `p = y` row gave stay. (b) t = 0.971 and 0.029, both exact. (c) Error +0.0024 at c = 1 and −0.0012 at c = −0.5, consistent with the 0.09 % relative bound on 1/v. | None |

Total used: **1,920 of 2,000**. The last 80 passes, about 2 rallies, were not worth spending. They could not change any decision.

## Why I did not tune constants

The only free behavioural constant is the dead band θ. Its value, half a paddle step, follows from the step size, and the
analysis shows every rally is a hit for any θ well below 0.08 − ε. With dev loss already exactly 0, a zero-order search over θ or the
knot counts could not show any improvement. I spent the budget on independent confirmation instead: 52 dev rallies, 1,863 passes,
all hits, loss 0.

## Files

- `policy.py`: final network.
- `EXPLANATION.md`: weight-group derivations, error bounds, failure conditions.
- `scratch/probe_rows.py`: generates the probe rows and the reference `t*` (task formula only).
  `scratch/probe1.json` holds the raw output of probe 1.
