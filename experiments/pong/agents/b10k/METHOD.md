# b10k: how the 10,000 forward passes were spent

## Before any forward pass

1. I read `game.py`. It gives `crossing = fold(U)` with `U = y + 1.5·b·(1−x)/a`, and shows that every target is reachable (14 steps needed at most, 25 available at least).
2. Design: multiplying the decision by `a > 0` removes the division. `a·fold(V/a)` is an exact sum of ReLU hinges, so only the products `a·y`, `b·x` and `a·p` need approximating. Each one comes from polarisation plus chord-interpolated squares.
3. I split hinges to equalise knot spacing within each layer and derived the error bound analytically: predicted crossing error ≤ 0.0074. Details are in EXPLANATION.md.
4. Checks without forward passes: building the model (no forward call) confirmed shapes 5→64→64→3 and that all 64+64 units are labelled. I hand-checked one square hinge (`(a+y)²` at `s = 0.65`: `0.25 + 1.15·0.15 = 0.4225 = 0.65²` ✓) and the fold coefficients (+1, −2, +2, …).
5. `scripts/probe_set.py` holds 16 hand-picked states and the analytic reference `fold(U)` for each (the task's reference, not the network). It decodes harness logits into the network's predicted crossing via `(up − down)/(2a) + p`.

## Harness calls, in order (the log is `log.jsonl`)

| # | call | policy (sha) | cost | cumulative | result | purpose / what I learned |
|---|---|---|---|---|---|---|
| 1 | probe, 16 rows | policy.py (79c14b9aa2af) | 16 | 16 | max \|pred − ref\| = 0.00625; all 16 actions as intended | Unit-level check of the whole predictor: no bounce, single/multiple bounces, range edges U = −3 and U = 4, paddle at both walls, very late states. The error is under the derived bound 0.0074, and the worst case is at `a = 0.5`, `|b| = 1`, as the bound predicts. No change needed. |
| 2 | eval 60 rallies, offset 0 | policy.py (79c14b9aa2af) | 2116 | 2132 | loss 0.0, hit 1.000 | Closed-loop check: the policy works end to end. |
| 3 | eval 30, offset 60 | candidates/deadband_000.py (DEADBAND = 0) | 1012 | 3144 | loss 0.0, hit 1.000 | Zero-order sweep of the dead band, low end. Removing it only adds dithering (±0.03), which is harmless. |
| 4 | eval 30, offset 60 | candidates/deadband_060.py (DEADBAND = 0.06) | 1012 | 4156 | loss 0.0, hit 1.000 | Dead band, high end. Still a hit, because 0.06 + 0.0074 < 0.08. The loss is flat over [0, 0.06], so I kept the derived value 0.015 (it settles closest to the target). |
| 5 | eval 30, offset 60 | candidates/coarse_grid.py (SEG_AY = 2, SEG_BX = 4, SEG_AP = 2, i.e. h = 0.75) | 1012 | 5168 | loss 0.00064, hit 0.967 | Margin test for the knot-spacing constants. Coarse grids (U-error bound about 0.18) start to miss, so grid resolution is what matters, and the chosen h₁ = 0.15 / h₂ = 0.056 grids have a large margin. I kept the fine grids. |
| — | eval 135, offset 90 | policy.py | refused (4964 > 4832) | 5168 | nothing charged | — |
| 6 | eval 130, offset 90 | policy.py (79c14b9aa2af, unchanged) | 4798 | 9966 | loss 0.0, hit 1.000 | Final confirmation on fresh dev rallies. |

**Total used: 9,966 of 10,000.**

## Outcome

- The final `policy.py` is the first derived construction. I changed no constant, because every sweep showed a flat optimum at loss 0, or showed that the chosen value was already on the safe side.
- On dev, `policy.py` was evaluated on 190 distinct rallies (0–59 and 90–219). Mean loss was 0.0 and the hit rate was 100%.
- The constants examined were the dead band (0, 0.015, 0.06) and the hinge counts / knot spacing (fine vs. coarse). Each has a stated meaning, and no fitting to simulated data was done.
- The held-out expectation (loss 0) rests on the analytic error bound (≤ 0.0074 versus a 0.08 margin), not only on these dev samples.
