# b50k: how the 50,000 forward passes were spent

Budget 50,000, used 49,950. Every call went through `uv run wsp-pong` and is in
`log.jsonl`.

## Before any pass: derivation

I read `game.py` and derived the crossing formula `fold(y + 1.5·vy_n·(1−x)/vx_n)`
and the reachability argument (≤ 14 steps needed, ≥ 25 available). Then I designed
the three-layer construction in EXPLANATION.md: 1/vx_n and a first product in
layer 1, a second product in layer 2, the fold in layer 3, and a dead-band argmax.
I also derived the error bounds analytically (about 4e-3 worst case on the target).

I checked that `build()` constructs the network and has the right shapes, without
calling it. All weights come from formulas. No simulated data was generated or
fitted.

`tools/probe_check.py` sends rows to `wsp-pong probe` and compares the implied
target (up-logit + paddle_y, since up = e = target − p) with the closed-form
*task reference*. It only computes the reference function. The network runs only
in the harness. From P3 on, it also saves results to `probe_results/`.

## Calls, in order

| # | kind | policy (sha) | rows / rallies (offset) | cost | cum. | result | what it tested, what I learned, what changed |
|---|---|---|---|---|---|---|---|
| P1 | probe | policy.py (5a7343) | 16 hand-picked states | 16 | 16 | worst target err 1.3e-3 | Extremes (x = 0, vx_n = 0.5, vy_n = ±1, which give u = ±3 with 3 bounces), late states (x = 0.97, 0.99) and mid states. The construction computes the right function. |
| E1 | eval | policy.py (5a7343), EPS = 0.02 | 100 (0) | 3504 | 3520 | loss 0.0, hit 100% | First closed-loop check. It works as derived. |
| P2 | probe | policy.py (5a7343) | 150-row grid: x × vx_n × vy_n | 150 | 3670 | visible lines showed errors up to 1.1e-3 (incomplete) | Grid over the domain. My sort/tail sorted by the signed error text and cut off the printed worst-case line, so the true grid maximum was lost. I made the probe script save results to disk, and P3 replaces this measurement. No change to the policy. |
| P3 | probe | policy.py (5a7343) | 200 uniform random rows in the input box | 200 | 3870 | worst 2.2e-3, mean |err| 6.4e-4, mean err −4e-5 | Measured error is under the analytic bound of 4.1e-3, with no bias. The worst cases are spread over all x, including x = 0.93, which confirms the error is absolute rather than relative. |
| E2 | eval | candidates/eps015 (38c9c9) | 100 (100) | 3674 | 7544 | 0.0, 100% | Dead band EPS = 0.015 (half a step) on fresh rallies. |
| E3 | eval | candidates/eps075 (c17041) | 100 (200) | 3502 | 11046 | 0.0, 100% | EPS = 0.075, near the predicted plateau edge (0.08 − error). Still a perfect score. |
| E4 | eval | candidates/eps090 (1a470e) | 100 (0) | 3504 | 14550 | 0.00089, 75% | EPS = 0.09 > 0.08. As predicted, the paddle can stop just outside the hit window: misses of ≤ 0.01, mean loss among misses ≈ 0.0036. This confirms the dead band works as intended and that EPS has a plateau of about [0, 0.078]. |
| E5 | eval | candidates/shift060 (db00c8) | 100 (300) | 3527 | 18077 | 0.0, 100% | Slack test. A diagnostic constant TARGET_SHIFT = 0.06 deliberately aims above the target (EPS = 0.015). Every hit means closed-loop error + resting offset < 0.02, so there is > 0.06 of margin. |
| E6 | eval | candidates/coarse4 (c04e51) | 100 (300) | 3527 | 21604 | 0.0, 100% | Robustness. Cutting every approximation to 4 knots (K_INV = M1 = M2 = 4) still gives a perfect score, so the design is not on a knife edge. |
| P4 | probe | candidates/coarse4 | same 200 rows as P3 | 200 | 21804 | worst 0.026, mean |err| 7.1e-3 | The coarse build's error is about 10× larger, matching the analytic h²/16 scaling (h ≈ 3× larger, so h² ≈ 9–14×). The bounds predict the error correctly. |
| E7 | eval | **policy.py (241f48)**, final, EPS = 0.015 | 700 (400) | 24696 | 46500 | 0.0, 100% | Confirmation of the final policy on 700 fresh rallies. |
| P5 | probe | policy.py (241f48) | 8 rows | 8 | 46508 | actions as predicted | Dead-band logic at p = target ± 0.014 (stay) and ± 0.016 (up/down), plus paddle at the top wall (down). |
| E8 | eval | policy.py (241f48) | 95 (1100) | 3442 | 49950 | 0.0, 100% | Used the remaining budget on more fresh rallies. |

## Zero-order search and what changed

- **EPS (dead band).** Values tried: 0.02 (E1), 0.015 (E2), 0.075 (E3), 0.09 (E4).
  The loss is flat at 0 throughout [0.015, 0.075] and breaks above 0.08, as
  derived. I picked 0.015 for the final policy, on an analytic argument rather
  than a measured difference. It is the smallest band without oscillation, so the
  worst-case resting offset is smallest (≤ 0.015). That keeps the most margin
  against estimation error.
- **TARGET_SHIFT.** Diagnostic only. It is 0.0 in the final policy, where it has no
  effect.
- **Resolution (K_INV, M1, M2).** 4/4/4 versus 14/12/15. Both give a perfect score.
  I kept the full-width 14/12/15 because it gives about 10× more accuracy at no cost
  (the width limit is 64 regardless).

The final policy differs from the probed and evaluated sha 5a7343 only in EPS
(0.02 → 0.015) and the added TARGET_SHIFT = 0 constant. The target computation is
identical, so P1–P3 apply to it.

## Summary

The final policy (sha 241f48) has a dev loss of 0.0 and a hit rate of 100% on 795
fresh rallies (offsets 400–1194). Earlier versions differing only in EPS scored 0
on another 300 rallies. Analytic and probed target error is ≤ 4e-3, against a
0.08 tolerance.

Note: after the last eval I ran `ruff format` on `policy.py`. It only rewrapped
long lines, so the constants and the weights are unchanged. The file's sha
changed from 241f48 to 8873d4. No passes were
left to re-run it.
