# b250k: how the forward-pass budget was spent

Budget 250,000. Used **237,707** (35 harness calls, all logged in `log.jsonl`).
12,293 remain.

## Ground rules I followed

- I derived the weights on paper first (EXPLANATION.md). I did not train,
  compute gradients, or fit anything to simulator data.
- Every forward pass went through `uv run wsp-pong eval|probe`. My own scripts
  (`tools/probes.py`) only generate probe inputs and compare the returned
  logits with the closed-form reference crossing,
  fold(y + 1.5·vyn·(1 − x)/vxn). They never run the network.
- Candidates (`candidates/*.py`) are produced by `tools/make_candidate.py`.
  Each is policy.py with one or more named constants overridden, so every
  search step changes a parameter whose meaning is stated in policy.py.
- **Data splits.** The search split is dev offsets 0–4999; all design decisions
  and ablations used it. The held-out split is offsets 15000–17999, used once
  for the final policy and for nothing else. Probe sets: `rand1000_1` was the
  search set, `rand2000_99` held out the allocation decision.
- **Probe readout.** The output layer is [−e, 0.015, +e] with e = target − p,
  so the `up` logit + p is the network's predicted crossing height. One probe
  row therefore measures the approximation error at any chosen state.

## Calls, in order

| # | kind | policy | rows / rallies | cost | result | purpose → what I learned / changed |
|---|---|---|---|---|---|---|
| 1 | probe | policy.py (v1) | 81 corner states | 81 | max err 1.0e-3 | First check of the derived weights at input extremes. Correct. Noticed the stay logit read 0.014999999665: the output bias was built through a float32 tensor. **Fixed** (dtype=float64). The action logic was not affected. |
| 2 | probe | v1 | rand1000_1 | 1,000 | max 2.79e-3, rms 8.6e-4 | Random states over the full box. Matches the analytic bound (≤ 4.1e-3). |
| 3 | probe | v1 | slow300_2 (vxn≈0.5, \|vyn\|≈1, x small) | 300 | max 2.90e-3 | Worst-case region (largest \|u − y\|, up to 3 wall bounces). Still fine. |
| 4 | eval | v1 | 500 @ 0 | 17,686 | loss 0, hit 100% | First closed-loop test. The loss is already saturated at 0, so from here the loss cannot rank variants. I switched the search objective to probe-measured target error, which tracks the safety margin. |
| 5–10 | probe | alloc_r{6,8,10,12,16,20} | rand1000_1 each | 6,000 | max err 3.77, 3.33, 2.48, 2.53, 2.90, 3.65 e-3 (v1 r14: 2.79e-3) | Zero-order sweep of the layer-1 width split N_RECIP vs N_SQ1 = (62 − N_RECIP)/2. The optimum is broad, around 10–14, consistent with the Lagrange estimate of 13.5. |
| 11–15 | probe | r10 with C1∈{1,3}, C2∈{1,3}, RECIP_SHIFT=0 | rand1000_1 each | 5,000 | max 2.84, 2.95, 2.59, 2.52, 3.40 e-3 (r10 base 2.48e-3) | One-at-a-time check of the derived constants. C1 = C2 = 2 (the (c+2)²/c optimum) is at or near the best. Removing the minimax offset on 1/v is clearly worse (+37%), which confirms the offset. **No change.** |
| 16–18 | probe | v1, r10, r12 | rand2000_99 (held-out probes) | 6,000 | max 2.87, 3.63, 3.07 e-3; p99 2.32, 2.02, 2.04 e-3 | Held-out check of the allocation winner. r10's advantage did not hold up (its worst case is worse). Taking the worst case over both probe sets, v1 (14/24) is best and matches the analysis. **Kept N_RECIP = 14, N_SQ1 = 24.** |
| 19 | eval | abl_follow (SLOPE_GAIN = 0, target = y) | 200 @ 500 | 7,142 | loss 0.0488, hit 65% | Alternative design: no anticipation. This is the TASK.md failure mode. |
| 20 | eval | abl_clamp (WALLS = "clamp") | 200 @ 500 | 7,142 | loss 0.0348, hit 74% | Alternative design: straight-line prediction with no bounce model. The fold layer is necessary. |
| 21–23 | eval | stress_aim (AIM_OFFSET = 0.055 / 0.065 / 0.075) | 200 @ 500 each | 21,426 | hit 100% / 99.5% / 64% | Closed-loop slack test. I aimed deliberately wrong by Δ. The prediction was that misses begin once Δ + deadband (0.015) + error approaches 0.08, i.e. Δ ≈ 0.062. Observed: the first miss is at 0.065. So the controller tolerates about 0.06 of systematic error, versus the actual ≤ 0.004. |
| 24–27 | probe | coarse_{2,3,4,6} (all PWL segment counts = n) | rand1000_1 each | 4,000 | max err 0.444 / 0.210 / 0.123 / 0.058 | Failure-mode sweep: how much resolution is needed. |
| 28–30 | eval | coarse_{3,4,6} | 200 @ 700 each | 21,051 | hit 76.5% / 100% / 100% | Open-loop error overstates closed-loop risk. coarse_4 has 0.12 worst-case open-loop error yet hits everything, because estimates late in the rally (small 1 − x) are accurate and the paddle corrects. It breaks at 3 segments. The final design (14/24/31 segments) has a large margin. |
| 31 | probe | policy.py (final text, with the AIM_OFFSET/WALLS switches added) | 81 corners | 81 | logits bit-identical to call 1 (apart from the fixed stay bias) | Confirms that adding the neutral switches did not change any weight. vyn = 0 rows: error 8.9e-16 (exact). |
| 32 | eval | **policy.py** | **3000 @ 15000 (held-out split)** | 105,487 | **loss 0, hit 100%** | Final held-out check, its only use. |
| 33 | eval | policy.py | 1000 @ 1000 | 35,160 | loss 0, hit 100% | Extra coverage on unused search-split rallies. |
| 34 | probe | policy.py | 61 states: x = 0, vyn = 1, y = 0, vxn swept over [0.5, 1] | 61 | r̂ error ≤ 8.7e-4 (theory: 4.4e-4 from 1/v + ≤ 5.3e-4 from layer 2) | Unit-group test of the 1/vxn units. P is exact here because the square inputs sit on knots, so the error isolates r. Consistent with the derivation. |
| 35 | probe | policy.py | 90 states: vxn = 1 (r exact), x × vyn grid | 90 | max err 1.0e-3 | Unit-group test of the P = vyn(1 − x) squares plus the layer-2 product. Consistent with the bound 1.5·(8.7e-4 + 5.3e-4). |

## Summary of what the budget bought

- **Correctness of the derivation.** 3,613 probed states on the final weights
  gave a maximum target error of 2.9e-3, below the analytic worst case of
  4.1e-3. Unit-group probes matched the per-group error bounds.
- **Closed loop.** The final policy played 4,500 dev rallies (500 + 3,000
  held-out + 1,000) with **loss 0.000 and a 100% hit rate**.
- **Search.** The zero-order search over interpretable constants (width split,
  polarisation scales C1 and C2, the 1/v offset) found nothing better than the
  analytically derived values, so the final constants are the derived ones.
- **Failure modes.** The ablations (no anticipation 65%, no bounce model 74%,
  3-segment PWLs 76.5%) and the aim-offset stress test (misses start at
  Δ ≈ 0.065) show which mechanisms matter and how large the margin is: about
  0.06 of tolerable error versus 0.004 actual.

Final dev loss: **0.0** (hit rate 1.0) on both the search and held-out splits.
