# b2k policy: how every weight group works

## Task, in closed form

In observation units (`a = 1 - ball_x`, `c = ball_vy_n`, `v = ball_vx_n`, `y = ball_y`, `p = paddle_y`) the ball
needs `a / vx` more steps to reach x = 1. Over that time it moves vertically by

    d = vy * a / vx = (VY_MAX / VX_MAX) * c * a / v = K * c * a / v,     K = 1.5.

Because `c` is the *current* (already sign-flipped) vertical velocity and `y` the current folded height, the
crossing height is `t* = fold(y + d)`. `fold` is the period-2 triangle wave used in `game.py`. This is exact and
matches `game.Rallies.crossing_y`. Over the input domain `a ∈ [0,1]`, `v ∈ [0.5,1]`, `c ∈ [-1,1]`, we have
`|d| ≤ 3`, so `u = y + d ∈ [-3, 4]`.

The network computes `t ≈ t*` and moves the paddle toward it. It needs two multiplications (`c·a`, then
`(c·a)·(1/v)`), one reciprocal and one fold. Each multiplication uses the polarization identity
`xy = ((x+y)² − (x−y)²)/4`, with each square written as a piecewise-linear (PWL) sum of hinges. The two squares are
built with the same knots, so their errors largely cancel. A hinge layer is needed before each nonlinearity can be read out
linearly, which fixes the depth at 3 hidden layers:

| layer | width | what its linear read-out provides |
|---|---|---|
| hidden 1 | 62 | `y`, `p`, `b ≈ 1/v`, `P1 ≈ c·a` |
| hidden 2 | 62 | `y`, `p`, `u ≈ y + K·P1·b` |
| hidden 3 | 10 | `p`, `t = fold(u)` (exact given u) |
| output | 3 | `[p − t − θ, 0, t − p − θ]` |

All constants are named in `policy.py`. `pwl_hinge_coeffs` turns "interpolate f at these knots" into hinge
coefficients: the first coefficient is the first segment's slope, and each later coefficient is the change of
slope at that knot. These are evaluations of known functions (1/v and z²) at hand-chosen knots. Nothing is fitted
to data.

## Hidden layer 1 (5 → 62)

**Unit 0: `ReLU(ball_y)`**, weight 1 on `ball_y`, bias 0. This is exact because `y ≥ 0`.

**Unit 1: `ReLU(paddle_y)`**, weight 1 on `paddle_y`, bias 0. This is exact because `p ∈ [0,1]`.

**Units 2–13: reciprocal hinges `B_j = ReLU(v − v_j)`**, weight 1 on `vx_n`, bias `−v_j`.
- Knots `v_j = 0.5 · 2^(j/12)`, j = 0..12. Only j = 0..11 need units. `B_0 = v − 0.5` is linear on the domain.
- Read-out: `b = 2 + Σ_j β_j B_j`, with `β_0 = −1/(v_0 v_1)` (first chord slope) and
  `β_j = −1/(v_j v_{j+1}) + 1/(v_{j−1} v_j)` (slope changes, all positive because 1/v is convex). The constant 2 = 1/v_0
  goes into the layer-2 biases.
- Why geometric knots: the chord error of 1/v on `[v_j, v_j r]` is about `h²/(4v³)` with `h = v(r−1)`. That gives
  relative error `(r−1)²/4`, which is the same in every segment. With `r = 2^(1/12)` the relative error is ≤ 8.8·10⁻⁴.
- Exact at the 13 knots. Elsewhere it overestimates 1/v by at most 0.09 %. It extrapolates linearly outside
  [0.5, 1], which never happens in this game.

**Units 14–37: square hinges for `r1 = a + c = 1 − x + c`**, `R1_j = ReLU(r1 − z_j)`, with weights `(−1 on ball_x, +1 on vy_n)`
and bias `1 − z_j`.
**Units 38–61: square hinges for `r2 = a − c = 1 − x − c`**, `R2_j = ReLU(r2 − z_j)`, with weights `(−1 on ball_x, −1 on vy_n)`
and bias `1 − z_j`.
- Both `r1, r2 ∈ [−1, 2]`. The knots are `z_j = −1 + j/8`, j = 0..24 (h = 1/8), so each square has 24 units.
- PWL `z²`: `sq(z) = 1 + (2z_0 + h)·R_0 + 2h Σ_{j≥1} R_j`. The first chord slope is `z_0 + z_1 = −1.875`, and every later
  slope change is `2h = 0.25`.
- Read-out: `P1 = (sq(r1) − sq(r2))/4 = Σ_j coef_j (R1_j − R2_j)/4`. The constants cancel.
- I used `a ± c` rather than `c ± a` so that both squares live on the same interval and share one knot grid.
  That grid is symmetric about 0 on [−1, 1].
- Error: the chord error `e(z) ∈ [0, h²/4]`, so `|P1 − c·a| ≤ h²/16 ≈ 9.8·10⁻⁴`. Near impact (`a → 0`) we get
  `r1 → c` and `r2 → −c`. The grid is symmetric, so `e(c) = e(−c)` and the error vanishes. Because e is h-Lipschitz,
  `|P1 − c·a| ≤ min(h²/16, h·a/2)`. The prediction gets *more* exact as the ball approaches, which is exactly when it
  matters most.
- It is exact when `r1` and `r2` are both on knots. In particular `c = 0` gives `P1 = 0` exactly, and so does `a = 0`.

## Hidden layer 2 (62 → 62)

**Units 0, 1:** pass-throughs of `y` and `p`, each with weight 1 from layer-1 units 0 and 1. They are exact because the values are ≥ 0.

**Units 2–31: `S3_k = ReLU(z3 − w_k)` with `z3 = b + P1`. Units 32–61: `S4_k = ReLU(z4 − w_k)` with `z4 = b − P1`.**
- The weight rows are the layer-1 read-outs `read_b ± read_P1`, and the bias is `2 − w_k`. These are linear functions of the
  layer-1 units, so they compose exactly.
- `b ∈ [1,2]` and `|P1| ≤ 1`, so `z3, z4 ∈ [0, 3]`. The knots are `w_k = k/10`, k = 0..30 (h₂ = 0.1). The PWL z² has first slope
  `w_0 + w_1 = 0.1`, every later slope change is `0.2`, and its value at `w_0` is 0.
- Read-out: `u = y + K·(sq(z3) − sq(z4))/4`, so the weights are `±K·coef_k/4 = ±0.375·coef_k`.
- Error: each square is off by at most `h₂²/4 = 0.0025`, so the product `P1·b` is off by ≤ 6.3·10⁻⁴ and `u` by ≤ 9.4·10⁻⁴. It is
  exact when `P1 = 0` (`z3 = z4`, the errors cancel identically). So this error also vanishes near impact, and when `c = 0`.
  If a slightly negative `z` appears through upstream error, the ReLU clamps its square to 0. The true square is ≈ 0 there.

## Hidden layer 3 (62 → 10)

**Unit 0:** paddle pass-through.

**Units 1–9: `F_k = ReLU(u − k)` for k = −4..4.** The weight row is `read_u` from layer 2, and the bias is `−k`.
- Output read-out: `t = F_{−4} + Σ_{k=−3..4} 2(−1)^k F_k`. The triangle wave satisfies `fold(−4) = 0` with slope +1 on [−4, −3],
  and its slope flips by −2 at odd integers and by +2 at even integers. This is exact for all `u ∈ [−4, 5]`, which contains the
  reachable range [−3, 4] with a margin of 1 for upstream error. fold is 1-Lipschitz, so it never amplifies the error in u.

## Output layer (10 → 3)

- `down = p − t − θ`, `stay = 0`, `up = t − p − θ`, with `θ = PADDLE_SPEED/2 = 0.015`.
- Argmax: go up iff `t − p > θ`, go down iff `p − t > θ`, otherwise stay. A tie at exactly ±θ goes to `stay`, the lower
  index. Moving 0.03 reduces `|t − p|` exactly when `|t − p| > 0.015`, so θ is the dead band that never makes the
  paddle worse with respect to its current estimate and stops it from jittering.

## Error budget and why every rally in the distribution is a hit

The worst-case total target error is `|t − t*| ≤ 0.0026 (1/v) + 0.0029 (c·a) + 0.0009 (product) ≈ 0.0065`. The real
error is smaller because the bounds are not all reached together. In 50 probe rows the largest observed error was 0.0028. Both multiplication
errors vanish as `a → 0`.

Control argument: every estimate `t̂_k` is within ε = 0.0065 of the true crossing T. While `|p − T| > θ + ε = 0.0215`, the paddle
steps 0.03 toward T. Once inside that band it stays inside: it only moves when `|t̂ − p| > θ`, and the move lands within θ of t̂.
From the start `p = 0.5`, so `|p − T| ≤ 0.5` and at most 16 steps are needed, while every rally has at least
`ceil(1/0.04) = 25` decisions. The final `|p − T| ≤ 0.0215 < 0.08 = PADDLE_HALF`, so every rally drawn from the game's
distribution should be a hit, with loss exactly 0. The clamp to [0, 1] does not interfere because `t ∈ [0, 1]`.

When it would fail: if `vx_n` fell outside [0.5, 1] the 1/v extrapolation would degrade. If `|u|` exceeded the [−4, 5] fold range,
which needs |d| > 4, the fold would be wrong. If the paddle were slower or rallies shorter than ~16 steps, a correct
prediction could still be out of reach. None of these can happen under `game.py`'s constants.
