# b50k Pong policy: how every weight is derived

Network: `5 -> 64 -> 62 -> 10 -> 3` (Linear/ReLU, three hidden layers). No weight
was trained or fitted. Every weight comes from a named helper in `policy.py`.
Each unit also carries a provenance name (`model.unit_names`), e.g.
`inv:knot3@0.5692`, `s2:sq-7`, `fold:hinge@-2`.

## The task in closed form

From `game.py`, the ball's height at the moment it crosses x = 1 is
`fold(y0 + vy/vx)`. Starting from the current observation, which gives the folded
height `y` and the *current* (post-bounce) vertical velocity, the remaining
horizontal distance is `a = 1 - x`, which takes `a / vx` steps. So

```
D      = vy * a / vx = (0.06 vy_n) * a / (0.04 vx_n) = 1.5 * vy_n * a / vx_n
u      = y + D                 (unfolded crossing height, u in [-3, 4])
target = fold(u)               (triangle wave of period 2)
```

The controller moves toward `target`. Starting from 0.5, it takes at most
(0.5 - 0.08)/0.03 = 14 steps to reach any crossing, and every rally lasts at least
25 steps. If the target estimate is accurate, every ball is reachable. The whole
design therefore comes down to computing `fold(1.5 vy_n (1-x)/vx_n + y)` accurately
with ReLUs. That needs two multiplications, one reciprocal and one fold.

Building block: **difference of squares.** `A*B = ((A+B)^2 - (A-B)^2)/4`. Each
square is a one-variable function, so one ReLU layer can approximate it with a
chord (piecewise-linear) interpolant. With knots `0, ±h, ±2h, …, ±(m-1)h`:

```
z^2 ≈ h*(relu(z) + relu(-z)) + Σ_{i=1}^{m-1} 2h*(relu(z - ih) + relu(-z - ih))
```

The chord slope on `[ih, (i+1)h]` is `(2i+1)h`, so the first hinge has slope `h`
and each later hinge adds `2h`. The interpolant is exact at the knots and
overestimates by at most `h²/4`. Both squares overestimate, so the product error
is at most `(h²/4)/4 = h²/16`. The t = 0 pair also gives the *exact* linear readout
`z = relu(z) - relu(-z)`. That is how the network carries `vy_n` and `c` forward
without spending extra units.

The scale factor λ in `A = λ·P`, `B = Q/λ` is chosen to balance the ranges of the
two factors (λ² = range(Q)/range(P)). This minimises the square's domain radius R,
and so the error for a given number of units.

## Hidden layer 1 (64 units): reciprocal and first product

| units | name | mechanism |
|---|---|---|
| 0 | `pass:y` | `relu(y) = y` (y ≥ 0): exact pass-through |
| 1 | `pass:p` | `relu(paddle_y)`: exact pass-through |
| 2–15 | `inv:knot j` | `relu(vx_n - v_j)`, j = 0..13: chord interpolant of `1/vx_n` on [0.5, 1] |
| 16–39 | `s1:sq±i` | square of `s1 = vy_n/√2 + √2(0.5 - x)`, m = 12, R = √2 |
| 40–63 | `d1:sq±i` | square of `d1 = vy_n/√2 − √2(0.5 - x)`, m = 12, R = √2 |

**Reciprocal (units 2–15).** On an interval [a, b], the chord of 1/v has maximum
error `(√b-√a)²/(ab) = (1/√a - 1/√b)²` (attained at v = √(ab)). This is the same
on every piece when the knots are uniform in `1/√v`. With K_INV = 14 pieces:
`v_j = 1/(√2 - j(√2-1)/14)²` = 0.5000, 0.5216, 0.5446, 0.5692, 0.5955, 0.6236,
0.6538, 0.6863, 0.7212, 0.7589, 0.7996, 0.8436, 0.8914, 0.9434, 1.0000. The
hinge at `v_j` has coefficient `slope_j − slope_{j−1}`, with
`slope_j = −1/(v_j v_{j+1})`. Per-piece error is `E = ((√2−1)/14)² = 8.75e-4`.
The readout bias `1/0.5 − E/2 − 1.5` centres the error to ±E/2 = ±4.4e-4 and
subtracts the midpoint 1.5. The layer-2 multiplier then sees `w' = 1/vx_n − 1.5 ∈
[−0.5, 0.5]`. The interpolant is exact at the knots. It is valid because vx_n ∈
[0.5, 1] always (vx ∈ [0.02, 0.04]). For vx_n < 0.5 it would extrapolate linearly,
but that never happens.

**First product (units 16–63).** We want `c = 1.5·vy_n·a`. Write `a = 0.5 + (0.5−x)`,
so `vy_n·a = vy_n·(0.5−x) + 0.5·vy_n`. The ranges are |vy_n| ≤ 1 and
|0.5−x| ≤ 0.5, so λ1 = 1/√2 and |s1|, |d1| ≤ 1/√2 + √2·0.5 = √2 = R1. With
m = 12, h = 0.1179, the product error is at most h²/16 = 8.7e-4 (×1.5 inside c).
The readouts are `vy_n(0.5−x) = (sq(s1) − sq(d1))/4` and
`vy_n = (lin(s1)+lin(d1))/(2λ1)`, which is exact. Then
`c = 1.5·(vy_n(0.5−x) + 0.5·vy_n)`.

## Hidden layer 2 (62 units): second product, D = c / vx_n

| units | name | mechanism |
|---|---|---|
| 0, 1 | `pass:y`, `pass:p` | exact pass-through of layer-1 units 0, 1 |
| 2–31 | `s2:sq±i` | square of `s2 = c/√3 + √3·w'`, m = 15, R = 1.75 |
| 32–61 | `d2:sq±i` | square of `d2 = c/√3 − √3·w'`, m = 15, R = 1.75 |

`c ∈ [−1.5, 1.5]` and `w' ∈ [−0.5, 0.5]` give λ2 = 1/√3 and |s2|, |d2| ≤ √3 = 1.732.
R2 = 1.75 leaves room for the layer-1 error. The hinge spacing is
h = 0.1167, so the product error is at most 8.5e-4. The readouts are
`c·w' = (sq(s2) − sq(d2))/4` and the exact `c = (lin(s2)+lin(d2))/(2λ2)`, so
`D = c·w' + 1.5·c = c / vx_n` and `u = y + D`. The input weights of these units are
layer-1 readout vectors (the linear forms for `c` and `w'`) combined with λ2. The
matrix is the composition of two named linear maps, not an opaque fit.

## Hidden layer 3 (10 units): the fold (wall bounces)

`fold` is piecewise linear with slope +1 on [2k, 2k+1] and −1 on [2k+1, 2k+2], and
fold(−4) = 0. Units `fold:hinge@t = relu(u − t)` for t = −4..4 have readout
coefficients +1 (t = −4, the initial slope) and then +2 at even t and −2 at odd t
(the slope flips by 2 at each integer). The readout bias is 0. On u ∈ [−4, 5] this
is **exact**. Physically u ∈ [−3, 4], and below −4 it returns 0 instead of folding,
which never happens. fold is 1-Lipschitz, so it never amplifies the upstream error,
even when u lands right on a bounce. Unit 9 is `pass:p`.

## Output layer (3 logits)

`e = fold(u) − p` is a linear readout of layer 3. The logits are
`[down, stay, up] = [−e, EPS, +e]`. The argmax picks up if e > EPS, down if e < −EPS,
and stay otherwise. Ties go to the lower index, so at exactly e = EPS the paddle
stays. EPS = 0.015 is half a paddle step (0.03/2). It is the smallest dead band that
cannot oscillate, so it minimises the worst-case resting offset between paddle and
target (≤ 0.015).

## Error budget: when it is exact and when it approximates

Worst-case error on D, derived analytically:

- layer-1 product: 1.5·8.7e-4 = 1.3e-3, multiplied by `1/vx_n ≤ 2` → 2.6e-3
- reciprocal: |c| ≤ 1.5 times 4.4e-4 → 6.6e-4
- layer-2 product: 8.5e-4
- total ≤ about 4.1e-3. Measured worst on 350 probed states: 2.2e-3 (mean |err| 6.4e-4).

The pass-throughs and the fold are exact. The products are exact at the square
knots, and `1/vx_n` is exact at the reciprocal knots. The error is an *absolute*
bump pattern and does not shrink as x → 1, but it stays at ≤ 0.004. The final
paddle offset is at most EPS + error ≤ 0.019, against a hit tolerance of 0.08,
which leaves a margin of more than 0.06. Measured in closed loop (METHOD.md, E5):
aiming deliberately 0.06 off-target still hits 100%.

**Failure modes.**

1. If EPS > 0.08 − error, the paddle can stop outside the hit window. This was
   confirmed at EPS = 0.09: 75% hits.
2. If the inputs left the ranges assumed above (vx_n < 0.5, |vy_n| > 1, or |u| > 4),
   the squares and the reciprocal would extrapolate linearly and the fold would stop
   folding. That cannot happen in this game.
3. The controller is greedy. It does not plan around reachability, because the
   physics guarantees reachability (14 ≤ 25 steps). With a faster ball or a slower
   paddle it would have to start earlier or prioritise.
4. The design degrades gracefully. With 4 knots everywhere (E6, P4), the worst error
   is 0.026 and it still hits 100%.
