# b250k: how the constructed Pong network works

Architecture: `5 → 64 → 64 → 10 → 3`, ReLU after each hidden layer, float64.
No weight was trained or fitted. Every weight comes out of `build()` in
`policy.py`, which derives it from named constants. `model.provenance` labels
every hidden unit, e.g. `"(2(1-x)+vyn)^2: ReLU(z - +0.33333), slope change +0.33333"`.

## 0. The function being computed

The ball moves in a straight line with reflections at y = 0 and y = 1. Reflection
is the same as moving freely in an "unfolded" coordinate u and then mapping back
with the triangle wave `fold(u)` (period 2, fold(u) = u on [0, 1], 2 − u on [1, 2]).
At the current observation the ball is at folded height y and moving with signed
vertical speed vy (game.py flips the sign at each bounce). It still needs
(1 − x)/vx steps to reach x = 1, so

    crossing = fold( y + vy (1 − x) / vx ) = fold(u),
    u = y + 1.5 · vyn · (1 − x) / vxn          (vy/vx = (0.06 vyn)/(0.04 vxn)).

This is **exact**. Restarting the unfolded line from the current folded state
(y, signed vy) gives the same crossing as `game.Rallies.crossing_y`, because
fold is even and 2-periodic. If u = y0 + k·vy and y = fold(u), then either
y = u − 2m with vy unchanged, or y = 2 + 2m − u with vy negated. In both cases
fold(y ± d) = fold(u + d). So the target is **the same at every step of a
rally**. The controller only has to move the paddle toward one fixed point,
and the estimate never has to "catch up".

The ranges that matter: x ∈ [0, 1), y ∈ [0, 1], vxn ∈ [0.5, 1], vyn ∈ [−1, 1].
So |u − y| ≤ 1.5 · 1 · 1/0.5 = 3 and u ∈ [−3, 4].

The network splits u − y = 1.5 · P · r into
- P = vyn · (1 − x), bilinear in the raw inputs, and
- r = 1/vxn, a 1-D function of one input.

Layer 1 computes both. Layer 2 multiplies them. Layer 3 applies fold. The output
layer compares the result with the paddle.

The building block used everywhere: a piecewise-linear (PWL) interpolant of a
1-D function f on knots k0 < … < kn is

    f̂(z) = f(k0) + Σ_{i<n} c_i · ReLU(z − k_i),   c_0 = first slope, c_i = slope change at k_i.

This holds exactly for z ≥ k0. Each term is one hidden unit with weight row =
the coefficients of z and bias = (constant of z) − k_i. The c_i and f(k0) go
into the next layer's weights (`pwl_terms`, `Layer.pwl`, `readout` in policy.py).

## Layer 1 (64 units): pass-through, 1/vxn, and the bilinear product P

| units | mechanism | weights |
|---|---|---|
| 0 | `ReLU(y)` = y, pass-through (y ≥ 0) | row e_y, bias 0 |
| 1 | `ReLU(p)` = p, pass-through (p ≥ 0) | row e_p, bias 0 |
| 2–15 | PWL of r(v) = 1/v on v = vxn ∈ [0.5, 1], 14 segments | row e_vxn, bias −k_i |
| 16–39 | PWL of z₊² with z₊ = 2(1 − x) + vyn ∈ [−1, 3], 24 uniform segments | row (−2, 0, 0, +1, 0), bias 2 − k_i |
| 40–63 | PWL of z₋² with z₋ = 2(1 − x) − vyn ∈ [−1, 3], 24 uniform segments | row (−2, 0, 0, −1, 0), bias 2 − k_i |

**P by polarisation.** With a = 1 − x ∈ [0, 1] and b = vyn ∈ [−1, 1],
a·b = ((c a + b)² − (c a − b)²)/(4c). Each square is a 1-D function of an
affine combination of the inputs, so one ReLU layer can represent it. The
layer-2 weights read P̂ = (ẑ₊² − ẑ₋²)/8 off these units.

*Why c = C1 = 2.* On a segment of width h, the PWL interpolant of z² overshoots
by t(1 − t)h², which is at most h²/4, the same everywhere. Both squares span
c + 2 (from −1 to c + 1), so with N segments the error in a·b is at most
((c+2)/N)²/4 / (4c) ∝ (c + 2)²/c. That is minimised at c = 2. The z² error
does not depend on position, so uniform knots are optimal.

*Error.* h = 4/24 = 1/6, so the overshoot of each square is in [0, 1/144] and
|P̂ − P| ≤ (1/144)/8 = 8.7·10⁻⁴. When vyn = 0, z₊ = z₋ and P̂ = 0 exactly.

**r = 1/vxn.** For a convex f, the interpolation error on a segment is about
h² f''/8. For f = 1/v, f'' = 2/v³, so equal error on every segment needs
h ∝ v^{3/2}. That means knots uniform in s = v^{−1/2}, from √2 down to 1. The
knots are 0.5, 0.5216, 0.5446, 0.5692, 0.5955, 0.6236, 0.6538, 0.6863, 0.7212,
0.7589, 0.7996, 0.8436, 0.8914, 0.9434, 1.0. With Δs = (√2 − 1)/14, the
per-segment overshoot is h² f''/8 = (2 v^{3/2} Δs)² · (2/v³)/8 = Δs² = 8.8·10⁻⁴,
the same on every segment. `RECIP_SHIFT = 1` subtracts Δs²/2 in the readout,
which centres the error at ±4.4·10⁻⁴ (minimax constant offset).

*Width split.* The errors in u are about 1.5·|r|·|δP| ≤ 1.5/N_SQ1² (from P) and
1.5·|P|·|δr| ≈ 0.13/N_RECIP² (from r), subject to N_RECIP + 2·N_SQ1 = 62.
Lagrange gives N_SQ1/N_RECIP ≈ 1.8, so N_RECIP ≈ 13.5. I chose 14/24. A
zero-order probe sweep over the allocation confirmed it (METHOD.md).

## Layer 2 (64 units): the product M = P · r

| units | mechanism |
|---|---|
| 0, 1 | y, p pass-through (`ReLU` of a non-negative unit) |
| 2–32 | PWL of w₊² with w₊ = 2r̂ + P̂, 31 uniform segments on [0.98, 5.02] |
| 33–63 | PWL of w₋² with w₋ = 2r̂ − P̂, same knots |

Each unit's weight row is the affine readout of w₊ or w₋ over layer-1 units,
i.e. 2·(r-PWL coefficients) ± (1/8)·(square coefficients), with bias = the
readout constants minus the knot. Same polarisation identity:
M̂ = (ŵ₊² − ŵ₋²)/(4·C2) with C2 = 2. Both terms range over [C2 − 1, 2C2 + 1] = [1, 5]
(same (c+2)²/c argument). `SQ2_MARGIN = 0.02` widens the domain by roughly ten
times the worst layer-1 error, so the inputs never fall off the first knot.
h = 4.04/31, so |δM| ≤ h²/4/8 = 5.3·10⁻⁴.

The readout from layer 2 is u = y + 1.5 M̂, which the layer-3 rows apply.

## Layer 3 (10 units): reflection off the walls

| unit | mechanism |
|---|---|
| 0 | p pass-through |
| 1–9 | `ReLU(u − k)` for k = −3.25, −3, −2, −1, 0, 1, 2, 3, 4 |

fold is linear between consecutive integers, so a PWL with a knot at every
integer in [−3.25, 4.25] reproduces it **exactly**, not approximately. The slopes
are +1 on [−3.25, −3] (fold(u) = fold(u + 4) and u + 4 ∈ [0.75, 1]), then they
alternate −1, +1, … In the output layer the slope changes are +1 at −3.25 and
∓2 at each integer, and the constant is fold(−3.25) = 0.75. The domain covers
u ∈ [−3, 4] with 0.25 to spare. The `WALLS="clamp"` ablation uses the same knots
with clip(u, 0, 1), which is also linear between integers.

## Output layer (3 logits)

e = target − p = 0.75 + Σ c_j·h3_j − h3_0, and logits = [−e, DEADBAND, +e].
- `up` wins iff e > 0.015, `down` wins iff e < −0.015, otherwise `stay`. On a
  tie stay beats up because it has the lower index.
- The weight rows are ±(1, −1, 2, −2, 2, −2, 2, −2, 2, −2) on (p, fold units)
  with the sign flipped for p, and the biases are ∓0.75 and 0.015.
- `DEADBAND = PADDLE_SPEED/2` is the smallest band that cannot oscillate. A
  move from just outside the band (distance d ∈ (0.015, 0.03]) lands at
  |d − 0.03| < 0.015, inside it.

## When it is exact, when it approximates, and when it fails

- **Exact:** the reduction to fold(u), the fold layer, the pass-throughs, the
  controller logic, and every state with vyn = 0. Probe 31 measured an error
  of 8.9·10⁻¹⁶ there.
- **Approximate:** u carries |δu| ≤ 1.5·(2·8.7e-4 + 1·4.4e-4 + 5.3e-4) ≈ 4.1·10⁻³
  (worst case). fold is 1-Lipschitz, so the target error is bounded the same
  way. Measured over 3,613 probed states (random, slow/steep, corners, unit
  tests), the maximum was **2.9·10⁻³** and the RMS 0.9·10⁻³. The error shrinks
  as x → 1 because P ∝ (1 − x).
- **Closed-loop guarantee:** the target estimate is within 0.0041 of the same
  fixed crossing at every step, so the paddle moves monotonically toward it.
  It needs at most 0.5/0.03 ≈ 17 decisions, and every rally has at least 25.
  Final |paddle − crossing| ≤ 0.015 + 0.0041 < 0.08, so **every rally from the
  physical distribution is a hit** (loss 0). The stress test agrees: a
  deliberate aiming error of 0.055 still hits 100%, misses start near
  0.065 ≈ 0.08 − 0.015, and 0.075 hits 64%. The slack is about 15 times the
  approximation error.
- **Fails:** only outside the physical input box, which the game never
  produces. Examples are vxn < 0.5 (the 1/v PWL extrapolates linearly from the
  first knot, so r is underestimated) and |u| beyond the fold domain (linear
  extrapolation). The design is also much more robust than its knot count
  suggests: with only 4 segments per PWL (14 + 10 + 10 units), probes showed
  open-loop errors up to 0.12, yet the closed loop still hit 200/200, because
  the late-rally estimates (small 1 − x) are accurate. At 3 segments the hit
  rate falls to 76%. Without bounce modelling (clamp) it is 74%, and following
  the ball is 65%.
