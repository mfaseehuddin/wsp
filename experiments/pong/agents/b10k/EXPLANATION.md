# b10k policy: how every weight group works

Network: `Linear(5,64) → ReLU → Linear(64,64) → ReLU → Linear(64,3)`. It has two hidden layers. The third allowed hidden layer is not needed. No weight was trained or fitted. Every value comes from the formulas below, which `policy.py` implements through the named constants and the `LayerBuilder.square` / perspective-fold helpers. `policy.UNIT_LABELS` (filled by `build()`) gives a label for every hidden unit.

Notation for the inputs: `x = ball_x`, `y = ball_y`, `a = vx_n ∈ [0.5, 1]`, `b = vy_n ∈ [−1, 1]`, `p = paddle_y`, `R(·) = ReLU`.

## 0. What has to be computed (derivation from `game.py`)

The ball's unfolded height is `u_k = y0 + k·vy`, and it crosses `x = 1` at time `1/vx`, so `crossing = fold(y0 + vy/vx)`. From any intermediate state, the remaining unfolded travel is `vy_cur·(1−x)/vx`. `fold` is a period-2 triangle wave that is symmetric under `u → −u` and `u → u + 2`, and the observed `vy_cur` carries the sign of the current fold branch. So, exactly:

```
crossing = fold(U),   U = y + (vy/vx)(1 − x) = y + SLOPE · b · (1 − x) / a,   SLOPE = VY_MAX/VX_MAX = 1.5
```

`U ∈ [−3, 4]`, because `|1.5·b·(1−x)/a| ≤ 1.5·1·1/0.5 = 3`.

The path never depends on the paddle, so a perfect predictor has a target that stays fixed for the whole rally. The paddle starts at 0.5. It needs at most `(0.5 − 0.08)/0.03 = 14` steps to reach a hit, and each rally gives it 25 to 50 steps. Steering toward an accurate prediction therefore hits every rally.

**Key trick (no division needed).** The action depends only on the sign of `fold(U) − p` compared with a dead band. Since `a > 0`, multiplying by `a` keeps the sign:

```
a·(fold(U) − p) = a·fold(V/a) − a·p,     V = a·U = a·y + 1.5·b − 1.5·b·x
```

`a·fold(V/a)` is the perspective of a piecewise-linear function, so it is an exact sum of hinges `R(V − k·a)`. What remains is three products of inputs: `a·y`, `b·x` and `a·p`. The polarisation identity `u·v = ((u+v)² − (u−v)²)/4` turns each product into squares of single linear forms, and ReLU hinges approximate each square.

### Square block (used for all six squares)

For a linear form `s ∈ [lo, hi]`, use `m` uniform segments with `h = (hi−lo)/m` and knots `t_j = lo + j·h`:

```
S(s) = lo² + (2·lo + h)·R(s − lo) + Σ_{j=1}^{m−1} 2h·R(s − t_j)
```

- The first hinge's output weight `2lo + h` is the slope of the first chord, `(t_1² − t_0²)/h`. Each later hinge adds the slope jump between consecutive chords, `2h`. The bias `lo²` is the value at `lo`.
- `S` equals `s²` at every knot (exact there). In between it overestimates by `(s − t_j)(t_{j+1} − s) ∈ [0, h²/4]`.
- Both squares of one product use the same `h`, so their errors have the same range. The polarised product's error is therefore `(e₁ − e₂)/4 ∈ [−h²/16, +h²/16]`. No minimax offset is needed, because a constant offset cancels in the difference.
- The block only works while `s ∈ [lo, hi]`. Below `lo` it is flat and above `hi` it extrapolates the last chord. The ranges below are the exact ranges of the inputs, so this never happens in the game.

## 1. Hidden layer 1 (64 units): input weights `W1`, biases `b1`

| units | mechanism | input weights / bias |
|---|---|---|
| 0 | pass `a` (always > 0) | `w[a]=1` |
| 1, 2 | `b⁺ = R(b)`, `b⁻ = R(−b)`, so `b = b⁺ − b⁻` exactly | `w[b]=±1` |
| 3 | pass `p` (always ≥ 0) | `w[p]=1` |
| 4–13 | hinges of `(a+y)²` on `[0.5, 2]`, `SEG_AY = 10`, `h = 0.15` | `w[a]=w[y]=1`, bias `−t_j` |
| 14–23 | hinges of `(a−y)²` on `[−0.5, 1]`, 10 segments, `h = 0.15` | `w[a]=1, w[y]=−1`, bias `−t_j` |
| 24–43 | hinges of `(b+x)²` on `[−1, 2]`, `SEG_BX = 20`, `h = 0.15` | `w[b]=w[x]=1`, bias `−t_j` |
| 44–63 | hinges of `(b−x)²` on `[−2, 1]`, 20 segments, `h = 0.15` | `w[b]=1, w[x]=−1`, bias `−t_j` |

How the 60 hinges are split: the four ranges have total length 9 (1.5 + 1.5 + 3 + 3), so 60 hinges give one common spacing of `h₁ = 0.15`. With equal spacing, the worst-case errors of the two products match.

Read out linearly from layer 1 (these are the layer-2 input weights):

- `a·y ≈ (S(a+y) − S(a−y))/4`, error ≤ `h₁²/16 = 0.0014`
- `b·x ≈ (S(b+x) − S(b−x))/4`, error ≤ `0.0014`
- `V = a·y + 1.5·(b⁺ − b⁻) − 1.5·b·x`, error ≤ `0.0014·(1 + 1.5) = 0.0035`

## 2. Hidden layer 2 (64 units): input weights `W2`, biases `b2`

Each layer-2 unit is a ReLU of one of the linear read-outs above.

| units | mechanism | definition |
|---|---|---|
| 0–8 | perspective-fold hinges `F_k = R(V − k·a)` for `k = −4 … 4`. Since `a > 0`, `F_k = a·R(U − k)` | weights = coefficients of `V` minus `k` times coefficients of `a` |
| 9 | pass `a` | `R(a)` |
| 10–36 | hinges of `(a+p)²` on `[0.5, 2]`, `SEG_AP = 27`, `h₂ = 1/18 ≈ 0.0556` | `R(a + p − t_j)` |
| 37–63 | hinges of `(a−p)²` on `[−0.5, 1]`, 27 segments | `R(a − p − t_j)` |

The split is 64 − 9 fold hinges − 1 pass = 54 hinges, 27 per square. The error of `a·p` is ≤ `h₂²/16 = 0.00019`, which is ≤ 0.0004 after dividing by `a`.

**Why the fold has 9 kinks.** `fold` has a kink at every integer. On `u ≥ −4`, `fold(u) = R(u+4) − 2R(u+3) + 2R(u+2) − 2R(u+1) + 2R(u) − 2R(u−1) + 2R(u−2) − 2R(u−3) + 2R(u−4)`: slope +1 on `[−4, −3]`, and the slope flips by ∓2 at each integer. The formula holds up to `u = 5`. The real range `[−3, 4]` therefore has a margin of 1 on each side, which is far larger than any approximation error. The output coefficients are `c_{−4} = 1` and `c_k = 2·(−1)^(k+4)` for the other `k`. These coefficients are exact.

## 3. Output layer (3 logits)

With `e = Σ_k c_k F_k − (S(a+p) − S(a−p))/4 ≈ a·(fold(U) − p)`, the logits are:

- `down = −e − DEADBAND·a`
- `stay = 0` (all-zero row)
- `up = e − DEADBAND·a`

The argmax is `up` iff `fold(U) − p > DEADBAND`, `down` iff it is `< −DEADBAND`, and otherwise `stay`. Multiplying the dead band by `a` keeps the threshold in screen heights. `DEADBAND = PADDLE_SPEED/2 = 0.015`: near the target, a step of 0.03 would overshoot by more than staying would miss, so the paddle settles within ±0.015 of the prediction. In a probe, `(up − down)/(2a) + p` recovers the network's predicted crossing, which is how probe 1 checked it.

## 4. When it is exact, when it approximates, when it fails

- **Exact parts:** the `b⁺/b⁻` split, the pass-throughs, the polarisation identity, all 9 fold hinges with their perspective scaling, and the decision rule. The squares are exact when their argument lies on a knot. For example, probe rows with `b = 0` or `x = 0, y = 0.5` came back with error ~1e−15.
- **Approximation:** only the six square interpolants. The worst-case error of the predicted crossing is `0.0035/a + 0.0004 ≤ 0.0074` (with `a ≥ 0.5`, and `fold` is 1-Lipschitz). The largest error measured by probe was 0.00625, at `a = 0.5`, `|b| = 1`.
- **Why that guarantees a hit:** the prediction stays within ε = 0.0074 of the true crossing at every step, so it never drifts. The paddle moves straight to it and settles within roughly `0.03 + 2ε ≈ 0.045` of the crossing. Even a conservative bound needs fewer than 17 of the at least 25 available steps, and 0.045 is less than the 0.08 paddle half-height. So the loss is 0 on every rally of every set, held-out test included. This depends on the analytic bound, not on dev data.
- **Failure modes (outside the game's ranges):** `a < 0.5` or `|b| > 1` would push square arguments out of range. `|U|` beyond the [−4, 5] fold margin would flatten the triangle wave. A much coarser hinge grid also breaks it: with `h = 0.75` the U-error bound grows to about 0.18, and the `coarse_grid` candidate missed 1 of 30 rallies. Inputs other than the stated observation layout are also out of scope.
