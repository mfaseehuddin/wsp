# b0: how the network works

Architecture: `5 -> 62 -> 64 -> 10 -> 3` (Linear/ReLU alternating, float64).
Inputs: `[x, y, vx_n, vy_n, p]`. Outputs: logits `[down, stay, up]`.
All constants are named in `policy.py`, and `model.unit_names` labels every unit.

## 0. The quantity being computed (derivation)

The crossing height is `fold(y0 + vy/vx)` (`game.Rallies.crossing_y`). At decision `k`
the ball's unfolded height is `u = y0 + k*vy` and `x = k*vx`, so the displacement still
to come is `vy*(1/vx - k) = vy*(1 - x)/vx`. `fold` is even and has period 2. On the rising
branch (`u mod 2 <= 1`) `fold(u + d) = fold(y + d)`. On the falling branch it equals
`fold(y - d)`. The observed `vy_n` already has the sign flipped on the falling branch, so

```
crossing = fold(z),   z = y + vy_cur*(1-x)/vx = y + 1.5 * vy_n * m * r,
m = 1 - x  in (0, 1],   r = 1/vx_n  in [1, 2],   1.5 = VY_MAX/VX_MAX.
```

This holds exactly at every step of every rally. The only approximation in the network
is the triple product `vy_n * m * r`. Everything else (passthroughs, fold, comparison)
is exact piecewise-linear algebra.

## Layer 1 (62 units): features of the inputs

| units | name | formula | purpose |
|---|---|---|---|
| 0 | `Y1` | `ReLU(y)` | passthrough (`y >= 0`) |
| 1 | `P1` | `ReLU(p)` | passthrough (`p >= 0`) |
| 2-16 | `S_j` | `ReLU(s - k_j)`, `s = vy_n + 1 - x`; weights `x:-1, vy_n:+1`, bias `1 - k_j` | hinge basis for `s^2` |
| 17-31 | `Q_j` | `ReLU(q - k_j)`, `q = 1 - x - vy_n`; weights `x:-1, vy_n:-1`, bias `1 - k_j` | hinge basis for `q^2` |
| 32-61 | `V_i` (i=1..30) | `ReLU(vx_n - v_i)`, `v_i = 1/(1 + i/31)` | speed gates: zero exactly when `r >= t_i` |

Knots: `k_j = -1 + 0.2 j`, j = 0..14 (`SQ_LO=-1`, `SQ_H=0.2`). `s` and `q` both lie in
`[-1, 2]`, because `vy_n` is in `[-1, 1]` and `m` is in `(0, 1]`. That makes
`ReLU(s - k_0) = s + 1` an exact linear term.

**Why squares.** `a = vy_n*m = (s^2 - q^2)/4`: the polarization identity turns a product
into two one-dimensional functions of input ridges, and one ReLU layer can represent those.
The piecewise-linear interpolant of `s^2` on the knots is

```
f(s) = k_0^2 + (2k_0 + h) ReLU(s - k_0) + 2h * sum_{j>=1} ReLU(s - k_j)
```

The first chord slope is `k_0 + k_1 = -1.8`, and each chord slope is 0.4 larger than the
one before. So `a_hat = alpha . H1` with `alpha(S_0) = -0.45`, `alpha(S_j>=1) = +0.1`,
`alpha(Q_0) = +0.45`, `alpha(Q_j>=1) = -0.1`. The `k_0^2` constants cancel. (`alpha` is
not a layer. It is the row that layer 2 applies to H1.)

**Exact when:** `s` and `q` both sit on knots, or both sit at the same offset inside
their cells. The interpolation error `e(s) = (s-k_j)(k_{j+1}-s)` is h-periodic and
symmetric, so it cancels in `f(s) - f(q)`. In particular `a_hat = 0` exactly when
`vy_n = 0` or when `m` is a multiple of h. **Error bound:** `e` is in `[0, h^2/4]`, so
`|a_hat - a| <= h^2/16 = 0.0025` everywhere. It is also `<= h*m/2`, which goes to 0 as
the ball reaches the paddle.

**Gate thresholds.** `r = 1/vx_n` is split into 31 equal cells on `[1, 2]`, with edges
`t_i = 1 + i/31`. The test `r > t_i` is the same as `vx_n < v_i = 1/t_i`, so the gates
threshold `vx_n` directly and `1/v` is never approximated. Evaluating `1/t` at the 30
edges is the only use of that function (e.g. `v_1 = 31/32 = 0.96875`,
`v_30 = 31/61 = 0.50820`).

## Layer 2 (64 units): gated product `a * r_hat`

| units | name | formula |
|---|---|---|
| 0, 1 | `Y2`, `P2` | passthroughs of `Y1`, `P1` |
| 2-32 | `G+_i` (i=0..30) | `ReLU( a_hat - 1e4 * V_i)` (`V_0 = 0`) |
| 33-63 | `G-_i` (i=0..30) | `ReLU(-a_hat - 1e4 * V_i)` |

`G+_i - G-_i = a_hat * [r > t_i]`. While the gate is open (`V_i = 0`) the pair passes
the positive and negative parts of `a_hat`. Once `vx_n > v_i + |a_hat|/GATE_GAIN` both
units are 0. Since `|a_hat| <= 1.0025`, the transition band is narrower than 1.1e-4 in
`vx_n`. Gate 0 has no `V` input and is always open, because `r >= 1` always.

## Layer 3 (10 units): fold of the unfolded target

`z = Y2 + 1.5 * sum_i c_i (G+_i - G-_i)`, with `c_0 = 1 + 1/62` and `c_i = 1/31`
(i >= 1). This gives `r_hat = 1 + (n + 1/2)/31` when `r` is in cell n, which is the
midpoint of r's cell. So `z ~= y + 1.5 * a * r`.

| units | name | formula |
|---|---|---|
| 0-8 | `F_b`, b = -4..4 | `ReLU(z - b)`: every layer-3 row is the same `z` row, with bias `-b` |
| 9 | `P3` | passthrough |

Error on `z`: `|z_hat - z| <= 1.5*r*|a_hat-a| + 1.5*|a|*|r_hat-r| <= 0.0075 + 0.0242*|vy_n|*m`.
This is at most 0.032 at the serve and at most 0.0094 in the last two steps (m <= 0.08).
Inside a gate transition band, `|r_hat - r|` can reach 1/31 instead of 1/62, which gives
at most 0.056 at the serve and 0.011 at the end. The range is
`z_hat in [-2.98, 3.98]`, inside the fold's valid domain `[-4, 5]`. As a result
`F_{-4}` always acts as a linear unit and `F_4` is never active. Both are kept as safety
margin.

## Output layer (3 logits): exact fold and a bang-bang controller with a deadband

`T = fold(z) = F_{-4} - 2F_{-3} + 2F_{-2} - 2F_{-1} + 2F_0 - 2F_1 + 2F_2 - 2F_3 + 2F_4`
holds exactly for `z >= -4`. The triangle wave starts at 0 at z = -4, rises with slope +1,
and flips slope at every integer, which is a change of -2 or +2.

```
down = P3 - T      stay = DEADBAND = 0.015      up = T - P3
```

`max(down, up) = |T - P|`, so the paddle stays when `|T - P| < 0.015` and otherwise steps
0.03 toward T. At exactly 0.015 a tie goes down when T < P and stays when T > P, which is
harmless. The deadband is half a paddle step, so a paddle that has settled never sits
more than 0.015 from T.

## Closed-loop guarantee, and when it fails

Let `g_k = |crossing - p_k|` and let `E_k` be the error bound above at step k. `E_k`
never increases, since `|vy_n|` is constant and m shrinks.

1. If `g_k > E_k + 0.015`, then T and the true crossing lie on the same side of p and
   `|T - p| > 0.015`. The paddle therefore moves 0.03 straight toward the crossing.
   Clamping to [0, 1] can only reduce `g`, because the crossing is in [0, 1].
2. Starting from `g_0 <= 0.5`, this takes at most `ceil(0.485/0.03) = 17` steps. Every
   rally has at least 25 steps (`vx <= 0.04`).
3. After that, a single step can raise `g` to at most `E + 0.015 + 0.03`. So the final
   gap is at most `E_{n-2} + 0.045 <= 0.0094 + 0.045 = 0.054`, or 0.057 in a gate band.
   Both are below `PADDLE_HALF = 0.08`.

**Prediction: every rally is a hit and the loss is 0.** This is derived, not measured,
because the budget was 0. The guarantee relies on the following:
- the physics constants in `game.py` (`VX in [0.02,0.04]`, `|vy_n| <= 1`,
  `PADDLE_SPEED = 0.03`, `PADDLE_HALF = 0.08`). The policy imports `VX_MAX`, `VY_MAX` and
  `PADDLE_SPEED`, but the knot ranges, the r-range `[1, 2]` and the fold domain assume the
  current values. A larger `VY_MAX/VX_MAX` or a smaller `VX_MIN` would push `s`, `q`, `r`
  or `z` outside their covered intervals;
- observations staying in the documented ranges. An off-distribution probe, for example
  `vx_n < 0.5` or `|vy_n| > 1`, leaves the ranges where the interpolants are valid.
