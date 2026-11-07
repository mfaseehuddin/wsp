# b0: method (forward-pass budget 0)

## Passes used: 0 of 0

| # | call | cost | purpose |
|---|---|---|---|
| 1 | `wsp-pong status --agent b0` | 0 | confirmed budget 0 / used 0 |

I made no `eval` and no `probe` call. I never ran the network, `game.play`, or any harness
internal, and I did not reimplement a forward pass. There is no dev loss.

## How correctness was established without running the network

### 1. Derive the target exactly, so that no part of the design is fitted
I read `game.py`. From `observe()` and `crossing_y()` I derived that
`crossing = fold(y + 1.5 * vy_n * (1-x) / vx_n)` holds exactly from any observation
(EXPLANATION section 0). This leaves the triple product as the only source of approximation.

### 2. Choose approximations with error bounds from analysis rather than from samples
- `vy_n*(1-x)`: polarization `(s^2 - q^2)/4` with a piecewise-linear interpolant of `t^2`
  at knot spacing 0.2. The bound `|err| <= h^2/16` comes from the standard interpolation
  error `(t-k_j)(k_{j+1}-t)`.
- `1/vx_n`: no approximation of the function at all. Gates threshold `vx_n` at
  `1/t_i` for the 31 equal cells `t_i` of `r` in `[1, 2]`, and the readout is the
  midpoint of the cell. The bound `|r_hat - r| <= 1/62` follows by construction.
- fold, passthroughs and comparison: exact identities.
- Width budget: layer 2 = 2 + 2*31 = 64 and layer 1 = 2 + 2*15 + 30 = 62. I chose 15
  square knots instead of 16 because `2/h` must be an integer to keep the knots symmetric
  about 0, and that symmetry is what makes `a_hat = 0` exactly when `vy_n = 0`.
- Closed-loop argument (EXPLANATION, "Closed-loop guarantee"). The worst-case final gap is
  0.054 (0.057 in a gate band), below 0.08, and convergence takes at most 17 of the 25 or
  more steps.

### 3. Structural inspection of the built weights (no forward pass)
I called `build()` and printed shapes and selected entries:
- layers `Linear(5->62), ReLU, Linear(62->64), ReLU, Linear(64->10), ReLU, Linear(10->3)`.
  All are float64 with `requires_grad=False`, and widths are at most 64 with 3 hidden layers.
- `S_0` row `[-1,0,0,1,0]`, bias 2.0. `S_14` bias -0.8 (= 1 - 1.8). `Q_0` row
  `[-1,0,0,-1,0]`, bias 2.0.
- `V_1` bias `-0.96875 = -31/32` and `V_30` bias `-0.508197 = -31/61`.
- `alpha` entries: `S_0: -0.45`, `S_1..14: +0.1`, `Q_0: +0.45`, `Q_1..14: -0.1`.
- `G+_5` and `G-_5` have a single `-1e4` on H1 unit 36, which is `V_5`. `G+_0` has no gate
  weight.
- Every layer-3 row: `Y2` 1.0, `G+_0` 1.52419 (= 1.5*(1 + 1/62)), `G+_i` 0.04839
  (= 1.5/31), with the negated values on the `G-` units. Biases `[4, 3, 2, 1, 0, -1, -2, -3, -4]`,
  and `P3` reads `P2`.
- Output rows: down `[-1, 2, -2, 2, -2, 2, -2, 2, -2 | +1]`, up = the negation, stay bias 0.015.

`ruff check` and `ruff format` both pass.

### 4. Hand-computed worked examples, unit by unit (in reasoning, not code)

**Example A (bounce off the floor, early in the rally).** obs `x=0.2, y=0.3, vx_n=0.6,
vy_n=-0.8, p=0.5`. The truth is `m = 0.8`, `r = 5/3`, `z = 0.3 - 1.6 = -1.3`, and the
crossing is `fold(-1.3) = 0.7`.
- `s = 0.0` and `q = 1.6`. `S`: `S_0=1.0`, `S_1..4 = .8,.6,.4,.2`, rest 0, sum(j>=1) = 2.0.
  `Q`: `Q_0=2.6`, `Q_1..12 = 2.4 ... 0.2` with sum 15.6, and `Q_13 = Q_14 = 0`.
- `a_hat = -0.45(1.0-2.6) + 0.1(2.0-15.6) = 0.72 - 1.36 = -0.64`, which equals `vy_n*m`
  exactly (both points are on knots).
- `r = 1.667` falls in cell 20 (`20.67`). `v_20 = 31/51 = 0.6078 > 0.6`, so the gate is
  open. `v_21 = 31/52 = 0.5962`, so `V_21 = 0.0038` and `1e4*V_21 = 38 > 0.64`, which
  closes it.
- `G+_i = 0` for all i. `G-_0..20 = 0.64` and `G-_21..30 = 0`.
- `z_hat = 0.3 - 1.5*0.64*(1 + 20.5/31) = 0.3 - 1.59484 = -1.29484`, an error of 0.005.
- `F_{-4,-3,-2} = 2.70516, 1.70516, 0.70516` and the rest are 0, so
  `T = 2.70516 - 3.41032 + 1.41032 = 0.70516`.
- logits `[-0.205, 0.015, +0.205]` give **up**, which is correct: the paddle at 0.5 must go
  toward 0.7.

**Example B (near the paddle, inside the deadband).** obs `x=0.97, y=0.42, vx_n=0.9,
vy_n=0.5, p=0.44`. The truth is `z = 0.42 + 1.5*0.5*0.03/0.9 = 0.445`.
- `s = 0.53` and `q = -0.47`. `S_0 = 1.53` and `S_1..7` sum to 5.11. `Q_0 = 0.53` and
  `Q_1..2` sum to 0.46.
- `a_hat = -0.45*1.0 + 0.1*4.65 = 0.015`, which equals `0.5*0.03` exactly. The two
  points share an in-cell offset, so the interpolation errors cancel: `f(.53)=.29` and
  `f(-.47)=.23`.
- `r = 1.111` falls in cell 3. `v_3 = 31/34 = 0.9118`, so the gate is open.
  `v_4 = 31/35 = 0.8857`, so `V_4 = 0.0143` and `1e4*V_4 = 143`, which closes it. Then
  `r_hat = 1.11290`.
- `z_hat = 0.42 + 1.5*0.015*1.11290 = 0.44504`, and `T = 4.445 - 6.89 + 4.89 - 2.89 + 0.89 = 0.44504`.
- logits `[-0.00504, 0.015, 0.00504]` give **stay**, which is correct: `|T-p| < 0.015`.

**Example C (worst-case quantization, two bounces).** obs `x=0, y=0.9, vx_n=0.5,
vy_n=1, p=0.5`. The truth is `z = 3.9` and the crossing is `fold(3.9) = 0.1`.
- `s = 2` and `q = 0`. `S_0 = 3` and `S_1..14` sum to `0.2*(1+...+14) = 21`. `Q_0 = 1` and
  `Q_1..4` sum to 2.0. So `a_hat = -0.9 + 1.9 = 1.0`, which is exact.
- `vx_n = 0.5 < v_30 = 0.5082`, so all 31 gates are open and `r_hat = 1 + 30.5/31 = 1.98387`.
- `z_hat = 0.9 + 2.97581 = 3.87581`, an error of 0.0242, the maximum quantization bound
  `1.5/62`.
- `F_{-4..3} = 7.8758, 6.8758, ..., 0.8758` and `F_4 = 0`. The alternating sum is
  `T = 0.12419`, which matches `fold(3.87581) = 2 - 1.87581`.
- logits `[0.376, 0.015, -0.376]` give **down**, which is correct.

**Example D (interpolation off-knot).** `vy_n=0.3, m=0.4`: `s=0.7` and `q=0.1`. Then
`S_0 - Q_0 = 0.6`, which contributes -0.27. `sum S_{j>=1} = 6.4` and `sum Q_{j>=1} = 2.5`,
which contribute +0.39. So `a_hat = 0.12`, equal to `vy_n*m`. The errors are
`e(0.7) = e(0.1) = 0.01 = h^2/4`, and they cancel.

**Fold spot checks.** `z=0.902` gives `T=0.902`, `z=1.3` gives `T=0.7`, and `z=-0.4` gives
`T=0.4`. All match the triangle wave.

### 5. Closed-loop sanity reasoning for the hardest reachable case
The hardest case is the fastest ball (`vx = 0.04`, 25 decisions) with a crossing at
0.0. The paddle must move from 0.5 to 0.08 or below, which takes 14 steps. By the
monotone-approach argument (EXPLANATION), the paddle steps down on every decision
while its gap exceeds `E + 0.015 <= 0.047`. It therefore reaches `p <= 0.047` by
decision 16 and holds within the deadband after that, finishing with `p` about
0.0-0.03. That is a hit.

## Predicted result
Hit rate 100% and mean loss 0.0 on dev and test. This is a derived prediction that I
have not measured, because I had no budget to measure it. The residual risks are an
implementation slip that the structural and hand checks above did not catch, or a
physics change in `game.py`.
