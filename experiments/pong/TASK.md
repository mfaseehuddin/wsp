# Task: generate a Pong-playing network without training

You are one of several agents. Each agent gets a different budget of **forward passes**. Your job is to write the weights of a small ReLU network that plays Pong, explain every weight group, and show how you used your budget.

## The game

Read `src/wsp/pong/game.py`. It is the complete physics, and you may study it as closely as you like.

- In each rally the ball starts at the left wall and moves right, bouncing off the top and bottom walls. Your paddle sits at x = 1, starts at y = 0.5, and moves at most 0.03 per step. The rally ends when the ball reaches x = 1.
- At every step your network receives `[ball_x, ball_y, ball_vx_n, ball_vy_n, paddle_y]` and outputs 3 logits: `[down, stay, up]`. The action is the argmax, with ties going to the lowest index.
- The paddle is slower than the ball's fastest vertical speed. Simply following the ball's current height is not enough; you need to anticipate where it will cross x = 1.

## Loss (fixed in advance)

For each rally, loss = `max(0, |paddle_y - crossing_y| - 0.08)`, i.e. how far the paddle's edge misses the ball, measured in screen heights. It is 0 for a hit. The score is the mean over rallies. Lower is better.

## Network contract

Your policy is `experiments/pong/agents/<you>/policy.py`. It defines `build()`, which returns a `torch.nn.Sequential` of alternating `nn.Linear` and `nn.ReLU` layers. The input width is 5 and the output width is 3, with at most 3 hidden layers of width ≤ 64. Set every weight and bias explicitly, from formulas you derived. The model runs in float64 on CPU.

## Your budget and the harness

Every forward pass costs 1. One observation row is one pass, and a rally costs about 35 passes (25–50). The only way to run your network is through the harness:

```bash
uv run wsp-pong status --agent <you>
uv run wsp-pong eval   --agent <you> --policy <file.py> --rallies N [--offset K] --note "why"
uv run wsp-pong probe  --agent <you> --policy <file.py> --obs '[[x, y, vx_n, vy_n, p], ...]' --note "why"
```

- `eval` plays N rallies from a fixed dev set (offset K picks which ones, 0 ≤ K, K+N ≤ 20000). It returns the mean loss and hit rate. The cost is computed before the run; if it is over budget, the request is refused and nothing is charged.
- `probe` returns logits for observations you choose, at 1 pass per row. It is useful for checking a single unit or layer cheaply.
- Every call is logged to `experiments/pong/agents/<you>/log.jsonl`. Always pass a `--note` that says what you are testing.
- You may keep several candidate files, e.g. in `experiments/pong/agents/<you>/candidates/`, and eval any of them.

After you finish, your final `policy.py` is scored on a held-out test set that you cannot access.

## Hard rules

1. **No backward pass and no training.** No autograd, `.backward()`, optimizers, or gradient estimates.
2. **No fitting to data.** Don't generate observation/target samples from the simulator and fit weights to them, whether by least squares, regression or anything else. Constants must come from analysis of the physics and of the functions involved. For example, you may choose piecewise-linear breakpoints for a known function such as 1/v by reasoning or by evaluating that function on paper. You may not fit to simulated game data.
3. **Every forward pass goes through the harness.** Don't call your network, `game.play`, `harness.evaluate` or any harness internals from your own code, and don't reimplement a network forward pass in numpy or by hand on many inputs. You may hand-check a few individual units in your reasoning. Don't import or run `src/wsp/pong/baselines.py`.
4. **Using forward passes to search is allowed** and is the point of the budget. You may compare candidates, tune a handful of interpretable constants by zero-order search, or probe units. Each change must be a change to a parameter whose meaning you can state.
5. **No internet.** Don't use WebSearch or WebFetch, and don't download weights or code.
6. **Stay in your own directory.** Don't read or write other agents' directories, don't edit ledgers, `src/`, `tests/` or this file, and don't set `WSP_ADMIN`.

## Deliverables (in `experiments/pong/agents/<you>/`)

1. `policy.py`: the final network. Build the weights from named constants and comments, not from unexplained numbers.
2. `EXPLANATION.md`: layer by layer and weight group by weight group:
   - what mechanism the group implements (e.g. "units 0–7: piecewise-linear approximation of the time to impact, (1 − x)/vx"),
   - why the values are what they are, including the derivation,
   - when the mechanism is exact and when it approximates or fails.
3. `METHOD.md`: how you spent your forward passes, in order, with each eval or probe, what you were testing, what you learned, and what you changed. If your budget is 0, explain how you made sure the network was correct without running it.

Finish with a short final reply: your final dev loss (if you have one), the passes you used, and a 3–5 sentence summary of your design.
