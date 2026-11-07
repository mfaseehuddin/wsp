# Addendum: you may design your own architecture

Everything in `experiments/bot/TASK.md` applies to you: the same task, hard rules, deliverables, scoring and token budget (250,000, plus the 100k wrap-up rule). Your run budget is **500**. The differences are below.

## What's different

1. **You may replace the fixed transformer** with an architecture of your own design. Write `experiments/bot/agents/<you>/arch.py`, defining `forward(params, tokens) -> logits` of shape [B, T, 2000]. Optionally also define `trace(params, tokens) -> dict` for `probe --trace`. In `weights.py`, return:

   ```python
   from wsp.bot.custom import CustomArch
   return CustomArch("experiments/bot/agents/<you>/arch.py", "one-line justification"), params
   ```

   The harness, tool loop, scoring and commands stay the same. Read `src/wsp/bot/custom.py` for the rules that `arch.py` is checked against when it loads.

2. **It must still be a neural network built only from tensor operations**: embeddings, matrix multiplies, attention, activations, norms, masks, gating, convolutions, recurrences written as `for i in range(...)`, and so on.
   - No Python branching on values: no if, while, and/or or ternaries.
   - No lookup tables or constants hidden in code. Every learned number goes in `params`, and all of `params` counts toward your size.
   - Using the standard transformer from `src/wsp/bot/model.py` is still allowed. Pick whichever you can justify.

3. **You may change any weight at any time**, between runs or during your work. The only thing that's forbidden is learning by gradients: no backward pass, autograd, optimizers or fitting.

## Justification (required, in `ARCHITECTURE.md`)

If you use a custom architecture, explain for a non-expert:
- **What it is**: a plain-language diagram of the data flow.
- **Why it's better for this job than a standard transformer**, measured on parameters, compute per token, editability for the learning rounds, and how it handles context and tools. Give numbers where you can, e.g. "a standard transformer would need ~X params for the same mechanism because …".
- **What it gives up**, and what it can't do.
- **Embedded fitness**, since a target is microcontrollers like the ESP32 (520 KB SRAM, single-precision FPU, no float64): the memory footprint in float32, operations per token, and whether it still works in float32.
  - Avoid needlessly huge constants. Magnitudes above about 6×10⁴ overflow float16.
  - Exact float64 tricks may fail in float32.

If you keep the standard transformer, justify that choice in `ARCHITECTURE.md` instead.
