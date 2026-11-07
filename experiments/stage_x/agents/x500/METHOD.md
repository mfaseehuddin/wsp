# METHOD (x500)

## Design process
1. Wrote the rules of a hand-compiled transformer on paper: one-hot tokens, shift heads from a sinusoid table, latest-of-class heads (recency vs USER null key), binary-gate neuron pairs, induction copy over +-1 token codes, emit channel decoded by cosine.
2. Built a small compiler inside `weights.py` (functions only, sanitizer-safe): items (rule / shift / latest / induct) declare reads and writes; a scheduler assigns layers; channels are allocated by name; groups are generated per head tensor and per rule tag.
3. Modules added in this order: greeting, calc tool (JSON chain + expression copy + answer frame), names/memory, capital facts + unknown fallback, social, safety refusals, read tool, delete confirmation, exact word echo.

## Run log (457 runs spent of 500: about 200 on 1-6 turn chats and 4 full-dev evals of 46-127 turns each)
- Batch 1 (calc, greeting, ~10 runs): greeting passed; calc looped. Learned: sinusoidal positions add noise to every channel and ruin flag keys -> switched to a learned table that is zero outside the position channels.
- Batch 2 (~20 runs): latest heads blended neighbours (recency step 0.3/token) -> scaled recency 100x and flag gain to 1e6 (weights stay under 60k via bias x weight split). JSON chain collided on short contexts -> keyed on 5 previous tokens plus a start token. Gate coefficient bug found by hand-checking one dot product.
- Batch 3 (~40 runs): copy heads: multiple writers of one channel broke the scheduler (fixed: max over all writers). Induction stopped early or leaked -> re-derived thresholds for exact 2-gram, added a start-flag bonus for the second character, tuned sink 163k -> 150k empirically.
- Batch 4 (~80 runs): added facts, names, social, read/delete/safety; fixed phase-anchor bugs ("is" anchor swallowed its own frame), class "no" containing the emitted "don't", heads over capacity (merged flag heads, M 512 -> 256 to stay under 20M).
- Batch 5 (path/text copy, ~30 runs): added a start-position bonus for the first five characters of a span (fixes notes.txt vs tell), a RESULT-region flag so text copy only sees tool output, inverted recency for text copy (first sentence of a file is copied: Meeting moved to Thursday). Tried a different code packing order: it caused loops, reverted.
- Final dev eval (50 cases, 127 turns): 14 of 46 mechanical passes, 0 safety violations, reuse: identical 5+ word replies only the fallback "I don't know that yet.", stock spans 0.

## Honest summary
Works: calc via tool with answer, names (copy + memory + correction), 20 capitals + unknown fallback, greetings/thanks/bye/how are you/who are you, secrets and ".." refusal, delete confirm/cancel/execute, exact word echo. Fails: most general knowledge, reasoning, instruction formats, supplied-fact answers, multi-turn arithmetic ("multiply that answer"), repeated-bigram copies, file reading when the name shares a bigram with later text ("notes.txt and tell").
Next steps would be a third induction gram (needs larger d_head or fewer layers), a fact store in superposition (random codes) and a pointer head for spans.
