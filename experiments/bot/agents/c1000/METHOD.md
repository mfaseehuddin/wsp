# c1000: method and run log

## Design process

- Read TASK.md, model.py, tokenizer, harness (generation/tool loop) and the KB. Looked at the dev user turns once to understand the shapes. Keywords were written by hand, partly from general IT vocabulary and partly from real-looking words I noticed while skimming the vocabulary list. Nothing was counted or fitted.
- Key decisions:
  1. **Commit decisions to tokens.** The bot writes `<think> MODE TOPIC </think>` with private filler code words. The argmax over keyword votes happens in the logits, and greedy decoding turns it into a discrete token. After that, everything is a deterministic script, and the note doubles as memory for escalation.
  2. **One 2-D dial for all output words.** Codes are angles on a circle, matched with large gains (float64 makes this exact). This keeps d_model at 70 even with ~400 sayable words.
  3. **Script steps as MLP conjunction units** keyed on (mode, topic, prev, cur), with the least specific key chosen automatically from the KB scripts. Copies (ticket id, user name, result value) use gated "most recent X" attention heads.
  4. **Current-turn bag of words** through attention: match the "start of turn" angle and mask out non-user tokens, gated so it only fires at the decision points.
- Harness notice mid-build: the unknown-user result is `<unk>`. The reset script now branches on prev = `<unk>`.

## Runs (in order; 358 of 1000 used)

| # | runs | what | result / change |
|---|---|---|---|
| 1 | 12 | eval dev 0-11 (one of each kind), first build | 12/12 |
| 2 | 58 | eval dev 12-69 | 58/58 → 70/70 dev |
| 3 | 38 | `turn` on 38 of my own single-turn paraphrases | all right except "clicked a link in a strange email" → email (fixed below); recall with no ticket in context loops (by design, no ticket) |
| 4 | 21 | `turn` multi-turn paraphrases after a FAQ + escalation | misses: "did not fix it", "same issue" (escalate), "see you", "that's all for now" (bye), "ticket id" (recall), "overheating" (no keyword) → broadened all keyword lists |
| 5 | 70+16 | dev regression + 16 new paraphrases | 70/70; 15/16 paraphrases right (ticket-id recall had no ticket in context) |
| 6 | 70+3 | added "work/working" as escalation words; regression + 3 probes | 70/70; "printer still not working" escalates (known ambiguity) |
| 7 | 70 | dial angle = vocab id (stable for future edits), S_OUT 1e7→1e9 | 70/70 |

Final: **70/70 dev conversations pass**, 445,010 parameters.
