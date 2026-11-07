# Stage X: a general conversational transformer, written by hand

Builders are Sonnet `wsp-builder` agents x500 and x2000 (run budgets 500 / 2,000, 300k tokens each). Each writes a **standard decoder transformer** (fixed forward in `src/wsp/stage_x/model.py`) aimed at open conversation, with no canned answers (TASK.md §3).

- Agent CLI: `uv run wsp-x …` (see CHEATSHEET.md). Admin: `WSP_ADMIN=1 uv run wsp-x init|test`.
- Public dev set: `dev.json`, 50 conversations, including a 64-turn session.
- **Hidden test:** authored by an independent agent only *after* both builders have stopped and their weights are frozen (`wsp-x test --model agents/<a>/weights.py --cases hidden.json --out results_<a>.json`). Rubric turns are graded by a separate blind judge.
- `_codex_archive/`: the earlier Codex attempt (x1 never produced a valid build). It is kept for the record and is not used.
