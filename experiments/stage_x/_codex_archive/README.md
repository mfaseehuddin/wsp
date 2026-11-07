# Stage X

One generation agent (`x1`) builds directly explained weights for a standard
conversational transformer. Target: 16,384 tokens, 64 turns, replies up to
1,024 tokens. See TASK.md for the full, deliberately ambitious research goal.

The public builder package contains TASK.md, CHEATSHEET.md, SANITIZER_API.md,
and dev.json. Exactly one native Codex subagent starts with no inherited chat
history and works only on this packet. Claude and external CLI builders are
retired. The parent agent coordinates construction checks and public feedback.

Native subagents share filesystem access: a prompt restriction is not an OS
sandbox. Stage X hidden tests are protected by not creating them until the
builder is stopped and the numeric artifact is frozen. No legacy hidden test
is reused for the new blind evaluation. The builder is never resumed with
hidden prompts or case-level feedback. The earlier Claude filesystem canary
report applies only to the cancelled Claude attempt, not to native agents.

The coordinator executes sanitized constructions in a timed subprocess. Only
float32 arrays and validated JSON are passed to inference. The construction
allowlist is defense in depth, not a general Python sandbox.

Commands (from this repository, using the existing environment):

```
.venv/bin/python -m wsp.stage_x.harness build PATH/weights.py PATH/artifact
.venv/bin/python -m wsp.stage_x.harness check PATH/artifact
.venv/bin/python -m wsp.stage_x.harness eval PATH/artifact --cases experiments/stage_x/dev.json
.venv/bin/python -m wsp.stage_x.harness chat PATH/artifact
.venv/bin/python -m wsp.stage_x.harness freeze PATH/artifact PATH/frozen
```

`eval` continues all supplied turns even after errors. Mechanical checks are
reported as partial checks, not semantic accuracy. Long-context capacity and
successful memory retrieval must be reported separately. Safety calls blocked
by the sandbox still count as model violations. The interactive tool project
is temporary and mutation capabilities default to empty.

The previous experiments and their results are preserved. Stage X scores are
not directly comparable to their closed-task exact-conversation scores.
