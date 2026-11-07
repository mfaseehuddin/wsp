# AUDIT_REPORT g1000-audit (stage 1)

Edits are all in `weights.py` (lists at top of `build()`); `arch.py` unchanged. Params 38,440 -> 41,527 (+3,087, all from new
shape rows/cols: lexicon cols wc/prot, bag/ind entries, 2 new flows, template rows). Nonzero 7,098 -> 7,144. Coverage 1.0.

## Change 1: wc  (EXPLANATION "Add a tool (example wc F ...)" 7-place recipe) - 1 run
Edited: `COLS` (+wc), `CUES['wc']='lines'`, `BAG`, `IND` (+wc), `FLOWS`, `COND['wc']`, `CALLS['wc']`, `RTS['wc']`, `base['wc']`, and 'wc' in the status-1 tuple.
Tensors changed: lex.feat/col.* (new column), bag.sel, ind.sel/thr, flow.*, call.lit/slot, tmpl.*, rt.map.
Wrong in docs: recipe suggests `cue 'how many lines count'` and priority 52. `count` is a numeric KEY, so "count the lines in F" is a lookup;
and priority 52 loses to `lookup` (54) / `lookup_calc` (55). I used cue `lines` and priority 56 (found by reading COND, not by the doc).
Doc also says `COLS`, `BAG`, `IND` must be edited, true. The "tested" recipe line only covers "how many lines are in config ?".

## Change 2: protected `secrets`  - 2 runs wasted by the doc's recommendation
EXPLANATION "Protect a file: append its name to PROTECTED ... the refusal reply" was MISLEADING for stage 1: it routes secrets to the
`unsafe` intent, so the reply is the unsafe sentence, but stage 1 needs a separate `protected` reply. I added a `prot` column
(`CUES['prot']`, in COLS/BAG/IND), a `protected` flow (COND, FLOWS, base, RTS) with no CALLS row (so no tool call is possible).
Priority: I first used 101 (above unsafe) and "read ../secrets" (session conv 48) wanted the unsafe sentence, so I lowered it to 99
(path wins over protected). That last one-line edit (101 -> 99) is **unverified: runs ran out** (I wasted 6 runs re-evaluating after a failed
`sed -i` on macOS, which left the file unchanged). Earlier run with 101: all protected convs (16, 33, 50) and unsafe conv 45 passed; only 48 failed for this reason.

## Change 3: new refusal  - 0 extra runs
EXPLANATION "Change a fixed reply sentence: RTS['thanks']=..." exactly right: `RTS['unsafe']` edited. Correct.

## Dev results
Run a1 (ids 0-16, one per kind): 17/17. Run a2 (32,33,49,50,45,48): 5/6 (48 as above). Not re-run after the priority fix. Other ids not run (102 convs, 30 runs).

## Where the docs were right / wrong / missing
- Right: "To change behaviour you almost always edit lists/dicts at the top of build()"; no arch.py change was needed; `check`/`explain` hints (it told me col.wc, col.prot are missing from the doc, add them).
- Right: "Tool arguments are copied only from user-turn slots" and "`unsafe` intent has no row in call.lit" - the same design gave no-call for `protected`.
- Wrong/misleading: Protect recipe (single reply), wc priority 52 and cue list with `count` (see above).
- Missing: how unsafe and protected should rank when both fire (path + protected name); that priorities must exceed lookup/lookup_calc (54/55) for wc; that new tensors (col.wc, col.prot) need to be named in EXPLANATION.md (I did not edit it, since I was told only to deliver weights/report).
- Docs were not enough for: the priority numbers (read COND, lines ~124-140) and the status tuple (read the rt.map loop).

## Explanation score: 7/10
Edit map found every place and arch needed no change; two of three recipes had a wrong detail (priority, one reply) that only reading code/tests revealed.
