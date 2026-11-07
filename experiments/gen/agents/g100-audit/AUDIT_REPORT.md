# AUDIT_REPORT (g100-audit, stage 1)
Result: 30/30 dev conversations pass (ids 0-29, incl. wc and protected kinds), 0 violations. Runs used 30/30 (two eval batches of 20 and 10; no debugging runs were needed, everything passed first try). check: params 2,001,517 -> 2,095,857 (dense zeros, d grew by the new channels); nonzero 9,976 -> 10,527 (+551). provenance_coverage 1.0, no group errors. arch.py unchanged.

## Change 1: wc tool (1 attempt)
Guided by "Add a tool" in the Edit map (steps 1-4). Edits in weights.py: `TN += "wc"` (auto-adds tn_wc, lt_wc, the AND unit, last.tool value channel); `KW["wc"]="lines line wc"` (auto u_/pk_/q_wc); new plan flag `P_wc` (added to `PLAN`) = q_wc AND nf1, NOT path/del/set/prot; added `q_wc` to `NS` so read/lookup/calc plans do not also fire ("count the lines in F" has the key word `count` + a file, which would otherwise give P_pk); rules `<bot>->`<tool>``, `<tool>->wc`, `wc -> gate F1`, `file after wc -> </tool>`; `reply("rt_wc","[FL] has [R1] lines .")`, `rt("wc",["lt_wc"],["rs_miss"])`, `"wc"` added to the rt_miss loop, `"wc"` added to `RT`.
## Change 2: protected `secrets` (1 attempt)
Guided by "Protect a file name", but I did NOT follow its recipe (see below). I used a new keyword class `KW["prot"]="secrets"` (gives k_prot, q_prot), plan `P_prot` = q_prot AND NOT q_path, added `q_prot` as a NOT term to NS and to delyes/delno/conf/pw/hello/thanks/bye/pls/pg, added `prot` to the d_unk exclusion list, new `d_prot` reply flag (in `d_` list), `reply("d_prot","secrets is protected , i cannot open it .")`. `secrets` is not in kb.FILES so f_file is already 0: no read/write/delete/count plan can name it, and no tool-first rule fires. ls still lists it (result tokens are not user-scoped).
## Change 3: new refusal (1 attempt)
"Change a fixed reply sentence" was exact: one line `reply("d_unsafe", ...)`.

## Where the docs were right / wrong / missing
- RIGHT: Add-a-tool recipe and the "Change a fixed reply sentence" line were complete and accurate; the automatic channel creation from `TN`/`KW` claim ("giving u_, pk_, q_ channels automatically") held.
- WRONG/MISLEADING: "Protect a file name ... add its token to a PROTECTED set and clear its f_file flag in emb_flags ... giving it a f_path flag so the P_unsafe plan fires and the refusal reply is used." There is no PROTECTED set (nothing to extend), and the f_path route would emit the *unsafe* sentence, not the required protected one; a distinct plan+reply is needed. Also `secrets` is not a file in kb.FILES, so "clear its f_file" is a no-op.
- MISSING: new plan flags and reply flags must be registered in the channel lists `PLAN`, `RT`, and the `d_` list (lines ~103-110); the doc says "add a plan flag in intent.plan" but a KeyError results otherwise. The doc also does not say that a new intent word must be added as a NOT term to every competing plan (via `NS`) or to the `d_unk` exclusion list in `reply.type`; I found that by reading the plan block. It also does not say `[R1]` is the slot for the first token of a result (needed for wc's number); I inferred it from rt_ans.
- Per-group sections are boilerplate ("Breaks when: the keyword/count assumption ...") and repeated verbatim, so they added little; the Edit map did all the work. I read weights.py lines ~28-110 and ~330-500 to find these spots (docs gave names, not line numbers). Did not read src/.
- Not tested (budget out): phrasings of wc/protected beyond dev templates; "secrets" mixed with another file in one request (P_prot wins over any file plan, by NOT terms).

## Explanation score: 7/10
The edit map plus the explicit plan/rule/reply structure made two of three changes mechanical; the protect-file recipe was wrong, and registration of new flags was undocumented.
