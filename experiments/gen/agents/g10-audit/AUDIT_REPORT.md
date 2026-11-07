# g10 audit report (stage 1)
Result: 27 of 27 dev conversations I ran pass (all 17 kinds, with the wc/protected/unsafe/session ones several times), no violations. Runs used 27 of 30. Params 1,611,268 -> 1,724,427 (nonzero 113,004 -> 113,328, +324). All edits are in weights.py (the generator); arch.py unchanged. Coverage 1.0, no group names missing from EXPLANATION.md.

## Change 3: new refusal sentence (1 run, shared)
Guide: "Edit map -> Change a fixed sentence: edit the word list in TEMPL ... e.g. r_bye". Edited `TEMPL["r_unsafe"]` (9 tokens -> 9 tokens, m4.emit units regenerate). Exact and sufficient.

## Change 1: wc (passed first run)
Guide: "Add a tool `foo A`" bullet. Changes: `TEMPL["tl_wc"]`, `r_wc`; INTENTS "wc"; `INT["wc"]` rule; RULES `tl_wc`, `r_wc`, and `[wc, nh1, g_missing]` appended to `r_missing`; new class `wc` (LEX `f_wc` = "lines line wc", CLASSES, channels bWC, boolean in m1). Touched rows: embed column f_wc, layer 0 u_wc unit, layer 1 head count_user_words output dim, bWC boolean units, 1 layer-2 intent unit, 3 layer-3 template units, layer-4 emit units.
Docs were not enough on: (a) "count" is also a KEY ("count the lines in data"), so the new intent must beat lookup/show: every old intent needs `nb(bWC)`; I did it by adding to BASE. Not mentioned. (b) A new class makes CLASSES 18 long but layer 1 head dim `e` was 16: I had to set `LAYERS[1]["e"]=18`. Not mentioned (would silently index out of range). (c) The bullet's step "(4) add the tool name word to the vocab flags nothing else needed" is garbled/misleading: a new keyword class (`lines`) was needed. (d) Reusing `r_missing` for the missing case was my inference, the doc says nothing about it.

## Change 2: protected `secrets` (passed first run)
Guide: "Add a protected file P: ... add a flag f_prot for token P in embed, count it as path (put P in LEX["f_path"]). Then any request naming it takes the unsafe rule". This is MISLEADING for this task: following it gives the unsafe sentence, but the spec needs a distinct `secrets is protected , ...` reply, and "read ../secrets" must still be `unsafe`. I instead added a separate class `prot` (LEX `f_prot`="secrets", u_prot, n_prot, bPROT), intent `protected` = BASE0 + bPROT (lower priority than bPATH, so ../secrets stays unsafe), `r_protected` template, `nb(bPROT)` in BASE so no old intent fires (e.g. "delete secrets" would otherwise be unknown/other), and "protected" added to INTENTS so `m3.unknown_default` is suppressed. The doc's final sentence "Grep hits are real tool output ... (not built)" is right and honest: I did not handle grep hits and dev conversations did not need it (not tested on a grep hitting secrets: only 3 spare runs). `ls` listing secrets works because list copy is unconditional, as predicted.

## Right / wrong / missing
- Right: "Each word is one unit in layer 4 (m4.emit); the builder regenerates them", the "Add a keyword/phrasing: ... Only the embed row changes", and the intent-rule table idea (mutually exclusive AND rules) all held. The group tables (what/why per group) let me find m0.user_class, m1.booleans, m2.intent_rules, m3.template_rules.
- Missing: layer-1 head width `e`; need to add `nb(new class)` to existing intents (mutual exclusivity is by construction, a new intent breaks it); that channels like bWC are registered in the channel list in the same file (the doc says "channel list" only for jh).
- Misleading: the protected-file recipe (above) and "(4) add the tool name word to the vocab flags nothing else needed".
- Also: weights.py is a generator script, and EXPLANATION.md names `TEMPL`, `INT`, `RULES`, `LEX` correctly, so locating things was easy. I read all of weights.py (447 lines), not only the pointed parts, since the edits needed cross-references.

## Score: 7/10
Docs located every place to edit and the edit map was mostly accurate, but two of the three changes needed knowledge the docs lacked or got wrong (priority/exclusivity with old intents, head width cap, protected-file recipe).
