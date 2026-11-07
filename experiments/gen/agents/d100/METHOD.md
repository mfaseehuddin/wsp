# METHOD d100
Design first: residual layout (codes + flags + pointers), then sections written by hand in model.wts (embed, attention layers, MLP script tables, position table P and P^2 for 512 positions).
Runs (budget 100):
- b1 (12 runs): run before the position table was in place (my mistake: a duplicate marker line blocked the edit). Wasted.
- b2/b3: with positions: ls, show, delete confirm, protected, lookup, find passed. Found that copied tokens carried their own predictions (copy source polluted) -> added clean code snapshot head L0H4.
- b4 (39 diverse): 13 pass; failures: key words polluted by predictions (fixed by clean keys in value finder), unsafe comma edge, note text containing `delete` selecting a file pointer, no calc yet.
- b5/b6: added calc family and lookup-calc, number pointer sharpening (weights 0.98 gave non-integer pointers), file-without-key coref rule, `see` removed from show; calc, followups, injection, unsafe pass.
- b7: found an ordering bug: sink flag exists only after L3 attention so nokey neurons must be in L3/L4 MLPs; fixed, then a regression (layer-4 neurons read class dims already overwritten by the copy head) fixed using clean snapshot dims, not re-run (runs exhausted).
Probes used for diagnosis: 7. Not built: find-lookup, sum2.
