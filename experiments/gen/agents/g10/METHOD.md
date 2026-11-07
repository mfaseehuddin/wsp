# Method
Design on paper first: lexicon classes -> counts -> boolean intents -> (intent, steps done, result branch) templates -> literal/slot emission by step index j. Replies copied by a fixed-lag head.
Runs (10 budget):
1. eval id 0 (ls-show): failed, answered `unknown`. Learned nothing yet.
2. probe+trace (debug): trace strings are 400 chars so rounded chunks were needed; first parse was misaligned.
3. probe+trace again, chunks of 40: n_* counts were all zero. Cause: the u_* units read `in_user` produced in the same layer. Fix: read `nbot` directly (relu(f - nbot)).
4. eval id 0: pass. 5. eval id 84 (delete-yes): pass. 6. eval 29 (16-turn session): pass. 7-8. eval 44, 89 sessions: pass. One run left unused.
Untested in dev: held-out compositions (find-lookup-calc, sum2-followup) are covered by the same templates but were not run.
