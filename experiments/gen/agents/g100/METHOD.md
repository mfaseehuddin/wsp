# Method
Design: wrote the intent/plan logic and a per-position next-token rule table first on paper, then built a channel allocator, a head builder (explicit score terms) and a unit builder in weights.py. Key tricks: sink tokens so counting heads return n/(n+2); ordinal tags to get first/second file; masked key codes so lookup sinks do not depend on the query; (token, prev, prev2) induction for list copy.
Runs (100 budget, 35 used):
1. eval ids 0,3,5,6,7 (ls-show, find, lookup-calc, find-lookup, sum2): tests that the arch loads and basic flows work -> 5/5.
2. eval ids 1,2,4,8-14 (one per other kind, one session): 10/10.
3. eval 29,44,59,74,89 (sessions) 21,22,16,19,26,35,50: 12/12.
4. turn on conv 6 "what is the batch in the file that mentions low plus 5 ?" (held-out composition): grep, read, calc 19 plus 5 -> answer 24, correct.
5. eval 13,14,29,44,59,74,89 after adding keyword synonyms (kw lists only): 7/7.
No fixes were needed for dev; synonyms were added blind for hidden phrasings.
