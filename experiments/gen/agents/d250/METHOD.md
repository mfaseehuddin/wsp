# Method (d250)
1. Design: residual layout first; CRT residue ids (no vocab access); think tag as intent memory; heads of the form "latest token of class C or sink" (recency 8 logit/token, class bonus 12000); previous-token head from pos features (A,B,A2,B2,BM,AM).
2. Wrote embed/pos/unembed, layer-0 presence heads, classify neurons, ~175 rule neurons, routing neurons (all literal lines, no code).
3. Run 1 (3 ids smoke): hello/unsafe/find-none passed. Run 2 (full dev, 108): 42 ok; failures were ls/show (no copy), calc args, chains.
4. Added induction head (P1/P2), calc arguments, chains. Run 3 (full dev): 29 ok, because neurons 155-175 lacked a -1 bias and fired always. Fixed default bias.
5. Run 4 (8 ids): 4 ok; fixed induction penalty (2-back too strong), then tool words polluting the intent-word head (added op-word head, double head). Run 5 (7 ids): 5 ok; SHOW rule restarted on file names inside content, fixed. Run 6 (7 ids): 5 ok; stale lookup (older result with same key) fixed by recency window, unverified (runs ran out at 0).
Final state: best full-dev score 42/108 before the later fixes; later subsets 5/7, 5/7. Not re-verified on full dev.
