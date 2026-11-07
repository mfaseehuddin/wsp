# nco60 notes

Design: K=2 levels (1 mode code, then 64 token codes), 6 layers x 7 heads, d_model 784, norm off.
Prefix layers build local structure (prev-token heads, counters, run starts); level rows gather the
question sentence, detect mode features, pick an answer token by attention, and copy a span by hops.

- run 1 (eval, v1 baseline): 2/33 mechanical. Greeting/name/bye/decline/purple work locally;
  knowledge, arithmetic and instructions are not attempted; recency (mango vs pear) and right-context
  answers (Zorbin) fail. Forgot to save the eval stdout; later evals go to scratch/evalN.txt.
- run 2 (eval, v2): 3/33. Added SELF/REQUEST modes, run-start copy (RSID), PM-gated ANSWER. Saved scratch/eval2.txt.
- run 3 (eval, v3): 3/33. CLARIFY (deictic and no first person). Requests now get honest refusals.
- run 4 (eval, v4): 6/33. Strict/broad stops fixed multi-piece names; REQUEST echoes topic (fox/polite pass).
- run 5 (eval, v5): 7/33. Pronoun and wh-word tokens made stops by concept geometry; fact recency kernel; greet/bye/harm anchors retuned. Fresh local set: teal, Lady Morn, Gwendolyn greeting work.
- run 6 (eval, v6): 7/33. ACK echoes the newest content run (but garbles instructions).
- run 7 (eval, v7): ACK echo only when first person present; hops/dots gated by copy validity; ET reduced to 2 freqs.
- run 8 (probe): level 0 code 6 (ANSWER), level 1 copies 'named Pumpernickel'; level-0 latent decoded to '!!!' so dec_b2[im_end] raised slightly.
- run 9 (eval, v8 final): 7/33 (name_intro, fox_story, polite, pelins x2, two_entities, coref). Tried own-meaning
  match (QS.SEMW) and a softer content curve: both broke teal/noon locally, so reverted.
