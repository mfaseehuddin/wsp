# NOTES (s60)
- Design decision (before any pass): fresh explicit circuit in 12 inserted layers at index 0 (embedding level), no repair.
  Constant C on one dim keeps RMSNorm scale fixed so scratch channels are exactly linear; all scratch is un-computed by mirror
  layers; final answer boost U written after mirrors. See EXPLANATION.md.
- Pass 1 (read): first build gives ppl ratio ~1e16 on guard, so something in the stack does not cancel. Reading resid after layers
  0,1,3,6,9,10,11,12 on one addition prompt + one prose text. Confirm: resid_L12 == resid_L0 on prose; if not, find the first layer
  where the difference appears (the mirror that fails to cancel).
- Pass 1 result: stack runs (lbo/iO/isE/hitP channels look right) but digit detectors fire for several digits (valD = 20 for '1').
  Fix: constrain the other 12 relevant embeddings to score 0 in each LDA detector. Next pass: read layers 3,5,6,7,8 again, expect valD=d+1,
  isD=1, then check GA/GB/BE/AE/trig/N/A/ANS values at the final '='. A correct build should have ANS_x one-hot at the last '='.
- Pass 2 result: digit detectors fixed (valD=d+1). At the last '=' of 123+456= the gather and chain work (lbo, iO, TRIG, carry, S, N, A all
  produce clean-looking values) but gating signals were 0.95 not 1, and gating amplification (x20) turned that into digit errors of 1-2
  (BE_0=4 instead of 6). Fix: added a cleanup layer (saturating step) between gathers and effective-digit gating; stack now 13+4 layers... (17 total).
  Next pass: read 3 texts (9999+1=, 58+7=, prose) at layers after L3 (4), L5 (6), L7 (8), L8 (9), end (13). Expect BE/AE exact digits, S one-hot of 10000 / 65,
  ANS_x one-hot at the last '=' (digits 1 then 0...), and resid_L13 == embeddings on prose.
- Pass 3 result: BE/AE digits are ~20% too small (AE=7.2 instead of 9), lbc=0.99 means the raw lbo hit weight is only ~0.6, i.e. the
  bank/gather attention is not as sharp as designed. Test: read attention maps of layer 1 (bank heads d=0..13) for '9999+1=' prompt;
  expect head d=2 at the last token to put >0.99 weight on the '+' key (offset 2). If not, my RoPE phase convention or score scale is off.
- Pass 4 result: attention maps are sharp (weight 1.0 at the designed offset). The 0.96-0.99 values were the '=' token's own embedding
  baseline in the scratch dims (~0.03), amplified x20 by the gating neurons. Fix: pick scratch dims where the 13 relevant tokens have ~0
  embedding, and subtract the '=' token baseline in each MLP neuron's bias (calibration). Next pass: same 3 texts; expect BE/AE exact
  digits (AE_0..3=9, BE_0=1 for 9999+1=), S one-hot (0,0,0,0,1), ANS_1 then at later positions ANS_0.
- Pass 5 result: BE/AE exact, S one-hot, ANS fires correctly (9999+1 -> 1, 58+7 -> 6). Next: guard, then eval.
- Guard after pass 5: ppl_ratio 1.21, top1 0.72. Prose cancels (2e-4) but digit tokens leave valD/isD residue: the L0 classifier mirror
  re-reads the valD/isD channel dims (u had support there). Fix: exclude those 4 dims from the classifier support. Rebuild + guard (free).
- Guard after classifier fix: ppl 1.126 top1 0.75, still bad. Pass 6: read code/prose with digits, compare resid_L13 vs L0 per token to locate the non-cancelling residue.
- Pass 6: plain code/prose cancel to 3e-4 (fp16 read). Guard still 1.13/0.75 so residue must come from specific patterns. Pass 7: read varied snippets (3+4=7, x=5, a+b=, tables, dates, lists) to find non-cancelling tokens.
- Pass 7: residue only at '=' and answer tokens (norm 300-480): that is U (boost) written at layer 13 and carried through the model. Cancelling is fine elsewhere. Pass 8: eval 6 digits fewshot n=128 to see accuracy with the current boost; expect high if ANS works end to end.
- Pass 8: 6-digit fewshot acc 100% (n=128) with GBOOST=60 (graft_v1_acc100.py saved). Guard bad (ppl 1.13, top1 .75) because U (|U|~400) rides the residual at '='/answer tokens. Trying GBOOST=12; then guard + eval 12 digits fewshot.
- Harness guard bug fixed by coordinator; earlier guard numbers (1.13/.75, 1.21/.72) are void. Re-running guard.
- Guard (fixed harness) with GBOOST=12: ppl 1.0 top1 1.0. Pass 9: eval 12 digits fewshot n=128 with GBOOST=12; expect ~100% if boost 12 (|U|~80) still wins over the model's own logits.
- Pass 9: 12-digit fewshot acc 0% with GBOOST=12 (outputs break after 1-3 digits; 6 digits at 60 was 100%). Back to GBOOST=60: rebuild, guard, eval 12 digits.
- Pass 10: GBOOST=60: 12-digit fewshot 100% (n=128), guard ppl 1.0 top1 0.9998. graft_v2_good.py saved. Next: eval zeroshot 8 digits, fewshot unequal 12/3 (expect fewshot OK; zeroshot may fail on la).
