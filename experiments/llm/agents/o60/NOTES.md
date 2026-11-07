# o60 notes

## Design decision (before any pass)
Weight analysis only so far. Plan: a self-contained exact adder in 9 inserted layers at the bottom
(before original layer 0), reading clean embeddings. It computes the answer digit at each output position
and "parks" a huge vector lambda*v_d there; v_d is built (pinv vs competitor tokens) so that after the final
norm + tied unembedding the digit d is top-1 with ~20-30 logit margin. Because every original layer
RMS-normalises its input, its output is bounded, so lambda >> those outputs makes the final logits follow v_d.
On non-addition text all gated neurons are off, flags and the normalising constant C are cancelled exactly,
and only tiny (1e-4 scale) attention scratch remains -> guard should be ~1.0.
Carry: lookahead via sign of sum_j (g_j-k_j) 2^j (exact in float32). Place alignment via RoPE offset-fetch
heads (constant q/k, peaked kernel over 6 high-frequency pairs). Zero-shot start-of-text (missing offsets)
handled by a content head that measures 1/(la+1) when only digits precede '+'.
Did not use the model's own addition circuit: analysing it would cost passes/tokens, and the failure
(carry chains) is exactly what a fresh lookahead circuit fixes.

## Pass 1 (eval fewshot, 8 digits, n=256)
Graft built (9 inserted layers). Testing whether the whole pipeline works end to end. Confirm: accuracy near 100%.
Refute: low accuracy -> then a `read` pass on inserted-layer residuals to locate the broken stage.
Guard findings (free): an UNEDITED build already scores ppl_ratio 1.1257 / top1 0.7476 (harness baseline offset,
not my graft). Full graft without park neurons: 1.1256/0.7471 (scratch is silent). With parks: 1.171/0.743, so parks
fire on some guard text (probably genuine 'a+b=' or multi-term sums, plus the forced newline). Now the first eval.

## Pass 1 result: acc 0 (outputs like '1132840'). Pass 2: read residuals after each inserted layer on two
teacher-forced problems and decode every scratch quantity to find the first stage that is wrong.
Pass 2 result: read arrays are float16, so small-scale scratch cannot be decoded from resid; large ones showed
V=0.67 and no F_p firing. Cause found by analysis: gates M*(flag-1) used stored flags whose read-back scale
drifts with rms by ~4e-5 at digits (x M=2e5 -> several units). Fix: gates now use token-detecting functionals
w (min-norm in span of relevant embeddings), exact at any rms. Pass 3: same eval; expect large accuracy jump.
Pass 3: acc 0 but parks now fire (digits come out, wrong values). Pass 4: read MLP activations (exact in fp16
relative terms) of layers 1-4 on the two dbg problems to find the first wrong neuron group.
Pass 4 result: zero-shot masks and validity steps are right, but fetched digit VALUES at '+' are off by up to 0.9
(e.g. offset 7 reads 0.09 instead of 1) while fetched DIG flags are ~exact. Pass 5: read layer-1 attention maps and
layer-0 MLP (token detectors) to tell apart "attention not sharp" from "wrong VAL flag".
Pass 5 result: layer-1 attention is perfectly sharp (weight 1.0 at offset d) and detectors are right, so the value
errors came from float32 cancellation in neurons with huge gate constants (error ~2e-6*M, M up to 1.75e5).
Rescaled: attention-written quantities 10x larger, masking moved to clean step outputs, no V in carry gates.
Max gate M now 5.4e3. Pass 6: eval 8-digit fewshot; expect most problems right.
Pass 6 result: 98.4% (252/256) at 8 digits fewshot; guard 1.1266/0.743 vs unedited-build 1.1257/0.7476.
Failures are single digits off by one. Pass 7: read MLP layers 3,4,5,7 on two failing problems to see which
stage carries the error (operand sums, carries, exports, or the select stage).
Pass 7 result: exports carried constant offsets (+0.08..0.16 on zero digits) and Y lost up to 0.43 at '=' -> traced
to the float32-rounded C*u0 vector leaking into every read row (1e-4..1e-3 per unit). Fix: constant channel is now
the exact axis 221, all read/write directions have an exact 0 there, and the gate functionals carry the e_221/C
correction. Pass 8: eval 12-digit fewshot (longest carries); expect >=99%.
Pass 8 result: 12-digit fewshot 256/256 = 100%. Guard 1.1267/0.743 (unedited build 1.1257/0.7476).
Pass 9: own problems, zero-shot, random lengths 1-12 each (unequal) + edge cases (0+0, 9..9+1, 1+9..9, 5+5).
Zero-shot exercises the start-of-text masking; failure would point at ZS/Z-head.
Pass 9 result: 256/256 zero-shot with mixed lengths 1-12 and carry edge cases.
Pass 10: gen on unseen-style prompts (prose/other few-shot prefixes, a long ~700-token prefix, problems after
other '+'/'=' text) to check generality and the forced newline. Expect exact answers followed by '\n'.
Pass 10 result: 32/32 exact on 9 prompt styles (prose prefix, code prefix, other few-shot lines, '5-3=2' lines,
a 700-token prefix), each answer followed by '\n'.
Passes 11-13: fresh-seed confirmation (6-digit fewshot seed 5; 3+11-digit zeroshot seed 6; 10-digit fewshot seed 9).
Expect 100%; any failure would be a precision edge case.
Passes 11-12 results: 6-digit fewshot seed 5 = 100%; 3+11-digit zeroshot seed 6 = 100% (the 10-digit run in a
batched command never executed). Final build + guard: ppl_ratio 1.1267, top1 0.743 (unedited build: 1.1257/0.7476).
Stopping: every dev test is at 100%. Open risk: false triggers on in-text sums like "1+2+3=" or "1.5+2=".
