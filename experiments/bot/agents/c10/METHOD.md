# Method (c10, 10-run budget)

## Design process
With only 10 runs I needed a design that is correct by construction and checkable without running the model.
1. I read the KB, the model, the tokenizer and the harness. Key facts:
   - The think text persists in the context.
   - The trigram (previous word, current word) is unique inside every reply, so the script plus two words determine the next word.
2. **Scripts plus codewords.** Using rare filler words as think codewords removes the need for "anchor" detection. Most-recent-codeword heads fit in layer 0, so the whole bot fits in 2 layers.
3. **Turn scoping.** A turn head copies the rotation code of the latest `<user>`. The vote head then matches equal codes, which gives a uniform average over the current turn only.
4. **Word coding.** A greedy colouring gives words distinct colours within each script (C=23), and groups make every (c, g) unique (G=18). The same colour serves as input feature and as half of the output code.
5. **Copying.** A gated attention head copies the latest id, user, service, state or status, switched on by an ALLOW flag written by layer-0 units. Topic and team come from TCUR through gate units.
6. `check_design.py` checks the design tables symbolically (sets of features, with no tensors and no forward pass):
   - every script position, with random slot fillers and stale codewords, fires exactly the intended unit (0 failures);
   - my hand-written keyword table picks the right script and topic on every dev user turn (0 failures). I wrote the keyword list by hand while reading the dev phrasings and the KB; it was not counted or fitted.

## Runs
1-2. `eval 3,11`.
   - 11 (hello, service, thanks, bye) passed.
   - 3 failed: after the FAQ codeword the bot wrote `<pad>`. The cause was the "suppress topics" unit at `<think>`, which sat in layer 0. Its −50 was written into that position's VOTE, and the layer-1 vote head then averaged it into the topic votes of later positions. **Fix:** moved both suppress units to the layer-1 MLP.
3-5. `eval 3,6,7`: 3/3 passed (faq→escalate→recall, urgent, reset of an unknown user). The coordinator's `<unk>` result fix is covered, because the unknown-user branch keys on "result is not `sent`".
6-9. `eval 4,10,55,9`: 4/4 passed (status, switch-escalate, reset of a known user, unknown-faq).
1 run left unused.
