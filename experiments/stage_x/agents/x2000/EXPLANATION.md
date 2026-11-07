# EXPLANATION (agent x2000)

## How the model works (about 15 lines)
1. A standard 7-layer, 3-head causal transformer (d=448, d_head=88, d_mlp=1024, learned position table, no LayerNorm). Nothing but weights.
2. Every token has a 64-d seeded +-1 code (greedy lowest-coherence pick). Codes live in residual block C; attention copies them, the unembedding reads them (block O) with logit = O . code.
3. Positions are 8 sinusoid pairs (periods 3,5,7,11,13,17,19,23). Rotating a position code by one step in W_q makes a "previous token" head; position 0 (BOS) has a zero code and acts as a null sink.
4. Layers 0-2 build context features at every token: previous-token codes (Vb = c(t-1)+0.7 Q c(t-2)+..., Q a signed permutation), the header of the current sentence (SPex), and a bag-of-letters signature of each word (Tb), which lets invented names be matched without a lexicon entry.
5. Layer 3 gathers, at the BOT token, the question's bag of word signatures, intent flags (greeting, thanks, "you", relation words such as capital/legs) and the known-entity code.
6. MLP 3 turns those into "fast" frame flags (greet, thank, about-me, knowledge-fact hit). Layer 4 retrieval matches the question bag against earlier sentences (own sentence excluded, BOS sink = no match, recency tie-break) and returns the header position of the best sentence. MLP 4 chooses a "slow" frame: ack (statement), qa (question with a match), unk (question without a match).
7. Layer 5 copies the first token of the chosen sentence (pointer = header position + 1, via rotated position code). Layer 6 continues by induction (key = previous tokens equal the current token, restricted to the retrieved sentence), so any bytes, including novel names, are copied verbatim.
8. MLP 6 holds the reusable text units (all <= 6 words): "Hello! How can I help?", "You are welcome.", "I am a small hand-built assistant.", "I do not know that yet.", " Noted.", and the knowledge table (entity, relation) -> value, e.g. France+capital -> Paris, each fact a pair of AND neurons.
9. Replies are composed: a statement is acknowledged by copying the user's own sentence then " Noted."; a question about supplied facts is answered by copying the best matching sentence; a known fact is answered with its value and a period; otherwise "I do not know that yet."
10. Memory and corrections: retrieval is recency-biased, so a later statement beats an earlier one with the same words (works within the 16k context; long-conversation recall is only partially tested).
11. Safety: this build never emits a tool call (TOOL token is never written), so it cannot delete, write or read outside the project. Tool use (calc/read/ls/grep) is NOT implemented in this version.
12. Knowledge is deliberately tiny (about 60 facts); everything else falls through to the honest "I do not know that yet."

## Where things live
- Language: lexicon `words` (294 words: function words, frame words, KB entities/relations/values); everything else is bytes.
- Facts: MLP 3 (hit detection) and MLP 6 (value emission) neurons, built from the `facts` list in build().
- Memory: attention layer 4 (retrieval) and layers 5-6 (copy).
- Safety: absence of any TOOL-emitting rule.

## Edit map
- Add a fact: append (entity, relation, value) to `facts` in build(); entity/relation/value words are added to the lexicon automatically; one neuron is added in MLP 3 and one in MLP 6.
- Add a relation word: add it to `rels` (alias list); its embedding row sets the relation register.
- Add a word: append to `words` (embedding row, unembed column and code are generated).
- Add a frame (social phrase): add a flag register (f_x/g_x), a neuron in MLP 3 that sets it, and `chain(...)` + an END neuron in MLP 6.
- Add a tool: needs a literal JSON chain in MLP 6 plus a digit/path start head (not done).

## Limits
Copying relies on 2-3 token context; sentences with repeated letter n-grams may loop or garble. Retrieval uses word signatures with a fixed threshold; multi-sentence turns use only the last sentence for the bag. No arithmetic, no tools, no reasoning, no instruction-format skills. Known failure: "My dog is called Biscotto." is echoed unreliably.

## Group sections
### embed
token embedding: 64-d seeded +-1 code (greedy lowest-coherence pick) in C, plus class flags F and word-feature regs. Why: codes give a shared token bus that attention copies and the unembedding reads.

### pos
learned position table: 8 sinusoid pairs (periods 3..23 coprime) + linear t/16. Why: relative-offset matching (prev token, latest boundary) by rotation of the pair code.

### unembed
reads output block O against token codes. Why: logit_i = O . code_i.

### b_unembed
END bias 30 (default when nothing writes O). Why: empty rules end the turn.

### layers.0.wq
attention wq:  [h0] previous-token head: query is position code rotated by -1 step, key is position code (8 sinusoid pairs, peak 8 vs 5 at neighbours); copies token code of t-1 into Vb [h1] latest sentence-boundary head (USER, BOT, . ? ! END, BOS): bnd flag + recency; copies that boundary position code into SPi [h2] latest word-separator head: copies position code of the nearest non-word-char token (<=t) into SPw. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.0.wk
attention wk:  [h0] previous-token head: query is position code rotated by -1 step, key is position code (8 sinusoid pairs, peak 8 vs 5 at neighbours); copies token code of t-1 into Vb [h1] latest sentence-boundary head (USER, BOT, . ? ! END, BOS): bnd flag + recency; copies that boundary position code into SPi [h2] latest word-separator head: copies position code of the nearest non-word-char token (<=t) into SPw. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.0.wv
attention wv:  [h0] previous-token head: query is position code rotated by -1 step, key is position code (8 sinusoid pairs, peak 8 vs 5 at neighbours); copies token code of t-1 into Vb [h1] latest sentence-boundary head (USER, BOT, . ? ! END, BOS): bnd flag + recency; copies that boundary position code into SPi [h2] latest word-separator head: copies position code of the nearest non-word-char token (<=t) into SPw. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.0.wo
attention wo:  [h0] previous-token head: query is position code rotated by -1 step, key is position code (8 sinusoid pairs, peak 8 vs 5 at neighbours); copies token code of t-1 into Vb [h1] latest sentence-boundary head (USER, BOT, . ? ! END, BOS): bnd flag + recency; copies that boundary position code into SPi [h2] latest word-separator head: copies position code of the nearest non-word-char token (<=t) into SPw. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.0.bq
attention bq:  [h0] previous-token head: query is position code rotated by -1 step, key is position code (8 sinusoid pairs, peak 8 vs 5 at neighbours); copies token code of t-1 into Vb [h1] latest sentence-boundary head (USER, BOT, . ? ! END, BOS): bnd flag + recency; copies that boundary position code into SPi [h2] latest word-separator head: copies position code of the nearest non-word-char token (<=t) into SPw. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.1.wq
attention wq:  [h0] previous-token carry: SPex(t)=SPi(t-1) (header of the sentence token t belongs to); Vb += 0.5*Vb(t-1) so Vb = code(t-1)+0.5code(t-2) [h1] word signature: average token code over the word-so-far (same SPw); invented names get a unique bag-of-letters signature; BOS sink gives 0 off-word. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.1.wk
attention wk:  [h0] previous-token carry: SPex(t)=SPi(t-1) (header of the sentence token t belongs to); Vb += 0.5*Vb(t-1) so Vb = code(t-1)+0.5code(t-2) [h1] word signature: average token code over the word-so-far (same SPw); invented names get a unique bag-of-letters signature; BOS sink gives 0 off-word. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.1.wv
attention wv:  [h0] previous-token carry: SPex(t)=SPi(t-1) (header of the sentence token t belongs to); Vb += 0.5*Vb(t-1) so Vb = code(t-1)+0.5code(t-2) [h1] word signature: average token code over the word-so-far (same SPw); invented names get a unique bag-of-letters signature; BOS sink gives 0 off-word. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.1.wo
attention wo:  [h0] previous-token carry: SPex(t)=SPi(t-1) (header of the sentence token t belongs to); Vb += 0.5*Vb(t-1) so Vb = code(t-1)+0.5code(t-2) [h1] word signature: average token code over the word-so-far (same SPw); invented names get a unique bag-of-letters signature; BOS sink gives 0 off-word. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.1.bq
attention bq:  [h0] previous-token carry: SPex(t)=SPi(t-1) (header of the sentence token t belongs to); Vb += 0.5*Vb(t-1) so Vb = code(t-1)+0.5code(t-2) [h1] word signature: average token code over the word-so-far (same SPw); invented names get a unique bag-of-letters signature; BOS sink gives 0 off-word. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.2.wq
attention wq:  [h1] second previous-token carry: Vb += 0.7*Q*Vb(t-1) extends the key context to three tokens (Q = signed permutation keeps orders distinguishable) [h0] previous-token carry of word signature to the separator after the word (full-word signature), SPq(t)=SPex(t-1), pcont flag. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.2.wk
attention wk:  [h1] second previous-token carry: Vb += 0.7*Q*Vb(t-1) extends the key context to three tokens (Q = signed permutation keeps orders distinguishable) [h0] previous-token carry of word signature to the separator after the word (full-word signature), SPq(t)=SPex(t-1), pcont flag. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.2.wv
attention wv:  [h1] second previous-token carry: Vb += 0.7*Q*Vb(t-1) extends the key context to three tokens (Q = signed permutation keeps orders distinguishable) [h0] previous-token carry of word signature to the separator after the word (full-word signature), SPq(t)=SPex(t-1), pcont flag. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.2.wo
attention wo:  [h1] second previous-token carry: Vb += 0.7*Q*Vb(t-1) extends the key context to three tokens (Q = signed permutation keeps orders distinguishable) [h0] previous-token carry of word signature to the separator after the word (full-word signature), SPq(t)=SPex(t-1), pcont flag. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.3.wq
attention wq:  [h0] question bag: at BOT average full-word signatures of content words in the last sentence (same header SPq) [h1] intent-flag gather: copies word-feature flags (?, greeting, thanks, you, intro, relation words) of the last sentence to BOT [h2] known-entity gather: copies code of KB entity word in the last sentence into Vb at BOT. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.3.wk
attention wk:  [h0] question bag: at BOT average full-word signatures of content words in the last sentence (same header SPq) [h1] intent-flag gather: copies word-feature flags (?, greeting, thanks, you, intro, relation words) of the last sentence to BOT [h2] known-entity gather: copies code of KB entity word in the last sentence into Vb at BOT. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.3.wv
attention wv:  [h0] question bag: at BOT average full-word signatures of content words in the last sentence (same header SPq) [h1] intent-flag gather: copies word-feature flags (?, greeting, thanks, you, intro, relation words) of the last sentence to BOT [h2] known-entity gather: copies code of KB entity word in the last sentence into Vb at BOT. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.3.wo
attention wo:  [h0] question bag: at BOT average full-word signatures of content words in the last sentence (same header SPq) [h1] intent-flag gather: copies word-feature flags (?, greeting, thanks, you, intro, relation words) of the last sentence to BOT [h2] known-entity gather: copies code of KB entity word in the last sentence into Vb at BOT. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.3.bq
attention bq:  [h0] question bag: at BOT average full-word signatures of content words in the last sentence (same header SPq) [h1] intent-flag gather: copies word-feature flags (?, greeting, thanks, you, intro, relation words) of the last sentence to BOT [h2] known-entity gather: copies code of KB entity word in the last sentence into Vb at BOT. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.3.w_in
MLP rules (neurons: AND of token/flag/register conditions -> writes frame flags or output token codes): about greet kb thank. Why: relu(sum of cond - (n-0.5)) fires only when all conditions hold.

### layers.3.w_out
MLP rules (neurons: AND of token/flag/register conditions -> writes frame flags or output token codes): about greet kb thank. Why: relu(sum of cond - (n-0.5)) fires only when all conditions hold.

### layers.3.b_in
MLP rules (neurons: AND of token/flag/register conditions -> writes frame flags or output token codes): about greet kb thank. Why: relu(sum of cond - (n-0.5)) fires only when all conditions hold.

### layers.4.wq
attention wq:  [h0] retrieval: question bag vs full-word signatures at earlier separators (own sentence excluded); value = header code of the matching sentence; BOS sink (threshold 64) = no match [h1] fetch fast-frame flags from the latest BOT to every reply position. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.4.wk
attention wk:  [h0] retrieval: question bag vs full-word signatures at earlier separators (own sentence excluded); value = header code of the matching sentence; BOS sink (threshold 64) = no match [h1] fetch fast-frame flags from the latest BOT to every reply position. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.4.wv
attention wv:  [h0] retrieval: question bag vs full-word signatures at earlier separators (own sentence excluded); value = header code of the matching sentence; BOS sink (threshold 64) = no match [h1] fetch fast-frame flags from the latest BOT to every reply position. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.4.wo
attention wo:  [h0] retrieval: question bag vs full-word signatures at earlier separators (own sentence excluded); value = header code of the matching sentence; BOS sink (threshold 64) = no match [h1] fetch fast-frame flags from the latest BOT to every reply position. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.4.bq
attention bq:  [h0] retrieval: question bag vs full-word signatures at earlier separators (own sentence excluded); value = header code of the matching sentence; BOS sink (threshold 64) = no match [h1] fetch fast-frame flags from the latest BOT to every reply position. Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.4.w_in
MLP rules (neurons: AND of token/flag/register conditions -> writes frame flags or output token codes): ack qa unk. Why: relu(sum of cond - (n-0.5)) fires only when all conditions hold.

### layers.4.w_out
MLP rules (neurons: AND of token/flag/register conditions -> writes frame flags or output token codes): ack qa unk. Why: relu(sum of cond - (n-0.5)) fires only when all conditions hold.

### layers.4.b_in
MLP rules (neurons: AND of token/flag/register conditions -> writes frame flags or output token codes): ack qa unk. Why: relu(sum of cond - (n-0.5)) fires only when all conditions hold.

### layers.5.wq
attention wq:  [h0] copy-start (QA): query = RC position code advanced one step, key = position code; copies first token of retrieved sentence into O [h1] fetch slow-frame flags, retrieved header RC and last-sentence header SPq from the latest BOT [h2] copy-start (ACK): same, but starting at the user's own last sentence (SPq). Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.5.wk
attention wk:  [h0] copy-start (QA): query = RC position code advanced one step, key = position code; copies first token of retrieved sentence into O [h1] fetch slow-frame flags, retrieved header RC and last-sentence header SPq from the latest BOT [h2] copy-start (ACK): same, but starting at the user's own last sentence (SPq). Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.5.wv
attention wv:  [h0] copy-start (QA): query = RC position code advanced one step, key = position code; copies first token of retrieved sentence into O [h1] fetch slow-frame flags, retrieved header RC and last-sentence header SPq from the latest BOT [h2] copy-start (ACK): same, but starting at the user's own last sentence (SPq). Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.5.wo
attention wo:  [h0] copy-start (QA): query = RC position code advanced one step, key = position code; copies first token of retrieved sentence into O [h1] fetch slow-frame flags, retrieved header RC and last-sentence header SPq from the latest BOT [h2] copy-start (ACK): same, but starting at the user's own last sentence (SPq). Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.5.bq
attention bq:  [h0] copy-start (QA): query = RC position code advanced one step, key = position code; copies first token of retrieved sentence into O [h1] fetch slow-frame flags, retrieved header RC and last-sentence header SPq from the latest BOT [h2] copy-start (ACK): same, but starting at the user's own last sentence (SPq). Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.6.wq
attention wq:  [h0] induction copy within the retrieved sentence: key = previous token(s) equals current token (Vb) inside region RCt; BOS sink = end of sentence (sets stop) [h1] same induction copy within the user's own last sentence (ACK). Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.6.wk
attention wk:  [h0] induction copy within the retrieved sentence: key = previous token(s) equals current token (Vb) inside region RCt; BOS sink = end of sentence (sets stop) [h1] same induction copy within the user's own last sentence (ACK). Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.6.wv
attention wv:  [h0] induction copy within the retrieved sentence: key = previous token(s) equals current token (Vb) inside region RCt; BOS sink = end of sentence (sets stop) [h1] same induction copy within the user's own last sentence (ACK). Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.6.wo
attention wo:  [h0] induction copy within the retrieved sentence: key = previous token(s) equals current token (Vb) inside region RCt; BOS sink = end of sentence (sets stop) [h1] same induction copy within the user's own last sentence (ACK). Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.6.bq
attention bq:  [h0] induction copy within the retrieved sentence: key = previous token(s) equals current token (Vb) inside region RCt; BOS sink = end of sentence (sets stop) [h1] same induction copy within the user's own last sentence (ACK). Why: gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits.

### layers.6.w_in
MLP rules (neurons: AND of token/flag/register conditions -> writes frame flags or output token codes): ack-note end kbdot kbval lit. Why: relu(sum of cond - (n-0.5)) fires only when all conditions hold.

### layers.6.w_out
MLP rules (neurons: AND of token/flag/register conditions -> writes frame flags or output token codes): ack-note end kbdot kbval lit. Why: relu(sum of cond - (n-0.5)) fires only when all conditions hold.

### layers.6.b_in
MLP rules (neurons: AND of token/flag/register conditions -> writes frame flags or output token codes): ack-note end kbdot kbval lit. Why: relu(sum of cond - (n-0.5)) fires only when all conditions hold.

