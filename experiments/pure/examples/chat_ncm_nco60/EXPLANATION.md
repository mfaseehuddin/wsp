# nco60: a hand-written Next-Concept chat model

Config: `D=75` (64 identity + 11 mode dims), `levels=(1, 64)`, `codebooks=(11, 151936)`, `d_model=784`,
6 layers x 7 heads, `d_head=192`, `d_mlp=1280`, `rope_theta=10^19.2`, `norm=False`. About 298M parameters, 57M of them nonzero.
Nothing is trained or fitted. Every number comes from a formula in `weights.py`.

## 1. What each level means (the coarse-to-fine split)

* **Level 0 (1 row, 11 codes): the speech act.** It picks one of ACK, GREET, GREETNAME, BYE, THANKS, DECLINE, ANSWER,
  UNKNOWN, SELF, REQUEST, CLARIFY. The code vector is a one-hot in latent dims 64..74. `w_in` copies it into the mode
  one-hot of every level-1 row. Decoding the latent after level 0 gives `<|im_end|>` everywhere, because the token
  dims are still empty and `dec_b2[im_end]` sits 0.05 above the rest. So it is an empty reply, and the level holds
  intent only. The probe (run 8) shows this: level 0 picked code 6 (ANSWER) and level 1 wrote " It is named Pumpernickel." 
* **Level 1 (64 rows, vocab codes): the words.** Row j emits token j of the reply. The reply has three parts:
  * a short template piece for the mode (at most 6 words per literal);
  * for copy modes (ANSWER, GREETNAME, REQUEST, ACK), a span copied from the conversation, then a `.`. ACK copies only
    if the turn has a first-person word ("Got it, I will remember Biscotto."). Otherwise it ends "... remember that.";
  * then `<|im_end|>`.

  The token code is a 64-d random identity vector `r_v` (unit norm, each half normalised separately). `head1` scores
  `20 * r_v . OUT`, and the decoder `dec_w2 = r^T`, `dec_b2 = -1/2` is the exact nearest-code inverse.

## 2. Token embedding: which Qwen structure is used, and why it carries meaning

All of it comes from Qwen2.5-0.5B's tied embedding E. Write C for E normalised and then centred.

* **SEMW (48-d):** PCA components 2..49 of C, normalised and multiplied by a content weight w. Cosine in this space
  reflects distributional similarity. PC1 is dropped because it is a word-frequency axis.
* **Content weight w, and two stop notions.**
  * There is a function-word axis `d = mean(the, and, of) - mean(" B", " K", " M")`, with 3+3 anchors. It is used as
    a projection `fs` together with PC1, and it separates function words (fs about 0.2-0.5) from content words and
    name pieces (fs <= 0.06).
  * `w = clip((0.21 - fs - 0.3*pc1)/0.08)`.
  * STOP (broad) marks function-like tokens. It defines run starts and keeps function words from becoming answers.
  * STC (strict) marks punctuation, clear function words and clitics. Only STC ends a copied span, so subword pieces
    such as "al", "el" and "o" stay inside names.
* **Flags from cosine similarity to named concept directions** (at most 3 anchors each, thresholded):
  * greeting (Hello / hi / hey), farewell (bye / goodbye / farewell), thanks (thanks / thank / Thanks);
  * harm (kill / bomb / weapon), second person (you / your / yourself), imperative (Write / Tell / Give);
  * deictic (it / this / that), first person (my / My / I);
  * punctuation (. , ?), sentence boundary (. ? !), and question (the "?" minus "." direction).

  Coverage of other words comes from the geometry, not from lists. Tokens near the pronoun / wh-word anchors are
  forced to STOP (never answers); tokens near the digit anchors (4 / 1 / 7, cos > 0.8) are content and continue spans.
* **ID (64-d):** a random identity code, used for exact copying and matching. It is a mechanism, not meaning.
* **Format tokens** (`<|im_start|>`, `<|im_end|>`, "user", "assistant", "\n") are flagged from the chat format.

## 3. Level-row inputs

`lvl_emb` sets the ONE and LVL flags. `lvl_pos[0]` sets SINK, so row 0 is the null key that every gated head can
fall back on. `lvl_pos[j]` (j < 16) sets a row one-hot J_j. Level rows carry no RoPE position in their keys,
because every `pos()` kernel is cancelled by `wk[LVL]`. So the sink bias is an absolute threshold.

## 4. Positional mechanisms (RoPE pairs 0..20 only)

The content dims use pairs 24..95, which are nearly rotation-free (omega <= 1.6e-5). The positional pairs implement
kernels `amp * sum_p cos((delta - s) * w_p)`:

* **Exact offset (amp 20, all pairs):** a peak at delta = s, with a gap of about 19 nats to the neighbours. Used for
  previous-token heads.
* **Context bag:** centred at delta = 2, width about 2.
* **Monotone recency (pairs 5..20):** decreasing for delta < 30.

## 5. Layers and heads

| layer | head | what it does |
|---|---|---|
| L0 | h0 | uniform average of BND: token 0 gets exactly 1, so MLP0 marks FIRST |
| L0 | h1-h4 | previous-token heads at delta = 1..4. They copy ID[:32] into P1..P4 and the strict-stop flags into SP1..4 (h1 also copies the broad stop into SB1) |
| L0 | h5 | left-context bags: CTX (semantic) and CTXR (identity, content-weighted) |
| MLP0 | | FIRST = relu(4*UNI-3); hop validity G_h = relu(1 - STC - SP1 - ... - SP_{h-1}); run start RS = SB1 |
| L1 | h0 | turn counter: attends to "user" tokens plus FIRST (with an extra ln 20) and reads FIRST, giving CT = 20/(20+N_user) |
| L1 | h1 | sentence counter: CS = 400/(400+N_bnd-1) |
| L1 | h2 | run-start identity RSID: the most recent token with RS=1 (sharp recency) |
| MLP1 | | 1.3k-knot piecewise-linear decoders. CT becomes enc(N) = (cos, sin) of N*{1, .13}. CS becomes the sentence index mod 17 as a one-hot (MOD), plus for level rows the target set {N-2, N-3} (TG). That target set is exactly the last sentence of the user turn, whether or not it ended with punctuation |
| L2 | h0 | question bag: level rows attend uniformly to tokens with MOD in TG and the same turn. They read SEMW into QS and IDW into QR |
| L2 | h1-h5 | presence detectors (greet, bye, thanks, harm, "?") over the last user turn. A same-turn score of 100*enc.enc, plus 30*flag, against a sink at 7. The output is the mass on prefix tokens |
| L2 | h6 | role: the most recent same-turn header. "assistant" outranks "user", which gives RA (in-reply flag) |
| L3 | h0 | fact match: earlier tokens that match the question semantically (QS.SEMW) and exactly (QR.ID). It excludes the question sentence (TG.MOD). It reads CTXR into RB (the left context of the matched words) and gives presence PM |
| L3 | h1 | name presence PN: a content run start in the last turn that is not a greeting |
| L3 | h2-h5 | presence of "you", imperative, deictic and first person |
| MLP3 | | (mode, row) gives hop-select HS_h and dot-select DS_h. In topic modes it zeroes QS, QR and TG, so the user's own words are not excluded |
| L4 | h0 | answer candidate. Score = 300*QS.CTX (left context matches the question) - 200*QR.ID (not a question word) + 15*RB.ID (right of a matched word) - 100*TG.MOD (not in the question sentence) + 5*W + 2*RS - penalties (stop, in-reply, role token) + recency, against a sink at 44. In GREETNAME/REQUEST mode the query adds W, RS, -greeting and strong recency, so it picks the newest content run of the last turn. Value: RSID (the run start of the chosen token) into A, and prefix mass into PC |
| MLP4 | | PCB = clip(5*(PC-0.6)), the binarised copy validity. Forced to 0 in ACK mode when there is no first person |
| L5 | h0-h3 | span hops: the token whose P_h equals A[:32] and whose G_h = 1. It reads ID into A_h and the valid flag into V_h, with the sink at 95 |
| MLP5 | | emission into OUT: the template token for (mode, row); A_h gated by HS_h and V_h (ReLU pairs with big offsets give exact pass-through); `.` after the last valid hop. Hops and dots are also gated by PCB. Empty rows fall to `<|im_end|>` through `head_b1` |

**Mode head (`head0`)** is a linear combination of the presence features. Examples:

* ANSWER = PQ + .5 PCB + .5 PM - 1.5
* UNKNOWN = PQ - .7
* SELF = PQ + PY - .5 PM - 1
* GREETNAME = 1.5 PG + .6 PN - PQ - 1
* CLARIFY = PD - PF - PQ - .5 PI
* REQUEST = PI - PQ - .3
* DECLINE = 3 PH - 1.2
* ACK = .25 (the fallback)

Other geometry-derived stop rules: tokens near the pronoun and wh-word anchors (you/your/yourself, my/My/I,
it/this/that, What/Who/Where) are STOP, so they are never answers. Tokens near the digit anchors (4/1/7, cos > 0.8)
count as content and continue spans.

## 6. Capacity limits

* There is no world knowledge, arithmetic, list manipulation or text generation beyond templates and copying. Qwen's
  embedding geometry alone does not give factual recall (the nearest neighbours of "capital + France" are spelling
  variants).
* There is one answer span per reply. Questions that ask for two things ("name and fruit") get one.
* The span is at most 5 tokens. It runs from the run start of the chosen token to the next strict stop, so it can
  over-run into a following content word ("Daro owns", "named Pumpernickel").
* Bag-of-words matching with 48-d semantics and 64-d random identities has cross-talk of about 5-10 nats, so
  thresholds trade false answers against false "I don't know".
* The sentence index is taken mod 17. Very old sentences can alias into the question bag. A turn match suppresses
  most of this.

## 7. Failure modes (observed locally)

* **Recency:** "pear" can beat "mango" after a correction, because recency is a mild additive term.
* **Answers that precede the matched words** ("Zorbin is the capital of Vantoria") are weak. The copy then picks a
  question word that leaked past the identity penalty.
* **Comparisons and ordering** ("Who is tallest?") are not reasoned. Usually the reply is "I don't know" or a name.
* **Imperatives** not near Write/Tell/Give fall into ACK or CLARIFY. Harm detection covers only words near
  kill/bomb/weapon.
* **The SELF answer is fixed.** "What should I call you?" gets "I'm a simple chat program..."
* **Repeated digits** ("4417") confuse the identity-based hops, because the same identity appears twice in the
  span, so numbers come out garbled.
* **Instructions not near Write/Tell/Give** get "Got it, I will remember that." or "Could you say more about that?"

Dev mechanical score: 7/33 (run 9). The passes are the name greeting, two topic echoes (fox/Quillo, report), the
pelins colour and time, Daro, and Biscotto's age.
