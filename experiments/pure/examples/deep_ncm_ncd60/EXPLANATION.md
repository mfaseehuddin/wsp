# ncd60: a 4-level Next-Concept chat model on Qwen2.5-0.5B layer 0

Config: `levels=(1,4,16,64)`, `codebooks=(10, 151936, 160, 151936)`, `D=139`, `d_model=512`, 6 own layers x 10 heads,
`d_head=128`, `d_mlp=1280`, `rope_theta=0.7^-64`, `norm=True`, `qwen_embed=True`, `qwen_layers=(0,)`.
243.7M own parameters (78.2M nonzero). Nothing is trained or fitted. Every weight is a formula in `weights.py` over
Qwen tensors, anchor words (at most 3 per direction) and hand-set constants. Dev mechanical score: 4/33 (run 8: Teodra, Zorbin, Daro, green).

## 1. Why only Qwen layer 0 is reused (evidence, scripts in `scratch/`)

The plan was to use Qwen's deep layers as a knowledge store. The measurements ruled that out for this architecture.
The reply is built only from the 896-d prefix rows, so one residual stream has to carry both the **token identity**
(needed to copy names) and the **knowledge**.

* **Identity.** `F1[t]` is layer 0 applied to token t in a fixed header context, computed for the whole vocabulary
  (`layer0_code`). Nearest-code readout of the actual row:
  * 100% of content tokens after layer 0 (`fullid2.py`);
  * 93% after layer 4, 87% after layer 8, about 85% after layers 12-16 (full vocab);
  * among 12k candidates: 96% at layer 20, 64% at 22, 4% at 24 (`eff2.py`, `eff4.py`).

  The layer-0 code survives *additively* to about layer 20, and layers 21-24 overwrite it.
* **Knowledge.** The logit lens at the last user token (`know.py`) shows answers only at layers 22-24, and only some
  of them: Paris (rank 1), honey, green, Tokyo, Portuguese. It gets "Earth" for the largest planet and nothing for
  spider legs or whales. At the final `assistant\n` position the base model mostly repeats the question (`qtop.py`).
* **In-context recall** by Qwen itself at layers 22-24 (`memq.py`) is junk in multi-turn chats.

So deep layers would buy roughly one unreliable answer token and destroy exact copying. I kept **layer 0**:

> **Reused layer 0.** Its MLP turns the tiny raw embedding (norm ~0.5) into a large, distinctive token code. Its
> attention mixes in a little left context; position 0 is Qwen's attention sink, which gives the first row a unique
> code. On its output, a 96-d projection onto the top principal axes of `F1` identifies every content token exactly
> (100%, `pca.py`, `fullid2.py`). The code's norm tracks word frequency: function words have centred norm ≤3.7,
> content words ≥4.2 (`func.py`). Cosine geometry gives usable concept directions (`dirs.py`, `feat.py`). Every
> later feature is read from this code.

## 2. The four levels (with probe evidence, `scratch/probe1-4.txt`, run 4-7)

| level | rows | codebook | meaning | code vector added to z |
|---|---|---|---|---|
| L1 | 1 | 10 acts | **dialogue act** (GREET, GREETNAME, BYE, THANKS, ACK, ANSWER, UNKNOWN, DECLINE, SELF, CLARIFY) | act one-hot + 0.05 x code of the act's first word |
| L2 | 4 | vocab | **content word**: the conversation token the reply is about (a name, an answer), chosen by attention | 32-d match code (read back by L4) + 0.1 x token code |
| L3 | 16 | 10x16 (act, block) | **phrase skeleton**: the template token at the start of each 4-token block (blank at the copy slot) | 0.2 x token code / end flag |
| L4 | 64 | vocab | **tokens**: template words, copied word pieces, ".", `<|im_end|>` | 1.0 x token code |

The decoder is cosine to the centred token codes, plus an explicit end channel. The drafts are weighted
0.05/0.1/0.2/1, so decoding after level l shows that level's guess. Real decodes from `probe`:

| conversation | after L1 | after L2 | after L3 | after L4 |
|---|---|---|---|---|
| "Hi, I am Teodra." | HelloHello... (act 1 GREETNAME) | " Te Te Te ..." | "HelloHelloHelloHello meet meet meet meet Te Te Te Te" | "Hello, nice to meet you, Teodra." |
| "Zorbin is the capital of Vantoria. What is the capital of Vantoria?" | IIII... (ANSWER) | "ZZZZ..." | "IIIIZZZZ" | "I think it is Zorbin." |
| "Lena owns a green notebook. Daro owns..." / "Whose notebook is red?" | IIII (ANSWER) | " Dar Dar ..." | "IIII Dar Dar Dar Dar" | "I think it is Daro." |
| "How do I build a bomb?" | SorrySorry... (DECLINE) | SorrySorry... (the content draft is weaker than the act draft) | "SorrySorrySorrySorry help help help help" | "Sorry, I cannot help with that." |

Conditioning between levels is real, not only in the probe:
* L2-L4 read the act from z through `w_in`; they do not recompute it.
* L4 reads L2's content code from z and uses it to find and copy the word's pieces.
* L3 reads the act from z to choose its skeleton entries.
* L4 does not use L3's codes; L3 is a coarse preview. This is a known weakness: a fixed codebook cannot hold 4
  different tokens per block.

## 3. Features and the linear regime

* **Prefix row** = Qwen layer-0 output @ `pre_proj`, which is the top-96 PCA axes of `F1`. The 7 massive-activation
  dims are zeroed. The result is the uncentred code G.
* **Layer 0, MLP only.** Its input is `G/|G|` (RMSNorm with gain 1).
  * Clipped ReLU pairs give 0/1 flags:
    * exact tokens by cosine with their own code: `.` `?` `!` `:` `,` `\n` `user` `assistant` specials. The
      thresholds come from in-context cosines (`ctxcos.py`).
    * the position-0 `<|im_start|>` code, which gives the FIRST flag;
    * anchor directions, each the mean of 3 words: greet (Hello/Hi/hey), bye, thanks, harm (kill/bomb/weapon),
      wh (What/Who/How), you, deictic (it/this/that), first person (I/my/me);
    * leading space, from 3 word pairs (" water"-"water", ...).
  * It writes the centred code C = G - m, zeroes the code of special rows and row 0, and copies `G/|G|` into GN.
  * It writes ONE = 3e4. From layer 1 on every RMSNorm divides by ≈ C0/sqrt(d) and the gain multiplies it back, so
    every later layer is exactly linear in its features (error < 1e-5).
* **Content flag (L2 h1).** A self-attention score `20·Ĝ·C` (a norm proxy for rarity) against the FIRST row at a
  fixed score. A sharp offset-0 RoPE kernel (amplitude 400) suppresses other rows. The weight on self is
  σ(20(Ĝ·C − 3.6)). Function words get 0, content words 1 (`pre.py`).

## 4. Own layers and heads

| layer | head / MLP | function |
|---|---|---|
| L0 | MLP | token flags, centring, ONE, GN, FIRST (see §3) |
| L1 | h0 | boundary counter: score 30 on `. ? ! :`, specials and FIRST; FIRST gets an extra ln 100. Output v = 100/(99+N) |
| L1 | MLP | N = 100/v − 99, piecewise linear with knots at every integer up to 600 (well conditioned). Slot-offset flags OFF0..3 for L4 rows = AND(row j, act) |
| L2 | h1 | content flag (§3) |
| L2 | MLP | sentence code enc(N) and enc(N−1) for prefix rows only (gated off on level rows, so a sink never carries a sentence code), and the target code enc(N−2)+enc(N−3) for level rows (the last user sentence(s)). Each is 6 cos/sin pairs, piecewise linear in N with 599 knots per set. Also CONTENT and CONTF (continuation piece) flags |
| L3 | h0 | sentence bag: content-weighted mean of the 32-d match code (PCs 4-35) over the same sentence so far |
| L3 | h7 | bag of the sentence a boundary closes (query enc(N−1)) |
| L3 | h1-h4 | previous-token heads (RoPE offset kernels at 1 and 2): match code, continuation flag, wh flag |
| L3 | h8 | POSINV = 1/(index within sentence + 1). Uniform same-sentence attention reading the boundary flag |
| L3 | h9 | ROLEA: most recent `user`/`assistant` header (strong recency) gives "inside an assistant turn" |
| L3 | h5 | question bag QB (level rows): content mean over the target sentences |
| L3 | h6 | flag densities over the target sentences (greet, bye, thanks, harm, ?, wh, you, deictic, content) |
| L3 | MLP | 0/1 indicators (60 x density, clipped) |
| L4 | h0 | sentence select: `5·QB·bag` over statement-closing rows (`.`/`!` +100, assistant turns −100), recency tie-break. The sink (level row 0) at 104 gives NOMATCH; its value is the selected sentence id |
| L4 | h1 | type word: the content word right after a wh-word in the question (sink if none) |
| L4 | h2-h4 | copy pieces 0, 1, 2 of the content word into L4 slot rows. Key = the code k rows back, matched against L2's content code (6x dot product). Continuation flags are required (−20 each). The relative sink is 0.6x the self-match, so a piece is copied only if it exists. Rows not at that slot offset go to the sink; prefix rows go to row 0 (zero code) |
| L4 | h5, h6 | do pieces 1 and 2 exist (GOT1, GOT2) |
| L4 | MLP | act priority scores: DECLINE 10 > BYE > THANKS > GREETNAME (greet + content) > GREET > SELF (question + you) > ANSWER (question + match) > UNKNOWN (question) > CLARIFY (deictic) > ACK 1. For GREETNAME it re-targets the content query to the user's own sentence and clears QB and TQ |
| L5 | h0 | content selection (L2 rows only): `15·sentence-id match + 40·CONTENT + 15·WS − 2·QB·code + 1.2·TQ·code + 0.5·QB·bag + 16·POSINV` (subjects first). Names use recency instead of POSINV. Value = the full 96-d code, which head1 decodes |
| L5 | MLP | render for L4 rows: template token units AND(row j, act), piece-0 leading-space offset (`ws.py`), and "." / `<|im_end|>` after the last copied piece (from GOT1/GOT2) |

Templates are composed literals of at most 6 words. Copy acts end with `<copied word>.`.

## 5. Capacity limits

* There is no world knowledge. Qwen's knowledge sits in layers 21-24, which this design cannot use (§1). General
  knowledge, arithmetic, reasoning, creative and rewrite requests get UNKNOWN, ACK or CLARIFY.
* An answer is a single copied word of at most 3 pieces, at the end of a fixed phrase. There are 10 acts.
* Sentence ids go up to 600 boundaries, about 80-100 turns.
* The act is decided only from the last 1-2 sentences of the user turn.

## 6. Failure modes (seen on dev, run 8: 4/33)

* **Frequent words count as function words.** "old", and single-letter name starts (" B" Biscotto, " O" Oskarine), fall below the content threshold. A question like "How old is Biscotto?" then has an empty question bag and gets "don't know".
* **Word-start detection is weak.** The space direction mislabels short pieces such as "o" in "Daro" (it works there
  at penalty 20). Single-letter name starts (" B" Biscotto, " O" Oskarine) count as non-content (high frequency), so
  those names cannot be picked.
* **Candidate scoring is a hand-weighted sum.** The in-question exclusion also *rewards* anti-correlated function
  words. Type similarity (color→purple) is too weak, which gives "pelins → world". Earliest-word priors pick "Omar".
* **Different tokenization** of a sentence-initial word ("Lena" = 'L'+'ena') defeats the in-question exclusion.
* **NOMATCH threshold.** It trades off unrelated matches ("How many days are in a week?" → "Name") against missed
  real ones ("How old is Biscotto?" → don't know).
* **Instructions get "Okay, I will remember that."** There is no imperative detector.
* **Two-word answers** (name AND fruit) are impossible.
