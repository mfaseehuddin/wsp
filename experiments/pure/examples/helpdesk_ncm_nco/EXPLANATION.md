# nco: a two-level hand-written Next-Concept Model

K = 2 forward passes. Levels (n=1, n=48), codebooks (56, V=3378). D=32, d_model=128, 2 layers, 4 heads, d_head=32,
d_mlp=311, linear decoder. 1.39M params, about 223k nonzero. Dev exact: 224/224.

## Level-by-level story
- **Level 0, n=1: the plan.** A single code c = 4*intent + 2*has_name + has_ticket (56 codes). `code0[c]` writes
  a one-hot intent (latent dims 16..29), has_name (30) and has_ticket (31). Nothing goes into the token dims
  (0..15), so the reply decoded after level 0 is empty (`probe`: `decoded_so_far: ""`). The reply's layout is
  fully fixed at this point: its length, where the greeting and tail go, and which canned sentence it uses.
- **Level 1, n=48: the tokens.** Each of the 48 rows reads the plan (via `w_in`) and its own position t
  (`lvl_pos`). It predicts the exact token for its slot: a canned word, the copied name, a copied ticket digit,
  or `<eos>`. Code = token id (codebook = vocab). `code1[v] = E[v]`, so after level 1 the latent's token
  dims hold E[token], and the decoder reads it back. After level 1 the reply is complete.

So level 0 settles *what* the reply is, and level 1 settles *which token goes where*. A middle level would add
nothing, because every position-dependent decision is an exact function of (plan, t) plus two copies.

## Weight groups
**Token codes** (`tok_emb[:,0:16]`, `head1`, `head_b1`, `code1`, `dec_w2`, `dec_b2`). Each token has a fixed
unit vector E[v] in 16 dims (seeded Gaussian, normalised; `<pad>` and `<eos>` get 0). Reading a token is a dot
product with E. For a clean E[t], token t scores 1 and every other token scores < 1 (max cosine 0.89), so the
argmax is exact. `<eos>` has code 0 and wins through a bias of 0.3 whenever nothing has been written.
The same E is used as the input identity (`tok_emb`, residual 0..15), the output subspace (residual 32..47,
read by `head1`), the level-1 codebook and the decoder.

**Prefix features in `tok_emb`** (hand tables in weights.py):
- KWD/VOT: intent keywords. Each has an attention strength (1 topic, 0.75 ambiguous, 0.5 greeting, 0.55
  thanks) and a vote vector normalised to sum 1. Ambiguous words split their vote (e.g. connect:
  vpn/wifi/monitor).
- NS (name score) gets +1 for a name lexicon and -3 for function words, intent keywords, cue words, digits
  and `<unk>`.
- DIG (digit flag) also seeds DS0..3.
- INTRO, INTRO2, FOLLOW, COM, GRT are cue features that neighbours read.

**`pre_pos`.** P1=i+1, P2=(i+1)^2, a prefix flag, and NS += 0.6(i+2), which is the end-of-message cue (see N1).
**`lvl_emb`/`lvl_pos`.** These hold a level flag (L0/L1), LKEY=1 (marks the row as a sink/level row) and t.

**Layer-0 heads: positional (prev1, prev2, prev3, next1).** q=(2c·target, -c), k=(p, p^2), so the score is
-c(p - target)^2 + const and peaks at the wanted neighbour, with c=8 sharp. The level rows sit at p=0 and act
as "before start". A missing previous token therefore reads the level rows, whose features are zero. Level rows
attend only to level rows (q = 20·Lflag on LKEY), so their feature dims stay 0. The values and outputs build
two linear evidence sums:
- NS(i) = lex - penalty + INTRO(prev1) [i'm, am 1, me 1, is .4, hi .5] + INTRO2(prev2) [this .8, it .6]
  + FOLLOW(next) [here/speaking 1.8, . , ! from .8] + 0.6·end. The end cue: N1 copies the next token's
  position. That cancels the pre_pos term unless next1 fell back to self, which happens only at the last token.
- DS_k(i) = d_i + d_{i-1..i-k} - d_{i-k-1} (and DS_3 = d_i+d_{i-1}+d_{i-2}+d_{i-3}). It equals k+1 exactly at
  the k-th digit of a 4-digit run and is <= k everywhere else. DS_3 = 4 means a 4-digit ticket number is
  present.
- N1 also copies "next token is a comma" (CNX).

**Layer-0 MLP: the layout bank.** This is the key trick. For a token sequence s_0..s_{L-1} and an integer
offset x, the piecewise-linear interpolant Σ_m ReLU(x-m)·(E[s_{m+1}] - 2E[s_m] + E[s_{m-1}]) for
m = -1..L equals E[s_x] exactly at integer x, and 0 outside 0..L-1. So L+2 units are an exact lookup table.
- 14 core banks, each with offset x = t - 3·has_name, gated by -100·(1 - intent_j).
- A greeting bank ("hi", name-slot, ",") with x = t, gated by has_name. Its middle slot emits the
  name-copy query QN instead of a token.
- A tail bank ("i added this to ticket d d d d .") with x = t - 3·has_name - len(core), where len(core) comes
  linearly from the intent one-hot. It is gated by has_ticket, and its digit slots emit queries QK_k.
- One AND unit, ReLU(greet + next_is_comma - 1). It moves a "hi ,"-style greeting vote to a separate HIC dim.

Prefix rows and the level-0 row have zero plan dims, so every bank stays off for them.

**Layer-1 heads.**
- Head 0, copy (level-1 rows): q = 20·QN on NS + 30·QK_k on DS_k + 12 on LKEY (the sink). A copy row lands
  on the max-NS token or the k-th digit and moves its identity E (dims 0..15) into the output dims. Rows
  that copy nothing attend the level rows, whose identity is 0.
- Heads 1 and 2, name and ticket flags (level-0 row): score 30·NS against a sink at 30·1.5, and 30·DS_3
  against a sink at 30·3.5. The value is the prefix flag, so NF and TF give the attention mass on the
  prefix, which is about 0 or 1.
- Head 3, intent (level-0 row): key = keyword strength, sink at 0.25, scale 24. The strongest tier dominates
  exponentially and the row averages that tier's votes.

**`head0`/`head_b0`.** logit(i,n,k) = vote_i + n(NF - .5) + k(TF - .5) + 0.05·[unknown]
+ 0.3·HIC·[hello, n=0]. Argmax over the product set factorises into three independent decisions.
- With no keyword, all votes are 0 and unknown wins.
- A "hi ," greeting counts as hello only when no name follows. For example "hi , i am X . <off-topic>" gets
  unknown, but "hello , can someone help" gets hello.

## Generalisation
Intent uses about 200 keywords and synonyms from my own knowledge, checked one word at a time with
`tokenize` (words not in the vocab are skipped automatically). Names are found by context cues plus a name
lexicon. A name needs NS >= 1.5: one strong cue, or a lexicon hit plus one cue. Tickets are detected as any
4-digit run, regardless of template ("ticket number 1234", "ticket # 5678", "update on ticket 4321 :" all
work in local tests). All positions are generated by exact arithmetic, so a new name or ticket number can't
break the layout.

## Known failure modes
- Names that tokenize to `<unk>` can't be copied. Several common names I tried (carlos, kim) are `<unk>`.
- An in-vocab name outside the lexicon needs a context cue worth 1.5 or more. "this is X ," and "i'm X ."
  have one; "X from finance" alone doesn't.
- An unlisted ordinary word in a strong name context can be taken as a name, e.g. "this is <adjective> ,".
  I mitigated this with a large penalty list.
- Intent phrasings with no listed keyword fall to unknown. Mixed topics go to the majority keyword
  ("vpn connection is very slow" gives vpn).
- A query with two separate 4-digit numbers would make the copy ambiguous.

## Path
I read the spec and dev, then designed (plan, tokens) as the minimal coarse-to-fine split. I chose a
vocab-sized level-1 codebook so copies are trivial. The piecewise-linear ReLU bank replaced a
shift-attention plan: offsets are linear in t and the plan, so no attention is needed for layout. Dev was
100% on the first eval. After that I tested only my own phrasings with `model.generate` and fixed these
issues:
- a greeting vote overriding unknown;
- the next1 head falling back to self at the message end ("grace speaking" copied "speaking");
- adjectives taken as names.

Finally I halved E (32 to 16 dims) to halve the nonzeros.
