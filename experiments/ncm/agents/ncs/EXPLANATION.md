# ncs: a two-level NCM, plan then tokens

## Levels (K=2)
- **Level 0 (n=1, 56 codes): the plan.** One code = (intent among 14) x (name present?) x (ticket digits present?).
  After this level the latent says what the reply is, but every position holds the same plan vector, so the decoded text is still all `<pad>`.
- **Level 1 (n=48, 189 codes): the tokens.** Each position picks its token code; the codebook is the reply alphabet
  (fixed reply words, 10 digits, ~55 names, `<eos>`). After this level the decoder reads out the full reply.
  Each position's code is "which token goes here", so after level 1 the reply is complete; before it, the reply is only the plan.

## Weight groups
- **Token codes** (`code1`, `dec_w2`, `head1`): each alphabet token has a +-1 code of width 128 (seeded by token id). Head, codebook and decoder are all this code, so the head correlates the written code with every token and argmax recovers it. Max cross-correlation between distinct codes is checked < 64 of 128.
- **Lexicon** (`tok_emb`): hand-written from language knowledge. Names (copy code + name flag), digits (copy code + digit flag), intent keywords (strength = logit, evidence one-hot). `pre_pos`, `lvl_pos` are one-hot; `lvl_emb` marks level rows; `w_in` routes the plan flags/intent into x.
- **Level 0** (`L1` heads 2-4, `head0`, `code0`): softmax pooling whose logit equals keyword strength: keywords dominate regardless of message length, giving per-intent evidence in [0,1]; separate heads give name presence and digit presence (logit 20). `head0` scores each of the 56 classes additively (7*evidence + 5*name-match + 5*digit-match, with biases for "absent"). `unknown` has no keywords and wins with a constant bias (evidence < ~0.3). `code0` writes intent one-hot + name/ticket flags into the latent.
- **Layer 0 attention** (3 heads): look 1,2,3 tokens back (one-hot position match) and read is-digit; the sum is a digit's rank in the ticket number. A fourth read on the 1-back head flags "greeting word before there".
- **Layer 0 MLP** (774 AND-neurons): one neuron per (intent, name?, position) writes the token code of the fixed reply token at that position (name shifts the core by 3, ticket tail follows the core); slot-marker neurons write WN (position 1 with a name) and WD_k (digit slots); rank one-hot neurons; "hi there" bigram neuron. Each neuron's bias is -(inputs-1), so it fires with value 1 iff all inputs are 1.
- **Layer 1 attention**: head 0, name slot rows attend the name token (logit 20) else a level row with zero code; head 1, digit slot k attends the digit token whose rank is k. Both copy its code. Gaps of 20 make it effectively hard attention.
- `head_b1[eos]=70`: with no code written the logits are all 0 so eos wins; a real token scores 128, noise <~40.

## Generalisation
Intent via a keyword lexicon (only vocabulary words), so new phrasings with those words work; names via a lexicon of ~55 vocabulary names, so new names are copied; tickets via digit presence and digit rank, so any template or number works. "my name is X", "X speaking", "ticket #N", "ticket id N" etc. work in my own tests (~100 phrasings).

## Failure modes
Names not in the lexicon are not copied (no name = no greeting). Other numbers in the message (e.g. "90%") trigger a ticket part. Intents with only unknown words go to `unknown`. Conflicting keywords (e.g. "wifi password") pick the stronger. Negation is ignored. Rank heads misbehave if a digit sits in the first 3 tokens.

## Path
First built bottom-up: template neurons + copy heads got 100% dev immediately after fixing a bias bug. Then tuned weak keywords for off-topic robustness, and moved pooling into layer 1 to allow the "hi there" bigram.
