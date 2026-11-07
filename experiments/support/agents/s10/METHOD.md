# s10 method (budget 10)

## Design reasoning
- The cost is dominated by V×d, so the aim was the smallest d. Tied embeddings save a further 470·d. A one-layer pipeline works: the intent and previous-token heads run in parallel, and one MLP acts as a lookup table.
- 2-D angle codes are used throughout. The 126 reply tokens sit on a circle. Any point on a circle is an argmax-extreme direction, so 2 dims are enough to output any reply token. The same codes serve as exact-match detectors: ReLU(G(a·a_c − 1) + 1).
- Analysis of the replies: (prev, cur) determines the next token in every case except 12 contexts that are shared across intents (e.g. `<sep>|restart` in vpn and slow, `your|computer` in email and slow). 98 of the 125 tokens have a unique successor. So I keep the bigram in the embedding (B dims plus 2 transfer neurons) and add neurons only where it is wrong.
- Intent is a hard max over hand-tiered keywords with a `<bos>` sink. Off-topic words get an "unknown" vote above greetings, so "hi, what's the weather" goes to unknown. App and apps were left out on purpose, because they occur in the slow reply ahead of a gated context.
- Static (non-forward) check: duplicate keywords = none, and every keyword exists in the vocabulary (`tid` asserts it). I listed the cross-intent keywords that occur inside replies: vpn has internet/connection, wifi has password, software has approved, mfa has request. None of those replies has an intent-gated context after the contaminating word, so they are harmless.

## Forward passes
1. `eval` on ids 8, 28, 21, 41, 16, 53, 47 (vpn, slow, printer, monitor, email, unknown, hello), 7 passes. Together these cover every kind of intent-gated correction. Result: 7/7 exact, token acc 1.0. Nothing changed.
2. `eval` on ids 50, 6, 3 (thanks, locked, password), 3 passes, for the remaining gated contexts (`you|are`, `minutes|.`, `and|choose`). Result: 3/3 exact. Nothing changed.

Total: 10/10 dev queries verified exact. The other 46 dev queries were not run because the budget was used up. By construction their correctness depends only on the keyword lists.
