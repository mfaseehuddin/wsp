# ARCHITECTURE (x500)

Standard causal decoder: 8 layers, d_model 656, 4 heads x d_head 64, d_mlp 256, learned position table (16384 x 656, almost all zeros), no LayerNorm, biases on. 19.34M parameters (about 0.65M nonzero; the position table and zero-padded matrices dominate). KV cache: 8 x 4 x 64 floats per token (about 256 MiB float32 at 16k, as reported by `wsp-x check`).

## Residual-stream layout (d = 656)
| channels | content |
|---|---|
| 0..31 | position pairs sin/cos(t*w_i), 16 high frequencies (read only by shift heads) |
| 32..~600 | named feature channels, allocated by name at build time: `t<id>` one-hot token (amplitude U=32); `c_*` token classes (set in `embed`); `p1:<ch>` channel `<ch>` at the previous position(s) (k = 1..4); `L_*` turn-local keyword flags; `ph_*` phase (latest anchor token in the reply); `f_*` computed intents/start flags; `m_*` mode switches that open copy heads; `p1c_j`, `p1:p1c_j` code of the previous two tokens; `cout_j` emit code (28 dims) |
| d-2, d-1 | slowest position pair, used as recency (monotone for t < 16k) |

All binary features have amplitude 32. A rule neuron pair saturates, so features stay exactly 0/32 (the position table is zero outside its channels, so there is no positional noise).

## Block plan (a scheduler assigns each head/rule to the earliest layer whose inputs exist)
- L0: shift heads (previous token, previous-token code), flag heads (digit, operator, greeting/thanks/bye, secret/dots, read/delete, yes/no/word, name word, question mark), calc intent rule.
- L1-L2: phase head (latest anchor token), more shifts (k=2..4), start-flag rules (expression start, name start, path start, result start), relation/entity heads.
- L3-L5: intent/mode rules, frame chains, fact rules (2 neurons per fact), namestart flag.
- L5-L7: gated copy heads (first-token heads: expression/word, name, result, path, text, relation, entity) and two induction heads (name; calc/result/path/text/word share one head with five class dims).
- Unembed: logit(v) = 20 * cos(code(v), cout); END bias 3.

## Float32 margins
Weights max 4e4 (attention gains); scores reach about 1e6 with float32 spacing under 0.1, recency steps 30 per token in latest heads. Induction scores: exact 2-gram+class 180k, sink 150k, 1-gram+class about 120k; margin about 30k in the worst case (code cross-correlation up to about 0.5).

## What composes
Flags -> intents -> frame chains -> copy heads -> tool calls -> RESULT copy -> answer frame. Facts, names and tool paths compose with the same frames.
## What does not
Arbitrary text generation, reasoning beyond one calc, multi-fact stores keyed by novel entities, repeated-bigram copying (11, te), anything needing counting.
