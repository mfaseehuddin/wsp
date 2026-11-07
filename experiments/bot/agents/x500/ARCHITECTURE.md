# Custom architecture: rule-memory machine

## What it is
```
token ids -> LEXICON (id -> features) -> COUNTERS (where in the turn?)
          -> evidence sum over user message -> pick script (think word)
          -> COPY HEADS (newest topic / id / service / result / user)
          -> PROGRAM MEMORY (script, step) rows -> vote for next token id
```
Only tensor ops are used (range matches, cummax, softmax attention, products, matmuls). There is no embedding matrix: a token id is a number and the "embedding" is a 386-row range-match table.

## Why it beats a standard transformer here
- Parameters: 4,897 total. The fixed transformer needs 2,000 x d_model for the embedding alone (d=32 gives 64,000, about 13x more) before any layer.
- Compute per token: about 386 box tests + 532 row matches + a 5-head lookup. Roughly 3 million MACs for a 256-token context, much less with caching.
- Editability: one fact is one row (one reply word = one `rom_tok`; one keyword = one lexicon row). A transformer spreads a fact over dense weights.
- Context and tools: counters and hard copy heads copy ids, names, services and states exactly, with no softmax blur.

## What it gives up
No generalisation beyond hand-written keywords; cost grows with the number of rows; new words need new rows; it is a table, not a learned model. It fixes the vocabulary at 2000 ids and 256 positions.

## Embedded fitness
- Memory (float32): 4,897 x 4 B = about 20 KB of weights (fits ESP32's 520 KB SRAM). Peak activations for one token with caching are small; the attention score tensor is T x T x 5 floats (about 1.3 MB at T=256, but a streaming implementation needs only one row).
- Ops per generated token: about 10 thousand MACs with caching.
- float32: yes. All ids are exact integers below 2,000; match tests use exact equality of small integers. Largest constant is 20,000 (attention mask), under the float16 limit of 6x10^4, and scores stay below that.
