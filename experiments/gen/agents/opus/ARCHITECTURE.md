# GenGPT-opus architecture

```
user turn tokens --lexicon--> word properties --bag + AND rules + priorities--> intent (26)
                                    |                                              |
                              slots (n-th file/key/name/number)                    v
previous <think> F K </think> ----> focus file FF, focus key FK -------> value vector (20 entries)
tool results ----hard attention----> grep hit, value after "key" at line start ---^
                                                                                   |
               intent x calls-done x step --> tool-call tokens  /  reply template + copy
               after the final "." or "?" --> <think> focus-file focus-key </think> <end>
```

**Why this, not a standard transformer.** It has 53k parameters (7.4k nonzero). A 4,096-word transformer spends 4,096 × d on its embedding alone, before doing anything. Every mechanism here is a named table you can edit (see EXPLANATION.md), and copying is exact hard attention, so there's no blur on ids.

**The new part: thinking as working memory.** The model writes its state into the context as think tokens and reads it back on the next turn. That is the transformer way to keep state across turns: no recurrent cell, and the state is visible text. A person can read it, so it can be checked. It is also *safe* state: the read mask only looks at the model's own think tokens, so injected file text can't move the focus.

**What it gives up.** It is a rule system expressed as tensor ops, not a learned distribution over language. Understanding is limited to the properties its lexicon assigns. Memory holds one file and one key.

**Embedded fitness.** About 212 KB in float32 (about 30 KB if only nonzero values are stored). Cost per token is O(T² × 14) for the masks, and a KV-cached version is O(T × 14). Magnitudes stay at or below 4,095, so float32 and even float16 are safe.
