# GenGPT d100: hand-written weights (5 layers, 18 heads of dim 36, d_model 416, tied embeddings)

## How it works
Every output token is chosen by a *tied* embedding: a token's row is its **code** (class dim + three digit one-hots); the logit of a token is the overlap of the residual with that code. Writing 16 on a token's four code dims emits it. Numbers get class 0 + hundreds/tens/ones digits (set with id ranges), files class 1, keys class 2, names/word values class 3, template words class 4 (+ a one-hot id channel 300+n for "current token is X" tests).

Pipeline per generated token (layer: job):
- L0: inUser/inResult markers (latest marker head), turn id = position and position^2 of the latest `<user>` (so a quadratic equality score isolates the current exchange), previous-token flags, clean snapshot of each token's code (dims 380-415).
- L1: user-message facts by priority attention: path/secrets, dominant verb (priority by key boost: wc > delete > set > find > list > show > no > yes > bye > thanks > greet), operator words, has file/key/number/value; pointer heads (latest file outside results = focus, latest key, first/last number, latest answer, result start/end, tool name, first result word, pending-confirm = token at turn-2). MLP1 turns facts into one-hot **intents** (shape based: key present = lookup, file only = show, file without key after a keyed question = lookup with the previous key, op word + number = calc, key + op + number = lookup-calc ...).
- L2: key-code copy; MLP2 is the script table: one neuron per edge (current token + intent/tool/result flags -> next template word, or raise a pointer selector).
- L3: value finder (key code vs line-start tokens, `<result>` is a sink so a missing key is detected); MLP3 gates the chosen pointer into channel 260 (relu(ptr + 1000*sel - 1000)).
- L4: universal copy head attends to position = pointer and copies the clean code x16 into the logits. `</result>` has the code of "." so copying past a result ends the sentence.

## Safety
A path token (`/`, `..`, `~`) has the highest verb priority and gives the unsafe reply with no tool; `secrets` gives the protected reply; delete never calls the tool: it replies confirm, and the next turn calls delete only if the previous reply ended with `?` and the user said yes. Result text is never used for focus (inResult penalty) and the value finder only matches keys at a line start, so `note ...` lines cannot act.

## Unseen phrasings
The decision uses structure (file/key/number/op/verb flags), not whole phrases; unknown words have no flags and are ignored. Coreference falls out of "latest file / latest key outside results".

## Extending
Synonym: add the token to a `flags.*` embed line. New word in a reply: give it a code and id channel. New intent: add an intent neuron in L1, its script neurons in L2. New tool: add a tool-name flag channel (65-71 style), a selector and edges.

## Known gaps
find-lookup chains (grep then read), sum of two lookups and think-memory are not built; the nokey path (L3/L4 neurons) was patched last and not re-verified.
