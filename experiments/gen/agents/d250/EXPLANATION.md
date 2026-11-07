# d250 GenGPT, weights written by hand (model.wts)

**How it works.** Every bot turn is `<bot> <think> TAG </think> ...`. TAG is a real word (hello, thanks, goodbye, sorry, path, protected, ls, says, in, mentions, answer, lines, set, sure, deleted, kept, calc, grep) chosen at `<think>` from the SHAPE of the user message. A later head copies the tag back into an intent register (INTR 176-194). After that each next word is `f(intent, current word, previous-word flags, result status)`.

**Residual layout.** 0-3 token id mod 7,8,9,11 (X) | 4-7 their squares | 8-11 constant 1 | 12-15 output id Y | 16-31 class flags (FILE, SECRETS, KEY, NAME, WORDV, NUM, markers, PATH) | 32-47 trigger classes (verbs, yes/no, ops) | 72-94 "latest trigger / path / secrets / file / key / value present" | 95-96, 256-268, 379 previous-token flags | 100-159 one-hot of template words | 176-194 intent register | 200-232 registers (F file, K key, W value, R first result token, status, MYK key-of-value, V looked-up value, NOKEY) | 272-279 WANT flags | 280-299 previous-token ids and induction output | 300-366, 440-442 output channels of template words.

**Copying without knowing token ids.** Residues mod 7,8,9,11 are CRT-unique over 4096 tokens. Unembed gives logit `-10*sum(Y_m-r_m)^2`, so any register copied into Y emits exactly that token. Template words use one-hot output channels (+5000).

**Unseen phrasing.** The intent word head takes the LATEST trigger word after `<user>`, with synonym sets (show/open/read/print/cat, delete/remove/erase/rm/rid, set/change/update/edit/make/write, mention*/contain*/find/search/grep/where/which, files/ls/list/folder, plus/add/sum, minus/subtract, times/multiply/product, double/twice, yes/yep/sure/ok/confirm/go, no/nope/cancel/never/keep, hello/hi/hey, thanks/thank/cheers, bye/goodbye/later). Unknown words have no class and are ignored. Structure: file+key = lookup, file alone = show, key alone = lookup in the focus file, file with no verb = repeat last key, file+key+operator = read then calc, find word+key = grep then read. Focus = latest file outside results.

**Safety.** Any `..`, `/` or `~` token gives the `path` tag (9000, beats everything). `secrets` gives `protected` (8000). No tool is emitted for either. Result text is data: files/keys/values inside `<result>` or `<tool>` are masked (INRES/INTOOL), and no rule reads `note` lines. delete always replies `confirm` first; only yes/no after a `sure` tag acts (7000).

**Extending.** New synonym: add the word to the right `embed @word CH = 1` line in group cls.triggers. New intent: add a tag token channel (tok.tags), a classify neuron in L1.mlp.classify, and rule neurons in L2. New tool: add its name to the tool-name rules and a result-status flag.

**Known gaps.** Add-two-lookups ("layers in A plus count in B") is not built. grep-none in find-then-read, and lookups with a key repeated in an old result, are approximate.
