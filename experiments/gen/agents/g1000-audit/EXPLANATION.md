# EXPLANATION: GenGPT g1000

Files: `weights.py` builds every tensor (all are hand-written tables made from the spec, nothing is trained), `arch.py` is the
fixed tensor program that reads those tables. To change behaviour you almost always edit **lists/dicts at the top of
`build()` in `weights.py`** (word lists, `FLOWS`, `COND`, `CALLS`, `RTS`, ...). Tensor shapes follow the list lengths automatically.
After an edit run `uv run wsp-gen check --model experiments/gen/agents/g1000-audit/weights.py` (coverage must stay 1.0) and `explain`
(every group name must appear in this file).

## How GenGPT works (plain language)
1. Every token is turned into a few yes/no properties by a word list (`lex.*`): is it a file name, a key, a number, a "delete"
   word, a "yes" word, a path piece (`/`, `..`, `~`)...
2. At the start of the bot's turn, the properties of **only the current user turn** are counted (`bag.*`). A set of AND-rules
   (`flow.*`) turns the counts into one **intent** (hello, ls, show, lookup, find, calc, followup, double, lookup_calc,
   find_lookup, find_lookup_calc, sum2, write, delete_ask, delete_yes, delete_no, unsafe, thanks, bye, unknown).
3. The words the bot will need are picked out of the user turn: first file, second file, first/second key, the name/word to
   search, the last value, first/second number, the operator (`slot.*`). Other values are fetched from the conversation:
   the last "the answer is R" (ANS), the file in the last "are you sure ... ?" question (PREVF), the first token of the first
   result (HIT = grep hit), the value on the line of a result that STARTS with the key (V1, V2), the newest result (RES0).
4. Each intent owns a short **script**: which tool call to make after `k` finished calls (`call.lit`, `call.slot`):
   `<tool> name args </tool>`. The harness inserts the result. When no call is left (or the last result is `missing`, `none`
   or `error`) the intent plus the result status picks a **reply template** (`rt.map`, `tmpl.*`).
5. Reply tokens come from the template (fixed words or a slot) or, for lists and file contents, are copied one by one from the newest
   result (`tmpl.k0` says where copying starts; `</result>` is turned into `.`). After `.` or `?` the bot says `<end>`.
6. The network outputs one **target token id** per position. The logits are `-40*min(|id - target|, 3)`, so the argmax is exactly
   that token (`k.gain`, `k.cap`, `k.vid`). Calculator arithmetic is done by the calc tool, never by the weights.

## How it stays safe
- **Escape:** a word list entry `path` (`/`, `..`, `~`, `\`, and every name in `PROTECTED`) makes the `unsafe` intent win with
  priority 100 (`flow.prio`). The `unsafe` intent has no row in `call.lit`, so NO tool call can be emitted; it replies the fixed
  refusal. Tool arguments are copied only from user-turn slots, so no path can reach a tool anyway.
- **Unrequested write/delete:** `write` is only a tool name in the script of the `write` intent (needs a "set/change" cue plus a
  file and a key in the USER turn); `delete` is only in the script of `delete_yes`, which needs `pend` (the previous bot
  turn ended with the confirm `?`) AND a "yes" cue. `delete_ask` has no calls, it only says "are you sure ...?". The file deleted is
  PREVF, the file of the question the user answered.
- **Prompt injection:** the bag and the slots are masked to the tokens between the last `<user>` and `<bot>` (`MU` in `arch.py`).
  Tokens inside `<result>...</result>` can never choose an intent, a tool name or a write/delete argument. Results are only
  *copied* into the reply or used as a value (V1/V2/HIT) in a read/calc argument. A `note` line is never matched by a lookup because a
  key only matches at a line start (after `<result>` or `;`) and `note` is not a key.
- **Confirm before delete:** see above; `no` (or anything not "yes") after a confirm gives `ok , i kept F .` with no tool call.

## Edit map (recipes; all in `weights.py` unless said otherwise)
**Add one more word to a cue (hello, thanks, bye, ls, find, set, del, yes, no, rev, dbl, path):** append the word to the string in
`CUES[...]`. It must exist in the vocabulary. Add a new file/key/name/value: it is picked up from `kb.FILES/KEYS/NAMES/WORD_KEYS`.
Add an operator word: add `word: 'plus'|'minus'|'times'` to `OPWORDS`.

**Protect a file (never read, written, deleted or counted):** append its name to `PROTECTED`. Effect: lexicon row of that word gets
`path=1` (tensor `lex.feat`, column `path`), so `bag` counts it, intent `unsafe` fires and the refusal reply is said with no
tool call (even inside "delete X", "set k in X to v", "how many lines in X"). The file still shows up in `ls` output because that is
the tool's real result. To hide it from `ls` the harness must change; not possible in weights.

**Add a tool (example `wc F`, reply `{F} has {n} lines .`):**
1. If the user phrases it with new words: add a cue list `CUES['wc'] = 'how many lines count'`; add `'wc'` to `COLS` (top of file), to the `BAG` list and to the `IND` list
   (`('wc','wc',1)`, put it before `('pend',...)`) inside `build()`, so the word list, the counter and an indicator exist.
2. Add the intent name to the `FLOWS` list (append at the end) and a rule `COND['wc'] = (priority, ['wc','F1'], ['del','set'])` (+ required, - forbidden;
   pick a priority above rules it overlaps, e.g. 52).
3. Add its script `CALLS['wc'] = [('wc', ['F1'])]` (tool word must be in the vocabulary; args are names from `SV`: F1, F2, K1, K2, W,
   VAL, N1, N2, OP, ANS, HIT, PREVF, V1, V2, ARGA, ARGB; up to 3 args and up to 3 calls).
4. Add the reply `RTS['wc'] = '{F1} has {RES0} lines .'` and map the flow to it: add `'wc': 'wc'` to the `base` dict (flow -> reply type, just above the status loop). (This 7-place recipe was tested: `how many lines are in config ?` then calls `wc config`.) `{RES0}` is the first token of the
   tool's result; use `COPY` as the last word to copy a whole multi-token result.
5. A tool whose result can be `missing` is already handled: `rt.map` row (intent, status missing) is set in the status loop;
   add the flow name to the tuple in `if f in (...)` for status 1 if you want `i could not find {F1} .`.
No change to `arch.py` is needed. If the tool takes a new *kind* of argument, add a slot row to `SLOTS` (class column, n, value kind).

**Change a fixed reply sentence:** edit the string in `RTS[...]` (tensors `tmpl.lit` row = that reply, one token per column; `{X}` marks a slot).
Example: `RTS['thanks'] = 'any time .'`. Max 24 tokens, every word must be in the vocabulary, the sentence must end in `.` or `?`.
Keep `COPY` only as the final word. Same for `hello`, `bye`, `unknown`, `unsafe`, `confirm`, `kept`, `deleted`, `write`, ...

**Add a new kind of user phrasing for an existing intent:** add its words to the right `CUES` list; if the phrasing changes the
*structure* (e.g. two files and a key) check the `COND` row of the intent.

**Add one more flow of calc type:** give it `COND`, `CALLS` with `('calc', ['ARGA','OP','ARGB'])`, a row in `opd.a/opd.b/opd.swap`
(the `{...}.items()` dict: first operand, second operand, may-swap), and `RTS` mapping like `answer`.

## Limits
- 38,440 parameters in 88 tensors, 7,098 non-zero (most zero entries are padding rows of the lexicon/script tables).
- Intent detection is keyword + counting. A phrasing with none of the cue words falls to `unknown`; a user turn that names a file
  twice for different reasons can be mis-slotted. Operands follow text order.
- One missing-file read inside a multi-step flow stops the flow with `i could not find F .` (gold may differ).
- Replies are capped at 24 template tokens; calls at 3 per turn with 3 args; values are single tokens.
- No arithmetic in weights: results come from the calc tool.

## Group reference (one section per prov group; tensor = same name in `params`)

### `lex.id`
- shape/size: 4096 values (1144 non-zero)
- computes: token id -> lexicon row (0 = ordinary word, 1 = any number, 2.. = spec words).
- why these values: a hand-written index; every word of the spec (files, keys, values, tools, cue words) has its own row.
- breaks when: a word of the user is missing from the word lists (it is then an ordinary word and invisible)

### `lex.feat`
- shape/size: 4818 values (207 non-zero)
- computes: lexicon row -> 33 yes/no features (file? key? yes-cue? ...) read by every later step.
- why these values: feature 1.0 = the word has the property; opid holds the id of plus/minus/times for operator words.
- breaks when: a word of the user is missing from the word lists (it is then an ordinary word and invisible)

### `col.user`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "user" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.bot`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "bot" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.tool`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "tool" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.etool`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "etool" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.res`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "res" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.eres`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "eres" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.end`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "end" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.dot`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "dot" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.q`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "q" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.semi`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "semi" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.is`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "is" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.answer`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "answer" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.file`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "file" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.key`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "key" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.num`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "num" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.name`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "name" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.wordv`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "wordv" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.opw`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "opw" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.opid`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "opid" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.hello`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "hello" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.thanks`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "thanks" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.bye`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "bye" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.ls`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "ls" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.find`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "find" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.set`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "set" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.del`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "del" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.yes`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "yes" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.no`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "no" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.path`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "path" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.rev`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "rev" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.dbl`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "dbl" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.wtok`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "wtok" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `col.valtok`
- shape/size: 33 values (1 non-zero)
- computes: selector that reads feature column "valtok" out of a lexicon row.
- why these values: one-hot over the 33 feature columns.
- breaks when: never (selector)

### `k.zero`
- shape/size: 1 values (0 non-zero)
- computes: constant zero.
- why these values: zero used for padding.
- breaks when: never (constant)

### `k.half`
- shape/size: 1 values (1 non-zero)
- computes: constant half.
- why these values: threshold between 0/1 flags.
- breaks when: never (constant)

### `k.neg1`
- shape/size: 1 values (1 non-zero)
- computes: constant neg1.
- why these values: position of "nothing yet".
- breaks when: never (constant)

### `k.onef`
- shape/size: 1 values (1 non-zero)
- computes: constant onef.
- why these values: one.
- breaks when: never (constant)

### `k.two`
- shape/size: 1 values (1 non-zero)
- computes: constant two.
- why these values: two (confirm "?" sits 2 tokens before the next <user>).
- breaks when: never (constant)

### `k.tok2`
- shape/size: 1 values (1 non-zero)
- computes: constant tok2.
- why these values: id of the number token 2, used by "double".
- breaks when: never (constant)

### `k.period`
- shape/size: 1 values (1 non-zero)
- computes: constant period.
- why these values: id of ".".
- breaks when: never (constant)

### `k.end`
- shape/size: 1 values (1 non-zero)
- computes: constant end.
- why these values: id of <end>.
- breaks when: never (constant)

### `k.cap`
- shape/size: 1 values (1 non-zero)
- computes: constant cap.
- why these values: readout distance is capped at 3 so logits stay within +-120 (float32/fp16 safe).
- breaks when: never (constant)

### `k.gain`
- shape/size: 1 values (1 non-zero)
- computes: constant gain.
- why these values: logit = -40*min(|id-target|,3): the target id scores 0, any other id at most -40, so greedy argmax is exact.
- breaks when: never (constant)

### `k.fgain`
- shape/size: 1 values (1 non-zero)
- computes: constant fgain.
- why these values: softmax sharpness for the intent choice; priorities differ by >=1 so the winner gets weight ~1.
- breaks when: never (constant)

### `k.pos`
- shape/size: 1024 values (1023 non-zero)
- computes: scalar position of each token.
- why these values: position index 0..1023.
- breaks when: never (constant)

### `k.vid`
- shape/size: 4096 values (4095 non-zero)
- computes: scalar id of each vocabulary entry (readout number line).
- why these values: vocabulary ids 0..4095.
- breaks when: never (constant)

### `k.jr`
- shape/size: 6 values (5 non-zero)
- computes: call-step index range.
- why these values: a call has at most 6 steps: <tool> name a0 a1 a2 </tool>.
- breaks when: never (constant)

### `k.cr`
- shape/size: 4 values (3 non-zero)
- computes: finished-call-count range.
- why these values: at most 3 calls per turn.
- breaks when: never (constant)

### `k.kr`
- shape/size: 24 values (23 non-zero)
- computes: reply token index range.
- why these values: longest reply (hello) has 20 tokens.
- breaks when: never (constant)

### `bag.sel`
- shape/size: 561 values (17 non-zero)
- computes: which lexicon features are summed over the current user turn (bag of words counts).
- why these values: one-hot columns of the 17 counted features.
- breaks when: a counted feature is added without a column here

### `bag.rev`
- shape/size: 17 values (1 non-zero)
- computes: reads the "subtract" count.
- why these values: subtract B from A flips the operand order.
- breaks when: a counted feature is added without a column here

### `ind.sel`
- shape/size: 360 values (20 non-zero)
- computes: picks one count per indicator.
- why these values: indicator j looks at one count (or the pending-confirm flag).
- breaks when: a new cue needs a new indicator row

### `ind.thr`
- shape/size: 20 values (3 non-zero)
- computes: indicator = clamp(count - (min-1), 0, 1).
- why these values: min count 2 means "at least two files".
- breaks when: a new cue needs a new indicator row

### `flow.and`
- shape/size: 400 values (64 non-zero)
- computes: per intent: +1 on required cues, -1 on forbidden cues.
- why these values: relu(sum - (m-1)) is 1 only if all m required cues are on and no forbidden one is.
- breaks when: two matched rules have equal priority, or a phrasing matches no rule (falls to unknown)

### `flow.bias`
- shape/size: 20 values (10 non-zero)
- computes: -(m-1) for an intent with m required cues.
- why these values: turns the sum into an AND.
- breaks when: two matched rules have equal priority, or a phrasing matches no rule (falls to unknown)

### `flow.prio`
- shape/size: 20 values (19 non-zero)
- computes: intent priority (score if matched).
- why these values: unsafe 100 > delete answers 90 > ... > greetings 20; ties impossible between overlapping rules.
- breaks when: two matched rules have equal priority, or a phrasing matches no rule (falls to unknown)

### `flow.base`
- shape/size: 20 values (1 non-zero)
- computes: fallback score of "unknown".
- why these values: score 1 beats nothing-matched (0) and loses to any matched intent (>=20).
- breaks when: two matched rules have equal priority, or a phrasing matches no rule (falls to unknown)

### `slot.flag`
- shape/size: 363 values (11 non-zero)
- computes: which token class each user slot scans for.
- why these values: F=file names, K=keys, W=name/word values, VAL=any value, N=numbers, OP=operator words.
- breaks when: the user repeats a file/key/number in an unexpected order

### `slot.n`
- shape/size: 11 values (11 non-zero)
- computes: which occurrence to take (-1 = the last).
- why these values: the 1st file is the one asked about; the value to write is the LAST value token.
- breaks when: the user repeats a file/key/number in an unexpected order

### `slot.val`
- shape/size: 33 values (11 non-zero)
- computes: what to read from the chosen token: its id, its operator id, or its position.
- why these values: ids are copied as numbers; positions give text order.
- breaks when: the user repeats a file/key/number in an unexpected order

### `sv.fromslot`
- shape/size: 198 values (9 non-zero)
- computes: copies the 9 user slots into the 18-entry value vector.
- why these values: identity on the first 9 entries; PK/PN are only used for ordering.
- breaks when: never (routing)

### `sv.ans`
- shape/size: 18 values (1 non-zero)
- computes: writes the computed value "ANS" into its entry of the value vector.
- why these values: last "the answer is R" in earlier bot text.
- breaks when: never (routing)

### `sv.hit`
- shape/size: 18 values (1 non-zero)
- computes: writes the computed value "HIT" into its entry of the value vector.
- why these values: first token of the first result (the grep hit).
- breaks when: never (routing)

### `sv.prevf`
- shape/size: 18 values (1 non-zero)
- computes: writes the computed value "PREVF" into its entry of the value vector.
- why these values: file right before the last "?" (the confirm question).
- breaks when: never (routing)

### `sv.v1`
- shape/size: 18 values (1 non-zero)
- computes: writes the computed value "V1" into its entry of the value vector.
- why these values: value of K1 on a key-starting line of result RA.
- breaks when: never (routing)

### `sv.v2`
- shape/size: 18 values (1 non-zero)
- computes: writes the computed value "V2" into its entry of the value vector.
- why these values: value of K2 on a line of result RB.
- breaks when: never (routing)

### `sv.res0`
- shape/size: 18 values (1 non-zero)
- computes: writes the computed value "RES0" into its entry of the value vector.
- why these values: first token of the newest result.
- breaks when: never (routing)

### `sv.two`
- shape/size: 18 values (1 non-zero)
- computes: writes the computed value "TWO" into its entry of the value vector.
- why these values: constant 2 for double.
- breaks when: never (routing)

### `sv.arga`
- shape/size: 18 values (1 non-zero)
- computes: writes the computed value "ARGA" into its entry of the value vector.
- why these values: first calc operand after swap.
- breaks when: never (routing)

### `sv.argb`
- shape/size: 18 values (1 non-zero)
- computes: writes the computed value "ARGB" into its entry of the value vector.
- why these values: second calc operand after swap.
- breaks when: never (routing)

### `sl.k1`
- shape/size: 11 values (1 non-zero)
- computes: selects slot K1 from the slot list.
- why these values: one-hot.
- breaks when: never (selector)

### `sl.k2`
- shape/size: 11 values (1 non-zero)
- computes: selects slot K2 from the slot list.
- why these values: one-hot.
- breaks when: never (selector)

### `sl.pk`
- shape/size: 11 values (1 non-zero)
- computes: selects slot PK from the slot list.
- why these values: one-hot.
- breaks when: never (selector)

### `sl.pn`
- shape/size: 11 values (1 non-zero)
- computes: selects slot PN from the slot list.
- why these values: one-hot.
- breaks when: never (selector)

### `flow.ra`
- shape/size: 20 values (5 non-zero)
- computes: which result of the turn holds the line for K1.
- why these values: grep-first flows read the file as result 2.
- breaks when: two matched rules have equal priority, or a phrasing matches no rule (falls to unknown)

### `flow.rb`
- shape/size: 20 values (1 non-zero)
- computes: which result holds the line for K2.
- why these values: sum2 reads file 2 as result 2.
- breaks when: two matched rules have equal priority, or a phrasing matches no rule (falls to unknown)

### `opd.a`
- shape/size: 360 values (6 non-zero)
- computes: first calc operand per intent.
- why these values: calc: 1st number; followup/double: previous answer; lookups: the looked-up value.
- breaks when: a new calc-like flow has no operand row

### `opd.b`
- shape/size: 360 values (6 non-zero)
- computes: second calc operand per intent.
- why these values: calc: 2nd number; followup: the number; double: 2; sum2: second value.
- breaks when: a new calc-like flow has no operand row

### `opd.swap`
- shape/size: 20 values (3 non-zero)
- computes: intent allows reordering of the operands.
- why these values: order follows the text (number first => swapped); "subtract" flips again.
- breaks when: a new calc-like flow has no operand row

### `call.lit`
- shape/size: 480 values (57 non-zero)
- computes: per (intent, calls done, step): literal token to emit: <tool>, tool name, </tool>.
- why these values: step 0 non-zero means "make another call"; zero means "start the reply".
- breaks when: a flow needs more than 3 calls or 3 args

### `call.slot`
- shape/size: 8640 values (32 non-zero)
- computes: per (intent, calls done, step): which value-vector entry to copy as an argument.
- why these values: e.g. read gets F1 (the user-named file) or HIT (a grep hit); delete gets PREVF (the confirmed file).
- breaks when: a flow needs more than 3 calls or 3 args

### `tmpl.lit`
- shape/size: 480 values (111 non-zero)
- computes: per (reply type, token index): literal word of the fixed sentence.
- why these values: the exact reply templates of the spec, one token per column.
- breaks when: a reply is longer than 24 tokens

### `tmpl.slot`
- shape/size: 8640 values (21 non-zero)
- computes: per (reply type, token index): which value-vector entry fills a {slot}.
- why these values: {f}->F1, {k}->K1, {v}->V1 or VAL, {w}->W, {r}->RES0.
- breaks when: a reply is longer than 24 tokens

### `tmpl.k0`
- shape/size: 20 values (20 non-zero)
- computes: token index where the copy of the newest result starts (99 = never).
- why these values: list/content replies copy the tool result then end with ".".
- breaks when: a reply is longer than 24 tokens

### `rt.map`
- shape/size: 2000 values (100 non-zero)
- computes: per (intent, result status) -> which reply type.
- why these values: status of the newest result: normal / missing / none / error / key-absent decides between e.g. lookup and "has no".
- breaks when: a result status has no reply row

### `st.ids`
- shape/size: 3 values (3 non-zero)
- computes: ids of the result words missing / none / error.
- why these values: first token of the newest result is compared to these.
- breaks when: the tool returns a new status word
