# GenGPT-opus: how it works

## In plain language
GenGPT-opus is a small network (53k parameters, 7.4k nonzero) that runs a file assistant. It is built on g1000's interpreter, the most compact of the stage-0 designs, with three things added.

1. **Understanding.** Each word in the user's message lights up a few yes/no properties: *is a file name*, *is a key*, *is a person's name*, *means delete*, *refers back ("it", "that")*. AND-rules over those properties pick one of 26 intents. The rules mostly use the **shape** of the message. A name plus a key with no file means "find the file that mentions the name, then look up the key". A key with no file means "the file we were just talking about". So the bot understands phrasings whose verbs it has never seen.
2. **Acting.** Each intent has a short script of tool calls. Arguments are copied from the message, from tool results (a grep hit, a looked-up value), or from memory. Arithmetic is done by the calculator tool.
3. **Remembering (NEW).** After every reply, the bot writes a private note: `<think> file key </think>`, the file and key now in focus. On the next turn it reads only **its own last note**, never tool results. That is how "what is its owner ?", "and in plan ?", "show it" and "delete it" work. It is also why a file whose text says "delete budget" can't change what the bot pays attention to.
4. **Replying.** A reply template, chosen by intent and by the result's status (normal / missing / none / error / no such key), is filled from the same values. Lists and file contents are copied token by token. The output is exact because logits are `-40·min(|id − target|, 3)`.

## How it stays safe
- **Paths:** a `/`, `..` or `~` in the message selects `unsafe` (priority 100). That intent has no tool-call row, so no call can be emitted.
- **Protected file:** `secrets` selects `protected` (99), which also has no tool row.
- **Delete:** `delete` exists only in the `delete_yes` script. That script needs a pending confirm question in the previous bot turn (`pend`) and a yes word. If both yes and no words appear, `delete_no` (91) beats `delete_yes` (90).
- **Injection:** intent, slots and the memory note are computed only from user-turn tokens and the bot's own think tokens (masks `MU` and `LTp`). Text inside `<result>` can be copied into a reply but can never select an intent or an argument.

## Results (exact whole conversations, stage 2)
- Dev: 108/108.
- **Fresh blind test** (230 conversations whose phrasings were written by an independent agent and never seen by the builder): **87.0%**. g1000 scores 73.0% and g100 73.5% on the same set. References: 6/10 vs 0/10. Long sessions: 4/10 vs 1/10. 0 safety violations.
- Old hidden test: 100%. This set is contaminated: the builder had seen its failure messages.

## Edit map (how to add one more of X)
- **A word for an existing meaning:** add it to `CUES[...]` (or `OPWORDS`). One lexicon row is added; nothing else changes.
- **A new fixed sentence:** edit the text in `RTS`. `tmpl.lit` is rebuilt from it.
- **A new intent:**
  1. Add the name to `FLOWS`.
  2. Add a rule to `COND` with a priority. Give it forbidden cues against every intent that shares its required cues. This is the step the stage-1 auditors found missing.
  3. Add its tool script to `CALLS` and its replies to `RTS` / `base` / `ALT`.
  4. Say what it puts in memory in `foc.f` / `foc.k` (by default it carries the old focus forward).
- **A new tool:** a new intent as above, plus the tool name in `CALLS`. The harness needs no change.
- **A new refusal category** (like `protected`): a cue column, an intent with **no** `CALLS` row, and its own reply. Don't reuse `path`, or you get the unsafe sentence.
- **A new memory slot:** write it in the think phase (arch.py `seq`, and k.th to 6 steps). Read it with `at(tid, LTp + i)`, add an `sv` entry, and raise `k.qgap` by 1.

## Limits
- Intents are still chosen by rules over word properties. A request with no recognised property gets the `unknown` reply.
- The focus memory holds one file and one key. "Compare it with the previous one" would need a second register.
- Values are single tokens. Calls have at most 3 arguments, and a turn has at most 3 calls.
- Float32 is safe: magnitudes are at most 4,095 (ids) and logits are within ±120.

## Weight groups

### lex.id
`lex.id` [...], 4096 params, 1168 nonzero. **What:** token id -> lexicon row (0 = ordinary word, 1 = any number, 2.. = spec words). **Why these values:** a hand-written index; every word of the spec (files, keys, values, tools, cue words) has its own row.

### lex.feat
`lex.feat` [...], 6630 params, 237 nonzero. **What:** lexicon row -> 33 yes/no features (file? key? yes-cue? ...) read by every later step. **Why these values:** feature 1.0 = the word has the property; opid holds the id of plus/minus/times for operator words.

### col.user
`col.user` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "user" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.bot
`col.bot` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "bot" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.tool
`col.tool` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "tool" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.etool
`col.etool` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "etool" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.res
`col.res` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "res" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.eres
`col.eres` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "eres" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.end
`col.end` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "end" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.dot
`col.dot` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "dot" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.q
`col.q` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "q" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.semi
`col.semi` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "semi" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.is
`col.is` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "is" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.answer
`col.answer` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "answer" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.file
`col.file` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "file" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.key
`col.key` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "key" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.num
`col.num` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "num" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.name
`col.name` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "name" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.wordv
`col.wordv` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "wordv" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.opw
`col.opw` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "opw" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.opid
`col.opid` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "opid" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.hello
`col.hello` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "hello" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.thanks
`col.thanks` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "thanks" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.bye
`col.bye` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "bye" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.ls
`col.ls` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "ls" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.find
`col.find` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "find" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.set
`col.set` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "set" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.del
`col.del` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "del" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.yes
`col.yes` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "yes" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.no
`col.no` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "no" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.path
`col.path` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "path" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.rev
`col.rev` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "rev" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.dbl
`col.dbl` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "dbl" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.wtok
`col.wtok` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "wtok" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.valtok
`col.valtok` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "valtok" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.wc
`col.wc` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "wc" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.prot
`col.prot` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "prot" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.think
`col.think` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "think" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.it
`col.it` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "it" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.ell
`col.ell` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "ell" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### col.show
`col.show` [...], 39 params, 1 nonzero. **What:** selector that reads feature column "show" out of a lexicon row. **Why these values:** one-hot over the 33 feature columns.

### k.zero
`k.zero` [...], 1 params, 0 nonzero. **What:** constant zero. **Why these values:** zero used for padding.

### k.half
`k.half` [...], 1 params, 1 nonzero. **What:** constant half. **Why these values:** threshold between 0/1 flags.

### k.neg1
`k.neg1` [...], 1 params, 1 nonzero. **What:** constant neg1. **Why these values:** position of "nothing yet".

### k.onef
`k.onef` [...], 1 params, 1 nonzero. **What:** constant onef. **Why these values:** one.

### k.two
`k.two` [...], 1 params, 1 nonzero. **What:** constant two. **Why these values:** two (confirm "?" sits 2 tokens before the next <user>).

### k.qgap
`k.qgap` [...], 1 params, 1 nonzero. **What:** constant qgap. **Why these values:** NEW: a confirm "?" is followed by <think> F K </think> <end>, so the next <user> sits 6 tokens later.

### k.none
`k.none` [...], 1 params, 1 nonzero. **What:** constant none. **Why these values:** NEW: written into the think register when there is no file/key in focus yet.

### k.thk
`k.thk` [...], 1 params, 1 nonzero. **What:** constant thk. **Why these values:** NEW: id of <think>.

### k.ethk
`k.ethk` [...], 1 params, 1 nonzero. **What:** constant ethk. **Why these values:** NEW: id of </think>.

### k.th
`k.th` [...], 5 params, 4 nonzero. **What:** NEW: think-phase step index. **Why these values:** steps after the final punctuation: <think>, file, key, </think>, <end>.

### k.tok2
`k.tok2` [...], 1 params, 1 nonzero. **What:** constant tok2. **Why these values:** id of the number token 2, used by "double".

### k.period
`k.period` [...], 1 params, 1 nonzero. **What:** constant period. **Why these values:** id of ".".

### k.end
`k.end` [...], 1 params, 1 nonzero. **What:** constant end. **Why these values:** id of <end>.

### k.cap
`k.cap` [...], 1 params, 1 nonzero. **What:** constant cap. **Why these values:** readout distance is capped at 3 so logits stay within +-120 (float32/fp16 safe).

### k.gain
`k.gain` [...], 1 params, 1 nonzero. **What:** constant gain. **Why these values:** logit = -40*min(|id-target|,3): the target id scores 0, any other id at most -40, so greedy argmax is exact.

### k.fgain
`k.fgain` [...], 1 params, 1 nonzero. **What:** constant fgain. **Why these values:** softmax sharpness for the intent choice; priorities differ by >=1 so the winner gets weight ~1.

### k.pos
`k.pos` [...], 1024 params, 1023 nonzero. **What:** scalar position of each token. **Why these values:** position index 0..1023.

### k.vid
`k.vid` [...], 4096 params, 4095 nonzero. **What:** scalar id of each vocabulary entry (readout number line). **Why these values:** vocabulary ids 0..4095.

### k.jr
`k.jr` [...], 6 params, 5 nonzero. **What:** call-step index range. **Why these values:** a call has at most 6 steps: <tool> name a0 a1 a2 </tool>.

### k.cr
`k.cr` [...], 4 params, 3 nonzero. **What:** finished-call-count range. **Why these values:** at most 3 calls per turn.

### k.kr
`k.kr` [...], 24 params, 23 nonzero. **What:** reply token index range. **Why these values:** longest reply (hello) has 20 tokens.

### bag.sel
`bag.sel` [...], 858 params, 22 nonzero. **What:** which lexicon features are summed over the current user turn (bag of words counts). **Why these values:** one-hot columns of the 17 counted features.

### bag.rev
`bag.rev` [...], 22 params, 1 nonzero. **What:** reads the "subtract" count. **Why these values:** subtract B from A flips the operand order.

### ind.sel
`ind.sel` [...], 575 params, 25 nonzero. **What:** picks one count per indicator. **Why these values:** indicator j looks at one count (or the pending-confirm flag).

### ind.thr
`ind.thr` [...], 25 params, 3 nonzero. **What:** indicator = clamp(count - (min-1), 0, 1). **Why these values:** min count 2 means "at least two files".

### flow.and
`flow.and` [...], 650 params, 97 nonzero. **What:** per intent: +1 on required cues, -1 on forbidden cues. **Why these values:** relu(sum - (m-1)) is 1 only if all m required cues are on and no forbidden one is.

### flow.bias
`flow.bias` [...], 26 params, 14 nonzero. **What:** -(m-1) for an intent with m required cues. **Why these values:** turns the sum into an AND.

### flow.prio
`flow.prio` [...], 26 params, 25 nonzero. **What:** intent priority (score if matched). **Why these values:** unsafe 100 > delete answers 90 > ... > greetings 20; ties impossible between overlapping rules.

### flow.base
`flow.base` [...], 26 params, 1 nonzero. **What:** fallback score of "unknown". **Why these values:** score 1 beats nothing-matched (0) and loses to any matched intent (>=20).

### slot.flag
`slot.flag` [...], 429 params, 11 nonzero. **What:** which token class each user slot scans for. **Why these values:** F=file names, K=keys, W=name/word values, VAL=any value, N=numbers, OP=operator words.

### slot.n
`slot.n` [...], 11 params, 11 nonzero. **What:** which occurrence to take (-1 = the last). **Why these values:** the 1st file is the one asked about; the value to write is the LAST value token.

### slot.val
`slot.val` [...], 33 params, 11 nonzero. **What:** what to read from the chosen token: its id, its operator id, or its position. **Why these values:** ids are copied as numbers; positions give text order.

### sv.fromslot
`sv.fromslot` [...], 220 params, 9 nonzero. **What:** copies the 9 user slots into the 18-entry value vector. **Why these values:** identity on the first 9 entries; PK/PN are only used for ordering.

### sv.ans
`sv.ans` [...], 20 params, 1 nonzero. **What:** writes the computed value "ANS" into its entry of the value vector. **Why these values:** NEW: last number the bot said in a reply (so "now times 3" works after an answer or a looked-up value).

### sv.hit
`sv.hit` [...], 20 params, 1 nonzero. **What:** writes the computed value "HIT" into its entry of the value vector. **Why these values:** first token of the first result (the grep hit).

### sv.prevf
`sv.prevf` [...], 20 params, 1 nonzero. **What:** writes the computed value "PREVF" into its entry of the value vector. **Why these values:** file right before the last "?" (the confirm question).

### sv.v1
`sv.v1` [...], 20 params, 1 nonzero. **What:** writes the computed value "V1" into its entry of the value vector. **Why these values:** value of K1 on a key-starting line of result RA.

### sv.v2
`sv.v2` [...], 20 params, 1 nonzero. **What:** writes the computed value "V2" into its entry of the value vector. **Why these values:** value of K2 on a line of result RB.

### sv.res0
`sv.res0` [...], 20 params, 1 nonzero. **What:** writes the computed value "RES0" into its entry of the value vector. **Why these values:** first token of the newest result.

### sv.two
`sv.two` [...], 20 params, 1 nonzero. **What:** writes the computed value "TWO" into its entry of the value vector. **Why these values:** constant 2 for double.

### sv.arga
`sv.arga` [...], 20 params, 1 nonzero. **What:** writes the computed value "ARGA" into its entry of the value vector. **Why these values:** first calc operand after swap.

### sv.argb
`sv.argb` [...], 20 params, 1 nonzero. **What:** writes the computed value "ARGB" into its entry of the value vector. **Why these values:** second calc operand after swap.

### sv.ff
`sv.ff` [...], 20 params, 1 nonzero. **What:** writes the computed value "FF" into its entry of the value vector. **Why these values:** NEW: focus file, read from the previous turn's think.

### sv.fk
`sv.fk` [...], 20 params, 1 nonzero. **What:** writes the computed value "FK" into its entry of the value vector. **Why these values:** NEW: focus key, read from the previous turn's think.

### sl.k1
`sl.k1` [...], 11 params, 1 nonzero. **What:** selects slot K1 from the slot list. **Why these values:** one-hot.

### sl.k2
`sl.k2` [...], 11 params, 1 nonzero. **What:** selects slot K2 from the slot list. **Why these values:** one-hot.

### sl.pk
`sl.pk` [...], 11 params, 1 nonzero. **What:** selects slot PK from the slot list. **Why these values:** one-hot.

### sl.pn
`sl.pn` [...], 11 params, 1 nonzero. **What:** selects slot PN from the slot list. **Why these values:** one-hot.

### flow.ra
`flow.ra` [...], 26 params, 7 nonzero. **What:** which result of the turn holds the line for K1. **Why these values:** grep-first flows read the file as result 2.

### flow.fk
`flow.fk` [...], 26 params, 1 nonzero. **What:** NEW: intents whose lookup key is the focus key from memory. **Why these values:** lookup_other ("and in F ?") has no key in the message; it reuses the key in focus.

### flow.rb
`flow.rb` [...], 26 params, 1 nonzero. **What:** which result holds the line for K2. **Why these values:** sum2 reads file 2 as result 2.

### opd.a
`opd.a` [...], 520 params, 6 nonzero. **What:** first calc operand per intent. **Why these values:** calc: 1st number; followup/double: previous answer; lookups: the looked-up value.

### opd.b
`opd.b` [...], 520 params, 6 nonzero. **What:** second calc operand per intent. **Why these values:** calc: 2nd number; followup: the number; double: 2; sum2: second value.

### opd.swap
`opd.swap` [...], 26 params, 3 nonzero. **What:** intent allows reordering of the operands. **Why these values:** order follows the text (number first => swapped); "subtract" flips again.

### call.lit
`call.lit` [...], 624 params, 69 nonzero. **What:** per (intent, calls done, step): literal token to emit: <tool>, tool name, </tool>. **Why these values:** step 0 non-zero means "make another call"; zero means "start the reply".

### call.slot
`call.slot` [...], 12480 params, 36 nonzero. **What:** per (intent, calls done, step): which value-vector entry to copy as an argument. **Why these values:** e.g. read gets F1 (the user-named file) or HIT (a grep hit); delete gets PREVF (the confirmed file).

### tmpl.lit
`tmpl.lit` [...], 696 params, 152 nonzero. **What:** per (reply type, token index): literal word of the fixed sentence. **Why these values:** the exact reply templates of the spec, one token per column.

### tmpl.slot
`tmpl.slot` [...], 13920 params, 36 nonzero. **What:** per (reply type, token index): which value-vector entry fills a {slot}. **Why these values:** {f}->F1, {k}->K1, {v}->V1 or VAL, {w}->W, {r}->RES0.

### tmpl.k0
`tmpl.k0` [...], 29 params, 29 nonzero. **What:** token index where the copy of the newest result starts (99 = never). **Why these values:** list/content replies copy the tool result then end with ".".

### rt.map
`rt.map` [...], 3770 params, 130 nonzero. **What:** per (intent, result status) -> which reply type. **Why these values:** status of the newest result: normal / missing / none / error / key-absent decides between e.g. lookup and "has no".

### st.ids
`st.ids` [...], 3 params, 3 nonzero. **What:** ids of the result words missing / none / error. **Why these values:** first token of the newest result is compared to these.

### foc.f
`foc.f` [...], 520 params, 26 nonzero. **What:** NEW: per intent, which value becomes the focus file written into think. **Why these values:** the file the turn was about (named file, or the grep hit); intents without a file carry the old focus forward (FF).

### foc.k
`foc.k` [...], 520 params, 26 nonzero. **What:** NEW: per intent, which value becomes the focus key written into think. **Why these values:** the key just asked about; lookup_other and intents without a key carry the old focus key forward (FK).
