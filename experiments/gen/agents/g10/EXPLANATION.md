# GenGPT g10 - explanation

## How GenGPT works
The model reads the whole conversation and writes one token at a time. It has six layers; each layer is "attention" (look at other tokens and copy or count something) followed by a small "logic" block of ReLU units that combines yes/no features.
1. Every token carries a 12-bit +-1 code of its id (so any token can be copied and compared) plus hand-written class flags (file name, key, number, value word, operator words, yes/hello/thanks/bye/delete cues, path characters). Table: `embed`.
2. Layer 0 learns where it is: how many tool results are finished in this turn (`nres`), the previous 1-3 tokens (`p1*`,`p2*`,`p3c`), and the position of the current step start (`Pst`) and last `<result>` (`Pre`). Its logic block keeps only the user's own words (`u_*`), one-hot `nh` (tool steps done) and `jh` (index inside the current step).
3. Layer 1 counts the user's words (`n_*`), fetches the first/last file, key, number, value word and last word (`UF1,UF2,UK1,UK2,UN1,UN2,UW,UV`) and reads whether the last bot reply was a delete question (`pend`, `PFc`). Logic: 0/1 booleans (`bF1`, `bK1`, ...).
4. Layer 2 looks into tool results: the value on the line that starts with the key (`LVa/LVb/LVc`), the file just read (`FREAD`), the first token of the latest result (`Gc`, with missing/none/error flags), the last "the answer is R" (`LAST`) and the aligned token for copying a result into the reply (`CPc`). Logic: the intent (`I_*`, mutually exclusive rules over booleans).
5. Layer 3 picks a template `T_*` from (intent, tool steps done, result branch). Layer 4 emits the template's fixed word at step index j or raises a slot flag `S_*` ("now write the key"). Layer 5 copies the slot's token code into `out`. The unembedding turns `out` (12 bits) into the token.
Reply and tool call are both "templates": a list of literal words and slots, e.g. `tl_read1 = <tool> read UF1 </tool>`, `r_lookup = the UK1 in FREAD is LVa .`. The tool result is appended by the harness, which moves the model to the next step (nres+1).

## How it stays safe
- escape: any `/`, `..` or `~` token in the user turn sets `bPATH` (layer 1 booleans) which makes intent `unsafe` (rule `m2.intent_rules`); template `r_unsafe` has no tool token, so no tool call can start.
- unrequested write/delete: `tl_write` needs the `write` intent (set/change shape: one file, one key, a value, no operator word). `tl_delete` needs `pend` (the previous bot reply ended in `?`, i.e. it asked) AND a yes word. A delete request only produces `r_confirm`.
- injection: `u_*` classes count only tokens before `<bot>` of the current turn (`nbot`), and every fetch head is restricted to the current turn or to specific result tokens. Words in results are only copied/looked up, never counted by the intent logic, so a `note ...` line cannot trigger a tool. Lookup requires the key to be the first token of a line (lag-2 token is `;` or `<result>`).

## Edit map
- Change a fixed sentence: edit the word list in `TEMPL` (weights.py), e.g. `r_bye`. Each word is one unit in layer 4 (`m4.emit`, `l4.w_in/b_in/w_out`); the builder regenerates them. Keep the same number of tokens or the `jh` index range (20) must cover it (`jh` has 20 slots; extend `("jh",20)` in the channel list and the `m0.j_onehot` loop).
- Add a keyword/phrasing: add the word to `LEX` (weights.py) in the right class (`f_opP`, `f_del`, `f_yes`, `f_ls`, ...) or to `FILES/KEYS/VALS`. Only the `embed` row changes.
- Add a protected file `P`: make intent rules never name it: add a flag `f_prot` for token P in `embed`, count it as `path` (put P in `LEX["f_path"]`). Then any request naming it takes the `unsafe` rule (no tool call, refusal). Grep hits are real tool output, so for `grep`/`ls` results you would also need to suppress copying P (not built).
- Add a tool `foo A`: (1) add `TEMPL["tl_foo"] = TL("foo", ("v","UF1"))` and a reply template `r_foo`; (2) add an intent `foo` in `INTENTS` and its rule in `INT`; (3) add RULES entries `tl_foo` (intent, nh(0)) and `r_foo` (intent, nh(1)); (4) add the tool name word to the vocab flags nothing else needed. Every new template needs `add("T_..")` which the loop does automatically.
- Add a reply branch: add a flag channel from the `first_of_result` head (`Gc` dims 12-14) and use it in `RULES`.

## Limits
About 1.6M parameters (most are zeros in dense blocks), float32-safe (largest score about 3e4). Known gaps: only the phrasings listed in `LEX` and structure (counts of files/keys/numbers/value words) are understood; "subtract B from A" is detected by the words subtract/from; follow-ups after a calc error reuse the last good answer; a reply longer than 20 tokens needs more `jh` slots; two operators in one sentence confuse the OP slot; `grep` result sets are copied as printed.

## Group sections (generated from weights.py; each name is a group)

### io.embedding
Tensors: embed.
- What: row of token v = 12-bit code (+-1) and hand-written class flags (file, key, number, value word, operator words, cues, specials)
- Why these values: the code makes every token copyable and comparable by dot product (match 12, any other <=10); flags are the lexicon. <bos>/<end> have code 0 so they never match a key
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### io.embedding_proj
Tensors: embproj.
- What: identity: static token features occupy the first K residual channels
- Why these values: keeps the embed table small (V x K instead of V x d)
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### io.position
Tensors: pos_dir.
- What: writes the integer position t into channel posv
- Why these values: heads that need the position of a flagged token read posv
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### io.turn_counter
Tensors: seg_in.
- What: turn id = running count of <user> tokens; used by 'local' heads
- Why these values: makes 'same turn' exact
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### io.big
Tensors: big.
- What: mask constant 1e9
- Why these values: causal mask and cross-turn penalty; any value >> score range works
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### io.unembed_proj
Tensors: out_proj.
- What: reads the 12 out channels with gain 5
- Why these values: logit of the intended token = 5*12=60, any other token <=50: margin 10
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### io.unembed_code
Tensors: code_table.
- What: logit(v) = sum_b out_b * bit_b(v)
- Why these values: binary code of the token id; dot with out=+-code(t) is maximal only for t
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.0.count_turn
Tensors: l0.dvec, l0.hp, l0.wk, l0.wo, l0.wq, l0.wv.
- What: counting head: how many <bot> and </result> tokens so far in this turn
- Why these values: ssum=1,local=1 sums the value over past tokens of the same turn; tells where in the turn we are (nbot) and how many tool steps are done (nres)
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.0.prev1
Tensors: l0.dvec, l0.hp, l0.wk, l0.wo, l0.wq, l0.wv.
- What: previous-token head (lag 1): copies code and a few flags of token t-1
- Why these values: A=20,shift=1 puts weight e^20 on the key exactly one step back
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.0.prev2
Tensors: l0.dvec, l0.hp, l0.wk, l0.wo, l0.wq, l0.wv.
- What: lag-2 head: flags of token t-2 (tool, ';', <result>, '?', 'answer')
- Why these values: A=20,shift=2
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.0.prev3
Tensors: l0.dvec, l0.hp, l0.wk, l0.wo, l0.wq, l0.wv.
- What: lag-3 head: code of token t-3 (the file in 'delete F ?' seen from the next <user>)
- Why these values: A=20,shift=3
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.0.step_start
Tensors: l0.dvec, l0.hp, l0.wk, l0.wo, l0.wq, l0.wv.
- What: position of the latest <bot>/</result> token (written to Pst)
- Why these values: key gain 16000 beats recency R*s (<=12*1023); R=12 makes the latest flagged token win by e^36 over the previous one
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.0.result_start
Tensors: l0.dvec, l0.hp, l0.wk, l0.wo, l0.wq, l0.wv.
- What: position of the latest <result> token (written to Pre)
- Why these values: key gain 16000 beats recency R*s (<=12*1023); R=12 makes the latest flagged token win by e^36 over the previous one
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m0.in_user
Tensors: l0.b_in, l0.w_in, l0.w_out.
- What: in_user = 1 until <bot> of this turn
- Why these values: relu(1-nbot): nbot is 0 for user tokens, 1 from <bot> on
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m0.user_class
Tensors: l0.b_in, l0.w_in, l0.w_out.
- What: u_X = class flag X AND token is in the user part of the turn
- Why these values: relu(f-nbot) = AND(f, in_user) because in_user = 1-nbot (read nbot directly: same-layer units cannot read each other), so bot-side tokens (file names in tool args) are not counted as user words
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m0.nres_onehot
Tensors: l0.b_in, l0.w_in, l0.w_out.
- What: nh[r] = 1 iff nres == r (tent function of an integer)
- Why these values: relu(n-r+1)-2relu(n-r)+relu(n-r-1) is exactly 1 at n=r and 0 at every other integer
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m0.j_onehot
Tensors: l0.b_in, l0.w_in, l0.w_out.
- What: jh[k] = 1 iff j==k where j = pos - Pst = index inside the current step (jg3 = j>=3)
- Why these values: same tent trick on the integer j=posv-Pst; jg3=relu(j-2)-relu(j-3) reuses two of the k=3 units
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.1.count_user_words
Tensors: l1.dvec, l1.hp, l1.wk, l1.wo, l1.wq, l1.wv.
- What: counting head over the user's words (gated class flags u_X) -> n_X counts
- Why these values: an exact count (not a mean), so 'two files named' is a threshold on an integer
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.1.pending
Tensors: l1.dvec, l1.hp, l1.wk, l1.wo, l1.wq, l1.wv.
- What: reads this turn's <user> token: did the previous bot reply end in '?' (pending delete) and which file
- Why these values: only <user> has f_user; its lag-2 flag p2_q and lag-3 code p3c were computed in layer 0
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.1.first_file
Tensors: l1.dvec, l1.hp, l1.wk, l1.wo, l1.wq, l1.wv.
- What: UF1: code of the first file name the user wrote
- Why these values: key gain 4000.0 on the class flag; R=-4 (negative = earliest, positive = latest) inside the turn
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.1.last_file
Tensors: l1.dvec, l1.hp, l1.wk, l1.wo, l1.wq, l1.wv.
- What: UF2: code of the last file name the user wrote
- Why these values: key gain 4000.0 on the class flag; R=4 (negative = earliest, positive = latest) inside the turn
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.1.first_key
Tensors: l1.dvec, l1.hp, l1.wk, l1.wo, l1.wq, l1.wv.
- What: UK1: first key word
- Why these values: key gain 4000.0 on the class flag; R=-4 (negative = earliest, positive = latest) inside the turn
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.1.last_key
Tensors: l1.dvec, l1.hp, l1.wk, l1.wo, l1.wq, l1.wv.
- What: UK2: last key word
- Why these values: key gain 4000.0 on the class flag; R=4 (negative = earliest, positive = latest) inside the turn
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.1.first_num
Tensors: l1.dvec, l1.hp, l1.wk, l1.wo, l1.wq, l1.wv.
- What: UN1: first number
- Why these values: key gain 4000.0 on the class flag; R=-4 (negative = earliest, positive = latest) inside the turn
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.1.last_num
Tensors: l1.dvec, l1.hp, l1.wk, l1.wo, l1.wq, l1.wv.
- What: UN2: last number
- Why these values: key gain 4000.0 on the class flag; R=4 (negative = earliest, positive = latest) inside the turn
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.1.first_val
Tensors: l1.dvec, l1.hp, l1.wk, l1.wo, l1.wq, l1.wv.
- What: UW: first value word (name/status/region/level word)
- Why these values: key gain 4000.0 on the class flag; R=-4 (negative = earliest, positive = latest) inside the turn
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.1.last_word
Tensors: l1.dvec, l1.hp, l1.wk, l1.wo, l1.wq, l1.wv.
- What: UV: the last content word of the user turn (the value in 'set K in F to V')
- Why these values: key gain 4000.0 on the class flag; R=4 (negative = earliest, positive = latest) inside the turn
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m1.booleans
Tensors: l1.b_in, l1.w_in, l1.w_out.
- What: b = [count >= k]
- Why these values: relu(n-k+1)-relu(n-k) is the exact step for integer counts
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.2.lookup_latest
Tensors: l2.dvec, l2.hp, l2.wk, l2.wo, l2.wq, l2.wv.
- What: value of key UK1 on a result line that STARTS with it (token after K whose lag-2 token is ';' or <result>) -> LVa; fu = 1 if not found
- Why these values: score = 8*code match(<=96) + 16 if the key is the line start + 16 if in the wanted result (latest/first) = 128 for a true match; the <user> sink has 120, a near miss at most 112
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.2.lookup_second_key
Tensors: l2.dvec, l2.hp, l2.wk, l2.wo, l2.wq, l2.wv.
- What: value of key UK2 on a result line that STARTS with it (token after K whose lag-2 token is ';' or <result>) -> LVb; fu = 1 if not found
- Why these values: score = 8*code match(<=96) + 16 if the key is the line start + 16 if in the wanted result (latest/first) = 128 for a true match; the <user> sink has 120, a near miss at most 112
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.2.lookup_first_result
Tensors: l2.dvec, l2.hp, l2.wk, l2.wo, l2.wq, l2.wv.
- What: value of key UK1 on a result line that STARTS with it (token after K whose lag-2 token is ';' or <result>) -> LVc; fu = 1 if not found
- Why these values: score = 8*code match(<=96) + 16 if the key is the line start + 16 if in the wanted result (latest/first) = 128 for a true match; the <user> sink has 120, a near miss at most 112
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.2.read_arg
Tensors: l2.dvec, l2.hp, l2.wk, l2.wo, l2.wq, l2.wv.
- What: FREAD: the file argument of the latest read call (token after 'read' that follows <tool>)
- Why these values: both lag flags needed: 2*4000.0 beats a single one
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.2.first_of_result
Tensors: l2.dvec, l2.hp, l2.wk, l2.wo, l2.wq, l2.wv.
- What: G: first token of the latest result (+ whether it is missing/none/error)
- Why these values: key = lag-1 token is <result>
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.2.last_answer
Tensors: l2.dvec, l2.hp, l2.wk, l2.wo, l2.wq, l2.wv.
- What: LAST: the number in the most recent 'the answer is R' of the whole conversation
- Why these values: key = lag-1 'is' and lag-2 'answer'; global scope; R=2
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### head.2.copy_result
Tensors: l2.dvec, l2.hp, l2.wk, l2.wo, l2.wq, l2.wv.
- What: CP: token of the latest result aligned with the reply token being generated (lag = Pst-Pre+2) + whether it is </result>
- Why these values: reply content starts 3 tokens into the step; source index = t-(Pst-Pre+2); A=20 pins exactly that key
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m2.intent_rules
Tensors: l2.b_in, l2.w_in, l2.w_out.
- What: I_x = AND of boolean features (user words + pending-confirm) selecting what the user wants
- Why these values: relu(sum(lits)-(m-1)) is an exact AND of 0/1 literals; the rules are mutually exclusive by construction (see EXPLANATION.md table)
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m3.template_rules
Tensors: l3.b_in, l3.w_in, l3.w_out.
- What: T_x = step (intent x tool-steps-done x result branch) selects reply/tool template x
- Why these values: AND of intent flag, one-hot nres and branch flags from the latest result; exact because all inputs are 0/1
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m3.unknown_default
Tensors: l3.b_in, l3.w_in, l3.w_out.
- What: unknown reply when no intent fired
- Why these values: relu(nh0 - sum I): exactly 1 only when no intent is on
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m4.emit
Tensors: l4.b_in, l4.w_in, l4.w_out.
- What: at step index j of template T emit literal token (writes its 12-bit code to out) or raise a slot flag S_var
- Why these values: relu(T+jh-1) is the AND of 'in this template' and 'at this index'; templates are exclusive so exactly one unit fires
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m4.end
Tensors: l4.b_in, l4.w_in, l4.w_out.
- What: after '.' or '?' emit <end>
- Why these values: f_endp (current token is . or ?) directly writes the code of <end>
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m5.gate_vars
Tensors: l5.b_in, l5.w_in, l5.w_out.
- What: out += code of the slot variable when its slot flag S is on
- Why these values: relu(sgn*c+2S-2) passes the +-1 code bit only when S=1 (S=0 leaves <=-1 -> 0); the + and - units give back the signed bit
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m5.operator
Tensors: l5.b_in, l5.w_in, l5.w_out.
- What: OP slot emits plus/minus/times according to the operator word the user used
- Why these values: AND of S_OP and the boolean for that operator
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.

### m5.copy_end
Tensors: l5.b_in, l5.w_in, l5.w_out.
- What: when the copy source is </result> emit '.'
- Why these values: AND of S_COPY and cp_end
- Breaks when: the input pattern it assumes (see the mechanism) does not hold, e.g. a phrasing whose words are not in LEX or a result longer than 20 tokens.
