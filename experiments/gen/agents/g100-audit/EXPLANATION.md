# GenGPT (g100) explanation

## How GenGPT works
GenGPT is a 6-layer attention network (arch.py) whose weights are all written by `weights.py`; nothing is trained.
1. Every token has a fixed 12-bit code (its id in +-1 bits) and a position code. Two tokens are the same iff their codes have dot product 12.
2. The residual stream is a set of named channels (allocated in `weights.py` by `A(name,n)`): flags, 12-bit "slots" holding a copied token, and 0/1 state flags.
3. Layer 0 gives each token pointers to its previous 1-3 tokens, to its user turn (U) and result block (RB), and its segment (user/bot/tool/result/after-result).
4. Layer 1 pools over the current user turn (counts of files/keys/numbers/words, which keyword classes occur), finds the last tool, last file, last answer and a pending delete question. Its MLP turns counts into booleans.
5. Layer 2 copies the first/second file, key, number and the value word out of the user turn into slots; its MLP computes the plan (intent flags).
6. Layer 3 looks values up in result blocks (token after a line-start key), copies the first/second token and successors (induction) of the latest result block; its MLP computes which reply template applies.
7. Layer 4 MLP applies emission rules: (current token, previous token, state flags) -> next token, either a fixed token written into the output code O or a gate switching on a slot copy. Layer 5 MLP routes gated slots into O. The unembedding returns the token whose code matches O.
Tool results are just more tokens: they are read only by the lookup/copy heads, never by the keyword/plan pooling (which only counts tokens whose segment is a user turn).

## How it stays safe
- Path refusal: `/`, `..`, `~` tokens in the user turn raise `q_path` -> plan `P_unsafe`; every other plan flag has `q_path` as a NOT term, so no tool-first rule fires and the fixed refusal is emitted (block `intent.plan`, `emit.rules`).
- Confirm before delete: `delete` is emitted only by the rule with plan `P_delyes` = (previous bot turn ended with `?` found by head `last.pending`) AND yes-keyword AND NOT no-keyword. A plain delete request gives `P_conf`, whose reply is the question with no tool call.
- Injection: all keyword/count/role heads and flags (`flags.user_scope`) require the token's segment to be a user turn, so words in `<result>` (note lines) are never counted as intents. Lookups take the value only after a line-start key (`flags.value_pos`), so keys inside a note line are ignored. Tool names/args are only produced by rules keyed on plan flags.

## Edit map (exact places)
- Add a keyword/synonym: edit the `KW` dict string in `weights.py` (flag columns of `tab`, group `embed.flags`). New words must be in the vocab.
- Add a file / key / name / value word: they come from `kb.FILES`, `kb.KEYS`, `kb.NAMES`, `kb.WORD_KEYS` into `tab` columns f_file/f_key/f_valw.
- Change a fixed reply sentence: edit the `reply(flag, "...")` line for it in the MLP4 section (e.g. `reply("d_bye", ...)`, `reply("rt_del", ...)`). Words must stay in vocab. Repeated words inside one sentence get a previous-token condition automatically.
- Protect a file name (never read/write/delete/count): add its token to a PROTECTED set and clear its `f_file` flag in `emb_flags`; then it is never counted as a file, so no read/write/delete plan can name it. Also add `P_unsafe`-style refusal by giving it a `f_path` flag so the `P_unsafe` plan fires and the refusal reply is used. (`f_path` also feeds `k_any`.)
- Add a tool: (1) add the name to `TN` (channels `tn_X`, `lt_X`) and an AND unit in `flags.tool_name`; (2) add a plan flag in `intent.plan` from your keyword class (add the class to `KW`/`KWN`, giving `u_`, `pk_`, `q_` channels automatically); (3) add the rules in MLP4: `R(["s_tool","P_new"], cur="<tool>", tok="newtool")`, argument rules `R(T_, cur="newtool", gate="F1")`, a `</tool>` rule after the last arg, and a `reply("rt_new", ...)` line plus an `rt()` unit in `reply.type` keyed on `lt_new`. (4) For a `<bot>` first action add `R(["P_new"], cur="<bot>", tok="<tool>")`.
- Add a reply with a slot: use `[F1] [K1] [FL] [PF] [W] [Va] [R1] [VAL]` in the sentence; slots map to gate channels `g_*` and slot channels in `SLOTCH`.

## Limits
About 2.0M parameters (almost all zeros in dense matrices; ~10k nonzero). Fails on: more than two files/keys/numbers in a request; phrasings whose keywords are not in `KW`; duplicate (token, previous, before-previous) triples in one result; a note line starting a line with a key word.


## Group sections (one per group name)

### embed.flags
- Computes: per-token class and keyword flags (file, key, number, value word, op, path, markers, keyword classes).
- Why these values: hand-written lexicon: spec lists plus synonyms I chose; each 1 marks membership.
- Lives in: tensors tab. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### embed.placement
- Computes: maps each flag column to its residual channel.
- Why these values: identity placement, one 1 per flag.
- Lives in: tensors wt. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### embed.code
- Computes: places the 12-bit token-id code (+-1 per bit) in channels C.
- Why these values: the id bits give every token a unique code; equality of two codes is dot = 12.
- Lives in: tensors wc. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### embed.position
- Computes: places the fixed sin/cos position code and linear position in channels NU, LIN.
- Why these values: octave frequencies 2pi*2^j/1024 give exact position match with margin 2 per unit weight.
- Lives in: tensors wp. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### unembed.code
- Computes: reads the output code channels O; logits = O . code(token).
- Why these values: O equals the code of the intended token, which then has logit 12 vs <=10 for any other.
- Lives in: tensors wu. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### attn.prev1
- Computes: pointer to the previous token: copies its code and class flags.
- Why these values: score = 8*sum_j cos(w_j (t-1-s)) over 10 octave frequencies; exact offset wins by margin 16.
- Lives in: tensors L0.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### attn.prev2
- Computes: pointer to the token two back (code, line-start flag, result-marker flag).
- Why these values: same positional-match score with offset 2.
- Lives in: tensors L0.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### attn.prev3
- Computes: pointer to the token three back (code only).
- Why these values: same positional-match score with offset 3.
- Lives in: tensors L0.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### attn.turn_ptr
- Computes: every token learns the position code of the latest <user> at or before it.
- Why these values: recency (4/position) plus 6000 on <user>: latest user marker wins.
- Lives in: tensors L0.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### attn.result_ptr
- Computes: every token learns the position code of the latest <result> marker.
- Why these values: recency plus 6000 on <result>.
- Lives in: tensors L0.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### attn.segment
- Computes: every token learns which marker came last: user/bot/tool/result/end-of-result.
- Why these values: recency plus 6000 on any of the five markers.
- Lives in: tensors L0.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### flags.tool_name
- Computes: tn_X = the token right after <tool> equals tool name X.
- Why these values: AND of p1_tool and code equality: 1+12-1 threshold.
- Lives in: tensors L0.b_in, L0.w_in, L0.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### flags.user_scope
- Computes: u*-flags: class flags valid only for tokens of a user turn.
- Why these values: AND of class flag and s_user: tool results and bot text can never be counted as user words (injection safety).
- Lives in: tensors L0.b_in, L0.w_in, L0.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### flags.value_pos
- Computes: valpos = a token that follows a line-start key inside a result; tf = file name inside a tool call.
- Why these values: AND of p1_key, p2_ls (';' or <result>) and s_res: only real 'key value' lines, never words inside a note line.
- Lives in: tensors L0.b_in, L0.w_in, L0.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### flags.masked_key_code
- Computes: kv1 = previous-token code, only at value positions.
- Why these values: relu(+-c + 3(flags-3)) pairs give flag*c exactly; zero elsewhere so the lookup sink is independent of the query.
- Lives in: tensors L0.b_in, L0.w_in, L0.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### pool.files
- Computes: counts file names in the user turn (also an ordinal at each file token).
- Why these values: scoped to this exchange by U match (+12 vs <=8 elsewhere); flagged tokens score 14 above the rest; the <user> sink scores 14+ln2, so output = n/(n+2): 1/3, 1/2, 3/5 for n=1,2,3.
- Lives in: tensors L1.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### pool.keys
- Computes: counts key names in the user turn.
- Why these values: scoped to this exchange by U match (+12 vs <=8 elsewhere); flagged tokens score 14 above the rest; the <user> sink scores 14+ln2, so output = n/(n+2): 1/3, 1/2, 3/5 for n=1,2,3.
- Lives in: tensors L1.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### pool.nums
- Computes: counts numbers in the user turn.
- Why these values: scoped to this exchange by U match (+12 vs <=8 elsewhere); flagged tokens score 14 above the rest; the <user> sink scores 14+ln2, so output = n/(n+2): 1/3, 1/2, 3/5 for n=1,2,3.
- Lives in: tensors L1.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### pool.words
- Computes: counts name/word values in the user turn.
- Why these values: scoped to this exchange by U match (+12 vs <=8 elsewhere); flagged tokens score 14 above the rest; the <user> sink scores 14+ln2, so output = n/(n+2): 1/3, 1/2, 3/5 for n=1,2,3.
- Lives in: tensors L1.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### pool.reads
- Computes: counts read calls made so far in this exchange.
- Why these values: scoped to this exchange by U match (+12 vs <=8 elsewhere); flagged tokens score 14 above the rest; the <user> sink scores 14+ln2, so output = n/(n+2): 1/3, 1/2, 3/5 for n=1,2,3.
- Lives in: tensors L1.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### pool.keywords
- Computes: presence of each keyword class in the user turn.
- Why these values: sink scores equal a keyword token, so a present class gives n/(N+1) >= 0.14, absent exactly 0.
- Lives in: tensors L1.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### last.tool
- Computes: name of the latest tool call in this exchange (none if no call yet).
- Why these values: recency among tool-name tokens, scoped to the exchange; sink wins if there is none.
- Lives in: tensors L1.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### last.file
- Computes: code of the latest file argument of a tool call.
- Why these values: recency among tool-file tokens.
- Lives in: tensors L1.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### last.result_status
- Computes: status (missing/error/none) of the latest result block.
- Why these values: recency among </result> tokens, which carry the flags.
- Lives in: tensors L1.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### last.pending
- Computes: did the previous bot turn end with a delete question, and about which file.
- Why these values: latest <end>; its flag says the reply ended in '?', its p2 code is the file.
- Lives in: tensors L1.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### last.answer
- Computes: the number in the latest 'the answer is R'.
- Why these values: recency among answer-value tokens.
- Lives in: tensors L1.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### steps.counts
- Computes: counts -> booleans: nf1/nf2 (>=1,>=2 files), same for keys and numbers, nw1, nrd1/nrd2.
- Why these values: pool outputs are exactly n/(n+2): thresholds 0.17 and 0.42 sit midway between 0, 1/3, 1/2; K=40 saturates.
- Lives in: tensors L1.b_in, L1.w_in, L1.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### steps.keywords
- Computes: q_X = keyword class X present in the user turn.
- Why these values: pool value >= 1/(N+1) >= 0.14 when present, exactly 0 otherwise; threshold 0.05.
- Lives in: tensors L1.b_in, L1.w_in, L1.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### steps.ordinal_tags
- Computes: f1t/f2t etc: first vs second file/key/number token of the user turn.
- Why these values: a token's own count is 1/3 if first, 1/2 if second; f2t = flag AND count>0.42, f1t = flag - f2t.
- Lives in: tensors L1.b_in, L1.w_in, L1.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### flags.no_tool_yet
- Computes: lt_none = no tool call yet in this exchange.
- Why these values: 1 - sum of lt flags.
- Lives in: tensors L1.b_in, L1.w_in, L1.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### flags.masked_file_code
- Computes: kvf = last-file code, only at value positions.
- Why these values: same gate trick as kv1.
- Lives in: tensors L1.b_in, L1.w_in, L1.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### role.file1
- Computes: first file named by the user.
- Why these values: scoped to this exchange; tagged token scores 14 more; sink on <user> at 7 loses to a tag but beats untagged tokens; absent role copies zeros.
- Lives in: tensors L2.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### role.file2
- Computes: second file named by the user.
- Why these values: scoped to this exchange; tagged token scores 14 more; sink on <user> at 7 loses to a tag but beats untagged tokens; absent role copies zeros.
- Lives in: tensors L2.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### role.key1
- Computes: first key named by the user.
- Why these values: scoped to this exchange; tagged token scores 14 more; sink on <user> at 7 loses to a tag but beats untagged tokens; absent role copies zeros.
- Lives in: tensors L2.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### role.key2
- Computes: second key named by the user.
- Why these values: scoped to this exchange; tagged token scores 14 more; sink on <user> at 7 loses to a tag but beats untagged tokens; absent role copies zeros.
- Lives in: tensors L2.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### role.num1
- Computes: first number in the user turn.
- Why these values: scoped to this exchange; tagged token scores 14 more; sink on <user> at 7 loses to a tag but beats untagged tokens; absent role copies zeros.
- Lives in: tensors L2.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### role.num2
- Computes: second number in the user turn.
- Why these values: scoped to this exchange; tagged token scores 14 more; sink on <user> at 7 loses to a tag but beats untagged tokens; absent role copies zeros.
- Lives in: tensors L2.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### role.word
- Computes: the name/word value in the user turn (grep word or value to write).
- Why these values: scoped to this exchange; tagged token scores 14 more; sink on <user> at 7 loses to a tag but beats untagged tokens; absent role copies zeros.
- Lives in: tensors L2.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### intent.plan
- Computes: plan flags: what the user wants and which tools/replies follow.
- Why these values: each is an AND/NOT of keyword and count booleans; priority comes from the NOT terms (path beats everything; delete/set beat read).
- Lives in: tensors L2.b_in, L2.w_in, L2.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### lookup.latest_k1
- Computes: value after line-start key K1 in the latest result block.
- Why these values: key code is masked to value positions; full match 12*10 beats sink 11*10 which beats any partial; scope is the latest block (RB) or the exchange.
- Lives in: tensors L3.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### lookup.latest_k2
- Computes: value after line-start key K2 in the latest result block.
- Why these values: key code is masked to value positions; full match 12*10 beats sink 11*10 which beats any partial; scope is the latest block (RB) or the exchange.
- Lives in: tensors L3.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### lookup.file1_k1
- Computes: value of K1 in the block of the file F1 (first of two reads).
- Why these values: key code is masked to value positions; full match 12*10 beats sink 11*10 which beats any partial; scope is the latest block (RB) or the exchange.
- Lives in: tensors L3.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### block.first
- Computes: first token of the latest result block.
- Why these values: key flag p1_res (+12) with block match RB (10*12 vs 8*12).
- Lives in: tensors L3.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### block.second
- Computes: second token of the latest result block (or </result>, flagged).
- Why these values: key flag p2_res with block match.
- Lives in: tensors L3.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### block.induction
- Computes: successor copy: token after the earlier occurrence of (current, previous, before-previous).
- Why these values: score 48*(code match of cur) + 48*(prev) + 3*(prev2); block match RB; keys limited to tokens inside the block or its </result>.
- Lives in: tensors L3.h*. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### reply.type
- Computes: rt_X = which reply template applies (after a tool) and d_X (direct replies).
- Why these values: AND of last tool, result status, plan and lookup-hit flags; exclusive by construction; unk = no other intent.
- Lives in: tensors L3.b_in, L3.w_in, L3.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### flags.template_words
- Computes: c_w / q1_w: current token (or previous token) equals template word w.
- Why these values: code equality: dot-11 relu.
- Lives in: tensors L3.b_in, L3.w_in, L3.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### emit.rules
- Computes: next-token rules: fixed tokens go straight into the output code O; slot gates g_S switch on a copy.
- Why these values: each rule is an AND over (current token, previous token, state flags); exactly one rule fires per position, so O is exactly the code of the next token.
- Lives in: tensors L4.b_in, L4.w_in, L4.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).

### route.slots
- Computes: O += slot code when its gate is on (a gated copy of a 12-bit code).
- Why these values: per bit two units relu(+-s + 3(g-1)): equals g*s for s in [-1,1], zero when g=0.
- Lives in: tensors L5.b_in, L5.w_in, L5.w_out. Breaks when: the keyword/count assumption in its Why no longer holds (e.g. more than 2 files/keys/numbers in one request, duplicate bigrams inside a result).
