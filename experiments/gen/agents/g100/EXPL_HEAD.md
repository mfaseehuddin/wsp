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
