# builds EXPLANATION.md group sections from weights.py (run with: uv run python gen_docs.py)
import re
ns = {}
exec(open('weights.py').read(), ns)
cfg, p, groups, words = ns['build']()
FULL = {
 'shift.': 'Previous-token style head. Query = position code rotated back by k steps (k = 1..4) on 16 sin/cos pairs of the learned position table; key = position code. The score peaks at exactly t-k, so the value (selected source channels) is copied one or more positions to the right as a "p1:" channel. All "previous token is X" conditions in the rules come from these. Breaks only if two positions get equal scores (not seen).',
 'shiftcode.': 'Same shift-by-1 head but the value is a 28-dim +-1 code of the previous token (every byte and word has a code from a seeded greedy near-orthogonal packing, max |cos| about 0.5). These codes are the keys for induction copying.',
 'latest.': 'Latest-of-class head. Key score = 1e6 for tokens of the class (and for BOS/USER) plus a recency term (3e5 * sin(w t - 0.3), w about 1e-4), so the newest flagged token wins by e^30 per token. USER tokens are "null" keys, which makes the flag turn-local: an older turn loses to the newest USER. Gated copy heads drop the null keys and add a BOS sink whose score is lowered by the mode channel, so the head only writes when its mode flag is on. Value = a flag channel (L_x) or the token code of the winner (the copy).',
 'induct.': 'Induction copy head. Query = code(current token) and code(previous token); key = codes of the two tokens before the source position; plus a class term (letters, digits, path chars...) and a BOS sink. Exact 2-gram match + class = 180k, 1-gram + class about 120k, sink = 150k, so the copy continues while the source successor is in class and the head falls silent at the end of the span. Output = token code written to the emit channels; the word->operator map (plus -> +) sits in the value matrix. Breaks on repeated bigrams (e.g. 11, or te in notes/tell): the latest exact match wins.',
 'rules.': 'MLP conjunction rules. Each rule uses a twin pair of ReLU neurons: weights 1 on every condition channel (negated conditions -2), thresholds (n-1)U+U/4 and +U/2 further; their difference is a saturating 0/U gate, so features stay clean. The output either writes a feature channel (flag/mode) or adds U*val*code(token) to the emit channels (val 1.0 = logit 20, 0.5 = fallback 10, 1.5 = override 30). Frame text is a chain of such rules, each keyed on the last 1..5 tokens (positions are "p1:" channels), so a stored frame is at most 6 words and slots are filled by copy heads.',
 'unembed': 'Decodes the 28-dim emit code: logit(v) = 20 * cos(code(v), emitted code). Greedy argmax therefore picks the emitted token; END has a bias of 3 so an empty turn ends.',
 'b_unembed': 'END logit 3 = default stop when no rule fires.',
 'embed': 'One-hot token channel (amplitude 32) plus class flags (digit, operator, name letter, path char, ...). Classes are the lexicon: add a token to a class by setting one embed entry.',
 'pos': 'Learned position table: 16 sin/cos pairs for shift heads and one slow pair for recency. Everything else is zero, so no positional noise reaches the feature channels.',
}
out = ['# EXPLANATION (agent x500)\n']
out.append('''## How the model works (about 15 lines)
1. It is a standard 8-layer causal transformer (d=%d, 4 heads x 64, MLP 256, learned position table). Every weight was written from formulas; nothing was trained.
2. The residual stream holds named channels: one-hot tokens, token classes, "previous token" copies (p1: channels), flags (L_*), mode switches (m_*), and a 28-dim emit code.
3. Layer 0 heads fetch the previous token(s) and the newest token of each keyword class in the current turn (digit, operator, greeting, secret, delete...).
4. MLP rules turn those into intents (calc, greet, fact, name, read, delete, ...). Intents are flags that persist through the reply, because the newest USER token never changes.
5. Reply text is composed token by token. A frame is a chain of rules keyed on the last few tokens; each rule emits the next token by writing its code to the emit channels. Frames are at most 6 words.
6. Slots (a copied name, expression, file path, number, relation word) are filled by gated copy heads: a first-token head finds a flagged start position, then an induction head continues the span until it leaves its character class.
7. Facts (capital of a country) are rules on two flags (relation, entity) that emit the value word. The same flags feed the frame "The capital of X is Y." and the unknown path.
8. Arithmetic uses the calc tool: the model writes a JSON tool call, copying the expression from the user text (with plus/minus/times translated), then reads the RESULT number back by copying.
9. Unknown questions fall to the fallback frame "I don't know that yet." (emit strength 0.5, so any real intent beats it).
10. Safety lives in negations: secrets / ".." flags block the read rule and trigger a refusal frame; delete needs "yes" in the same message and no "no", otherwise it asks to confirm or cancels; text inside RESULT never raises a flag because flags are keyed to user tokens.
11. Memory: the name head has no turn-boundary key, so it finds the newest "I am X / name is X" anywhere in context; later corrections win by recency.
12. Limits: it only knows 20 capitals, calc, names, a few social phrases, exact-word echo and file read/delete. Repeated bigrams break copying (e.g. 11, te in notes/tell).
''' % cfg['d_model'])
out.append('\n## Where things live\n- Language: word list (%d words) + class rows of `embed` + frame rule chains (`rules.social.*`, `rules.fact.frame`, `rules.name.*`).\n- Facts: `rules.fact.known` and `rules.fact.value` (two neurons per fact, in layers 3-4).\n- Memory: head `latest.name.first` (no USER null key) + `induct.name.cont`.\n- Safety: `rules.safe.*`, negation lists on `rules.tool.*`, `rules.tool.delete`.\n- Tools: `rules.calc.*`, `rules.tool.*`.\n' % len(words))
out.append('\n## Edit map\n- Add a capital: append (Country, City) to `CAPITALS` in `modules()`; it registers both words, adds the class rows, the E_ flag value row in head `ent1` (max 64 entities per head) and two rules.\n- Add a word: call `wid(S, "word")`; to make it a keyword add it to a `tokclass` list.\n- Add a frame: `frame(S, tag, trigger_conds, parts, tagconds, negs)`; parts are strings (words if registered, else bytes); keep it at most 6 words.\n- Add a tool: copy the `tool.read` block: a JSON prefix chain, a start-flag rule, a mode rule, a close chain.\n- Add an intent: one flag head or a class in an existing `flag.*` head (`latest(...)`), one rule setting a flag, one `frame`. Add the new flag to `negs` of `fallback`.\n')
out.append('\n## Limits\nNo free-form generation, explanation, ordering, syllogisms, word problems, uppercase, lists or supplied-fact answers. Copying fails on repeated bigrams inside a span. "Done." after delete does not check whether the tool succeeded. The model is a rule machine in transformer form, not a language model.\n')
out.append('\n## Group sections\nEvery group name from `weights.py` follows. Names carry layer/head (L#h#) or rule tag.\n')
seen = {}
for g in groups:
    key = None
    for k in FULL:
        if g['name'].startswith(k) or g['name'] == k:
            key = k
            break
    body = FULL.get(key, g['mechanism'])
    out.append('\n### %s\n%s Tensor `%s`. Why: %s.\n' % (g['name'], body, g['tensor'], g['why']))
open('EXPLANATION.md', 'w').write(''.join(out))
print(len(groups))
