"""GenGPT-opus: hand-derived weights. Base: g1000's script interpreter (audited version). Additions, all marked NEW:
- a think register (working memory): each turn ends with <think> focus-file focus-key </think>;
- reference intents that read it: lookup_it ("what is its K ?"), lookup_other ("and in F ?"), show_it, delete_it;
- structural intent rules: value word + key = find-then-lookup, value word alone = find, key alone = focus file,
  so understanding no longer depends on "find/which/where" keywords;
- safety tie-break: a reply with both yes and no words counts as no.

Everything below is a table written from the spec.  Each table is a named parameter tensor and a prov group.
Edit recipes are in EXPLANATION.md (Edit map)."""
import torch
from wsp.gen import tokenizer as T, kb
from wsp.gen.prov import group, S
from wsp.bot.custom import CustomArch

D = torch.float64
V = 4096
tid = T.tid
# ---------------------------------------------------------------- lexicon (hand-written word lists)
COLS = ['user', 'bot', 'tool', 'etool', 'res', 'eres', 'end', 'dot', 'q', 'semi', 'is', 'answer',
        'file', 'key', 'num', 'name', 'wordv', 'opw', 'opid', 'hello', 'thanks', 'bye', 'ls', 'find', 'set',
        'del', 'yes', 'no', 'path', 'rev', 'dbl', 'wtok', 'valtok', 'wc', 'prot', 'think', 'it', 'ell', 'show']
CI = {c: i for i, c in enumerate(COLS)}
CUES = {
    'hello': 'hello hi hey greetings howdy morning evening',
    'thanks': 'thanks thank cheers',
    'bye': 'bye goodbye later farewell',
    'ls': 'list files ls everything folder',
    'find': 'find mentions mention mentioned mentioning search searching contains contain containing grep which where has match',
    'set': 'set change update make write edit modify assign',
    'del': 'delete remove removes erase drop rm rid trash destroy wipe discard purge kill',
    'yes': 'yes yep yeah sure ok okay go ahead proceed confirm confirmed absolutely definitely',
    'no': 'no nope cancel keep stop abort never not dont wait nah hold',
    # NEW (references and structure)
    'it': 'it its that this',                                   # points back to the file in focus
    'ell': 'and about then now what how also same too it its that this',  # ellipsis: "and in F ?" "now F ?"
    'show': 'show read open print cat see display view say says contents content dump',
    'path': '/ .. ~ \\',
    'rev': 'subtract away deduct',
    'dbl': 'double twice',
    'wc': 'lines',
}
# Protected names (stage 1 style): a word listed here counts as a path cue, so any request that mentions it
# is refused with the unsafe reply and NO tool call (never read, written, deleted or counted).
PROTECTED = ['secrets']
CUES['prot'] = ' '.join(PROTECTED)
OPWORDS = {'plus': 'plus', 'add': 'plus', 'sum': 'plus', 'total': 'plus', 'minus': 'minus', 'subtract': 'minus',
           'less': 'minus', 'away': 'minus', 'deduct': 'minus', 'times': 'times', 'multiplied': 'times', 'multiply': 'times', 'product': 'times', 'double': 'times', 'twice': 'times'}
SPECIAL = {'<user>': 'user', '<bot>': 'bot', '<tool>': 'tool', '</tool>': 'etool', '<result>': 'res',
           '</result>': 'eres', '<end>': 'end', '.': 'dot', '?': 'q', ';': 'semi', 'is': 'is', 'answer': 'answer',
           '<think>': 'think'}
vocab = T.vocab()
inv = {w: i for i, w in enumerate(vocab)}


def lexicon():
    rows = {}  # word -> {col: value}
    def add(w, c, v=1.0):
        if w in inv:
            rows.setdefault(w, {})[c] = v
    for w, c in SPECIAL.items(): add(w, c)
    for w in kb.FILES: add(w, 'file')
    for w in kb.KEYS: add(w, 'key')
    for w in kb.NAMES: add(w, 'name'); add(w, 'wtok'); add(w, 'valtok')
    for k, ws in kb.WORD_KEYS.items():
        for w in ws: add(w, 'wordv'); add(w, 'wtok'); add(w, 'valtok')
    for c, ws in CUES.items():
        for w in ws.split(): add(w, c)
    for w, o in OPWORDS.items(): add(w, 'opw'); add(w, 'opid', float(tid(o)))
    return rows


def build():
    rows = lexicon()
    words = sorted(rows)
    lex_id = torch.zeros(V, dtype=D)           # row 0 = "ordinary word": all-zero features
    lex_feat = torch.zeros(len(words) + 2, len(COLS), dtype=D)
    for i, w in enumerate(words, 2):
        lex_id[inv[w]] = i
        for c, v in rows[w].items(): lex_feat[i, CI[c]] = v
    numrow = 1                                  # row 1 = a number token (ids 12..1011)
    lex_id[12:1012] = numrow
    for c in ('num', 'valtok'): lex_feat[numrow, CI[c]] = 1.0
    p, G = {}, []

    def g(name, tensor, index, mech, why):
        p[name] = tensor
        G.append(group(name, name, index, mechanism=mech, why=why))

    g('lex.id', lex_id, S[...], 'token id -> lexicon row (0 = ordinary word, 1 = any number, 2.. = spec words)',
      'a hand-written index; every word of the spec (files, keys, values, tools, cue words) has its own row')
    g('lex.feat', lex_feat, S[...], 'lexicon row -> 33 yes/no features (file? key? yes-cue? ...) read by every later step',
      'feature 1.0 = the word has the property; opid holds the id of plus/minus/times for operator words')
    for c in COLS:
        v = torch.zeros(len(COLS), dtype=D); v[CI[c]] = 1
        g('col.' + c, v, S[...], f'selector that reads feature column "{c}" out of a lexicon row', 'one-hot over the 33 feature columns')

    # ---- constants
    def const(name, val, why):
        g('k.' + name, torch.full((), float(val), dtype=D), S[...], 'constant ' + name, why)
    const('zero', 0, 'zero used for padding'); const('half', 0.5, 'threshold between 0/1 flags')
    const('neg1', -1, 'position of "nothing yet"'); const('onef', 1, 'one'); const('two', 2, 'two (confirm "?" sits 2 tokens before the next <user>)')
    const('qgap', 6, 'NEW: a confirm "?" is followed by <think> F K </think> <end>, so the next <user> sits 6 tokens later')
    const('none', tid('none'), 'NEW: written into the think register when there is no file/key in focus yet')
    const('thk', tid('<think>'), 'NEW: id of <think>'); const('ethk', tid('</think>'), 'NEW: id of </think>')
    g('k.th', torch.arange(5, dtype=D), S[...], 'NEW: think-phase step index', 'steps after the final punctuation: <think>, file, key, </think>, <end>')
    const('tok2', tid('2'), 'id of the number token 2, used by "double"'); const('period', tid('.'), 'id of "."')
    const('end', T.tid('<end>'), 'id of <end>')
    const('cap', 3, 'readout distance is capped at 3 so logits stay within +-120 (float32/fp16 safe)')
    const('gain', 40, 'logit = -40*min(|id-target|,3): the target id scores 0, any other id at most -40, so greedy argmax is exact')
    const('fgain', 20, 'softmax sharpness for the intent choice; priorities differ by >=1 so the winner gets weight ~1')
    g('k.pos', torch.arange(1024, dtype=D), S[...], 'scalar position of each token', 'position index 0..1023')
    g('k.vid', torch.arange(V, dtype=D), S[...], 'scalar id of each vocabulary entry (readout number line)', 'vocabulary ids 0..4095')
    g('k.jr', torch.arange(6, dtype=D), S[...], 'call-step index range', 'a call has at most 6 steps: <tool> name a0 a1 a2 </tool>')
    g('k.cr', torch.arange(4, dtype=D), S[...], 'finished-call-count range', 'at most 3 calls per turn')
    g('k.kr', torch.arange(24, dtype=D), S[...], 'reply token index range', 'longest reply (hello) has 20 tokens')

    # ---- intent (flow) selection
    BAG = ['hello', 'thanks', 'bye', 'ls', 'find', 'set', 'del', 'yes', 'no', 'path', 'rev', 'dbl', 'file', 'key', 'num', 'wtok', 'opw', 'wc', 'prot', 'it', 'ell', 'show']
    bag_sel = torch.zeros(len(COLS), len(BAG), dtype=D)
    for j, b in enumerate(BAG): bag_sel[CI[b], j] = 1
    g('bag.sel', bag_sel, S[...], 'which lexicon features are summed over the current user turn (bag of words counts)', 'one-hot columns of the 17 counted features')
    rev = torch.zeros(len(BAG), dtype=D); rev[BAG.index('rev')] = 1
    g('bag.rev', rev, S[...], 'reads the "subtract" count', 'subtract B from A flips the operand order')
    # indicators: (name, bag feature, min count)
    IND = [('hello', 'hello', 1), ('thanks', 'thanks', 1), ('bye', 'bye', 1), ('ls', 'ls', 1), ('find', 'find', 1), ('set', 'set', 1),
           ('del', 'del', 1), ('yes', 'yes', 1), ('no', 'no', 1), ('path', 'path', 1), ('wc', 'wc', 1), ('prot', 'prot', 1), ('dbl', 'dbl', 1),
           ('F1', 'file', 1), ('F2', 'file', 2), ('K1', 'key', 1), ('K2', 'key', 2), ('N1', 'num', 1), ('N2', 'num', 2),
           ('W1', 'wtok', 1), ('OP', 'opw', 1), ('it', 'it', 1), ('ell', 'ell', 1), ('show', 'show', 1), ('pend', None, 1)]
    ind_sel = torch.zeros(len(BAG) + 1, len(IND), dtype=D); ind_thr = torch.zeros(len(IND), dtype=D)
    for j, (n, b, th) in enumerate(IND):
        ind_sel[BAG.index(b) if b else len(BAG), j] = 1; ind_thr[j] = th - 1
    g('ind.sel', ind_sel, S[...], 'picks one count per indicator', 'indicator j looks at one count (or the pending-confirm flag)')
    g('ind.thr', ind_thr, S[...], 'indicator = clamp(count - (min-1), 0, 1)', 'min count 2 means "at least two files"')
    II = {n: j for j, (n, _, _) in enumerate(IND)}
    FLOWS = ['hello', 'thanks', 'bye', 'unknown', 'unsafe', 'ls', 'show', 'lookup', 'find', 'calc', 'followup', 'double',
             'lookup_calc', 'find_lookup', 'find_lookup_calc', 'sum2', 'write', 'delete_ask', 'delete_yes', 'delete_no', 'wc', 'protected',
             'lookup_it', 'lookup_other', 'show_it', 'delete_it']
    FI = {f: i for i, f in enumerate(FLOWS)}
    # name: (priority, +required, -forbidden)
    COND = {  # name: (priority, required indicators, forbidden indicators). Higher priority wins among matches.
        'unsafe': (100, ['path'], []), 'protected': (99, ['prot'], []),
        'delete_no': (91, ['pend', 'no'], ['path']),       # NEW: no beats yes when both appear (the safe reading)
        'delete_yes': (90, ['pend', 'yes'], ['path']),
        'delete_ask': (80, ['del', 'F1'], ['path']),
        'delete_it': (79, ['del', 'it'], ['F1', 'path']),   # NEW: "delete it" -> confirm the focus file
        'write': (70, ['set', 'F1', 'K1'], []),
        'sum2': (60, ['F2', 'K2'], ['set', 'del']),
        'wc': (59, ['wc', 'F1'], ['del', 'set']),
        # NEW structural rules: a value word (name / state) with a key and no file => find the file, then look up
        'find_lookup_calc': (58, ['W1', 'K1', 'OP', 'N1'], ['F1', 'set', 'del']),
        'find_lookup': (57, ['W1', 'K1'], ['F1', 'set', 'del']),
        'lookup_calc': (55, ['F1', 'K1', 'N1', 'OP'], ['F2', 'set', 'del']),
        'lookup': (54, ['F1', 'K1'], ['N1', 'set', 'del']),
        'find': (53, ['W1'], ['K1', 'F1', 'set', 'del']),                      # NEW: a value word alone => search
        'lookup_it': (52, ['K1'], ['F1', 'W1', 'N1', 'set', 'del']),           # NEW: a key alone => the focus file
        'lookup_other': (51, ['F1', 'ell'], ['K1', 'show', 'del', 'set']),     # NEW: "and in F ?" => focus key on F
        'show': (50, ['F1'], ['K1', 'del', 'set']),
        'show_it': (49, ['show', 'it'], ['F1', 'K1', 'del', 'set']),           # NEW: "show it"
        'double': (45, ['dbl'], ['F1', 'K1']),
        'followup': (44, ['OP', 'N1'], ['N2', 'F1', 'K1', 'W1']),
        'calc': (43, ['N2'], ['F1', 'K1', 'W1']),
        'ls': (40, ['ls'], ['F1', 'K1', 'W1']),
        'hello': (20, ['hello'], []), 'thanks': (20, ['thanks'], []), 'bye': (20, ['bye'], []),
    }
    fand = torch.zeros(len(IND), len(FLOWS), dtype=D); fbias = torch.zeros(len(FLOWS), dtype=D)
    fprio = torch.zeros(len(FLOWS), dtype=D); fbase = torch.zeros(len(FLOWS), dtype=D)
    for f, (pr, pos_, neg_) in COND.items():
        j = FI[f]
        for n in pos_: fand[II[n], j] = 1
        for n in neg_: fand[II[n], j] = -1
        fbias[j] = -(len(pos_) - 1); fprio[j] = pr
    fbase[FI['unknown']] = 1
    g('flow.and', fand, S[...], 'per intent: +1 on required cues, -1 on forbidden cues', 'relu(sum - (m-1)) is 1 only if all m required cues are on and no forbidden one is')
    g('flow.bias', fbias, S[...], '-(m-1) for an intent with m required cues', 'turns the sum into an AND')
    g('flow.prio', fprio, S[...], 'intent priority (score if matched)', 'unsafe 100 > delete answers 90 > ... > greetings 20; ties impossible between overlapping rules')
    g('flow.base', fbase, S[...], 'fallback score of "unknown"', 'score 1 beats nothing-matched (0) and loses to any matched intent (>=20)')

    # ---- user slots: n-th flagged token of the current user turn
    SLOTS = [('F1', 'file', 1, 'tid'), ('F2', 'file', 2, 'tid'), ('K1', 'key', 1, 'tid'), ('K2', 'key', 2, 'tid'),
             ('W', 'wtok', 1, 'tid'), ('VAL', 'valtok', -1, 'tid'), ('N1', 'num', 1, 'tid'), ('N2', 'num', 2, 'tid'),
             ('OP', 'opw', 1, 'opid'), ('PK', 'key', 1, 'pos'), ('PN', 'num', 1, 'pos')]
    NS = len(SLOTS)
    sflag = torch.zeros(NS, len(COLS), dtype=D); sn = torch.zeros(NS, dtype=D); sval = torch.zeros(NS, 3, dtype=D)
    for i, (n, col, k, vk) in enumerate(SLOTS):
        sflag[i, CI[col]] = 1; sn[i] = k; sval[i, ['tid', 'opid', 'pos'].index(vk)] = 1
    g('slot.flag', sflag, S[...], 'which token class each user slot scans for', 'F=file names, K=keys, W=name/word values, VAL=any value, N=numbers, OP=operator words')
    g('slot.n', sn, S[...], 'which occurrence to take (-1 = the last)', 'the 1st file is the one asked about; the value to write is the LAST value token')
    g('slot.val', sval, S[...], 'what to read from the chosen token: its id, its operator id, or its position', 'ids are copied as numbers; positions give text order')
    SV = ['F1', 'F2', 'K1', 'K2', 'W', 'VAL', 'N1', 'N2', 'OP', 'ANS', 'HIT', 'PREVF', 'V1', 'V2', 'RES0', 'TWO', 'ARGA', 'ARGB', 'FF', 'FK']
    NV = len(SV); SI = {s: i for i, s in enumerate(SV)}
    frm = torch.zeros(NS, NV, dtype=D)
    for i, (n, *_r) in enumerate(SLOTS):
        if n in SI: frm[i, SI[n]] = 1
    g('sv.fromslot', frm, S[...], 'copies the 9 user slots into the 18-entry value vector', 'identity on the first 9 entries; PK/PN are only used for ordering')
    for n in ['ANS', 'HIT', 'PREVF', 'V1', 'V2', 'RES0', 'TWO', 'ARGA', 'ARGB', 'FF', 'FK']:
        v = torch.zeros(NV, dtype=D); v[SI[n]] = 1
        g('sv.' + n.lower(), v, S[...], f'writes the computed value "{n}" into its entry of the value vector',
          {'ANS': 'NEW: last number the bot said in a reply (so "now times 3" works after an answer or a looked-up value)', 'HIT': 'first token of the first result (the grep hit)',
           'PREVF': 'file right before the last "?" (the confirm question)', 'V1': 'value of K1 on a key-starting line of result RA',
           'V2': 'value of K2 on a line of result RB', 'RES0': 'first token of the newest result', 'TWO': 'constant 2 for double',
           'ARGA': 'first calc operand after swap', 'ARGB': 'second calc operand after swap',
           'FF': 'NEW: focus file, read from the previous turn\'s think', 'FK': 'NEW: focus key, read from the previous turn\'s think'}[n])
    for n, k in (('k1', 'K1'), ('k2', 'K2'), ('pk', 'PK'), ('pn', 'PN')):
        v = torch.zeros(NS, dtype=D); v[[s[0] for s in SLOTS].index(k)] = 1
        g('sl.' + n, v, S[...], f'selects slot {k} from the slot list', 'one-hot')

    # ---- per-flow lookups (result index of V1/V2) and operands
    ra = torch.zeros(len(FLOWS), dtype=D); rb = torch.zeros(len(FLOWS), dtype=D)
    for f in ('lookup', 'lookup_calc', 'sum2', 'lookup_it', 'lookup_other'): ra[FI[f]] = 1
    for f in ('find_lookup', 'find_lookup_calc'): ra[FI[f]] = 2
    rb[FI['sum2']] = 2
    g('flow.ra', ra, S[...], 'which result of the turn holds the line for K1', 'grep-first flows read the file as result 2')
    fk = torch.zeros(len(FLOWS), dtype=D); fk[FI['lookup_other']] = 1
    g('flow.fk', fk, S[...], 'NEW: intents whose lookup key is the focus key from memory', 'lookup_other ("and in F ?") has no key in the message; it reuses the key in focus')
    g('flow.rb', rb, S[...], 'which result holds the line for K2', 'sum2 reads file 2 as result 2')
    opa = torch.zeros(len(FLOWS), NV, dtype=D); opb = torch.zeros(len(FLOWS), NV, dtype=D); swap = torch.zeros(len(FLOWS), dtype=D)
    for f, (a, b, sw) in {'calc': ('N1', 'N2', 1), 'followup': ('ANS', 'N1', 0), 'double': ('ANS', 'TWO', 0), 'lookup_calc': ('V1', 'N1', 1),
                          'find_lookup_calc': ('V1', 'N1', 1), 'sum2': ('V1', 'V2', 0)}.items():
        opa[FI[f], SI[a]] = 1; opb[FI[f], SI[b]] = 1; swap[FI[f]] = sw
    g('opd.a', opa, S[...], 'first calc operand per intent', 'calc: 1st number; followup/double: previous answer; lookups: the looked-up value')
    g('opd.b', opb, S[...], 'second calc operand per intent', 'calc: 2nd number; followup: the number; double: 2; sum2: second value')
    g('opd.swap', swap, S[...], 'intent allows reordering of the operands', 'order follows the text (number first => swapped); "subtract" flips again')

    # ---- scripts
    CALLS = {  # per flow: list of (tool, [slot,...]) ; one entry per call index c
        'ls': [('ls', [])], 'show': [('read', ['F1'])], 'lookup': [('read', ['F1'])], 'find': [('grep', ['W'])],
        'find_lookup': [('grep', ['W']), ('read', ['HIT'])],
        'find_lookup_calc': [('grep', ['W']), ('read', ['HIT']), ('calc', ['ARGA', 'OP', 'ARGB'])],
        'calc': [('calc', ['ARGA', 'OP', 'ARGB'])], 'followup': [('calc', ['ARGA', 'OP', 'ARGB'])], 'double': [('calc', ['ARGA', 'OP', 'ARGB'])],
        'lookup_calc': [('read', ['F1']), ('calc', ['ARGA', 'OP', 'ARGB'])],
        'sum2': [('read', ['F1']), ('read', ['F2']), ('calc', ['ARGA', 'OP', 'ARGB'])],
        'write': [('write', ['F1', 'K1', 'VAL'])], 'wc': [('wc', ['F1'])], 'delete_yes': [('delete', ['PREVF'])],
        'lookup_it': [('read', ['FF'])], 'lookup_other': [('read', ['F1'])], 'show_it': [('read', ['FF'])]}
    clit = torch.zeros(len(FLOWS), 4, 6, dtype=D); cslot = torch.zeros(len(FLOWS), 4, 6, NV, dtype=D)
    for f, calls in CALLS.items():
        for c, (tool, args) in enumerate(calls):
            clit[FI[f], c, 0] = tid('<tool>'); clit[FI[f], c, 1] = inv[tool]
            for a, s in enumerate(args): cslot[FI[f], c, 2 + a, SI[s]] = 1
            clit[FI[f], c, 2 + len(args)] = tid('</tool>')
    g('call.lit', clit, S[...], 'per (intent, calls done, step): literal token to emit: <tool>, tool name, </tool>', 'step 0 non-zero means "make another call"; zero means "start the reply"')
    g('call.slot', cslot, S[...], 'per (intent, calls done, step): which value-vector entry to copy as an argument', 'e.g. read gets F1 (the user-named file) or HIT (a grep hit); delete gets PREVF (the confirmed file)')

    RTS = {  # reply templates; {X} = slot of the value vector, COPY = copy the newest result
        'hello': 'hello ! i can list , read , search and edit files in this project , and do math .',
        'thanks': 'you are welcome .', 'bye': 'goodbye .',
        'unknown': 'sorry , i can only help with files in this project and with math .',
        'unsafe': 'sorry , that path is outside this project .',
        'protected': 'secrets is protected , i cannot open it .', 'wc': '{F1} has {RES0} lines .',
        'confirm': 'are you sure you want to delete {F1} ?', 'kept': 'ok , i kept {PREVF} .',
        'ls': 'the files are COPY', 'show': '{F1} says : COPY', 'missing': 'i could not find {F1} .',
        'lookup': 'the {K1} in {F1} is {V1} .', 'nokey': '{F1} has no {K1} .', 'find': '{W} is in COPY',
        'notfound': 'no file mentions {W} .', 'answer': 'the answer is {RES0} .', 'calc_error': 'i could not compute that .',
        'write': 'i set {K1} in {F1} to {VAL} .', 'deleted': 'i deleted {PREVF} .',
        'lookup_h': 'the {K1} in {HIT} is {V1} .', 'nokey_h': '{HIT} has no {K1} .',
        'lookup_ff': 'the {K1} in {FF} is {V1} .', 'nokey_ff': '{FF} has no {K1} .', 'missing_ff': 'i could not find {FF} .',
        'show_ff': '{FF} says : COPY', 'confirm_ff': 'are you sure you want to delete {FF} ?',
        'lookup_fk': 'the {FK} in {F1} is {V1} .', 'nokey_fk': '{F1} has no {FK} .'}
    RN = list(RTS); RI = {r: i for i, r in enumerate(RN)}
    tlit = torch.zeros(len(RN), 24, dtype=D); tslot = torch.zeros(len(RN), 24, NV, dtype=D); tk0 = torch.full((len(RN),), 99.0, dtype=D)
    for r, text in RTS.items():
        for k, w in enumerate(text.split()):
            if w == 'COPY': tk0[RI[r]] = k; break
            if w.startswith('{'): tslot[RI[r], k, SI[w[1:-1]]] = 1
            else: tlit[RI[r], k] = inv[w]
    g('tmpl.lit', tlit, S[...], 'per (reply type, token index): literal word of the fixed sentence', 'the exact reply templates of the spec, one token per column')
    g('tmpl.slot', tslot, S[...], 'per (reply type, token index): which value-vector entry fills a {slot}', '{f}->F1, {k}->K1, {v}->V1 or VAL, {w}->W, {r}->RES0')
    g('tmpl.k0', tk0, S[...], 'token index where the copy of the newest result starts (99 = never)', 'list/content replies copy the tool result then end with "."')
    NST = 5  # status of the newest result: 0 normal, 1 missing, 2 none, 3 error, 4 key absent
    base = {'hello': 'hello', 'thanks': 'thanks', 'bye': 'bye', 'unknown': 'unknown', 'unsafe': 'unsafe', 'ls': 'ls', 'show': 'show', 'lookup': 'lookup',
            'find': 'find', 'calc': 'answer', 'followup': 'answer', 'double': 'answer', 'lookup_calc': 'answer', 'find_lookup': 'lookup_h',
            'find_lookup_calc': 'answer', 'sum2': 'answer', 'write': 'write', 'delete_ask': 'confirm', 'delete_yes': 'deleted', 'delete_no': 'kept',
            'wc': 'wc', 'protected': 'protected', 'lookup_it': 'lookup_ff', 'lookup_other': 'lookup_fk', 'show_it': 'show_ff', 'delete_it': 'confirm_ff'}
    ALT = {  # (intent, status) -> reply type, where it differs from the base reply
        **{(f, 1): 'missing' for f in ('show', 'lookup', 'lookup_calc', 'sum2', 'write', 'wc', 'lookup_other')},
        ('lookup_it', 1): 'missing_ff', ('show_it', 1): 'missing_ff', ('find', 2): 'notfound',
        **{(f, 3): 'calc_error' for f in FLOWS if base[f] == 'answer'},
        ('lookup', 4): 'nokey', ('find_lookup', 4): 'nokey_h', ('lookup_it', 4): 'nokey_ff', ('lookup_other', 4): 'nokey_fk'}
    rmap = torch.zeros(len(FLOWS), NST, len(RN), dtype=D)
    for f in FLOWS:
        for s_ in range(NST):
            rmap[FI[f], s_, RI[ALT.get((f, s_), base[f])]] = 1
    g('rt.map', rmap, S[...], 'per (intent, result status) -> which reply type', 'status of the newest result: normal / missing / none / error / key-absent decides between e.g. lookup and "has no"')
    st = torch.tensor([tid('missing'), tid('none'), tid('error')], dtype=D)
    g('st.ids', st, S[...], 'ids of the result words missing / none / error', 'first token of the newest result is compared to these')
    # NEW: what the think register records at the end of each turn (default: carry the old focus forward)
    focf = torch.zeros(len(FLOWS), NV, dtype=D); fock = torch.zeros(len(FLOWS), NV, dtype=D)
    for f in FLOWS:
        focf[FI[f], SI[{'show': 'F1', 'lookup': 'F1', 'lookup_calc': 'F1', 'write': 'F1', 'wc': 'F1', 'delete_ask': 'F1', 'sum2': 'F1',
                        'lookup_other': 'F1', 'find_lookup': 'HIT', 'find_lookup_calc': 'HIT'}.get(f, 'FF')]] = 1
        fock[FI[f], SI['K1' if f in ('lookup', 'lookup_calc', 'lookup_it', 'find_lookup', 'find_lookup_calc', 'write', 'sum2') else 'FK']] = 1
    g('foc.f', focf, S[...], 'NEW: per intent, which value becomes the focus file written into think',
      'the file the turn was about (named file, or the grep hit); intents without a file carry the old focus forward (FF)')
    g('foc.k', fock, S[...], 'NEW: per intent, which value becomes the focus key written into think',
      'the key just asked about; lookup_other and intents without a key carry the old focus key forward (FK)')
    return CustomArch('experiments/gen/agents/opus/arch.py',
                      'tensor-op script interpreter: tables for intent, slots, tool scripts and reply templates; hard attention over the current turn; scalar-id readout'), p, G
