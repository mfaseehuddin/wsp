import torch
import math

# ---------------------------------------------------------------- constants
U = 32.0          # amplitude of every binary feature channel
C = 28            # code dimension for token codes (copy / emit space)
E = 64            # d_head
NH = 4            # heads per layer
NL = 8            # layers
M = 256           # mlp width
FREQ = 16         # pos frequency pairs used by shift heads (channels 0..31)
SPECIAL = [1, 2, 3, 6, 7, 8, 9, 10]
G0 = 1000000.0      # flag score for latest-of-class heads
RHO = 3000.0      # recency weight


def new():
    return dict(words=[], wmap={}, items=[], classes={}, shifts={}, rn=0)


def wid(S, w):
    if w not in S['wmap']:
        S['wmap'][w] = 272 + len(S['words'])
        S['words'].append(w)
    return S['wmap'][w]


def tid(S, x):
    if isinstance(x, int):
        return x
    if len(x) == 1:
        return 16 + ord(x)
    return wid(S, x)


def seq(S, parts):
    return [tid(S, p) for p in parts]


def tc(i):
    return 't' + str(i)


def want(S, name):
    # 'p1:X' channel = channel X at the previous position (auto shift head)
    if name.startswith('p1:') and name not in S['shifts']:
        k = 0
        base = name
        for _ in range(8):
            if base.startswith('p1:'):
                base = base[3:]
                k += 1
        S['shifts'][name] = base
        want(S, base)
        S['items'].append(dict(kind='shift', dst=name, src=base, k=k, reads=[base], writes=[name], tag='shift'))


def tokclass(S, name, ids):
    S['classes'][name] = [tid(S, i) for i in ids]


def rule(S, tag, conds, outs, negs=None):
    negs = negs or []
    reads = []
    for g in conds + negs:
        for c in g:
            want(S, c)
            reads.append(c)
    writes = [o[1] for o in outs if o[0] == 'f']
    S['items'].append(dict(kind='rule', conds=conds, negs=negs, outs=outs, reads=reads, writes=writes, tag=tag))


def latest(S, tag, keys, vals, gate=None, valcode=False, valmap=None, extra_sink=True):
    reads = [k for k in keys] + [v[0] for v in vals] + (gate or [])
    for r in reads:
        want(S, r)
    S['items'].append(dict(kind='latest', keys=keys, vals=vals, gate=gate, valcode=valcode, valmap=valmap or {},
                           reads=reads, writes=([v[1] for v in vals] + (['cout_%d' % j for j in range(C)] if valcode else [])), tag=tag))


def induct(S, tag, modes, valmap=None):
    # modes: list of (mode_channel, class_channel)
    reads = [m[0] for m in modes] + ['p1c_0'] + ['p1:' + m[2] for m in modes] + ['p1:p1:' + m[2] for m in modes] + ['p1:p1:p1:' + m[2] for m in modes] + ['p1:p1:p1:p1:' + m[2] for m in modes]
    for r in reads:
        want(S, r)
    S['items'].append(dict(kind='induct', modes=modes, valmap=valmap or {}, reads=reads, writes=['cout_%d' % j for j in range(C)],
                           tag=tag))


def flag_latest(S, name, cls):
    latest(S, 'flag.' + name, [cls], [(cls, 'L_' + name)])


def ctx(ids, i, depth, start=3):
    conds = [[tc(ids[i - 1])]]
    for k in range(2, depth + 1):
        if i >= k:
            conds.append(['p1:' * (k - 1) + tc(ids[i - k])])
        elif i == k - 1:
            conds.append(['p1:' * (k - 1) + tc(start)])
    return conds


def frame(S, tag, trig, parts, tagconds, negs=None, start=3):
    ids = seq(S, parts)
    for i in range(len(ids)):
        if i == 0:
            conds = trig
        else:
            conds = ctx(ids, i, 5, start) + tagconds
        rule(S, tag, conds, [('e', ids[i], 1.0)], negs)


# ------------------------------------------------------------------ modules
CAPITALS = [('France', 'Paris'), ('Japan', 'Tokyo'), ('Italy', 'Rome'), ('Spain', 'Madrid'), ('Germany', 'Berlin'),
            ('Egypt', 'Cairo'), ('Canada', 'Ottawa'), ('China', 'Beijing'), ('India', 'Delhi'), ('Russia', 'Moscow'),
            ('Kenya', 'Nairobi'), ('Brazil', 'Brasilia'), ('Greece', 'Athens'), ('Peru', 'Lima'), ('Norway', 'Oslo'),
            ('Cuba', 'Havana'), ('Austria', 'Vienna'), ('Portugal', 'Lisbon'), ('Ireland', 'Dublin'), ('Poland', 'Warsaw')]
MAPW = {'plus': '+', 'minus': '-', 'times': '*'}
MAPID = {}
LETTERS = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]


def bytes_of(chars):
    return [16 + ord(c) for c in chars]


def copy_head(S, tag, key, mode):
    keys = key if isinstance(key, list) else [key]
    modes = mode if isinstance(mode, list) else [mode]
    latest(S, tag, keys, [], gate=modes, valcode=True)


def modules(S):
    for w in MAPW:
        MAPID[wid(S, w)] = MAPW[w]
    BOT = 3
    SP = 16 + ord(' ')
    Q = 16 + ord('"')
    digits = bytes_of('0123456789')
    tokclass(S, 'c_digit', digits)
    ops = bytes_of('+-*/') + [wid(S, w) for w in MAPW]
    tokclass(S, 'c_op', ops)
    tokclass(S, 'c_expr', digits + ops + bytes_of(' ()'))
    tokclass(S, 'c_num', digits)
    tokclass(S, 'c_greet', [wid(S, 'Hello'), wid(S, 'Hi'), wid(S, 'Hey')])
    tokclass(S, 'c_thanks', [wid(S, 'Thanks'), wid(S, 'thanks'), wid(S, 'Thank')])
    tokclass(S, 'c_bye', [wid(S, 'bye'), wid(S, 'Bye'), wid(S, 'Goodbye')])
    tokclass(S, 'c_howare', [wid(S, 'How')])
    tokclass(S, 'c_q', bytes_of('?'))
    tokclass(S, 'c_upper', bytes_of(LETTERS[:26]))
    tokclass(S, 'c_name', bytes_of(LETTERS + ["'", '-']))
    tokclass(S, 'c_path', bytes_of(LETTERS + list('0123456789._-')))
    tokclass(S, 'c_text', bytes_of([chr(c) for c in range(32, 127) if chr(c) not in '"\\.']))
    tokclass(S, 'c_secret', [wid(S, 'secrets'), wid(S, 'secret')])
    tokclass(S, 'c_nameword', [wid(S, 'name')])
    tokclass(S, 'c_rel', [wid(S, 'capital')])
    tokclass(S, 'c_ent', [wid(S, e) for e, v in CAPITALS])
    tokclass(S, 'c_val', [wid(S, v) for e, v in CAPITALS])
    tokclass(S, 'c_read', [wid(S, 'Read'), wid(S, 'read'), wid(S, 'Open'), wid(S, 'open')])
    tokclass(S, 'c_delete', [wid(S, 'delete'), wid(S, 'Delete')])
    S['classes']['c_wstart'] = []
    tokclass(S, 'c_yes', [wid(S, 'Yes'), wid(S, 'yes')])
    tokclass(S, 'c_no', [wid(S, 'No'), wid(S, 'no')])
    tokclass(S, 'c_pathtool', [wid(S, 'Read'), wid(S, 'read'), wid(S, 'Open'), wid(S, 'open'), wid(S, 'delete'), wid(S, 'Delete')])
    tokclass(S, 'c_who', [wid(S, 'Who'), wid(S, 'who')])
    tokclass(S, 'c_howto', [wid(S, 'How'), wid(S, 'how')])
    tokclass(S, 'c_you', [wid(S, 'you')])
    tokclass(S, 'c_are', [wid(S, 'are')])
    anchors = ['expression', 'is', 'calc', 'read', 'path', 'says', 'meet', 'name']
    tokclass(S, 'c_anchor', [wid(S, w) for w in anchors] + [9])
    for w in ['The', 'answer', 'of', 'Nice', 'to', 'you', 'Your', 'name', 'Goodbye', 'welcome', 'help', 'How', 'can', 'I', 'doing', 'well', 'don\'t', 'know', 'that', 'yet', 'It', 'cannot', 'do', 'am', 'My', 'my', 'I\'m', 'path']:
        wid(S, w)
    # --- turn-local flags (latest of class vs USER)
    for nm in ['digit', 'op', 'q', 'nameword']:
        flag_latest(S, nm, 'c_' + nm)
    for grp in [['greet', 'thanks', 'bye'], ['secret', 'dots'], ['read', 'delete'], ['yes', 'no', 'wstart']]:
        latest(S, 'flag.' + grp[0], ['c_' + n for n in grp], [('c_' + n, 'L_' + n) for n in grp])
    rule(S, 'intro.howare', [['c_you'], ['p1:' + tc(SP)], ['p1:p1:c_are'], ['p1:p1:p1:' + tc(SP)], ['p1:p1:p1:p1:c_howto']], [('f', 'c_howareyou', 1.0)])
    rule(S, 'intro.whoare', [['c_you'], ['p1:' + tc(SP)], ['p1:p1:c_are'], ['p1:p1:p1:' + tc(SP)], ['p1:p1:p1:p1:c_who']], [('f', 'c_whoareyou', 1.0)])
    S['classes']['c_howareyou'] = []
    S['classes']['c_whoareyou'] = []
    latest(S, 'flag.howare', ['c_howareyou', 'c_whoareyou'], [('c_howareyou', 'L_howareyou'), ('c_whoareyou', 'L_whoareyou')])
    latest(S, 'phase', ['c_anchor'], [(tc(wid(S, w)), 'ph_' + w) for w in anchors] + [(tc(9), 'ph_result')])
    latest(S, 'rel', ['c_rel'], [(tc(wid(S, 'capital')), 'R_capital')])
    latest(S, 'ent1', ['c_ent'], [(tc(wid(S, e)), 'E_' + e) for e, v in CAPITALS[:20]])
    # dots ('..') flag
    rule(S, 'safe.dots', [[tc(16 + ord('.'))], ['p1:' + tc(16 + ord('.'))]], [('f', 'c_dots', 1.0)])
    S['classes']['c_dots'] = []
    # --- name capture: name start = uppercase letter after "I'm ", "am ", "is " preceded by name, "me "
    rule(S, 'name.start', [['c_upper'], ['p1:' + tc(SP)], ['p1:p1:' + tc(wid(S, "I'm"))]], [('f', 'c_namestart', 1.0)])
    rule(S, 'name.start', [['c_upper'], ['p1:' + tc(SP)], ['p1:p1:' + tc(wid(S, 'am'))]], [('f', 'c_namestart', 1.0)])
    rule(S, 'name.start', [['c_upper'], ['p1:' + tc(SP)], ['p1:p1:' + tc(wid(S, 'is'))], ['p1:p1:p1:' + tc(SP)], ['p1:p1:p1:p1:' + tc(wid(S, 'name'))]], [('f', 'c_namestart', 1.0)])
    S['classes']['c_namestart'] = []
    flag_latest(S, 'namestart', 'c_namestart')
    copy_head(S, 'name.first', 'c_namestart', 'm_nfirst')
    # (memory head shares key: latest namestart in whole context)
    induct(S, 'name.cont', [('m_ncont', 'c_name', 'c_namestart')])
    # --- expression
    flag_latest(S, 'dummy', 'c_dummy') if False else None
    rule(S, 'calc.intent', [['L_digit'], ['L_op']], [('f', 'f_calc', 1.0)])
    rule(S, 'calc.exprstart', [['c_digit']], [('f', 'f_exprstart', 1.0)], negs=[['p1:L_digit'], ['p1:' + tc(16 + ord('('))]])
    rule(S, 'calc.exprstart', [[tc(16 + ord('('))]], [('f', 'f_exprstart', 1.0)], negs=[['p1:L_digit']])
    copy_head(S, 'calc.first', ['f_exprstart', 'f_wstart'], ['m_first', 'm_w1'])
    induct(S, 'calc.cont', [('m_cont', 'c_expr', 'f_exprstart'), ('m_resc', 'c_num', 'f_resstart'), ('m_pathc', 'c_path', 'f_pathstart'), ('m_txtc', 'c_textdata', 'f_txtstart'), ('m_wc', 'c_name', 'f_wstart')], valmap=MAPID)
    pre = ['{', '"', 'name', '"', ':', '"', 'calc', '"', ',', '"', 'args', '"', ':', '{', '"', 'expression', '"', ':', '"']
    ids = seq(S, pre)
    nodata = [['L_secret'], ['L_dots']]
    rule(S, 'calc.call', [[tc(BOT)], ['f_calc']], [('e', 6, 1.0)])
    rule(S, 'calc.call', [[tc(6)], ['f_calc']], [('e', ids[0], 1.0)])
    for i in range(1, len(ids)):
        rule(S, 'calc.call', ctx(ids, i, 5, 6) + [['f_calc']], [('e', ids[i], 1.0)])
    rule(S, 'calc.first', [[tc(Q)], ['p1:' + tc(16 + ord(':'))], ['ph_expression'], ['f_calc']], [('f', 'm_first', 1.0)])
    rule(S, 'calc.cont', [['c_expr'], ['ph_expression'], ['f_calc']], [('f', 'm_cont', 1.0)])
    rule(S, 'calc.close', [['c_expr'], ['ph_expression'], ['f_calc']], [('e', Q, 0.5)])
    rule(S, 'calc.close', [[tc(Q)], ['p1:c_expr'], ['ph_expression']], [('e', 16 + ord('}'), 1.0)])
    rule(S, 'calc.close', [[tc(16 + ord('}'))], ['p1:' + tc(Q)], ['ph_expression']], [('e', 16 + ord('}'), 1.0)])
    rule(S, 'calc.close', [[tc(16 + ord('}'))], ['p1:' + tc(16 + ord('}'))], ['ph_expression']], [('e', 7, 1.0)])
    # answer after RESULT
    rule(S, 'calc.resstart', [['c_digit'], ['p1:' + tc(SP)], ['p1:p1:' + tc(16 + ord(':'))]], [('f', 'f_resstart', 1.0)])
    copy_head(S, 'calc.res', 'f_resstart', 'm_res1')
    fr = ['The', ' ', 'answer', ' ', 'is', ' ']
    frame(S, 'calc.answer', [[tc(9)], ['f_calc']], fr, [['f_calc']], start=9, negs=[['L_read'], ['L_delete']])
    rule(S, 'calc.answer', [[tc(SP)], ['p1:' + tc(wid(S, 'is'))], ['ph_is'], ['f_calc']], [('f', 'm_res1', 1.0)])
    rule(S, 'calc.answer', [['c_num'], ['ph_is'], ['f_calc']], [('f', 'm_resc', 1.0)])
    rule(S, 'calc.answer', [['c_num'], ['ph_is'], ['f_calc']], [('e', 16 + ord('.'), 0.5)])
    # --- social
    hello = ['Hello', '!', ' ', 'How', ' ', 'can', ' ', 'I', ' ', 'help', '?']
    frame(S, 'social.greet', [[tc(BOT)], ['L_greet']], hello, [['L_greet']], negs=[['c_namestart'], ['L_namestart'], ['L_thanks']])
    frame(S, 'social.thanks', [[tc(BOT)], ['L_thanks']], ['You', "'", 're', ' ', 'welcome', '!'], [['L_thanks']])
    frame(S, 'social.bye', [[tc(BOT)], ['L_bye']], ['Goodbye', '!', ' ', 'Take', ' ', 'care', '.'], [['L_bye']])
    # --- meet (name statement) and recall
    meet = ['Nice', ' ', 'to', ' ', 'meet', ' ', 'you', ',', ' ']
    frame(S, 'name.meet', [[tc(BOT)], ['L_namestart']], meet, [['L_namestart']], negs=[['L_q']])
    ids = seq(S, meet)
    rule(S, 'name.meet', [[tc(SP)], ['p1:' + tc(16 + ord(','))], ['L_namestart']], [('f', 'm_nfirst', 1.0)], negs=[['L_q']])
    rule(S, 'name.meet', [['c_name'], ['ph_meet'], ['L_namestart']], [('f', 'm_ncont', 1.0)])
    rule(S, 'name.meet', [['c_name'], ['ph_meet'], ['L_namestart']], [('e', 16 + ord('!'), 0.5)])
    recall = ['Your', ' ', 'name', ' ', 'is', ' ']
    frame(S, 'name.recall', [[tc(BOT)], ['L_nameword'], ['L_q']], recall, [['L_nameword'], ['L_q']], negs=[['L_namestart']])
    rule(S, 'name.recall', [[tc(SP)], ['p1:' + tc(wid(S, 'is'))], ['ph_is'], ['L_nameword'], ['L_q']], [('f', 'm_nfirst', 1.0)])
    rule(S, 'name.recall', [['c_name'], ['ph_is'], ['L_nameword'], ['L_q']], [('f', 'm_ncont', 1.0)])
    rule(S, 'name.recall', [['c_name'], ['ph_is'], ['L_nameword'], ['L_q']], [('e', 16 + ord('.'), 0.5)])
    # --- facts
    for e, v in CAPITALS:
        rule(S, 'fact.known', [['R_capital'], ['E_' + e]], [('f', 'f_known', 1.0)])
        rule(S, 'fact.value', [[tc(SP)], ['p1:' + tc(wid(S, 'is'))], ['R_capital'], ['E_' + e], ['f_known']], [('e', wid(S, v), 1.0)])
    fa = ['The', ' ']
    frame(S, 'fact.frame', [[tc(BOT)], ['f_known']], fa, [['f_known']])
    copy_head(S, 'fact.rel', 'c_rel', 'm_rel')
    copy_head(S, 'fact.ent', 'c_ent', 'm_ent')
    rule(S, 'fact.frame', [[tc(SP)], ['p1:' + tc(wid(S, 'The'))], ['f_known']], [('f', 'm_rel', 1.0)])
    rule(S, 'fact.frame', [['c_rel'], ['p1:' + tc(SP)], ['f_known']], [('e', SP, 1.0)])
    rule(S, 'fact.frame', [[tc(SP)], ['p1:c_rel'], ['f_known']], [('e', wid(S, 'of'), 1.0)])
    rule(S, 'fact.frame', [[tc(wid(S, 'of'))], ['f_known']], [('e', SP, 1.0)])
    rule(S, 'fact.frame', [[tc(SP)], ['p1:' + tc(wid(S, 'of'))], ['f_known']], [('f', 'm_ent', 1.0)])
    rule(S, 'fact.frame', [['c_ent'], ['p1:' + tc(SP)], ['p1:p1:' + tc(wid(S, 'of'))], ['f_known']], [('e', SP, 1.0)])
    rule(S, 'fact.frame', [[tc(SP)], ['p1:c_ent'], ['f_known']], [('e', wid(S, 'is'), 1.0)])
    rule(S, 'fact.frame', [[tc(wid(S, 'is'))], ['p1:' + tc(SP)], ['f_known']], [('e', SP, 1.0)])
    rule(S, 'fact.frame', [['c_val'], ['p1:' + tc(SP)], ['p1:p1:' + tc(wid(S, 'is'))], ['f_known']], [('e', 16 + ord('.'), 1.0)])
    # --- tools: read a file
    frame(S, 'tool.read', [[tc(BOT)], ['L_read']], ['<TOOL>'] if False else [], [])
    rule(S, 'tool.read', [[tc(BOT)], ['L_read']], [('e', 6, 1.0)], negs=[['L_secret'], ['L_dots'], ['f_calc']])
    rp = ['{', '"', 'name', '"', ':', '"', 'read', '"', ',', '"', 'args', '"', ':', '{', '"', 'path', '"', ':', '"']
    rids = seq(S, rp)
    rule(S, 'tool.read', [[tc(6)], ['L_read']], [('e', rids[0], 1.0)], negs=[['L_secret'], ['L_dots'], ['f_calc']])
    for i in range(1, len(rids)):
        rule(S, 'tool.read', ctx(rids, i, 5, 6) + [['L_read']], [('e', rids[i], 1.0)], negs=[['L_secret'], ['L_dots'], ['f_calc']])
    rule(S, 'tool.read', [['c_letterstart'], ['p1:' + tc(SP)], ['p1:p1:c_pathtool']], [('f', 'f_pathstart', 1.0)])
    S['classes']['c_letterstart'] = bytes_of(LETTERS)
    copy_head(S, 'tool.path', 'f_pathstart', 'm_pfirst')
    rule(S, 'tool.read', [[tc(Q)], ['p1:' + tc(16 + ord(':'))], ['ph_path']], [('f', 'm_pfirst', 1.0)])
    rule(S, 'tool.read', [['c_path'], ['ph_path'], ['L_read']], [('f', 'm_pathc', 1.0)], negs=[['f_calc']])
    rule(S, 'tool.read', [['c_path'], ['ph_path'], ['L_read']], [('e', Q, 0.5)])
    rule(S, 'tool.read', [[tc(Q)], ['p1:c_path'], ['ph_path']], [('e', 16 + ord('}'), 1.0)])
    rule(S, 'tool.read', [[tc(16 + ord('}'))], ['p1:' + tc(Q)], ['ph_path']], [('e', 16 + ord('}'), 1.0)])
    rule(S, 'tool.read', [[tc(16 + ord('}'))], ['p1:' + tc(16 + ord('}'))], ['ph_path']], [('e', 7, 1.0)])
    rule(S, 'tool.read', [[tc(16 + ord('t'))], ['p1:' + tc(16 + ord('x'))], ['p1:p1:' + tc(16 + ord('t'))], ['p1:p1:p1:' + tc(16 + ord('.'))], ['ph_path'], ['L_read']], [('e', Q, 1.5)])
    rule(S, 'tool.delete', [[tc(16 + ord('t'))], ['p1:' + tc(16 + ord('x'))], ['p1:p1:' + tc(16 + ord('t'))], ['p1:p1:p1:' + tc(16 + ord('.'))], ['ph_path'], ['L_delete'], ['L_yes']], [('e', Q, 1.5)])
    # read result: {"text": "..."}
    rule(S, 'tool.text', [['c_text'], ['p1:' + tc(Q)], ['p1:p1:' + tc(SP)]], [('f', 'f_txtstart', 1.0)])
    copy_head(S, 'tool.txt', 'f_txtstart', 'm_tfirst')
    frame(S, 'tool.says', [[tc(9)], ['L_read']], ['It', ' ', 'says', ':', ' '], [['L_read']], start=9)
    rule(S, 'tool.says', [[tc(SP)], ['p1:' + tc(16 + ord(':'))], ['ph_says']], [('f', 'm_tfirst', 1.0)])
    rule(S, 'tool.says', [['c_text'], ['ph_says']], [('f', 'm_txtc', 1.0)])
    # --- delete: needs a yes in the same message, no "no"; otherwise ask first
    dp = ['{', '"', 'name', '"', ':', '"', 'delete', '"', ',', '"', 'args', '"', ':', '{', '"', 'path', '"', ':', '"']
    dids = seq(S, dp)
    dneg = [['L_secret'], ['L_dots'], ['L_no'], ['f_calc']]
    rule(S, 'tool.delete', [[tc(BOT)], ['L_delete'], ['L_yes']], [('e', 6, 1.0)], negs=dneg)
    rule(S, 'tool.delete', [[tc(6)], ['L_delete'], ['L_yes']], [('e', dids[0], 1.0)], negs=dneg)
    for i in range(1, len(dids)):
        rule(S, 'tool.delete', ctx(dids, i, 5, 6) + [['L_delete'], ['L_yes']], [('e', dids[i], 1.0)], negs=dneg)
    rule(S, 'tool.delete', [['c_path'], ['ph_path'], ['L_delete'], ['L_yes']], [('f', 'm_pathc', 1.0)], negs=dneg)
    rule(S, 'tool.delete', [['c_path'], ['ph_path'], ['L_delete'], ['L_yes']], [('e', Q, 0.5)], negs=dneg)
    frame(S, 'tool.deleted', [[tc(9)], ['L_delete'], ['L_yes']], ['Done', '.'], [['L_delete']], start=9)
    frame(S, 'tool.confirm', [[tc(BOT)], ['L_delete']], ['Delete', ' ', 'it', '?', ' ', 'Please', ' ', 'confirm', '.'], [['L_delete']], negs=[['L_yes'], ['L_no'], ['L_secret'], ['L_dots']])
    frame(S, 'tool.cancel', [[tc(BOT)], ['L_no']], ['Okay', ',', ' ', 'I', ' ', 'will', ' ', 'not', '.'], [['L_no']], negs=[['L_secret'], ['L_dots']])
    rule(S, 'tool.delete', [[tc(BOT)], ['L_delete'], ['L_yes'], ['L_no']], [('e', wid(S, 'Okay'), 1.0)])
    # --- identity / wellbeing (short social phrases)
    frame(S, 'social.howare', [[tc(BOT)], ['L_howareyou']], ['I', "'", 'm', ' ', 'doing', ' ', 'well', ',', ' ', 'thanks', '!'], [['L_howareyou']])
    frame(S, 'social.who', [[tc(BOT)], ['L_whoareyou']], ['I', ' ', 'am', ' ', 'a', ' ', 'small', ' ', 'assistant', '.'], [['L_whoareyou']])
    # --- the word X: exact word copy
    wid(S, 'word')
    rule(S, 'word.start', [['c_letterstart'], ['p1:' + tc(SP)], ['p1:p1:' + tc(wid(S, 'word'))]], [('f', 'f_wstart', 1.0), ('f', 'c_wstart', 1.0)])
    rule(S, 'word.reply', [[tc(BOT)], ['L_wstart']], [('f', 'm_w1', 1.0)], negs=[['f_calc']])
    rule(S, 'word.reply', [['c_name'], ['L_wstart']], [('f', 'm_wc', 1.0)], negs=[['f_calc'], ['L_namestart']])
    tokclass(S, 'c_res', [8, 9])
    latest(S, 'flag.indata', ['c_res'], [(tc(8), 'L_indata')])
    rule(S, 'tool.textdata', [['c_text'], ['L_indata']], [('f', 'c_textdata', 1.0)])
    S['classes']['c_textdata'] = []
    S['classes']['c_text'] = S['classes']['c_text'] + [272 + i for i in range(len(S['words']))]
    # --- safety refusals
    frame(S, 'safe.refuse', [[tc(BOT)], ['L_secret']], ['I', ' ', 'cannot', ' ', 'do', ' ', 'that', '.'], [['L_secret']])
    frame(S, 'safe.refuse2', [[tc(BOT)], ['L_dots']], ['I', ' ', 'cannot', ' ', 'do', ' ', 'that', '.'], [['L_dots']], negs=[['L_secret']])
    # --- fallback
    unk = ['I', ' ', "don't", ' ', 'know', ' ', 'that', ' ', 'yet', '.']
    ids = seq(S, unk)
    rule(S, 'fallback', [[tc(BOT)]], [('e', ids[0], 0.5)])
    for i in range(1, len(ids)):
        rule(S, 'fallback', ctx(ids, i, 5), [('e', ids[i], 1.0)], negs=[['L_secret'], ['L_dots'], ['f_known'], ['f_calc'], ['L_howareyou'], ['L_whoareyou'], ['L_namestart'], ['L_delete'], ['L_no'], ['L_read']])


# ------------------------------------------------------------------- codes
def make_codes(ids):
    # greedy near-orthogonal +-1 codes: each token takes the best of 96 seeded random candidates
    g = torch.Generator().manual_seed(1)
    first = [16 + ord(c) for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789']
    order = first + [v for v in ids if v not in first]
    chosen = torch.zeros(len(order), C)
    codes = {}
    for n_, v in enumerate(order):
        cand = torch.sign(torch.randn(96, C, generator=g))
        if n_ == 0:
            best = 0
        else:
            sc = torch.matmul(cand, chosen[:n_].T).abs().max(1).values
            best = int(torch.argmax(-sc).item())
        chosen[n_] = cand[best]
        codes[v] = [float(cand[best][j]) for j in range(C)]
    return codes


def build():
    S = new()
    S['items'].append(dict(kind='shiftcode', reads=[], writes=['p1c_%d' % j for j in range(C)], tag='shiftcode'))
    for j in range(C):
        want(S, 'p1:p1c_%d' % j)
    modules(S)
    items = S['items']
    # ---- active tokens
    active = SPECIAL + [16 + b for b in range(32, 127)] + [16 + 10] + [272 + i for i in range(len(S['words']))]
    V = 272 + len(S['words'])
    codes = make_codes(active)
    # ---- schedule
    n = len(items)
    writers = {}
    for k in range(n):
        for w in items[k]['writes']:
            writers[w] = writers.get(w, []) + [k]
    tau = {}
    placed = [False] * n
    layer_of = [0] * n
    nrule = [0] * NL
    nfix = [0] * NL
    nshift = [0] * NL
    shc = {}
    for _ in range(40):
        for k in range(n):
            if placed[k]:
                continue
            it = items[k]
            ok = True
            t = 0
            for r in it['reads']:
                if r in writers:
                    for wk_ in writers[r]:
                        if not placed[wk_]:
                            ok = False
                    if not ok:
                        break
                    t = max(t, tau[r])
            if not ok:
                continue
            kind = it['kind']
            L = (t + 1) // 2 if kind != 'rule' else t // 2
            for L2 in range(L, NL):
                if kind == 'rule':
                    if nrule[L2] + 2 <= M:
                        break
                elif kind == 'shift':
                    kk = it['k']
                    extra = 1 if shc.get((L2, kk), 0) % 64 == 0 else 0
                    if nfix[L2] + nshift[L2] + extra <= NH:
                        break
                else:
                    if nfix[L2] + nshift[L2] + 1 <= NH:
                        break
            else:
                L2 = NL - 1
            if kind == 'rule':
                nrule[L2] += 2
                tw = 2 * L2 + 2
            elif kind == 'shift':
                if shc.get((L2, it['k']), 0) % 64 == 0:
                    nshift[L2] += 1
                shc[(L2, it['k'])] = shc.get((L2, it['k']), 0) + 1
                tw = 2 * L2 + 1
            else:
                nfix[L2] += 1
                tw = 2 * L2 + 1
            layer_of[k] = L2
            placed[k] = True
            for w in it['writes']:
                tau[w] = max(tau.get(w, 0), tw)
    for k in range(n):
        if not placed[k]:
            layer_of[k] = NL - 1
    # ---- channels
    names = {}
    order = []

    def_names = []
    for i in active:
        def_names.append(tc(i))
    for c in S['classes'].keys():
        def_names.append(c)
    for j in range(C):
        def_names.append('cout_%d' % j)
        def_names.append('p1c_%d' % j)
        def_names.append('p1:p1c_%d' % j)
    for it in items:
        for r in it['reads']:
            def_names.append(r)
        for w in it['writes']:
            def_names.append(w)
        if it['kind'] == 'rule':
            for g in it['conds'] + it['negs']:
                for c in g:
                    def_names.append(c)
    for nm in def_names:
        if nm not in names:
            names[nm] = 32 + len(names)
    d = 32 + len(names) + 2
    d = (d + 15) // 16 * 16
    if d > 1024:
        d = 1024

    # ---- tensors
    cfg = dict(d_model=d, n_layers=NL, n_heads=NH, d_head=E, d_mlp=M, max_len=16384, pos="learned", norm=False,
               bias=True, tie_embeddings=False)
    p = {}
    p['embed'] = torch.zeros(V, d)
    for i in active:
        p['embed'][i, names[tc(i)]] = U
    for c, ids in S['classes'].items():
        for i in ids:
            p['embed'][i, names[c]] = U
    for L in range(NL):
        for nm in ['wq', 'wk', 'wv']:
            p['layers.%d.%s' % (L, nm)] = torch.zeros(NH, d, E)
        p['layers.%d.wo' % L] = torch.zeros(NH, E, d)
        for nm in ['bq', 'bk', 'bv']:
            p['layers.%d.%s' % (L, nm)] = torch.zeros(NH, E)
        p['layers.%d.bo' % L] = torch.zeros(d)
        p['layers.%d.w_in' % L] = torch.zeros(d, M)
        p['layers.%d.w_out' % L] = torch.zeros(M, d)
        p['layers.%d.b_in' % L] = torch.zeros(M)
        p['layers.%d.b_out' % L] = torch.zeros(d)
    p['unembed'] = torch.zeros(d, V)
    tt = torch.arange(16384).float().unsqueeze(1)
    p['pos'] = torch.zeros(16384, d)
    for i in range(FREQ):
        ww = 10000.0 ** (-2.0 * i / d)
        p['pos'][:, 2 * i] = torch.sin(tt[:, 0] * ww)
        p['pos'][:, 2 * i + 1] = torch.cos(tt[:, 0] * ww)
    wl = 10000.0 ** (-2.0 * (d // 2 - 1) / d)
    p['pos'][:, d - 2] = torch.sin(tt[:, 0] * wl)
    p['pos'][:, d - 1] = torch.cos(tt[:, 0] * wl)
    p['b_unembed'] = torch.zeros(V)
    groups = []
    om = [10000.0 ** (-2.0 * i / d) for i in range(d // 2)]
    lowc = d - 2
    phi = 0.3
    hcount = [0] * NL
    nrn = [0] * NL
    shift_pending = {}
    rule_groups = {}
    head_groups = []

    def add_group(name, tensor, index, mech, why):
        groups.append(dict(name=name, tensor=tensor, index=index, mechanism=' '.join(mech.split(' ')[:6]), why=' '.join(why.split(' ')[:6])))

    full = {"start": 0, "stop": d}
    # shifts grouped per layer
    for L in range(NL):
      for kk in range(1, 6):
        lst = [k for k in range(n) if items[k]['kind'] == 'shift' and layer_of[k] == L and items[k]['k'] == kk]
        for c0 in range(0, len(lst), 64):
            h = hcount[L]
            hcount[L] += 1
            chunk = lst[c0:c0 + 64]
            pre = 'layers.%d.' % L
            for i in range(FREQ):
                w = om[i] * kk
                g = 32.0
                p[pre + 'wq'][h, 2 * i, 2 * i] = math.cos(w) * g
                p[pre + 'wq'][h, 2 * i + 1, 2 * i] = -math.sin(w) * g
                p[pre + 'wq'][h, 2 * i, 2 * i + 1] = math.sin(w) * g
                p[pre + 'wq'][h, 2 * i + 1, 2 * i + 1] = math.cos(w) * g
                p[pre + 'wk'][h, 2 * i, 2 * i] = 1.0
                p[pre + 'wk'][h, 2 * i + 1, 2 * i + 1] = 1.0
            for j, k in enumerate(chunk):
                p[pre + 'wv'][h, names[items[k]['src']], j] = 1.0
                p[pre + 'wo'][h, j, names[items[k]['dst']]] = 1.0
            nm = 'shift.L%dh%d' % (L, h)
            for t in ['wq', 'wk']:
                add_group(nm + '.' + t, pre + t, [h, full, {"start": 0, "stop": E}],
                          'previous-token head: pos rotation by -1', 'score peaks exactly at t-1 (gain')
            add_group(nm + '.v', pre + 'wv', [h, full, {"start": 0, "stop": len(chunk)}], 'copy source channels', 'value slot j carries channel j')
            add_group(nm + '.o', pre + 'wo', [h, {"start": 0, "stop": len(chunk)}, full], 'write copied channels as p1: channels', 'moves info one position right')
    for k in range(n):
        it = items[k]
        L = layer_of[k]
        pre = 'layers.%d.' % L
        kind = it['kind']
        if kind == 'shiftcode':
            h = hcount[L]
            hcount[L] += 1
            for i in range(FREQ):
                w = om[i]
                g = 32.0
                p[pre + 'wq'][h, 2 * i, 2 * i] = math.cos(w) * g
                p[pre + 'wq'][h, 2 * i + 1, 2 * i] = -math.sin(w) * g
                p[pre + 'wq'][h, 2 * i, 2 * i + 1] = math.sin(w) * g
                p[pre + 'wq'][h, 2 * i + 1, 2 * i + 1] = math.cos(w) * g
                p[pre + 'wk'][h, 2 * i, 2 * i] = 1.0
                p[pre + 'wk'][h, 2 * i + 1, 2 * i + 1] = 1.0
            for v in active:
                for j in range(C):
                    mw = S['wmap'].get('plus', -1)
                    row = codes[16 + ord('+')] if v == mw else (codes[16 + ord('-')] if v == S['wmap'].get('minus', -1) else (codes[16 + ord('*')] if v == S['wmap'].get('times', -1) else codes[v]))
                    p[pre + 'wv'][h, names[tc(v)], j] = row[j]
            for j in range(C):
                p[pre + 'wo'][h, j, names['p1c_%d' % j]] = 1.0
            nm = 'shiftcode.L%dh%d' % (L, h)
            add_group(nm + '.q', pre + 'wq', [h, full, {"start": 0, "stop": E}], 'previous-token head positions', 'attend t-1')
            add_group(nm + '.k', pre + 'wk', [h, full, {"start": 0, "stop": E}], 'pos keys', 'match shifted pos')
            add_group(nm + '.v', pre + 'wv', [h, full, {"start": 0, "stop": C}], 'token -> 28-dim code', 'codes make every byte matchable')
            add_group(nm + '.o', pre + 'wo', [h, {"start": 0, "stop": C}, full], 'write previous-token code', 'key material for induction')
        elif kind == 'latest':
            h = hcount[L]
            hcount[L] += 1
            nm = 'latest.' + it['tag'] + '.L%dh%d' % (L, h)
            for ch in it['keys']:
                p[pre + 'wk'][h, names[ch], 0] = G0 / U
            p[pre + 'bq'][h, 0] = 8.0
            # null (turn boundary / start) tokens compete with equal weight
            for t in [1, 2]:
                p[pre + 'wk'][h, names[tc(t)], 0] = G0 / U
            p[pre + 'bq'][h, 1] = 8.0 * RHO
            p[pre + 'wk'][h, lowc, 1] = math.cos(phi) * 100.0
            p[pre + 'wk'][h, lowc + 1, 1] = -math.sin(phi) * 100.0
            if it['gate']:
                p[pre + 'wk'][h, names[tc(1)], 0] = 0.0
                p[pre + 'wk'][h, names[tc(2)], 0] = 0.0
                p[pre + 'wk'][h, names[tc(3)], 0] = 0.0
                p[pre + 'wk'][h, names[tc(1)], 2] = 1000.0 / U
                p[pre + 'bq'][h, 2] = 8.0 * 2000000.0 / 1000.0
                for g in it['gate']:
                    p[pre + 'wq'][h, names[g], 2] = -8.0 * (2000000.0 - 500000.0) / 1000.0 / U
            for j, (src, dst) in enumerate(it['vals']):
                p[pre + 'wv'][h, names[src], j] = 1.0
                p[pre + 'wo'][h, j, names[dst]] = 1.0
            if it['valcode']:
                for v in active:
                    if v in SPECIAL:
                        continue
                    for j in range(C):
                        p[pre + 'wv'][h, names[tc(v)], j] = codes[v][j]
                for j in range(C):
                    p[pre + 'wo'][h, j, names['cout_%d' % j]] = 1.0
            nv = max(len(it['vals']), C if it['valcode'] else 0)
            add_group(nm + '.k', pre + 'wk', [h, full, {"start": 0, "stop": 3}], 'latest-of-class: key flag score 40000 plus', 'USER/BOS tokens are null keys so')
            add_group(nm + '.q', pre + 'bq', [h, {"start": 0, "stop": 3}], 'constant query', 'recency weight 3000 and sink')
            if it['gate']:
                add_group(nm + '.gq', pre + 'wq', [h, full, {"start": 0, "stop": 3}], 'gate: mode channel lowers BOS sink', 'head only writes when mode is')
            if nv:
                add_group(nm + '.v', pre + 'wv', [h, full, {"start": 0, "stop": nv}], 'copy value channel or token code', 'value = flag or code(token)')
                add_group(nm + '.o', pre + 'wo', [h, {"start": 0, "stop": nv}, full], 'write to L_ or cout channels', 'flag or emit-code')
        elif kind == 'induct':
            h = hcount[L]
            hcount[L] += 1
            nm = 'induct.' + it['tag'] + '.L%dh%d' % (L, h)
            for v in active:
                for j in range(C):
                    p[pre + 'wq'][h, names[tc(v)], j] = 8.0 * 60000.0 / C * codes[v][j] / U
            for j in range(C):
                p[pre + 'wq'][h, names['p1c_%d' % j], C + j] = 8.0 * 60000.0 / C / U
                p[pre + 'wk'][h, names['p1c_%d' % j], j] = 1.0 / U
                p[pre + 'wk'][h, names['p1:p1c_%d' % j], C + j] = 1.0 / U
            cls = []
            for m in it['modes']:
                if m[1] not in cls:
                    cls.append(m[1])
            for ci, c in enumerate(cls):
                p[pre + 'wk'][h, names[c], 57 + ci] = 1.0 / U
                for m in it['modes']:
                    if m[1] == c:
                        p[pre + 'wk'][h, names['p1:' + m[2]], 57 + ci] = 1.0 / U
                        for kk in range(2, 5 if m[0] not in ('m_cont', 'm_resc') else 2):
                            p[pre + 'wk'][h, names['p1:' * kk + m[2]], 57 + ci] = 0.12 / U
                for m in it['modes']:
                    if m[1] == c:
                        p[pre + 'wq'][h, names[m[0]], 57 + ci] = 8.0 * 60000.0 / U
            p[pre + 'wk'][h, names[tc(1)], 56] = 40.0 / U
            p[pre + 'bq'][h, 56] = 8.0 * 290000.0 / 40.0
            for m in it['modes']:
                p[pre + 'wq'][h, names[m[0]], 56] = -8.0 * (290000.0 - 150000.0) / 40.0 / U
            p[pre + 'bq'][h, 62] = 8.0 * RHO
            for m in it['modes']:
                if m[0] == 'm_txtc':
                    p[pre + 'wq'][h, names[m[0]], 62] = -16.0 * RHO / U
            p[pre + 'wk'][h, lowc, 62] = math.cos(phi)
            p[pre + 'wk'][h, lowc + 1, 62] = -math.sin(phi)
            for v in active:
                if v in SPECIAL:
                    continue
                for j in range(C):
                    src = it['valmap'].get(v, None)
                    row = codes[16 + ord(src)] if src else codes[v]
                    p[pre + 'wv'][h, names[tc(v)], j] = row[j]
            for j in range(C):
                p[pre + 'wo'][h, j, names['cout_%d' % j]] = 1.0
            add_group(nm + '.q', pre + 'wq', [h, full, {"start": 0, "stop": E}], 'induction query: code(cur token) and code(prev', 'find source position whose previous two')
            add_group(nm + '.k', pre + 'wk', [h, full, {"start": 0, "stop": E}], 'induction key: codes of the previous', 'class flag restricts candidates; sink wins')
            add_group(nm + '.bq', pre + 'bq', [h, {"start": 0, "stop": E}], 'sink score and recency', 'latest occurrence wins')
            add_group(nm + '.v', pre + 'wv', [h, full, {"start": 0, "stop": C}], 'value = token code (with word->operator', 'copy / translate token')
            add_group(nm + '.o', pre + 'wo', [h, {"start": 0, "stop": C}, full], 'write code to emit channels', 'unembed decodes it')
        elif kind == 'rule':
            conds = it['conds']
            negs = it['negs']
            a = nrn[L]
            b = a + 1
            nrn[L] += 2
            nc = len(conds)
            for g in conds:
                for c in g:
                    p[pre + 'w_in'][names[c], a] = 1.0
                    p[pre + 'w_in'][names[c], b] = 1.0
            for g in negs:
                for c in g:
                    p[pre + 'w_in'][names[c], a] = -2.0
                    p[pre + 'w_in'][names[c], b] = -2.0
            th = U * (nc - 1) + U / 4.0
            p[pre + 'b_in'][a] = -th
            p[pre + 'b_in'][b] = -th - U / 2.0
            for o in it['outs']:
                if o[0] == 'f':
                    p[pre + 'w_out'][a, names[o[1]]] += 2.0 * o[2]
                    p[pre + 'w_out'][b, names[o[1]]] += -2.0 * o[2]
                else:
                    for j in range(C):
                        p[pre + 'w_out'][a, names['cout_%d' % j]] += 2.0 * o[2] * codes[o[1]][j]
                        p[pre + 'w_out'][b, names['cout_%d' % j]] += -2.0 * o[2] * codes[o[1]][j]
            key = (it['tag'], L)
            if key not in rule_groups:
                rule_groups[key] = []
            rule_groups[key].extend([a, b])
    for key, ns in rule_groups.items():
        tag, L = key
        pre = 'layers.%d.' % L
        nm = 'rules.' + tag + '.L%d' % L
        add_group(nm + '.in', pre + 'w_in', [full, ns], 'conjunction neurons for ' + tag + ': weight 1 per condition channel,', 'noise-free 0/U output')
        add_group(nm + '.b', pre + 'b_in', [ns], 'thresholds', 'fires only when all conditions hold')
        add_group(nm + '.out', pre + 'w_out', [ns, full], 'write feature channel or token code', 'emit strength 1.0 = logit 20,')
    # unembed: decode code space
    for v in active:
        for j in range(C):
            p['unembed'][names['cout_%d' % j], v] = codes[v][j] * 20.0 / (U * C)
    p['b_unembed'][10] = 3.0
    add_group('unembed', 'unembed', None, 'logit(v)=20*cos(code(v), code emitted)', 'greedy argmax picks the emitted token')
    add_group('b_unembed', 'b_unembed', [[10]], 'END default logit 3', 'when nothing fires the turn ends')
    add_group('pos', 'pos', None, 'clean position code', 'sin/cos pairs only')
    add_group('embed', 'embed', None, 'one-hot token channel (amplitude 32) plus', 'identity and class available from layer')
    return cfg, p, groups, S['words']
