"""GenGPT weights, derived by hand. See EXPLANATION.md (operating manual) and ARCHITECTURE.md.

Residual stream = named channels (allocated by `A`). Every block below is registered
with provenance. Layer plan:
 L0 attn: previous-token(1,2,3) pointers, turn pointer U, latest-result pointer RB, segment
 L0 mlp : token-level flags that need context (tool names, user-turn flags, value positions)
 L1 attn: counting/pooling over the user turn, last tool, last file, pending confirm, last answer
 L1 mlp : saturating steps (counts->booleans), ordinal tags
 L2 attn: role copies (first/second file, key, number, word)
 L2 mlp : intent / plan flags
 L3 attn: lookup heads (value after key), result-block copy heads (first, second, induction)
 L3 mlp : reply-type flags
 L4 mlp : emission rules (fixed tokens) and slot gates
 L5 mlp : slot routing (gated copy of a 12-bit slot into the output code)
"""
import math
import torch
from wsp.gen import tokenizer as T, kb
from wsp.gen.prov import group, S
from wsp.bot.custom import CustomArch

D64 = torch.float64
VOC = set(T.vocab())
tid = T.tid


def code(t):
    return [1.0 if (t >> b) & 1 else -1.0 for b in range(12)]


# ------------------------------------------------------------------ lexicon (hand written)
KW = {
    "ls": "files list ls dir directory folder contents everything",
    "del": "delete remove erase rm trash discard drop wipe destroy rid clear nuke kill unlink purge",
    "set": "set change update write edit modify put assign replace adjust alter store rewrite",
    "find": "find search locate grep where which mentions mentioned mention contains contain containing includes appears look who",
    "yes": "yes sure yep yeah ok okay confirm proceed ahead go y",
    "no": "no nope cancel keep stop abort never nah wait dont negative",
    "hello": "hello hi hey howdy greetings hiya yo morning",
    "thanks": "thanks thank thx cheers ty appreciate appreciated",
    "bye": "bye goodbye later farewell cya quit exit",
    "swap": "subtract",
    "dbl": "double twice",
    "cal": "plus add added sum total minus subtract subtracted less difference times multiply multiplied double twice product x",
    "plus": "plus add added sum total",
    "minus": "minus subtract subtracted less difference",
    "times": "times multiply multiplied double twice product x",
}
KWN = list(KW) + ["path"]
EF = (["mk_user", "mk_bot", "mk_tool", "mk_res", "mk_eres", "mk_end", "mk_bos",
       "f_file", "f_key", "f_num", "f_valw", "f_op", "f_path", "f_semi", "k_any"]
      + ["k_" + k for k in KW])
VALW = list(kb.NAMES) + [w for v in kb.WORD_KEYS.values() for w in v]


def emb_flags(i, w):
    f = set()
    sp = {2: "mk_user", 3: "mk_bot", 6: "mk_tool", 8: "mk_res", 9: "mk_eres", 10: "mk_end", 1: "mk_bos"}
    if i in sp:
        f.add(sp[i])
    if w in kb.FILES: f.add("f_file")
    if w in kb.KEYS: f.add("f_key")
    if 12 <= i < 1012: f.add("f_num")
    if w in VALW: f.add("f_valw")
    if w in ("plus", "minus", "times"): f.add("f_op")
    if w in ("/", "..", "~"): f.add("f_path"); f.add("k_any")
    if w == ";": f.add("f_semi")
    for k, s in KW.items():
        if w in s.split():
            f.add("k_" + k); f.add("k_any")
    return f


# ------------------------------------------------------------------ channel allocation
CH, ND = {}, [0]


def A(name, n=1):
    CH[name] = list(range(ND[0], ND[0] + n)); ND[0] += n


A("C", 12); A("NU", 20); A("LIN")
for f in EF: A(f)
A("p1c", 12)
for f in ["p1_key", "p1_ls", "p1_tool", "p1_op", "p1_file", "p1_eres", "p1_res"]: A(f)
A("p2c", 12); A("p2_ls"); A("p2_res"); A("p3c", 12); A("U", 20); A("RB", 20)
for f in ["s_user", "s_bot", "s_tool", "s_res", "s_eres"]: A(f)
TN = ["ls", "read", "grep", "calc", "write", "delete"]
for t in TN: A("tn_" + t)
for f in ["ufile", "ukey", "unum", "uvalw", "ukw", "tf", "valpos", "ansval", "askflag", "er_miss", "er_err", "er_none"]: A(f)
for k in KWN: A("u_" + k)
A("kv1", 12)
for f in ["cnt_file", "cnt_key", "cnt_num", "cnt_valw", "cnt_read"]: A(f)
for k in KWN: A("pk_" + k)
for t in TN: A("lt_" + t)
A("FLc", 12); A("rs_miss"); A("rs_err"); A("rs_none"); A("asked"); A("PF", 12); A("ANSc", 12)
for f in ["nf1", "nf2", "nk1", "nk2", "nn1", "nn2", "nw1", "nrd1", "nrd2", "lt_none", "f1t", "f2t", "k1t", "k2t", "n1t", "n2t"]: A(f)
for k in KWN: A("q_" + k)
A("kvf", 12)
for s in ["F1c", "F2c", "K1c", "K2c", "N1c", "N2c", "Wc"]: A(s, 12)
PLAN = ["unsafe", "delyes", "delno", "conf", "pw", "hello", "thanks", "bye", "pls", "pg", "pr", "pr2", "pk", "pc",
        "A_va", "A_vc", "A_ans", "A_n1", "A_n2", "B_vb", "B_two", "B_n1", "B_n2", "ou_plus", "ou_minus", "ou_times"]
for p in PLAN: A("P_" + p)
A("Va", 12); A("hVa"); A("Vb", 12); A("Vc", 12); A("R1c", 12); A("R2c", 12); A("R2E")
A("INDc", 12); A("INDE")
RT = ["ls", "show", "look", "nokey", "find", "nf", "ans", "cerr", "miss", "write", "del"]
for r in RT: A("rt_" + r)
for r in ["hello", "thanks", "bye", "unk", "unsafe", "conf", "delno"]: A("d_" + r)
CW = ["the", "files", "are", "says", ":", "is", "in", "."]
for w in CW: A("c_" + w)
for w in ["are", ":", "in"]: A("q1_" + w)
SLOTS = ["F1", "F2", "K1", "N1", "N2", "W", "PF", "FL", "ANS", "Va", "Vb", "Vc", "R1", "R2", "IND"]
SLOTCH = {"F1": "F1c", "F2": "F2c", "K1": "K1c", "N1": "N1c", "N2": "N2c", "W": "Wc", "PF": "PF", "FL": "FLc",
          "ANS": "ANSc", "Va": "Va", "Vb": "Vb", "Vc": "Vc", "R1": "R1c", "R2": "R2c", "IND": "INDc"}
for s in SLOTS: A("g_" + s)
A("O", 12)
d = ND[0]

# ------------------------------------------------------------------ heads
HEADS = {l: [] for l in range(6)}
OM = math.log(2.0)
G, GL, AU = 6000.0, 4096.0, 12.0   # flag weight, recency slope, turn-match weight


class Head:
    def __init__(s, layer, name, mech, why):
        s.l, s.name, s.mech, s.why = layer, name, mech, why
        s.wq, s.wk, s.wv, s.wo, s.bq = [], [], [], [], []
        s.e = 0; s.ev = 0
        HEADS[layer].append(s)

    def ne(s):
        s.e += 1; return s.e - 1

    def qk(s, qc, kc, c):
        for a, b in zip(CH[qc] if isinstance(qc, str) else qc, CH[kc] if isinstance(kc, str) else kc):
            e = s.ne(); s.wq.append((a, e, c)); s.wk.append((b, e, 1.0))

    def key(s, items, c):
        e = s.ne(); s.bq.append((e, 1.0))
        for n, w in items:
            s.wk.append((CH[n][0], e, c * w))

    def shift(s, k, g):
        for j in range(10):
            th = 2 * math.pi * (2 ** j) / 1024 * k
            e1, e2 = s.ne(), s.ne()
            cc, ss = CH["NU"][j], CH["NU"][10 + j]
            s.wq += [(cc, e1, g * math.cos(th)), (ss, e1, g * math.sin(th)), (ss, e2, g * math.cos(th)), (cc, e2, -g * math.sin(th))]
            s.wk += [(cc, e1, 1.0), (ss, e2, 1.0)]

    def val(s, src, dst):
        for a, b in zip(CH[src], CH[dst]):
            v = s.ev; s.ev += 1; s.wv.append((a, v, 1.0)); s.wo.append((v, b, 1.0))

    def valsum(s, items, dst):
        v = s.ev; s.ev += 1
        for n, w in items:
            s.wv.append((CH[n][0], v, w))
        s.wo.append((v, CH[dst][0], 1.0))

    def scoped(s, sink):   # same-exchange scope: UM term and sink on <user>
        s.qk("U", "U", AU)
        s.key([("mk_user", 1)], sink)

    def recency(s):
        s.key([("LIN", 1)], GL)


# ---- L0
h = Head(0, "attn.prev1", "pointer to the previous token: copies its code and class flags",
         "score = 8*sum_j cos(w_j (t-1-s)) over 10 octave frequencies; exact offset wins by margin 16")
h.shift(1, 8.0); h.val("C", "p1c")
for f in ["f_key", "f_op", "f_file", "mk_eres", "mk_res", "mk_tool"]:
    h.valsum([(f, 1)], {"f_key": "p1_key", "f_op": "p1_op", "f_file": "p1_file", "mk_eres": "p1_eres", "mk_res": "p1_res", "mk_tool": "p1_tool"}[f])
h.valsum([("f_semi", 1), ("mk_res", 1)], "p1_ls")
h = Head(0, "attn.prev2", "pointer to the token two back (code, line-start flag, result-marker flag)", "same positional-match score with offset 2")
h.shift(2, 8.0); h.val("C", "p2c"); h.valsum([("f_semi", 1), ("mk_res", 1)], "p2_ls"); h.valsum([("mk_res", 1)], "p2_res")
h = Head(0, "attn.prev3", "pointer to the token three back (code only)", "same positional-match score with offset 3")
h.shift(3, 8.0); h.val("C", "p3c")
h = Head(0, "attn.turn_ptr", "every token learns the position code of the latest <user> at or before it", "recency (4/position) plus 6000 on <user>: latest user marker wins")
h.key([("mk_user", 1)], G); h.recency(); h.val("NU", "U")
h = Head(0, "attn.result_ptr", "every token learns the position code of the latest <result> marker", "recency plus 6000 on <result>")
h.key([("mk_res", 1)], G); h.recency(); h.val("NU", "RB")
h = Head(0, "attn.segment", "every token learns which marker came last: user/bot/tool/result/end-of-result", "recency plus 6000 on any of the five markers")
h.key([(m, 1) for m in ["mk_user", "mk_bot", "mk_tool", "mk_res", "mk_eres"]], G); h.recency()
for m, c in [("mk_user", "s_user"), ("mk_bot", "s_bot"), ("mk_tool", "s_tool"), ("mk_res", "s_res"), ("mk_eres", "s_eres")]:
    h.valsum([(m, 1)], c)

# ---- L1
def counter(name, flag, out, mech):
    h = Head(1, name, mech, "scoped to this exchange by U match (+12 vs <=8 elsewhere); flagged tokens score 14 above the rest; the <user> sink scores 14+ln2, so output = n/(n+2): 1/3, 1/2, 3/5 for n=1,2,3")
    h.scoped(14 + OM); h.key([(flag, 1)], 14.0); h.valsum([(flag, 1)], out)

counter("pool.files", "ufile", "cnt_file", "counts file names in the user turn (also an ordinal at each file token)")
counter("pool.keys", "ukey", "cnt_key", "counts key names in the user turn")
counter("pool.nums", "unum", "cnt_num", "counts numbers in the user turn")
counter("pool.words", "uvalw", "cnt_valw", "counts name/word values in the user turn")
counter("pool.reads", "tn_read", "cnt_read", "counts read calls made so far in this exchange")
h = Head(1, "pool.keywords", "presence of each keyword class in the user turn", "sink scores equal a keyword token, so a present class gives n/(N+1) >= 0.14, absent exactly 0")
h.scoped(14.0); h.key([("ukw", 1)], 14.0)
for k in KWN:
    h.valsum([("u_" + k, 1)], "pk_" + k)
h = Head(1, "last.tool", "name of the latest tool call in this exchange (none if no call yet)", "recency among tool-name tokens, scoped to the exchange; sink wins if there is none")
h.scoped(G - AU); h.recency(); h.key([("tn_" + t, 1) for t in TN], G)
for t in TN: h.valsum([("tn_" + t, 1)], "lt_" + t)
h = Head(1, "last.file", "code of the latest file argument of a tool call", "recency among tool-file tokens")
h.key([("tf", 1)], G); h.recency(); h.val("C", "FLc")
h = Head(1, "last.result_status", "status (missing/error/none) of the latest result block", "recency among </result> tokens, which carry the flags")
h.key([("mk_eres", 1)], G); h.recency()
for f in ["miss", "err", "none"]: h.valsum([("er_" + f, 1)], "rs_" + f)
h = Head(1, "last.pending", "did the previous bot turn end with a delete question, and about which file", "latest <end>; its flag says the reply ended in '?', its p2 code is the file")
h.key([("mk_end", 1)], G); h.recency(); h.valsum([("askflag", 1)], "asked"); h.val("p2c", "PF")
h = Head(1, "last.answer", "the number in the latest 'the answer is R'", "recency among answer-value tokens")
h.key([("ansval", 1)], G); h.recency(); h.val("C", "ANSc")

# ---- L2 role copies
def role(name, tag, dst, mech):
    h = Head(2, name, mech, "scoped to this exchange; tagged token scores 14 more; sink on <user> at 7 loses to a tag but beats untagged tokens; absent role copies zeros")
    h.scoped(7.0); h.key([(tag, 1)], 14.0); h.val("C", dst)

role("role.file1", "f1t", "F1c", "first file named by the user")
role("role.file2", "f2t", "F2c", "second file named by the user")
role("role.key1", "k1t", "K1c", "first key named by the user")
role("role.key2", "k2t", "K2c", "second key named by the user")
role("role.num1", "n1t", "N1c", "first number in the user turn")
role("role.num2", "n2t", "N2c", "second number in the user turn")
role("role.word", "uvalw", "Wc", "the name/word value in the user turn (grep word or value to write)")

# ---- L3
GV = 10.0
def lookup(name, kslot, mech, file_match):
    h = Head(3, name, mech, "key code is masked to value positions; full match 12*10 beats sink 11*10 which beats any partial; scope is the latest block (RB) or the exchange")
    h.qk(kslot, "kv1", GV)
    if file_match:
        h.qk("F1c", "kvf", GV); h.qk("U", "U", AU); h.key([("mk_user", 1)], 23 * GV)
    else:
        h.qk("RB", "RB", AU); h.key([("mk_res", 1)], 11 * GV)
    return h

h = lookup("lookup.latest_k1", "K1c", "value after line-start key K1 in the latest result block", False)
h.val("C", "Va"); h.valsum([("valpos", 1)], "hVa")
h = lookup("lookup.latest_k2", "K2c", "value after line-start key K2 in the latest result block", False)
h.val("C", "Vb")
h = lookup("lookup.file1_k1", "K1c", "value of K1 in the block of the file F1 (first of two reads)", True)
h.val("C", "Vc")
h = Head(3, "block.first", "first token of the latest result block", "key flag p1_res (+12) with block match RB (10*12 vs 8*12)")
h.key([("p1_res", 1)], 12.0); h.qk("RB", "RB", AU); h.val("C", "R1c")
h = Head(3, "block.second", "second token of the latest result block (or </result>, flagged)", "key flag p2_res with block match")
h.key([("p2_res", 1)], 12.0); h.qk("RB", "RB", AU); h.val("C", "R2c"); h.valsum([("mk_eres", 1)], "R2E")
h = Head(3, "block.induction", "successor copy: token after the earlier occurrence of (current, previous, before-previous)", "score 48*(code match of cur) + 48*(prev) + 3*(prev2); block match RB; keys limited to tokens inside the block or its </result>")
h.qk("C", "p1c", 48.0); h.qk("p1c", "p2c", 48.0); h.qk("p2c", "p3c", 3.0); h.qk("RB", "RB", AU)
h.key([("s_res", 1), ("mk_eres", 1), ("mk_res", -1)], 12.0)
h.val("C", "INDc"); h.valsum([("mk_eres", 1)], "INDE")

# ------------------------------------------------------------------ MLP units
UN = {l: [] for l in range(6)}   # (ins, bias, outs)
BLK = {l: [] for l in range(6)}  # (start, end, name, mech, why)
cur = [None]


def block(l, name, mech, why):
    BLK[l].append([len(UN[l]), None, name, mech, why])
    if len(BLK[l]) > 1: BLK[l][-2][1] = len(UN[l])


def close(l):
    BLK[l][-1][1] = len(UN[l])


def unit(l, ins, bias, outs):
    UN[l].append((ins, bias, outs))


def eqin(g, t):
    return [(CH[g][b], code(t)[b]) for b in range(12)]


def AND(l, pos=(), neg=(), eqs=(), outs=()):
    ins, th = [], 0
    for c in pos: ins.append((CH[c][0], 1.0)); th += 1
    for g, t in eqs: ins += eqin(g, t); th += 12
    for c in neg: ins.append((CH[c][0], -2.0))
    unit(l, ins, -(th - 1.0), [(CH[o][0], 1.0) for o in outs])


def tokout(t):
    return [(CH["O"][b], code(t)[b]) for b in range(12)]


def step(l, xs, flag_chs, th, K, out, extra=0.0):
    """out = 1 if sum(xs)*... : relu(K(x-th)+K*(flags-nflags)) - relu(... -1)."""
    ins = [(CH[x][0], K) for x in xs] + [(CH[f][0], K) for f in flag_chs]
    b = -K * (th + len(flag_chs))
    unit(l, ins, b, [(CH[out][0], 1.0)]); unit(l, ins, b - 1.0, [(CH[out][0], -1.0)])


# ---- MLP0
block(0, "flags.tool_name", "tn_X = the token right after <tool> equals tool name X", "AND of p1_tool and code equality: 1+12-1 threshold")
for t in TN: AND(0, ["p1_tool"], eqs=[("C", tid(t))], outs=["tn_" + t])
block(0, "flags.user_scope", "u*-flags: class flags valid only for tokens of a user turn", "AND of class flag and s_user: tool results and bot text can never be counted as user words (injection safety)")
for a, b in [("f_file", "ufile"), ("f_key", "ukey"), ("f_num", "unum"), ("f_valw", "uvalw"), ("k_any", "ukw")]:
    AND(0, [a, "s_user"], outs=[b])
for k in KWN: AND(0, ["k_" + k if k != "path" else "f_path", "s_user"], outs=["u_" + k])
block(0, "flags.value_pos", "valpos = a token that follows a line-start key inside a result; tf = file name inside a tool call", "AND of p1_key, p2_ls (';' or <result>) and s_res: only real 'key value' lines, never words inside a note line")
AND(0, ["f_file", "s_tool"], outs=["tf"])
AND(0, ["p1_key", "p2_ls", "s_res"], outs=["valpos"])
AND(0, ["f_num"], eqs=[("p1c", tid("is")), ("p2c", tid("answer"))], outs=["ansval"])
AND(0, ["mk_end"], eqs=[("p1c", tid("?"))], outs=["askflag"])
for f, w in [("miss", "missing"), ("err", "error"), ("none", "none")]:
    AND(0, ["mk_eres"], eqs=[("p1c", tid(w))], outs=["er_" + f])
block(0, "flags.masked_key_code", "kv1 = previous-token code, only at value positions", "relu(+-c + 3(flags-3)) pairs give flag*c exactly; zero elsewhere so the lookup sink is independent of the query")
for b in range(12):
    ins = [(CH["p1_key"][0], 3.0), (CH["p2_ls"][0], 3.0), (CH["s_res"][0], 3.0)]
    unit(0, ins + [(CH["p1c"][b], 1.0)], -9.0, [(CH["kv1"][b], 1.0)])
    unit(0, ins + [(CH["p1c"][b], -1.0)], -9.0, [(CH["kv1"][b], -1.0)])
close(0)

# ---- MLP1
block(1, "steps.counts", "counts -> booleans: nf1/nf2 (>=1,>=2 files), same for keys and numbers, nw1, nrd1/nrd2", "pool outputs are exactly n/(n+2): thresholds 0.17 and 0.42 sit midway between 0, 1/3, 1/2; K=40 saturates")
for c, n in [("cnt_file", "nf"), ("cnt_key", "nk"), ("cnt_num", "nn")]:
    step(1, [c], [], 0.17, 40.0, n + "1"); step(1, [c], [], 0.42, 40.0, n + "2")
step(1, ["cnt_valw"], [], 0.17, 40.0, "nw1")
step(1, ["cnt_read"], [], 0.17, 40.0, "nrd1"); step(1, ["cnt_read"], [], 0.42, 40.0, "nrd2")
block(1, "steps.keywords", "q_X = keyword class X present in the user turn", "pool value >= 1/(N+1) >= 0.14 when present, exactly 0 otherwise; threshold 0.05")
for k in KWN: step(1, ["pk_" + k], [], 0.05, 100.0, "q_" + k)
block(1, "steps.ordinal_tags", "f1t/f2t etc: first vs second file/key/number token of the user turn", "a token's own count is 1/3 if first, 1/2 if second; f2t = flag AND count>0.42, f1t = flag - f2t")
for c, f, t1, t2 in [("cnt_file", "ufile", "f1t", "f2t"), ("cnt_key", "ukey", "k1t", "k2t"), ("cnt_num", "unum", "n1t", "n2t")]:
    step(1, [c], [f], 0.42, 40.0, t2)
    unit(1, [(CH[f][0], 1.0)], 0.0, [(CH[t1][0], 1.0)])
    unit(1, [(CH[c][0], 40.0), (CH[f][0], 40.0)], -40 * 1.42, [(CH[t1][0], -1.0)]); unit(1, [(CH[c][0], 40.0), (CH[f][0], 40.0)], -40 * 1.42 - 1, [(CH[t1][0], 1.0)])
block(1, "flags.no_tool_yet", "lt_none = no tool call yet in this exchange", "1 - sum of lt flags")
unit(1, [(CH["lt_" + t][0], -1.0) for t in TN], 1.0, [(CH["lt_none"][0], 1.0)])
block(1, "flags.masked_file_code", "kvf = last-file code, only at value positions", "same gate trick as kv1")
for b in range(12):
    unit(1, [(CH["valpos"][0], 3.0), (CH["FLc"][b], 1.0)], -3.0, [(CH["kvf"][b], 1.0)])
    unit(1, [(CH["valpos"][0], 3.0), (CH["FLc"][b], -1.0)], -3.0, [(CH["kvf"][b], -1.0)])
close(1)

# ---- MLP2 intents
P = lambda n: "P_" + n
def plan(out, pos, neg=()):
    AND(2, pos, neg, outs=[P(out)])
block(2, "intent.plan", "plan flags: what the user wants and which tools/replies follow", "each is an AND/NOT of keyword and count booleans; priority comes from the NOT terms (path beats everything; delete/set beat read)")
NS = ["q_path", "q_del", "q_set"]
plan("unsafe", ["q_path"])
plan("delyes", ["asked", "q_yes"], ["q_path", "q_no"])
plan("delno", ["asked", "q_no"], ["q_path"])
plan("conf", ["q_del", "nf1"], ["q_path"])
plan("pw", ["q_set", "nf1", "nk1", "nn1"], ["q_path", "q_del"])
plan("pw", ["q_set", "nf1", "nk1", "nw1"], ["q_path", "q_del", "nn1"])
for k in ["hello", "thanks", "bye"]: plan(k, ["q_" + k], ["q_path"])
plan("pls", ["q_ls"], ["q_path", "nf1", "q_del", "q_find"])
plan("pg", ["q_find", "nw1"], NS[:1] + ["nf1", "q_del", "q_set"])
plan("pr", ["nf1"], NS)
plan("pr", ["q_find", "nw1", "nk1"], NS + ["nf1"])
plan("pr2", ["nf2", "nk2"], NS)
plan("pk", ["nk1", "nf1"], NS)
plan("pk", ["nk1", "q_find", "nw1"], NS + ["nf1"])
plan("pc", ["q_cal", "nk1"], NS)
plan("pc", ["q_cal", "nn1"], NS + ["nk1"])
plan("pc", ["q_cal", "q_dbl"], NS + ["nk1", "nn1"])
plan("A_va", ["q_cal", "nk1"], NS + ["nk2"])
plan("A_vc", ["q_cal", "nk2"], NS)
plan("A_ans", ["q_cal"], NS + ["nk1", "nn2"])
plan("A_n1", ["q_cal", "nn2"], NS + ["nk1", "q_swap"])
plan("A_n2", ["q_cal", "nn2", "q_swap"], NS + ["nk1"])
plan("B_vb", ["q_cal", "nk2"], NS)
plan("B_two", ["q_cal", "q_dbl"], NS + ["nn1", "nk1"])
plan("B_n1", ["q_cal", "nn1"], NS + ["nn2", "nk2"])
plan("B_n1", ["q_cal", "nn2", "q_swap"], NS + ["nk2"])
plan("B_n2", ["q_cal", "nn2"], NS + ["q_swap", "nk2"])
for o in ["plus", "minus", "times"]: plan("ou_" + o, ["q_cal", "q_" + o], NS)
close(2)

# ---- MLP3 reply types
block(3, "reply.type", "rt_X = which reply template applies (after a tool) and d_X (direct replies)", "AND of last tool, result status, plan and lookup-hit flags; exclusive by construction; unk = no other intent")
E, Bt = "s_eres", "s_bot"
def rt(n, pos, neg=()): AND(3, [E] + pos, neg, outs=["rt_" + n])
rt("ls", ["lt_ls"]); rt("show", ["lt_read"], ["rs_miss", "P_pk", "P_pc"])
rt("look", ["lt_read", "P_pk", "hVa"], ["rs_miss", "P_pc"]); rt("nokey", ["lt_read", "P_pk"], ["rs_miss", "P_pc", "hVa"])
rt("find", ["lt_grep"], ["rs_none", "P_pr"]); rt("nf", ["lt_grep", "rs_none"])
rt("ans", ["lt_calc"], ["rs_err"]); rt("cerr", ["lt_calc", "rs_err"])
for t in ["read", "write", "delete"]: rt("miss", ["rs_miss", "lt_" + t])
rt("write", ["lt_write"], ["rs_miss"]); rt("del", ["lt_delete"], ["rs_miss"])
for a, b in [("hello", "hello"), ("thanks", "thanks"), ("bye", "bye"), ("unsafe", "unsafe"), ("conf", "conf"), ("delno", "delno")]:
    AND(3, [Bt, "P_" + b], outs=["d_" + a])
unit(3, [(CH[Bt][0], 1.0)] + [(CH["P_" + p][0], -1.0) for p in ["unsafe", "delyes", "delno", "conf", "pw", "hello", "thanks", "bye", "pls", "pg", "pr", "pc"]], 0.0, [(CH["d_unk"][0], 1.0)])
block(3, "flags.template_words", "c_w / q1_w: current token (or previous token) equals template word w", "code equality: dot-11 relu")
for w in CW: AND(3, [], eqs=[("C", tid(w))], outs=["c_" + w])
for w in ["are", ":", "in"]: AND(3, [], eqs=[("p1c", tid(w))], outs=["q1_" + w])
close(3)

# ---- MLP4 emission rules and gates
block(4, "emit.rules", "next-token rules: fixed tokens go straight into the output code O; slot gates g_S switch on a copy", "each rule is an AND over (current token, previous token, state flags); exactly one rule fires per position, so O is exactly the code of the next token")
GATE = lambda s: [("g_" + s, 1.0)]


def R(pos=(), neg=(), cur=None, p1=None, tok=None, gate=None):
    eqs = ([("C", tid(cur))] if cur else []) + ([("p1c", tid(p1))] if p1 else [])
    ins, th = [], 0
    for c in pos: ins.append((CH[c][0], 1.0)); th += 1
    for g, t in eqs: ins += eqin(g, t); th += 12
    for c in neg: ins.append((CH[c][0], -2.0))
    outs = tokout(tid(tok)) if tok else [(CH["g_" + gate][0], 1.0)]
    unit(4, ins, -(th - 1.0), outs)


def reply(flag, items, start="bot"):
    """Transitions of one reply template. items: literal words or [SLOT]."""
    n = len(items)
    for i in range(-1, n - 1):
        nxt = items[i + 1]
        pos, neg, cu, p1 = [flag], [], None, None
        if i == -1:
            if start == "bot": cu = "<bot>"; pos = [flag]
            else: pos = [flag, "mk_eres"]
        else:
            it = items[i]
            prev = items[i - 1] if i > 0 else None
            if it.startswith("["):
                cls = {"K1": "f_key", "FL": "f_file", "PF": "f_file", "R1": "f_num"}.get(it[1:-1])
                if cls: pos.append(cls)
                if prev and not prev.startswith("["): p1 = prev
                if i == 0: pos.append("p1_eres")
            else:
                cu = it
                if prev and not prev.startswith("[") and items.count(it) > 1: p1 = prev
        if nxt.startswith("["):
            s = nxt[1:-1]
            if s == "VAL":
                R(pos + ["P_pw", "nn1"], neg, cu, p1, gate="N1"); R(pos + ["P_pw"], neg + ["nn1"], cu, p1, gate="W")
            else:
                R(pos, neg, cu, p1, gate=s)
        else:
            R(pos, neg, cu, p1, tok=nxt)


sp = lambda s: s.split()
reply("d_hello", sp("hello ! i can list , read , search and edit files in this project , and do math ."))
reply("d_thanks", sp("you are welcome ."))
reply("d_bye", sp("goodbye ."))
reply("d_unk", sp("sorry , i can only help with files in this project and with math ."))
reply("d_unsafe", sp("i can only access files inside this project ."))
reply("d_conf", sp("are you sure you want to delete [F1] ?"))
reply("d_delno", sp("ok , i kept [PF] ."))
reply("rt_ls", sp("the files are"), "res")
reply("rt_show", sp("[FL] says :"), "res")
reply("rt_look", sp("the [K1] in [FL] is [Va] ."), "res")
reply("rt_nokey", sp("[FL] has no [K1] ."), "res")
reply("rt_find", sp("[W] is in"), "res")
reply("rt_nf", sp("no file mentions [W] ."), "res")
reply("rt_ans", sp("the answer is [R1] ."), "res")
reply("rt_cerr", sp("i could not compute that ."), "res")
reply("rt_miss", sp("i could not find [FL] ."), "res")
reply("rt_write", sp("i set [K1] in [FL] to [VAL] ."), "res")
reply("rt_del", sp("i deleted [FL] ."), "res")
# 'i could not' + compute/find: the shared prefix is resolved by the flags rt_cerr/rt_miss. (handled above)
# content copy (ls list, show content, find list)
for r, trig, tmpl in [("rt_ls", "are", ["c_the", "c_files", "c_are"]), ("rt_show", ":", ["c_says", "c_:"]), ("rt_find", "in", ["c_is", "c_in"])]:
    cn = "c_" + trig; qn = "q1_" + trig
    R([r, cn], [], gate="R1")
    R([r, qn], ["R2E"], gate="R2"); R([r, qn, "R2E"], [], tok=".")
    base_neg = [qn, "p1_eres", "mk_eres", "c_."] + tmpl
    R([r], base_neg + ["INDE"], gate="IND"); R([r, "INDE"], base_neg, tok=".")
# end of reply
R([], [], cur=".", tok="<end>"); R([], [], cur="?", tok="<end>")
# <bot>: first action
R(["P_pg"], [], cur="<bot>", tok="<tool>"); R(["P_pr"], ["P_pg"], cur="<bot>", tok="<tool>")
R(["P_pls"], [], cur="<bot>", tok="<tool>"); R(["P_pw"], [], cur="<bot>", tok="<tool>")
R(["P_delyes"], [], cur="<bot>", tok="<tool>"); R(["P_pc"], ["P_pr", "P_pg"], cur="<bot>", tok="<tool>")
# <tool>: which tool
T_ = ["s_tool"]
R(T_ + ["P_pls"], [], cur="<tool>", tok="ls"); R(T_ + ["P_pw"], [], cur="<tool>", tok="write")
R(T_ + ["P_delyes"], [], cur="<tool>", tok="delete"); R(T_ + ["P_pg", "lt_none"], [], cur="<tool>", tok="grep")
R(T_ + ["P_pr", "lt_none"], ["P_pg"], cur="<tool>", tok="read"); R(T_ + ["P_pr", "P_pg", "lt_grep"], [], cur="<tool>", tok="read")
R(T_ + ["P_pr2", "lt_read", "nrd1"], ["nrd2"], cur="<tool>", tok="read")
R(T_ + ["P_pc", "lt_none"], ["P_pr", "P_pg"], cur="<tool>", tok="calc")
R(T_ + ["P_pc", "lt_read"], ["P_pr2"], cur="<tool>", tok="calc"); R(T_ + ["P_pc", "lt_read", "P_pr2", "nrd2"], [], cur="<tool>", tok="calc")
# arguments
R(T_ + ["P_pg"], [], cur="read", gate="R1"); R(T_, ["P_pg", "nrd2"], cur="read", gate="F1"); R(T_ + ["nrd2"], ["P_pg"], cur="read", gate="F2")
R(T_, [], cur="grep", gate="W"); R(T_, [], cur="write", gate="F1"); R(T_, [], cur="delete", gate="PF"); R(T_, [], cur="ls", tok="</tool>")
R(T_ + ["P_A_va"], ["P_pr2"], cur="calc", gate="Va"); R(T_ + ["P_A_va", "P_pr2"], [], cur="calc", gate="Vc")
R(T_ + ["P_A_vc"], [], cur="calc", gate="Vc"); R(T_ + ["P_A_ans"], [], cur="calc", gate="ANS")
R(T_ + ["P_A_n1"], [], cur="calc", gate="N1"); R(T_ + ["P_A_n2"], [], cur="calc", gate="N2")
R(T_ + ["f_file"], [], p1="read", tok="</tool>"); R(T_ + ["f_file"], [], p1="delete", tok="</tool>"); R(T_ + ["f_file"], [], p1="write", gate="K1")
R(T_, [], p1="grep", tok="</tool>")
R(T_ + ["f_key", "p1_file", "P_pw", "nn1"], [], gate="N1"); R(T_ + ["f_key", "p1_file", "P_pw"], ["nn1"], gate="W")
R(T_ + ["p1_key"], [], tok="</tool>")
for o in ["plus", "minus", "times"]: R(T_ + ["f_num", "P_ou_" + o], [], p1="calc", tok=o)
R(T_ + ["f_op", "P_B_vb"], [], gate="Vb"); R(T_ + ["f_op", "P_B_n1"], [], gate="N1"); R(T_ + ["f_op", "P_B_n2"], [], gate="N2")
R(T_ + ["f_op", "P_B_two"], [], tok="2"); R(T_ + ["f_num", "p1_op"], [], tok="</tool>")
# after a result: continue with another tool or start the reply
E_ = ["mk_eres"]
R(E_ + ["lt_grep", "P_pr"], ["rs_none"], tok="<tool>")
R(E_ + ["lt_read", "P_pr2", "nrd1"], ["rs_miss", "nrd2"], tok="<tool>")
R(E_ + ["lt_read", "P_pc"], ["rs_miss", "P_pr2"], tok="<tool>"); R(E_ + ["lt_read", "P_pc", "P_pr2", "nrd2"], ["rs_miss"], tok="<tool>")
close(4)

# ---- MLP5 routing
block(5, "route.slots", "O += slot code when its gate is on (a gated copy of a 12-bit code)", "per bit two units relu(+-s + 3(g-1)): equals g*s for s in [-1,1], zero when g=0")
for s in SLOTS:
    for b in range(12):
        for sg in (1.0, -1.0):
            unit(5, [(CH[SLOTCH[s]][b], sg), (CH["g_" + s][0], 3.0)], -3.0, [(CH["O"][b], sg)])
close(5)

# ------------------------------------------------------------------ materialise
def build():
    p = {}
    nF = len(EF)
    tab = torch.zeros(4096, nF, dtype=D64)
    for i, w in enumerate(T.vocab()):
        for f in emb_flags(i, w): tab[i, EF.index(f)] = 1.0
    p["tab"] = tab
    wt = torch.zeros(nF, d, dtype=D64)
    for j, f in enumerate(EF): wt[j, CH[f][0]] = 1.0
    p["wt"] = wt
    wc = torch.zeros(12, d, dtype=D64)
    for b in range(12): wc[b, CH["C"][b]] = 1.0
    p["wc"] = wc
    wp = torch.zeros(21, d, dtype=D64)
    for j in range(20): wp[j, CH["NU"][j]] = 1.0
    wp[20, CH["LIN"][0]] = 1.0
    p["wp"] = wp
    wu = torch.zeros(d, 12, dtype=D64)
    for b in range(12): wu[CH["O"][b], b] = 1.0
    p["wu"] = wu
    groups = [
        group("embed.flags", "tab", S[:, :], mechanism="per-token class and keyword flags (file, key, number, value word, op, path, markers, keyword classes)", why="hand-written lexicon: spec lists plus synonyms I chose; each 1 marks membership"),
        group("embed.placement", "wt", S[:, :], mechanism="maps each flag column to its residual channel", why="identity placement, one 1 per flag"),
        group("embed.code", "wc", S[:, :], mechanism="places the 12-bit token-id code (+-1 per bit) in channels C", why="the id bits give every token a unique code; equality of two codes is dot = 12"),
        group("embed.position", "wp", S[:, :], mechanism="places the fixed sin/cos position code and linear position in channels NU, LIN", why="octave frequencies 2pi*2^j/1024 give exact position match with margin 2 per unit weight"),
        group("unembed.code", "wu", S[:, :], mechanism="reads the output code channels O; logits = O . code(token)", why="O equals the code of the intended token, which then has logit 12 vs <=10 for any other"),
    ]
    for l in range(6):
        hs = HEADS[l]
        p[f"L{l}.nh"] = torch.zeros(len(hs), dtype=D64)
        for i, hd in enumerate(hs):
            e, ev = max(hd.e, 1), max(hd.ev, 1)
            wq, wk, wv, wo, bq = torch.zeros(d, e, dtype=D64), torch.zeros(d, e, dtype=D64), torch.zeros(d, ev, dtype=D64), torch.zeros(ev, d, dtype=D64), torch.zeros(e, dtype=D64)
            for a, b, c in hd.wq: wq[a, b] += c
            for a, b, c in hd.wk: wk[a, b] += c
            for a, b, c in hd.wv: wv[a, b] += c
            for a, b, c in hd.wo: wo[a, b] += c
            for b, c in hd.bq: bq[b] += c
            pf = f"L{l}.h{i}."
            p[pf + "wq"], p[pf + "wk"], p[pf + "wv"], p[pf + "wo"], p[pf + "bq"] = wq, wk, wv, wo, bq
            for t in ["wq", "wk", "wv", "wo", "bq"]:
                groups.append(group(hd.name, pf + t, ..., mechanism=hd.mech, why=hd.why))
        us = UN[l]
        m = len(us)
        wi, bi, wo_ = torch.zeros(d, m, dtype=D64), torch.zeros(m, dtype=D64), torch.zeros(m, d, dtype=D64)
        for j, (ins, b, outs) in enumerate(us):
            for c, w in ins: wi[c, j] += w
            bi[j] = b
            for c, w in outs: wo_[j, c] += w
        p[f"L{l}.w_in"], p[f"L{l}.b_in"], p[f"L{l}.w_out"] = wi, bi, wo_
        for a, b, name, mech, why in BLK[l]:
            groups.append(group(name, f"L{l}.w_in", S[:, a:b], mechanism=mech, why=why))
            groups.append(group(name, f"L{l}.b_in", S[a:b], mechanism=mech, why=why))
            groups.append(group(name, f"L{l}.w_out", S[a:b, :], mechanism=mech, why=why))
    return CustomArch("experiments/gen/agents/g100-audit/arch.py", "attention network with explicit pointer, pooling and copy heads plus rule MLPs; fixed bit-code embeddings"), p, groups
