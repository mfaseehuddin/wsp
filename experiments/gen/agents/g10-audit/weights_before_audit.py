"""GenGPT g10: hand-derived weights.  See EXPLANATION.md / ARCHITECTURE.md.
Everything is built from a channel registry (named slices of the residual stream),
heads (attention circuits) and neurons (ReLU units) added by small helpers."""
import torch
from wsp.bot.custom import CustomArch
from wsp.gen import tokenizer as T, kb
from wsp.gen.prov import group, S

F64 = torch.float64
V = 4096
GC = 4000.0  # key gain: a flagged key beats any recency/earliest tie-break inside one turn

# ----------------------------------------------------------------- lexicon (hand written)
FILES, KEYS = kb.FILES, kb.KEYS
VALS = list(kb.NAMES) + [w for ws in kb.WORD_KEYS.values() for w in ws]
LEX = {
    "f_opP": "plus add sum total",
    "f_opM": "minus subtract less",
    "f_opT": "times multiply multiplied double twice",
    "f_dbl": "double twice",
    "f_swap": "subtract from",
    "f_del": "delete remove erase drop",
    "f_yes": "yes yep yeah sure ok okay confirm proceed",
    "f_hello": "hello hi hey greetings howdy",
    "f_thanks": "thanks thank thx cheers",
    "f_bye": "bye goodbye farewell later",
    "f_ls": "files list ls",
    "f_path": "/ .. ~",
    "f_qmark": "?", "f_sep": ";", "f_missing": "missing", "f_none": "none",
    "f_error": "error", "f_read": "read", "f_answer": "answer", "f_is": "is",
    "f_endp": ". ?",
}
SPECIAL_FLAG = {"f_user": "<user>", "f_bot": "<bot>", "f_tool": "<tool>", "f_toolend": "</tool>",
                "f_res": "<result>", "f_resend": "</result>", "f_end": "<end>"}
CLASSES = ["file", "key", "num", "val", "opP", "opM", "opT", "dbl", "swap", "del", "yes",
           "hello", "thanks", "bye", "ls", "path"]
STATIC = ["code"] + list(SPECIAL_FLAG) + ["f_file", "f_key", "f_num", "f_val"] + \
    [k for k in LEX if k not in ("f_qmark",)] + ["f_qmark", "f_content", "f_one"]
STATIC = list(dict.fromkeys(STATIC))

# ----------------------------------------------------------------- channel registry
CH = {}
_n = [0]


def add(name, w=1):
    CH[name] = (_n[0], w)
    _n[0] += w


for s in STATIC:
    add(s, 12 if s == "code" else 1)
K = _n[0]
add("posv")
for n_, w_ in [("nbot", 1), ("nres", 1), ("p1c", 12), ("p1_read", 1), ("p1_res", 1), ("p1_sep", 1), ("p1_is", 1),
               ("p2_tool", 1), ("p2_sep", 1), ("p2_res", 1), ("p2_q", 1), ("p2_answer", 1), ("p3c", 12),
               ("Pst", 1), ("Pre", 1), ("in_user", 1)] + [("u_" + c, 1) for c in CLASSES] + \
        [("nh", 4), ("jh", 20), ("jg3", 1)] + [("n_" + c, 1) for c in CLASSES] + \
        [("pend", 1), ("PFc", 12)] + [(n + "c", 12) for n in ("UF1", "UF2", "UK1", "UK2", "UN1", "UN2", "UW", "UV")] + \
        [(n, 1) for n in ("bF1", "bF2", "bK1", "bK2", "bN1", "bN2", "bV1", "bOP", "bOPP", "bOPM", "bOPT", "bDBL",
                          "bSW", "bDEL", "bYES", "bHELLO", "bTHANKS", "bBYE", "bLS", "bPATH", "bVAL")] + \
        [("LVa", 12), ("fu_a", 1), ("LVb", 12), ("fu_b", 1), ("LVc", 12), ("fu_c", 1), ("FREAD", 12),
         ("Gc", 12), ("g_missing", 1), ("g_none", 1), ("g_error", 1), ("LAST", 12), ("CPc", 12), ("cp_end", 1)]:
    add(n_, w_)
INTENTS = ["delyes", "kept", "unsafe", "confirm", "write", "sum2", "lcalc", "lookup", "show", "flcalc", "flook",
           "find", "calc", "follow", "ls", "hello", "thanks", "bye"]
for i_ in INTENTS:
    add("I_" + i_)

# step templates: items are literal words, ("v", var) slots, or ("copy",)
TL = lambda *a: ["<tool>", *a, "</tool>"]
TEMPL = {
    "tl_ls": TL("ls"), "tl_read1": TL("read", ("v", "UF1")), "tl_read2": TL("read", ("v", "UF2")),
    "tl_readG": TL("read", ("v", "G")), "tl_grep": TL("grep", ("v", "UW")),
    "tl_calc": TL("calc", ("v", "UN1"), ("v", "OP"), ("v", "UN2")),
    "tl_calcs": TL("calc", ("v", "UN2"), ("v", "OP"), ("v", "UN1")),
    "tl_calcv": TL("calc", ("v", "LVa"), ("v", "OP"), ("v", "UN1")),
    "tl_calcsum": TL("calc", ("v", "LVc"), ("v", "OP"), ("v", "LVb")),
    "tl_calcf1": TL("calc", ("v", "LAST"), ("v", "OP"), ("v", "UN1")),
    "tl_calcf2": TL("calc", ("v", "LAST"), ("v", "OP"), "2"),
    "tl_write": TL("write", ("v", "UF1"), ("v", "UK1"), ("v", "UV")),
    "tl_delete": TL("delete", ("v", "PF")),
    "r_hello": "hello ! i can list , read , search and edit files in this project , and do math .".split(),
    "r_thanks": "you are welcome .".split(), "r_bye": ["goodbye", "."],
    "r_unknown": "sorry , i can only help with files in this project and with math .".split(),
    "r_unsafe": "i can only access files inside this project .".split(),
    "r_ls": ["the", "files", "are", ("copy",)], "r_show": [("v", "UF1"), "says", ":", ("copy",)],
    "r_missing": ["i", "could", "not", "find", ("v", "UF1"), "."],
    "r_lookup": ["the", ("v", "UK1"), "in", ("v", "FREAD"), "is", ("v", "LVa"), "."],
    "r_nokey": [("v", "FREAD"), "has", "no", ("v", "UK1"), "."],
    "r_find": [("v", "UW"), "is", "in", ("copy",)],
    "r_notfound": ["no", "file", "mentions", ("v", "UW"), "."],
    "r_answer": ["the", "answer", "is", ("v", "G"), "."],
    "r_calcerr": "i could not compute that .".split(),
    "r_write": ["i", "set", ("v", "UK1"), "in", ("v", "UF1"), "to", ("v", "UV"), "."],
    "r_confirm": ["are", "you", "sure", "you", "want", "to", "delete", ("v", "UF1"), "?"],
    "r_deleted": ["i", "deleted", ("v", "PF"), "."], "r_kept": ["ok", ",", "i", "kept", ("v", "PF"), "."],
}
for t_ in list(TEMPL) + ["r_unknown"]:
    add("T_" + t_)
VARS = {"UF1": "UF1c", "UF2": "UF2c", "UK1": "UK1c", "UW": "UWc", "UN1": "UN1c", "UN2": "UN2c", "UV": "UVc",
        "LVa": "LVa", "LVb": "LVb", "LVc": "LVc", "FREAD": "FREAD", "G": "Gc", "LAST": "LAST", "PF": "PFc",
        "COPY": "CPc"}
for v_ in list(VARS) + ["OP"]:
    add("S_" + v_)
add("out", 12)
D = _n[0]


def ci(s):
    if isinstance(s, str):
        return CH[s][0]
    return CH[s[0]][0] + s[1]


def vec(ch, dst0, w=1.0, width=12):
    return [((ch, i), dst0 + i, w) for i in range(width)]


# ----------------------------------------------------------------- layers
LAYERS = [{"e": 16, "heads": [], "neurons": []}, {"e": 16, "heads": [], "neurons": []},
          {"e": 18, "heads": [], "neurons": []}] + [{"e": 1, "heads": [], "neurons": []} for _ in range(3)]
GROUPS = []
WHY = {}


def head(L, name, mech, why, q=(), k=(), v=(), o=(), hp=(0, 0, 0, 0, 0), dvec=()):
    LAYERS[L]["heads"].append(dict(name=f"head.{L}.{name}", q=q, k=k, v=v, o=o, hp=hp, dvec=dvec, mech=mech, why=why))


def neuron(L, grp, mech, why, ins, bias, outs):
    LAYERS[L]["neurons"].append(dict(grp=grp, ins=ins, bias=bias, outs=outs))
    WHY[grp] = (mech, why)


def lit_terms(lits):
    ins, bias = [], 0.0
    for spec, sign in lits:
        items = [(spec, 1.0)] if isinstance(spec, (str, tuple)) else spec
        if sign > 0:
            ins += [(s, w) for s, w in items]
        else:
            ins += [(s, -w) for s, w in items]
            bias += 1.0
    return ins, bias - (len(lits) - 1)


def AND(L, grp, mech, why, lits, outs):
    ins, b = lit_terms(lits)
    neuron(L, grp, mech, why, ins, b, outs)


def code_of(tok):
    i = T.tid(tok)
    return [(("out", b), 1.0 if (i >> b) & 1 else -1.0) for b in range(12)]


# ---- L0 attention
head(0, "count_turn", "counting head: how many <bot> and </result> tokens so far in this turn",
     "ssum=1,local=1 sums the value over past tokens of the same turn; tells where in the turn we are (nbot) and how many tool steps are done (nres)",
     v=[("f_bot", 0, 1.0), ("f_resend", 1, 1.0)], o=[(0, "nbot", 1.0), (1, "nres", 1.0)], hp=(0, 0, 0, 1, 1))
head(0, "prev1", "previous-token head (lag 1): copies code and a few flags of token t-1",
     "A=20,shift=1 puts weight e^20 on the key exactly one step back",
     v=vec("code", 0) + [("f_read", 12, 1.0), ("f_res", 13, 1.0), ("f_sep", 14, 1.0), ("f_is", 15, 1.0)],
     o=[(i, ("p1c", i), 1.0) for i in range(12)] + [(12, "p1_read", 1.0), (13, "p1_res", 1.0), (14, "p1_sep", 1.0), (15, "p1_is", 1.0)],
     hp=(20, 1, 0, 0, 0))
head(0, "prev2", "lag-2 head: flags of token t-2 (tool, ';', <result>, '?', 'answer')", "A=20,shift=2",
     v=[("f_tool", 0, 1.0), ("f_sep", 1, 1.0), ("f_res", 2, 1.0), ("f_qmark", 3, 1.0), ("f_answer", 4, 1.0)],
     o=[(0, "p2_tool", 1.0), (1, "p2_sep", 1.0), (2, "p2_res", 1.0), (3, "p2_q", 1.0), (4, "p2_answer", 1.0)], hp=(20, 2, 0, 0, 0))
head(0, "prev3", "lag-3 head: code of token t-3 (the file in 'delete F ?' seen from the next <user>)", "A=20,shift=3",
     v=vec("code", 0), o=[(i, ("p3c", i), 1.0) for i in range(12)], hp=(20, 3, 0, 0, 0))
for nm, flags, dst in [("step_start", ["f_bot", "f_resend"], "Pst"), ("result_start", ["f_res"], "Pre")]:
    head(0, nm, f"position of the latest {'<bot>/</result>' if dst == 'Pst' else '<result>'} token (written to {dst})",
         "key gain 16000 beats recency R*s (<=12*1023); R=12 makes the latest flagged token win by e^36 over the previous one",
         q=[("f_one", 0, 1.0)], k=[(f, 0, 16000.0) for f in flags], v=[("posv", 0, 1.0)], o=[(0, dst, 1.0)], hp=(0, 0, 12, 0, 0))

# ---- M0
neuron(0, "m0.in_user", "in_user = 1 until <bot> of this turn", "relu(1-nbot): nbot is 0 for user tokens, 1 from <bot> on",
       [("nbot", -1.0)], 1.0, [("in_user", 1.0)])
for c in CLASSES:
    AND(0, "m0.user_class", "u_X = class flag X AND token is in the user part of the turn",
        "relu(f-nbot) = AND(f, in_user) because in_user = 1-nbot (read nbot directly: same-layer units cannot read each other), so bot-side tokens (file names in tool args) are not counted as user words",
        [(("f_" + c if c in ("file", "key", "num", "val") else {"path": "f_path"}.get(c, "f_" + c)), 1), ("nbot", -1)], [("u_" + c, 1.0)])
for r in range(4):
    for dlt, w in [(1, 1.0), (0, -2.0), (-1, 1.0)]:
        neuron(0, "m0.nres_onehot", "nh[r] = 1 iff nres == r (tent function of an integer)",
               "relu(n-r+1)-2relu(n-r)+relu(n-r-1) is exactly 1 at n=r and 0 at every other integer",
               [("nres", 1.0)], -r + dlt, [(("nh", r), w)])
for kk in range(20):
    for dlt, w in [(1, 1.0), (0, -2.0), (-1, 1.0)]:
        outs = [(("jh", kk), w)] + ([("jg3", w)] if kk == 3 and dlt != -1 else [])
        if kk == 3 and dlt == -1:
            outs = [(("jh", kk), w)]
        neuron(0, "m0.j_onehot", "jh[k] = 1 iff j==k where j = pos - Pst = index inside the current step (jg3 = j>=3)",
               "same tent trick on the integer j=posv-Pst; jg3=relu(j-2)-relu(j-3) reuses two of the k=3 units",
               [("posv", 1.0), ("Pst", -1.0)], -kk + dlt, outs)
# jg3 = relu(j-2) - relu(j-3): units kk=3 with dlt=+1 -> relu(j-2) (w=+1) ok; dlt=0 -> relu(j-3) with w=-2 -> need -1
LAYERS[0]["neurons"] = [n for n in LAYERS[0]["neurons"]]
for n in LAYERS[0]["neurons"]:
    for idx_, (s, w) in enumerate(n["outs"]):
        if s == "jg3" and w == -2.0:
            n["outs"][idx_] = ("jg3", -1.0)

# ---- L1 attention
head(1, "count_user_words", "counting head over the user's words (gated class flags u_X) -> n_X counts",
     "an exact count (not a mean), so 'two files named' is a threshold on an integer",
     v=[(("u_" + c), i, 1.0) for i, c in enumerate(CLASSES)], o=[(i, "n_" + c, 1.0) for i, c in enumerate(CLASSES)], hp=(0, 0, 0, 1, 1))
head(1, "pending", "reads this turn's <user> token: did the previous bot reply end in '?' (pending delete) and which file", 
     "only <user> has f_user; its lag-2 flag p2_q and lag-3 code p3c were computed in layer 0",
     q=[("f_one", 0, 1.0)], k=[("f_user", 0, GC)], v=[("p2_q", 0, 1.0)] + vec("p3c", 1), o=[(0, "pend", 1.0)] + [(1 + i, ("PFc", i), 1.0) for i in range(12)], hp=(0, 0, 0, 1, 0))


def fetch(name, mech, keys, R, dst):
    head(1, name, mech, f"key gain {GC} on the class flag; R=" + str(R) + " (negative = earliest, positive = latest) inside the turn",
         q=[("f_one", 0, 1.0)], k=[(c, 0, GC) for c in keys], v=vec("code", 0), o=[(i, (dst, i), 1.0) for i in range(12)], hp=(0, 0, R, 1, 0))


fetch("first_file", "UF1: code of the first file name the user wrote", ["u_file"], -4, "UF1c")
fetch("last_file", "UF2: code of the last file name the user wrote", ["u_file"], 4, "UF2c")
fetch("first_key", "UK1: first key word", ["u_key"], -4, "UK1c")
fetch("last_key", "UK2: last key word", ["u_key"], 4, "UK2c")
fetch("first_num", "UN1: first number", ["u_num"], -4, "UN1c")
fetch("last_num", "UN2: last number", ["u_num"], 4, "UN2c")
fetch("first_val", "UW: first value word (name/status/region/level word)", ["u_val"], -4, "UWc")
fetch("last_word", "UV: the last content word of the user turn (the value in 'set K in F to V')", ["f_content", "in_user"], 4, "UVc")

# ---- M1: booleans (b = min(n,1))
def boolean(out, lin, k=1):
    # [n>=k] = relu(n-k+1) - relu(n-k)
    for dlt, w in [(1, 1.0), (0, -1.0)]:
        neuron(1, "m1.booleans", "b = [count >= k]", "relu(n-k+1)-relu(n-k) is the exact step for integer counts",
               lin, -k + dlt, [(out, w)])


for out, c, k in [("bF1", "file", 1), ("bF2", "file", 2), ("bK1", "key", 1), ("bK2", "key", 2), ("bN1", "num", 1), ("bN2", "num", 2),
                  ("bV1", "val", 1), ("bOPP", "opP", 1), ("bOPM", "opM", 1), ("bOPT", "opT", 1), ("bDBL", "dbl", 1), ("bSW", "swap", 1),
                  ("bDEL", "del", 1), ("bYES", "yes", 1), ("bHELLO", "hello", 1), ("bTHANKS", "thanks", 1), ("bBYE", "bye", 1),
                  ("bLS", "ls", 1), ("bPATH", "path", 1)]:
    boolean(out, [("n_" + c, 1.0)], k)
boolean("bOP", [("n_opP", 1.0), ("n_opM", 1.0), ("n_opT", 1.0)])
boolean("bVAL", [("n_num", 1.0), ("n_val", 1.0)])

# ---- L2 attention: lookups
G8 = 8.0
for nm, kc, ordq in [("lookup_latest", "UK1c", "latest"), ("lookup_second_key", "UK2c", "latest"), ("lookup_first_result", "UK1c", "zero")]:
    dst, fu = {"lookup_latest": ("LVa", "fu_a"), "lookup_second_key": ("LVb", "fu_b"), "lookup_first_result": ("LVc", "fu_c")}[nm]
    q = vec(kc, 0, G8) + [("f_one", 16, 16.0), ("f_one", 17, 120.0)]
    q += [(("nh", r + 1), 12 + r, 16.0) for r in range(3)] if ordq == "latest" else [("f_one", 12, 16.0)]
    kk = vec("p1c", 0) + [(("nh", r), 12 + r, 1.0) for r in range(4)] + [("f_user", 12, -1.0), ("p2_sep", 16, 1.0), ("p2_res", 16, 1.0), ("f_user", 17, 1.0)]
    head(2, nm, f"value of key {kc[:3]} on a result line that STARTS with it (token after K whose lag-2 token is ';' or <result>) -> {dst}; fu = 1 if not found",
         "score = 8*code match(<=96) + 16 if the key is the line start + 16 if in the wanted result (latest/first) = 128 for a true match; the <user> sink has 120, a near miss at most 112",
         q=q, k=kk, v=vec("code", 0) + [("f_user", 12, 1.0)], o=[(i, (dst, i), 1.0) for i in range(12)] + [(12, fu, 1.0)], hp=(0, 0, 0, 1, 0))
head(2, "read_arg", "FREAD: the file argument of the latest read call (token after 'read' that follows <tool>)", f"both lag flags needed: 2*{GC} beats a single one",
     q=[("f_one", 0, 1.0)], k=[("p1_read", 0, GC), ("p2_tool", 0, GC)], v=vec("code", 0), o=[(i, ("FREAD", i), 1.0) for i in range(12)], hp=(0, 0, 4, 1, 0))
head(2, "first_of_result", "G: first token of the latest result (+ whether it is missing/none/error)", "key = lag-1 token is <result>",
     q=[("f_one", 0, 1.0)], k=[("p1_res", 0, GC)], v=vec("code", 0) + [("f_missing", 12, 1.0), ("f_none", 13, 1.0), ("f_error", 14, 1.0)],
     o=[(i, ("Gc", i), 1.0) for i in range(12)] + [(12, "g_missing", 1.0), (13, "g_none", 1.0), (14, "g_error", 1.0)], hp=(0, 0, 4, 1, 0))
head(2, "last_answer", "LAST: the number in the most recent 'the answer is R' of the whole conversation", "key = lag-1 'is' and lag-2 'answer'; global scope; R=2",
     q=[("f_one", 0, 1.0)], k=[("p1_is", 0, 8000.0), ("p2_answer", 0, 8000.0)], v=vec("code", 0), o=[(i, ("LAST", i), 1.0) for i in range(12)], hp=(0, 0, 2, 0, 0))
head(2, "copy_result", "CP: token of the latest result aligned with the reply token being generated (lag = Pst-Pre+2) + whether it is </result>",
     "reply content starts 3 tokens into the step; source index = t-(Pst-Pre+2); A=20 pins exactly that key",
     v=vec("code", 0) + [("f_resend", 12, 1.0)], o=[(i, ("CPc", i), 1.0) for i in range(12)] + [(12, "cp_end", 1.0)], hp=(20, 2, 0, 1, 0), dvec=[("Pst", 1.0), ("Pre", -1.0)])

# ---- M2: intents
nb = lambda c: (c, -1)
pos = lambda c: (c, 1)
E1f = [("bF1", 1.0), ("bF2", -1.0)]
BASE = [nb("pend"), nb("bPATH")]
INT = {
    "delyes": [pos("pend"), pos("bYES")], "kept": [pos("pend"), nb("bYES")], "unsafe": [nb("pend"), pos("bPATH")],
    "confirm": BASE + [pos("bDEL"), pos("bF1")],
    "write": BASE + [nb("bDEL"), (E1f, 1), pos("bK1"), nb("bOP"), pos("bVAL")],
    "sum2": BASE + [nb("bDEL"), pos("bF2"), pos("bK2")],
    "lcalc": BASE + [nb("bDEL"), (E1f, 1), pos("bK1"), pos("bOP"), pos("bN1")],
    "lookup": BASE + [nb("bDEL"), (E1f, 1), pos("bK1"), nb("bOP"), nb("bVAL")],
    "show": BASE + [nb("bDEL"), (E1f, 1), nb("bK1")],
    "flcalc": BASE + [nb("bF1"), pos("bV1"), pos("bK1"), pos("bOP"), pos("bN1")],
    "flook": BASE + [nb("bF1"), pos("bV1"), pos("bK1"), nb("bOP")],
    "find": BASE + [nb("bF1"), pos("bV1"), nb("bK1")],
    "calc": BASE + [nb("bF1"), nb("bV1"), pos("bN2"), pos("bOP")],
    "follow": BASE + [nb("bF1"), nb("bV1"), nb("bN2"), pos("bOP")],
    "ls": BASE + [nb("bF1"), nb("bV1"), nb("bOP"), nb("bN1"), pos("bLS")],
    "hello": BASE + [nb("bF1"), nb("bV1"), nb("bOP"), nb("bN1"), pos("bHELLO")],
    "thanks": BASE + [nb("bF1"), nb("bV1"), nb("bOP"), nb("bN1"), pos("bTHANKS")],
    "bye": BASE + [nb("bF1"), nb("bV1"), nb("bOP"), nb("bN1"), pos("bBYE")],
}
for nm, lits in INT.items():
    AND(2, "m2.intent_rules", "I_x = AND of boolean features (user words + pending-confirm) selecting what the user wants",
        "relu(sum(lits)-(m-1)) is an exact AND of 0/1 literals; the rules are mutually exclusive by construction (see EXPLANATION.md table)",
        lits, [("I_" + nm, 1.0)])

# ---- M3: template flags
nh = lambda r: (("nh", r), 1)
I = lambda n: ("I_" + n, 1)
fnd, nfnd = (("fu_a", -1)), None
ANS = [("calc", 1), ("follow", 1), ("lcalc", 2), ("sum2", 3), ("flcalc", 3)]
LOOK = [("lookup", 1), ("flook", 2)]
RULES = {  # template: list of AND-alternatives (list of literals)
    "tl_ls": [[I("ls"), nh(0)]], "tl_read1": [[I(i), nh(0)] for i in ("show", "lookup", "lcalc", "sum2")],
    "tl_read2": [[I("sum2"), nh(1)]], "tl_readG": [[I(i), nh(1)] for i in ("flook", "flcalc")],
    "tl_grep": [[I(i), nh(0)] for i in ("find", "flook", "flcalc")],
    "tl_calc": [[I("calc"), nh(0), ("bSW", -1)]], "tl_calcs": [[I("calc"), nh(0), ("bSW", 1)]],
    "tl_calcv": [[I("lcalc"), nh(1)], [I("flcalc"), nh(2)]], "tl_calcsum": [[I("sum2"), nh(2)]],
    "tl_calcf1": [[I("follow"), nh(0), ("bDBL", -1)]], "tl_calcf2": [[I("follow"), nh(0), ("bDBL", 1)]],
    "tl_write": [[I("write"), nh(0)]], "tl_delete": [[I("delyes"), nh(0)]],
    "r_ls": [[I("ls"), nh(1)]], "r_show": [[I("show"), nh(1), ("g_missing", -1)]],
    "r_missing": [[I("show"), nh(1), ("g_missing", 1)], [I("write"), nh(1), ("g_missing", 1)]],
    "r_lookup": [[I(i), nh(r), ("fu_a", -1)] for i, r in LOOK], "r_nokey": [[I(i), nh(r), ("fu_a", 1)] for i, r in LOOK],
    "r_find": [[I("find"), nh(1), ("g_none", -1)]], "r_notfound": [[I("find"), nh(1), ("g_none", 1)]],
    "r_answer": [[I(i), nh(r), ("g_error", -1)] for i, r in ANS], "r_calcerr": [[I(i), nh(r), ("g_error", 1)] for i, r in ANS],
    "r_write": [[I("write"), nh(1), ("g_missing", -1)]], "r_confirm": [[I("confirm"), nh(0)]],
    "r_deleted": [[I("delyes"), nh(1)]], "r_kept": [[I("kept"), nh(0)]], "r_unsafe": [[I("unsafe"), nh(0)]],
    "r_hello": [[I("hello"), nh(0)]], "r_thanks": [[I("thanks"), nh(0)]], "r_bye": [[I("bye"), nh(0)]],
}
for t, alts in RULES.items():
    for lits in alts:
        AND(3, "m3.template_rules", "T_x = step (intent x tool-steps-done x result branch) selects reply/tool template x",
            "AND of intent flag, one-hot nres and branch flags from the latest result; exact because all inputs are 0/1",
            lits, [("T_" + t, 1.0)])
neuron(3, "m3.unknown_default", "unknown reply when no intent fired", "relu(nh0 - sum I): exactly 1 only when no intent is on",
       [(("nh", 0), 1.0)] + [("I_" + i, -1.0) for i in INTENTS], 0.0, [("T_r_unknown", 1.0)])

# ---- M4: slot flags / literals
for t, items in TEMPL.items():
    for j, it in enumerate(items):
        ins, outs, b = [("T_" + t, 1.0), (("jh", j), 1.0)], None, -1.0
        if isinstance(it, tuple) and it[0] == "copy":
            ins = [("T_" + t, 1.0), ("jg3", 1.0), ("f_endp", -1.0)]
            outs = [("S_COPY", 1.0)]
        elif isinstance(it, tuple):
            outs = [("S_" + it[1], 1.0)]
        else:
            outs = code_of(it)
        neuron(4, "m4.emit", "at step index j of template T emit literal token (writes its 12-bit code to out) or raise a slot flag S_var",
               "relu(T+jh-1) is the AND of 'in this template' and 'at this index'; templates are exclusive so exactly one unit fires",
               ins, b, outs)
neuron(4, "m4.end", "after '.' or '?' emit <end>", "f_endp (current token is . or ?) directly writes the code of <end>", [("f_endp", 1.0)], 0.0, code_of("<end>"))
neuron(4, "m4.unknown_T", "r_unknown template literals use T_r_unknown (registered in TEMPL)", "same as emit", [], 0.0, [])
LAYERS[4]["neurons"].pop()

# ---- M5: gate variable codes into out; operators; copy end
for v_, src in VARS.items():
    for bit in range(12):
        sx = ("S_" + v_, 2.0)
        extra = [("cp_end", -2.0)] if v_ == "COPY" else []
        for sgn in (1.0, -1.0):
            neuron(5, "m5.gate_vars", "out += code of the slot variable when its slot flag S is on",
                   "relu(sgn*c+2S-2) passes the +-1 code bit only when S=1 (S=0 leaves <=-1 -> 0); the + and - units give back the signed bit",
                   [((src, bit), sgn), sx] + extra, -2.0, [(("out", bit), sgn)])
for op, w in [("bOPP", "plus"), ("bOPM", "minus"), ("bOPT", "times")]:
    neuron(5, "m5.operator", "OP slot emits plus/minus/times according to the operator word the user used", "AND of S_OP and the boolean for that operator",
           [("S_OP", 1.0), (op, 1.0)], -1.0, code_of(w))
neuron(5, "m5.copy_end", "when the copy source is </result> emit '.'", "AND of S_COPY and cp_end", [("S_COPY", 1.0), ("cp_end", 1.0)], -1.0, code_of("."))


# ----------------------------------------------------------------- assemble
def build():
    p, groups = {}, []
    # embedding
    E = torch.zeros(V, K, dtype=F64)
    vocab = T.vocab()
    tid = T.token_to_id()
    for v_ in range(V):
        w = vocab[v_]
        if w not in ("<end>", "<bos>"):
            for b in range(12):
                E[v_, b] = 1.0 if (v_ >> b) & 1 else -1.0
    for f, w in SPECIAL_FLAG.items():
        E[tid[w], ci(f)] = 1
    for w in FILES:
        E[tid[w], ci("f_file")] = 1
    for w in KEYS:
        E[tid[w], ci("f_key")] = 1
    for n in range(1000):
        E[12 + n, ci("f_num")] = 1
    for w in VALS:
        E[tid[w], ci("f_val")] = 1
    for f, words in LEX.items():
        for w in words.split():
            if w in tid:
                E[tid[w], ci(f)] = 1
    for v_ in range(V):
        w = vocab[v_]
        if w not in T.SPECIALS and w not in "? ! , . : ;".split() and w != "<unk>":
            E[v_, ci("f_content")] = 1
    E[:, ci("f_one")] = 1
    p["embed"] = E
    p["embproj"] = torch.zeros(K, D, dtype=F64)
    p["embproj"][torch.arange(K), torch.arange(K)] = 1
    p["pos_dir"] = torch.zeros(D, dtype=F64); p["pos_dir"][ci("posv")] = 1
    p["seg_in"] = torch.zeros(D, dtype=F64); p["seg_in"][ci("f_user")] = 1
    p["big"] = torch.tensor([1e9], dtype=F64)
    p["out_proj"] = torch.zeros(D, 12, dtype=F64)
    for b in range(12):
        p["out_proj"][ci(("out", b)), b] = 5.0
    ids = torch.arange(V)
    p["code_table"] = torch.stack([((ids >> b) & 1).to(F64) * 2 - 1 for b in range(12)])
    groups += [
        group("io.embedding", "embed", S[:, :], mechanism="row of token v = 12-bit code (+-1) and hand-written class flags (file, key, number, value word, operator words, cues, specials)",
              why="the code makes every token copyable and comparable by dot product (match 12, any other <=10); flags are the lexicon. <bos>/<end> have code 0 so they never match a key"),
        group("io.embedding_proj", "embproj", S[:, :], mechanism="identity: static token features occupy the first K residual channels", why="keeps the embed table small (V x K instead of V x d)"),
        group("io.position", "pos_dir", S[:], mechanism="writes the integer position t into channel posv", why="heads that need the position of a flagged token read posv"),
        group("io.turn_counter", "seg_in", S[:], mechanism="turn id = running count of <user> tokens; used by 'local' heads", why="makes 'same turn' exact"),
        group("io.big", "big", S[:], mechanism="mask constant 1e9", why="causal mask and cross-turn penalty; any value >> score range works"),
        group("io.unembed_proj", "out_proj", S[:, :], mechanism="reads the 12 out channels with gain 5", why="logit of the intended token = 5*12=60, any other token <=50: margin 10"),
        group("io.unembed_code", "code_table", S[:, :], mechanism="logit(v) = sum_b out_b * bit_b(v)", why="binary code of the token id; dot with out=+-code(t) is maximal only for t"),
    ]
    for L, lay in enumerate(LAYERS):
        e, hs, ns = lay["e"], lay["heads"], lay["neurons"]
        h = len(hs)
        wq, wk, wv = [torch.zeros(h, D, e, dtype=F64) for _ in range(3)]
        wo = torch.zeros(h, e, D, dtype=F64)
        hp = torch.zeros(h, 5, dtype=F64)
        dv = torch.zeros(h, D, dtype=F64)
        for i, hd in enumerate(hs):
            for s_, d_, w_ in hd["q"]:
                wq[i, ci(s_), d_] += w_
            for s_, d_, w_ in hd["k"]:
                wk[i, ci(s_), d_] += w_
            for s_, d_, w_ in hd["v"]:
                wv[i, ci(s_), d_] += w_
            for s_, d_, w_ in hd["o"]:
                wo[i, s_, ci(d_)] += w_
            hp[i] = torch.tensor(hd["hp"], dtype=F64)
            for s_, w_ in hd["dvec"]:
                dv[i, ci(s_)] += w_
            for tname, t_ in [("wq", wq), ("wk", wk), ("wv", wv), ("wo", wo), ("hp", hp), ("dvec", dv)]:
                groups.append(group(hd["name"], f"l{L}.{tname}", S[i], mechanism=hd["mech"], why=hd["why"]))
        p[f"l{L}.wq"], p[f"l{L}.wk"], p[f"l{L}.wv"], p[f"l{L}.wo"], p[f"l{L}.hp"], p[f"l{L}.dvec"] = wq, wk, wv, wo, hp, dv
        m = len(ns)
        w_in = torch.zeros(D, m, dtype=F64); b_in = torch.zeros(m, dtype=F64); w_out = torch.zeros(m, D, dtype=F64)
        fam = {}
        for j, nr in enumerate(ns):
            for s_, w_ in nr["ins"]:
                w_in[ci(s_), j] += w_
            b_in[j] = nr["bias"]
            for s_, w_ in nr["outs"]:
                w_out[j, ci(s_)] += w_
            fam.setdefault(nr["grp"], []).append(j)
        p[f"l{L}.w_in"], p[f"l{L}.b_in"], p[f"l{L}.w_out"] = w_in, b_in, w_out
        for g, js in fam.items():
            mech, why = WHY[g]
            groups.append(group(g, f"l{L}.w_in", S[:, js], mechanism=mech, why=why))
            groups.append(group(g, f"l{L}.b_in", S[js], mechanism=mech, why=why))
            groups.append(group(g, f"l{L}.w_out", S[js, :], mechanism=mech, why=why))
    arch = CustomArch("experiments/gen/agents/g10/arch.py", "transformer with relative-position, recency, same-turn and counting head modes: copy/lookup/count by attention, logic by ReLU AND units")
    return arch, p, groups
