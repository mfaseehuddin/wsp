import torch
import math


def isal(c):
    return ("a" <= c and c <= "z") or ("A" <= c and c <= "Z") or ("0" <= c and c <= "9")


def isdg(c):
    return "0" <= c and c <= "9"


def build():
    d = 448
    L = 7
    H = 3
    E = 88
    M = 1024
    MAXLEN = 16384
    cfg = dict(d_model=d, n_layers=L, n_heads=H, d_head=E, d_mlp=M, max_len=MAXLEN,
               pos="learned", norm=False, bias=True, tie_embeddings=False)
    sqE = math.sqrt(E)

    # ---------------- lexicon ----------------
    func = ["the", "a", "an", "is", "are", "was", "were", "am", "be", "it", "its", "this", "that", "of", "in", "on",
            "at", "to", "for", "with", "by", "from", "and", "or", "but", "not", "no", "yes", "do", "does", "did",
            "have", "has", "had", "what", "who", "whom", "whose", "when", "where", "why", "how", "which", "can",
            "will", "would", "should", "I", "you", "he", "she", "we", "they", "me", "my", "your", "his", "her",
            "our", "their", "there", "here", "please", "also", "so", "as", "if", "then", "now", "many", "much",
            "more", "one", "two", "today", "tell", "say", "know", "call", "make", "makes", "give", "get", "go"]
    funcwords = []
    for w in func:
        funcwords.append(w)
        if (w[0] in "abcdefghijklmnopqrstuvwxyz"):
            funcwords.append(w[0].upper() + w[1:])
    frame_words = ["Hello", "Hi", "Hey", "hello", "hi", "hey", "Noted", "welcome", "assistant", "remember", "that",
                   "yet", "small", "hand", "built", "help", "I'm", "I'll", "thanks", "Thanks", "thank", "Thank",
                   "name", "called", "am", "doing", "well", "program", "Tell", "Nice", "meet"]
    rels = {"capital": ["capital"], "legs": ["legs", "leg"], "make": ["make", "makes", "produce", "produces"],
            "class": ["fish", "mammal", "amphibian", "reptile", "kind", "class"], "boil": ["boil", "boils", "boiling"],
            "freeze": ["freeze", "freezes", "freezing"], "color": ["color", "colour"]}
    facts = []
    for e, v in [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"),
                 ("Germany", "Berlin"), ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Brazil", "Brasilia"),
                 ("Australia", "Canberra"), ("India", "Delhi"), ("China", "Beijing"), ("Russia", "Moscow"),
                 ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Greece", "Athens"), ("England", "London"),
                 ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Norway", "Oslo"), ("Mexico", "Mexico")]:
        facts.append((e, "capital", v))
    for e, v in [("spider", "8"), ("spiders", "8"), ("ant", "6"), ("insect", "6"), ("dog", "4"), ("cat", "4"),
                 ("bird", "2"), ("human", "2"), ("cow", "4"), ("horse", "4")]:
        facts.append((e, "legs", v))
    for e, v in [("bees", "honey"), ("bee", "honey"), ("cows", "milk"), ("hens", "eggs"), ("chickens", "eggs"),
                 ("cow", "milk"), ("hen", "eggs")]:
        facts.append((e, "make", v))
    for e, v in [("whale", "mammal"), ("dolphin", "mammal"), ("shark", "fish"), ("bat", "mammal"),
                 ("frog", "amphibian"), ("snake", "reptile"), ("eagle", "bird"), ("salmon", "fish")]:
        facts.append((e, "class", v))
    facts.append(("water", "boil", "100"))
    facts.append(("water", "freeze", "0"))
    for e, v in [("grass", "green"), ("sky", "blue"), ("banana", "yellow"), ("snow", "white"), ("blood", "red"),
                 ("coal", "black")]:
        facts.append((e, "color", v))
    ents = []
    for f in facts:
        if f[0] not in ents:
            ents.append(f[0])
    relwords = []
    for k in rels:
        for w in rels[k]:
            relwords.append(w)
    words = []
    for w in funcwords + frame_words + ents + relwords + [f[2] for f in facts]:
        ok = True
        for c in w:
            if not (isal(c) or c == "_" or c == "'"):
                ok = False
        if ok and w not in words and not (isdg(w[0]) and len(w) == 1):
            words.append(w)
    V = 272 + len(words)
    wid = {}
    for i, w in enumerate(words):
        wid[w] = 272 + i
    funcset = set(funcwords)

    def tok(s):
        out = []
        i = 0
        n = len(s)
        for _ in range(n):
            if i >= n:
                break
            c = s[i]
            if isal(c) or c == "_":
                j = i
                for _ in range(n):
                    if j < n and (isal(s[j]) or s[j] == "_"):
                        j += 1
                    elif j + 1 < n and s[j] in "'-" and (isal(s[j + 1]) or s[j + 1] == "_"):
                        j += 1
                    else:
                        break
                w = s[i:j]
                if w in wid:
                    out.append(wid[w])
                else:
                    for ch in w:
                        out.append(16 + ord(ch))
                i = j
            else:
                out.append(16 + ord(c))
                i += 1
        return out

    # ---------------- layout ----------------
    C = 0
    Vb = 64
    O = 128
    Tb = 192
    P = 256
    TL = 272
    F = 273
    R0 = 289
    fl = {}
    for i, n in enumerate(["bos", "user", "bot", "bnd", "wordc", "content", "digit", "op", "qm", "sep", "dot",
                           "excl", "lexw", "space", "sp1", "sp2"]):
        fl[n] = F + i
    rn = ["SPi", "SPex", "SPw", "SPq", "RC", "RCt", "SPqt"]
    reg = {}
    pos_ = R0
    for n in rn:
        reg[n] = pos_
        pos_ += 16
    for n in ["tqm", "tgreet", "tthank", "tyou", "tintro", "tent", "pcont", "nomatch", "stop_q", "stop_a", "tdig", "top",
              "f_greet", "f_thank", "f_kb", "f_about", "g_greet", "g_thank", "g_kb", "g_about",
              "s_ack", "s_qa", "s_unk", "u_ack", "u_qa", "u_unk"]:
        reg[n] = pos_
        pos_ += 1
    relreg = {}
    for k in rels:
        reg["rel_" + k] = pos_
        relreg[k] = pos_
        pos_ += 1

    # ---------------- codes (seeded, greedy low coherence) ----------------
    g = torch.Generator().manual_seed(1)
    codes = torch.zeros(V, 64)
    for i in range(V):
        cand = torch.sign(torch.randn(24, 64, generator=g))
        if i == 0:
            codes[i] = cand[0]
        else:
            m = torch.matmul(cand, codes[:i].T).abs().max(dim=1).values
            codes[i] = cand[int(torch.argmax(-m).item())]
    codes[1] = torch.zeros(64)  # BOS = null sink, code 0
    perm = torch.randperm(64, generator=g)
    sgn = torch.sign(torch.randn(64, generator=g))
    Qm = torch.zeros(64, 64)
    for i in range(64):
        Qm[int(perm[i].item()), i] = sgn[i]  # Q c = signed permutation of c (orthogonal)

    # ---------------- params ----------------
    p = {}
    p["embed"] = torch.zeros(V, d)
    p["pos"] = torch.zeros(MAXLEN, d)
    p["unembed"] = torch.zeros(d, V)
    p["b_unembed"] = torch.zeros(V)
    for l in range(L):
        for n, s in [("wq", (H, d, E)), ("wk", (H, d, E)), ("wv", (H, d, E)), ("wo", (H, E, d)),
                     ("bq", (H, E)), ("bk", (H, E)), ("bv", (H, E)), ("bo", (d,)),
                     ("w_in", (d, M)), ("w_out", (M, d)), ("b_in", (M,)), ("b_out", (d,))]:
            p["layers." + str(l) + "." + n] = torch.zeros(s)

    per = [3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0]
    t = torch.arange(MAXLEN).float()
    for k in range(8):
        p["pos"][:, P + 2 * k] = torch.cos(2 * math.pi * t / per[k])
        p["pos"][:, P + 2 * k + 1] = torch.sin(2 * math.pi * t / per[k])
    p["pos"][:, TL] = t / 16.0
    p["pos"][0, P:P + 16] = 0.0  # position 0 (BOS) carries a zero code so the BOS sink has no kernel noise

    def rot(sign):
        Rm = torch.zeros(16, 16)
        for k in range(8):
            th = sign * 2 * math.pi / per[k]
            c = math.cos(th)
            s_ = math.sin(th)
            Rm[2 * k, 2 * k] = c
            Rm[2 * k, 2 * k + 1] = -s_
            Rm[2 * k + 1, 2 * k] = s_
            Rm[2 * k + 1, 2 * k + 1] = c
        return Rm

    Rp = rot(1.0)
    Rm1 = rot(-1.0)

    # embedding rows
    greetw = ["Hello", "Hi", "Hey", "hello", "hi", "hey"]
    thankw = ["thanks", "Thanks", "thank", "Thank"]
    introw = ["I'm", "my", "My", "name", "called", "I", "am"]
    youw = ["you", "You", "your", "Your"]
    opw = ["plus", "minus", "times"]
    relof = {}
    for k in rels:
        for w in rels[k]:
            relof[w] = k
    for i in range(V):
        row = p["embed"][i]
        row[C:C + 64] = codes[i]
        fi = {}
        if i == 1:
            fi = {"bos": 1, "bnd": 1}
        elif i == 2:
            fi = {"user": 1, "bnd": 1, "sep": 1}
        elif i == 3:
            fi = {"bot": 1, "bnd": 1, "sep": 1}
        elif i < 16:
            fi = {"sep": 1, "bnd": 1 if i in (8, 10) else 0}
        elif i < 272:
            ch = chr(i - 16)
            if isal(ch) and ord(ch) < 128 or ch in "_'-":
                fi = {"wordc": 1, "sep": 0}
                if (isal(ch) and not isdg(ch) or isdg(ch)):
                    fi["content"] = 1
                if isdg(ch):
                    fi["digit"] = 1
                    row[reg["tdig"]] = 1.0
            else:
                fi = {"sep": 1}
                if ch in ".?!\n":
                    fi["bnd"] = 1
                if ch == ".":
                    fi["dot"] = 1
                if ch == "?":
                    fi["qm"] = 1
                    fi["bnd"] = 1
                    row[reg["tqm"]] = 1.0
                if ch == "!":
                    fi["excl"] = 1
                if ch == " ":
                    fi["space"] = 1
                if ch in "+*/%":
                    fi["op"] = 1
                    row[reg["top"]] = 1.0
        else:
            w = words[i - 272]
            fi = {"wordc": 1, "lexw": 1}
            if w not in funcset:
                fi["content"] = 1
            if w in greetw:
                row[reg["tgreet"]] = 1.0
            if w in thankw:
                row[reg["tthank"]] = 1.0
            if w in introw:
                row[reg["tintro"]] = 1.0
            if w in youw:
                row[reg["tyou"]] = 1.0
            if w in opw:
                row[reg["top"]] = 1.0
            if w in relof:
                row[relreg[relof[w]]] = 1.0
            if w in ents:
                row[reg["tent"]] = 1.0
        for n in fi:
            row[fl[n]] = float(fi[n])
    for i in range(V):
        p["unembed"][O:O + 64, i] = codes[i]
    p["b_unembed"][10] = 30.0

    # ---------------- attention helpers ----------------
    DESC = {}

    def note(l, h, text):
        k = "layers." + str(l) + ".attn"
        DESC[k] = DESC.get(k, "") + " [h" + str(h) + "] " + text

    def Wq(l, h):
        return p["layers." + str(l) + ".wq"][h]

    def Wk(l, h):
        return p["layers." + str(l) + ".wk"][h]

    def pairqk(l, h, col, qs, ks, n, coef, Rm=None):
        # logit += coef * (Rm q) . k
        if Rm is None:
            Rm = torch.eye(n)
        Wq(l, h)[qs:qs + n, col:col + n] = coef * sqE * Rm.T
        Wk(l, h)[ks:ks + n, col:col + n] = torch.eye(n)
        return col + n

    def flagk(l, h, col, kidx, coef, qidx=None, qcoef=0.0):
        # logit += coef * kfeat(s) (constant in q); large coefs use a big key scale so weights stay < 60000
        ks = 1.0
        if abs(coef) > 1000.0:
            ks = abs(coef) / 1000.0
        p["layers." + str(l) + ".bq"][h, col] = coef * sqE / ks
        if isinstance(kidx, list):
            for ki in kidx:
                Wk(l, h)[ki, col] = ks
        else:
            Wk(l, h)[kidx, col] = ks
        return col + 1

    def sinkcol(l, h, col, level, gate=None, gscale=0.0):
        base = 0.0
        # BOS sink: logit = level + gscale*(1-gate)
        p["layers." + str(l) + ".bq"][h, col] = (level + gscale) * sqE
        if gate is not None:
            Wq(l, h)[gate, col] = -gscale * sqE
        Wk(l, h)[fl["bos"], col] = 1.0
        return col + 1

    def val(l, h, vcol, src, dst, n, scale=1.0):
        p["layers." + str(l) + ".wv"][h][src:src + n, vcol:vcol + n] = scale * torch.eye(n)
        p["layers." + str(l) + ".wo"][h][vcol:vcol + n, dst:dst + n] = torch.eye(n)
        return vcol + n

    def recency(l, h, col, coef):
        return flagk(l, h, col, TL, coef)

    # ---- L0 ----
    c = pairqk(0, 0, 0, P, P, 16, 4.0, Rm1)
    val(0, 0, 0, C, Vb, 64)
    note(0, 0, "previous-token head: query is position code rotated by -1 step, key is position code (8 sinusoid pairs, peak 8 vs 5 at neighbours); copies token code of t-1 into Vb")
    c = flagk(0, 1, 0, fl["bnd"], 2000000.0)
    c = recency(0, 1, c, 128.0)
    val(0, 1, 0, P, reg["SPi"], 16)
    note(0, 1, "latest sentence-boundary head (USER, BOT, . ? ! END, BOS): bnd flag + recency; copies that boundary position code into SPi")
    c = flagk(0, 2, 0, fl["sep"], 2000000.0)
    c = recency(0, 2, c, 128.0)
    val(0, 2, 0, P, reg["SPw"], 16)
    note(0, 2, "latest word-separator head: copies position code of the nearest non-word-char token (<=t) into SPw")

    # ---- L1 ----
    c = pairqk(1, 0, 0, P, P, 16, 4.0, Rm1)
    v = val(1, 0, 0, reg["SPi"], reg["SPex"], 16)
    p["layers.1.wv"][0][Vb:Vb + 64, v:v + 64] = 0.7 * Qm.T
    p["layers.1.wo"][0][v:v + 64, Vb:Vb + 64] = torch.eye(64)
    v = v + 64
    note(1, 0, "previous-token carry: SPex(t)=SPi(t-1) (header of the sentence token t belongs to); Vb += 0.5*Vb(t-1) so Vb = code(t-1)+0.5code(t-2)")
    c = pairqk(1, 1, 0, reg["SPw"], reg["SPw"], 16, 12.0)
    c = flagk(1, 1, c, fl["wordc"], 30.0)
    c = sinkcol(1, 1, c, 108.0)
    val(1, 1, 0, C, Tb, 64)
    note(1, 1, "word signature: average token code over the word-so-far (same SPw); invented names get a unique bag-of-letters signature; BOS sink gives 0 off-word")

    # ---- L2 ----
    c = pairqk(2, 0, 0, P, P, 16, 4.0, Rm1)
    v = val(2, 0, 0, Tb, Tb, 64)
    v = val(2, 0, v, reg["SPex"], reg["SPq"], 16)
    v = val(2, 0, v, fl["content"], reg["pcont"], 1)
    v = val(2, 0, v, reg["tqm"], reg["tqm"], 1)
    pairqk(2, 1, 0, P, P, 16, 4.0, Rm1)
    p["layers.2.wv"][1][Vb:Vb + 64, 0:64] = 0.7 * Qm.T
    p["layers.2.wo"][1][0:64, Vb:Vb + 64] = torch.eye(64)
    note(2, 1, "second previous-token carry: Vb += 0.7*Q*Vb(t-1) extends the key context to three tokens (Q = signed permutation keeps orders distinguishable)")
    note(2, 0, "previous-token carry of word signature to the separator after the word (full-word signature), SPq(t)=SPex(t-1), pcont flag")

    # ---- L3 ----
    c = pairqk(3, 0, 0, reg["SPq"], reg["SPex"], 16, 12.0)
    c = flagk(3, 0, c, fl["sep"], 30.0)
    c = flagk(3, 0, c, reg["pcont"], 30.0)
    c = sinkcol(3, 0, c, 140.0)
    val(3, 0, 0, Tb, Tb, 64)
    note(3, 0, "question bag: at BOT average full-word signatures of content words in the last sentence (same header SPq)")
    tfs = [reg[n] for n in ["tgreet", "tthank", "tyou", "tintro"]] + [relreg[k] for k in rels]
    c = pairqk(3, 1, 0, reg["SPq"], reg["SPex"], 16, 12.0)
    c = flagk(3, 1, c, tfs, 40.0)
    c = sinkcol(3, 1, c, 116.0)
    v = 0
    for r_ in tfs:
        v = val(3, 1, v, r_, r_, 1)
    note(3, 1, "intent-flag gather: copies word-feature flags (?, greeting, thanks, you, intro, relation words) of the last sentence to BOT")
    c = pairqk(3, 2, 0, reg["SPq"], reg["SPex"], 16, 12.0)
    c = flagk(3, 2, c, reg["tent"], 40.0)
    c = sinkcol(3, 2, c, 116.0)
    val(3, 2, 0, C, Vb, 64, 2.0)
    note(3, 2, "known-entity gather: copies code of KB entity word in the last sentence into Vb at BOT")

    # ---- L4 ----
    c = pairqk(4, 0, 0, Tb, Tb, 64, 4.0)
    c = pairqk(4, 0, c, reg["SPq"], reg["SPex"], 16, -30.0)
    c = flagk(4, 0, c, fl["sep"], 30.0)
    c = flagk(4, 0, c, reg["pcont"], 30.0)
    c = flagk(4, 0, c, fl["bot"], -100.0)
    c = recency(4, 0, c, 0.003)
    c = sinkcol(4, 0, c, 100.0)
    v = val(4, 0, 0, reg["SPex"], reg["RC"], 16)
    v = val(4, 0, v, fl["bos"], reg["nomatch"], 1)
    note(4, 0, "retrieval: question bag vs full-word signatures at earlier separators (own sentence excluded); value = header code of the matching sentence; BOS sink (threshold 64) = no match")
    c = flagk(4, 1, 0, fl["bot"], 2000000.0)
    c = recency(4, 1, c, 128.0)
    v = 0
    for n in ["f_greet", "f_thank", "f_kb", "f_about"]:
        v = val(4, 1, v, reg[n], reg["g_" + n[2:]], 1)
    note(4, 1, "fetch fast-frame flags from the latest BOT to every reply position")

    # ---- L5 ----
    def starthead(h, rs, gname):
        c0 = pairqk(5, h, 0, rs, P, 16, 6.0, Rp)
        c0 = sinkcol(5, h, c0, 20.0, reg[gname], 100.0)
        flagk(5, h, c0, [fl["bot"], fl["user"]], -400.0)
        val(5, h, 0, C, O, 64)

    starthead(0, reg["RC"], "s_qa")
    note(5, 0, "copy-start (QA): query = RC position code advanced one step, key = position code; copies first token of retrieved sentence into O")
    c = flagk(5, 1, 0, fl["bot"], 2000000.0)
    c = recency(5, 1, c, 128.0)
    v = 0
    for n in ["s_ack", "s_qa", "s_unk"]:
        v = val(5, 1, v, reg[n], reg["u_" + n[2:]], 1)
    v = val(5, 1, v, reg["RC"], reg["RCt"], 16)
    v = val(5, 1, v, reg["SPq"], reg["SPqt"], 16)
    note(5, 1, "fetch slow-frame flags, retrieved header RC and last-sentence header SPq from the latest BOT")
    starthead(2, reg["SPq"], "s_ack")
    note(5, 2, "copy-start (ACK): same, but starting at the user's own last sentence (SPq)")

    # ---- L6 ----
    def conthead(h, rs, gname, stopname):
        c0 = pairqk(6, h, 0, rs, reg["SPex"], 16, 14.0)
        Wq(6, h)[C:C + 64, c0:c0 + 64] = 1.0 * sqE * torch.eye(64)
        Wq(6, h)[Vb:Vb + 64, c0:c0 + 64] = 0.7 * sqE * Qm.T
        Wk(6, h)[Vb:Vb + 64, c0:c0 + 64] = torch.eye(64)
        c0 += 64
        c0 = sinkcol(6, h, c0, 155.0, reg[gname], 100.0)
        flagk(6, h, c0, [fl["bot"], fl["user"]], -400.0)  # never copy protocol tokens
        Wq(6, h)[fl["bot"], c0 - 1] = 100.0 * sqE  # never copy continuation at BOT itself
        v0 = val(6, h, 0, C, O, 64)
        val(6, h, v0, fl["bos"], reg[stopname], 1)

    conthead(0, reg["RCt"], "u_qa", "stop_q")
    note(6, 0, "induction copy within the retrieved sentence: key = previous token(s) equals current token (Vb) inside region RCt; BOS sink = end of sentence (sets stop)")
    conthead(1, reg["SPqt"], "u_ack", "stop_a")
    note(6, 1, "same induction copy within the user's own last sentence (ACK)")

    # ---------------- MLP rule engine ----------------
    cnt = {}
    ml = {}

    def neuron(l, conds, outs, tag):
        n = cnt.get(l, 0)
        cnt[l] = n + 1
        key = "layers." + str(l)
        win = p[key + ".w_in"]
        pos_w = 0.0
        for cd in conds:
            k, a, w = cd
            if k == "t":
                win[C:C + 64, n] += w * codes[a] / 64.0
            elif k == "p":
                win[Vb:Vb + 64, n] += w * codes[a] / 64.0
            elif k == "f":
                win[fl[a], n] += w
            elif k == "v":  # Vb dot code scaled for gathered entity (x2)
                win[Vb:Vb + 64, n] += w * codes[a] / 128.0
            else:
                win[reg[a], n] += w
            if w > 0:
                pos_w += w
        p[key + ".b_in"][n] = -(pos_w - 0.5)
        wout = p[key + ".w_out"]
        for ot in outs:
            k, a, w = ot
            if k == "o":
                wout[n, O:O + 64] += 6.0 * w * codes[a]
            else:
                wout[n, reg[a]] += 2.0 * w
        ml.setdefault(l, []).append(tag)

    # fast frames (L3)
    neuron(3, [("f", "bot", 1.0), ("r", "tgreet", 1.0), ("r", "tintro", -1.0)], [("r", "f_greet", 1.0)], "greet")
    neuron(3, [("f", "bot", 1.0), ("r", "tthank", 1.0)], [("r", "f_thank", 1.0)], "thank")
    neuron(3, [("f", "bot", 1.0), ("r", "tyou", 1.0), ("r", "tgreet", -1.0)], [("r", "f_about", 1.0)], "about")
    for (e, rl, vv) in facts:
        neuron(3, [("f", "bot", 1.0), ("v", wid[e], 1.0), ("r", "rel_" + rl, 1.0)], [("r", "f_kb", 1.0)], "kb")
    # slow frames (L4)
    ffs = [("r", "f_greet", -2.0), ("r", "f_thank", -2.0), ("r", "f_kb", -2.0), ("r", "f_about", -2.0)]
    neuron(4, [("f", "bot", 1.0), ("r", "f_greet", -2.0), ("r", "f_thank", -2.0), ("r", "f_kb", -2.0),
               ("r", "f_about", -2.0), ("r", "tqm", -2.0), ("r", "tdig", -1.0), ("r", "top", -1.0)],
           [("r", "s_ack", 1.0)], "ack")
    neuron(4, [("f", "bot", 1.0), ("r", "tqm", 1.0), ("r", "nomatch", -2.0), ("r", "f_greet", -2.0), ("r", "f_thank", -2.0),
               ("r", "f_kb", -2.0), ("r", "f_about", -2.0)], [("r", "s_qa", 1.0)], "qa")
    neuron(4, [("f", "bot", 1.0), ("r", "tqm", 1.0), ("r", "nomatch", 1.0), ("r", "f_greet", -2.0), ("r", "f_thank", -2.0),
               ("r", "f_kb", -2.0), ("r", "f_about", -2.0)], [("r", "s_unk", 1.0)], "unk")

    # literals (L6)
    def chain(fr, text, prefix_ctx, first=True):
        ids = tok(text)
        for i in range(len(ids)):
            cd = list(fr)
            if i == 0:
                if first:
                    neuron(6, cd + [("f", "bot", 1.0)], [("o", ids[0], 1.0)], "lit")
            if i + 1 < len(ids):
                c2 = cd + [("t", ids[i], 1.0)]
                if i > 0:
                    c2.append(("p", ids[i - 1], 1.0))
                neuron(6, c2, [("o", ids[i + 1], 1.0)], "lit")
        return ids

    ids = chain([("r", "g_greet", 1.0)], "Hello! How can I help?", None)
    neuron(6, [("r", "g_greet", 1.0), ("t", ids[-1], 1.0), ("p", ids[-2], 1.0)], [("o", 10, 1.0)], "end")
    ids = chain([("r", "g_thank", 1.0)], "You are welcome.", None)
    neuron(6, [("r", "g_thank", 1.0), ("t", ids[-1], 1.0), ("p", ids[-2], 1.0)], [("o", 10, 1.0)], "end")
    ids = chain([("r", "g_about", 1.0)], "I am a small hand-built assistant.", None)
    neuron(6, [("r", "g_about", 1.0), ("t", ids[-1], 1.0), ("p", ids[-2], 1.0)], [("o", 10, 1.0)], "end")
    allids = chain([("r", "u_unk", 1.0)], "I do not know that yet.", None)
    neuron(6, [("r", "u_unk", 1.0), ("t", allids[-1], 1.0), ("p", allids[-2], 1.0)], [("o", 10, 1.0)], "end")
    # QA: stop after copied sentence -> END
    for fn in ["dot", "qm", "excl"]:
        neuron(6, [("r", "u_qa", 1.0), ("f", fn, 1.0)], [("o", 10, 1.0)], "end")
    # ACK: after copy, " Noted."
    nt = tok(" Noted.")
    for fn in ["dot", "qm", "excl"]:
        neuron(6, [("r", "u_ack", 1.0), ("f", fn, 1.0), ("p", nt[-2], -1.0)],
               [("o", nt[0], 1.0)], "ack-note")
    for i in range(len(nt) - 1):
        c2 = [("r", "u_ack", 1.0), ("t", nt[i], 1.0)]
        if i > 0:
            c2.append(("p", nt[i - 1], 1.0))
            neuron(6, c2, [("o", nt[i + 1], 1.0)], "ack-note")
        else:
            for pc in ".?!":
                neuron(6, c2 + [("p", 16 + ord(pc), 1.0)], [("o", nt[i + 1], 1.0)], "ack-note")
    neuron(6, [("r", "u_ack", 1.0), ("t", nt[-1], 1.0), ("p", nt[-2], 1.0)], [("o", 10, 1.0)], "end")
    # KB: value at BOT, then '.', END
    for (e, rl, vv) in facts:
        vi = tok(vv)
        neuron(6, [("f", "bot", 1.0), ("v", wid[e], 1.0), ("r", "rel_" + rl, 1.0)], [("o", vi[0], 1.0)], "kbval")
    neuron(6, [("r", "g_kb", 1.0), ("f", "wordc", 1.0)], [("o", 16 + ord("."), 1.0)], "kbdot")
    neuron(6, [("r", "g_kb", 1.0), ("f", "dot", 1.0)], [("o", 10, 1.0)], "end")

    # ---------------- groups ----------------
    groups = []
    gen = {
        "embed": ("token embedding: 64-d seeded +-1 code (greedy lowest-coherence pick) in C, plus class flags F and word-feature regs",
                  "codes give a shared token bus that attention copies and the unembedding reads"),
        "pos": ("learned position table: 8 sinusoid pairs (periods 3..23 coprime) + linear t/16",
                "relative-offset matching (prev token, latest boundary) by rotation of the pair code"),
        "unembed": ("reads output block O against token codes", "logit_i = O . code_i"),
        "b_unembed": ("END bias 30 (default when nothing writes O)", "empty rules end the turn"),
    }
    for k in p:
        nz = float(p[k].abs().sum().item()) > 0
        if not nz:
            continue
        if k in gen:
            mech, why = gen[k]
        else:
            ls = k.split(".")
            kind = ls[2]
            ad = DESC.get("layers." + ls[1] + ".attn", "")
            if kind in ("w_in", "w_out", "b_in", "b_out"):
                mech = "MLP rules (neurons: AND of token/flag/register conditions -> writes frame flags or output token codes): " + " ".join(sorted(set(ml.get(int(ls[1]), []))))
                why = "relu(sum of cond - (n-0.5)) fires only when all conditions hold"
            else:
                mech = "attention " + kind + ": " + ad
                why = "gains chosen so the intended key beats the BOS sink / neighbours by >=6 logits"
        groups.append(dict(name=k, tensor=k, index=None, mechanism=mech, why=why))
    return cfg, p, groups, words
