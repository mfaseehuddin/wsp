"""NCM help-desk bot, hand-written. K=2 levels: (1) one 'plan' code, (2) 48 per-position token codes. See EXPLANATION.md."""
import math, sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import lex
from wsp.ncm import model, tokenizer as T

REPLIES = {
 "password": "go to reset.corp.com and choose forgot password . the link expires in 15 minutes .",
 "locked": "your account unlocks after 30 minutes . to unlock it now , call the help desk on 4357 .",
 "vpn": "restart the vpn client , check your internet connection , then sign in again .",
 "wifi": "forget the corp wifi network , reconnect , and enter your corp password when asked .",
 "email": "close outlook , open it again and check you are online . if mail still fails , restart your computer .",
 "printer": "check the printer has paper and toner , then remove it and add it again in settings .",
 "software": "request software in the self service portal . most requests are approved within one day .",
 "slow": "restart your computer and close unused apps . if it stays slow , open a ticket at help.corp.com .",
 "mfa": "open the authenticator app and approve the request . for a new phone , call the help desk on 4357 .",
 "disk": "empty the recycle bin and move large files to onedrive , then run disk cleanup .",
 "monitor": "check the cable at both ends , then press windows p and choose extend .",
 "hello": "hello , i am the it support bot . what is the problem ?",
 "thanks": "you are welcome . have a good day !",
 "unknown": "sorry , i can only help with it problems . please open a ticket at help.corp.com .",
}
INTENTS = lex.INTENTS
NI = len(INTENTS)
Cc = 128          # width of the token-code space
dh = Cc
TAIL = "i added this to ticket".split()


def tok1(w):
    ids = T.encode(w)
    return ids[0] if len(ids) == 1 and ids[0] != 2 else None


def code(tid):  # pseudo-random +-1 code for a token id (formula: seeded by the id)
    return np.random.default_rng(1000 + tid).choice([-1.0, 1.0], size=Cc)


def build():
    V = len(T.vocab())
    ix = T.encode
    NAMES = sorted({tok1(w) for w in lex.NAMES if tok1(w)} )
    DIG = [ix(str(d))[0] for d in range(10)]
    EOS = 1
    cores = {i: ix(REPLIES[i]) for i in INTENTS}
    for i, c in cores.items():
        assert 2 not in c, (i, c)
    hi, comma, dot = ix("hi")[0], ix(",")[0], ix(".")[0]
    tail = ix(" ".join(TAIL))
    # alphabet: every token that may appear in a reply
    A = sorted(set([EOS, hi, comma, dot] + tail + DIG + NAMES + [t for c in cores.values() for t in c]))
    aidx = {t: a for a, t in enumerate(A)}
    codes = np.stack([code(t) for t in A])
    G = codes @ codes.T
    np.fill_diagonal(G, 0)
    assert abs(G).max() < 0.5 * Cc, abs(G).max()

    # ---- dm channel layout
    ch, n = {}, 0
    def alloc(name, k=1):
        nonlocal n
        ch[name] = n; n += k
    alloc("C", Cc); alloc("O", Cc)
    alloc("KW"); alloc("EV", NI); alloc("EVO", NI); alloc("ISN"); alloc("ISD"); alloc("ISQ")
    alloc("NPO"); alloc("DPO"); alloc("PP", 32); alloc("P", 48)
    alloc("INT", NI); alloc("NM"); alloc("NM0"); alloc("TK"); alloc("TK0")
    alloc("PD", 3); alloc("GR"); alloc("TH"); alloc("PH"); alloc("U", 4); alloc("WN"); alloc("WD", 4); alloc("WDS")
    dm = n
    # D: [code space | intent one-hot | flags]
    D = Cc + NI + 4
    Dint, Dnm, Dnm0, Dtk, Dtk0 = Cc, Cc + NI, Cc + NI + 1, Cc + NI + 2, Cc + NI + 3

    # ---- MLP neurons for layer 0: list of (input {channel:weight}, bias, output {channel:weight})
    neu = []
    def lit(c, o=0): return ch[c] + o
    for k in range(4):  # rank one-hot  u_k = hat(s-k), s = sum of the 3 offset-head outputs
        for d, w in ((1, 1), (0, -2), (-1, 1)):
            neu.append(({lit("PD", 0): 1, lit("PD", 1): 1, lit("PD", 2): 1}, -(k - d) * 1.0 + 0.0 * 0, {lit("U", k): w}))
    # fix biases: relu(s - k + d) where d in (1,0,-1)  -> bias = -k + d
    neu = []
    for k in range(4):
        for d, w in ((1, 1), (0, -2), (-1, 1)):
            neu.append(({lit("PD", 0): 1, lit("PD", 1): 1, lit("PD", 2): 1}, -k + d, {lit("U", k): w}))
    def emit(tid, pos, i=None, nm=None, tk=None):
        """output code(tid) at reply position pos when intent i / name flag nm / ticket flag tk match"""
        inp = {lit("P", pos): 1}
        b = 0
        if i is not None: inp[lit("INT", INTENTS.index(i))] = 1; b -= 1
        if nm is not None: inp[lit("NM" if nm else "NM0")] = 1; b -= 1
        if tk is not None: inp[lit("TK" if tk else "TK0")] = 1; b -= 1
        b -= 0
        neu.append((inp, b, {ch["O"] + c: float(codes[aidx[tid]][c]) for c in range(Cc)}))
    def slot(chan_out, pos, i=None, nm=None, tk=None):
        inp = {lit("P", pos): 1}; b = 0
        if i is not None: inp[lit("INT", INTENTS.index(i))] = 1; b -= 1
        if nm is not None: inp[lit("NM" if nm else "NM0")] = 1; b -= 1
        if tk is not None: inp[lit("TK" if tk else "TK0")] = 1; b -= 1
        neu.append((inp, b, chan_out))
    emit(hi, 0, nm=True); emit(comma, 2, nm=True)
    slot({lit("WN"): 1}, 1, nm=True)
    for i in INTENTS:
        L = len(cores[i])
        for nm in (0, 1):
            o = 3 * nm
            for j, t in enumerate(cores[i]):
                emit(t, o + j, i=i, nm=bool(nm))
            e = o + L
            for j, t in enumerate(tail):
                emit(tid=t, pos=e + j, i=i, nm=bool(nm), tk=True)
            for k in range(4):
                slot({lit("WD", k): 1, lit("WDS"): 1}, e + 5 + k, i=i, nm=bool(nm), tk=True)
            emit(dot, e + 9, i=i, nm=bool(nm), tk=True)
            assert e + 10 <= 48
    neu.append(({ch["TH"]: 1, ch["PH"]: 1}, -1, {ch["KW"]: 4.0}))
    d_mlp = len(neu)

    cfg = model.Config(vocab=V, D=D, levels=(1, 48), codebooks=(NI * 4, len(A)), d_model=dm, n_layers=2,
                       n_heads=6, d_head=dh, d_mlp=d_mlp, dec_hidden=0, norm=False)
    P = {k: torch.zeros(*s, dtype=torch.float64) for k, s in model.shapes(cfg).items()}
    sq = math.sqrt(dh)

    # ---- embeddings
    for t in NAMES:
        P["tok_emb"][t, ch["C"]:ch["C"] + Cc] = torch.tensor(code(t)); P["tok_emb"][t, ch["ISN"]] = 1
    for t in DIG:
        P["tok_emb"][t, ch["C"]:ch["C"] + Cc] = torch.tensor(code(t)); P["tok_emb"][t, ch["ISD"]] = 1
    for ii, intent in enumerate(INTENTS[:-1]):
        for w, s in lex.KW[intent].items():
            t = tok1(w)
            if t is None: continue
            P["tok_emb"][t, ch["KW"]] = max(float(P["tok_emb"][t, ch["KW"]]), s)
            P["tok_emb"][t, ch["EV"] + ii] = 1.0
    # tokens with several intents: normalise EV to the intent with the max strength
    for ii, intent in enumerate(INTENTS[:-1]):
        pass
    for w in ("hi", "hello", "hey", "hiya", "greetings", "morning"):
        if tok1(w): P["tok_emb"][tok1(w), ch["GR"]] = 1
    P["tok_emb"][tok1("there"), ch["TH"]] = 1
    for p in range(32): P["pre_pos"][p, ch["PP"] + p] = 1
    for t in range(48): P["lvl_pos"][t, ch["P"] + t] = 1
    for l in range(2): P["lvl_emb"][l, ch["ISQ"]] = 1
    for c in range(NI): P["w_in"][Dint + c, ch["INT"] + c] = 1
    for d, c in ((Dnm, "NM"), (Dnm0, "NM0"), (Dtk, "TK"), (Dtk0, "TK0")): P["w_in"][d, ch[c]] = 1

    # ---- layer 0 attention (6 heads)
    L0 = "L0."
    for o in (1, 2, 3):                                                # h3..5: look o tokens back, read isdigit
        h = 2 + o
        for j in range(o, 32):
            P[L0 + "wq"][h, ch["PP"] + j, j - o] = 20 * sq
        for j in range(32): P[L0 + "wk"][h, ch["PP"] + j, j] = 1
        P[L0 + "wv"][h, ch["ISD"], 0] = 1; P[L0 + "wo"][h, 0, ch["PD"] + o - 1] = 1
    # ---- layer 0 MLP
    for u, (inp, b, out) in enumerate(neu):
        for c, w in inp.items(): P[L0 + "w1"][c, u] = w
        P[L0 + "b1"][u] = b
        for c, w in out.items(): P[L0 + "w2"][u, c] = w
    L1 = "L1."
    # layer 1 heads 2-4 pool over the user tokens for level 0 (they run at level 1 too, harmlessly)
    P[L1 + "bq"][2, 0] = sq; P[L1 + "wk"][2, ch["KW"], 0] = 1          # h2: keyword pooling (logit = strength)
    for c in range(NI): P[L1 + "wv"][2, ch["EV"] + c, c] = 1; P[L1 + "wo"][2, c, ch["EVO"] + c] = 1
    P[L1 + "bq"][3, 0] = sq; P[L1 + "wk"][3, ch["ISN"], 0] = 20       # h3: name pooling
    P[L1 + "wv"][3, ch["ISN"], 0] = 1; P[L1 + "wo"][3, 0, ch["NPO"]] = 1
    P[L1 + "bq"][4, 0] = sq; P[L1 + "wk"][4, ch["ISD"], 0] = 20       # h4: digit pooling
    P[L1 + "wv"][4, ch["ISD"], 0] = 1; P[L1 + "wo"][4, 0, ch["DPO"]] = 1
    # bigram "hi there": offset-1 head (h3 of layer 0) also reads the greeting flag of the previous token
    P[L0 + "wv"][3, ch["GR"], 1] = 1; P[L0 + "wo"][3, 1, ch["PH"]] = 1
    # ---- layer 1 attention: head 0 copies the name, head 1 copies the k-th ticket digit
    P[L1 + "wq"][0, ch["WN"], 0] = 20 * sq; P[L1 + "wk"][0, ch["ISN"], 0] = 1
    P[L1 + "bq"][0, 1] = 20 * sq; P[L1 + "wq"][0, ch["WN"], 1] = -20 * sq; P[L1 + "wk"][0, ch["ISQ"], 1] = 1
    P[L1 + "wq"][1, ch["WDS"], 0] = 20 * sq; P[L1 + "wk"][1, ch["ISD"], 0] = 1
    for k in range(4): P[L1 + "wq"][1, ch["WD"] + k, 1 + k] = 20 * sq; P[L1 + "wk"][1, ch["U"] + k, 1 + k] = 1
    P[L1 + "bq"][1, 5] = 20 * sq; P[L1 + "wq"][1, ch["WDS"], 5] = -20 * sq; P[L1 + "wk"][1, ch["ISQ"], 5] = 1
    for h in (0, 1):
        for c in range(Cc): P[L1 + "wv"][h, ch["C"] + c, c] = 1; P[L1 + "wo"][h, c, ch["O"] + c] = 1

    # ---- level 0 (plan) head: class c=(intent,name,ticket)
    for ii in range(NI):
        for nm in (0, 1):
            for tk in (0, 1):
                c = (ii * 2 + nm) * 2 + tk
                if ii < NI - 1: P["head0"][ch["EVO"] + ii, c] = 7.0
                else: P["head_b0"][c] += 2.0         # 'unknown' wins when no keyword evidence exceeds ~0.3
                P["head0"][ch["NPO"], c] = 5.0 if nm else -5.0
                P["head0"][ch["DPO"], c] = 5.0 if tk else -5.0
                P["head_b0"][c] += (0 if nm else 5.0) + (0 if tk else 5.0)
                P["code0"][c, Dint + ii] = 1
                P["code0"][c, Dnm if nm else Dnm0] = 1
                P["code0"][c, Dtk if tk else Dtk0] = 1
    # ---- level 1 (token) head, codebook, decoder
    for a, t in enumerate(A):
        P["head1"][ch["O"]:ch["O"] + Cc, a] = torch.tensor(codes[a])
        P["code1"][a, :Cc] = torch.tensor(codes[a])
        P["dec_w2"][:Cc, t] = torch.tensor(codes[a])
    P["head_b1"][aidx[EOS]] = 70.0

    groups = [
        {"name": "token codes", "tensors": ["code1", "dec_w2", "head1"], "mechanism": "every reply token has a fixed +-1 code of width 128 (seeded by token id); codebook 1, the decoder and the level-1 head are all that code", "why": "random +-1 codes are nearly orthogonal (max cross-correlation < 64 of 128), so correlation retrieves the token"},
        {"name": "token embeddings (lexicon)", "tensors": ["tok_emb", "pre_pos", "lvl_pos", "lvl_emb", "w_in"], "mechanism": "each user token carries: its copy code (names, digits), name/digit flags, keyword strength KW and intent evidence EV; positions are one-hot; w_in routes plan flags into x", "why": "hand-written lexicon of names and intent keywords from language knowledge"},
        {"name": "level 0: plan (intent x name x ticket)", "tensors": ["head0", "head_b0", "code0", "L1.wq", "L1.wk", "L1.wv", "L1.wo"], "mechanism": "layer-1 heads 2-4 soft-max-pool the user tokens: keyword evidence per intent, name presence, digit presence; head0 scores 56 plan classes additively", "why": "softmax with logit = keyword strength focuses on keywords, so evidence is length independent; 'unknown' wins by a constant bias when evidence < ~0.3"},
        {"name": "layer 0 attention: ticket digit rank", "tensors": ["L0.wq", "L0.wk", "L0.wv", "L0.wo"], "mechanism": "three heads look 1/2/3 tokens back (one-hot positions) and read isdigit; the sum is the rank of a digit within its run", "why": "the k-th ticket digit is the digit with k digits before it"},
        {"name": "layer 0 MLP: reply template", "tensors": ["L0.w1", "L0.b1", "L0.w2", "L0.b2"], "mechanism": "one AND-neuron per (intent, name?, position) writes the token code for fixed reply tokens, plus slot markers WN/WD_k, rank one-hot U_k, and the 'hi there' bigram", "why": "neuron fires iff all inputs are 1 (bias = -(inputs-1))"},
        {"name": "layer 1 attention: slot copying", "tensors": ["L1.wq", "L1.wk", "L1.wv", "L1.wo", "L1.bq"], "mechanism": "head 0: name slot rows attend the name token; head 1: digit slot k attends the digit with rank k; both copy its code to O; other rows attend a level row with zero code", "why": "logit gaps of 20 make attention effectively hard"},
        {"name": "level 1 bias", "tensors": ["head_b1"], "mechanism": "eos bias 70 beats noise when no token code is written", "why": "positions after the reply stay empty"},
    ]
    return cfg, P, groups
