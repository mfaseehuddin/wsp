"""nco: hand-written 2-level Next-Concept Model for the IT help desk.

Level 0 (n=1, 56 codes): the PLAN of the reply = (intent, has_name, has_ticket).
Level 1 (n=48, codebook = whole vocab): the exact TOKEN at every position, computed in parallel from
(plan, position) by a piecewise-linear "layout" MLP plus a copy head for the name and the ticket digits.
See EXPLANATION.md for the meaning of every weight group.
"""
import math

import torch

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
INTENTS = list(REPLIES)
UNK = INTENTS.index("unknown")
HELLO = INTENTS.index("hello")

# ---- intent keywords: word -> (attention strength, {intent: vote}) -------------------------------------------
S, M, W = 1.0, 0.75, 0.5          # strong / medium / weak (greetings, thanks) key strength
KW = {}


def kw(intent, strength, words, vote=1.0):
    for w in words.split():
        KW.setdefault(w, [strength, {}])
        KW[w][0] = max(KW[w][0], strength)
        KW[w][1][intent] = KW[w][1].get(intent, 0) + vote


kw("password", S, "password passwords passcode pwd passphrase pin")
kw("password", M, "forgot forgotten reset expired expire expiring credentials")
kw("locked", S, "locked lockout unlock unlocked lock locking disabled suspended blocked")
kw("locked", M, "attempts account")
kw("vpn", S, "vpn anyconnect globalprotect forticlient tunnel")
kw("vpn", M, "remote remotely")
kw("wifi", S, "wifi wi-fi wireless wlan hotspot ssid router")
kw("wifi", M, "internet network signal")
kw("email", S, "email emails mail mails outlook inbox mailbox calendar gmail attachment attachments "
               "sync syncing synced newsletter messages")
kw("printer", S, "printer printers print printing printed prints toner scanner scan scanning jam jammed "
                 "cartridge ink paper")
kw("software", S, "software install installed installing installation program programs application "
                  "applications license licence excel photoshop adobe zoom download visio tableau")
kw("software", M, "app apps tool")
kw("slow", S, "slow slower slowly sluggish freezing freezes frozen freeze lag laggy lagging hangs hanging "
              "forever performance unresponsive crawling")
kw("mfa", S, "mfa authenticator authentication otp verification duo okta factor 2fa")
kw("mfa", M, "code codes phone verify sms")
kw("disk", S, "disk disks space storage gb capacity cleanup")
kw("disk", M, "drive full")
kw("monitor", S, "monitor monitors screen screens display displays hdmi displayport projector resolution")
kw("monitor", M, "external")
kw("hello", W, "hello hi hey morning afternoon evening greetings howdy hiya")
kw("thanks", W + 0.05, "thanks thank thx cheers appreciated appreciate grateful worked solved")
kw("password", M, "login change wrong")
kw("locked", M, "login")
kw("vpn", M, "connection connect connecting disconnecting disconnected drops dropping home")
kw("wifi", S, "lan")
kw("wifi", M, "connection connect connecting disconnecting disconnected drops dropping")
kw("monitor", M, "connect cable dock docking black blank second dual")
kw("email", M, "send message spam")
kw("printer", M, "queue offline")
kw("software", M, "update admin access word")
kw("slow", M, "crashing stuck ages")
kw("wifi", M, "online")
kw("mfa", M, "token prompt approve push text")
kw("disk", S, "recycle onedrive")
kw("disk", M, "files hard")
kw("thanks", W + 0.05, "great awesome perfect sorted")
GREET = set("hello hi hey".split())

# ---- name evidence ---------------------------------------------------------------------------------------------
NAMES = """ali anna chen david emma fatima james kenji leo lucy maria nina omar priya ravi sam sara sofia tom zara
aaron abdul adam adrian ahmed aisha akira alan albert alex alexander alice alicia amal amanda amir amit amy ana
andrea andrew andy angela anil anita ann anne anthony anton arjun arthur asha ashley aya ayesha barbara ben
benjamin beth bilal bob boris brian bruce carl carla carlos carmen carol caroline catherine charles charlie
chris christina christine claire clara daniel danny dario deepak diana diego dmitri dylan ed edward elena eli
elif elizabeth ella ellen emily emir emre eric erik ethan eva fahad farah felix fernando fiona francesca frank
fred gabriel gary george gina greg hamza hana hannah hans harry hassan helen henry hiro hiroshi hugo hussein
ian igor ines irene isaac isabel isabella ivan jack jacob jake jamal jane jason javier jean jeff jennifer jenny
jessica jim joe john jonathan jorge jose joseph josh juan julia julian julie justin kai kamal karen karim kate
katie keith kevin khalid kim kumar kyle lara laura lee lena leila liam lily lin linda lisa liu luca lucas luis
luke mai maya manuel marco marcus maria mario marta martin mary matt matthew max maxim mehmet mei mia michael
michelle miguel mike min mohamed mohammed monica nadia naomi natalia nathan neha nick nicole nikita noah noor
olga oliver olivia omar oscar pablo paolo patrick paul pedro peter philip pierre pooja rachel rahul raj rajesh
ramon raul rebecca reza ricardo richard rita robert rohan ron rosa ryan sabrina sakura salma samir samuel sandra
sanjay santiago sarah sean sebastian sergei shreya simon sina sonia sophie stefan steve steven sunil susan
takeshi tanya tara teresa thomas tim timothy tina tony tyler uma usman valentina vera victor vikram vincent
wang wei william xavier yan yara yasmin yi yuki yusuf zain zhang zoe
aria dan dave erin grace heidi jade mateo miro ruth taha tariq jasper keiko""".split()
# words that are never a name (function words, help-desk vocabulary); intent keywords are added below
NOT_NAME = """i me my mine you your we our they them he she it its is am are was were be been being have has had
do does did not no yes a an the this that these those there here what who why how when where which can could
would should will shall may might must to of in on at for from with without by about into over after before
again and or but so if then than too very really still just also only please help hello hi hey thanks thank
new old good bad fine ok okay sure sorry unable trying getting having going working broken down up out off
back home office work today now ready done stuck lost confused frustrated busy away all any some every
everything nothing something anything one two three four five six seven eight nine ten first second last
keeps keep says said get got need needs want wants like see seen try tried call called ticket number re
issue issues problem problems error errors fail fails failed failing crash crashing crashed computer laptop
pc mac machine device desktop user login log sign signed again anymore more much many less few same other
another , . ? ! ' " - : ; ( ) #
urgent important quick quickly asap weird strange annoying impossible terrible awful horrible wrong dead gone
missing empty black blank white red green blue big small huge tiny late early fast slow easy hard simple
happy sad angry upset tired sick available free open closed shut stopped started finished new old
afraid worried sorry glad pleased able allowed supposed meant told asked using used trying wondering
desperate stuck annoyed lucky done broke fixed sorted unhappy tech it support desk team manager boss
finance hr sales marketing accounting legal admin office building floor room meeting today tomorrow
yesterday morning afternoon evening night week weekend monday tuesday wednesday thursday friday
again also always never sometimes often anything nothing everyone someone somebody nobody""".split()
INTRO = {"hi": 0.5, "hey": 0.5, "hello": 0.5, "i'm": 1.0, "am": 1.0, "is": 0.4, "me": 1.0, "called": 1.0, "named": 1.0, "name": 0.4}
INTRO2 = {"this": 0.8, "it": 0.6, "name": 0.8}
FOLLOW = {".": 0.8, ",": 0.8, "here": 1.8, "speaking": 1.8, "from": 0.8, "!": 0.8}
W_END, W_LEX, W_PEN, THETA = 0.6, 1.0, -3.0, 1.5

# ---- residual layout (d_model = 128) ---------------------------------------------------------------------------
E_DIM = 16
I0, O0 = 0, 32                         # token identity (input) / token output subspaces
P1, P2, PF, LP1, LKEY, L0, L1 = 64, 65, 66, 67, 68, 69, 70
KWD, VOT, HIC = 71, 72, 86             # keyword key strength, 14 intent votes, "greeting followed by comma"
NF, TF, NS, DS = 87, 88, 89, 90        # name flag, ticket flag, name score, digit scores DS0..3 (90..93)
QN, QK = 94, 95                        # copy queries: name, digit k (95..98)
CI, CN_, CT = 99, 113, 114             # plan dims from latent: intent one-hot (99..112), has_name, has_ticket
DIG, INT1, INT2, FOL, COM, GRT = 115, 116, 117, 118, 119, 120
CNX = 121                              # "next token is a comma"
# latent (D = 32): 0..15 token code, 16..29 intent, 30 has_name, 31 has_ticket
ZI, ZN, ZT = 16, 30, 31


def one(w):
    ids = T.encode(w)
    return ids[0] if len(ids) == 1 and ids[0] > 2 else None


def build():
    V = len(T.vocab())
    H, dh, A = 4, 32, 100.0
    rs = math.sqrt(dh)
    reps = {k: T.encode(v) for k, v in REPLIES.items()}
    Lc = [len(reps[k]) for k in INTENTS]
    tail = T.encode("i added this to ticket") + [None] * 4 + T.encode(".")
    greet = [T.encode("hi")[0], None, T.encode(",")[0]]
    n_core = sum(L + 2 for L in Lc)
    d_mlp = n_core + 5 + 12 + 1
    cfg = model.Config(vocab=V, D=32, levels=(1, 48), codebooks=(56, V), d_model=128, n_layers=2, n_heads=H,
                       d_head=dh, d_mlp=d_mlp, dec_hidden=0, norm=False)
    P = {k: torch.zeros(*s, dtype=torch.float64) for k, s in model.shapes(cfg).items()}

    # token codes: fixed pseudo-random unit vectors (near-orthogonal); <pad>,<eos> get the zero code
    g = torch.Generator().manual_seed(0)
    E = torch.randn(V, E_DIM, generator=g, dtype=torch.float64)
    E = E / E.norm(dim=1, keepdim=True)
    E[:2] = 0

    # ---- embeddings ----
    te = P["tok_emb"]
    te[:, I0:I0 + E_DIM] = E
    digits = [T.encode(str(d))[0] for d in range(10)]
    for d in digits:
        te[d, DIG] = 1
        te[d, DS:DS + 4] = 1
        te[d, NS] += W_PEN
    te[2, NS] += W_PEN                                    # <unk> is never copied as a name
    for w in NAMES:
        if (i := one(w)) is not None:
            te[i, NS] += W_LEX
    for w in set(NOT_NAME) | set(KW) | set(INTRO) | set(INTRO2) | set(FOLLOW):
        if (i := one(w)) is not None and w not in NAMES:
            te[i, NS] += W_PEN
    for tab, dim in ((INTRO, INT1), (INTRO2, INT2), (FOLLOW, FOL)):
        for w, v in tab.items():
            if (i := one(w)) is not None:
                te[i, dim] = v
    te[T.encode(",")[0], COM] = 1
    for w, (s, votes) in KW.items():
        if (i := one(w)) is None:
            continue
        te[i, KWD] = s
        tot = sum(votes.values())
        for it, v in votes.items():
            te[i, VOT + INTENTS.index(it)] = v / tot
        if w in GREET:
            te[i, GRT] = 1
    pp = P["pre_pos"]
    for i in range(32):
        pp[i, P1], pp[i, P2], pp[i, PF] = i + 1, (i + 1) ** 2, 1
        pp[i, NS] = W_END * (i + 2)                       # end-of-message cue, cancelled by N1 head unless last
    P["lvl_emb"][0, LKEY] = P["lvl_emb"][0, L0] = 1
    P["lvl_emb"][1, LKEY] = P["lvl_emb"][1, L1] = 1
    for t in range(48):
        P["lvl_pos"][t, LP1] = t
    wi = P["w_in"]
    for j in range(14):
        wi[ZI + j, CI + j] = 1
    wi[ZN, CN_] = wi[ZT, CT] = 1

    # ---- layer 0 attention: positional heads (prev1, prev2, prev3, next1) for prefix rows ----
    c, B = 8.0, 20.0
    for h, k in enumerate((1, 2, 3, -1)):
        wq, wk, wv, wo = (P[f"L0.{x}"][h] for x in ("wq", "wk", "wv", "wo"))
        wq[P1, 0], wq[PF, 0], wq[PF, 1] = 2 * c * rs, -2 * c * k * rs, -c * rs
        wq[L0, 2] = wq[L1, 2] = B * rs
        wk[P1, 0], wk[P2, 1], wk[LKEY, 2] = 1, 1, 1
        if h == 0:
            wv[INT1, 0], wo[0, NS] = 1, 1
            wv[DIG, 1] = 1
            wo[1, DS + 0], wo[1, DS + 1], wo[1, DS + 2], wo[1, DS + 3] = -1, 1, 1, 1
        elif h == 1:
            wv[INT2, 0], wo[0, NS] = 1, 1
            wv[DIG, 1] = 1
            wo[1, DS + 1], wo[1, DS + 2], wo[1, DS + 3] = -1, 1, 1
        elif h == 2:
            wv[DIG, 1] = 1
            wo[1, DS + 2], wo[1, DS + 3] = -1, 1
        else:
            wv[FOL, 0], wo[0, NS] = 1, 1
            wv[P1, 1], wo[1, NS] = 1, -W_END
            wv[COM, 2], wo[2, CNX] = 1, 1

    # ---- layer 0 MLP ----
    w1, b1, w2 = P["L0.w1"], P["L0.b1"], P["L0.w2"]
    u = 0

    def bank(seq_tok, seq_q, pre):          # piecewise-linear interpolation of a token sequence over offset x
        nonlocal u
        n = len(seq_tok)
        vec = []
        for x in range(n):
            v = torch.zeros(128, dtype=torch.float64)
            if seq_tok[x] is not None:
                v[O0:O0 + E_DIM] = E[seq_tok[x]]
            if seq_q[x] is not None:
                v[seq_q[x]] = 1
            vec.append(v)
        get = lambda x: vec[x] if 0 <= x < n else torch.zeros(128, dtype=torch.float64)
        for m in range(-1, n + 1):
            pre(u, m)
            w2[u] = get(m + 1) - 2 * get(m) + get(m - 1)
            u += 1

    for j, it in enumerate(INTENTS):        # core reply of intent j at offset x = t - 3*has_name
        def pre(uu, m, j=j):
            w1[LP1, uu], w1[CN_, uu], w1[CI + j, uu], b1[uu] = 1, -3, A, -m - A
        bank(reps[it], [None] * Lc[j], pre)

    def pre_g(uu, m):                       # greeting "hi <name> ," at t = 0..2 when has_name
        w1[LP1, uu], w1[CN_, uu], b1[uu] = 1, A, -m - A
    bank(greet, [None, QN, None], pre_g)

    def pre_t(uu, m):                       # ticket tail at offset x = t - 3*has_name - len(core)
        w1[LP1, uu], w1[CN_, uu], w1[CT, uu], b1[uu] = 1, -3, A, -m - A
        for j in range(14):
            w1[CI + j, uu] = -Lc[j]
    bank(tail, [None] * 5 + [QK + k for k in range(4)] + [None], pre_t)
    # greeting followed by a comma ("hi , i am ...") moves its vote to the HIC dim
    w1[GRT, u], w1[CNX, u], b1[u] = 1, 1, -1
    w2[u, VOT + HELLO], w2[u, HIC] = -1, 1
    u += 1
    assert u == d_mlp

    # ---- layer 1 attention: copy (level-1 rows); name flag, ticket flag, intent (level-0 row) ----
    bn, bd, C, bf, bi = 20.0, 30.0, 12.0, 30.0, 24.0
    for h in range(4):
        wq, wk, wv, wo = (P[f"L1.{x}"][h] for x in ("wq", "wk", "wv", "wo"))
        if h < 3:
            wk[NS, 0], wk[LKEY, 5] = 1, 1
            for k in range(4):
                wk[DS + k, 1 + k] = 1
        if h == 0:
            wq[QN, 0] = bn * rs
            for k in range(4):
                wq[QK + k, 1 + k] = bd * rs
            wq[L1, 5] = C * rs
            for e in range(E_DIM):
                wv[I0 + e, e], wo[e, O0 + e] = 1, 1
        elif h == 1:
            wq[L0, 0], wq[L0, 5] = bf * rs, bf * THETA * rs
            wv[PF, 0], wo[0, NF] = 1, 1
        elif h == 2:
            wq[L0, 4], wq[L0, 5] = bf * rs, bf * 3.5 * rs
            wv[PF, 0], wo[0, TF] = 1, 1
        else:
            wk[KWD, 0], wk[LKEY, 1] = 1, 1
            wq[L0, 0], wq[L0, 1] = bi * rs, 0.25 * bi * rs
            for e in range(15):
                wv[VOT + e, e], wo[e, VOT + e] = 1, 1

    # ---- level heads and codebooks ----
    for j in range(14):
        for n in range(2):
            for k in range(2):
                cc = j * 4 + n * 2 + k
                P["head0"][VOT + j, cc] = 1
                if j == HELLO and n == 0:
                    P["head0"][HIC, cc] = 0.3
                P["head0"][NF, cc], P["head0"][TF, cc] = n, k
                P["head_b0"][cc] = -0.5 * n - 0.5 * k + (0.05 if j == UNK else 0)
                P["code0"][cc, ZI + j], P["code0"][cc, ZN], P["code0"][cc, ZT] = 1, n, k
    P["head1"][O0:O0 + E_DIM] = E.T
    P["head_b1"][1] = 0.3
    P["code1"][:, :E_DIM] = E
    P["dec_w2"][:E_DIM] = E.T
    P["dec_b2"][1] = 0.3

    groups = [
        {"name": "token codes", "tensors": ["tok_emb[:, 0:32]", "head1", "code1", "dec_w2", "dec_b2", "head_b1"],
         "mechanism": "every token v has a fixed unit vector E[v] (seeded Gaussian); reading/writing tokens is a dot "
                      "product with E; <eos> has the zero code and wins by a 0.3 bias when nothing is written",
         "why": "for a clean E[v] the dot product is 1 for v and <1 for every other unit vector, so argmax is exact"},
        {"name": "plan code (level 0)", "tensors": ["head0", "head_b0", "code0", "w_in"],
         "mechanism": "56 codes = intent x has_name x has_ticket; logit = intent vote + n(NF-.5) + k(TF-.5)",
         "why": "argmax over a product set factorises into three independent decisions"},
        {"name": "intent keywords", "tensors": ["tok_emb KWD/VOT", "L1 head 3"],
         "mechanism": "level-0 row attends to keywords (strength 1/.75/.5, sink .25) and averages their votes",
         "why": "strong topic words dominate greetings exponentially; no keyword -> sink -> unknown bias wins"},
        {"name": "positional heads", "tensors": ["pre_pos", "L0 heads 0-3"],
         "mechanism": "score -c(p - target)^2 via q=(2c target, -c), k=(p, p^2); prev1/2/3 and next1",
         "why": "give each user token its neighbours' features: name cues and digit-run counts"},
        {"name": "name/digit scores", "tensors": ["tok_emb NS/DS", "L0 wo", "pre_pos NS"],
         "mechanism": "NS = lexicon + intro(prev) + intro2(prev2) + follow(next) + end - common-word penalty; "
                      "DS_k = digit_i + digits_{i-1..i-k} - digit_{i-k-1} peaks (k+1) at the k-th ticket digit",
         "why": "linear evidence sums; copy and flag heads read them as attention keys"},
        {"name": "flags and copy", "tensors": ["L1 heads 0-2"],
         "mechanism": "NF/TF: attention mass on prefix vs a sink at threshold; copy: query from layout picks "
                      "the max-NS token or k-th digit and moves its identity code into the output subspace",
         "why": "softmax with large scale = hard argmax; sink absorbs rows that copy nothing"},
        {"name": "layout MLP", "tensors": ["L0.w1", "L0.b1", "L0.w2"],
         "mechanism": "for each intent a bank of ReLU(x - m), x = t - 3n, whose second differences carry the "
                      "reply tokens: the piecewise-linear interpolant equals E[reply[x]] at integer x; same for "
                      "greeting (t) and ticket tail (x - len core) gated by flags",
         "why": "exact lookup of token-by-offset with Lc+2 units per sequence; gates are -A(1-flag)"},
    ]
    return cfg, P, groups
