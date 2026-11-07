"""nco60: a hand-derived multi-layer Next-Concept chat model (see EXPLANATION.md).

Every weight is written by formula. Qwen2.5-0.5B's tied embedding is the only learned material: it supplies
semantic codes (PCA), a function-word axis, punctuation/boundary similarity and a few named concept directions
(<= 3 anchors each). Everything else is mechanism: positional RoPE kernels, counters, sinks, copy heads.
"""

import math

import torch
from safetensors import safe_open

from wsp.ncmchat import model
from wsp.ncmchat.harness import ids, special

QW = "models/Qwen2.5-0.5B/model.safetensors"
V = 151936
R = 64  # identity code width (latent D for tokens)
RP = 32  # identity prefix used for matching
S = 48  # semantic code width
DH = 192
HALF = DH // 2
THETA = 10 ** 19.2
NPOS = 21  # positional pairs 0..20
CPAIRS = list(range(24, HALF))  # near rotation-free pairs used for content (omega <= 1.6e-5)
CDIMS = [p for p in CPAIRS] + [p + HALF for p in CPAIRS]  # 132 content dims
NMODE = 11
MODES = ["ACK", "GREET", "GREETNAME", "BYE", "THANKS", "DECLINE", "ANSWER", "UNKNOWN", "SELF", "REQUEST", "CLARIFY"]
M = {m: i for i, m in enumerate(MODES)}
# short composable pieces (<= 6 words each); copied material is appended after COPY modes
TMPL = {
    "ACK": "Got it, I will remember",
    "GREET": "Hello! How can I help you?",
    "GREETNAME": "Hello, nice to meet you,",
    "BYE": "Goodbye, take care!",
    "THANKS": "You're welcome, happy to help!",
    "DECLINE": "Sorry, I can't help with that.",
    "ANSWER": " It is",
    "UNKNOWN": "Sorry, I don't know that.",
    "SELF": "I'm a simple chat program." + " I remember what you tell me.",
    "REQUEST": "I can't do that yet. Topic:",
    "CLARIFY": "Could you say more about that?",
}
COPY = {"GREETNAME", "ANSWER", "REQUEST", "ACK"}
TOPIC = ["GREETNAME", "REQUEST", "ACK"]  # copy modes whose span is the newest content run of the last turn
NHOP = 5  # copied span length cap (A0..A4)
NPOSJ = 16  # rows with an explicit position indicator

# ---------------- residual layout ----------------
L = {}
_o = 0


def _slot(name, n):
    global _o
    L[name] = slice(_o, _o + n)
    _o += n


_slot("ID", R)  # level rows: OUT (written only by the last MLP)
_slot("SEMW", S)  # level rows: A1 (last layer)
_slot("CTX", S)  # level rows: A2
_slot("P", 4 * RP)  # P1..P4 ; level rows: A3, A4
_slot("CTXR", RP)
_slot("IDW", R)
_slot("RSID", R)  # identity of the run start (most recent token after a stop)
_slot("F", 128)
_slot("QS", S)
_slot("QR", R)
_slot("RB", RP)
_slot("A", R)
NMOD = 17
DM = _o
F0 = L["F"].start
FN = ["STC", "SB1", "DEI", "FP", "PD", "PF", "YOU", "IMP", "PY", "PI", "PM", "ASST", "RA", "ONE", "PRE", "LVL", "SINK", "W", "STOP", "BND", "USER", "GREET", "BYE", "THANKS", "HARM", "QM",
      "UNI", "FIRST", "SP1", "SP2", "SP3", "SP4", "G1", "G2", "G3", "G4", "RS", "CT", "CS"]
FN += [f"ET{i}" for i in range(4)] + [f"MOD{i}" for i in range(NMOD)] + [f"TG{i}" for i in range(NMOD)]
FN += [f"HS{i}" for i in range(NHOP)] + [f"DS{i}" for i in range(1, NHOP + 1)]
FN += ["PG", "PB", "PT", "PH", "PQ", "PN", "PC", "PCB", "V1", "V2", "V3", "V4"]
FN += [f"MD{i}" for i in range(NMODE)] + [f"J{i}" for i in range(NPOSJ)]
assert len(FN) <= 128, len(FN)
f = {n: F0 + i for i, n in enumerate(FN)}
A_SLOTS = [L["A"]] + [slice(R + R * i, 2 * R + R * i) for i in range(4)]
assert A_SLOTS[-1].stop <= L["IDW"].stop

NL = 6
H = 7
DMLP = 1280
PHI = [1.0, 0.13]  # counter encoding frequencies
NT_MAX, NS_MAX = 120, 1100
KT, KS = 20.0, 400.0  # sink weights: counter c = K / (K + N) keeps knots well separated


def qwen(name):
    with safe_open(QW, "pt") as fh:
        return fh.get_tensor(name).float()


def one_id(s):
    t = ids(s)
    assert len(t) == 1, (s, t)
    return t[0]


def enc(n):
    out = []
    for p in PHI:
        out += [math.cos(n * p), math.sin(n * p)]
    return torch.tensor(out, dtype=torch.float64)


def omega():
    return THETA ** (-torch.arange(HALF, dtype=torch.float64) / HALF)


def pos_kernel(shift, pairs, amp):
    # (q, k) bias vectors so that q.k = amp * sum_p cos((delta - shift) * w_p).
    w = omega()
    q = torch.zeros(DH, dtype=torch.float64)
    k = torch.zeros(DH, dtype=torch.float64)
    for p in pairs:
        q[p], q[p + HALF] = amp * math.cos(shift * w[p]), -amp * math.sin(shift * w[p])
        k[p] = 1.0
    return q, k


class Head:
    def __init__(self, P, li, h):
        self.P, self.p, self.h, self.c = P, f"L{li}.", h, 0

    def content(self, qsrc, ksrc, scale=1.0):
        # Score += scale * <q-part, k-part>; qsrc/ksrc: list of (residual index or slice, weight-matrix/scalar).
        qsrc = qsrc if isinstance(qsrc, list) else [qsrc]
        ksrc = ksrc if isinstance(ksrc, list) else [ksrc]
        n = qsrc[0][1].shape[1] if torch.is_tensor(qsrc[0][1]) else 1
        dims = CDIMS[self.c:self.c + n]
        assert len(dims) == n, "out of content dims"
        self.c += n
        sq = math.sqrt(DH) * scale
        for src in qsrc:
            self._put("wq", "bq", src, dims, sq)
        for src in ksrc:
            self._put("wk", "bk", src, dims, 1.0)

    def _put(self, wn, bn, src, dims, s):
        idx, W = src
        W = W if torch.is_tensor(W) else torch.tensor([[float(W)]])
        if idx == "bias":
            self.P[self.p + bn][self.h, dims] += s * W[0].float()
        else:
            rows = list(range(idx.start, idx.stop)) if isinstance(idx, slice) else [idx]
            for r, row in enumerate(rows):
                self.P[self.p + wn][self.h, row, dims] += s * W[r].float()

    def pos(self, shift, pairs, amp):
        q, k = pos_kernel(shift, pairs, amp)
        self.P[self.p + "bq"][self.h] += (q * math.sqrt(DH)).float()
        self.P[self.p + "bk"][self.h] += k.float()
        self.P[self.p + "wk"][self.h, f["LVL"]] -= k.float()  # level rows (sinks) carry no position

    def value(self, src, dst, scale=1.0):
        # Copy residual src (index or slice) of the attended row into residual dst of the query row.
        rs = list(range(src.start, src.stop)) if isinstance(src, slice) else [src]
        ds = list(range(dst.start, dst.stop)) if isinstance(dst, slice) else [dst]
        assert len(rs) == len(ds)
        used = getattr(self, "vused", 0)
        for i, (r, d) in enumerate(zip(rs, ds)):
            self.P[self.p + "wv"][self.h, r, used + i] = 1.0
            self.P[self.p + "wo"][self.h, used + i, d] = scale
        self.vused = used + len(rs)


def sl(n):
    return slice(f[n], f[n] + 1)


def build():
    cfg = model.Config(vocab=V, D=R + NMODE, levels=(1, 64), codebooks=(NMODE, V), d_model=DM, n_layers=NL,
                       n_heads=H, d_head=DH, d_mlp=DMLP, dec_hidden=0, rope_theta=THETA, norm=False)
    P = {k: torch.zeros(*s) for k, s in model.shapes(cfg).items()}
    groups = []
    IS = torch.eye(S)

    # ---------- material from Qwen's tied embedding ----------
    E = qwen("model.embed_tokens.weight")
    En = E / E.norm(dim=1, keepdim=True)
    del E
    mu0 = En[:100000].mean(0)
    C = En - mu0
    del En
    torch.manual_seed(0)
    _, _, Vp = torch.pca_lowrank(C[:100000], q=S + 1, center=False)
    pc1 = C @ Vp[:, 0]
    if pc1[one_id(" the")] < 0:
        pc1 = -pc1
    sem = C @ Vp[:, 1:S + 1]
    sem = sem / sem.norm(dim=1, keepdim=True).clamp(min=1e-6)
    Cn = C / C.norm(dim=1, keepdim=True)

    def cen(ws):
        v = sum(Cn[one_id(w)] for w in ws)
        return v / v.norm()

    letters = C[[one_id(" " + c) for c in "BKM"]].mean(0)
    funcw = C[[one_id(w) for w in [" and", " the", " of"]]].mean(0)
    dfun = funcw - letters
    dfun = dfun / dfun.norm()
    fs = C @ dfun  # function-word axis (letters/content low, function words high)
    del C
    pu = Cn @ cen([".", ",", "?"])
    bn = Cn @ cen([".", "?", "!"])
    cl = Cn @ cen(["'m", "'s", "'re"])
    dq = Cn[one_id("?")] - Cn[one_id(".")]
    qm = (Cn @ dq) / dq.norm()

    def concept(ws, tau):
        c = torch.stack([Cn @ Cn[one_id(w)] for w in ws]).max(0).values
        return ((c - tau) / (1 - tau)).clamp(min=0)

    x_greet = concept(["Hello", " hi", " hey"], 0.3)
    x_bye = concept([" bye", "bye", " goodbye"], 0.55)
    x_thx = concept([" thanks", " thank", "Thanks"], 0.5)
    x_harm = concept([" kill", " bomb", " hack"], 0.45)
    x_you = concept([" you", " your", " yourself"], 0.6)
    x_imp = concept(["Write", "Tell", "Give"], 0.4)
    x_dei = concept([" it", " this", " that"], 0.7)
    x_fp = concept([" my", "My", " I"], 0.7)
    g = fs + 0.3 * pc1  # combined function-word score (function axis + frequency axis PC1)
    stop = ((fs > 0.11) | (g > 0.19) | ((pu > 0.55) & (fs > -0.05)) | (cl > 0.6)).float()
    pron = (concept([" you", " your", " yourself"], 0.6) + concept([" my", "My", " I"], 0.7)
            + concept([" it", " this", " that"], 0.7) + concept(["What", " Who", " Where"], 0.6)) > 0  # pronouns, wh-words
    stop = torch.maximum(stop, pron.float())
    digit = concept(["4", "1", "7"], 0.8) > 0  # number pieces are content and continue a span
    del Cn
    stop = stop * (~digit).float()
    w = ((0.21 - g) / 0.08).clamp(0, 1) * (1 - stop)
    w[digit] = 1.0
    stopc = ((fs > 0.11) | ((pu > 0.55) & (fs > -0.05)) | (cl > 0.6)).float()  # strict: ends a copied span
    stopc = stopc * (~digit).float()
    bnd = (bn > 0.84).float()
    qmk = (qm > 0.3).float()
    sp = [special("<|im_start|>"), special("<|im_end|>"), special("<|endoftext|>")]
    role = [one_id("user"), one_id("assistant"), one_id("\n")]
    for t in sp:
        stop[t], w[t], bnd[t], stopc[t] = 1, 0, 1, 1
    for t in role:
        stop[t], w[t], stopc[t] = 1, 0, 1

    gen = torch.Generator().manual_seed(1234)
    rid = torch.randn(V, R, generator=gen)
    rid[:, :RP] = rid[:, :RP] / rid[:, :RP].norm(dim=1, keepdim=True) / math.sqrt(2)
    rid[:, RP:] = rid[:, RP:] / rid[:, RP:].norm(dim=1, keepdim=True) / math.sqrt(2)

    TE = P["tok_emb"]
    TE[:, L["ID"]] = rid
    TE[:, L["SEMW"]] = sem * w[:, None]
    TE[:, L["IDW"]] = rid * w[:, None]
    TE[:, f["ONE"]] = 1
    TE[:, f["PRE"]] = 1
    TE[:, f["W"]] = w
    TE[:, f["STOP"]] = stop
    TE[:, f["STC"]] = stopc
    TE[:, f["BND"]] = bnd
    TE[one_id("user"), f["USER"]] = 1
    TE[one_id("assistant"), f["ASST"]] = 1
    TE[:, f["GREET"]] = x_greet
    TE[:, f["BYE"]] = x_bye
    TE[:, f["THANKS"]] = x_thx
    TE[:, f["HARM"]] = x_harm
    TE[:, f["YOU"]] = x_you
    TE[:, f["IMP"]] = x_imp
    TE[:, f["DEI"]] = x_dei
    TE[:, f["FP"]] = x_fp
    TE[:, f["QM"]] = qmk
    groups.append({"name": "tok_emb", "tensors": ["tok_emb"], "mechanism": "identity, semantics, scalar flags", "why": "see EXPLANATION.md section 2"})

    # level-row inputs
    P["lvl_emb"][:, f["ONE"]] = 1
    P["lvl_emb"][:, f["LVL"]] = 1
    P["lvl_pos"][0, f["SINK"]] = 1
    for j in range(NPOSJ):
        P["lvl_pos"][j, f["J0"] + j] = 1
    for m in range(NMODE):
        P["w_in"][R + m, f["MD0"] + m] = 1
    groups.append({"name": "level inputs", "tensors": ["lvl_emb", "lvl_pos", "w_in"], "mechanism": "sink, row index, mode one-hot", "why": "see EXPLANATION.md section 3"})

    ALLP = list(range(NPOS))
    # ================= L0: local structure =================
    h = Head(P, 0, 0)  # uniform average of im_start flag -> UNI (token 0 gets exactly 1)
    h.value(f["BND"], f["UNI"])
    for hh in range(1, 5):  # previous-token heads Δ = hh
        h = Head(P, 0, hh)
        h.pos(hh, ALLP, 20.0)
        h.content((f["LVL"], 2000.0), (f["LVL"], 1.0))  # level rows look at level rows (zero values)
        p0 = L["P"].start + (hh - 1) * RP
        h.value(slice(L["ID"].start, L["ID"].start + RP), slice(p0, p0 + RP))
        h.value(f["STC"], f[f"SP{hh}"])
        if hh == 1:
            h.value(f["STOP"], f["SB1"])
    h = Head(P, 0, 5)  # left-context bag (centred at Δ=2, width ~2)
    h.pos(2, list(range(1, 8)), 1.0)
    h.pos(0, list(range(8, NPOS)), 0.8)
    h.content((f["LVL"], 400.0), (f["LVL"], 1.0))
    h.value(L["SEMW"], L["CTX"])
    h.value(slice(L["IDW"].start, L["IDW"].start + RP), L["CTXR"], scale=2.0)
    groups.append({"name": "L0", "tensors": ["L0.*"], "mechanism": "previous-token heads, context bag", "why": "see EXPLANATION.md section 4"})
    # MLP0: FIRST flag and hop validity G_h, run start RS
    u = 0
    W1, B1, W2 = P["L0.w1"], P["L0.b1"], P["L0.w2"]
    W1[f["UNI"], u], B1[u], W2[u, f["FIRST"]] = 4.0, -3.0, 1.0
    u += 1
    for hh in range(1, 5):
        B1[u] = 1.0
        W1[f["STC"], u] = -1.0
        W1[f["LVL"], u] = -1.0
        for t in range(1, hh):
            W1[f[f"SP{t}"], u] = -1.0
        W2[u, f[f"G{hh}"]] = 1.0
        u += 1
    W1[f["SB1"], u], W2[u, f["RS"]] = 1.0, 1.0  # run start: previous token is a stop
    u += 1

    # ================= L1: counters =================
    h = Head(P, 1, 0)
    h.content((f["ONE"], 45.0), (f["USER"], 1.0))
    h.content((f["ONE"], 45.0 + math.log(KT)), (f["FIRST"], 1.0))
    h.value(f["FIRST"], f["CT"])
    h = Head(P, 1, 1)
    h.content((f["ONE"], 45.0), (f["BND"], 1.0))
    h.content((f["ONE"], math.log(KS)), (f["FIRST"], 1.0))
    h.value(f["FIRST"], f["CS"])
    h = Head(P, 1, 2)  # run-start identity: most recent token whose predecessor was a stop
    h.content((f["ONE"], 40.0), (f["RS"], 1.0))
    h.pos(0, ALLP, 15.0)
    h.content((f["LVL"], 2000.0), (f["LVL"], 1.0))
    h.value(L["ID"], L["RSID"])
    # MLP1: piecewise-linear decoders 1/(1+N) -> enc(N), 1/N -> enc(N), enc(N-2)+enc(N-3)
    W1, B1, W2 = P["L1.w1"], P["L1.b1"], P["L1.w2"]
    u = 0

    def knots(src, xs, ys, dsts):
        nonlocal u
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = torch.stack([ys[i] for i in order])
        slopes = (ys[1:] - ys[:-1]) / torch.tensor([xs[i + 1] - xs[i] for i in range(len(xs) - 1)])[:, None]
        # y(c) = ys0 + s0*relu(c-x0) + sum (s_i - s_{i-1}) relu(c - x_i)
        for d, j in zip(dsts, range(ys.shape[1])):
            P["L1.b2"][d] += ys[0, j].float()
        prev = torch.zeros(ys.shape[1], dtype=torch.float64)
        for i in range(len(xs) - 1):
            W1[src, u] = 1.0
            B1[u] = -xs[i]
            for d, j in zip(dsts, range(ys.shape[1])):
                W2[u, d] = float(slopes[i, j] - prev[j])
            prev = slopes[i]
            u += 1

    et = [f[f"ET{i}"] for i in range(4)]
    mods = [f[f"MOD{i}"] for i in range(NMOD)]
    tgs = [f[f"TG{i}"] for i in range(NMOD)]
    knots(f["CT"], [KT / (KT + n) for n in range(NT_MAX)], [enc(n) for n in range(NT_MAX)], et)
    E32 = torch.eye(NMOD, dtype=torch.float64)
    ys = [torch.cat([E32[n % NMOD], E32[(n - 2) % NMOD] + E32[(n - 3) % NMOD]]) for n in range(1, NS_MAX)]
    knots(f["CS"], [KS / (KS + n - 1) for n in range(1, NS_MAX)], ys, mods + tgs)
    MODS, TGS = slice(mods[0], mods[-1] + 1), slice(tgs[0], tgs[-1] + 1)
    ETS = slice(et[0], et[-1] + 1)
    assert u <= DMLP

    # ================= L2: question sentence bag and mode presence (level rows) =================
    lam = torch.tensor([100.0, 100.0]).repeat_interleave(2)
    h = Head(P, 2, 0)
    h.content((TGS, 40.0 * torch.eye(NMOD)), (MODS, torch.eye(NMOD)))
    h.content((ETS, torch.diag(lam)), (ETS, torch.eye(4)))
    h.value(L["SEMW"], L["QS"])
    h.value(L["IDW"], L["QR"])
    pres = [("GREET", "PG", 40.0, 7.0), ("BYE", "PB", 30.0, 7.0), ("THANKS", "PT", 30.0, 7.0),
            ("HARM", "PH", 30.0, 7.0), ("QM", "PQ", 30.0, 8.0)]
    for i, (src, dst, beta, tau) in enumerate(pres):
        h = Head(P, 2, 1 + i)
        h.content((slice(et[0], et[-1] + 1), torch.diag(lam)), (slice(et[0], et[-1] + 1), torch.eye(4)))
        h.content(("bias", torch.tensor([[beta]])), (f[src], 1.0))
        h.content(("bias", torch.tensor([[tau]])), (f["SINK"], 1.0))
        h.value(f["PRE"], f[dst])

    h = Head(P, 2, 6)  # role: is the most recent same-turn header 'assistant'?
    h.content((ETS, torch.diag(lam)), (ETS, torch.eye(4)))
    h.content(("bias", torch.tensor([[40.0]])), (f["USER"], 1.0))
    h.content(("bias", torch.tensor([[60.0]])), (f["ASST"], 1.0))
    h.content((f["LVL"], 2000.0), (f["LVL"], 1.0))  # level rows read themselves (RA = 0)
    h.value(f["ASST"], f["RA"])

    # ================= L3: fact match (right-support) and name candidate =================
    h = Head(P, 3, 0)
    h.content((L["QS"], 200.0 * IS), (L["SEMW"], IS))
    h.content((L["QR"], 200.0 * torch.eye(R)), (L["ID"], torch.eye(R)))
    h.content((TGS, -100.0 * torch.eye(NMOD)), (MODS, torch.eye(NMOD)))
    h.content(("bias", torch.tensor([[2.0]])), (f["W"], 1.0))
    h.content(("bias", torch.tensor([[-30.0]])), (f["RA"], 1.0))
    h.content(("bias", torch.tensor([[-20.0]])), (f["STOP"], 1.0))
    h.content(("bias", torch.tensor([[38.0]])), (f["SINK"], 1.0))
    h.pos(0, list(range(14, 21)), 2.0)
    h.value(L["CTXR"], L["RB"])
    h.value(f["PRE"], f["PM"])
    for i, (src, dst) in enumerate([("YOU", "PY"), ("IMP", "PI"), ("DEI", "PD"), ("FP", "PF")]):  # more presence detectors
        h = Head(P, 3, 2 + i)
        h.content((ETS, torch.diag(lam)), (ETS, torch.eye(4)))
        h.content(("bias", torch.tensor([[30.0]])), (f[src], 1.0))
        h.content(("bias", torch.tensor([[7.0]])), (f["SINK"], 1.0))
        h.value(f["PRE"], f[dst])
    h = Head(P, 3, 1)  # name: content run-start in the last user turn, not a greeting word
    h.content((slice(et[0], et[-1] + 1), torch.diag(lam)), (slice(et[0], et[-1] + 1), torch.eye(4)))
    h.content(("bias", torch.tensor([[6.0]])), (f["W"], 1.0))
    h.content(("bias", torch.tensor([[4.0]])), (f["RS"], 1.0))
    h.content(("bias", torch.tensor([[-12.0]])), (f["GREET"], 1.0))
    h.content(("bias", torch.tensor([[-8.0]])), (f["USER"], 1.0))
    h.content(("bias", torch.tensor([[8.0]])), (f["SINK"], 1.0))
    h.value(f["PRE"], f["PN"])

    W1, B1, W2 = P["L3.w1"], P["L3.b1"], P["L3.w2"]
    u = 0
    for m in COPY:
        T0 = len(ids(TMPL[m]))
        for hh in range(NHOP + 1):
            j = T0 + hh
            if j >= NPOSJ:
                continue
            W1[f["MD0"] + M[m], u], W1[f["J0"] + j, u], B1[u] = 1, 1, -1
            if hh < NHOP:
                W2[u, f[f"HS{hh}"]] = 1
            if hh >= 1:
                W2[u, f[f"DS{hh}"]] = 1
            u += 1

    BIG = 30.0
    gn = f["MD0"] + M["GREETNAME"]
    for sl_ in (L["QS"], L["QR"], TGS):
        for c in range(sl_.start, sl_.stop):
            for sgn in (1, -1):
                W1[c, u], B1[u] = sgn, -BIG
                for m_ in TOPIC:
                    W1[f["MD0"] + M[m_], u] = BIG
                W2[u, c] = -sgn
                u += 1

    # ================= L4: answer candidate =================
    h = Head(P, 4, 0)
    h.content((L["QS"], 300.0 * IS), (L["CTX"], IS))
    h.content([(L["QR"], -400.0 * torch.eye(R)), (L["RB"], 15.0 * torch.eye(RP, R))], (L["ID"], torch.eye(R)))
    h.content((TGS, -100.0 * torch.eye(NMOD)), (MODS, torch.eye(NMOD)))
    tm = [f["MD0"] + M[m_] for m_ in TOPIC]
    b5, b2 = ("bias", torch.tensor([[5.0]])), ("bias", torch.tensor([[2.0]]))
    kw, krs = (f["W"], 1.0), (f["RS"], 1.0)
    h.content([b5] + [(t_, 3.0) for t_ in tm], kw)
    h.content([b2] + [(t_, 4.0) for t_ in tm], krs)
    h.content([(t_, -12.0) for t_ in tm], (f["GREET"], 1.0))
    h.content(("bias", torch.tensor([[-8.0]])), (f["USER"], 1.0))
    h.content(("bias", torch.tensor([[-60.0]])), (f["RA"], 1.0))
    h.content(("bias", torch.tensor([[-60.0]])), (f["STOP"], 1.0))
    h.content(("bias", torch.tensor([[68.0]])), (f["SINK"], 1.0))
    REC = list(range(5, NPOS))  # monotone recency kernel for delta < ~30
    h.pos(0, list(range(6, NPOS)), 5.0)  # fact recency: newer statements win (corrections)
    q, _ = pos_kernel(0, REC, 4.0)  # GREETNAME: strong recency (the name is in the last turn)
    for t_ in tm:
        P["L4.wq"][0, t_] += (q * math.sqrt(DH)).float()
    h.value(L["RSID"], L["A"])
    h.value(f["PRE"], f["PC"])

    W1, B1, W2 = P["L4.w1"], P["L4.b1"], P["L4.w2"]  # binarize copy validity: PCB = clip(5*(PC-0.4), 0, 1)
    W1[f["PC"], 0], B1[0], W2[0, f["PCB"]] = 5.0, -3.0, 1.0
    W1[f["PC"], 1], B1[1], W2[1, f["PCB"]] = 5.0, -4.0, -1.0
    W1[f["MD0"] + M["ACK"], 2], W1[f["PF"], 2], W2[2, f["PCB"]] = 1.0, -1.0, -1.0  # ACK echoes only first-person facts

    # ================= L5: span hops =================
    PSL = [L["P"].start + i * RP for i in range(4)]
    for hh in range(1, NHOP):
        h = Head(P, 5, hh - 1)
        h.content((slice(L["A"].start, L["A"].start + RP), 160.0 * torch.eye(RP)),
                  (slice(PSL[hh - 1], PSL[hh - 1] + RP), torch.eye(RP)))
        h.content(("bias", torch.tensor([[30.0]])), (f[f"G{hh}"], 1.0))
        h.content(("bias", torch.tensor([[95.0]])), (f["SINK"], 1.0))
        h.pos(0, list(range(14, 21)), 1.0)
        h.value(L["ID"], A_SLOTS[hh])
        h.value(f["PRE"], f[f"V{hh}"])
    # MLP5: emission. OUT (=ID slot) <- template token / gated copy / '.' / nothing (-> <|im_end|> by bias)
    W1, B1, W2 = P["L5.w1"], P["L5.b1"], P["L5.w2"]
    u = 0
    dot = one_id(".")
    valid = [f["PCB"]] + [f[f"V{i}"] for i in range(1, NHOP)]
    for m in MODES:
        tt = ids(TMPL[m])
        for j, t in enumerate(tt):
            W1[f["MD0"] + M[m], u], W1[f["J0"] + j, u], B1[u] = 1, 1, -1
            W2[u, L["ID"]] = rid[t]
            u += 1
    for hh in range(NHOP):  # copy gate: A_hh[c] passes when hop-select HS_hh and valid_hh
        for c in range(R):
            for sgn in (1, -1):
                W1[A_SLOTS[hh].start + c, u] = sgn
                W1[f[f"HS{hh}"], u] = BIG
                W1[valid[hh], u] = BIG
                B1[u] = -2 * BIG
                if hh > 0:
                    W1[f["PCB"], u] = BIG
                    B1[u] = -3 * BIG
                W2[u, L["ID"].start + c] = sgn
                u += 1
    for hh in range(1, NHOP + 1):  # '.' right after the last valid copied token
        W1[f[f"DS{hh}"], u] = 1
        W1[valid[hh - 1], u] = 1
        if hh < NHOP:
            W1[valid[hh], u] = -1
        B1[u] = -1
        if hh > 1:
            W1[f["PCB"], u], B1[u] = 1, -2
        W2[u, L["ID"]] = rid[dot]
        u += 1
    t0 = len(ids(TMPL["ACK"]))  # ACK with nothing to copy ends with: that.
    for j, t in [(t0, one_id(" that")), (t0 + 1, dot)]:
        W1[f["MD0"] + M["ACK"], u], W1[f["J0"] + j, u], W1[f["PCB"], u], B1[u] = 1, 1, -1, -1
        W2[u, L["ID"]] = rid[t]
        u += 1
    assert u <= DMLP, u

    for li, what in [(1, "turn, sentence, run-start counters"), (2, "question bag, presence, role"),
                     (3, "fact match, name presence"), (4, "answer candidate selection"), (5, "span hops, emission MLP")]:
        groups.append({"name": f"L{li}", "tensors": [f"L{li}.*"], "mechanism": what, "why": "see EXPLANATION.md section 5"})
    # ================= heads, codes, decoder =================
    # level 0: mode from presence features
    hd = P["head0"]
    lin = {"ACK": {"ONE": 0.25}, "GREET": {"PG": 1.2, "PN": -0.4, "ONE": -0.1}, "GREETNAME": {"PG": 1.5, "PN": 0.6, "PQ": -1.0, "ONE": -1.0},
           "BYE": {"PB": 2.0, "ONE": -0.2}, "THANKS": {"PT": 1.8, "ONE": -0.2}, "DECLINE": {"PH": 3.0, "ONE": -1.2},
           "ANSWER": {"PQ": 1.0, "PCB": 0.5, "PM": 0.5, "ONE": -1.5},
           "SELF": {"PQ": 1.0, "PY": 1.0, "PM": -0.5, "ONE": -1.0}, "REQUEST": {"PI": 1.0, "PQ": -1.0, "ONE": -0.3},
           "CLARIFY": {"PD": 1.0, "PF": -1.0, "PQ": -1.0, "PI": -0.5}, "UNKNOWN": {"PQ": 1.0, "ONE": -0.7}}
    for m, d in lin.items():
        for k, v in d.items():
            hd[f[k], M[m]] = v
    for m in range(NMODE):
        P["code0"][m, R + m] = 1.0
    # level 1: token from OUT by nearest identity code; empty OUT -> <|im_end|>
    P["head1"][L["ID"]] = rid.T * 20.0
    P["head_b1"][special("<|im_end|>")] = 6.0
    P["code1"][:, :R] = rid
    P["dec_w2"][:R] = rid.T
    P["dec_b2"][:] = -0.5
    P["dec_b2"][special("<|im_end|>")] = -0.45  # an empty latent (after level 0) decodes to the end token
    groups.append({"name": "codes/decoder", "tensors": ["head0", "code0", "head1", "head_b1", "code1", "dec_w2", "dec_b2"], "mechanism": "mode choice, nearest identity code", "why": "see EXPLANATION.md section 6"})
    return cfg, P, groups
