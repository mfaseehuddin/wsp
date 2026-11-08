"""ncd60: a 4-level Next-Concept chat model on top of Qwen2.5-0.5B layer 0 (verbatim). No training, no fitting.

Levels (coarse -> fine):  L1 dialogue act (1 row) -> L2 content word (4 rows, vocab codes, copied from the
conversation) -> L3 phrase skeleton (16 rows, (act, block) codes) -> L4 tokens (64 rows).
Every number below is a formula over Qwen weights, anchor words (<= 3 per direction) or hand-set constants.
See EXPLANATION.md for the mechanism of every block.
"""
import math

import torch

from wsp.ncmchat import model, qwen
from wsp.ncmchat.harness import ids

S = math.sqrt(128.0)          # attention divides by sqrt(d_head)
C0 = 3.0e4                    # constant ONE feature: makes every later RMSNorm a fixed scale (layers become linear)
NA = 10                       # acts
ACTS = ["GREET", "GREETNAME", "BYE", "THANKS", "ACK", "ANSWER", "UNKNOWN", "DECLINE", "SELF", "CLARIFY"]
# Template pieces (each <= 6 words). Copy acts put a copied word + "." after their piece.
TPL = {"GREET": "Hello! How can I help you?", "GREETNAME": "Hello, nice to meet you,", "BYE": "Goodbye! Have a great day.",
       "THANKS": "You are welcome! Happy to help.", "ACK": "Okay, I will remember that.", "ANSWER": "I think it is",
       "UNKNOWN": "Sorry, I do not know that.", "DECLINE": "Sorry, I cannot help with that.",
       "SELF": ["I am a small chat program.", " I remember what you tell me."], "CLARIFY": "What do you mean by it?"}
COPY = ("GREETNAME", "ANSWER")
PRIO = {"DECLINE": 10, "BYE": 9, "THANKS": 8, "GREETNAME": 7, "GREET": 6, "SELF": 5, "ANSWER": 4, "UNKNOWN": 3, "CLARIFY": 2, "ACK": 1}


def tid(w):
    t = ids(w)
    assert len(t) == 1, (w, t)
    return t[0]


def layer0_code():
    """F1[t] = verbatim Qwen layer 0 applied to token t right after '<|im_start|>user\\n' (whole vocab, chunked)."""
    E = qwen.embed(); W = qwen.layer(0)
    hdr = torch.tensor(ids("<|im_start|>user\n"))
    _, ck, cv = qwen.block(0, E[hdr], torch.arange(3), torch.zeros(2, 0, 64), torch.zeros(2, 0, 64))
    Kc, Vc = ck.repeat_interleave(7, 0), cv.repeat_interleave(7, 0)
    out = torch.empty(E.shape[0], 896)
    for s in range(0, E.shape[0], 8192):
        x = E[s:s + 8192]; n = x.shape[0]
        h = qwen.rms(x, W["input_layernorm.weight"])
        q = (h @ W["self_attn.q_proj.weight"].T + W["self_attn.q_proj.bias"]).view(n, 14, 64)
        k = (h @ W["self_attn.k_proj.weight"].T + W["self_attn.k_proj.bias"]).view(n, 2, 64)
        v = (h @ W["self_attn.v_proj.weight"].T + W["self_attn.v_proj.bias"]).view(n, 2, 64)
        pos = torch.full((n,), 3)
        q = qwen._rope(q.transpose(0, 1), pos).transpose(0, 1); k = qwen._rope(k.transpose(0, 1), pos).transpose(0, 1)
        kk, vv = k.repeat_interleave(7, 1), v.repeat_interleave(7, 1)
        sc = torch.cat([torch.einsum("nhd,hmd->nhm", q, Kc), (q * kk).sum(-1, keepdim=True)], -1) / 8
        a = sc.softmax(-1)
        o = torch.einsum("nhm,hmd->nhd", a[..., :3], Vc) + a[..., 3:] * vv
        x = x + o.reshape(n, 896) @ W["self_attn.o_proj.weight"].T
        h = qwen.rms(x, W["post_attention_layernorm.weight"])
        x = x + (torch.nn.functional.silu(h @ W["mlp.gate_proj.weight"].T) * (h @ W["mlp.up_proj.weight"].T)) @ W["mlp.down_proj.weight"].T
        out[s:s + n] = x
    return out


class Dims:
    def __init__(self):
        self.n = 0; self.names = {}

    def __call__(self, name, k=1):
        r = list(range(self.n, self.n + k)); self.names[name] = r; self.n += k
        return r if k > 1 else r[0]


def build():
    # ---------------- material from Qwen ----------------
    F = layer0_code()
    mu = F.mean(0); big = mu.abs() > 10 * mu.abs().median()          # 7 massive-activation dims, ignored
    Fc = F - mu; Fc[:, big] = 0
    ev, U = torch.linalg.eigh(Fc.T @ Fc / Fc.shape[0]); U = U.flip(1)[:, :96].contiguous(); U[big] = 0
    del Fc
    G = F @ U; del F                                                   # uncentred 96-d token code (what pre_proj gives)
    m = G.mean(0); C = G - m                                           # centred code
    Gn = G / G.norm(dim=1, keepdim=True).clamp(min=1e-6)
    Cn = C / C.norm(dim=1, keepdim=True).clamp(min=1e-6)
    V = G.shape[0]; END = tid("<|im_end|>")

    def dirn(pos, neg=None):
        v = Gn[[tid(w) for w in pos]].mean(0)
        if neg: v = v - Gn[[tid(w) for w in neg]].mean(0)
        return v / v.norm()

    # ---------------- dims ----------------
    d = Dims()
    ID = d("ID", 96)            # prefix: token code (centred after MLP0); level rows: OUT (L4) / CSTAR (L2)
    ONE, LVL, SINK = d("ONE"), d("LVL"), d("SINK")
    LEV = d("LEV", 4); J = d("J", 25)
    FL = {k: d(k) for k in ["SPC", "PERIOD", "QMARK", "EXCL", "COLON", "NL", "COMMA", "GREET", "BYE", "THX", "HARM",
                             "WH", "YOU", "DEIX", "WS", "USER", "ASSIST", "FIRSTP"]}
    GN = d("GN", 96)            # prefix-only (read at L2); level rows reuse these dims for QB/TQ/DENS/IND
    POSINV, ROLEA = d("POSINV"), d("ROLEA")
    NL = d("NL")
    MEANSPC, FIRST, INVB, NRM, CONTENT, CONTF = d("MEANSPC"), d("FIRST"), d("INVB"), d("NRM"), d("CONTENT"), d("CONTF")
    ENC0, ENCT, ENCM1 = d("ENC0", 12), d("ENCT", 12), d("ENCM1", 12)
    BAGB = d("BAGB", 32)
    BAG, PREV1, PREV2 = d("BAG", 32), d("PREV1", 32), d("PREV2", 32)
    P1CONTF, P1WH, P2CONTF = d("P1CONTF"), d("P1WH"), d("P2CONTF")
    ACTZ, CZ = d("ACTZ", NA), d("CZ", 32)
    SID, NOMATCH, GOT1, GOT2 = d("SID", 12), d("NOMATCH"), d("GOT1"), d("GOT2")
    OFF = d("OFF", 4); ACTS_ = d("ACTSCORE", NA); OUTEND = d("OUTEND")
    QB, TQ = GN[0:32], GN[32:64]
    DN = {k: GN[64 + i] for i, k in enumerate(["GREET", "BYE", "THX", "HARM", "QMARK", "WH", "YOU", "DEIX", "CONTENT"])}
    IND = {k: GN[73 + i] for i, k in enumerate(["H", "B", "T", "G", "C", "Q", "Y", "D"])}
    global DIMS; DIMS = d.names
    dm = ((d.n + 31) // 32) * 32
    ID32 = ID[4:36]; IDt = tuple(ID)                                  # match code: PCs 4..35 (PCs 0-3 are shared by all content words)

    TOKS = {a: (ids(TPL[a]) if isinstance(TPL[a], str) else sum((ids(p) for p in TPL[a]), [])) for a in ACTS}
    D_ACT, D_CONT, D_TOK = list(range(0, NA)), list(range(NA, NA + 32)), list(range(NA + 32, NA + 128))
    Dlat = NA + 129; D_END = NA + 128
    cfg = model.Config(vocab=V, D=Dlat, levels=(1, 4, 16, 64), codebooks=(NA, V, NA * 16, V), d_model=dm, n_layers=6,
                       n_heads=10, d_head=128, d_mlp=1280, dec_hidden=0, rope_theta=0.7 ** -64, norm=True,
                       qwen_embed=True, qwen_layers=(0,), qwen_final_norm=False,
                       level_meanings=("dialogue act", "content word copied from the conversation",
                                       "phrase skeleton: template token per 4-token block", "tokens"))
    P = {k: torch.zeros(*s) for k, s in model.shapes(cfg).items()}
    om = 0.7 ** torch.arange(64, dtype=torch.float64)                 # RoPE angular speeds (theta = 0.7^-64)
    STATIC = list(range(32, 64)) + list(range(96, 128))
    groups = []

    # ---------------- helpers ----------------
    def slots(L, h):
        key = (L, h)
        used.setdefault(key, 0)
        return key

    used = {}

    def take(L, h, k):
        i = used.get((L, h), 0); used[(L, h)] = i + k
        assert i + k <= 64
        return STATIC[i:i + k]

    def term(L, h, q=(0.0, {}), k=(0.0, {})):
        s = take(L, h, 1)[0]; p = f"L{L}."
        P[p + "bq"][h, s] += q[0] * S
        for dd, w in q[1].items(): P[p + "wq"][h, dd, s] += w * S
        P[p + "bk"][h, s] += k[0]
        for dd, w in k[1].items(): P[p + "wk"][h, dd, s] += w

    def dot(L, h, qd, kd, scale, sl=None):
        sl = sl or take(L, h, len(qd)); p = f"L{L}."
        for a, b, s in zip(qd, kd, sl):
            P[p + "wq"][h, a, s] += scale * S; P[p + "wk"][h, b, s] += 1.0

    def kern(L, h, s, amp):                                             # amp * sum_p cos((delta - s) w_p), pairs 0..7
        p = f"L{L}."
        for i in range(8):
            P[p + "bq"][h, i] += amp * S * math.cos(s * om[i]); P[p + "bq"][h, i + 64] += -amp * S * math.sin(s * om[i])
            P[p + "bk"][h, i] += 1.0

    def recency(L, h, mu_):                                             # -mu sin(delta w_21): prefers recent rows
        p = f"L{L}."
        P[p + "bq"][h, 85] += mu_ * S; P[p + "bk"][h, 21] += 1.0

    vslot = {}

    def val(L, h, src, dst, const=0.0):                                # one value channel: sum src -> dst
        i = vslot.get((L, h), 0); vslot[(L, h)] = i + 1; p = f"L{L}."
        for dd, w in src.items(): P[p + "wv"][h, dd, i] += w
        P[p + "bv"][h, i] += const; P[p + "wo"][h, i, dst] += 1.0

    def copyvec(L, h, src, dst):
        for a, b in zip(src, dst): val(L, h, {a: 1.0}, b)

    def no_level_keys(L, h, B=300.0, sink=None):
        term(L, h, (-B, {}), (0.0, {LVL: 0.01}))                        # LVL = 100 on level rows
        if sink is not None:                                            # sink = level row 0, score sink[0] + sink gate
            term(L, h, (B + sink[0], sink[1]), (0.0, {SINK: 1.0}))

    mlp = {}

    def unit(L, win, b, wout):
        i = mlp.get(L, 0); mlp[L] = i + 1; p = f"L{L}."
        assert i < cfg.d_mlp, L
        for dd, w in win.items(): P[p + "w1"][dd, i] += w
        P[p + "b1"][i] += b
        for dd, w in wout.items():
            if isinstance(dd, tuple): P[p + "w2"][i, list(dd)] += w
            else: P[p + "w2"][i, dd] += w

    def clip01(L, win, b, dst, scale=1.0):                              # dst += scale * clip(win.x + b, 0, 1)
        unit(L, win, b, {dst: scale}); unit(L, win, b - 1.0, {dst: -scale})

    # norms: layer 0 sees raw rows (cosine geometry); later layers see x * C0/rms ~ x  (ONE dominates rms)
    for L in range(6):
        P[f"L{L}.n1"][:] = 1.0 if L == 0 else C0 / math.sqrt(dm)
        P[f"L{L}.n2"][:] = 1.0 if L == 0 else C0 / math.sqrt(dm)
        if L: P[f"L{L}.n1"][ONE] = 0; P[f"L{L}.n2"][ONE] = 0

    # ---------------- inputs ----------------
    P["pre_proj"][:, ID] = U                                            # 896 -> 96-d token code (top PCs of layer-0 codes)
    for l in range(4):
        P["lvl_emb"][l, LVL] = 100.0; P["lvl_emb"][l, LEV[l]] = 1.0; P["lvl_emb"][l, ID] = m   # +m cancels MLP0's -m
    for j in range(64): P["lvl_pos"][j, J[min(j, 24)]] = 1.0
    P["lvl_pos"][0, SINK] = 1.0
    for a in range(NA): P["w_in"][D_ACT[a], ACTZ[a]] = 1.0
    for i in range(32): P["w_in"][D_CONT[i], CZ[i]] = 1.0
    groups.append({"name": "inputs", "tensors": ["pre_proj", "lvl_emb", "lvl_pos", "w_in"],
                   "mechanism": "prefix row = Qwen layer-0 residual projected on the top-96 principal axes of the layer-0 token codes; level rows carry level flag, row one-hot, sink flag, and the act/content read back from z",
                   "why": "the 96-d code identifies every content token exactly (100% on test text), so copying is exact"})

    # ---------------- layer 0: token features (MLP only; input = unit vector of the code) ----------------
    r = 1.0 / math.sqrt(dm); KILL = {LVL: -10.0}
    exact = {"USER": "user", "ASSIST": "assistant", "SPC": "<|im_start|>", "PERIOD": ".", "QMARK": "?", "EXCL": "!", "COLON": ":", "NL": "\n", "COMMA": ","}
    dirs = {"GREET": ([" Hello", "Hi", " hey"], 0.65), "BYE": ([" bye", " goodbye", " farewell"], 0.65),
            "THX": ([" thanks", " Thank", " thank"], 0.65), "HARM": ([" kill", " bomb", " weapon"], 0.48),
            "WH": ([" What", " Who", " How"], 0.60), "YOU": ([" you", " your", " yourself"], 0.60),
            "DEIX": ([" it", " this", " that"], 0.50), "FIRSTP": ([" I", " my", " me"], 0.60)}
    C_spc = C[tid("<|im_start|>")]
    for k, w in exact.items():
        u = Gn[tid(w)]; a = 1 / 0.02
        win = {ID[i]: a * r * u[i].item() for i in range(96)} | KILL
        extra = {IDt: -C_spc} if k == "SPC" else {}
        th = {"USER": 0.93, "ASSIST": 0.92, "SPC": 0.95, "EXCL": 0.85, "QMARK": 0.8, "COLON": 0.75}.get(k, 0.7)
        unit(0, win, -a * th, {FL[k]: 1.0} | {kk: vv for kk, vv in extra.items()})
        unit(0, win, -a * th - 1, {FL[k]: -1.0} | {kk: -vv for kk, vv in extra.items()})
    for k, (anc, tau) in dirs.items():
        u = dirn(anc); a = 1 / 0.05
        clip01(0, {ID[i]: a * r * u[i].item() for i in range(96)} | KILL, -a * tau, FL[k])
    E0 = qwen.embed()[tid("<|im_start|>")][None]                     # row 0 code: <|im_start|> at position 0 (sink row)
    x0, _, _ = qwen.block(0, E0, torch.arange(1), torch.zeros(2, 0, 64), torch.zeros(2, 0, 64)); g0 = (x0[0] @ U); g0 = g0 / g0.norm()
    c0 = x0[0] @ U - m                                                  # row 0's centred code: removed (it is not a word)
    win0 = {ID[i]: 100 * r * g0[i].item() for i in range(96)} | KILL
    unit(0, win0, -100 * 0.985, {FIRST: 1.0, IDt: -c0}); unit(0, win0, -100 * 0.985 - 1, {FIRST: -1.0, IDt: c0})
    u = dirn([" water", " run", " blue"], ["water", "run", "blue"])     # leading-space direction (3 word pairs)
    WS_TAU = 0.0
    clip01(0, {ID[i]: 20 * r * u[i].item() for i in range(96)} | KILL, -20 * WS_TAU + 0.5, FL["WS"])
    for i in range(96):                                                 # pass the unit code on (for the norm head)
        unit(0, {ID[i]: r} | KILL, 0.0, {GN[i]: 1.0}); unit(0, {ID[i]: -r} | KILL, 0.0, {GN[i]: -1.0})
    P["L0.b2"][ONE] = C0; P["L0.b2"][ID] = -m
    groups.append({"name": "L0 MLP token features", "tensors": ["L0.w1", "L0.b1", "L0.w2", "L0.b2"],
                   "mechanism": "cosine of the code with anchor directions / exact token codes -> 0/1 flags; centres the code; writes ONE",
                   "why": "everything later is linear in these flags"})

    # ---------------- layer 1: sentence counter, render offsets ----------------
    BNDS = {FL[k]: 1.0 for k in ["SPC", "PERIOD", "QMARK", "EXCL", "COLON"]} | {FIRST: 1.0}
    term(1, 0, (30.0, {}), (0.0, BNDS)); term(1, 0, (math.log(100.0), {}), (0.0, {FIRST: 1.0}))
    val(1, 0, {FIRST: 1.0}, INVB)                                       # v = 100 / (99 + N), N = boundaries so far
    v = lambda n: 100.0 / (99.0 + n)
    P["L1.b2"][NL] += 600.0; prevs = 0.0                                 # MLP1: N = 100/v - 99, knots at v(N)
    for n in range(600, 1, -1):
        sl_ = (n - 1 - n) / (v(n - 1) - v(n))
        unit(1, {INVB: 1.0}, -v(n), {NL: sl_ - prevs}); prevs = sl_
    for a in COPY:
        La = len(TOKS[a])
        for k in range(4):
            pi = La + k
            if pi < 24: unit(1, {J[pi]: 1.0, LEV[3]: 1.0, ACTZ[ACTS.index(a)]: 1.0}, -2.0, {OFF[k]: 1.0})
    groups.append({"name": "L1", "tensors": ["L1.*"], "mechanism": "h0 boundary counter v=100/(99+N); MLP1 turns v into N (piecewise linear) and makes slot-offset flags OFF0..3 for L4 rows", "why": "N numbers the sentences; OFF tells a token row which copied piece it holds"})

    # ---------------- layer 2: sentence counter, content (self-norm) ----------------
    # head 1: self score lam*Ghat.C (sharp offset-0 kernel) against the FIRST row at threshold tau
    lam, tau_s, Pk = 20.0, 3.6, 400.0
    kern(2, 1, 0, Pk)
    P["L2.wk"][1, FIRST, list(range(8))] += -1.0                        # FIRST row: no positional bonus ...
    sd = [i for i in range(128) if not (i < 8 or 64 <= i < 72)][:97]
    for i in range(96):
        P["L2.wq"][1, GN[i], sd[i]] += lam * S; P["L2.wk"][1, ID[i], sd[i]] += 1.0
    P["L2.bq"][1, sd[96]] += (8 * Pk + lam * tau_s) * S; P["L2.wk"][1, FIRST, sd[96]] += 1.0   # ... but a fixed score
    val(2, 1, {FIRST: -1.0}, NRM, const=1.0)
    # MLP2: piecewise-linear decoders 1/N -> enc(N), enc(N-2)+enc(N-3)
    NMAX, MC = 600, 100.0
    wf = 2 * math.pi / torch.tensor([997., 131., 37., 11., 5.3, 3.1], dtype=torch.float64)
    def enc(n): return torch.cat([torch.cos(n * wf), torch.sin(n * wf)])
    def fvec(n): return torch.cat([enc(n), enc(n - 2) + enc(n - 3), enc(n - 1)])
    dst = ENC0 + ENCT + ENCM1
    P["L2.b2"][dst] += fvec(1).float()                                  # MLP2: N -> enc codes, knots at integers
    pidx = list(range(12)) + list(range(24, 36)); lidx = list(range(12, 24))
    unit(2, {LVL: 0.01}, 0.0, {dst[i]: -fvec(1)[i].item() for i in pidx})   # level rows: no own sentence code
    for idx, gate in ((pidx, {LVL: -10.0}), (lidx, {})):               # prefix codes (off on level rows); target code
        prev = torch.zeros(36, dtype=torch.float64)
        for n in range(1, NMAX):
            sl_ = fvec(n + 1) - fvec(n)
            unit(2, {NL: 1.0} | gate, -float(n), {dst[i]: (sl_ - prev)[i].item() for i in idx}); prev = sl_
    PUNC = {FL[k]: -3.0 for k in ["SPC", "PERIOD", "QMARK", "EXCL", "COLON", "NL", "COMMA", "GREET", "BYE", "THX", "WH", "YOU", "DEIX", "USER", "ASSIST", "FIRSTP"]}
    clip01(2, {NRM: 3.0} | PUNC, -1.5, CONTENT)
    clip01(2, {FL["WS"]: -1.0} | {k: -1.0 for k in PUNC}, 1.0, CONTF)
    groups.append({"name": "L2", "tensors": ["L2.*"], "mechanism": "h1: self-attention whose score is Ghat.C (a norm proxy = word frequency) vs the FIRST row -> content flag; MLP2: N -> sentence-id code enc(N) and target code enc(N-2)+enc(N-3)", "why": "sentence ids let level rows find the question and the fact sentences"})

    # ---------------- layer 3: bags, previous tokens, question bag, act densities ----------------
    A = 15.0
    dot(3, 0, ENC0, ENC0, A); term(3, 0, (8.0, {}), (0.0, {CONTENT: 1.0}))
    copyvec(3, 0, ID32, BAG)
    dot(3, 7, ENCM1, ENC0, A); term(3, 7, (8.0, {}), (0.0, {CONTENT: 1.0})); copyvec(3, 7, ID32, BAGB)   # bag of the sentence a boundary closes
    for h, s_, dstc in ((1, 1, PREV1), (3, 2, PREV2)):
        kern(3, h, s_, 60.0); copyvec(3, h, ID32, dstc)
    dot(3, 8, ENC0, ENC0, A); val(3, 8, BNDS, POSINV)                  # 1/(index within sentence + 1)
    term(3, 9, (30.0, {}), (0.0, {FL["USER"]: 1.0, FL["ASSIST"]: 1.0})); recency(3, 9, 2000.0)
    val(3, 9, {FL["ASSIST"]: 1.0}, ROLEA)                              # 1 inside assistant turns
    for h in (0, 1, 2, 3, 4, 7, 8, 9):
        term(3, h, (0.0, {LVL: 3.0}), (0.0, {SINK: 1.0}))                # level rows -> sink (value 0)
    kern(3, 2, 1, 60.0); val(3, 2, {CONTF: 1.0}, P1CONTF); val(3, 2, {FL["WH"]: 1.0}, P1WH)
    kern(3, 4, 2, 60.0); val(3, 4, {CONTF: 1.0}, P2CONTF)
    dot(3, 5, ENCT, ENC0, A); term(3, 5, (8.0, {}), (0.0, {CONTENT: 1.0})); no_level_keys(3, 5)
    copyvec(3, 5, ID32, QB)
    dot(3, 6, ENCT, ENC0, A); no_level_keys(3, 6)
    for k in DN: val(3, 6, {(CONTENT if k == "CONTENT" else FL[k]): 1.0}, DN[k])
    # MLP3: act indicators from densities (level rows)
    for k, src in (("H", ["HARM"]), ("B", ["BYE"]), ("T", ["THX"]), ("G", ["GREET"]), ("C", ["CONTENT"]), ("Q", ["QMARK", "WH"]),
                   ("Y", ["YOU"]), ("D", ["DEIX"])):
        clip01(3, {DN[s]: 60.0 for s in src}, 0.0, IND[k])
    groups.append({"name": "L3", "tensors": ["L3.*"], "mechanism": "h0 sentence bag (content mean); h1-h4 previous-token copies (code, continuation flag, wh flag); h5 question bag over the target sentences; h6 flag densities over the target sentences; MLP3 0/1 indicators", "why": "inputs for sentence selection, continuation copying and the act decision"})

    # ---------------- layer 4: sentence select, type word, continuation pieces, act ----------------
    lam_s, theta_s = 5.0, 4.0
    dot(4, 0, QB, BAGB, lam_s); term(4, 0, (100.0, {}), (0.0, {FL["PERIOD"]: 1.0, FL["EXCL"]: 1.0, ROLEA: -1.0})); recency(4, 0, 60.0)
    no_level_keys(4, 0, sink=(100.0 + theta_s, {}))
    copyvec(4, 0, ENCM1, SID); val(4, 0, {SINK: 1.0}, NOMATCH)
    dot(4, 1, ENCT, ENC0, A); term(4, 1, (20.0, {}), (0.0, {P1WH: 1.0, CONTENT: 1.0})); no_level_keys(4, 1, sink=(9.76 * A + 30.0, {}))
    copyvec(4, 1, ID32, TQ)
    lam_m = 6.0; B2 = 400.0
    # h2: piece 0 (the content token itself), h3: piece 1 code, h4: piece 2 code; h5/h6: piece-1/2 existence flags
    for h, kd, extra, gate in ((2, ID32, {CONTENT: 4.0}, [OFF[0]]), (3, PREV1, {CONTF: 20.0}, [OFF[1]]),
                               (4, PREV2, {CONTF: 20.0, P1CONTF: 20.0}, [OFF[2]]),
                               (5, PREV1, {CONTF: 20.0}, [OFF[1], OFF[2]]), (6, PREV2, {CONTF: 20.0, P1CONTF: 20.0}, [OFF[2], OFF[3]])):
        sl = take(4, h, 32)
        for i in range(32):                                             # prefix key: code of the row k back; sink key: 0.6*CZ
            P["L4.wq"][h, CZ[i], sl[i]] += lam_m * S; P["L4.wk"][h, kd[i], sl[i]] += 1.0; P["L4.wk"][h, CZ[i], sl[i]] += 0.6
        term(4, h, (1.0, {}), (-sum(extra.values()), extra))           # rows lacking the flags lose 8 each
        no_level_keys(4, h, sink=(B2, {g: -B2 for g in gate}))         # rows not at this slot offset -> sink
        term(4, h, (B2, {LVL: -B2 / 100.0}), (0.0, {FIRST: 1.0}))      # prefix rows -> row 0 (code zeroed): no write
        recency(4, h, 30.0)
        if h <= 4: copyvec(4, h, ID, ID)
        else: val(4, h, {SINK: -1.0}, GOT1 if h == 5 else GOT2, const=1.0)
    # level rows other than the sink must not match through dot(CZ, CZ): they are removed by no_level_keys (-300)
    # MLP4: act scores + GREETNAME query gating
    P["L4.b2"][ACTS_[ACTS.index("ACK")]] += PRIO["ACK"]
    for a, ks in (("DECLINE", ["H"]), ("BYE", ["B"]), ("THANKS", ["T"]), ("GREET", ["G"]), ("UNKNOWN", ["Q"]), ("CLARIFY", ["D"]),
                  ("GREETNAME", ["G", "C"]), ("SELF", ["Q", "Y"])):
        clip01(4, {IND[k]: 1.0 for k in ks}, -(len(ks) - 1.0), ACTS_[ACTS.index(a)], scale=PRIO[a])
    clip01(4, {IND["Q"]: 1.0, NOMATCH: -1.0}, 0.0, ACTS_[ACTS.index("ANSWER")], scale=PRIO["ANSWER"])
    gn = ACTZ[ACTS.index("GREETNAME")]; R = 30.0
    for i in range(12):                                                 # SID <- ENCT when the act is GREETNAME
        unit(4, {ENCT[i]: 1.0, SID[i]: -1.0, gn: R}, -R, {SID[i]: 1.0}); unit(4, {ENCT[i]: -1.0, SID[i]: 1.0, gn: R}, -R, {SID[i]: -1.0})
    for i in range(32):                                                 # QB, TQ <- 0 when GREETNAME
        for blk in (QB, TQ):
            unit(4, {blk[i]: 1.0, gn: R}, -R, {blk[i]: -1.0}); unit(4, {blk[i]: -1.0, gn: R}, -R, {blk[i]: 1.0})
    groups.append({"name": "L4", "tensors": ["L4.*"], "mechanism": "h0 picks the earlier sentence whose bag best matches the question bag (sink = no match); h1 type word after a wh-word; h2-h4 copy word pieces 0..2 of the content token for slot rows; h5/h6 piece-exists flags; MLP4 act priority scores, GREETNAME re-targets the content query to the user's own sentence", "why": "act and copy pointers"})

    # ---------------- layer 5: content selection, rendering ----------------
    dot(5, 0, SID, ENC0, A)
    term(5, 0, (10.0, {}), (0.0, {CONTENT: 4.0, FL["WS"]: 1.5, FL["GREET"]: -2.0, FIRST: -2.0, FL["SPC"]: -2.0, ROLEA: -2.0} | {k: -2.0 for k in BNDS}))
    gnz = ACTZ[ACTS.index("GREETNAME")]
    term(5, 0, (16.0, {gnz: -16.0}), (0.0, {POSINV: 1.0}))               # answers: prefer early words (subjects)
    P["L5.wq"][0, gnz, 85] += 40.0 * S; P["L5.bk"][0, 21] += 1.0        # names: prefer the most recent word
    sl = take(5, 0, 32)
    for i in range(32):                                                 # -1.5 * in-question + 0.6 * type similarity
        P["L5.wq"][0, QB[i], sl[i]] += -2.0 * S; P["L5.wq"][0, TQ[i], sl[i]] += 3.0 * S; P["L5.wk"][0, ID32[i], sl[i]] += 1.0
    dot(5, 0, QB[:16], BAG[:16], 0.5)
    no_level_keys(5, 0, sink=(B2, {LEV[1]: -B2}))
    copyvec(5, 0, ID, ID)                                               # L2 rows: CSTAR = code of the chosen token
    for a in ACTS:
        La, T = len(TOKS[a]), TOKS[a]; ai = ACTZ[ACTS.index(a)]
        for pi in range(24):
            if pi < La: tok = T[pi]
            elif a in COPY and pi < La + 4: continue
            else: tok = tid("<|im_end|>")
            unit(5, {J[pi]: 1.0, LEV[3]: 1.0, ai: 1.0}, -2.0, {IDt: C[tok], OUTEND: float(tok == END)})
    unit(5, {J[24]: 1.0, LEV[3]: 1.0}, -1.0, {IDt: C[END], OUTEND: 1.0})
    dotc, endc = C[tid(".")], C[tid("<|im_end|>")]
    u_ws = C[[tid(w) for w in [" water", " run", " blue"]]].mean(0) - C[[tid(w) for w in ["water", "run", "blue"]]].mean(0)
    unit(5, {OFF[0]: 1.0}, 0.0, {IDt: u_ws})                           # piece 0 always gets a leading space
    unit(5, {OFF[1]: 1.0, GOT1: -1.0}, 0.0, {IDt: dotc})
    unit(5, {OFF[2]: 1.0, GOT1: 1.0, GOT2: -1.0}, -1.0, {IDt: dotc}); unit(5, {OFF[2]: 1.0, GOT1: -1.0}, 0.0, {IDt: endc, OUTEND: 1.0})
    unit(5, {OFF[3]: 1.0, GOT2: 1.0}, -1.0, {IDt: dotc}); unit(5, {OFF[3]: 1.0, GOT2: -1.0}, 0.0, {IDt: endc, OUTEND: 1.0})
    groups.append({"name": "L5", "tensors": ["L5.*"], "mechanism": "h0 (L2 rows only) picks the content token: in the selected sentence, content word-start, not in the question, similar to the type word, after the overlap; MLP5 (L4 rows) writes template tokens, '.', <|im_end|>", "why": "content and surface form"})

    # ---------------- heads, codes, decoder ----------------
    for a in range(NA): P["head0"][ACTS_[a], a] = 1.0
    P["head1"][ID, :] = C.T; P["head3"][ID, :] = C.T; P["head_b1"][:] = -0.5 * (C * C).sum(1); P["head_b3"][:] = -0.5 * (C * C).sum(1)
    P["head3"][OUTEND, END] = 30.0                                   # nearest-code (L2) decoding; explicit end channel
    for a in range(NA):
        for j in range(16):
            P["head2"][ACTZ[a], a * 16 + j] = 1.0; P["head2"][J[j], a * 16 + j] = 1.0
    for a, nm in enumerate(ACTS):
        P["code0"][a, D_ACT[a]] = 1.0; P["code0"][a, D_TOK] = 0.05 * C[TOKS[nm][0]]
        La = len(TOKS[nm])
        for j in range(16):
            pi = 4 * j
            if pi < La: P["code2"][a * 16 + j, D_TOK] = 0.2 * C[TOKS[nm][pi]]
            elif nm in COPY and pi < La + 3: pass
            else: P["code2"][a * 16 + j, D_END] = 0.2
    P["code1"][:, D_CONT] = C[:, 4:36]; P["code1"][:, D_TOK] = 0.1 * C
    P["code3"][:, D_TOK] = C; P["code3"][END, D_END] = 1.0
    P["dec_w2"][D_TOK, :] = 10 * Cn.T; P["dec_w2"][D_END, END] = 300.0
    groups.append({"name": "codes", "tensors": ["head0-3", "code0-3", "dec_w2"], "mechanism": "act one-hot; vocab codes are the centred 96-d layer-0 token codes (cosine decoding); skeleton codes = template token at each 4-block; drafts weighted 0.05/0.1/0.2/1 so each level's guess overrides the previous one", "why": "decoding the latent after any level gives a readable reply draft"})
    return cfg, P, groups
