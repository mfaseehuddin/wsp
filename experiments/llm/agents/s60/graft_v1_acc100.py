"""s60 graft: explicit adder circuit written into 12 inserted layers at the embedding level. See EXPLANATION.md."""
import numpy as np, torch
from wsp.llm import surgery

C = 4096.0            # constant written on dim c0 inside the stack (keeps RMSNorm scale fixed); removed again at the end
D = 896
R0 = C / np.sqrt(D)   # nominal rms of the residual inside the stack
K = 30.0              # neuron sharpness
NF = 18               # RoPE pairs used for position tests
AMP = 20.0            # attention score amplitude per RoPE pair (peak margin ~0.815*AMP ~ 16)
KAP = 12.65           # key amplitude; query amplitude = 8*AMP/KAP
TH = 1e6 ** (-np.arange(32) / 32)
GBOOST = 60.0         # target logit boost for the answer digit
RMSF = 12.0           # assumed rms of final residual (tuned after measurement)
NL = 13


class Chans:
    def __init__(self, dims):
        self.dims, self.m = list(dims), {}

    def __call__(self, name):
        if name not in self.m:
            self.m[name] = self.dims[len(self.m)]
        return self.m[name]


def phase(t, amp=None):
    a = 8 * AMP / KAP if amp is None else amp
    v = np.zeros(64)
    v[:NF] = a * np.cos(t * TH[:NF])
    v[32:32 + NF] = -a * np.sin(t * TH[:NF])
    return v


class Layer:
    """Collects an attention half and an MLP half for one inserted layer."""

    def __init__(self):
        self.Wq = np.zeros((21 * 64, D)); self.bq = np.zeros(21 * 64)
        self.Wk = np.zeros((3 * 64, D)); self.bk = np.zeros(3 * 64)
        self.Wv = np.zeros((3 * 64, D)); self.bv = np.zeros(3 * 64)
        self.Wo = np.zeros((D, 21 * 64))
        self.neurons = []   # (inputs{dim:coef}, bias, out{dim:coef})
        self.attn_used = False

    def keys(self):
        for g in range(3):
            self.bk[g * 64:g * 64 + NF] = KAP
        self.attn_used = True

    def vdim(self, g, dim, coefs, bias=0.0):
        for c, a in coefs.items():
            self.Wv[g * 64 + dim, c] += a * R0
        self.bv[g * 64 + dim] += bias

    def head_const(self, h, delta):
        self.bq[h * 64:(h + 1) * 64] += phase(delta)

    def head_onehot(self, h, pairs):  # pairs: [(channel dim, delta)]
        for c, dl in pairs:
            self.Wq[h * 64:(h + 1) * 64, c] += phase(dl) * R0

    def head_out(self, h, vdim, ch, coef=1.0):
        self.Wo[ch, h * 64 + vdim] += coef

    def neuron(self, inputs, bias, out):
        self.neurons.append((inputs, bias, out))

    def apply(self, layer, c0=None, Eeq=None, calib=True):
        with torch.no_grad():
            at = layer.self_attn
            f = lambda a: torch.tensor(a, dtype=torch.float32)
            if self.attn_used:
                at.q_proj.weight.copy_(f(self.Wq)); at.q_proj.bias.copy_(f(self.bq))
                at.k_proj.weight.copy_(f(self.Wk)); at.k_proj.bias.copy_(f(self.bk))
                at.v_proj.weight.copy_(f(self.Wv)); at.v_proj.bias.copy_(f(self.bv))
                at.o_proj.weight.copy_(f(self.Wo))
            n = len(self.neurons)
            if n:
                G = np.zeros((n, D)); U = np.zeros((n, D)); Dn = np.zeros((D, n))
                for i, (inp, b, out) in enumerate(self.neurons):
                    for c, a in inp.items():
                        G[i, c] += K * a * R0
                    if calib:   # remove the '=' token's own embedding baseline from the scratch dims it reads
                        b = b - sum(a * Eeq[c] for c, a in inp.items())
                    G[i, c0] += K * b * R0 / C
                    U[i, c0] = R0 / (K * C)
                    for c, a in out.items():
                        Dn[c, i] += a
                m = layer.mlp
                m.gate_proj.weight[:n].copy_(f(G)); m.up_proj.weight[:n].copy_(f(U))
                m.down_proj.weight[:, :n].copy_(f(Dn))


def negneurons(ns):
    return [(i, b, {c: -a for c, a in o.items()}) for i, b, o in ns]


def copy_attn(dst, src, sign=-1.0):
    for k_ in ("Wq", "bq", "Wk", "bk", "Wv", "bv"):
        setattr(dst, k_, getattr(src, k_).copy())
    dst.Wo = src.Wo.copy() * sign
    dst.attn_used = src.attn_used


def graft(model, tok):
    for i in range(NL):
        surgery.insert_layer(model, i)
    surgery.add_head_group(model)
    E = model.model.embed_tokens.weight.detach().double().numpy()
    N = E.shape[0]
    ids = surgery.digit_ids(tok)
    plus, eq, nl = tok.convert_tokens_to_ids("+"), tok.convert_tokens_to_ids("="), 198
    var = np.abs(E[ids + [plus, eq, nl]]).max(0) + 0.1 * E.std(0)   # dims where the 13 relevant tokens have ~0 embedding
    order = np.argsort(var)
    c0 = int(order[0])
    ch = Chans(order[1:])
    ids = surgery.digit_ids(tok)
    plus, eq, nl = tok.convert_tokens_to_ids("+"), tok.convert_tokens_to_ids("="), 198

    # ---- channel names
    P = lambda *a: "_".join(str(x) for x in a)
    d_ = range(1, 14)

    # ---- token classifiers (LDA direction per class, closed form ridge solve)
    mask = np.ones(D, bool); mask[c0] = False
    for nm in ("valD", "isD", "isP", "isE"):
        mask[ch(nm)] = False      # classifier must not read the channels it writes (mirror re-reads them)
    Em = E[:, mask]
    S = Em.T @ Em / N
    S += 1e-3 * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    sp = Em[ids + [plus, eq, nl]]
    S += 1.0 * sp.T @ sp          # the 13 relevant tokens are constrained (score 0) for every other class's detector
    Si = np.linalg.inv(S)

    def cls_vec(tid):
        e = Em[tid]
        u = Si @ e
        u /= e @ u
        full = np.zeros(D); full[mask] = u
        return full

    layers = [Layer() for _ in range(NL)]
    L0, L1, L2, L3, L4, L5, L6, L7, L8, R1, R2, R3, R4 = layers

    # ---- L0: attention writes +C on c0 (value bias, softmax weights sum to 1); MLP classifies the token
    L0.attn_used = True
    L0.bv[0] = C; L0.Wo[c0, 0] = 1.0       # head 0, value dim 0 -> c0
    cls = {}
    for d in range(10):
        cls[("d", d)] = (cls_vec(ids[d]), {ch("valD"): 2.0 * (d + 1), ch("isD"): 2.0})
    cls["+"] = (cls_vec(plus), {ch("isP"): 2.0})
    cls["="] = (cls_vec(eq), {ch("isE"): 2.0})
    for key, (u, out) in cls.items():
        L0.neuron({j: u[j] for j in np.nonzero(u)[0]}, -0.5, out)

    # ---- common value layout for the gather layers (same on all 3 kv groups)
    def std_v(L):
        L.keys()
        for g in range(3):
            L.vdim(g, 0, {ch("isP"): 1.0})
            L.vdim(g, 1, {ch("isD"): -1.0}, bias=1.0)       # nondigit
            L.vdim(g, 2, {ch("valD"): 1.0})
            for d in d_:
                L.vdim(g, 2 + d, {ch(P("LAo", d)): 1.0})     # la one-hot (dims 3..15)
            L.vdim(g, 16, {ch("isE"): 1.0})

    # ---- L1: banks at offsets d (is '+', is non-digit, is '=') + b-gather k=0..6
    std_v(L1)
    for d in range(0, 14):
        L1.head_const(d, d)
        if d >= 1:
            L1.head_out(d, 0, ch(P("hitP", d))); L1.head_out(d, 1, ch(P("nd", d)))
        L1.head_out(d, 16, ch(P("hitE", d)))
    for k in range(7):
        h = 14 + k
        L1.head_const(h, k + 1); L1.head_out(h, 2, ch(P("GB", k)))
    # MLP: first-hit one-hots: lb (offset of '+' from '='), la (first non-digit left of '+'), i (nearest '=')
    for d in d_:
        L1.neuron({**{ch(P("hitP", d)): 1.0}, **{ch(P("hitP", e)): -1.0 for e in range(1, d)}}, 0.0, {ch(P("lbo", d)): 1.0})
        L1.neuron({**{ch(P("nd", d)): 1.0}, **{ch(P("nd", e)): -1.0 for e in range(1, d)}}, 0.0, {ch(P("LAo", d)): 1.0})
    for i in range(0, 14):
        L1.neuron({**{ch(P("hitE", i)): 1.0}, **{ch(P("hitE", e)): -1.0 for e in range(0, i)}}, 0.0, {ch(P("iO", i)): 1.0})

    # ---- L2: b-gather k=7..11, a-gather k=0..11 (offset lb+2+k), la transfer from '+' to '='
    std_v(L2)
    for k in range(7, 12):
        h = k - 7; L2.head_const(h, k + 1); L2.head_out(h, 2, ch(P("GB", k)))
    for k in range(12):
        h = 5 + k
        L2.head_onehot(h, [(ch(P("lbo", d)), d + 1 + k) for d in d_]); L2.head_out(h, 2, ch(P("GA", k)))
    h = 17
    L2.head_onehot(h, [(ch(P("lbo", d)), d) for d in d_])
    for d in d_:
        L2.head_out(h, 2 + d, ch(P("LAt", d)))
    big = 20.0
    def clean(L, src, dst):   # saturating step: 0 below 0.5, exactly 1 above ~0.6
        L.neuron({src: 10.0}, -5.0, {dst: 1.0}); L.neuron({src: 10.0}, -6.0, {dst: -1.0})
    for d in d_:
        clean(L2, ch(P("lbo", d)), ch(P("lbc", d))); clean(L2, ch(P("LAt", d)), ch(P("LAc", d)))
    clean(L2, ch("isE"), ch("iEc"))
    lbT = lambda k: {ch(P("lbc", d)): 1.0 for d in range(k + 2, 14)}
    laT = lambda k: {ch(P("LAc", d)): 1.0 for d in range(k + 2, 14)}
    add = lambda a, b, s=1.0: {**a, **{k_: a.get(k_, 0) + s * v for k_, v in b.items()}}
    for k in range(12):
        inp = {ch(P("GB", k)): 1.0, ch("iEc"): big}; inp.update({c: big for c in lbT(k)})
        L3.neuron(inp, -1 - 2 * big, {ch(P("BE", k)): 1.0})
        inp = {ch(P("GA", k)): 1.0, ch("iEc"): big}; inp.update({c: big for c in laT(k)})
        L3.neuron(inp, -1 - 2 * big, {ch(P("AE", k)): 1.0})
    L3.neuron({ch(P("GB", 0)): 1.0}, 0.0, {ch("IB"): 1.0}); L3.neuron({ch(P("GB", 0)): 1.0}, -1.0, {ch("IB"): -1.0})
    L3.neuron({ch(P("GA", 0)): 1.0}, 0.0, {ch("IA"): 1.0}); L3.neuron({ch(P("GA", 0)): 1.0}, -1.0, {ch("IA"): -1.0})

    # ---- L4: G/P indicators, existence excess, trigger
    for k in range(12):
        g = {ch(P("BE", k)): 1.0, ch(P("AE", k)): 1.0}
        for j, out in ((8, {ch(P("P", k)): 1.0}), (9, {ch(P("G", k)): 1.0, ch(P("P", k)): -2.0}),
                       (10, {ch(P("G", k)): -1.0, ch(P("P", k)): 1.0})):
            L4.neuron(g, -float(j), out)
        L4.neuron(add(laT(k), lbT(k), -1.0), 0.0, {ch(P("EX", k)): 1.0})
    trig = {ch("IB"): 1, ch("IA"): 1, ch("iEc"): 1}
    trig.update({ch(P("LAc", d)): 1 for d in range(2, 14)})
    trig.update({ch(P("lbc", d)): 1 for d in d_})
    L4.neuron(trig, -4.0, {ch("TRIG"): 1.0})

    # ---- L5: carries c_k = OR_j [G_j and P_{j+1..k-1}]
    for k in range(1, 13):
        for j in range(k):
            inp = {ch(P("G", j)): 1.0}; inp.update({ch(P("P", m)): 1.0 for m in range(j + 1, k)})
            L5.neuron(inp, -float(k - j - 1), {ch(P("C", k)): 1.0})

    # ---- L6: sum-digit one-hots and answer length one-hot
    for k in range(13):
        t = {}
        if k < 12:
            t = {ch(P("BE", k)): 1.0, ch(P("AE", k)): 1.0}
        if k >= 1:
            t[ch(P("C", k))] = 1.0
        for j in range(-1, 22):
            out = {}
            for x in range(10):
                for off, co in ((-1, 1), (0, -2), (1, 1), (9, 1), (10, -2), (11, 1)):
                    if j == x + off:
                        out[ch(P("S", k, x))] = out.get(ch(P("S", k, x)), 0) + co
            if out:
                L6.neuron(t, -float(j), out)
    Ek = lambda k: add(lbT(k), {ch(P("EX", k)): 1.0}) if k < 12 else {}
    for v in range(1, 14):
        if v - 1 <= 11:
            inp = add(add(Ek(v - 1), Ek(v), -1.0), {ch("TRIG"): 1.0}); 
            if v <= 12: inp[ch(P("C", v))] = inp.get(ch(P("C", v)), 0) - 1.0
            L6.neuron(inp, -1.0, {ch(P("N", v)): 1.0})
        if v >= 2:
            inp = add(add(Ek(v - 2), Ek(v - 1), -1.0), {ch("TRIG"): 1.0})
            inp[ch(P("C", v - 1))] = inp.get(ch(P("C", v - 1)), 0) + 1.0
            L6.neuron(inp, -2.0, {ch(P("N", v)): 1.0})

    # ---- L7: A_{i,x} = s_{n-1-i, x} AND n (MSB-first answer digit i); A_{n,EOS} = n
    for v in range(1, 14):
        for k in range(v):
            i = v - 1 - k
            for x in range(10):
                L7.neuron({ch(P("S", k, x)): 1.0, ch(P("N", v)): 1.0}, -1.0, {ch(P("A", i, x)): 1.0})
        L7.neuron({ch(P("N", v)): 1.0}, 0.0, {ch(P("A", v, 10)): 1.0})

    # ---- L8: answer read: head i attends offset i, reads block A_i -> ANSi channels; gated by nearest '=' one-hot
    L8.keys()
    grp = {0: list(range(0, 5)), 1: list(range(5, 10)), 2: list(range(10, 14))}
    for g, iis in grp.items():
        for jj, i in enumerate(iis):
            h = g * 7 + jj
            for x in range(11):
                L8.vdim(g, jj * 11 + x, {ch(P("A", i, x)): 1.0})
                L8.head_out(h, jj * 11 + x, ch(P("ANSi", i, x)))
            L8.head_const(h, i)
    for i in range(14):
        for x in range(11):
            L8.neuron({ch(P("ANSi", i, x)): 1.0, ch(P("iO", i)): 1.0}, -1.5, {ch(P("ANS", x)): 2.0})

    # ---- mirrors (un-compute): same weights with negated writes
    copy_attn(R1, L8); R1.neurons = negneurons(L7.neurons + L6.neurons + L5.neurons + L4.neurons + L3.neurons + L2.neurons)
    copy_attn(R2, L2); R2.neurons = negneurons(L1.neurons)
    copy_attn(R3, L1); R3.neurons = negneurons(L0.neurons)
    R4.attn_used = True; R4.bv[0] = -C; R4.Wo[c0, 0] = 1.0

    # ---- conversion: ANS_x -> boost vector U_x (written after all mirrors), erases ANS_x
    fw = model.model.norm.weight.detach().double().numpy()
    toks = ids + [nl]
    Gm = np.stack([fw * E[t] for t in toks])               # logit_j = (x/rms) . Gm_j
    Gp = np.linalg.pinv(Gm)                                 # D x 11
    conv = []
    for x, t in enumerate(toks):
        U = Gp[:, x] * GBOOST * RMSF
        conv.append(U)
        out = {ch(P("ANS", x)): -2.0}
        out.update({j: 2.0 * U[j] for j in range(D)})
        R3.neurons.append(({ch(P("ANS", x)): 1.0}, -0.5, out))
    graft.Unorm = float(np.mean([np.linalg.norm(u) for u in conv]))

    # ---- write weights
    for i, L in enumerate(layers):
        L.apply(model.model.layers[i], c0=c0, Eeq=E[eq], calib=(i != 0 and i != 11))
    assert len(ch.m) < len(ch.dims), len(ch.m)
    graft.nch = len(ch.m)
    import json; json.dump({'c0': c0, 'ch': {k_: int(v) for k_, v in ch.m.items()}, 'Unorm': graft.Unorm, 'fw': float(np.abs(fw).mean())}, open('experiments/llm/agents/s60/chmap.json', 'w'))
    return [{"name": "stack", "tensors": ["model.layers.0-11"], "mechanism": "see EXPLANATION.md", "why": "see EXPLANATION.md"}]
