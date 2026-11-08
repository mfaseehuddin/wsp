"""o60 graft: an exact adder written into 9 new layers inserted before original layer 0.

See EXPLANATION.md. Units: every scratch quantity q lives on its own unit direction u_q of a basis B
that is orthogonal to the embeddings of the 12 "relevant" tokens (0-9,'+','=') and to axis 221, and is
stored with scale sigma_q (residual += sigma_q * value * u_q). A big constant C*u0 (u0 orthogonal to the
relevant embeddings) is added first so every RMSNorm divides by ~C/sqrt(896): readings become linear.
"""
import math
import os
import torch
from wsp.llm import surgery

C = 1000.0          # normalising constant (removed again in the last inserted layer)
G = 200.0           # gate gain: relu(y) ~= silu(G*y)/G
LAM = 1.0e4         # park rms
NLAY = 10
JS = 221            # residual axis kept free of scratch (carries most of C)


class Builder:
    def __init__(self, model, tok):
        self.m = model
        self.tok = tok
        E = model.model.embed_tokens.weight.detach()
        self.E = E
        self.dig = surgery.digit_ids(tok)
        self.PLUS = tok.convert_tokens_to_ids("+")
        self.EQ = tok.convert_tokens_to_ids("=")
        self.NL = tok("\n")["input_ids"][0]
        self.R = self.dig + [self.PLUS, self.EQ]
        Er = E[self.R].double()
        d = E.shape[1]
        Qr, _ = torch.linalg.qr(Er.T)                       # span of relevant embeddings
        e = torch.zeros(d, dtype=torch.float64); e[JS] = 1
        self.u0 = e.clone()   # constant direction = axis JS exactly (exact in float32; nothing else reads/writes JS)
        Q, _ = torch.linalg.qr(torch.cat([Er.T, e[:, None]], 1))
        P = torch.eye(d, dtype=torch.float64) - Q @ Q.T
        cov = torch.zeros(d, d, dtype=torch.float64)
        for i in range(0, E.shape[0], 16384):
            b = E[i:i + 16384].double(); cov += b.T @ b
        ev, V = torch.linalg.eigh(P @ cov @ P)
        keep = (ev > ev.max() * 1e-9).nonzero().squeeze(1)  # drop the 13-dim null space
        self.B = V[:, keep[:320]].clone()                    # lowest-variance complement directions
        self.B[JS] = 0.0                                     # exactly blind to the constant channel
        self.cont = (E @ self.B.float()).abs().max(0).values.double()  # worst-case |<u_q, e_x>|
        self.rmsC = C / math.sqrt(d)
        self.names = {}
        self.sig = {}
        order = self.cont[:40].argsort().tolist()
        for nm, j in zip(["P", "DIG", "E", "VAL"], order[:4]):
            self.names[nm] = j; self.sig[nm] = 1.0
        self.free = [j for j in range(self.B.shape[1]) if j not in order[:4]]
        self.L = model.model.layers
        self.neur = {i: 0 for i in range(NLAY)}
        self.groups = []
        self.log = {}
        self.gw = {}
        self.labels = {}

    # ---- quantities -------------------------------------------------------------------------
    def q(self, name, sigma):
        if name not in self.names:
            self.names[name] = self.free.pop(0); self.sig[name] = sigma
        return name

    def read(self, spec):
        v = torch.zeros(self.B.shape[0], dtype=torch.float64)
        for k, c in spec.items():
            if k == "one":
                v += c * self.u0 * self.rmsC / C
            else:
                v += c * self.B[:, self.names[k]] * self.rmsC / self.sig[k]
        return v

    def write(self, spec):
        v = torch.zeros(self.B.shape[0], dtype=torch.float64)
        for k, c in spec.items():
            v += c * self.B[:, self.names[k]] * self.sig[k]
        return v

    def gate_vec(self, toks, M):
        """M*(<w,e_x>-1): exactly 0 on the target tokens (any layer, any rms), <= -M*margin elsewhere.
        w = min-norm vector in span(relevant embeddings) with <w,e_r> = target(r)."""
        key = "".join(sorted(toks))
        if key not in self.gw:
            tv = [1.0 if ((r in self.dig and "DIG" in toks) or (r == self.PLUS and "P" in toks)
                          or (r == self.EQ and "E" in toks)) else 0.0 for r in self.R]
            Er = self.E[self.R].double().clone(); e21 = Er[:, JS].clone(); Er[:, JS] = 0.0
            tgt = torch.tensor(tv, dtype=torch.float64) * (1.0 + e21 / C)   # cancels the 'one' read exactly
            w = Er.T @ torch.linalg.solve(Er @ Er.T, tgt)
            sc = (self.E @ w.float()).double() - 1.0; sc[self.R] = -9
            self.gw[key] = (w, -sc.max().item())
        w, marg = self.gw[key]
        M = M / marg
        self.maxM = max(getattr(self, "maxM", 0.0), M)
        return M * w * self.rmsC - self.read({"one": M})

    def garbage(self, spec):
        return sum(abs(c) * self.cont[self.names[k]].item() / self.sig[k] for k, c in spec.items() if k != "one")

    # ---- MLP neurons --------------------------------------------------------------------------
    def neuron(self, li, gate, up, down, label=""):
        n = self.neur[li]; self.neur[li] += 1
        self.labels.setdefault(li, []).append(label)
        self.log.setdefault(li, []).append((gate, up, down))
        mlp = self.L[li].mlp
        with torch.no_grad():
            mlp.gate_proj.weight[n] = gate.float()
            mlp.up_proj.weight[n] = up.float()
            mlp.down_proj.weight[:, n] = down.float()
        return n

    def relu(self, li, y, out, gate_tokens=None, gain=G):
        """neuron = relu(y) (y a read-spec), written as out (write-spec). gate_tokens: list of flags; the
        neuron is forced off unless the current token is one of them (M_tok*(sum flags - 1))."""
        g = self.read(y)
        if gate_tokens:
            g = g + self.gate_vec(gate_tokens, 2.0 * self.garbage(y) + 50.0)
        lab = ",".join(f"{k}" for k in out) + "|" + ",".join(f"{k}:{c:g}" for k, c in y.items())
        return self.neuron(li, gain * g, self.read({"one": 1.0 / gain}), self.write(out), lab[:60])

    def ramp(self, li, y, lo, hi, out, gate_tokens=None):
        """clamp((y-lo)/(hi-lo),0,1) * out, two neurons."""
        s = 1.0 / (hi - lo)
        a = self.relu(li, {**y, "one": y.get("one", 0) - lo}, {k: s * c for k, c in out.items()}, gate_tokens)
        b = self.relu(li, {**y, "one": y.get("one", 0) - hi}, {k: -s * c for k, c in out.items()}, gate_tokens)
        return a, b

    def step(self, li, y, n, out, gate_tokens=None):
        """[y >= n] for integer-valued y."""
        return self.ramp(li, y, n - 0.75, n - 0.25, out, gate_tokens)

    # ---- attention ----------------------------------------------------------------------------
    def set_q(self, li, h, bias):
        with torch.no_grad():
            self.L[li].self_attn.q_proj.bias[h * 64:(h + 1) * 64] = bias.float()

    def set_kv(self, li, j, kbias, krows=None, vrows=None, vbias=None):
        a = self.L[li].self_attn
        with torch.no_grad():
            a.k_proj.bias[j * 64:(j + 1) * 64] = kbias.float()
            for r, vec in (krows or {}).items():
                a.k_proj.weight[j * 64 + r] = vec.float()
            for r, vec in (vrows or {}).items():
                a.v_proj.weight[j * 64 + r] = vec.float()
            if vbias is not None:
                a.v_proj.bias[j * 64:(j + 1) * 64] = vbias.float()

    def set_o(self, li, h, dim, vec):
        with torch.no_grad():
            self.L[li].self_attn.o_proj.weight[:, h * 64 + dim] = vec.float()


def omega(i):
    return 1e6 ** (-i / 32)


def graft(model, tok):
    for i in range(NLAY):
        surgery.insert_layer(model, i)
    b = Builder(model, tok)
    Bq = b.q
    Z64 = lambda: torch.zeros(64, dtype=torch.float64)
    nd = {}  # neuron ranges per group for provenance
    def mark(li, key):
        nd.setdefault(key, [li, b.neur[li], None])
    def done(key):
        nd[key][2] = b.neur[nd[key][0]]

    # ============ N0: constant C, token flags ============
    vb = Z64(); vb[0] = 1.0
    b.set_kv(0, 0, Z64(), vbias=vb)
    b.set_o(0, 0, 0, C * b.u0)                       # every position: += C*u0 exactly
    E = b.E
    det = []
    for r in b.R:
        er = E[r].double().clone(); er[JS] = 0.0
        s = (E @ er.float()).double(); S = s[r].item(); s[r] = -1e9; O = s.max().item()
        out = {}
        if r in b.dig:
            out = {"DIG": 1.0, "VAL": float(b.dig.index(r))}
        elif r == b.PLUS:
            out = {"P": 1.0}
        else:
            out = {"E": 1.0}
        det.append((er, S, O, out))
    def detect(li, sign):
        for er, S, O, out in det:
            for lo, sg in ((0.25, 2.0), (0.75, -2.0)):
                g = G * (er * b.rmsC - (O + lo * (S - O)) * b.read({"one": 1.0})) / (S - O)
                b.neuron(li, g, b.read({"one": 1.0 / G}), b.write({k: sign * sg * c for k, c in out.items()}))
    mark(0, "flags"); detect(0, 1.0); done("flags")

    # ============ N1: offset fetch heads d=1..13, zero-shot head Z ============
    A, cr = 40.0, 480.0
    kb = Z64()
    for i in list(range(6)) + [15]:
        kb[i] = 1.0
    kb[31] = 30.0
    krows = {31: -30.0 * b.read({"DIG": 1, "P": 1})}
    vrows = {0: b.read({"DIG": 1}), 1: b.read({"VAL": 1}), 2: b.read({"E": 1}), 3: b.read({"P": 1})}
    for j in (0, 1):
        b.set_kv(1, j, kb, krows, vrows)
    for d in range(1, 14):
        h = d - 1
        qb = Z64()
        for i in range(6):
            qb[i] = 8 * A * math.cos(d * omega(i)); qb[i + 32] = -8 * A * math.sin(d * omega(i))
        qb[15] = 8 * cr
        b.set_q(1, h, qb)
        for k, nm in enumerate(["FD", "FV", "FE", "FP"]):
            b.set_o(1, h, k, b.write({Bq(f"{nm}{d}", 3e-3): 1.0}))
    qz = Z64(); qz[31] = 8.0
    b.set_q(1, 13, qz)
    b.set_o(1, 13, 3, b.write({Bq("ZO", 0.1): 1.0}))

    mark(1, "anchor")
    for p in range(12):
        sumd = {f"FD{d}": 1.0 for d in range(1, p + 2)}
        b.step(1, sumd, p + 1, {Bq(f"VLD{p}", 1e-2): 1.0}, ["P", "E"])
        thr = 0.5 * (1 / (p + 1) + 1 / (p + 2)); hg = 0.5 * (1 / (p + 1) - 1 / (p + 2))
        b.ramp(1, {"ZO": 1 / hg, "one": -thr / hg}, -0.5, 0.5, {Bq(f"ZS{p}", 1e-2): 1.0}, ["P"])
    for d in range(2, 14):
        b.step(1, {f"FP{d}": 1.0, **{f"FD{x}": 1.0 for x in range(1, d)}}, d, {Bq("PB", 1e-2): 1.0}, ["E"])
    done("anchor"); mark(1, "kpos")
    for d in range(1, 14):
        b.step(1, {f"FE{d}": 1.0, **{f"FD{x}": 1.0 for x in range(1, d)}}, d,
               {Bq("R", 1e-2): 1.0, Bq("K", 1e-2): float(d)}, ["DIG"])
    done("kpos")

    # ============ N2: zero-shot masking of A at '+' ============
    mark(2, "zsmask")
    for p in range(12):
        b.step(2, {f"VLD{p}": 1.0, f"ZS{p}": -1.0}, 1, {Bq(f"AVm{p}", 0.1): 1.0}, ["P"])
        b.relu(2, {f"FV{p+1}": 1.0, f"VLD{p}": 12.0, f"ZS{p}": -12.0, "one": -12.0},
               {Bq(f"MM{p}", 0.1): 1.0}, ["P", "E"])
    done("zsmask")

    # ============ recency copy head helper ============
    Bk, Sr = 5000.0, 3250.0
    def copy_head(li, flag, srcs, dsts):
        kb = Z64(); kb[15] = 1.0
        vrows = {i: b.read(s) for i, s in enumerate(srcs)}
        b.set_kv(li, 0, kb, {31: Bk * b.read({flag: 1.0})}, vrows)
        qb = Z64(); qb[31] = 8.0; qb[15 + 32] = 8.0 * Sr
        b.set_q(li, 0, qb)
        for i, dst in enumerate(dsts):
            b.set_o(li, 0, i, b.write(dst))

    # ============ N3: copy A from '+' to '=', generate/kill, validity, length ============
    copy_head(3, "P", [{f"AVm{p}": 1.0} for p in range(12)] + [{f"MM{p}": 1.0} for p in range(12)],
              [{Bq(f"AV{p}", 1e-3): 1.0} for p in range(12)] + [{Bq(f"AM{p}", 1e-3): 1.0} for p in range(12)])
    for p in range(1, 13):
        Bq(f"CP{p}", 1e-4)
    mark(3, "genkill")
    for j in range(12):
        x = {f"AM{j}": 1.0, f"MM{j}": 1.0}
        b.step(3, x, 10, {f"CP{p}": 2.0 ** j for p in range(j + 1, 13)}, ["E"])
        b.step(3, {k: -v for k, v in x.items()}, -8, {f"CP{p}": -(2.0 ** j) for p in range(j + 1, 13)}, ["E"])
    b.step(3, {"PB": 1.0, "AV0": 1.0}, 2, {Bq("V", 0.1): 1.0}, ["E"])
    for p in range(1, 13):
        b.step(3, {f"AV{p-1}": 1.0, f"VLD{p-1}": 1.0}, 1, {Bq(f"GL{p}", 0.1): 1.0}, ["E"])
    done("genkill")

    # ============ N4: carries (lookahead), top carry ============
    M2 = 1.0e4
    for p in range(13):
        Bq(f"SC{p}", 1e-3)
    mark(4, "carry")
    for p in range(1, 13):
        b.step(4, {f"CP{p}": 1.0}, 1, {f"SC{p}": 1.0, f"SC{p-1}": -10.0, Bq(f"CC{p}", 0.1): 1.0}, ["E"])
    done("carry")

    # ============ N5: copy sum digits from '=' to answer positions; place index ============
    s_spec = [({f"AM{p}": 1.0, f"MM{p}": 1.0, f"SC{p}": 1.0} if p < 12 else {"SC12": 1.0}) for p in range(13)]
    mark(5, "export")
    for p in range(13):
        b.relu(5, {**s_spec[p], "V": 30.0, "one": -30.0}, {Bq(f"XS{p}", 0.1): 1.0}, ["E"])
    b.relu(5, {**{f"GL{p}": 1.0 for p in range(1, 13)}, "V": 30.0, "one": -30.0}, {Bq("XN", 0.1): 1.0}, ["E"])
    for p in range(1, 13):   # top carry: c_p at p == L adds one digit
        y = {f"CC{p}": 1.0, f"GL{p}": 1.0, "V": 1.0}
        if p < 12:
            y[f"GL{p+1}"] = -1.0
        b.step(5, y, 3, {"XN": 1.0}, ["E"])
    b.step(5, {"V": 1.0}, 1, {Bq("XV", 0.1): 1.0}, ["E"])
    done("export")
    copy_head(6, "E", [{f"XS{p}": 1.0} for p in range(13)] + [{"XN": 1.0}, {"XV": 1.0}],
              [{Bq(f"QS{p}", 1e-3): 1.0} for p in range(13)] + [{Bq("QN", 1e-3): 1.0}, {Bq("QV", 1e-2): 1.0}])
    M3 = 20.0
    mark(6, "place")
    for p in range(-1, 13):
        y = {"QN": 1.0, "K": -1.0, "one": -1.0 - p - 2 * M3, "R": M3, "E": M3, "QV": M3}
        b.step(6, y, 0, {Bq(f"F{p}", 1e-2): 1.0}, ["DIG", "E"])
    done("place")

    # ============ N6: select the digit of the current place ============
    Bq("Y", 1e-2)
    mark(7, "select")
    for p in range(13):
        y = {f"F{p}": 1.0, "one": -0.5}
        if p < 12:
            y[f"F{p+1}"] = -1.0
        gsp = b.read(y)
        M = 2.0 * b.garbage(y) + 50.0
        gsp = gsp + b.gate_vec(["DIG", "E"], M)
        b.neuron(7, G * gsp, b.read({f"QS{p}": 2.0 / G}), b.write({"Y": 1.0}))
    done("select")

    # ============ N7: park the answer digit (huge, dominates every later layer), cancel flags ============
    with torch.no_grad():
        Ef = b.E
        gf = model.model.norm.weight.detach().double()
    park = {}
    tgt = b.dig + [b.NL]
    for t in tgt:
        Mt = Ef[t].double() * gf
        comp = [x for x in tgt if x != t]
        for _ in range(2):
            Cm = (Ef[comp].double() * gf).T
            v = Mt - Cm @ torch.linalg.lstsq(Cm, Mt).solution
            lg = (Ef @ (v * gf).float()).double(); lg[t] = -1e9
            comp = list(dict.fromkeys(comp + lg.topk(20).indices.tolist()))
        park[t] = LAM * v / v.pow(2).mean().sqrt()
    P = lambda t: park[t]
    mark(8, "park")
    def pneur(y, outvec):
        M = 2.0 * b.garbage(y) + 50.0
        for lo, sg in ((0.75, 2.0), (0.25, -2.0)):
            g = b.read({**y, "one": y.get("one", 0) + lo}) + b.gate_vec(["DIG", "E"], M)
            b.neuron(8, G * g, b.read({"one": 1.0 / G}), sg * outvec)
    pneur({"F0": 1.0, "one": -1.0}, P(b.dig[0]))
    for d in range(1, 10):
        pneur({"Y": 1.0, "one": -float(d)}, P(b.dig[d]) - P(b.dig[d - 1]))
    pneur({"F-1": 1.0, "F0": -1.0, "one": -1.0}, P(b.NL))
    done("park"); mark(8, "unflag"); detect(8, -1.0)
    for li in (2, 5):   # cancel the 0.1-scale exports exactly (same neurons, negated outputs)
        for g_, u_, d_ in list(b.log[li]):
            b.neuron(8, g_, u_, -d_)
    done("unflag")

    # ============ N8: remove C ============
    b.set_kv(9, 0, Z64(), vbias=vb)
    b.set_o(9, 0, 0, -C * b.u0)

    def T(li, s):
        return f"model.layers.{li}.{s}"
    def nr(key):
        li, a, z = nd[key]
        return [T(li, f"mlp.gate_proj.weight[{a}:{z}]"), T(li, f"mlp.up_proj.weight[{a}:{z}]"),
                T(li, f"mlp.down_proj.weight[:, {a}:{z}]")]
    G_ = [
        ("normaliser C", [T(0, "self_attn.v_proj.bias[0]"), T(0, "self_attn.o_proj.weight[:, 0]"),
                          T(9, "self_attn.v_proj.bias[0]"), T(9, "self_attn.o_proj.weight[:, 0]")],
         "bias-only head adds exactly C*u0 at every position (layer 0) and removes it (layer 8)",
         "makes every RMSNorm in the inserted layers divide by ~C/sqrt(896), so reads are linear; u0 is orthogonal to the digit/+/= embeddings so the constant channel is exact there"),
        ("token flags", nr("flags"), "step detectors for 0-9,'+','=' -> DIG, VAL, P, E flags",
         "dot with the token's own embedding; threshold between self-similarity and the best other vocab token"),
        ("offset fetch", [T(1, "self_attn.q_proj.bias"), T(1, "self_attn.k_proj"), T(1, "self_attn.v_proj.weight[0:4]"),
                          T(1, "self_attn.v_proj.weight[64:68]"), T(1, "self_attn.o_proj.weight[:, 0:896]")],
         "head d (0..12) attends exactly to offset d+1 and copies its flags; head 13 measures 1/(la+1) for zero-shot",
         "constant q/k biases: score = 40*sum_{i<6} cos(w_i(D-d)) + 480 cos(w_15 D); margin >= 32 for D<=4000"),
        ("anchor logic", nr("anchor"), "digit-run validity, masked digits per place, '+' before B, zero-shot masks", "integer thresholds"),
        ("answer position", nr("kpos"), "k = distance to nearest '=' through digits", "integer thresholds"),
        ("zero-shot mask", nr("zsmask"), "mask A places beyond the true length at text start", "integer thresholds"),
        ("copy A", [T(3, "self_attn")], "'=' copies A's digits from the most recent '+'", "content key 5000*P + recency -3250 sin(w15 D)"),
        ("generate/kill", nr("genkill"), "g_j=[a_j+b_j>=10], k_j=[a_j+b_j<=8] summed into C_p=sum_{j<p}(g_j-k_j)2^j; V, [L>=p]",
         "sign of C_p = carry lookahead (highest non-9 place decides)"),
        ("carries", nr("carry"), "c_p=[C_p>=1], top carry -> sum digits s_p and answer length n", "s_p=a_p+b_p+c_p-10c_{p+1}"),
        ("export sums", nr("export"), "s_p, n, V re-written at 0.1 scale at valid '=' only", "so copy heads never amplify embedding contamination"),
        ("copy sums", [T(6, "self_attn")], "answer positions copy s_p, n, V from the most recent '='", "as copy A, key on E"),
        ("place index", nr("place"), "F_p=[n-1-k>=p]", "integer thresholds gated by R and V"),
        ("select", nr("select"), "Y = s_p of the current place", "SwiGLU product gate(onehot)*up(s_p)"),
        ("park", nr("park"), "writes LAM*v_digit (or v_newline after the last digit)",
         "v_t = M_t minus its projection on competitor tokens (pinv); dominates all later layers"),
        ("unflag", nr("unflag"), "cancels the token flags and the 0.1-scale exports exactly", "same detectors as layer 0, negated"),
    ]
    return [{"name": n, "tensors": t, "mechanism": m, "why": w} for n, t, m, w in G_]
