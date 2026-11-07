"""Hand-constructed weights for the IT-support bot (agent s50). See EXPLANATION.md.

One layer, two heads, one ReLU MLP, tied embeddings, d_model = 11.
Residual layout:
  ID   0..4  token identity code (output / <sep> detection), reply tokens only
  CODE 5..8  intent code (keyword tokens) / intent evidence written by head 0 (x A)
  STR  9     keyword strength (key of head 0)
  P    10    position t (from pos); head 1 subtracts the <sep> position -> step n
"""

import itertools

import torch

from wsp.support import data, tokenizer as tk
from wsp.support.model import Config

ID, CODE, STR, P = list(range(5)), list(range(5, 9)), 9, 10
T_KEY, MU = 120.0, 1.6  # head 0: score = T*strength - MU*position
LAM = 100.0  # head 1: <sep> detection sharpness
A = 2000.0  # head 0 output gain (keeps the token's own code negligible)
BETA = 50.0  # MLP intent gate: ramps shift by BETA*(1 - intent match)
S = 2e5  # MLP output gain onto identity codes

# Hand-written keyword lists: intent -> (strong, weak). Strength 1.0 / 0.5.
# Greetings/thanks are all weak so that 'hi, my vpn is down' resolves to the IT intent.
KW = {
    "password": (
        "password pwd forgot reset expired",
        "login logins log sign remember change",
    ),
    "locked": ("locked unlock unlocks blocked", "account accounts attempts logged"),
    "vpn": ("vpn globalprotect", "remote internal server sharepoint sites"),
    "wifi": ("wifi wi-fi wireless", "internet network dropping drops"),
    "email": (
        "outlook email emails mail inbox outbox",
        "send syncing arriving arrives calendar",
    ),
    "printer": ("printer print printing toner paper jammed", "queue"),
    "software": (
        "software install installing installed license program",
        "excel adobe photoshop python visual studio application download admin rights tool",
    ),
    "slow": (
        "slow sluggish freezing freezes frozen lagging hangs",
        "performance forever ages cpu fan boot crash crashes crashed",
    ),
    "mfa": (
        "mfa 2fa authenticator authentication factor",
        "code phone prompt push notification approve sso multi",
    ),
    "disk": ("disk storage space cleanup", "drive full onedrive free"),
    "monitor": (
        "monitor screen screens display hdmi projector",
        "external second signal flickers black blank cable docking station",
    ),
    "hello": ("", "hello hi hey morning afternoon evening anyone"),
    "thanks": ("", "thanks thank cheers awesome perfect great worked works fixed"),
    "unknown": ("", ""),
}
# Out-of-domain words: carry the 'unknown' code with strength 0.75.
OOD = "weather lunch joke pizza parking flight coffee vacation payday london ceo game book meeting room won floor"


def _codes(n, w):  # all {-1,0,1}^n vectors with exactly w nonzeros, unit-normalised
    out = []
    for supp in itertools.combinations(range(n), w):
        for sg in itertools.product((1, -1), repeat=w):
            v = torch.zeros(n, dtype=torch.float64)
            v[list(supp)] = torch.tensor(sg, dtype=torch.float64)
            out.append(v / w**0.5)
    return out


def build():
    V, L = tk.vocab_size(), tk.MAX_LEN
    cfg = Config(
        d_model=11,
        n_layers=1,
        n_heads=2,
        d_head=4,
        d_mlp=0,
        pos="learned",
        tie_embeddings=True,
        norm=False,
        bias=True,
    )
    replies = {k: tk.split(r) + ["<eos>"] for k, r in data.REPLIES.items()}
    d_mlp = sum(len(r) for r in replies.values())
    cfg = Config(**{**cfg.to_dict(), "d_mlp": d_mlp})
    d = cfg.d_model
    f64 = dict(dtype=torch.float64)

    # Identity codes (5-d, cos <= 0.82) for every token the bot can emit, plus <sep>.
    out_toks = ["<sep>"] + sorted({w for r in replies.values() for w in r})
    idc = dict(zip(out_toks, _codes(5, 3) + _codes(5, 2) + _codes(5, 1)))
    assert len(idc) == len(out_toks)
    # Intent codes (4-d, pairwise cos in {0, +-1/2, -1}).
    icode = dict(zip(data.INTENTS, _codes(4, 2)))
    unk = icode["unknown"]

    embed = torch.zeros(V, d, **f64)
    for w, c in idc.items():
        embed[tk.tid(w), ID] = c

    def put(word, intent, strength):
        i = tk.tid(word)  # asserts the word is in the vocabulary
        embed[i, CODE] = icode[intent] - unk  # value = code_X after bv adds code_unk
        embed[i, STR] = strength

    for intent, (strong, weak) in KW.items():
        for w in strong.split():
            put(w, intent, 1.0)
        for w in weak.split():
            put(w, intent, 0.5)
    for w in OOD.split():
        put(w, "unknown", 0.75)

    pos = torch.zeros(L, d, **f64)
    pos[:, P] = torch.arange(L, **f64)  # P carries the absolute position t

    h, e = cfg.n_heads, cfg.d_head
    wq, wk, wv = (torch.zeros(h, d, e, **f64) for _ in range(3))
    wo = torch.zeros(h, e, d, **f64)
    bq, bk, bv = (torch.zeros(h, e, **f64) for _ in range(3))
    sq = e**0.5  # cancels the 1/sqrt(d_head) in the score
    # Head 0: intent detector. Constant query; key = T*strength - MU*t (earliest strong keyword wins);
    # value = intent code (neutral tokens -> 'unknown' via bv); written to CODE with gain A.
    bq[0, 0] = sq
    wk[0, STR, 0], wk[0, P, 0] = T_KEY, -MU
    for j in range(4):
        wv[0, CODE[j], j] = 1.0
        wo[0, j, CODE[j]] = A
    bv[0] = unk
    # Head 1: <sep> locator. Key = LAM * (identity . code_sep); value = t; output -t_sep into P.
    bq[1, 0] = sq
    wk[1, ID, 0] = LAM * idc["<sep>"]
    wv[1, P, 0] = 1.0
    wo[1, 0, P] = -1.0
    bo = torch.zeros(d, **f64)

    # MLP: per intent X, ramps r_k = ReLU(n - k - BETA*(1 - code_X . I)), k = -1..len-2.
    # Output = sum_k r_k * (second difference of the reply's identity codes), i.e. the piecewise-linear
    # interpolation that equals S*idcode(reply_X[n]) at integer step n.
    w_in = torch.zeros(d, d_mlp, **f64)
    b_in = torch.zeros(d_mlp, **f64)
    w_out = torch.zeros(d_mlp, d, **f64)
    j = 0
    for x, r in replies.items():
        v = [idc[w] for w in r]
        for k in range(-1, len(r) - 1):
            w_in[P, j] = 1.0
            w_in[CODE, j] = BETA * icode[x] / A
            b_in[j] = -k - BETA
            if k == -1:
                delta = v[0]
            elif k == 0:
                delta = v[1] - 2 * v[0]
            else:
                delta = v[k + 1] - 2 * v[k] + v[k - 1]
            w_out[j, ID] = S * delta
            j += 1
    b_out = torch.zeros(d, **f64)
    b_unembed = torch.zeros(V, **f64)

    p = "layers.0."
    params = {
        "embed": embed,
        "pos": pos,
        p + "wq": wq,
        p + "wk": wk,
        p + "wv": wv,
        p + "wo": wo,
        p + "bq": bq,
        p + "bk": bk,
        p + "bv": bv,
        p + "bo": bo,
        p + "w_in": w_in,
        p + "b_in": b_in,
        p + "w_out": w_out,
        p + "b_out": b_out,
        "b_unembed": b_unembed,
    }
    return cfg, params
