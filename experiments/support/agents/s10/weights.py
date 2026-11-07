"""s10: hand-constructed IT-support transformer (1 layer, 2 heads, d_model=11).

Residual stream layout (every dimension has one job):
  A  = 0,1   token identity / output code: unit vector at angle 2*pi*j/N for the
             j-th reply token (N=126 incl. <sep>,<eos>); zero for other tokens.
             Tied unembedding => logit[n] = x . E[n] reads this code back out.
  B  = 2,3   "default successor" code of a reply token: A-code of its most common
             next token in the replies (bigram table stored in the embedding).
  K  = 4     keyword strength (tier + intent tie-break) -> key of the intent head.
  I  = 5,6   intent vote (unit vector at angle 2*pi*i/14); the intent head writes
             the winning vote back into these same dims (J).
  P  = 7,8   learned position (cos wt, sin wt), w = 2*pi/64 -> prev-token head.
  H  = 9,10  A-code of the previous token (written by the prev-token head).
Head 0 (intent): constant query, key = K  -> hard attention to the strongest keyword
                 (or the <bos> sink = unknown); copies its vote I -> I.
Head 1 (prev):   query = pos rotated by one step, key = pos -> attends t-1;
                 copies A -> H.
MLP: 2 linear "transfer" neurons (R*B -> A, the bigram), 14 <sep> neurons
     (cur==<sep> AND intent==i -> first reply token), and correction neurons
     (cur==c AND prev==p [AND intent==i]) where the bigram default is wrong.
"""

import math
from collections import Counter, defaultdict

import torch

from wsp.support import data, tokenizer as T
from wsp.support.model import Config, param_shapes

D = 11
A, B, K, I, P, H = [0, 1], [2, 3], 4, [5, 6], [7, 8], [9, 10]
R = 1e6  # output scale: logit gap to nearest wrong code = R*(1-cos(2pi/126)) ~ 1243
GH = 1e4  # identity-match gate gain (mismatch costs >= GH*0.00124 = 12.4)
GI = 200.0  # intent-match gate gain (mismatch costs >= GI*(1-cos(2pi/14)) = 19.8)
GK = 600.0  # intent-head score per unit of keyword strength
GP = 1e4  # prev-head score scale (gap to s=t, t-2: GP*(1-cos w) = 48)
W = 2 * math.pi / 64

# --- intent keyword lists, hand-written from the vocabulary -------------------
# tier: 6 specific noun, 5 strong cue, 4 weak cue, 3 off-topic (votes unknown),
# 2 greeting/thanks, 1 = <bos> sink (unknown), 0 = every other token.
KW = {
    "password": {
        6: "password pwd",
        5: "forgot reset expired expires remember",
        4: "login logins log change",
    },
    "locked": {
        6: "locked unlock unlocks blocked",
        5: "attempts",
        4: "account accounts user",
    },
    "vpn": {6: "vpn globalprotect", 4: "remote home server"},
    "wifi": {
        6: "wifi wi-fi wireless",
        5: "internet network",
        4: "signal drops dropping connection connect connecting connected disconnected disconnecting reconnect weak",
    },
    "email": {
        6: "email emails outlook mail inbox outbox",
        5: "calendar syncing",
        4: "send message arriving arrives",
    },
    "printer": {6: "printer print printing toner jammed", 5: "paper", 4: "queue"},
    "software": {
        6: "software install installed installing license program",
        5: "excel adobe photoshop python visual studio application download admin rights",
        4: "update updating tool setup request",
    },
    "slow": {
        6: "slow sluggish lagging freezing freezes frozen hangs",
        5: "performance cpu forever",
        4: "crash crashes crashing crashed",
    },
    "mfa": {
        6: "mfa 2fa authenticator authentication factor multi",
        5: "code sso prompt push notification approve approved",
        4: "phone",
    },
    "disk": {6: "disk storage space", 5: "full drive cleanup"},
    "monitor": {
        6: "monitor screen screens display hdmi projector docking",
        5: "external flickers station",
        4: "second black blank cable",
    },
    "hello": {2: "hello hi hey morning afternoon evening"},
    "thanks": {2: "thanks thank cheers awesome perfect great worked fixed"},
    "unknown": {
        3: "weather lunch pizza joke parking vacation payday flight coffee game meeting room book ceo london floor"
    },
}
INTENTS = data.INTENTS
# tie-break inside a tier (higher wins): specific problems before generic ones
PRIO = [
    "unknown",
    "hello",
    "thanks",
    "slow",
    "software",
    "disk",
    "monitor",
    "printer",
    "email",
    "wifi",
    "vpn",
    "password",
    "locked",
    "mfa",
]


def unit(a):
    return torch.tensor([math.cos(a), math.sin(a)], dtype=torch.float64)


def build():
    cfg = Config(
        d_model=D,
        n_layers=1,
        n_heads=2,
        d_head=2,
        d_mlp=0,
        pos="learned",
        tie_embeddings=True,
        norm=False,
        bias=True,
    )
    V = T.vocab_size()
    reps = {k: ["<sep>"] + T.split(r) + ["<eos>"] for k, r in data.REPLIES.items()}
    rtok = list(dict.fromkeys(t for w in reps.values() for t in w))  # 126 reply tokens
    code = {t: unit(2 * math.pi * j / len(rtok)) for j, t in enumerate(rtok)}
    icode = {k: unit(2 * math.pi * i / len(INTENTS)) for i, k in enumerate(INTENTS)}

    # bigram default successor B(c) = most common next token over all replies
    succ = defaultdict(Counter)
    ctx = defaultdict(dict)  # (prev, cur) -> {intent: next}
    for k, w in reps.items():
        for p, c, n in zip(["Q"] + w, w, w[1:]):
            succ[c][n] += 1
            ctx[(p, c)][k] = n
    Bdef = {c: s.most_common(1)[0][0] for c, s in succ.items() if c != "<sep>"}

    # --- embedding --------------------------------------------------------------
    E = torch.zeros(V, D, dtype=torch.float64)
    for t in rtok:
        E[T.tid(t), A] = code[t]
        if t in Bdef:
            E[T.tid(t), B] = code[Bdef[t]]
    for k, tiers in KW.items():
        for tier, words in tiers.items():
            for w in words.split():
                E[T.tid(w), K] = tier + 0.05 * PRIO.index(k)
                E[T.tid(w), I] = icode[k]
    E[T.tid("<bos>"), K] = 1.0  # sink: wins only if no keyword is present
    E[T.tid("<bos>"), I] = icode["unknown"]

    pos = torch.zeros(T.MAX_LEN, D, dtype=torch.float64)
    for t in range(T.MAX_LEN):
        pos[t, P] = unit(W * t)

    # --- attention ----------------------------------------------------------------
    wq = torch.zeros(2, D, 2, dtype=torch.float64)
    wk, wv = wq.clone(), wq.clone()
    wo = torch.zeros(2, 2, D, dtype=torch.float64)
    bq = torch.zeros(2, 2, dtype=torch.float64)
    # head 0: score = GK * strength(key token); copy vote I -> I
    bq[0, 0] = GK * math.sqrt(2)
    wk[0, K, 0] = 1.0
    wv[0, I, :] = torch.eye(2, dtype=torch.float64)
    wo[0, :, I] = torch.eye(2, dtype=torch.float64)
    # head 1: q_t = c*Rot(-w) p_t, k_s = c*p_s -> score = GP cos(w(t-1-s)); copy A -> H
    c = math.sqrt(GP * math.sqrt(2))
    rot = torch.tensor(
        [[math.cos(W), -math.sin(W)], [math.sin(W), math.cos(W)]], dtype=torch.float64
    )
    wq[1][P, :] = c * rot.T.T  # (p @ M)[j] = sum_i p_i M_ij; M = Rot(-w)^T = Rot(w)
    wk[1][P, :] = c * torch.eye(2, dtype=torch.float64)
    wv[1][A, :] = torch.eye(2, dtype=torch.float64)
    wo[1][:, H] = torch.eye(2, dtype=torch.float64)

    # --- MLP neurons: (name, w_in vector, b_in, w_out vector) -------------------------
    neurons = []
    z = lambda: torch.zeros(D, dtype=torch.float64)  # noqa: E731
    for j in range(
        2
    ):  # linear bigram transfer: relu(B_j + 2) * R -> A_j (minus 2R in b_out)
        wi, wo_ = z(), z()
        wi[B[j]] = 1.0
        wo_[A[j]] = R
        neurons.append((f"transfer{j}", wi, 2.0, wo_))

    def gate(cur=None, prev=None, intent=None, out=None, name=""):
        wi, b = z(), 1.0
        if cur:
            wi[A] = GH * code[cur]
            b -= GH
        if prev:
            wi[H] = GH * code[prev]
            b -= GH
        if intent:
            wi[I] = GI * icode[intent]
            b -= GI
        wo_ = z()
        wo_[A] = R * out
        neurons.append((name, wi, b, wo_))

    for k, w in reps.items():  # <sep> -> first reply token of the detected intent
        gate(cur="<sep>", intent=k, out=code[w[1]], name=f"sep:{k}")
    for (p, cur), byk in ctx.items():  # corrections where the bigram default is wrong
        if cur == "<sep>":
            continue
        nexts = set(byk.values())
        if len(nexts) == 1:
            n = nexts.pop()
            if n != Bdef[cur]:
                gate(
                    cur=cur,
                    prev=p,
                    out=code[n] - code[Bdef[cur]],
                    name=f"{p}|{cur}->{n}",
                )
        else:
            for k, n in byk.items():
                if n != Bdef[cur]:
                    gate(
                        cur=cur,
                        prev=p,
                        intent=k,
                        out=code[n] - code[Bdef[cur]],
                        name=f"{k}:{p}|{cur}->{n}",
                    )

    m = len(neurons)
    cfg = Config(**{**cfg.to_dict(), "d_mlp": m})
    w_in = torch.stack([n[1] for n in neurons], 1)
    b_in = torch.tensor([n[2] for n in neurons], dtype=torch.float64)
    w_out = torch.stack([n[3] for n in neurons], 0)
    b_out = z()
    b_out[A] = -2 * R

    L = "layers.0."
    params = {
        "embed": E,
        "pos": pos,
        L + "wq": wq,
        L + "wk": wk,
        L + "wv": wv,
        L + "wo": wo,
        L + "bq": bq,
        L + "bk": torch.zeros(2, 2, dtype=torch.float64),
        L + "bv": torch.zeros(2, 2, dtype=torch.float64),
        L + "bo": z(),
        L + "w_in": w_in,
        L + "b_in": b_in,
        L + "w_out": w_out,
        L + "b_out": b_out,
        "b_unembed": torch.zeros(V, dtype=torch.float64),
    }
    assert set(params) == set(param_shapes(cfg))
    build.neurons = [n[0] for n in neurons]  # provenance
    return cfg, params
