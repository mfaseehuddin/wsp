"""c1: hand-written weights for the IT support bot ("keyword router + n-gram script reader").

2 layers, no LayerNorm, tied embeddings. Every residual dimension has a name
(see `Space`); every MLP unit is one line of a reply script. See EXPLANATION.md.
"""

from __future__ import annotations

import math

import torch

from wsp.bot import kb
from wsp.bot import tokenizer as T
from wsp.bot.model import Config, param_shapes

# ------------------------------------------------------------------ knowledge
N_TOPIC_SLOTS = 25  # 22 topics + 3 spare slots for new FAQ topics
SPARE_WORDS = 30  # spare word dims for new reply words
PSEUDO = ["recall", "escalate", "hello", "thanks", "bye"]  # non-FAQ classes in head T
FLOWS = ["status", "reset", "urgent", "service"]  # classes in head F (override T)

# Hand-written keyword lists (topic -> synonyms). Topic names themselves are added
# automatically with a higher strength. Words missing from the vocab are skipped.
SYN = {
    "password": "password passwords forgot expired reset",
    "locked": "locked unlock attempts",
    "vpn": "vpn",
    "wifi": "wifi wireless wi-fi",
    "email": "outlook mail inbox emails",
    "printer": "print toner",
    "software": "install excel license application adobe installed",
    "slow": "freezing sluggish hangs freezes frozen",
    "mfa": "authenticator 2fa code",
    "disk": "storage space full",
    "monitor": "screen display external signal hdmi projector",
    "audio": "sound hear speakers microphone mic headset muted",
    "camera": "webcam video",
    "teams": "meeting meetings chat",
    "onedrive": "sync syncing uploading",
    "access": "permission shared folder denied",
    "browser": "chrome edge cookies",
    "battery": "charging charge charger power dies drains",
    "bluetooth": "pair headphones mouse earbuds",
    "phishing": "!suspicious !weird !fake",
    "newhire": "!hire !starter !onboarding starts joining person",
    "restore": "!deleted !recover",
}
PSEUDO_KW = {  # class -> (strength, words)
    "recall": (3.6, "number repeat remind"),
    "thanks": (3.5, "thanks thank worked cheers great perfect fixed"),
    "bye": (3.45, "bye goodbye later"),
    "escalate": (3.4, "still tried help fails failed didn't nothing"),
    "hello": (3.3, "hello hi hey morning afternoon evening"),
}
RECALL_WEAK = (3.2, "ticket")
FLOW_KW = {  # flow -> (strength, words); ids and user names are added in code
    "urgent": (4.6, "urgent asap emergency critical"),
    "service": (4.4, "down outage offline status"),
}

# ------------------------------------------------------------------ constants
A_CLS = 1e5  # attention bonus for "token is of the class this head looks for"
R_REC = 25600.0  # recency: score gain per position = R_REC/256 = 100 nats
C_SEG = 1e5  # same-turn preference in heads F/T
BIG = 1e6  # bot-side tokens are invisible to heads F/T
BETA = 4000.0  # keyword strength scale (strength stored /10, so 400 nats per unit)
K_PREV = 65536 * 60.0  # sharpness of the previous-token head
OUT = 20.0  # logit written by one MLP script unit


class Space:
    """Named residual dimensions."""

    def __init__(self):
        self.idx: dict[str, int] = {}

    def add(self, name):
        if name not in self.idx:
            self.idx[name] = len(self.idx)
        return self.idx[name]

    def __getitem__(self, k):
        return self.idx[k]


def tok(w):
    return T.token_to_id().get(w)


def templates():
    """intent -> list of token scripts. $X = slot, '#R' + tokens = harness result."""
    sp = T.split
    out = {}
    for t, (team, r) in kb.FAQ.items():
        out[t] = [["<bot>", "<think>", t, "</think>", *sp(r), "<end>"]]
    for k in ["hello", "thanks", "bye", "unknown"]:
        out[k] = [["<bot>", *sp(kb.SOCIAL[k]), "<end>"]]
    out["recall"] = [["<bot>", "your", "ticket", "number", "is", "$ID", ".", "<end>"]]
    out["escalate"] = [["<bot>", "<tool>", "create_ticket", "$TOPIC", "</tool>", "#R", "ticket", "@ID",
                        "</result>", "i", "opened", "ticket", "$ID", "about", "$TOPIC", ".", "the",
                        "$TEAMH", "team", "will", "contact", "you", "today", ".", "<end>"]]
    out["status"] = [["<bot>", "<tool>", "lookup_ticket", "$ID", "</tool>", "#R", "@ST", "</result>",
                      "ticket", "$ID", "is", "$ST", ".", "<end>"]]
    out["service"] = [["<bot>", "<tool>", "service_status", "$SVC", "</tool>", "#R", "@ST", "</result>",
                       "$SVC", "is", "$ST", "right", "now", ".", "<end>"]]
    out["urgent"] = [["<bot>", "<tool>", "page_team", "$TEAMT", "</tool>", "#R", "paged", "</result>",
                      "i", "paged", "the", "$TEAMT", "team", ".", "they", "will", "call", "you",
                      "within", "15", "minutes", ".", "<end>"]]
    pre = ["<bot>", "<tool>", "reset_password", "$USR", "</tool>", "#R"]
    out["reset"] = [pre + ["sent", "</result>", *sp("a reset link was sent to"), "$USR", ".", "<end>"],
                    pre + ["<unk>", "</result>", *sp("i could not find the user"), "$USR", ".", "<end>"]]
    return out


SLOT_CLASS = {"$ID": "isID", "@ID": "isID", "$TOPIC": "isTopic", "$TEAMH": "isTeam",
              "$TEAMT": "isTeam", "$SVC": "isSVC", "$ST": "isST", "@ST": "isST", "$USR": "isUSR"}


def feat(x):
    """Key feature of a script token: a class flag for slot tokens, else the word."""
    return ("c", SLOT_CLASS[x]) if x in SLOT_CLASS else ("w", x)


def script_units():
    """Turn the scripts into MLP units: (intent, cur_feat, prev_feat|None, output)."""
    units = []
    for intent, seqs in templates().items():
        rows = []  # (cur, prev, out)
        for s in seqs:
            s = [x for x in s if x != "#R"]
            harness = set()
            for i, x in enumerate(s):  # result block tokens are written by the harness
                if x == "</tool>":
                    j = i + 1
                    while s[j] != "</result>":
                        harness.add(j)
                        j += 1
                    harness.add(j)
            for i in range(len(s) - 1):
                if i + 1 in harness:
                    continue
                rows.append((feat(s[i]), feat(s[i - 1]) if i else None, s[i + 1]))
        rows = list(dict.fromkeys(rows))
        by_cur = {}
        for c, p, o in rows:
            by_cur.setdefault(c, set()).add(o)
        for c, p, o in rows:
            if len(by_cur[c]) == 1:
                units.append((intent, c, None, o))
            else:
                units.append((intent, c, p, o))
    return list(dict.fromkeys(units))


def class_of(w):
    c = []
    if w in kb.FAQ:
        c.append("isTopic")
    if w in kb.TEAMS:
        c.append("isTeam")
    if w in kb.SERVICES:
        c.append("isSVC")
    if w in kb.STATES + kb.STATUSES:
        c.append("isST")
    if w in kb.TICKET_IDS:
        c.append("isID")
    if w in kb.USERS + kb.NON_USERS:
        c.append("isUSR")
    return c


def build():
    topics = list(kb.FAQ)
    tclasses = topics + [f"spare{i}" for i in range(N_TOPIC_SLOTS - len(topics))] + PSEUDO
    teams = kb.TEAMS
    users = kb.USERS + kb.NON_USERS
    states = kb.STATES + kb.STATUSES
    units = script_units()

    # ---- word vocabulary W (one-hot "current word" dims, also the output dims)
    words = []
    for _, c, p, o in units:
        for f in (c, p):
            if f and f[0] == "w":
                words.append(f[1])
        if not o.startswith("$"):
            words.append(o)
    words += topics + teams + kb.SERVICES + states + ["sent", "<unk>", "paged", "error"]
    words = [w for w in dict.fromkeys(words) if tok(w) is not None and w not in kb.TICKET_IDS]

    # ---- previous-token features needed to disambiguate scripts
    pv = ["w:<think>"] + sorted({f"{p[0]}:{p[1]}" for _, _, p, _ in units if p})
    pv = list(dict.fromkeys(pv))

    S = Space()
    for w in words:
        S.add("W:" + w)
    for i in range(SPARE_WORDS):
        S.add(f"W:spare{i}")
    for a in range(10):
        S.add(f"tens{a}")
    for a in range(10):
        S.add(f"units{a}")
    for u in users:
        S.add("U:" + u)
    for c in ["isID", "isUSR", "isSVC", "isST", "isTopic", "isTeam", "isUserTag", "isBotTag"]:
        S.add(c)
    for c in tclasses:
        S.add("vote:" + c)
    for t in teams:
        S.add("team:" + t)
    S.add("sT")
    for f in FLOWS:
        S.add("fvote:" + f)
    S.add("sF")
    S.add("pos1"), S.add("pos2")
    # attention outputs
    S.add("seg"), S.add("botside")
    for x in pv:
        S.add("PV:" + x)
    for c in tclasses:
        S.add("Tout:" + c)
    for t in teams:
        S.add("Tteam:" + t)
    for f in FLOWS:
        S.add("Fout:" + f)
    for c in topics + tclasses[len(topics):N_TOPIC_SLOTS]:
        S.add("THout:" + c)
    for t in teams:
        S.add("THteam:" + t)
    for a in range(10):
        S.add(f"IDc_t{a}")
    for a in range(10):
        S.add(f"IDc_u{a}")
    for u in users:
        S.add("USc:" + u)
    for s_ in kb.SERVICES:
        S.add("USsvc:" + s_)
    for s_ in states:
        S.add("STc:" + s_)
    d = len(S.idx)
    e = max(36, len(pv), 2)
    V = T.vocab_size()

    # ---- token embeddings (tied: also the unembedding)
    E = torch.zeros(V, d, dtype=torch.float64)

    def put(w, name, val=1.0):
        i = tok(w)
        if i is not None:
            E[i, S[name]] = val

    for w in words:
        put(w, "W:" + w)
        for c in class_of(w):
            put(w, c)
    for n in kb.TICKET_IDS:
        put(n, f"tens{int(n[2])}"), put(n, f"units{int(n[3])}"), put(n, "isID")
        put(n, "fvote:status"), put(n, "sF", 0.50)
    for u in users:
        put(u, "U:" + u), put(u, "isUSR"), put(u, "fvote:reset"), put(u, "sF", 0.48)
    put("<user>", "isUserTag"), put("<bot>", "isBotTag")
    for f, (s, ws) in FLOW_KW.items():
        for w in ws.split():
            put(w, "fvote:" + f), put(w, "sF", s / 10)

    def topic_kw(w, c, s):
        if tok(w) is None:
            return
        E[tok(w), [S["vote:" + x] for x in tclasses]] = 0
        put(w, "vote:" + c), put(w, "sT", s / 10)
        if c in kb.FAQ:
            E[tok(w), [S["team:" + x] for x in teams]] = 0
            put(w, "team:" + kb.FAQ[c][0])

    s0, w0 = RECALL_WEAK
    for w in w0.split():
        topic_kw(w, "recall", s0)
    for c, (s, ws) in PSEUDO_KW.items():
        for w in ws.split():
            topic_kw(w, c, s)
    for i, t in enumerate(topics):  # synonyms: strength 3.9 .. 4.6 (distinct per topic)
        for w in SYN[t].split():
            if not w.startswith("!"):
                topic_kw(w, t, 3.9 + 0.03 * i)
    for i, t in enumerate(topics):  # topic names: strength 5.0 .. 5.7
        topic_kw(t, t, 5.0 + 0.03 * i)
    for i, t in enumerate(topics):  # strong synonyms ("!word"): 6.0 .. 6.7, beat names
        for w in SYN[t].split():
            if w.startswith("!"):
                topic_kw(w[1:], t, 6.0 + 0.03 * i)

    P = torch.zeros(T.MAX_LEN, d, dtype=torch.float64)
    for t in range(T.MAX_LEN):
        P[t, S["pos1"]] = t / 256
        P[t, S["pos2"]] = (t / 256) ** 2

    cfg = Config(d_model=d, n_layers=2, n_heads=4, d_head=e, d_mlp=0, pos="learned",
                 tie_embeddings=True, norm=False, bias=True)
    n_units = None  # filled below
    params = {"embed": E, "pos": P}
    sq = math.sqrt(e)

    def head_params():
        return {k: torch.zeros(4, d, e, dtype=torch.float64) for k in ("wq", "wk", "wv")} | {
            "wo": torch.zeros(4, e, d, dtype=torch.float64),
            "bq": torch.zeros(4, e, dtype=torch.float64),
            "bk": torch.zeros(4, e, dtype=torch.float64),
            "bv": torch.zeros(4, e, dtype=torch.float64),
            "bo": torch.zeros(d, dtype=torch.float64)}

    def copy(L, h, pairs):
        """Head h moves residual dim src -> dst through value slot j."""
        for j, (src, dst) in enumerate(pairs):
            L["wv"][h, S[src], j] = 1.0
            L["wo"][h, j, S[dst]] = 1.0

    def recency(L, h, flags, a=A_CLS):
        """Score = a * (sum of flags) + recency: picks the latest flagged token."""
        L["bq"][h, 0] = sq
        for f in flags:
            L["wk"][h, S[f], 0] = a
        L["wk"][h, S["pos1"], 0] = R_REC

    # ---- layer 0: P (previous token), U (segment), ID (latest ticket id), US (latest user/service)
    L0 = head_params()
    L0["wq"][0, S["pos1"], 0] = sq
    L0["bq"][0, 0] = -sq / 256
    L0["bq"][0, 1] = sq
    L0["wk"][0, S["pos1"], 0] = 2 * K_PREV
    L0["wk"][0, S["pos2"], 1] = -K_PREV
    src = {}
    for x in pv:
        kind, name = x.split(":", 1)
        src[x] = ("W:" + name) if kind == "w" else name
    copy(L0, 0, [(src[x], "PV:" + x) for x in pv])
    recency(L0, 1, ["isUserTag", "isBotTag"])
    copy(L0, 1, [("pos1", "seg"), ("isBotTag", "botside")])
    recency(L0, 2, ["isID"])
    copy(L0, 2, [(f"tens{a}", f"IDc_t{a}") for a in range(10)] + [(f"units{a}", f"IDc_u{a}") for a in range(10)])
    recency(L0, 3, ["isUSR", "isSVC"])
    copy(L0, 3, [("U:" + u, "USc:" + u) for u in users] + [("W:" + s_, "USsvc:" + s_) for s_ in kb.SERVICES])

    # ---- layer 1: F (flow keyword), T (topic/social keyword), TH (latest think topic), ST (latest state)
    L1 = head_params()
    for h, sdim in ((0, "sF"), (1, "sT")):
        L1["bq"][h, 0] = sq
        L1["wk"][h, S["seg"], 0] = C_SEG
        L1["wk"][h, S["botside"], 0] = -BIG
        L1["wk"][h, S[sdim], 0] = BETA
    copy(L1, 0, [("fvote:" + f, "Fout:" + f) for f in FLOWS])
    copy(L1, 1, [("vote:" + c, "Tout:" + c) for c in tclasses] + [("team:" + t, "Tteam:" + t) for t in teams])
    recency(L1, 2, ["isTopic", "PV:w:<think>"])
    copy(L1, 2, [("vote:" + c, "THout:" + c) for c in tclasses[:N_TOPIC_SLOTS]] + [("team:" + t, "THteam:" + t) for t in teams])
    recency(L1, 3, ["isST"])
    copy(L1, 3, [("W:" + s_, "STc:" + s_) for s_ in states])

    # ---- layer 1 MLP: one unit per script line (and per value for slot lines)
    rows = []  # (list of (dim, weight), bias, list of (out_dim, weight))
    f_dims = [S["Fout:" + f] for f in FLOWS]
    t_dims = [S["Tout:" + c] for c in tclasses]

    def intent_terms(intent):
        if intent in FLOWS:
            return [(S["Fout:" + intent], 1.0)], 0.0
        if intent == "unknown":
            return [(i, -2.0) for i in f_dims + t_dims], 1.0
        return [(S["Tout:" + intent], 1.0)] + [(i, -2.0) for i in f_dims], 0.0

    def fdim(f, pref=""):
        k, n = f
        return S[pref + (k + ":" + n if pref else ("W:" + n if k == "w" else n))]

    slot_vals = {
        "$TOPIC": [(f"THout:{t}", [("W:" + t, OUT)]) for t in topics],
        "$TEAMH": [(f"THteam:{t}", [("W:" + t, OUT)]) for t in teams],
        "$TEAMT": [(f"Tteam:{t}", [("W:" + t, OUT)]) for t in teams],
        "$ID": [(f"IDc_t{a}", [(f"tens{a}", OUT)]) for a in range(10)]
        + [(f"IDc_u{a}", [(f"units{a}", OUT)]) for a in range(10)],
        "$SVC": [(f"USsvc:{s_}", [("W:" + s_, OUT)]) for s_ in kb.SERVICES],
        "$ST": [(f"STc:{s_}", [("W:" + s_, OUT)]) for s_ in states],
        "$USR": [(f"USc:{u}", [("U:" + u, OUT)]) for u in users],
    }
    for intent, c, p, o in units:
        terms, b = intent_terms(intent)
        terms = terms + [(fdim(c), 1.0)]
        n = 2
        if p is not None:
            terms.append((S["PV:" + p[0] + ":" + p[1]], 1.0))
            n += 1
        if o.startswith("$"):
            for vdim, outs in slot_vals[o]:
                rows.append((terms + [(S[vdim], 1.0)], b - n, [(S[x], wt) for x, wt in outs]))
        else:
            rows.append((terms, b - (n - 1), [(S["W:" + o], OUT)]))
    m = len(rows)
    cfg = Config(d_model=d, n_layers=2, n_heads=4, d_head=e, d_mlp=m, pos="learned",
                 tie_embeddings=True, norm=False, bias=True)
    w_in = torch.zeros(d, m, dtype=torch.float64)
    b_in = torch.zeros(m, dtype=torch.float64)
    w_out = torch.zeros(m, d, dtype=torch.float64)
    for j, (terms, b, outs) in enumerate(rows):
        for i, wt in terms:
            w_in[i, j] += wt
        b_in[j] = b
        for i, wt in outs:
            w_out[j, i] += wt
    for li, L in enumerate((L0, L1)):
        for k, v in L.items():
            params[f"layers.{li}.{k}"] = v
        z = li == 0  # layer-0 MLP is unused (d_mlp is shared by both layers)
        params[f"layers.{li}.w_in"] = torch.zeros(d, m, dtype=torch.float64) if z else w_in
        params[f"layers.{li}.b_in"] = torch.zeros(m, dtype=torch.float64) if z else b_in
        params[f"layers.{li}.w_out"] = torch.zeros(m, d, dtype=torch.float64) if z else w_out
        params[f"layers.{li}.b_out"] = torch.zeros(d, dtype=torch.float64)
    params["b_unembed"] = torch.zeros(V, dtype=torch.float64)
    assert set(params) == set(param_shapes(cfg)), set(param_shapes(cfg)) ^ set(params)
    build.space, build.units = S, units  # for inspection / docs
    return cfg, params
