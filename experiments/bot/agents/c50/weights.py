"""c50: hand-written weights for the IT help-desk bot (no training).

Two-layer transformer, every number placed by a named construction below.
Layer 0 attention: previous-token copy, "which user turn am I in", "am I inside a bot turn".
Layer 0 MLP:       "copy triggers" - fires when the next token must be copied from context.
Layer 1 attention: flow head (what kind of request), topic head (latest topic), copy head.
Layer 1 MLP:       reply table - one unit per (reply template, step) writes the next token.
See EXPLANATION.md for the edit map.
"""

from __future__ import annotations

import math

import torch

from wsp.bot import kb, model
from wsp.bot import tokenizer as T

TID = T.token_to_id()
V = T.vocab_size()

# ----------------------------------------------------------------- hand-written keyword lists
# topic -> {word: strength 1..8}. Higher strength wins inside one user message.
TOPIC_WORDS = {
    "password": {"login": 2, "password": 3, "passwords": 3, "forgot": 4, "expired": 3},
    "locked": {"account": 3, "locked": 6, "lockout": 6, "unlock": 6, "attempts": 5},
    "vpn": {"vpn": 7, "tunnel": 5, "globalprotect": 7, "anyconnect": 7},
    "wifi": {"internet": 3, "connection": 2, "wifi": 7, "wi-fi": 7, "wireless": 6, "network": 2},
    "email": {"send": 3, "calendar": 4, "email": 4, "outlook": 6, "mail": 5, "inbox": 5, "emails": 4},
    "printer": {"printer": 7, "print": 6, "printing": 6, "toner": 6, "jammed": 5},
    "software": {"adobe": 6, "photoshop": 6, "zoom": 4, "installed": 4, "software": 5, "install": 5, "installing": 5, "excel": 5, "license": 5,
                 "program": 4, "application": 4},
    "slow": {"lagging": 5, "fan": 3, "crash": 2, "crashes": 2, "crashing": 2, "slow": 5, "freezing": 5, "freezes": 5, "frozen": 5, "sluggish": 5, "lag": 5,
             "laggy": 5, "hangs": 4},
    "mfa": {"authentication": 6, "push": 5, "mfa": 7, "authenticator": 7, "2fa": 7, "otp": 6, "verification": 5, "code": 4},
    "disk": {"cleanup": 5, "disk": 6, "space": 5, "storage": 5, "full": 4},
    "monitor": {"docking": 5, "monitor": 6, "screen": 5, "display": 5, "signal": 5, "hdmi": 5,
                "projector": 5, "external": 3},
    "audio": {"audio": 6, "sound": 6, "headset": 5, "speaker": 5, "speakers": 5, "hear": 5,
              "microphone": 5, "mic": 5, "muted": 5},
    "camera": {"image": 3, "picture": 3, "camera": 7, "webcam": 7, "video": 5},
    "teams": {"chat": 4, "teams": 5, "meeting": 3, "meetings": 3},
    "onedrive": {"files": 2, "file": 2, "cloud": 4, "sharepoint": 4, "backup": 4, "onedrive": 5, "sync": 6, "syncing": 6, "synced": 6, "uploading": 5,
                 "upload": 5},
    "access": {"folders": 4, "rights": 5, "admin": 4, "access": 6, "permission": 6, "permissions": 6, "denied": 5, "shared": 5,
               "folder": 4, "drive": 3},
    "browser": {"loading": 3, "load": 3, "pages": 4, "cache": 4, "browser": 7, "chrome": 7, "firefox": 7, "edge": 5, "website": 5,
                "websites": 5, "cookies": 6, "page": 4},
    "battery": {"plug": 4, "drains": 6, "battery": 7, "charging": 7, "charge": 6, "charger": 6, "dies": 4,
                "power": 4},
    "bluetooth": {"bluetooth": 8, "pair": 7, "pairing": 7, "paired": 7, "mouse": 4,
                  "keyboard": 4, "headphones": 4, "earbuds": 5, "airpods": 5},
    "phishing": {"phishing": 8, "suspicious": 7, "scam": 7, "spam": 6, "link": 5,
                 "fake": 5, "weird": 3},
    "newhire": {"new": 2, "hire": 7, "onboarding": 7, "person": 5, "starts": 5, "starter": 6,
                "employee": 5, "joiner": 6, "joining": 5},
    "restore": {"recycle": 5, "removed": 5, "document": 2, "recover": 7, "restore": 7, "deleted": 6, "undelete": 7, "mistake": 4,
                "lost": 4},
}
# flow channels in priority order (highest first); each gets a priority score.
FLOWS = ["urgent", "service", "status", "recall", "reset", "esc", "topic", "escw",
         "bye", "thanks", "hello"]
PRIO = dict(zip(FLOWS, [110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10], strict=True))
FLOW_WORDS = {
    "urgent": ["urgent", "emergency", "asap", "immediately", "critical"],
    "service": ["down", "outage", "outages", "offline"],
    "status": list(kb.TICKET_IDS),  # a ticket number in the message => status lookup
    "recall": ["number", "id", "remind", "repeat", "reference"],
    "reset": list(kb.USERS) + list(kb.NON_USERS),  # a user name => reset that user
    "esc": ["still", "tried", "didn't", "doesn't", "persists", "worse"],
    "escw": ["help", "nothing", "same", "fixed", "luck", "again", "fails", "failed", "failing", "working"],  # only if no topic
    "bye": ["bye", "goodbye", "all", "done", "cya", "later"],
    "thanks": ["thanks", "thank", "thx", "cheers", "appreciate", "worked", "great", "awesome", "perfect"],
    "hello": ["hello", "hi", "hey", "morning", "afternoon", "evening", "greetings"],
}

# ----------------------------------------------------------------- reply templates (from KB)
FAQ = {t: (team, reply) for t, (team, reply) in kb.faq(0).items()} if hasattr(kb, "faq") else {}
SOC = kb.SOCIAL
FL = kb.FLOWS
SLOTS = {"{id}": "ID", "{user}": "USER", "{service}": "SVC", "{state}": "STATE",
         "{status}": "STATUS", "{topic}": "TOPIC", "{team}": "TEAM"}
COPY = ["ID", "USER", "SVC", "STATE", "STATUS"]  # filled by the copy head
CLASSES = COPY + ["TOPIC", "TEAM"]
CLASS_TOKENS = {
    "ID": kb.TICKET_IDS, "USER": list(kb.USERS) + list(kb.NON_USERS), "SVC": kb.SERVICES,
    "STATE": kb.STATES, "STATUS": kb.STATUSES, "TOPIC": list(FAQ), "TEAM": kb.TEAMS,
}


def words(s):
    out = []
    for w in s.split():
        out.append(SLOTS.get(w, w))
    return [x for w in out for x in ([w] if w in CLASSES else T.split(w))]


def R(*xs):  # harness-inserted (result) tokens: context only, never generated
    return [("R", x) for x in xs]


def templates():
    """name -> (decision, token list). Tokens after <bot>; ('R',x) = harness-written."""
    t = {}
    for topic, (_, reply) in FAQ.items():
        t["faq:" + topic] = (("faq", topic), words(reply) + ["<end>"])
    for k in ("hello", "thanks", "bye", "unknown"):
        t["soc:" + k] = ((k,), words(SOC[k]) + ["<end>"])
    t["esc"] = (("esc",), ["<tool>", "create_ticket", "TOPIC", "</tool>",
                           *R("<result>", "ticket", "ID", "</result>"),
                           *words(FL["escalate"]), "<end>"])
    t["recall"] = (("recall",), words(FL["recall"]) + ["<end>"])
    t["status"] = (("status",), ["<tool>", "lookup_ticket", "ID", "</tool>",
                                 *R("<result>", "STATUS", "</result>"),
                                 *words(FL["status"]), "<end>"])
    t["service"] = (("service",), ["<tool>", "service_status", "SVC", "</tool>",
                                   *R("<result>", "STATE", "</result>"),
                                   *words(FL["service"]), "<end>"])
    t["urgent"] = (("urgent",), ["<tool>", "page_team", "TEAM", "</tool>",
                                 *R("<result>", "paged", "</result>"),
                                 *words(FL["urgent"]), "<end>"])
    pre = ["<tool>", "reset_password", "USER", "</tool>", ("R", "<result>")]
    t["reset_sent"] = (("reset",), pre + R("sent", "</result>") + words(FL["reset_sent"])
                       + ["<end>"])
    # unknown users come back as <result> <unk> </result>
    t["reset_unk"] = (("reset",), pre + R("<unk>", "</result>") + words(FL["reset_unknown"])
                      + ["<end>"])
    return t


def plain(x):
    return x[1] if isinstance(x, tuple) else x


# ----------------------------------------------------------------- residual layout
K, KP = 48, 24  # identity code size, prev-code size (first KP dims of the code)
LAYOUT = [("IN", K), ("OUT", K), ("PREV", KP), ("PFL", 7), ("FL", 7), ("ISUSER", 1),
          ("ISBOT", 1), ("ISBOS", 1), ("ISKW", 1), ("KWF", 1), ("FLV", len(FLOWS)),
          ("KWT", 1), ("TPV", 22), ("P", 1), ("P2", 1), ("U", 2), ("INB", 1),
          ("FLOW", len(FLOWS)), ("TL", 22), ("GATE", 5)]
S = {}
_o = 0
for _n, _w in LAYOUT:
    S[_n] = list(range(_o, _o + _w))
    _o += _w
D_MODEL = _o
E = 32
SQ = math.sqrt(E)

# constants
G = 50.0  # output logit scale for generated / copied tokens
A_, B_, DELTA = 10.0, 10.0, 1.0  # MLP unit: decision weight, match weight, firing margin


def code_table():
    """Identity code for every token the bot can see at a generating position or emit."""
    toks = {"<bot>", "<tool>", "</tool>", "<result>", "</result>", "<end>", "<unk>",
            "ticket", "paged", "sent", "error"}
    for _, tl in templates().values():
        toks |= {plain(x) for x in tl if plain(x) not in CLASSES}
    for c in CLASSES:
        toks |= set(CLASS_TOKENS[c])
    toks = sorted(t for t in toks if t in TID)
    g = torch.Generator().manual_seed(50)
    codes = {}
    for t in toks:
        a = torch.randn(KP, generator=g, dtype=torch.float64)
        b = torch.randn(K - KP, generator=g, dtype=torch.float64)
        # first KP dims (prev code) and rest each carry half the norm
        codes[t] = torch.cat([a / a.norm(), b / b.norm()]) / math.sqrt(2)
    return codes


def units():
    """Reply-table units: (decision terms, cur match, prev match, output token)."""
    out, gates = [], []
    for name, (dec, tl) in templates().items():
        seen = {}
        for i in range(len(tl) - 1):
            nxt = tl[i + 1]
            if isinstance(nxt, tuple):
                continue  # harness writes it
            cur = plain(tl[i])
            prev = "<bot>" if i == 0 else plain(tl[i - 1])
            key = (cur, prev)
            if key in seen and seen[key] != nxt:
                raise ValueError(f"collision in {name}: {key}")
            seen[key] = nxt
            pm = prev
            if nxt in COPY:
                gates.append((cur, pm, nxt))
            elif nxt == "TOPIC":
                for tp in FAQ:
                    out.append((dec, ("topic", tp), cur, pm, tp))
            elif nxt == "TEAM":
                for tm in kb.TEAMS:
                    out.append((dec, ("team", tm), cur, pm, tm))
            else:
                out.append((dec, None, cur, pm, nxt))
    # first token of each template: cur=<bot>
    first = []
    for name, (dec, tl) in templates().items():
        first.append((dec, None, "<bot>", None, plain(tl[0])))
    return first + out, sorted(set(gates), key=str)


def _m(a, b):
    return a == b or (b in CLASSES and a in CLASS_TOKENS[b]) or (
        a in CLASSES and b in CLASS_TOKENS[a])


def split_units():
    """Reply units whose (cur, prev) context occurs in only one place across all
    templates need no decision check: they go to the layer-0 MLP (free capacity).
    Returns (layer0 literal units, layer1 decision-gated units, gate units)."""
    tu, gu = units()
    allv = [(c, p, n) for (_, _, c, p, n) in tu] + [(c, p, "GATE:" + k) for c, p, k in gu]
    l0, l1 = [], []
    for u in tu:
        dec, extra, c, p, n = u
        ok = extra is None and p is not None and c not in CLASSES and p not in CLASSES
        if ok:
            for c2, p2, n2 in allv:
                if n2 != n and _m(c, c2) and (p2 is None or _m(p, p2)):
                    ok = False
                    break
        (l0 if ok else l1).append(u)
    # de-duplicate identical shared units in layer 0
    l0 = list({(c, p, n): (d, e, c, p, n) for d, e, c, p, n in l0}.values())
    # balance the two MLPs (both layers pay for d_mlp): surplus literal units go to layer 1
    while len(l0) + len(gu) > len(l1) + 1:
        l1.append(l0.pop())
    return l0, l1, gu


def decision_terms(dec, extra):
    """Linear decision D (value 1 when this reply is chosen, <=0 otherwise)."""
    F = {f: S["FLOW"][i] for i, f in enumerate(FLOWS)}
    topics = list(FAQ)
    terms, const = {}, 0.0

    def add(dim, w):
        terms[dim] = terms.get(dim, 0.0) + w

    k = dec[0]
    if k == "faq":
        add(S["TL"][topics.index(dec[1])], 1)
        add(F["topic"], 1)
        const = -1
    elif k in ("hello", "thanks", "bye"):
        add(F[k], 1)
    elif k == "unknown":
        const = 1
        for f in FLOWS:
            add(F[f], -1)
    elif k == "esc":
        add(F["esc"], 1)
        add(F["escw"], 1)
    else:
        add(F[k], 1)
    if extra is not None:  # slot unit: also require the topic / team
        kind, val = extra
        sel = [val] if kind == "topic" else [t for t in topics if FAQ[t][0] == val]
        for tp in sel:
            add(S["TL"][topics.index(tp)], 1)
        const -= 1
    return terms, const


def build():
    lit_units, tmpl_units, gate_units = split_units()
    codes = code_table()
    n0, n1 = len(gate_units) + len(lit_units), len(tmpl_units)
    cfg = model.Config(d_model=D_MODEL, n_layers=2, n_heads=4, d_head=E,
                       d_mlp=max(n0, n1), pos="learned", tie_embeddings=True,
                       norm=False, bias=True)
    P = {k: torch.zeros(s, dtype=torch.float64) for k, s in model.param_shapes(cfg).items()}
    emb = P["embed"]
    topics = list(FAQ)

    # ---------------- embedding: identity code (in IN and OUT) + hand-set features
    for t, c in codes.items():
        emb[TID[t], S["IN"]] = c
        emb[TID[t], S["OUT"]] = c
    for ci, c in enumerate(CLASSES):
        for t in CLASS_TOKENS[c]:
            if t in TID:
                emb[TID[t], S["FL"][ci]] = 1
    emb[TID["<user>"], S["ISUSER"][0]] = 1
    emb[TID["<bot>"], S["ISBOT"][0]] = 1
    emb[TID["<bos>"], S["ISBOS"][0]] = 1
    for ti, tp in enumerate(topics):
        for w, s in TOPIC_WORDS[tp].items():
            if w in TID:
                r = TID[w]
                emb[r, S["ISKW"][0]] = 1
                emb[r, S["KWT"][0]] = s / 8
                emb[r, S["TPV"]] = 0
                emb[r, S["TPV"][ti]] = 1
    # flow cue words; topic keywords are flow channel "topic"
    role = {}
    for tp in topics:
        for w in TOPIC_WORDS[tp]:
            role[w] = "topic"
    for f in FLOWS:
        for w in FLOW_WORDS.get(f, []):
            if PRIO[f] > PRIO.get(role.get(w, ""), 0):
                role[w] = f
    for w, f in role.items():
        if w in TID:
            emb[TID[w], S["KWF"][0]] = PRIO[f] / 110
            emb[TID[w], S["FLV"]] = 0
            emb[TID[w], S["FLV"][FLOWS.index(f)]] = 1
    # positions: p and p^2 (used for "exact previous position" and recency)
    p = torch.arange(T.MAX_LEN, dtype=torch.float64)
    P["pos"][:, S["P"][0]] = p
    P["pos"][:, S["P2"][0]] = p * p

    def W(layer, name):
        return P[f"layers.{layer}.{name}"]

    # ---------------- layer 0 attention
    # head 0: previous token. score = -10 (j-(t-1))^2 (+const), copies prev code + class flags
    mu = 10.0
    W(0, "wq")[0, S["P"][0], 0] = 2 * mu * SQ
    W(0, "bq")[0, 0] = -2 * mu * SQ
    W(0, "wk")[0, S["P"][0], 0] = 1
    W(0, "bq")[0, 1] = -mu * SQ
    W(0, "wk")[0, S["P2"][0], 1] = 1
    for i in range(KP):
        W(0, "wv")[0, S["IN"][i], i] = 1
        W(0, "wo")[0, i, S["PREV"][i]] = 1
    for c in range(7):
        W(0, "wv")[0, S["FL"][c], KP + c] = 1
        W(0, "wo")[0, KP + c, S["PFL"][c]] = 1
    # head 1: latest <user> -> copy its (p, p^2) = "turn id" U
    # head 2: latest marker (<user> or <bot>) -> copy is_bot = INB (inside a bot turn)
    for h, keyf in ((1, ["ISUSER"]), (2, ["ISUSER", "ISBOT"])):
        W(0, "bq")[h, 0] = 5000 * SQ
        for f in keyf:
            W(0, "wk")[h, S[f][0], 0] = 1
        W(0, "bq")[h, 1] = 10 * SQ
        W(0, "wk")[h, S["P"][0], 1] = 1
    W(0, "wv")[1, S["P"][0], 0] = 1
    W(0, "wv")[1, S["P2"][0], 1] = 1
    W(0, "wo")[1, 0, S["U"][0]] = 1
    W(0, "wo")[1, 1, S["U"][1]] = 1
    W(0, "wv")[2, S["ISBOT"][0], 0] = 1
    W(0, "wo")[2, 0, S["INB"][0]] = 1

    # ---------------- layer 0 MLP: copy triggers (cur, prev) -> GATE[class]
    def match_in(wcol, cur, prev):
        nb = 0
        if cur in CLASSES:
            wcol[S["FL"][CLASSES.index(cur)]] += B_
        else:
            wcol[S["IN"]] += B_ * codes[cur]
        nb += 1
        if prev is not None:
            if prev in CLASSES:
                wcol[S["PFL"][CLASSES.index(prev)]] += B_
            else:
                q = codes[prev][:KP]
                wcol[S["PREV"]] += B_ * q / (q @ q)
            nb += 1
        return nb

    for u, (cur, prev, cls) in enumerate(gate_units):
        nb = match_in(W(0, "w_in")[:, u], cur, prev)
        W(0, "b_in")[u] = DELTA - B_ * nb
        W(0, "w_out")[u, S["GATE"][COPY.index(cls)]] = 1 / DELTA

    # literal units (context unique across all replies): next token, no decision needed
    for j, (_, _, cur, prev, nxt) in enumerate(lit_units):
        u = len(gate_units) + j
        nb = match_in(W(0, "w_in")[:, u], cur, prev)
        W(0, "b_in")[u] = DELTA - B_ * nb
        W(0, "w_out")[u, S["OUT"]] = (G / DELTA) * codes[nxt]

    # ---------------- layer 1 attention
    # head 0: flow. same user turn (-5 (u_j-u_t)^2), priority score, exclude bot turns
    m2 = 5.0
    W(1, "wq")[0, S["U"][0], 0] = 2 * m2 * SQ
    W(1, "wk")[0, S["U"][0], 0] = 1
    W(1, "bq")[0, 1] = -m2 * SQ
    W(1, "wk")[0, S["U"][1], 1] = 1
    W(1, "bq")[0, 2] = 110 * SQ
    W(1, "wk")[0, S["KWF"][0], 2] = 1
    W(1, "bq")[0, 3] = -5000 * SQ
    W(1, "wk")[0, S["INB"][0], 3] = 1
    for i in range(len(FLOWS)):
        W(1, "wv")[0, S["FLV"][i], i] = 1
        W(1, "wo")[0, i, S["FLOW"][i]] = 1
    # head 1: latest topic keyword in any user turn (latest turn first, then strength)
    W(1, "bq")[1, 0] = 3000 * SQ
    W(1, "wk")[1, S["ISKW"][0], 0] = 1
    W(1, "bq")[1, 1] = 10 * SQ
    W(1, "wk")[1, S["U"][0], 1] = 1
    W(1, "bq")[1, 2] = 48 * SQ
    W(1, "wk")[1, S["KWT"][0], 2] = 1
    W(1, "bq")[1, 3] = -5000 * SQ
    W(1, "wk")[1, S["INB"][0], 3] = 1
    for i in range(22):
        W(1, "wv")[1, S["TPV"][i], i] = 1
        W(1, "wo")[1, i, S["TL"][i]] = 1
    # heads 2,3: copy the latest token of the gated class (else attend <bos>, value 0)
    gam, gam0, lam = 3000.0, 2000.0, 5.0
    for h in (2, 3):
        for c in range(5):
            W(1, "wq")[h, S["GATE"][c], c] = gam * SQ
            W(1, "wk")[h, S["FL"][c], c] = 1
            W(1, "wq")[h, S["GATE"][c], 6] = -gam0 * SQ
        W(1, "bq")[h, 5] = lam * SQ
        W(1, "wk")[h, S["P"][0], 5] = 1
        W(1, "bq")[h, 6] = gam0 * SQ
        W(1, "wk")[h, S["ISBOS"][0], 6] = 1
        for i in range(K // 2):
            W(1, "wv")[h, S["IN"][K // 2 * (h - 2) + i], i] = 1
            W(1, "wo")[h, i, S["OUT"][K // 2 * (h - 2) + i]] = G

    # ---------------- layer 1 MLP: reply table
    for u, (dec, extra, cur, prev, nxt) in enumerate(tmpl_units):
        col = W(1, "w_in")[:, u]
        terms, const = decision_terms(dec, extra)
        for dim, w in terms.items():
            col[dim] += A_ * w
        nb = match_in(col, cur, prev)
        W(1, "b_in")[u] = A_ * (const - 1) + DELTA - B_ * nb
        W(1, "w_out")[u, S["OUT"]] = (G / DELTA) * codes[nxt]
    return cfg, P


if __name__ == "__main__":
    l0, l1, gu = split_units()
    print("l0", len(l0), "l1", len(l1), "gates", len(gu))
    miss = sorted({w for d in TOPIC_WORDS.values() for w in d if w not in TID}
                  | {w for f in FLOW_WORDS.values() for w in f if w not in TID})
    print("missing words", miss)
    cd = code_table()
    print("codes", len(cd), "d_model", D_MODEL)
    M = torch.stack(list(cd.values()))
    g = M @ M.T - torch.eye(len(M))
    print("max coherence", g.abs().max().item())
