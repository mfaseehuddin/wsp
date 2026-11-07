"""c10: a hand-written 2-layer transformer IT help-desk bot ("script player").

Idea in one paragraph: every bot turn is a *script*. The bot first writes a
private codeword into its think area that names the script (K) and, for FAQ /
urgent turns, a second codeword naming the topic (T). From then on each next
token is looked up by an MLP unit keyed on (script K, topic T, current token,
previous token); copy slots (ticket id, user, service, state, status) are
filled by a gated attention head, and topic / team slots are written from the
topic codeword. All facts come from the knowledge base (`kb`), and the keyword
list in KEYWORDS is hand-written. See EXPLANATION.md for the edit map.
"""

from __future__ import annotations

import math

import torch

from wsp.bot import kb
from wsp.bot import tokenizer as T
from wsp.bot.model import Config

V = T.vocab_size()
VOC = set(T.vocab())

# ------------------------------------------------------------------ the spec
FAQ = {t: (team, T.split(r)) for t, (team, r) in kb.FAQ.items()}
TOPICS = list(FAQ)
SOCIAL = ["hello", "thanks", "bye", "unknown"]
INTENTS = ["faq", *SOCIAL, "esc", "recall", "status", "service", "page", "reset"]
# private think vocabulary: obscure filler words that never occur in user text
_CODE = ("jackleg jajman jantu jass jiltee jiva joinant joskin joyancy jungli "
         "kairine kana kareeta katun keeler kelp keratto ketose khanjar kiby "
         "kilhig kirkify kitter knape knived knutty koko koppa kraal kudzu "
         "kuskite quale quatre").split()
K_WORD = dict(zip(INTENTS, _CODE[: len(INTENTS)]))          # 11 script codewords
T_WORD = dict(zip(TOPICS, _CODE[len(INTENTS):]))            # 22 topic codewords
CODEWORDS = list(K_WORD.values()) + list(T_WORD.values())   # = the VOTE axis

CLASSES = {
    "ID": list(kb.TICKET_IDS), "USER": list(kb.USERS) + list(kb.NON_USERS),
    "SERVICE": list(kb.SERVICES), "STATE": list(kb.STATES),
    "STATUS": list(kb.STATUSES), "TOPIC": TOPICS, "TEAM": list(kb.TEAMS),
}
COPY_CLASSES = ["ID", "USER", "SERVICE", "STATE", "STATUS"]  # copied by attention
SLOT = {"{id}": "{ID}", "{topic}": "{TOPIC}", "{team}": "{TEAM}", "{user}": "{USER}",
        "{status}": "{STATUS}", "{service}": "{SERVICE}", "{state}": "{STATE}"}

# Hand-written keyword votes: word -> [(script or topic, weight)].
# Topic words vote for the topic codeword and (same weight) for the FAQ script.
TOPIC_KW = {
    "password": "password 2 forgot 2 expired 1 passcode 2",
    "locked": "locked 3 lock 3 unlock 3 lockout 3 attempts 1 account 1",
    "vpn": "vpn 3 tunnel 2 remote 1",
    "wifi": "wifi 3 wireless 3 network 1 hotspot 2",
    "email": "email 2 emails 2 outlook 3 mail 3 inbox 3",
    "printer": "printer 3 print 3 printing 3 toner 3 paper 2",
    "software": "software 3 install 3 excel 2 license 2 application 2 program 2 app 1",
    "slow": "slow 3 freezing 3 freezes 3 frozen 3 sluggish 3 lagging 3 hangs 2",
    "mfa": "mfa 3 authenticator 3 code 2 phone 1 2fa 3 approve 1",
    "disk": "disk 3 space 3 storage 2 full 2",
    "monitor": "monitor 3 screen 3 display 3 signal 2 external 1",
    "audio": "sound 3 audio 3 headset 3 hear 3 speaker 3 speakers 3 microphone 3 mic 3 calls 1 headphones 1",
    "camera": "camera 3 webcam 4 video 2 image 1",
    "teams": "teams 3 meeting 1 chat 1",
    "onedrive": "onedrive 3 sync 3 syncing 3 uploading 2",
    "access": "access 3 denied 3 permission 3 permissions 3 shared 2 folder 2 drive 1",
    "browser": "browser 3 chrome 3 firefox 3 edge 2 website 2 cookies 3 page 1",
    "battery": "battery 3 charging 3 charge 3 charger 3 power 2 drains 3 dies 2",
    "bluetooth": "bluetooth 3 pair 3 pairing 3 mouse 2 keyboard 2 headphones 2",
    "phishing": "phishing 4 suspicious 4 weird 3 link 3 scam 4 spam 3",
    "newhire": "hire 3 new 1 person 2 starts 2 starter 3 onboarding 3 joining 3 employee 2",
    "restore": "deleted 4 recover 4 restore 4 mistake 2 recycle 2 lost 2",
}
INTENT_KW = {
    "hello": "hello 0.8 hi 0.8 hey 0.8 morning 0.8 afternoon 0.8 evening 0.8 greetings 0.8",
    "thanks": "thanks 5 thank 5 worked 5 cheers 5 fixed 5",
    "bye": "bye 7 goodbye 7 all 3",
    "esc": "still 10 tried 8 help 6 didn't 6 nope 6 persists 10 worse 6",
    "recall": "number 10 repeat 8 remind 8 ticket 3",
    "service": "down 10 outage 10 status 8 up 6 service 6",
    "page": "urgent 20 asap 20 emergency 20 immediately 15 critical 15",
}


def keyword_votes() -> dict[str, list[tuple[str, float]]]:
    kw: dict[str, list] = {}
    for topic, s in TOPIC_KW.items():
        p = s.split()
        for w, x in zip(p[::2], p[1::2]):
            kw.setdefault(w, []).extend([(T_WORD[topic], float(x)), (K_WORD["faq"], float(x))])
    for it, s in INTENT_KW.items():
        p = s.split()
        for w, x in zip(p[::2], p[1::2]):
            kw.setdefault(w, []).append((K_WORD[it], float(x)))
    for u in CLASSES["USER"]:      # a user name means "reset that password"
        kw.setdefault(u, []).append((K_WORD["reset"], 30.0))
    for i in CLASSES["ID"]:        # a ticket number means "look that ticket up"
        kw.setdefault(i, []).append((K_WORD["status"], 25.0))
    return {w: v for w, v in kw.items() if w in VOC}


# ------------------------------------------------------------------ scripts
def _toks(text):
    return [SLOT[w] if w in SLOT else x for w in text.split() for x in ([w] if w in SLOT else T.split(w))]


def scripts() -> list[tuple[str, str | None, list[str]]]:
    """(script, topic, tokens after <bot>) for every bot-turn shape."""
    F = {k: _toks(v) for k, v in kb.FLOWS.items()}
    S = {k: T.split(v) for k, v in kb.SOCIAL.items()}
    K = K_WORD
    out = [("faq", t, ["<think>", K["faq"], T_WORD[t], "</think>", *r, "<end>"])
           for t, (_, r) in FAQ.items()]
    out += [(s, None, ["<think>", K[s], "</think>", *S[s], "<end>"]) for s in SOCIAL]
    out.append(("recall", None, ["<think>", K["recall"], "</think>", *F["recall"], "<end>"]))

    def tool(it, name, arg, res, reply, topic=None):
        head = ["<think>", K[it], *([T_WORD[topic]] if topic else []), "</think>"]
        return (it, topic, [*head, "<tool>", name, arg, "</tool>", "<result>", *res,
                            "</result>", *reply, "<end>"])

    out.append(tool("esc", "create_ticket", "{TOPIC}", ["ticket", "{ID}"], F["escalate"]))
    out.append(tool("status", "lookup_ticket", "{ID}", ["{STATUS}"], F["status"]))
    out.append(tool("service", "service_status", "{SERVICE}", ["{STATE}"], F["service"]))
    for t in TOPICS:
        out.append(tool("page", "page_team", "{TEAM}", ["paged"], F["urgent"], t))
    out.append(tool("reset", "reset_password", "{USER}", ["sent"], F["reset_sent"]))
    out.append(tool("reset", "reset_password", "{USER}", ["<unk>"], F["reset_unknown"]))
    return out


SPECIAL_FLAG = {"<bos>": "bos", "<user>": "user", "<bot>": "bot", "<think>": "think",
                "</think>": "/think", "<tool>": "tool", "</tool>": "/tool",
                "</result>": "/result", "<result>": "result"}
FLAGS = ["user", "bos", "bot", "think", "/think", "tool", "/tool", "result", "/result",
         "isK", "isT", *CLASSES]


def class_flags(w):
    return {c for c, m in CLASSES.items() if w in m}


# ------------------------------------------------------------------ colours
def colouring(scr):
    """Give every output word a (colour c, group g) pair, unique per word.
    Inside one script all fixed words get different colours, and the words
    that can fill a slot avoid those colours. Then (script, colour of current
    token, colour of previous token) identifies the next token exactly."""
    words = set()
    for _, _, s in scr:
        words |= {w for w in s if not w.startswith("{") and w not in SPECIAL_FLAG
                  and w not in CODEWORDS and w != "<unk>"}
    for c in CLASSES.values():
        words |= set(c)
    words |= {"<think>", "</think>", "<tool>", "</tool>", "<end>"}
    conflict = {w: set() for w in words}
    for _, _, s in scr:
        fixed = {w for w in s if w in conflict and not w.startswith("<")}
        fill = set().union(*[set(CLASSES[w[1:-1]]) for w in s if w.startswith("{")])
        for a in fixed:
            conflict[a] |= fixed - {a}
            conflict[a] |= fill - {a}
        for a in fill - fixed:
            conflict[a] |= fixed - {a}
    fillers = set().union(*CLASSES.values())
    order = sorted(words, key=lambda w: (w in fillers, -len(conflict[w]), w))
    best = None
    for extra in range(0, 30):
        col, use, C1 = {}, {}, 0
        for w in order:
            bad = {col[x] for x in conflict[w] if x in col}
            if w not in fillers:
                c = min(c for c in range(999) if c not in bad)
                C1 = max(C1, c + 1)
            else:
                ok = [c for c in range(C1 + extra) if c not in bad] or [
                    min(c for c in range(999) if c not in bad)]
                c = min(ok, key=lambda c: (use.get(c, 0), c))
            col[w], use[c] = c, use.get(c, 0) + 1
        cost = len(use) + max(use.values())
        if best is None or cost < best[0]:
            best = (cost, col)
    col = best[1]
    used = sorted(set(col.values()))
    remap = {c: i for i, c in enumerate(used)}
    code, cnt = {}, {}
    for w in sorted(words):
        c = remap[col[w]]
        code[w] = (c, cnt.get(c, 0))
        cnt[c] = cnt.get(c, 0) + 1
    return code, len(used), max(cnt.values())


# ------------------------------------------------------------------ units
def feats(w, code):
    """Symbolic features of a token at the input."""
    if w in SPECIAL_FLAG:
        return {SPECIAL_FLAG[w]}
    if w in K_WORD.values():
        return {"isK"}
    if w in T_WORD.values():
        return {"isT"}
    f = class_flags(w)
    if w in code:
        f.add(("c", code[w][0]))
    return f


def primary(w, code):
    if w.startswith("{"):
        return w[1:-1]
    if w in SPECIAL_FLAG:
        return SPECIAL_FLAG[w]
    if w in K_WORD.values():
        return "isK"
    if w in T_WORD.values():
        return "isT"
    return ("c", code[w][0])


def units(scr, code):
    """MLP units: (conditions, action). action = ('out', word) | ('allow', class)
    | ('suppress', 'K'|'T'). Conditions are ANDed."""
    table = {
        frozenset({"bot"}): ("out", "<think>"),                 # <bot> -> <think>
        frozenset({"think"}): ("suppress", "T"),                # choose a script
        frozenset({"isK", ("K", "faq")}): ("suppress", "K"),    # choose a topic
        frozenset({"isK", ("K", "page")}): ("suppress", "K"),
    }
    for it, topic, s in scr:
        seq = ["<bot>", *s]
        model_pos, inside = [], False
        for i in range(len(seq) - 1):
            if seq[i] == "<result>":
                inside = True
            if seq[i] == "</result>":
                inside = False
            if i >= 2 and not inside and seq[i] != "</tool>":
                model_pos.append(i)
        for i in model_pos:
            if i == 2 and it in ("faq", "page"):
                continue  # topic is chosen by the vote, not by a unit
            nxt = seq[i + 1]
            if seq[i - 1] == "<unk>":  # result "unknown" is not in the vocabulary
                cond = {("K", it), primary(seq[i], code), ("not", ("p", primary("sent", code)))}
            else:
                cond = {("K", it), primary(seq[i], code), ("p", primary(seq[i - 1], code))}
            if topic and seq[i] != K_WORD[it] and it == "faq":
                cond.add(("T", topic))
            if it == "page" and seq[i] == T_WORD[topic]:
                cond = {("K", it), "isT", ("p", "isK")}
            act = ("allow", nxt[1:-1]) if nxt.startswith("{") else ("out", nxt)
            key = frozenset(cond)
            if table.get(key, act) != act:
                raise ValueError(f"conflict {it} {topic} {seq[i]} -> {nxt} vs {table[key]}")
            table[key] = act
    return table


# ------------------------------------------------------------------ build
OMEGA = 2 * math.pi / 300   # one rotation frequency for all positional tricks
M = 100.0                   # logit size of a chosen word


def build():
    scr = scripts()
    code, C, G = colouring(scr)
    tab = units(scr, code)
    kw = keyword_votes()
    nK, nT = len(K_WORD), len(T_WORD)
    nV = nK + nT

    # ---- residual layout (named slices) ----
    L, o = {}, 0
    for name, n in [("POS", 3), ("TURN", 2), ("CURC", C), ("FL", len(FLAGS)),
                    ("PREV", C + len(FLAGS)), ("KCUR", nK), ("TCUR", nT), ("VOTE", nV),
                    ("OUT", C + G), ("SELFG", G), ("ALLOW", len(CLASSES))]:
        L[name], o = (o, n), o + n
    d = o
    ix = lambda name, j=0: L[name][0] + j  # noqa: E731
    fl = lambda f: ix("FL", FLAGS.index(f))  # noqa: E731
    E_ = max(C + len(FLAGS), C + G, nV, 8)
    # suppress units must sit in layer 1, after the vote head has read the
    # keys; otherwise their -50 would leak into later votes of the same turn.
    trans = [(k, a) for k, a in tab.items() if a[0] == "out"]
    supp = [(k, a) for k, a in tab.items() if a[0] == "suppress"]
    allows = [(k, a) for k, a in tab.items() if a[0] == "allow"]
    gate = [("TOPIC", j) for j in range(nT)] + [("TEAM", j) for j in range(nT)]
    n0 = len(allows) + (len(trans) + len(supp) + len(gate) - len(allows) + 1) // 2
    mlp0 = allows + trans[: n0 - len(allows)]
    mlp1 = trans[n0 - len(allows):] + supp
    dm = max(len(mlp0), len(mlp1) + len(gate))
    cfg = Config(d_model=d, n_layers=2, n_heads=4, d_head=E_, d_mlp=dm,
                 pos="learned", tie_embeddings=False, norm=False, bias=True)
    z = lambda *s: torch.zeros(*s, dtype=torch.float64)  # noqa: E731
    P = {"embed": z(V, d), "pos": z(T.MAX_LEN, d), "unembed": z(d, V), "b_unembed": z(V)}
    for li in range(2):
        p = f"layers.{li}."
        for w in ("wq", "wk", "wv"):
            P[p + w] = z(4, d, E_)
        P[p + "wo"] = z(4, E_, d)
        for b in ("bq", "bk", "bv"):
            P[p + b] = z(4, E_)
        P[p + "bo"] = z(d)
        P[p + "w_in"], P[p + "w_out"] = z(d, dm), z(dm, d)
        P[p + "b_in"], P[p + "b_out"] = z(dm), z(d)
    tid = T.tid
    vidx = {w: j for j, w in enumerate(CODEWORDS)}

    # ---- embedding: token features ----
    Em = P["embed"]
    for w in VOC:
        r = tid(w)
        for f in feats(w, code):
            if isinstance(f, tuple):
                Em[r, ix("CURC", f[1])] = 1.0
            else:
                Em[r, fl(f)] = 1.0
        if any(w in CLASSES[c] for c in COPY_CLASSES) and w in code:
            Em[r, ix("SELFG", code[w][1])] = 1.0
        if w in vidx:                                  # codeword identity
            Em[r, ix("VOTE", vidx[w])] = 1.0
    Em[tid("<bos>"), fl("user")] = 1.0
    for w, vs in kw.items():                           # keyword votes
        for target, x in vs:
            Em[tid(w), ix("VOTE", vidx[target])] += x
    # ---- positions: a rotation (cos, sin) and a ramp t/256 ----
    t = torch.arange(T.MAX_LEN, dtype=torch.float64)
    P["pos"][:, ix("POS", 0)] = torch.cos(OMEGA * t)
    P["pos"][:, ix("POS", 1)] = torch.sin(OMEGA * t)
    P["pos"][:, ix("POS", 2)] = t / 256
    rt = math.sqrt(E_)
    c0, s0, u0 = ix("POS", 0), ix("POS", 1), ix("POS", 2)

    def recent_head(p, h, flag, src, dst, n):
        """Attend to the most recent token carrying `flag` (else <bos>)."""
        P[p + "bq"][h, 0], P[p + "bq"][h, 1], P[p + "bq"][h, 2] = 4000 * rt, 5 * rt, 2000 * rt
        P[p + "wk"][h, fl(flag), 0] = 1.0
        P[p + "wk"][h, u0, 1] = 256.0
        P[p + "wk"][h, fl("bos"), 2] = 1.0
        for j in range(n):
            P[p + "wv"][h, src + j, j] = 1.0
            P[p + "wo"][h, j, dst + j] = 1.0

    # L0 H0: previous-token head (copies colour + flags of token t-1)
    p, A = "layers.0.", 1.5e5 * rt
    P[p + "wq"][0, c0, 0], P[p + "wq"][0, s0, 0] = A * math.cos(OMEGA), A * math.sin(OMEGA)
    P[p + "wq"][0, c0, 1], P[p + "wq"][0, s0, 1] = -A * math.sin(OMEGA), A * math.cos(OMEGA)
    P[p + "wk"][0, c0, 0], P[p + "wk"][0, s0, 1] = 1.0, 1.0
    for j in range(C + len(FLAGS)):
        P[p + "wv"][0, ix("CURC") + j, j] = 1.0
        P[p + "wo"][0, j, ix("PREV") + j] = 1.0
    # L0 H1: turn head (where did the current user turn start?)
    P[p + "bq"][1, 0], P[p + "bq"][1, 1] = 1000 * rt, 5 * rt
    P[p + "wk"][1, fl("user"), 0], P[p + "wk"][1, u0, 1] = 1.0, 256.0
    P[p + "wv"][1, c0, 0], P[p + "wv"][1, s0, 1] = 1.0, 1.0
    P[p + "wo"][1, 0, ix("TURN")], P[p + "wo"][1, 1, ix("TURN", 1)] = 1.0, 1.0
    # L0 H2 / H3: current script codeword K and topic codeword T
    recent_head(p, 2, "isK", ix("VOTE"), ix("KCUR"), nK)
    recent_head(p, 3, "isT", ix("VOTE", nK), ix("TCUR"), nT)

    # L1 H0: vote head = average of keyword votes over the current turn
    p = "layers.1."
    P[p + "wq"][0, ix("TURN"), 0] = P[p + "wq"][0, ix("TURN", 1), 1] = 2000 * rt
    P[p + "wk"][0, ix("TURN"), 0] = P[p + "wk"][0, ix("TURN", 1), 1] = 1.0
    for j in range(nV):
        P[p + "wv"][0, ix("VOTE", j), j] = 1.0
        P[p + "wo"][0, j, ix("VOTE", j)] = 1.0
    # L1 H1: gated copy head: when ALLOW[X] is on, copy the latest class-X token
    for j, cl in enumerate(COPY_CLASSES):
        P[p + "wq"][1, ix("ALLOW", list(CLASSES).index(cl)), j] = 10000 * rt
        P[p + "wk"][1, fl(cl), j] = 1.0
    P[p + "bq"][1, 5], P[p + "bq"][1, 6] = 5000 * rt, 5 * rt
    P[p + "wk"][1, fl("bos"), 5], P[p + "wk"][1, u0, 6] = 1.0, 256.0
    for j in range(C):
        P[p + "wv"][1, ix("CURC", j), j] = 1.0
        P[p + "wo"][1, j, ix("OUT", j)] = M
    for j in range(G):
        P[p + "wv"][1, ix("SELFG", j), C + j] = 1.0
        P[p + "wo"][1, C + j, ix("OUT", C + j)] = M

    # ---- MLP units ----
    def cdim(c):
        if isinstance(c, tuple) and c[0] == "K":
            return ix("KCUR", INTENTS.index(c[1]))
        if isinstance(c, tuple) and c[0] == "T":
            return ix("TCUR", TOPICS.index(c[1]))
        if isinstance(c, tuple) and c[0] == "p":
            q = c[1]
            return ix("PREV", q[1]) if isinstance(q, tuple) else ix("PREV", C + FLAGS.index(q))
        if isinstance(c, tuple) and c[0] == "c":
            return ix("CURC", c[1])
        return fl(c)

    def outvec(word):
        v = z(d)
        c, g = code[word]
        v[ix("OUT", c)] = v[ix("OUT", C + g)] = 2 * M
        return v

    def put(li, u, conds, act):
        p = f"layers.{li}."
        pos = [c for c in conds if not (isinstance(c, tuple) and c[0] == "not")]
        for c in pos:
            P[p + "w_in"][cdim(c), u] = 1.0
        for c in set(conds) - set(pos):
            P[p + "w_in"][cdim(c[1]), u] = -1.0
        P[p + "b_in"][u] = -(len(pos) - 0.5)
        if act[0] == "out":
            P[p + "w_out"][u] = outvec(act[1])
        elif act[0] == "allow":
            P[p + "w_out"][u, ix("ALLOW", list(CLASSES).index(act[1]))] = 2.0
        else:  # suppress the K or T half of the vote logits
            a, n = (ix("VOTE"), nK) if act[1] == "K" else (ix("VOTE", nK), nT)
            P[p + "w_out"][u, a:a + n] = -100.0

    for u, (k, a) in enumerate(mlp0):
        put(0, u, k, a)
    for u, (k, a) in enumerate(mlp1):
        put(1, u, k, a)
    for j, (cl, tj) in enumerate(gate):            # topic / team slot writers
        u = len(mlp1) + j
        P["layers.1.w_in"][ix("ALLOW", list(CLASSES).index(cl)), u] = 1.0
        P["layers.1.w_in"][ix("TCUR", tj), u] = 1.0
        P["layers.1.b_in"][u] = -1.5
        topic = TOPICS[tj]
        P["layers.1.w_out"][u] = outvec(topic if cl == "TOPIC" else FAQ[topic][0])

    # ---- unembedding ----
    for w, (c, g) in code.items():
        P["unembed"][ix("OUT", c), tid(w)] = 1.0
        P["unembed"][ix("OUT", C + g), tid(w)] = 1.0
    P["b_unembed"][:] = -1.0
    for w, j in vidx.items():
        P["unembed"][ix("VOTE", j), tid(w)] = 1.0
        P["b_unembed"][tid(w)] = 0.0
    P["b_unembed"][tid(K_WORD["unknown"])] = 0.005   # wins only if nothing votes
    return cfg, P
