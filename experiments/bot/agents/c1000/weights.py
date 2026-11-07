"""c1000: a hand-written IT help-desk transformer ("dial + script" design).

Every weight is set by the named constructions below; see EXPLANATION.md.
Short version:
  * Every word the bot can say sits at its own angle on a "dial" (a 2-D unit
    circle, dims CUR).  Writing a word = pointing the OUT dial at its angle.
  * Layer 0 attention looks up: the previous token, where the current user
    turn starts, whether we are inside a user turn, and the mode / topic code
    words the bot wrote in its private <think> notes.
  * Each bot turn starts "<think> MODE [TOPIC] </think>".  MODE and TOPIC are
    picked by a keyword vote over the current user turn (layer 1 "bag" heads,
    read straight out by the unembedding = an argmax).
  * After that the turn is a fixed script.  One MLP unit per script step
    fires on (mode, [topic], [previous word], current word) and points the OUT
    dial at the next word, or opens a "copy gate".
  * Layer 1 copy heads, when their gate is open, copy a ticket id, a user name
    or a tool result from the context onto the OUT dial.
"""

from __future__ import annotations

import math

import torch

from wsp.bot import kb
from wsp.bot import tokenizer as T
from wsp.bot.model import Config

V = T.vocab()
TID = T.token_to_id()
TOPICS = list(kb.FAQ)  # 22 FAQ topics
TOPICS_X = TOPICS + ["internet", "phone"]  # + 2 services that are not FAQ topics
TEAM = {t: kb.FAQ[t][0] for t in TOPICS}

# ---------------------------------------------------------------- private code words
# Rare dictionary words (vocab filler, never used by people) used only inside
# <think>.  One per mode and one per topic, so the bot can "remember" them.
MODES = ["faq", "hello", "thanks", "bye", "unknown", "esc", "recall", "status",
         "service", "urgent", "reset"]
_CW = ("abiotic aborad acanth achime aclys actify adducer adipsy adoral aerator "
       "affinal aftmost agelast aglow agreer ainhum aition akoasma alation aldane "
       "alexia alichel alkali allose almug alpaca amandin ambier amerce aminize "
       "amoke amra amylon ananas andric").split()
MODE_CW = dict(zip(MODES, _CW[:11]))
TOPIC_CW = dict(zip(TOPICS_X, _CW[11:35]))
TMODES = ["faq", "urgent", "service"]  # modes whose topic is chosen by the vote

# ---------------------------------------------------------------- keyword lists (hand written)
# word -> weight, per topic.  Words missing from the vocabulary are skipped.
TOPIC_KW = {
    "password": "password:1.2 pw:1.2 forgot:1 reset:0.5 expired:1 expires:1 change:0.3",
    "locked": "locked:1.5 blocked:1 unlock:1 attempts:1 account:0.5 login:0.5 logins:0.5 logged:0.5",
    "vpn": "vpn:1.5 globalprotect:1.5 cisco:1 remote:0.5 home:0.3 disconnecting:0.5 connect:0.3 connecting:0.3",
    "wifi": "wifi:1.5 wireless:1.5 wi-fi:1.5 network:0.6 dropping:0.5 drops:0.5 connection:0.3 connect:0.3 connecting:0.3",
    "email": "email:1 emails:1 outlook:1.2 mail:1 inbox:1 outbox:1 sending:0.5 calendar:0.8",
    "printer": "printer:1.5 print:1.5 reprint:1 toner:1 paper:0.5 jammed:1",
    "software": "software:1.5 install:1.2 installing:1.2 installed:1 license:1 excel:1 application:0.8 adobe:1 photoshop:1 spreadsheet:0.5 microsoft:0.5 python:1",
    "slow": "slow:1.5 sluggish:1.5 freezing:1.5 freezes:1.5 frozen:1.5 hangs:1 stuck:0.7 unresponsive:1 lagging:1.5 cpu:1 performance:1",
    "mfa": "mfa:1.5 2fa:1.5 authenticator:1.5 sso:0.7 code:1 authentication:1 prompt:0.7 phone:0.6",
    "disk": "disk:1.5 space:1.2 storage:1.2 full:1 large:0.3 drive:0.4",
    "monitor": "monitor:1.5 screen:1 display:1 projector:1.2 docking:1 hdmi:1 cable:0.7 external:0.8 signal:1 second:0.5 detected:0.5 flickers:1 black:0.3 blank:0.5",
    "audio": "audio:1.5 sound:1.5 headset:1.2 speakers:1 microphone:1 mic:1 hear:1 muted:1 crackling:1 loudly:0.5 headphones:0.8",
    "camera": "camera:1.5 webcam:1.5 video:1 image:0.8 picture:0.8 photo:0.5 shutter:1 black:0.3",
    "teams": "teams:1.5 meeting:0.5 meetings:0.5 chat:0.5 calls:0.3 call:0.2",
    "onedrive": "onedrive:1.2 sharepoint:0.8 sync:1 syncing:1 uploading:1 upload:1 pause:0.8 paused:1 cloud:0.5",
    "access": "access:1.2 permission:1 rights:1 approval:0.5 shared:1.5 folder:1 folders:1 denied:1.2 drive:0.6",
    "browser": "browser:1.5 chrome:1.5 edge:1 web:1 site:1 sites:1 websites:1 page:0.5 pages:0.5 cookies:1 loading:0.3",
    "battery": "battery:1.5 batteries:1.5 charging:1.5 charger:1.5 charge:1.2 drains:1 power:0.8 dies:1 outlet:1 unplugged:0.7 plug:0.5",
    "bluetooth": "bluetooth:1.5 pair:2 mouse:1 keyboard:1 keys:0.5 typing:0.5 earbuds:1 receiver:0.7 usb:0.5",
    "phishing": "phishing:2 suspicious:2 fake:1.5 weird:1.2 stranger:1.2 warning:0.8 report:0.5 link:1.5 links:1.5 invoice:0.7",
    "newhire": "hire:2 newhire:2 new:0.4 person:1 starts:1 starter:1.5 employee:1 onboarding:1.5 join:0.8 joining:1 recruit:1.2 laptops:0.6 equipment:0.6",
    "restore": "deleted:1.5 delete:0.6 restore:1.5 recover:1.5 reclaim:1 recycle:1 lost:1 missing:0.8 mistake:1 mistook:1 undo:1 backup:1 accidentally:1 removed:1",
    "internet": "internet:1.5 web:0.5",
    "phone": "phone:0.6",
}
# intent words; FAQ intent additionally gets the sum of all FAQ-topic votes.
INTENT_KW = {
    "hello": "hello:0.9 hi:0.9 hey:0.9 morning:0.9 afternoon:0.9 evening:0.9 good:0.3",
    "thanks": "thanks:1 thank:1 cheers:1 worked:1 works:0.8 great:0.5 awesome:0.5 perfect:0.5 fixed:0.8 welcome:0.5",
    "bye": "bye:1.5 goodbye:1.5 later:1 all:0.7 done:0.7 enough:0.7 see:0.7 night:0.5",
    "esc": "still:2 tried:1.5 help:1 fix:1 fail:1 failed:1 failing:1 fails:1 same:1.2 again:0.5 nothing:1 didn't:1 doesn't:0.5 nope:1 luck:1 worse:1 working:0.7 work:0.7",
    "recall": "number:2 repeat:1 remind:1 remember:1 id:1 ticket:0.7",
    "status": "update:1 status:1",
    "service": "down:2.5 outage:2.5 offline:1.5 up:1 service:1 server:0.5 everyone:0.7",
    "urgent": "urgent:4 asap:4 emergency:4 critical:3",
    "reset": "reset:0.5 user:0.5",
}
UNKNOWN_BASE = 0.6  # the <user> token votes "unknown" with this weight
ID_VOTE, USER_VOTE = 4.0, 4.0  # a ticket id votes "status", a user name votes "reset"
PRIORITY = ["unknown", "bye", "thanks", "hello", "faq", "recall", "esc", "service",
            "status", "reset", "urgent"]  # tiny tie-break bias, later = preferred


def _kw(s):
    return {w: float(x) for w, x in (p.rsplit(":", 1) for p in s.split()) if w in TID}


# ---------------------------------------------------------------- scripts (from the KB)
def _scripts():
    """Concrete token sequences that follow '</think>' for every mode/topic.

    Symbols ID / USER / RV stand for values the bot must copy from context.
    Returns list of (mode, topic_or_None, tokens starting at the think word)."""
    F, out = kb.FLOWS, []
    sp = T.split

    def fill(tpl, **kw):
        toks = []
        for w in tpl.split():
            toks += [kw[w[1:-1]]] if w.startswith("{") else sp(w)
        return toks

    for t in TOPICS:
        out.append(("faq", t, [TOPIC_CW[t], "</think>", *sp(kb.FAQ[t][1]), "<end>"]))
        out.append(("esc", t, [TOPIC_CW[t], "</think>", "<tool>", "create_ticket", t,
                               "</tool>", "</result>",
                               *fill(F["escalate"], id="ID", topic=t, team=TEAM[t]), "<end>"]))
        out.append(("urgent", t, [TOPIC_CW[t], "</think>", "<tool>", "page_team", TEAM[t],
                                  "</tool>", "</result>", *fill(F["urgent"], team=TEAM[t]), "<end>"]))
    for s in kb.SERVICES:
        out.append(("service", s, [TOPIC_CW[s], "</think>", "<tool>", "service_status", s,
                                   "</tool>", "</result>",
                                   *fill(F["service"], service=s, state="RV"), "<end>"]))
    for m, k in [("hello", "hello"), ("thanks", "thanks"), ("bye", "bye"), ("unknown", "unknown")]:
        out.append((m, None, [MODE_CW[m], "</think>", *sp(kb.SOCIAL[k]), "<end>"]))
    out.append(("recall", None, [MODE_CW["recall"], "</think>", *fill(F["recall"], id="ID"), "<end>"]))
    out.append(("status", None, [MODE_CW["status"], "</think>", "<tool>", "lookup_ticket", "ID",
                                 "</tool>", "RV", "</result>",
                                 *fill(F["status"], id="ID", status="RV"), "<end>"]))
    for res, key in [("sent", "reset_sent"), ("<unk>", "reset_unknown")]:
        out.append(("reset", None, [MODE_CW["reset"], "</think>", "<tool>", "reset_password", "USER",
                                    "</tool>", res, "</result>", *fill(F[key], user="USER"), "<end>"]))
    return out


NO_PREDICT = {"</tool>"}  # after </tool> the harness writes <result> ... </result>


def _transitions():
    """(mode, topic, prev, cur, next) for every position where the model predicts."""
    tr = []
    for mode, topic, seq in _scripts():
        for i in range(1, len(seq) - 1):
            prev, cur, nxt = seq[i - 1], seq[i], seq[i + 1]
            if cur in NO_PREDICT or (cur in ("sent", "<unk>", "RV") and nxt == "</result>"):
                continue
            if cur == "</result>":  # prev of </result> is the result value
                prev = seq[i - 1] if seq[i - 1] in ("sent", "<unk>", "RV") else "*"
            tr.append((mode, topic, prev, cur, nxt))
    return tr


def _units():
    """Pick, for each transition, the least specific key that is unambiguous."""
    tr = _transitions()
    levels = [lambda m, t, p, c: (m, None, None, c), lambda m, t, p, c: (m, None, p, c),
              lambda m, t, p, c: (m, t, None, c), lambda m, t, p, c: (m, t, p, c)]
    units = {}
    for m, t, p, c, n in tr:
        for f in levels:
            k = f(m, t, p, c)
            if k[2] == "*":
                continue
            nxts = {n2 for m2, t2, p2, c2, n2 in tr if f(m2, t2, p2, c2) == k}
            if len(nxts) == 1:
                units[k] = n
                break
        else:
            raise ValueError(f"ambiguous transition {(m, t, p, c, n)}")
    return units


# ---------------------------------------------------------------- residual layout
LAYOUT = [("pcos", 1), ("psin", 1), ("plin", 1),            # position
          ("CUR", 2),                                       # dial code of this token
          ("f_user", 1), ("f_bot", 1), ("f_bos", 1),         # boundary flags
          ("f_mode", 1), ("f_topic", 1),                     # private code-word flags
          ("f_id", 1), ("f_uname", 1), ("f_rv", 1),          # copyable value classes
          ("f_gate", 1),                                     # "read the keyword vote here"
          ("INT", len(MODES)), ("TOP", len(TOPICS_X)),       # keyword votes
          ("s_topic", 1), ("s_mode", 1), ("s_xp", 1), ("s_nsvc", 1),  # logit suppressors
          ("U", 2), ("in_user", 1),                          # L0: turn start, inside user turn
          ("PREV", 2), ("p_id", 1), ("p_uname", 1), ("p_rv", 1),     # L0: previous token
          ("MODE", 2), ("TOPC", 2),                          # L0: mode/topic code words
          ("OUT", 2), ("g_id", 1), ("g_uname", 1), ("g_rv", 1)]      # next-word dial, copy gates
D, _o = {}, 0
for _n, _k in LAYOUT:
    D[_n] = slice(_o, _o + _k)
    _o += _k
DM = _o
NH, DH = 6, 12
OMEGA = math.pi / 256  # position angle step
A_POS = 4e5            # sharpness of position matching
K_FLAG, R_REC, B_BOS = 1e5, 256 * 30.0, 1e5
S_OUT = 1e9            # logit scale of the OUT dial
VS = 10.0              # logit scale of keyword votes
SUP = 1e4              # logit suppression


def _coded_tokens():
    toks = set()
    for _, _, seq in _scripts():
        toks.update(x for x in seq if x not in ("ID", "USER", "RV"))
    toks.update(["<bot>", "<think>", *kb.TICKET_IDS, *kb.USERS, *kb.NON_USERS,
                 *kb.STATUSES, *kb.STATES, *MODE_CW.values()])
    return sorted(t for t in toks if t in TID)


def _all_units(units):
    """Universal think-section units + the script units."""
    allu = [((None, None, None, "<bot>"), "<think>"), ("TOPICWORD", "</think>")]
    allu += [((None, None, None, MODE_CW[m]), "</think>") for m in MODES
             if m not in TMODES and m != "esc"]
    allu += [((None, TOPIC_CW[t], None, MODE_CW["esc"]), TOPIC_CW[t]) for t in TOPICS]
    return allu + list(units.items())


def build():
    cfg = Config(d_model=DM, n_layers=2, n_heads=NH, d_head=DH, d_mlp=0, pos="learned",
                 tie_embeddings=False, norm=False, bias=True)
    units = _units()
    allu = _all_units(units)
    dm = (len(allu) + 1) // 2
    cfg = Config(**{**cfg.to_dict(), "d_mlp": dm})
    z = lambda *s: torch.zeros(*s, dtype=torch.float64)  # noqa: E731
    Vn, L = len(V), T.MAX_LEN
    P = {"embed": z(Vn, DM), "pos": z(L, DM), "unembed": z(DM, Vn), "b_unembed": z(Vn)}
    E, U = P["embed"], P["unembed"]
    c = lambda n: D[n].start  # noqa: E731

    # --- dial codes: every coded token gets an angle on the unit circle
    # The angle of a token is fixed by its vocabulary id (2*pi*id/2000), so adding
    # a word later never moves any other word on the dial.
    coded = _coded_tokens()
    N = len(V)
    code = {t: torch.tensor([math.cos(2 * math.pi * TID[t] / N), math.sin(2 * math.pi * TID[t] / N)],
                            dtype=torch.float64) for t in coded}
    delta = 1 - math.cos(2 * math.pi / N)
    S_CODE = 10.0 / delta  # unit sharpness: a neighbour on the dial is 10 below threshold
    for t, v in code.items():
        E[TID[t], D["CUR"]] = v
        U[D["OUT"], TID[t]] = S_OUT * v

    # --- positions: angle OMEGA*t (position 0 = <bos> left at zero) and a linear ramp
    for t in range(1, L):
        P["pos"][t, c("pcos")] = math.cos(OMEGA * t)
        P["pos"][t, c("psin")] = math.sin(OMEGA * t)
    P["pos"][:, c("plin")] = torch.arange(L, dtype=torch.float64) / L

    # --- flags
    E[TID["<user>"], c("f_user")] = 1
    E[TID["<bot>"], c("f_bot")] = 1
    E[TID["<bos>"], c("f_bos")] = 1
    for w in MODE_CW.values():
        E[TID[w], c("f_mode")] = 1
    for w in TOPIC_CW.values():
        E[TID[w], c("f_topic")] = 1
    for w in kb.TICKET_IDS:
        E[TID[w], c("f_id")] = 1
        E[TID[w], c("INT") + MODES.index("status")] += ID_VOTE
    for w in kb.USERS + kb.NON_USERS:
        if w in TID:
            E[TID[w], c("f_uname")] = 1
            E[TID[w], c("INT") + MODES.index("reset")] += USER_VOTE
    for w in kb.STATUSES + kb.STATES:
        E[TID[w], c("f_rv")] = 1
    for w in ["<think>"] + [MODE_CW[m] for m in TMODES]:
        E[TID[w], c("f_gate")] = 1

    # --- keyword votes (embedding rows of user words)
    E[TID["<user>"], c("INT") + MODES.index("unknown")] = UNKNOWN_BASE
    for ti, t in enumerate(TOPICS_X):
        for w, x in _kw(TOPIC_KW[t]).items():
            E[TID[w], c("TOP") + ti] += x
            if t in TOPICS:
                E[TID[w], c("INT") + MODES.index("faq")] += x
    for m, s in INTENT_KW.items():
        for w, x in _kw(s).items():
            E[TID[w], c("INT") + MODES.index(m)] += x

    # --- vote read-out (argmax happens in the logits) + suppressors
    for mi, m in enumerate(MODES):
        U[c("INT") + mi, TID[MODE_CW[m]]] = VS
        P["b_unembed"][TID[MODE_CW[m]]] = 1e-3 * PRIORITY.index(m)
        U[c("s_mode"), TID[MODE_CW[m]]] = -SUP
    for ti, t in enumerate(TOPICS_X):
        U[c("TOP") + ti, TID[TOPIC_CW[t]]] = VS
        P["b_unembed"][TID[TOPIC_CW[t]]] = 1e-4
        U[c("s_topic"), TID[TOPIC_CW[t]]] = -SUP
        if t in ("internet", "phone"):
            U[c("s_xp"), TID[TOPIC_CW[t]]] = -SUP
        if t not in kb.SERVICES:
            U[c("s_nsvc"), TID[TOPIC_CW[t]]] = -SUP
    E[TID["<think>"], c("s_topic")] = 1          # choosing MODE: no topic words
    for m in TMODES:                              # choosing TOPIC: no mode words
        E[TID[MODE_CW[m]], c("s_mode")] = 1
    E[TID[MODE_CW["faq"]], c("s_xp")] = 1
    E[TID[MODE_CW["urgent"]], c("s_xp")] = 1
    E[TID[MODE_CW["service"]], c("s_nsvc")] = 1

    sq = math.sqrt(DH)

    def head(layer, h, q, k, v, bq=None, bk=None):
        """q,k: list of (residual dim, coeff) per head column; v: list of (src, dst)."""
        p = f"layers.{layer}."
        for j, terms in enumerate(q):
            for dim, x in terms:
                P[p + "wq"][h, dim, j] += x * sq
        for j, terms in enumerate(k):
            for dim, x in terms:
                P[p + "wk"][h, dim, j] += x
        if bq:
            for j, x in bq.items():
                P[p + "bq"][h, j] += x * sq
        if bk:
            for j, x in bk.items():
                P[p + "bk"][h, j] += x
        for j, (src, dst) in enumerate(v):
            P[p + "wv"][h, src, j] = 1
            P[p + "wo"][h, j, dst] = 1

    for li in range(2):
        p = f"layers.{li}."
        for w in ("wq", "wk", "wv"):
            P[p + w] = z(NH, DM, DH)
        P[p + "wo"] = z(NH, DH, DM)
        for b in ("bq", "bk", "bv"):
            P[p + b] = z(NH, DH)
        P[p + "bo"] = z(DM)
        P[p + "w_in"], P[p + "w_out"] = z(DM, dm), z(dm, DM)
        P[p + "b_in"], P[p + "b_out"] = z(dm), z(DM)

    co, si = math.cos(OMEGA), math.sin(OMEGA)
    pc, ps, pl = c("pcos"), c("psin"), c("plin")

    def recent(flag):
        """'most recent position carrying flag' (bos is the fallback)."""
        return dict(q=[[], []], k=[[(c(flag), 1)], [(pl, 1)]], bq={0: K_FLAG, 1: R_REC})

    # ---- layer 0 attention
    # H0 previous token: q = A*(angle of t-1), k = angle of s  -> peak at s = t-1
    head(0, 0, q=[[(pc, A_POS * co), (ps, A_POS * si)], [(pc, -A_POS * si), (ps, A_POS * co)]],
         k=[[(pc, 1)], [(ps, 1)]],
         v=[(c("CUR"), c("PREV")), (c("CUR") + 1, c("PREV") + 1), (c("f_id"), c("p_id")),
            (c("f_uname"), c("p_uname")), (c("f_rv"), c("p_rv"))])
    # H1 most recent <user>: copy its position angle -> U (turn start)
    head(0, 1, **recent("f_user"), v=[(pc, c("U")), (ps, c("U") + 1)])
    # H2 most recent of <user>/<bot>/<bos>: copy "was it <user>" -> in_user
    head(0, 2, q=[[], []], k=[[(c("f_user"), 1), (c("f_bot"), 1), (c("f_bos"), 1)], [(pl, 1)]],
         bq={0: K_FLAG, 1: R_REC}, v=[(c("f_user"), c("in_user"))])
    # H3 most recent mode code word -> MODE ;  H4 most recent topic code word -> TOPC
    head(0, 3, q=[[], []], k=[[(c("f_mode"), 1), (c("f_bos"), 0.5)], [(pl, 1)]],
         bq={0: K_FLAG, 1: R_REC}, v=[(c("CUR"), c("MODE")), (c("CUR") + 1, c("MODE") + 1)])
    head(0, 4, q=[[], []], k=[[(c("f_topic"), 1), (c("f_bos"), 0.5)], [(pl, 1)]],
         bq={0: K_FLAG, 1: R_REC}, v=[(c("CUR"), c("TOPC")), (c("CUR") + 1, c("TOPC") + 1)])

    # ---- layer 1 attention
    # H0-2 keyword bag: average of current-user-turn tokens, only where f_gate = 1
    G_OUT, H_BOS = 1e3, 1e3
    vote_dims = list(range(c("INT"), D["TOP"].stop))
    for h in range(3):
        dims = vote_dims[h * DH:(h + 1) * DH]
        head(1, h,
             q=[[(c("U"), A_POS)], [(c("U") + 1, A_POS)], [], [(c("f_gate"), -2 * A_POS)]],
             k=[[(c("U"), 1)], [(c("U") + 1, 1)], [(c("in_user"), 1)], [(c("f_bos"), 1)]],
             bq={2: G_OUT, 3: 2 * A_POS - H_BOS}, bk={2: -1},
             v=[(d_, d_) for d_ in dims])
    # H3-5 copy heads: most recent id / user name / result value, only when gated
    for h, (g, f) in zip((3, 4, 5), [("g_id", "f_id"), ("g_uname", "f_uname"), ("g_rv", "f_rv")]):
        head(1, h, q=[[(c(g), K_FLAG)], [], [(c(g), -B_BOS)]],
             k=[[(c(f), 1)], [(pl, 1)], [(c("f_bos"), 1)]], bq={1: R_REC, 2: B_BOS},
             v=[(c("CUR"), c("OUT")), (c("CUR") + 1, c("OUT") + 1)])

    # ---- MLP units (layer 0 first; extra units spill into layer 1)
    cls = {"ID": ("f_id", "p_id", "g_id"), "USER": ("f_uname", "p_uname", "g_uname"),
           "RV": ("f_rv", "p_rv", "g_rv")}
    gate_units = [u for u in allu if u[1] in cls]
    rest = [u for u in allu if u[1] not in cls]
    order = gate_units + rest
    assert len(order) <= 2 * dm
    for i, (key, nxt) in enumerate(order):
        li, j = (0, i) if i < dm else (1, i - dm)
        p = f"layers.{li}."
        W, tot = P[p + "w_in"], 0.0
        if key == "TOPICWORD":
            W[c("f_topic"), j] = 2
            tot = 2
        else:
            m, t, pv, cu = key
            if m is not None:
                W[D["MODE"], j] += S_CODE * code[MODE_CW[m]]
                tot += S_CODE
            if t is not None:
                tw = t if t in TOPIC_CW.values() else TOPIC_CW[t]
                W[D["TOPC"], j] += S_CODE * code[tw]
                tot += S_CODE
            if pv is not None:
                if pv in cls:
                    W[c(cls[pv][1]), j] += 2
                    tot += 2
                else:
                    W[D["PREV"], j] += S_CODE * code[pv]
                    tot += S_CODE
            if cu in cls:
                W[c(cls[cu][0]), j] += 2
                tot += 2
            else:
                W[D["CUR"], j] += S_CODE * code[cu]
                tot += S_CODE
        P[p + "b_in"][j] = -(tot - 1)
        if nxt in cls:
            P[p + "w_out"][j, c(cls[nxt][2])] = 1
        else:
            P[p + "w_out"][j, D["OUT"]] = code[nxt]
    return cfg, P
