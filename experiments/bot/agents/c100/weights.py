"""c100: hand-written weights for the IT help-desk bot ("register + ramp" design).

Two layers, no training. See EXPLANATION.md for the plain-language story.

Residual stream layout (named slices, see DIMS):
  CODE  48  token identity code (embedding) == output code (tied unembedding)
  VOTE  34  keyword votes of a token; on register tokens: one-hot intent id
  I     34  intent of the current bot turn (copied from latest register token)
  flags     f_bos f_user f_bot f_think f_areg f_treg f_tid f_status f_state f_name f_svc
  P1        position t (from learned pos table);  SU / SB: position of latest <user>/<bot>
  dirty     1 where layer-0 MLP wrote an output (such positions are never copy sources)
  gates     g_* : which copy channel is open at this step
  TEAM5 / SVC8  payloads: team of a topic register, canonical service of a service word
Layer 0: attn  {SU head, SB head, intent-reader head};  MLP  ramp tables (copy intents, social, some FAQs)
Layer 1: attn  {vote head, copy-ID head, copy-payload head};  MLP  intent readout + rest of FAQ tables
"""

from __future__ import annotations

import math

import torch

from wsp.bot import kb
from wsp.bot import tokenizer as T
from wsp.bot.model import Config

# ----------------------------------------------------------------- spec: intents
TOPICS = list(kb.FAQ)  # 22 FAQ topics, KB order
ACTIONS = ["hello", "thanks", "bye", "unknown", "escalate", "recall", "status",
           "service", "urgent", "reset", "reset_sent", "reset_unknown"]
INTENTS = TOPICS + ACTIONS  # VOTE / I index order
NI = len(INTENTS)
# register tokens written in <think>: obscure filler words never used by people
REG_WORDS = ("abbassi affinal alumina athenee banago begloom byrlaw cavish chorion cyanus "
             "deplume dittay duncery flavic foxbane gelilah hunkies inswept joskin knutty "
             "leptid mathes mimmest neolith outhold pinweed prating puntel relevel retrue "
             "rouleau saveloy").split()
REG = dict(zip(INTENTS[:32], REG_WORDS))
REG["reset_sent"], REG["reset_unknown"] = "sent", "<unk>"  # tool results act as registers

# ----------------------------------------------------------------- hand-written keywords
# weight units: 2 core word, 1 secondary, 3 action word, 5 name/ticket id. Scaled by 0.1.
KW = {
    "password": "password:2 forgot:1 expired:1 reset:1 passcode:1",
    "locked": "locked:2 lockout:2 unlock:2 attempts:1 account:0.5",
    "vpn": "vpn:2 tunnel:1 anyconnect:2 remote:0.5",
    "wifi": "wifi:2 wi-fi:2 wireless:2 ssid:2 hotspot:1 internet:0.5",
    "email": "email:1 outlook:2 mail:1 inbox:1 emails:1",
    "printer": "printer:2 print:2 printing:2 toner:2 scanner:1 printers:2",
    "software": "software:2 install:2 excel:1 license:1 installed:1 program:1 adobe:1 office:1 application:1",
    "slow": "slow:2 freezing:2 freezes:2 frozen:2 sluggish:2 lag:2 laggy:2 hangs:1 freeze:2 slowly:2",
    "mfa": "mfa:2 authenticator:2 2fa:2 code:1 otp:2 verification:1 phone:1",
    "disk": "disk:2 space:2 storage:2 full:1",
    "monitor": "monitor:2 screen:2 display:2 signal:1 external:1 hdmi:2 projector:2 monitors:2",
    "audio": "sound:2 audio:2 hear:2 microphone:2 mic:2 speaker:2 speakers:2 headset:1 headphones:1 volume:2 mute:1",
    "camera": "camera:2 webcam:2 video:1 image:1 cam:2",
    "teams": "teams:2 meeting:1 chat:1",
    "onedrive": "onedrive:2 sync:1 syncing:1 synced:1 uploading:1 upload:1",
    "access": "access:2 permission:2 permissions:2 shared:2 folder:1 denied:1 drive:1",
    "browser": "browser:2 chrome:2 edge:1 firefox:2 website:2 cookies:2 page:1 site:1",
    "battery": "battery:2 charging:2 charge:2 charger:2 power:1 dies:1",
    "bluetooth": "bluetooth:2 pair:2 pairing:2 mouse:1 keyboard:1 earbuds:1",
    "phishing": "phishing:2 suspicious:2 scam:2 spam:2 weird:2 link:1 strange:1 links:1",
    "newhire": "hire:2 hires:2 onboarding:2 person:1 starts:1 starting:1 employee:1 joiner:2 new:0.5",
    "restore": "recover:2 deleted:2 restore:2 undelete:2 mistake:1 lost:1 delete:1",
    "hello": "hello:3 hi:3 hey:3 morning:3 afternoon:3 evening:3 greetings:3",
    "thanks": "thanks:3 thank:3 thx:3 worked:2 appreciate:3 cheers:3",
    "bye": "bye:4 goodbye:4 later:1 all:1",
    "escalate": "still:3 tried:3 help:1.5 again:1 ticket:1 same:1",
    "recall": "number:3 repeat:1",
    "status": "update:1 status:1",
    "service": "down:3 outage:3 degraded:3 up:2 outages:3",
    "urgent": "urgent:3 asap:3 emergency:3 critical:2 immediately:2",
}
UNKNOWN_DEFAULT = 0.4  # vote of the <user> token for "unknown"
SVC_SYN = {"email": "email outlook mail", "vpn": "vpn", "wifi": "wifi wi-fi wireless",
           "teams": "teams", "onedrive": "onedrive", "printer": "printer printing",
           "internet": "internet", "phone": "phone"}
TEAMS = list(kb.TEAMS)
STATUSES, STATES = list(kb.STATUSES), list(kb.STATES)
NAMES = list(kb.USERS) + list(kb.NON_USERS)
TIDS = [str(i) for i in kb.TICKET_IDS] if not isinstance(kb.TICKET_IDS, str) else []

# ----------------------------------------------------------------- residual layout
FLAGS = ["f_bos", "f_user", "f_bot", "f_think", "f_areg", "f_treg", "f_tid", "f_status",
         "f_state", "f_name", "f_svc"]
GATES = ["g_topic", "g_tid", "g_status", "g_state", "g_name", "gany_id", "g_team", "g_svc", "gany_pay"]
C = 48
DIMS: dict[str, int] = {}
_o = 0
for name, n in [("CODE", C), ("VOTE", NI), ("I", NI), *[(f, 1) for f in FLAGS],
                ("P1", 1), ("SU", 1), ("SB", 1), ("dirty", 1), *[(g, 1) for g in GATES],
                ("TEAM", len(TEAMS)), ("SVC", len(SVC_SYN))]:
    DIMS[name] = _o
    _o += n
D = _o
ID_CH = ["topic", "tid", "status", "state", "name"]  # copy-ID head channels (value = CODE)
PAY_CH = ["team", "svc"]  # copy-payload head channels

# strengths
A = 40.0  # logit of a table / copy output
KF, LAM, BSINK = 1e4, 20.0, 7000.0  # attention: flag, recency, sink
MR = 1000.0  # ramp intent gate
S_V, M_V = 1e5, 1e6  # vote readout gain and gate (gate > S_V*(1+votes): register tokens carry VOTE=1)


def tid(w):
    return T.tid(w)


def build_codes():
    g = torch.Generator().manual_seed(100)
    c = torch.randn(T.vocab_size(), C, generator=g, dtype=torch.float64)
    c = c / c.norm(dim=1, keepdim=True)
    for i in INTENTS[:32]:
        c[tid(REG[i])] = 0  # register tokens are read out through VOTE, never through CODE
    c[tid("<bos>")] = 0  # <bos> is the attention sink: its value must be empty
    return c


# ----------------------------------------------------------------- templates (spec -> steps)
def turn_positions(intent):
    """Positions of one bot turn from <bot> (offset 0). Each item (token_or_copy, made_by_model)."""
    F, faqrep = kb.FLOWS, {t: r for t, (tm, r) in kb.faq(0).items()}
    th = [("<think>", 1), ("REG", 1)] + ([("REG2", 1)] if intent == "urgent" else []) + [("</think>", 1)]
    tool, res, reply = None, 0, None
    if intent in TOPICS:
        reply = faqrep[intent]
    elif intent in ("hello", "thanks", "bye", "unknown"):
        reply = kb.SOCIAL[intent]
    elif intent == "escalate":
        tool, res, reply = ("create_ticket", "@topic"), 2, F["escalate"]
    elif intent == "recall":
        reply = F["recall"]
    elif intent == "status":
        tool, res, reply = ("lookup_ticket", "@tid"), 1, F["status"]
    elif intent == "service":
        tool, res, reply = ("service_status", "@svc"), 1, F["service"]
    elif intent == "urgent":
        tool, res, reply = ("page_team", "@team"), 1, F["urgent"]
    else:  # reset family
        tool, res = ("reset_password", "@name"), 1
        reply = F["reset_unknown"] if intent == "reset_unknown" else F["reset_sent"]
    P = [("<bot>", 0)] + th
    if tool:
        P += [("<tool>", 1), (tool[0], 1), (tool[1], 1), ("</tool>", 1), ("<result>", 0)]
        P += [("RES", 0)] * res + [("</result>", 0)]
    sub = {"{id}": "@tid", "{topic}": "@topic", "{team}": "@team", "{status}": "@status",
           "{service}": "@svc", "{state}": "@state", "{user}": "@name"}
    for w in reply.split():
        P.append((sub.get(w, w), 1))
    P.append(("<end>", 1))
    return P


def template(intent):
    """{k: item} = what the model must emit when at offset k (item is a word or '@channel')."""
    P = turn_positions(intent)
    k0 = {"urgent": 3}.get(intent, 3 if intent in TOPICS else 2)
    if intent in ("reset_sent", "reset_unknown"):
        k0 = [w for w, _ in P].index("</result>")
    tmpl = {}
    for k in range(k0, len(P) - 1):
        w, made = P[k + 1]
        if intent == "reset" and w == "<result>":
            break
        if made:
            tmpl[k] = w
    return tmpl


L0_FAQ = TOPICS[:6]  # FAQ tables placed in layer 0 (rest in layer 1) to balance MLP widths
L0_INTENTS = L0_FAQ + [a for a in ACTIONS if a != "urgent"] + ["urgent"]


def ramp_units(intent, gate_ok, codes):
    """Ramp construction: R_j = relu(d - j + 1 + MR*(I_intent - 1)), d = t - SB.
    Indicator(d == k) = R_k - 2R_{k+1} + R_{k+2}, so unit j carries v_j - 2v_{j-1} + v_{j-2}."""
    tm = template(intent)
    ks = sorted(tm)
    vec = {}
    for k, w in tm.items():
        v = torch.zeros(D, dtype=torch.float64)
        if w.startswith("@"):
            ch = w[1:]
            assert gate_ok, (intent, w)
            v[DIMS["g_" + ch]] = 1
            v[DIMS["gany_id" if ch in ID_CH else "gany_pay"]] = 1
        else:
            v[DIMS["CODE"]:DIMS["CODE"] + C] = A * codes[tid(w)]
            if gate_ok:
                v[DIMS["dirty"]] = 1
        vec[k] = v
    z = torch.zeros(D, dtype=torch.float64)
    units = []
    for j in range(ks[0], ks[-1] + 3):
        w_out = vec.get(j, z) - 2 * vec.get(j - 1, z) + vec.get(j - 2, z)
        w_in = torch.zeros(D, dtype=torch.float64)
        w_in[DIMS["P1"]], w_in[DIMS["SB"]] = 1, -1
        if intent is not None:
            w_in[DIMS["I"] + INTENTS.index(intent)] = MR
        b = 1 - j - (MR if intent is not None else 0)
        units.append((w_in, b, w_out))
    return units


def build():
    V = T.vocab_size()
    codes = build_codes()
    E = torch.zeros(V, D, dtype=torch.float64)
    E[:, DIMS["CODE"]:DIMS["CODE"] + C] = codes
    vo = DIMS["VOTE"]

    def setf(w, f, val=1.0):
        if w in T.token_to_id():
            E[tid(w), DIMS[f]] = val

    # ---- keyword votes (hand-written lists above), scale 0.1
    for intent, spec in KW.items():
        for item in spec.split():
            w, s = item.rsplit(":", 1)
            if w in T.token_to_id():
                E[tid(w), vo + INTENTS.index(intent)] += 0.1 * float(s)
    for n in NAMES:
        setf(n, "f_name")
        E[tid(n), vo + INTENTS.index("reset")] += 0.5
    for t in TIDS:
        setf(t, "f_tid")
        E[tid(t), vo + INTENTS.index("status")] += 0.5
    E[tid("<user>"), vo + INTENTS.index("unknown")] = 0.1 * UNKNOWN_DEFAULT
    # ---- structural flags
    for w, f in [("<bos>", "f_bos"), ("<user>", "f_user"), ("<bot>", "f_bot"), ("<think>", "f_think")]:
        setf(w, f)
    for s in STATUSES:
        setf(s, "f_status")
    for s in STATES:
        setf(s, "f_state")
    for k, (svc, syn) in enumerate(SVC_SYN.items()):
        for w in syn.split():
            if w in T.token_to_id():
                setf(w, "f_svc")
                E[tid(w), DIMS["SVC"] + k] = 1
    # ---- register tokens: one-hot intent in VOTE, flags, team payload
    for i in INTENTS:
        r = tid(REG[i])
        if i not in ("reset_sent", "reset_unknown"):
            E[r, vo:vo + NI] = 0
        E[r, vo + INTENTS.index(i)] = 1
        E[r, DIMS["f_areg"]] = 1
        if i in TOPICS:
            E[r, DIMS["f_treg"]] = 1
            E[r, DIMS["TEAM"] + TEAMS.index(kb.FAQ[i][0] if isinstance(kb.FAQ[i], tuple) else kb.faq(0)[i][0])] = 1

    pos = torch.zeros(T.MAX_LEN, D, dtype=torch.float64)
    pos[:, DIMS["P1"]] = torch.arange(T.MAX_LEN, dtype=torch.float64)

    H, dh = 3, C
    sq = math.sqrt(dh)
    P = {"embed": E, "pos": pos}

    def attn_layer(i):
        p = f"layers.{i}."
        for n in ("wq", "wk", "wv"):
            P[p + n] = torch.zeros(H, D, dh, dtype=torch.float64)
        P[p + "wo"] = torch.zeros(H, dh, D, dtype=torch.float64)
        for n in ("bq", "bk", "bv"):
            P[p + n] = torch.zeros(H, dh, dtype=torch.float64)
        P[p + "bo"] = torch.zeros(D, dtype=torch.float64)
        return p

    def recency_key(p, h, flag):
        """score = KF*flag + LAM*position + BSINK*is_bos  (query = constant 1)."""
        P[p + "bq"][h, 0] = sq
        P[p + "wk"][h, DIMS[flag], 0] = KF
        P[p + "wk"][h, DIMS["P1"], 0] = LAM
        P[p + "wk"][h, DIMS["f_bos"], 0] = BSINK

    # ---- layer 0 attention
    p = attn_layer(0)
    for h, (flag, dst) in enumerate([("f_user", "SU"), ("f_bot", "SB")]):
        recency_key(p, h, flag)
        P[p + "wv"][h, DIMS["P1"], 0] = 1
        P[p + "wo"][h, 0, DIMS[dst]] = 1
    recency_key(p, 2, "f_areg")  # intent reader: latest register -> I
    for a in range(NI):
        P[p + "wv"][2, vo + a, a] = 1
        P[p + "wo"][2, a, DIMS["I"] + a] = 1

    # ---- layer 1 attention
    p = attn_layer(1)
    # head 0: vote average over the current user turn (keys with the largest SU)
    P[p + "bq"][0, 0] = 10 * sq
    P[p + "wk"][0, DIMS["SU"], 0] = 1
    for a in range(NI):
        P[p + "wv"][0, vo + a, a] = 1
        P[p + "wo"][0, a, vo + a] = 1
    # heads 1, 2: gated copy heads
    for h, chans in [(1, ID_CH), (2, PAY_CH)]:
        gany = "gany_id" if h == 1 else "gany_pay"
        flag_of = {"topic": "f_treg", "tid": "f_tid", "status": "f_status", "state": "f_state",
                   "name": "f_name", "team": "f_treg", "svc": "f_svc"}
        for c, ch in enumerate(chans):
            P[p + "wq"][h, DIMS["g_" + ch], c] = KF * sq
            P[p + "wk"][h, DIMS[flag_of[ch]], c] = 1
        n = len(chans)
        P[p + "wq"][h, DIMS[gany], n] = LAM * sq  # recency only when a gate is open
        P[p + "wk"][h, DIMS["P1"], n] = 1
        P[p + "bq"][h, n + 1] = sq  # sink on <bos>
        P[p + "wk"][h, DIMS["f_bos"], n + 1] = BSINK
        if h == 1:
            P[p + "wq"][h, DIMS[gany], n + 2] = -KF * sq  # never copy from "dirty" positions
            P[p + "wk"][h, DIMS["dirty"], n + 2] = 1
    cs = slice(DIMS["CODE"], DIMS["CODE"] + C)
    # copy-ID value: own CODE, plus topic register one-hot -> code of the topic word
    vv = torch.zeros(D, C, dtype=torch.float64)
    vv[cs, :] = torch.eye(C, dtype=torch.float64)
    for t in TOPICS:
        vv[vo + INTENTS.index(t), :] = codes[tid(t)]
    P[p + "wv"][1] = vv
    P[p + "wo"][1][:, cs] = A * torch.eye(C, dtype=torch.float64)
    vv = torch.zeros(D, C, dtype=torch.float64)
    for k, tm in enumerate(TEAMS):
        vv[DIMS["TEAM"] + k, :] = codes[tid(tm)]
    for k, s in enumerate(SVC_SYN):
        vv[DIMS["SVC"] + k, :] = codes[tid(s)]
    P[p + "wv"][2] = vv
    P[p + "wo"][2][:, cs] = A * torch.eye(C, dtype=torch.float64)

    # ---- MLPs
    mlp = {0: [], 1: []}
    # fixed: offset 0 (<bot>) -> <think>, independent of intent
    for j in range(0, 3):
        w_in = torch.zeros(D, dtype=torch.float64)
        w_in[DIMS["P1"]], w_in[DIMS["SB"]], w_in[DIMS["f_bot"]] = 1, -1, MR
        v = torch.zeros(D, dtype=torch.float64)
        v[cs] = A * codes[tid("<think>")] * [1, -2, 1][j]
        mlp[0].append((w_in, 1.0 - j - MR, v))  # only on the <bot> token itself
    for i in INTENTS:
        layer = 0 if i in L0_INTENTS else 1
        mlp[layer] += ramp_units(i, layer == 0, codes)
    # layer 1: offset-1 readout (current token is <think>): relu(S*vote_a + M*f_think - M) -> VOTE_a
    for a in INTENTS:
        if a in ("urgent", "reset_sent", "reset_unknown"):
            continue
        w_in = torch.zeros(D, dtype=torch.float64)
        w_in[vo + INTENTS.index(a)] = S_V
        w_in[DIMS["f_think"]] = M_V
        v = torch.zeros(D, dtype=torch.float64)
        v[vo + INTENTS.index(a)] = 1
        mlp[1].append((w_in, -M_V, v))
    # offset 2 after a topic register: "</think>", unless the turn is urgent -> urgent register
    ui = vo + INTENTS.index("urgent")
    w_in = torch.zeros(D, dtype=torch.float64)
    w_in[DIMS["f_treg"]], w_in[ui] = 1, -1e4
    v = torch.zeros(D, dtype=torch.float64)
    v[cs] = A * codes[tid("</think>")]
    mlp[1].append((w_in, 0.0, v))
    w_in = torch.zeros(D, dtype=torch.float64)
    w_in[ui], w_in[DIMS["f_treg"]] = S_V, M_V
    v = torch.zeros(D, dtype=torch.float64)
    v[ui] = 1
    mlp[1].append((w_in, -M_V, v))

    dm = max(len(mlp[0]), len(mlp[1]))
    for li in (0, 1):
        p = f"layers.{li}."
        Win = torch.zeros(D, dm, dtype=torch.float64)
        Wout = torch.zeros(dm, D, dtype=torch.float64)
        b = torch.zeros(dm, dtype=torch.float64)
        for u, (wi, bi, wo) in enumerate(mlp[li]):
            Win[:, u], b[u], Wout[u] = wi, bi, wo
        P[p + "w_in"], P[p + "w_out"], P[p + "b_in"] = Win, Wout, b
        P[p + "b_out"] = torch.zeros(D, dtype=torch.float64)
    P["b_unembed"] = torch.zeros(V, dtype=torch.float64)
    cfg = Config(d_model=D, n_layers=2, n_heads=H, d_head=dh, d_mlp=dm, pos="learned",
                 tie_embeddings=True, norm=False, bias=True)
    return cfg, P
