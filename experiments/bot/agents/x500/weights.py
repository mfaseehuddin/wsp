"""x500 weights: lexicon rows + program-memory rows, all built from the KB and hand-written keyword lists."""
import torch
from wsp.bot import kb, tokenizer as T
from wsp.bot.custom import CustomArch

ARCH = "experiments/bot/agents/x500/arch.py"
tid = lambda w: T.encode(w)[0]
FAQ = kb.faq(0)
TOPICS = list(FAQ)
PLAN_TOK = TOPICS + ["hello", "thanks", "bye", "other", "ticket", "remind", "status", "service", "urgent", "reset"]
PIDX = {n: i for i, n in enumerate(TOPICS + ["hello", "thanks", "bye", "unknown", "escalate", "recall", "status", "service", "urgent", "reset"])}
TEAM_TOK = {t: tid(t) for t in kb.TEAMS}
# structural flag dims (after the 32 evidence dims)
D0 = 32
SD = {n: D0 + i for i, n in enumerate("user bot think thinkend tool toolend result resultend end PI TM ftop fid fsvc fres fname emit vt2".split())}

# hand-written keyword evidence: plan -> {word: weight}
KW = {
 "password": dict(password=2, forgot=1, expired=1, login=1, passwords=2),
 "locked": dict(locked=3, lock=2, unlock=2, lockout=3),
 "vpn": dict(vpn=3, tunnel=1),
 "wifi": {"wifi": 3, "wireless": 3, "wi-fi": 3, "internet": 1, "network": 1},
 "email": dict(email=2, emails=2, outlook=3, mail=2, inbox=2, mailbox=2),
 "printer": dict(printer=3, print=2, printing=2, toner=2, paper=1, scanner=1, printed=2),
 "software": dict(software=3, install=3, installed=2, license=2, excel=2, application=1, app=1, program=1, download=1),
 "slow": dict(slow=3, freezing=3, freezes=3, frozen=2, lag=2, sluggish=3, performance=2, hanging=2),
 "mfa": {"mfa": 4, "authenticator": 3, "authentication": 3, "2fa": 3, "code": 1, "token": 1, "verification": 2},
 "disk": dict(disk=3, space=2, storage=2, full=2, cleanup=2),
 "monitor": dict(monitor=3, screen=2, display=2, hdmi=2, signal=1, projector=1, external=1),
 "audio": dict(audio=3, sound=3, hear=3, headset=2, microphone=3, mic=3, speaker=2, speakers=2, headphones=1, volume=2),
 "camera": dict(camera=3, webcam=3, video=1, black=1, image=1),
 "teams": dict(teams=3, chat=1),
 "onedrive": dict(onedrive=3, sync=1, syncing=1, cloud=1, uploading=1),
 "access": dict(access=3, permission=3, permissions=3, folder=2, shared=2, denied=2, share=1, finance=1, drive=1),
 "browser": dict(browser=3, chrome=3, edge=3, firefox=3, safari=3, website=2, websites=2, page=1, web=1),
 "battery": dict(battery=3, charging=3, charger=2, drains=2, power=1, dies=1),
 "bluetooth": dict(bluetooth=4, pair=3, pairing=3, paired=3, headphones=1, mouse=1, keyboard=1),
 "phishing": dict(phishing=4, suspicious=3, scam=3, spam=2, weird=1, link=2, links=2, fraud=2, malicious=3),
 "newhire": dict(hire=3, hires=3, employee=3, starts=2, starting=2, onboarding=3, person=1, new=0.5, start=2),
 "restore": dict(deleted=3, delete=2, recover=3, restore=3, recycle=2, recovery=3, lost=1, mistake=1, accidentally=2, undelete=3),
 "hello": dict(hello=3, hi=3, hey=3, morning=3, afternoon=2, evening=2, greetings=3, howdy=3, good=0.5),
 "thanks": dict(thx=3, ty=3, resolved=2, sorted=2, thanks=3, thank=3, worked=3, great=1, perfect=2, solved=2, fixed=2, appreciated=2, cheers=3, awesome=2, helpful=2),
 "bye": dict(cheerio=3, bye=3, goodbye=3, all=1.5, farewell=3, later=1, see=1, cya=3, done=1),
 "escalate": {"still": 3, "tried": 2, "fails": 2, "failed": 2, "help": 1.5, "did": 1, "didn't": 2, "doesn't": 1, "nothing": 1.5, "again": 1, "same": 1.5, "helped": 1.5, "luck": 2, "ticket": 1, "open": 1, "escalate": 3, "not": 0.3},
 "recall": dict(number=2.5, repeat=2.5, remind=2.5, my=0.3, ticket=0.7, again=0.5, id=1, reference=2),
 "status": dict(update=2.5, status=2, progress=2, ticket=1),
 "service": dict(down=3, outage=3, outages=3, up=2, available=2, operational=2),
 "urgent": dict(urgent=15, urgently=15, asap=15, emergency=15, immediately=12, critical=12),
 "reset": dict(reset=1, user=1.5),
}


def build():
    rows = []  # (lo, hi, dim, w)
    def add(a, b, d, w=1.0): rows.append((a, b, d, w))
    for n, d in (("<user>", "user"), ("<bot>", "bot"), ("<think>", "think"), ("</think>", "thinkend"), ("<tool>", "tool"),
                 ("</tool>", "toolend"), ("<result>", "result"), ("</result>", "resultend"), ("<end>", "end")):
        add(tid(n), tid(n), SD[d])
    add(tid("<bot>"), tid("<bot>"), SD["emit"], tid("<think>"))
    for i, w in enumerate(PLAN_TOK):
        t = tid(w); assert t != 11, w
        add(t, t, SD["PI"], i + 1)
        if w == "urgent": add(t, t, SD["vt2"])
        else: add(t, t, SD["emit"], tid("</think>"))
    for tp in TOPICS:
        t = tid(tp); add(t, t, SD["ftop"]); add(t, t, SD["TM"], TEAM_TOK[FAQ[tp][0]])
    for tm, t in TEAM_TOK.items():
        add(t, t, SD["ftop"]); add(t, t, SD["TM"], t)
    add(tid("4000"), tid("4099"), SD["fid"]); add(tid("4000"), tid("4099"), PIDX["status"], 5)
    for s in kb.SERVICES:
        add(tid(s), tid(s), SD["fsvc"]); add(tid(s), tid(s), PIDX["service"], 1.5)
    for s in kb.STATES + kb.STATUSES + ["sent", "unknown"]:
        add(tid(s), tid(s), SD["fres"])
    for s in kb.USERS + kb.NON_USERS:
        assert tid(s) != 11, s
        add(tid(s), tid(s), SD["fname"]); add(tid(s), tid(s), PIDX["reset"], 5)
    for p, d in KW.items():
        for w, wt in d.items():
            t = tid(w)
            if t != 11: add(t, t, PIDX[p], wt)
    # program memory rows (plan, after, k, res, slot, tok)
    rom = []
    SL = dict(topic=1, team=2, id=3, service=4, status=5, state=5, user=6)
    def script(plan, after, text, res=0):
        for k, w in enumerate(text.split() + ["<end>"]):
            if w.startswith("{"): rom.append((PIDX[plan] + 1, after, k, res, SL[w[1:-1]], 0))
            else: rom.append((PIDX[plan] + 1, after, k, res, 0, tid(w)))
    for tp in TOPICS: script(tp, 0, FAQ[tp][1])
    for s in ("hello", "thanks", "bye", "unknown"): script(s, 0, kb.SOCIAL[s])
    fl = kb.FLOWS
    call = lambda tool, slot: f"<tool> {tool} {{{slot}}} </tool>"
    def strip_end(t): return t
    def tool_script(plan, tool, slot):
        for k, w in enumerate(call(tool, slot).split()):
            if w.startswith("{"): rom.append((PIDX[plan] + 1, 0, k, 0, SL[slot], 0))
            else: rom.append((PIDX[plan] + 1, 0, k, 0, 0, tid(w)))
    tool_script("escalate", "create_ticket", "topic"); script("escalate", 1, fl["escalate"])
    script("recall", 0, fl["recall"])
    tool_script("status", "lookup_ticket", "id"); script("status", 1, fl["status"])
    tool_script("service", "service_status", "service"); script("service", 1, fl["service"])
    tool_script("urgent", "page_team", "team"); script("urgent", 1, fl["urgent"])
    tool_script("reset", "reset_password", "user")
    script("reset", 1, fl["reset_sent"], tid("sent")); script("reset", 1, fl["reset_unknown"], 11)

    f = lambda xs: torch.tensor(xs, dtype=torch.float64)
    topic_mask = f([0.0] * 22 + [-100.0] * 10)
    bias = [0.0] * 32; bias[PIDX["unknown"]] = 0.5
    P = {
        "lex_lo": f([r[0] for r in rows]), "lex_hi": f([r[1] for r in rows]),
        "lex_dim": f([r[2] for r in rows]), "lex_w": f([r[3] for r in rows]),
        "dim_grid": f(list(range(50))), "plan_bias": f(bias), "plan_id": f([tid(w) for w in PLAN_TOK]),
        "topic_mask": topic_mask, "beta": f([6.0]),
        "head_req": f([1, 0, 0, 0, 0]), "att": f([4000.0, 5.0, 20000.0]), "slot_grid": f([1, 2, 3, 4, 5, 6]),
    }
    for i, n in enumerate(["plan", "after", "k", "res", "slot", "tok"]):
        P["rom_" + n] = f([r[i] for r in rom])
    return CustomArch(ARCH, "rule-memory machine: lexicon + counters + copy heads + program memory"), P
