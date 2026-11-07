"""NCM help-desk task: the support intents plus slots copied from the user's message.

Reply = [ "hi <name> ," ]  +  REPLIES[intent]  +  [ "i added this to ticket d d d d ." ]
- the name part appears iff the user says who they are;
- the ticket part appears iff the user mentions a ticket number (copied digit by digit).

Hidden test: the held-out support queries (never shown to any agent), with names and ticket numbers not used
in dev, and one unseen template each for names and tickets.
"""

from __future__ import annotations

import random

from wsp.support import data as S

REPLIES = S.REPLIES
INTENTS = S.INTENTS

DEV_NAMES = ["priya", "tom", "maria", "chen", "omar", "lucy", "ravi", "anna", "james", "sofia",
             "ali", "emma", "kenji", "sara", "david", "fatima", "leo", "nina", "sam", "zara"]
HIDDEN_NAMES = ["mateo", "aisha", "ivan", "grace", "noah", "hana", "felix", "olga", "yusuf", "clara",
                "pablo", "mei", "oscar", "lena", "tariq", "ruth", "hugo", "keiko", "amir", "jade"]

DEV_NAME_TPL = ["i'm {n} .", "this is {n} ,", "{n} here .", "hi , i am {n} ."]
DEV_TICKET_TPL = ["ticket {t}", "re ticket {t} .", "my ticket is {t} ."]
HIDDEN_NAME_TPL = DEV_NAME_TPL[:2] + ["{n} speaking ."]
HIDDEN_TICKET_TPL = DEV_TICKET_TPL[:2] + ["ticket number {t} ."]


def reply(intent: str, name: str | None, ticket: str | None) -> str:
    out = f"hi {name} , " if name else ""
    out += REPLIES[intent]
    if ticket:
        out += " i added this to ticket " + " ".join(ticket) + " ."
    return out


def query(q: str, name: str | None, ticket: str | None, rng: random.Random, name_tpl, ticket_tpl) -> str:
    parts = []
    if name:
        parts.append(rng.choice(name_tpl).format(n=name))
    parts.append(q)
    if ticket:
        parts.append(rng.choice(ticket_tpl).format(t=ticket))
    return " ".join(parts)


def _make(pairs, names, name_tpl, ticket_tpl, seed) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for q, intent in pairs:
        for has_n, has_t in ((0, 0), (1, 0), (0, 1), (1, 1)):
            n = rng.choice(names) if has_n else None
            t = f"{rng.randrange(1000, 10000)}" if has_t else None
            out.append({"id": len(out), "query": query(q, n, t, rng, name_tpl, ticket_tpl),
                        "intent": intent, "name": n, "ticket": t, "reply": reply(intent, n, t)})
    return out


def dev_set() -> list[dict]:
    return _make(S.DEV, DEV_NAMES, DEV_NAME_TPL, DEV_TICKET_TPL, seed=1)


def _hidden_set() -> list[dict]:
    """Admin only."""
    pairs = [(d["query"], d["intent"]) for d in S._test_set()]
    return _make(pairs, HIDDEN_NAMES, HIDDEN_NAME_TPL, HIDDEN_TICKET_TPL, seed=7)
