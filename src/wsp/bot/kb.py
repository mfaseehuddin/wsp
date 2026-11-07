"""Knowledge base, mock tools and conversation generator for the IT support bot.

Everything the bot must know lives here: FAQ replies, which team owns each
topic, the mock tools and their fake database, and the reply templates that
use tool results. Conversations are generated from phrasing pools.

- dev pools are in this file and agents may read them;
- test / field pools and the stage-1 topics are in `hidden.blob`
  (zlib+base64). Only the admin harness decodes them. Agents must not.

Stages (set by the admin in experiments/bot/STAGE):
  0  initial build
  1  learning round 1: new FAQ topics are added (badge, keyboard, expenses)
  2  learning round 2: no KB change; feedback from failed field conversations
  3  learning round 3: knowledge update (help desk number 4357 -> 4400,
     reset links expire in 30 minutes instead of 15) + more field feedback
"""

from __future__ import annotations

import base64
import json
import random
import zlib
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------- knowledge

TEAMS = ["network", "desktop", "accounts", "messaging", "security"]

FAQ: dict[str, tuple[str, str]] = {  # topic -> (owning team, canonical reply)
    "password": (
        "accounts",
        "go to reset.corp.com and choose forgot password . the link expires in 15 minutes .",
    ),
    "locked": (
        "accounts",
        "your account unlocks after 30 minutes . to unlock it now , call the help desk on 4357 .",
    ),
    "vpn": (
        "network",
        "restart the vpn client , check your internet connection , then sign in again .",
    ),
    "wifi": (
        "network",
        "forget the corp wifi network , reconnect , and enter your corp password when asked .",
    ),
    "email": (
        "messaging",
        "close outlook , open it again and check you are online . if mail still fails , restart your computer .",
    ),
    "printer": (
        "desktop",
        "check the printer has paper and toner , then remove it and add it again in settings .",
    ),
    "software": (
        "desktop",
        "request software in the self service portal . most requests are approved within one day .",
    ),
    "slow": (
        "desktop",
        "restart your computer and close unused apps . if it stays slow , i can open a ticket .",
    ),
    "mfa": (
        "security",
        "open the authenticator app and approve the request . for a new phone , call the help desk on 4357 .",
    ),
    "disk": (
        "desktop",
        "empty the recycle bin and move large files to onedrive , then run disk cleanup .",
    ),
    "monitor": (
        "desktop",
        "check the cable at both ends , then press windows p and choose extend .",
    ),
    "audio": (
        "desktop",
        "check the headset is the default device in sound settings , then rejoin the call .",
    ),
    "camera": (
        "desktop",
        "close other apps that use the camera , check the privacy shutter , then restart teams .",
    ),
    "teams": (
        "messaging",
        "sign out of teams , clear the cache in settings , then sign in again .",
    ),
    "onedrive": (
        "messaging",
        "click the onedrive cloud icon , choose resume syncing , and check you have free space .",
    ),
    "access": (
        "security",
        "ask your manager to approve access in the self service portal under shared folders .",
    ),
    "browser": (
        "desktop",
        "clear the browser cache and cookies , then try again in a private window .",
    ),
    "battery": (
        "desktop",
        "try a different power outlet and charger . if the battery still drains fast , i can open a ticket .",
    ),
    "bluetooth": (
        "desktop",
        "turn bluetooth off and on , remove the device , then pair it again .",
    ),
    "phishing": (
        "security",
        "do not click any links . use the report phishing button in outlook , then delete the email .",
    ),
    "newhire": (
        "accounts",
        "your manager orders new laptops in the self service portal two weeks before the start date .",
    ),
    "restore": (
        "messaging",
        "deleted files stay in the onedrive recycle bin for 90 days . open it and choose restore .",
    ),
}

SOCIAL = {
    "hello": "hello , i am the it support bot . what is the problem ?",
    "thanks": "you are welcome . is there anything else ?",
    "bye": "goodbye , have a good day !",
    "unknown": "sorry , i can only help with it problems .",
}

# Replies that use tool results. {slots} are filled from the conversation.
FLOWS = {
    "escalate": "i opened ticket {id} about {topic} . the {team} team will contact you today .",
    "recall": "your ticket number is {id} .",
    "status": "ticket {id} is {status} .",
    "service": "{service} is {state} right now .",
    "urgent": "i paged the {team} team . they will call you within 15 minutes .",
    "reset_sent": "a reset link was sent to {user} .",
    "reset_unknown": "i could not find the user {user} .",
}

# ---------------------------------------------------------------- mock tools

TOOLS = {
    "create_ticket": "create_ticket <topic>  -> ticket <id>",
    "lookup_ticket": "lookup_ticket <id>     -> <status>",
    "service_status": "service_status <service> -> up | down | degraded",
    "page_team": "page_team <team>       -> paged | error",
    "reset_password": "reset_password <user>  -> sent | unknown",
}
SERVICES = ["email", "vpn", "wifi", "teams", "onedrive", "printer", "internet", "phone"]
STATES = ["up", "down", "degraded"]
STATUSES = ["open", "assigned", "waiting", "resolved", "closed"]
TICKET_IDS = [str(i) for i in range(4000, 4100)]
USERS = [
    "jsmith",
    "adoe",
    "bkhan",
    "cwong",
    "dpatel",
    "egarcia",
    "fnguyen",
    "gmuller",
    "hkim",
    "irossi",
    "jlee",
    "kbrown",
    "lsilva",
    "mali",
    "nchen",
    "olsen",
    "pdubois",
    "qwang",
    "rsingh",
    "styler",
]
NON_USERS = [
    "tgray",
    "uberg",
    "vlopez",
    "wbaker",
    "xmoore",
    "yhassan",
    "zortiz",
    "abell",
]


class MockDB:
    """Deterministic fake back end. One instance per conversation (seeded)."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.state = {s: self.rng.choice(STATES) for s in SERVICES}

    def call(self, name: str, args: list[str]) -> list[str]:
        a = args[0] if len(args) == 1 else None
        if name == "create_ticket" and a in all_topics(3):
            return ["ticket", self.rng.choice(TICKET_IDS)]
        if name == "lookup_ticket" and a in TICKET_IDS:
            return [STATUSES[int(a) * 7 % len(STATUSES)]]
        if name == "service_status" and a in SERVICES:
            return [self.state[a]]
        if name == "page_team" and a in TEAMS:
            return ["paged"]
        if name == "reset_password" and a is not None:
            return ["sent" if a in USERS else "unknown"]
        return ["error"]


# ---------------------------------------------------------------- hidden data

_BLOB = Path(__file__).with_name("hidden.blob")


@lru_cache(maxsize=1)
def _hidden() -> dict:
    """Admin only. Agents must not call this or read hidden.blob."""
    return json.loads(zlib.decompress(base64.b64decode(_BLOB.read_text())))


def all_topics(stage: int) -> list[str]:
    topics = list(FAQ)
    if stage >= 1:
        topics += list(_hidden()["stage1_faq"])
    return topics


def faq(stage: int) -> dict[str, tuple[str, str]]:
    out = dict(FAQ)
    if stage >= 1:
        out.update({k: tuple(v) for k, v in _hidden()["stage1_faq"].items()})
    if stage >= 3:
        out = {k: (t, r.replace("4357", "4400")) for k, (t, r) in out.items()}
        t, r = out["password"]
        out["password"] = (
            t,
            r.replace("expires in 15 minutes", "expires in 30 minutes"),
        )
    return out


# ---------------------------------------------------------------- dev phrasing pools

DEV_POOLS: dict = {
    "faq": {
        "password": [
            "i forgot my password",
            "how do i reset my password ?",
            "my password expired",
        ],
        "locked": [
            "my account is locked",
            "i am locked out of my account",
            "account locked after too many attempts",
        ],
        "vpn": [
            "vpn is not connecting",
            "i can't connect to the vpn",
            "vpn keeps disconnecting",
        ],
        "wifi": [
            "wifi is not working",
            "i can't connect to the wifi",
            "the wireless keeps dropping",
        ],
        "email": [
            "outlook is not syncing",
            "i am not getting emails",
            "outlook keeps crashing",
        ],
        "printer": [
            "the printer is not working",
            "i can't print",
            "printer says offline",
        ],
        "software": [
            "i need to install software",
            "can you install excel for me",
            "request a software license",
        ],
        "slow": [
            "my computer is very slow",
            "laptop is running slow",
            "everything is freezing",
        ],
        "mfa": [
            "mfa is not working",
            "i did not get the authenticator code",
            "i got a new phone and need mfa",
        ],
        "disk": [
            "my disk is full",
            "no space left on my laptop",
            "low disk space warning",
        ],
        "monitor": [
            "my second monitor is not working",
            "external screen not detected",
            "monitor shows no signal",
        ],
        "audio": [
            "nobody can hear me in calls",
            "my headset has no sound",
            "microphone not working",
        ],
        "camera": [
            "my camera is not working",
            "webcam shows a black image",
            "teams can't find my camera",
        ],
        "teams": [
            "teams won't load",
            "teams keeps signing me out",
            "teams messages are not sending",
        ],
        "onedrive": [
            "onedrive is not syncing",
            "onedrive says sync paused",
            "my files are not uploading to onedrive",
        ],
        "access": [
            "i need access to the finance folder",
            "can't open the shared drive",
            "access denied on a shared folder",
        ],
        "browser": [
            "websites won't load in chrome",
            "my browser keeps crashing",
            "page shows an error in edge",
        ],
        "battery": [
            "my laptop battery drains fast",
            "laptop is not charging",
            "battery dies after an hour",
        ],
        "bluetooth": [
            "bluetooth mouse not connecting",
            "my headphones won't pair",
            "bluetooth device keeps disconnecting",
        ],
        "phishing": [
            "i got a suspicious email",
            "i think this email is phishing",
            "someone sent me a weird link",
        ],
        "newhire": [
            "a new person starts next week",
            "how do i get a laptop for a new hire",
            "setting up a new employee",
        ],
        "restore": [
            "i deleted a file by mistake",
            "how do i recover a deleted document",
            "restore my deleted folder",
        ],
    },
    "hello": ["hello", "hi there", "good morning"],
    "thanks": ["thanks", "thank you", "that worked thanks"],
    "bye": ["bye", "goodbye", "that is all , bye"],
    "unknown": [
        "what is the weather today",
        "who won the game last night",
        "what is for lunch",
    ],
    "followup": [
        "it is still not working",
        "that did not help",
        "i tried that and it still fails",
    ],
    "recall": ["what is my ticket number ?", "can you repeat my ticket number ?"],
    "status": ["what is the status of ticket {id} ?", "any update on ticket {id} ?"],
    "service": ["is {service} down ?", "is there an outage with {service} ?"],
    "urgent": ["urgent :", "this is urgent ,"],
    "reset": ["please reset the password for {user}", "reset password for user {user}"],
}

SPLIT_SIZES = {"dev": 70, "test": 160, "field": 80}
SPLIT_SEEDS = {"dev": 11, "test": 22, "field": 33}


def pools(split: str, stage: int = 0) -> dict:
    if split != "dev":
        return _hidden()["pools"][split]
    if stage < 1:
        return DEV_POOLS
    return {**DEV_POOLS, "faq": {**DEV_POOLS["faq"], **_hidden()["stage1_dev_faq"]}}


# ---------------------------------------------------------------- conversations


def _fill(t: str, **kw) -> str:
    return t.format(**kw)


def conversations(split: str, stage: int) -> list[dict]:
    """Deterministic conversations for a split at a stage.

    Each conversation: {"id", "seed", "turns": [{"user", "tools": [[name, arg]], "reply"}]}.
    Tool results are not stored: they come from MockDB(seed) at run time, and
    the gold replies below were filled using the same MockDB.
    """
    P = pools(split, stage)
    F = faq(stage)
    topics = [t for t in all_topics(stage) if t in P["faq"]]
    rng = random.Random(SPLIT_SEEDS[split] * 1000 + stage)
    out = []
    for i in range(SPLIT_SIZES[split]):
        seed = SPLIT_SEEDS[split] * 100000 + stage * 1000 + i
        db = MockDB(seed)
        turns: list[dict] = []

        def say(user: str, tools: list, reply: str) -> None:
            turns.append({"user": user, "tools": tools, "reply": reply})

        def faq_turn(topic: str) -> None:
            say(rng.choice(P["faq"][topic]), [], F[topic][1])

        def escalate(topic: str) -> str:
            tid = db.call("create_ticket", [topic])[1]
            say(
                rng.choice(P["followup"]),
                [["create_ticket", topic]],
                _fill(FLOWS["escalate"], id=tid, topic=topic, team=F[topic][0]),
            )
            return tid

        def social(kind: str) -> None:
            say(rng.choice(P[kind]), [], SOCIAL[kind])

        kind = i % 12
        a, b = rng.sample(topics, 2)
        if kind == 0:
            faq_turn(a)
        elif kind == 1:
            social("hello"), faq_turn(a), social("thanks")
        elif kind == 2:
            faq_turn(a), escalate(a)
        elif kind == 3:
            faq_turn(a)
            tid = escalate(a)
            say(rng.choice(P["recall"]), [], _fill(FLOWS["recall"], id=tid))
        elif kind == 4:
            tid = rng.choice(TICKET_IDS)
            st = db.call("lookup_ticket", [tid])[0]
            say(
                _fill(rng.choice(P["status"]), id=tid),
                [["lookup_ticket", tid]],
                _fill(FLOWS["status"], id=tid, status=st),
            )
        elif kind == 5:
            s = rng.choice(SERVICES)
            say(
                _fill(rng.choice(P["service"]), service=s),
                [["service_status", s]],
                _fill(
                    FLOWS["service"], service=s, state=db.call("service_status", [s])[0]
                ),
            )
        elif kind == 6:
            team = F[a][0]
            say(
                rng.choice(P["urgent"]) + " " + rng.choice(P["faq"][a]),
                [["page_team", team]],
                _fill(FLOWS["urgent"], team=team),
            )
        elif kind == 7:
            u = rng.choice(USERS + NON_USERS)
            res = db.call("reset_password", [u])[0]
            say(
                _fill(rng.choice(P["reset"]), user=u),
                [["reset_password", u]],
                _fill(
                    FLOWS["reset_sent" if res == "sent" else "reset_unknown"], user=u
                ),
            )
        elif kind == 8:
            faq_turn(a), faq_turn(b), social("bye")
        elif kind == 9:
            social("unknown"), faq_turn(a)
        elif kind == 10:
            faq_turn(a), faq_turn(b), escalate(b)
        else:
            social("hello")
            s = rng.choice(SERVICES)
            say(
                _fill(rng.choice(P["service"]), service=s),
                [["service_status", s]],
                _fill(
                    FLOWS["service"], service=s, state=db.call("service_status", [s])[0]
                ),
            )
            social("thanks"), social("bye")
        out.append({"id": i, "seed": seed, "turns": turns})
    return out
