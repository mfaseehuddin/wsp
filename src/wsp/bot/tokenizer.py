"""Word-level tokenizer for the IT support bot (2,000-token fixed vocabulary).

Splits lowercase text into special tags (<user>, <tool>, ...), words
(letters/digits/underscore, with internal ' . - allowed) and punctuation.
Unknown words map to <unk>. The vocabulary is stored in vocab.txt.

Conversation layout (one sequence, at most MAX_LEN tokens):
  <bos> <user> words <bot> [<think> any tokens </think>]
        [<tool> name arg </tool> <result> result tokens </result>]*  reply words <end>
        <user> ... <bot> ... <end> ...
The harness writes <bos>, every <user> turn, <bot>, and each <result> ... </result>
block. The model writes everything else in a bot turn, ending with <end>.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

SPECIALS = [
    "<pad>",
    "<bos>",
    "<user>",
    "<bot>",
    "<think>",
    "</think>",
    "<tool>",
    "</tool>",
    "<result>",
    "</result>",
    "<end>",
    "<unk>",
]
PAD, BOS, USER, BOT, THINK, THINK_END, TOOL, TOOL_END, RESULT, RESULT_END, END, UNK = (
    SPECIALS
)
VOCAB_SIZE = 2000
MAX_LEN = 256

_VOCAB = Path(__file__).with_name("vocab.txt")
_TOK = re.compile(r"</?[a-z]+>|[a-z0-9_]+(?:['.\-][a-z0-9_]+)*|[?!,.:]")


def split(text: str) -> list[str]:
    return _TOK.findall(text.lower())


@lru_cache(maxsize=1)
def vocab() -> list[str]:
    return _VOCAB.read_text().split("\n")


@lru_cache(maxsize=1)
def token_to_id() -> dict[str, int]:
    return {t: i for i, t in enumerate(vocab())}


def vocab_size() -> int:
    return len(vocab())


def tid(token: str) -> int:
    return token_to_id()[token]


def encode(text: str) -> list[int]:
    m = token_to_id()
    return [m.get(w, m[UNK]) for w in split(text)]


def decode(ids: list[int]) -> str:
    v = vocab()
    return " ".join(v[i] for i in ids)


def build_vocab() -> list[str]:
    """Admin: build and save the vocabulary (run once; result is committed)."""
    from wsp.support.data import EXTRA_WORDS

    from . import kb

    words: set[str] = set()

    def add(text: str) -> None:
        words.update(split(re.sub(r"\{[a-z]+\}", " ", text)))

    for st in range(4):
        for team, reply in kb.faq(st).values():
            add(team + " " + reply)
    for t in [*kb.SOCIAL.values(), *kb.FLOWS.values(), *kb.all_topics(3)]:
        add(t)
    for group in (
        kb.TOOLS,
        kb.SERVICES,
        kb.STATES,
        kb.STATUSES,
        kb.TICKET_IDS,
        kb.USERS,
        kb.NON_USERS,
        kb.TEAMS,
        EXTRA_WORDS,
    ):
        for w in group:
            add(w)
    h = kb._hidden()
    all_pools = [kb.DEV_POOLS, *h["pools"].values(), {"faq": h["stage1_dev_faq"]}]
    for P in all_pools:
        for k, v in P.items():
            for phrase in [x for xs in v.values() for x in xs] if k == "faq" else v:
                add(phrase)
    words -= set(SPECIALS)
    # Fill to VOCAB_SIZE with common-looking dictionary words (deterministic).
    with open("/usr/share/dict/words") as f:
        filler = [w.strip() for w in f]
    filler = sorted(
        {w for w in filler if w.isalpha() and w.islower() and 3 <= len(w) <= 7} - words
    )
    need = VOCAB_SIZE - len(SPECIALS) - len(words)
    step = max(1, len(filler) // need)
    words |= set(filler[::step][:need])
    v = SPECIALS + sorted(words)
    assert len(v) == VOCAB_SIZE, len(v)
    _VOCAB.write_text("\n".join(v))
    vocab.cache_clear()
    token_to_id.cache_clear()
    return v
