"""Word-level tokenizer for the IT support task.

Lowercases, then splits into words (letters/digits, with internal ' . -
allowed, e.g. "can't", "help.corp.com", "wi-fi") and single punctuation marks.
Unknown words map to <unk>. The vocabulary is fixed and sorted, so token ids
are stable.

Sequence layout for one example (MAX_LEN positions at most):
    <bos> user tokens (truncated to MAX_QUERY) <sep> reply tokens <eos>
The model is trained-free: at the position of <sep> and of every reply token it
must predict the next reply token, and finally <eos>.
"""

from __future__ import annotations

import re
from functools import lru_cache

from . import data

PAD, BOS, SEP, EOS, UNK = "<pad>", "<bos>", "<sep>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, SEP, EOS, UNK]
MAX_QUERY = 24
MAX_LEN = 64

_WORD = re.compile(r"[a-z0-9]+(?:['.\-][a-z0-9]+)*|[?!,.]")


def split(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@lru_cache(maxsize=1)
def vocab() -> list[str]:
    words: set[str] = set(data.EXTRA_WORDS)
    for r in data.REPLIES.values():
        words.update(split(r))
    for q, _ in data.DEV:
        words.update(split(q))
    words.update(w for q in data._test_vocab_words() for w in split(q))
    return SPECIALS + sorted(words - set(SPECIALS))


@lru_cache(maxsize=1)
def token_to_id() -> dict[str, int]:
    return {t: i for i, t in enumerate(vocab())}


def vocab_size() -> int:
    return len(vocab())


def tid(token: str) -> int:
    return token_to_id()[token]


def encode_words(text: str) -> list[int]:
    m = token_to_id()
    return [m.get(w, m[UNK]) for w in split(text)]


def decode(ids: list[int]) -> str:
    v = vocab()
    return " ".join(v[i] for i in ids)


def prompt_ids(query: str) -> list[int]:
    """<bos> query <sep>  -- what the model sees before it starts replying."""
    return [tid(BOS), *encode_words(query)[:MAX_QUERY], tid(SEP)]


def example_ids(query: str, reply: str) -> list[int]:
    """Full teacher-forced sequence: <bos> query <sep> reply <eos>."""
    ids = prompt_ids(query) + encode_words(reply) + [tid(EOS)]
    assert len(ids) <= MAX_LEN, "example too long"
    return ids
