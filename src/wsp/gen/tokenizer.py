"""Word-level tokenizer for GenGPT (4,096-token fixed vocabulary).

- the 12 special tags are the same as the bot experiment's (<bos>, <user>, <tool>, ...);
- the numbers 0..999 are single tokens (ids 12..1011, so `tid(str(n)) == 12 + n`);
- words are letters/digits/underscore with internal ' or - ;
- punctuation and path pieces are single tokens: ? ! , . : ; = / ~ and `..`.
  So `../etc/passwd` tokenizes as `.. / etc / passwd`, and a path argument in a
  tool call is its tokens joined with no spaces.
Unknown words map to <unk>. The vocabulary is stored in vocab.txt.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from wsp.bot.tokenizer import (  # noqa: F401  (re-exported)
    BOS,
    BOT,
    END,
    PAD,
    RESULT,
    RESULT_END,
    SPECIALS,
    THINK,
    THINK_END,
    TOOL,
    TOOL_END,
    UNK,
    USER,
)

VOCAB_SIZE = 4096
MAX_LEN = 1024
NUM0 = len(SPECIALS)  # id of "0"

_VOCAB = Path(__file__).with_name("vocab.txt")
_TOK = re.compile(r"</?[a-z]+>|\.\.|[a-z0-9_]+(?:['\-][a-z0-9_]+)*|[?!,.:;=/~]")


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
    """Admin: build and save the vocabulary (run once; the result is committed)."""
    from . import kb

    words: set[str] = set()

    def add(text: str) -> None:
        words.update(split(re.sub(r"\{[a-z0-9]+\}", " ", text)))

    for st in (0, 1):
        s = kb.spec(st)
        for t in [
            *s["tools"],
            *s["tools"].values(),
            *s["replies"].values(),
            *s["rules"],
        ]:
            add(t)
    for t in [*kb.FILES, *kb.KEYS, *kb.NAMES, *kb.OPS, kb.PROTECTED, "mallory", "ok", "missing",
              "denied", "error", "none", "note", *(x for v in kb.WORD_KEYS.values() for x in v),
              *(x for ln in kb.SECRETS_FILE for x in ln)]:  # fmt: skip
        add(t)
    for split_name in ("dev", "test"):
        P, paths, inj = kb.pools(split_name)
        for t in [*(x for v in P.values() for x in v), *paths, *inj]:
            add(t)
    nums = [str(n) for n in range(1000)]
    words -= set(SPECIALS) | set(nums)
    with open("/usr/share/dict/words") as f:
        filler = sorted(
            {
                w.strip()
                for w in f
                if w.strip().isalpha()
                and w.strip().islower()
                and 3 <= len(w.strip()) <= 7
            }
            - words
        )
    need = VOCAB_SIZE - len(SPECIALS) - len(nums) - len(words)
    step = max(1, len(filler) // need)
    v = SPECIALS + nums + sorted(words | set(filler[::step][:need]))
    assert len(v) == VOCAB_SIZE, len(v)
    _VOCAB.write_text("\n".join(v))
    vocab.cache_clear()
    token_to_id.cache_clear()
    return v
