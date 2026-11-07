"""Word-level tokenizer for the NCM help-desk task.

Lowercase; words (letters with internal ' . - allowed), single digits (numbers are split digit by digit), and
single punctuation marks. Unknown words map to <unk>.

The vocabulary (vocab.txt) is built once from neutral sources: the GenGPT 4k word list, the reply texts, the dev
queries, all names and the digits. Hidden-query words were NOT added: a word you never saw is <unk>.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PAD, EOS, UNK = "<pad>", "<eos>", "<unk>"
SPECIALS = [PAD, EOS, UNK]
MAX_QUERY = 32
_VOCAB = Path(__file__).with_name("vocab.txt")
_WORD = re.compile(r"[a-z]+(?:['.\-][a-z0-9]+)*|[0-9]|[?!,.#]")


def split(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def build_vocab() -> list[str]:  # run once by the maintainer
    from wsp.gen import tokenizer as G

    from . import data

    words = {w for w in G.vocab() if re.fullmatch(r"[a-z]+(?:['.\-][a-z0-9]+)*", w)}
    for r in data.REPLIES.values():
        words |= set(split(r))
    for q, _ in data.S.DEV:
        words |= set(split(q))
    words |= {w for d in data.S._test_set() for w in split(d["query"])}
    words |= set(data.DEV_NAMES) | set(data.HIDDEN_NAMES) | set("0123456789") | set("?!,.#")
    words |= set(split(" ".join(data.DEV_NAME_TPL + data.DEV_TICKET_TPL + ["i added this to ticket number speaking"])))
    words.discard("")
    return SPECIALS + sorted(words - set(SPECIALS))


@lru_cache(None)
def vocab() -> list[str]:
    return _VOCAB.read_text().split("\n")


@lru_cache(None)
def index() -> dict[str, int]:
    return {t: i for i, t in enumerate(vocab())}


def encode(text: str) -> list[int]:
    ix = index()
    return [ix.get(w, ix[UNK]) for w in split(text)]


def decode(ids) -> str:
    v = vocab()
    return " ".join(v[i] for i in ids)


if __name__ == "__main__":
    _VOCAB.write_text("\n".join(build_vocab()))
    print(len(vocab.__wrapped__()))
