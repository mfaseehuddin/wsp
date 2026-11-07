"""Reversible word-plus-byte tokenizer; protocol tags never come from text."""

import re

(
    PAD,
    BOS,
    USER,
    BOT,
    THINK,
    THINK_END,
    TOOL,
    TOOL_END,
    RESULT,
    RESULT_END,
    END,
    UNK,
    MEMORY,
    MEMORY_END,
) = range(14)
BYTE0, WORD0 = 16, 272
WORD = re.compile(r"[A-Za-z0-9_]+(?:['-][A-Za-z0-9_]+)*")


class Tokenizer:
    def __init__(self, words=()):
        if not isinstance(words, (list, tuple)) or len(words) > 8192:
            raise ValueError("word lexicon must contain at most 8192 words")
        if any(
            not isinstance(w, str)
            or not WORD.fullmatch(w)
            or not w.isascii()
            or len(w) > 128
            for w in words
        ):
            raise ValueError("invalid word")
        if len(set(words)) != len(words):
            raise ValueError("duplicate words")
        self.words = list(words)
        self.ids = {w: WORD0 + i for i, w in enumerate(words)}
        self.vocab_size = WORD0 + len(words)

    def encode(self, text):
        ids, start = [], 0
        for match in WORD.finditer(text):
            ids.extend(BYTE0 + b for b in text[start : match.start()].encode("utf-8"))
            word = match.group()
            ids.extend(
                [self.ids[word]]
                if word in self.ids
                else [BYTE0 + b for b in word.encode("utf-8")]
            )
            start = match.end()
        ids.extend(BYTE0 + b for b in text[start:].encode("utf-8"))
        return ids

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if BYTE0 <= i < WORD0:
                out.append(i - BYTE0)
            elif WORD0 <= i < self.vocab_size:
                out.extend(self.words[i - WORD0].encode("utf-8"))
            else:
                raise ValueError(f"protocol/invalid token {i} in text")
        return out.decode("utf-8", errors="replace")
