# 05_rag/tools/rag_utils.py
"""Pure-Python RAG helpers for Navasena Module 05.

GPU/model-free so they can be unit-tested locally (CPU) and imported by the
Colab notebooks. This file is the SOURCE OF TRUTH — notebooks import these
helpers and must not re-implement them.

Note on offsets: Chunk.start/end are approximate char offsets into the
space-normalised text and are not currently used to slice the original document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORDish = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def simple_token_count(text: str) -> int:
    """Whitespace+punctuation token estimate. Notebooks override with a real tokenizer."""
    return len(_WORDish.findall(text))


@dataclass
class TokenCounter:
    """Counts tokens via an injectable function (default: `simple_token_count`).

    In notebooks: TokenCounter(tokenize_fn=lambda t: len(qwen_tok.encode(t))).
    """
    tokenize_fn: Callable[[str], int] = simple_token_count

    def count(self, text: str) -> int:
        return self.tokenize_fn(text)


def split_sentences(text: str) -> List[str]:
    """Split on sentence terminators; trims and drops empties."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


@dataclass
class Chunk:
    text: str
    start: int  # approximate char offset into the space-normalized text (not for slicing the raw source)
    end: int
    n_tokens: int


class TextChunker:
    """Three chunking strategies, all token-budgeted via a TokenCounter.

    Offsets (Chunk.start/end) are approximate and not currently used to slice
    the original document — they index the space-normalized (joined) text.
    """

    def __init__(self, counter: TokenCounter, max_tokens: int = 128, overlap_words: int = 24):
        # overlap_words is measured in WORDS, not subword tokens; with a subword
        # tokenizer the actual token overlap will be larger than this value.
        if not (0 <= overlap_words < max_tokens):
            raise ValueError("overlap_words must satisfy 0 <= overlap_words < max_tokens")
        self.counter = counter
        self.max_tokens = max_tokens
        self.overlap_words = overlap_words

    def _emit(self, segments: List[str], cursor: int) -> Chunk:
        text = " ".join(segments)
        return Chunk(text=text, start=cursor, end=cursor + len(text), n_tokens=self.counter.count(text))

    def fixed(self, text: str) -> List[Chunk]:
        """Sliding word window packed to the token budget, with word overlap."""
        words = text.split()
        if not words:
            return []
        chunks: List[Chunk] = []
        i, cursor = 0, 0
        while i < len(words):
            window: List[str] = []
            j = i
            while j < len(words) and self.counter.count(" ".join(window + [words[j]])) <= self.max_tokens:
                window.append(words[j])
                j += 1
            if not window:  # single word exceeds budget — emit it alone to make progress
                window = [words[i]]
                j = i + 1
            chunks.append(self._emit(window, cursor))
            if j >= len(words):
                break
            # step back `overlap_words` words for the next window
            step = max(1, len(window) - self.overlap_words)
            consumed = " ".join(words[i:i + step])
            cursor += len(consumed) + 1
            i += step
        return chunks

    def sentence(self, text: str) -> List[Chunk]:
        """Greedily pack whole sentences until the token budget is reached.

        A sentence that alone exceeds max_tokens is emitted as a single oversized chunk (no sub-word splitting).
        """
        sents = split_sentences(text)
        chunks: List[Chunk] = []
        buf: List[str] = []
        cursor = 0
        for s in sents:
            trial = " ".join(buf + [s])
            if buf and self.counter.count(trial) > self.max_tokens:
                chunks.append(self._emit(buf, cursor))
                cursor += len(" ".join(buf)) + 1
                buf = [s]
            else:
                buf.append(s)
        if buf:
            chunks.append(self._emit(buf, cursor))
        return chunks

    def semantic(self, text: str) -> List[Chunk]:
        """Paragraph-based: split on blank lines; each paragraph becomes its own chunk.

        Paragraphs that exceed the token budget are sub-chunked via `sentence()`.
        """
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[Chunk] = []
        cursor = 0
        for p in paras:
            if self.counter.count(p) > self.max_tokens:
                # oversized paragraph: fall back to sentence chunking
                sub = self.sentence(p)
                for c in sub:
                    chunks.append(Chunk(text=c.text, start=cursor + c.start,
                                        end=cursor + c.end, n_tokens=c.n_tokens))
            else:
                chunks.append(Chunk(text=p, start=cursor, end=cursor + len(p),
                                    n_tokens=self.counter.count(p)))
            cursor += len(p) + 2  # +2 approximates the blank-line separator; offsets may drift for non-standard gaps
        return chunks


def chunk_quality_score(chunks: List[Chunk], max_tokens: int, original_len: int) -> float:
    """0-100 quality score combining three sub-metrics:

    - budget_adherence: fraction of chunks whose n_tokens <= max_tokens
    - size_consistency:  1 - coefficient_of_variation(n_tokens), clamped to [0,1]
    - coverage:          min(1, sum(chunk_chars) / original_len)  (overlap can push >1)

    Returns 0.0 for an empty chunk list.
    """
    if not chunks:
        return 0.0
    sizes = [c.n_tokens for c in chunks]
    budget_adherence = sum(1 for n in sizes if n <= max_tokens) / len(sizes)
    m = mean(sizes)
    cov = (pstdev(sizes) / m) if m > 0 else 1.0
    size_consistency = max(0.0, min(1.0, 1.0 - cov))
    total_chars = sum(len(c.text) for c in chunks)
    coverage = min(1.0, total_chars / original_len) if original_len > 0 else 0.0
    score = (0.4 * budget_adherence + 0.3 * size_consistency + 0.3 * coverage) * 100.0
    return round(score, 1)
