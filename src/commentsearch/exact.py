"""Exact (keyword) matching. Every word must appear (after stemming); "quoted phrases" must appear in that order.

Two engines follow the same rule so their results can be compared:
- ``bm25s``: our own inverted index of stems (NLTK's original Porter stemmer), BM25 scores from the passage index.
- ``fts5``: SQLite's built-in full-text search (its Porter stemmer, the same algorithm), BM25 scores.
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache

import numpy as np
from bm25s.stopwords import STOPWORDS_EN

WORD_RE = re.compile(r"[A-Za-z0-9]+")
PHRASE_RE = re.compile(r'"([^"]+)"')
STOP = frozenset(STOPWORDS_EN)
ENGINES = ("fts5", "bm25s")
# "all": every word and phrase must appear (AND). "any": at least one must appear (OR).
MODE_ALL = "all"
MODE_ANY = "any"


class PorterWords:
    """Original Porter stemmer (NLTK), the same algorithm SQLite's FTS5 ``porter`` tokenizer uses.

    Cached per word, so each distinct word is stemmed once. bm25s accepts an object with ``stemWords``.
    """

    def __init__(self):
        from nltk.stem.porter import PorterStemmer

        self._stemmer = PorterStemmer(PorterStemmer.ORIGINAL_ALGORITHM)
        self._cache: dict[str, str] = {}

    def stem(self, word: str) -> str:
        if word not in self._cache:
            self._cache[word] = self._stemmer.stem(word)
        return self._cache[word]

    def stemWords(self, words: list[str]) -> list[str]:
        return [self.stem(w) for w in words]


@lru_cache(maxsize=1)
def get_stemmer() -> PorterWords:
    return PorterWords()


def stem_words(words: list[str]) -> list[str]:
    return get_stemmer().stemWords([w.lower() for w in words]) if words else []


def parse_query(query: str) -> dict:
    """Split a query into loose words (stop words dropped) and quoted phrases (kept whole).

    Returns lowercase words, plus their stems: ``loose`` / ``loose_stems`` / ``phrases`` / ``phrase_stems``.
    """
    phrases = [WORD_RE.findall(p.lower()) for p in PHRASE_RE.findall(query or "")]
    phrases = [p for p in phrases if p]
    rest = PHRASE_RE.sub(" ", query or "")
    loose: list[str] = []
    for w in WORD_RE.findall(rest.lower()):
        if w not in STOP and w not in loose:
            loose.append(w)
    return {
        "loose": loose,
        "loose_stems": stem_words(loose),
        "phrases": phrases,
        "phrase_stems": [stem_words(p) for p in phrases],
    }


def is_empty(plan: dict) -> bool:
    return not plan["loose"] and not plan["phrases"]


def _phrase_at(stems: list[str], phrase: list[str]) -> bool:
    k = len(phrase)
    return any(stems[i : i + k] == phrase for i in range(len(stems) - k + 1))


# ---------------------------------------------------------------- bm25s engine: stem postings

class StemPostings:
    """Which comments contain each stem. Built once per index; a query is a set intersection."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.stem_of: dict[str, str] = {}
        postings: dict[str, list[int]] = {}
        for row, text in enumerate(texts):
            words = set(WORD_RE.findall(text.lower()))
            new = [w for w in words if w not in self.stem_of]
            for w, s in zip(new, stem_words(new)):
                self.stem_of[w] = s
            for s in {self.stem_of[w] for w in words}:
                postings.setdefault(s, []).append(row)
        self.postings = {s: np.asarray(rows, dtype=np.int64) for s, rows in postings.items()}

    def stems_of_text(self, text: str) -> list[str]:
        return [self.stem_of.get(w, "") for w in WORD_RE.findall(text.lower())]

    def _postings_all(self, stems: list[str]) -> np.ndarray:
        """Rows of comments that contain every one of the stems."""
        rows = None
        for stem in dict.fromkeys(stems):
            found = self.postings.get(stem, np.array([], dtype=np.int64))
            rows = found if rows is None else np.intersect1d(rows, found, assume_unique=True)
            if rows.size == 0:
                break
        return np.arange(len(self.texts), dtype=np.int64) if rows is None else rows

    def _phrase_rows(self, phrase_stems: list[str]) -> np.ndarray:
        """Rows of comments that contain the phrase, words in order."""
        rows = self._postings_all(phrase_stems)
        keep = [r for r in rows if _phrase_at(self.stems_of_text(self.texts[r]), phrase_stems)]
        return np.asarray(keep, dtype=np.int64)

    def match(self, plan: dict, mode: str = MODE_ALL) -> np.ndarray:
        """Rows of comments that match: all of the words and phrases (mode 'all'), or any of them ('any')."""
        if is_empty(plan):
            return np.array([], dtype=np.int64)
        if mode == MODE_ANY:
            parts = [self._postings_all([s]) for s in plan["loose_stems"]]
            parts += [self._phrase_rows(ps) for ps in plan["phrase_stems"]]
            out = np.array([], dtype=np.int64)
            for part in parts:
                out = np.union1d(out, part)
            return out.astype(np.int64)
        # Phrase words (stop words included) are a cheap pre-filter; the exact order is checked below.
        rows = self._postings_all(plan["loose_stems"] + [s for ps in plan["phrase_stems"] for s in ps])
        if plan["phrases"]:
            keep = [
                r for r in rows
                if all(_phrase_at(self.stems_of_text(self.texts[r]), ps) for ps in plan["phrase_stems"])
            ]
            rows = np.asarray(keep, dtype=np.int64)
        return rows


# ---------------------------------------------------------------- fts5 engine

def build_fts(texts: list[str]) -> sqlite3.Connection:
    """In-memory SQLite table, one row per comment (rowid = comment row)."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='porter unicode61 remove_diacritics 2')")
    conn.executemany("INSERT INTO docs(rowid, body) VALUES (?, ?)", enumerate(texts))
    conn.commit()
    return conn


def fts_query(plan: dict, mode: str = MODE_ALL) -> str:
    """FTS5 syntax: each loose word and each phrase is a quoted term. Adjacent terms are ANDed, or ORed in 'any' mode."""
    parts = [f'"{w}"' for w in plan["loose"]] + [f'"{" ".join(p)}"' for p in plan["phrases"]]
    return (" OR " if mode == MODE_ANY else " ").join(parts)


def match_fts(conn: sqlite3.Connection, plan: dict, mode: str = MODE_ALL) -> tuple[np.ndarray, np.ndarray]:
    """Rows and BM25 scores (higher is better), most relevant first."""
    if is_empty(plan):
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    rows = conn.execute(
        "SELECT rowid, -bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY bm25(docs)", (fts_query(plan, mode),)
    ).fetchall()
    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0].astype(np.int64), arr[:, 1]


# ---------------------------------------------------------------- highlighting

def highlight_exact(text: str, plan: dict) -> str:
    """HTML: escaped text with each matched word (or phrase word) wrapped in <mark class="cs-exact">."""
    import html

    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    stems = stem_words([text[s:e] for s, e in spans])
    hit = [False] * len(spans)
    loose = set(plan["loose_stems"])
    for i, s in enumerate(stems):
        if s in loose:
            hit[i] = True
    for ps in plan["phrase_stems"]:
        k = len(ps)
        for i in range(len(stems) - k + 1):
            if stems[i : i + k] == ps:
                for j in range(i, i + k):
                    hit[j] = True
    out, pos = [], 0
    for (s, e), h in zip(spans, hit):
        out.append(html.escape(text[pos:s]))
        word = html.escape(text[s:e])
        out.append(f'<mark class="cs-exact">{word}</mark>' if h else word)
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


def passage_hits(text: str, plan: dict) -> int:
    """How many matched words (or phrase words) a passage contains. Used to pick the passage to show."""
    stems = stem_words(WORD_RE.findall(text))
    loose = set(plan["loose_stems"])
    n = sum(1 for s in stems if s in loose)
    for ps in plan["phrase_stems"]:
        if _phrase_at(stems, ps):
            n += len(ps)
    return n
