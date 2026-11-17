"""Load, clean, flag and chunk comments."""

from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_PUNKT = None
_BOILERPLATE_RE = re.compile(
    r"\b(?:see|refer to|review)\s+(?:the\s+|our\s+|my\s+)?(?:attached|attachment|enclosed)\b"
    r"|\battached\s+(?:comments?|letters?|files?|documents?|testimony|statements?)\b",
    re.IGNORECASE,
)
_MISSING_DATES = {"", "none", "nan", "nat", "null"}


def load_table(path) -> pd.DataFrame:
    """Read a CSV, XLSX/XLS or Parquet file. CSV and XLSX are read as text."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str).fillna("")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix} ({path})")


def clean_text(value) -> str:
    """Strip HTML tags, unescape entities and collapse whitespace."""
    if value is None or pd.isna(value):
        return ""
    text = _TAG_RE.sub(" ", str(value))
    text = html.unescape(text).replace("\xa0", " ")
    return _WS_RE.sub(" ", text).strip()


def flag_boilerplate(text: str, max_words: int = 60) -> bool:
    """True for very short comments or short 'see attached' stubs. Flagged, never removed."""
    n_words = len(text.split())
    if n_words < 5:
        return True
    return n_words <= max_words and _BOILERPLATE_RE.search(text) is not None


def _punkt():
    """NLTK's Punkt sentence tokenizer (punkt_tab), the same one the Preparer and Grouper notebooks use."""
    global _PUNKT
    if _PUNKT is None:
        import nltk
        from nltk.tokenize import PunktTokenizer

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        _PUNKT = PunktTokenizer()
    return _PUNKT


def split_sentences(text: str) -> list[str]:
    """Sentences of a comment, split with NLTK Punkt (handles abbreviations such as 'Dr.' and 'U.S.')."""
    return [s.strip() for s in _punkt().tokenize(text or "") if s.strip()]


def chunk_text(text: str, target_words: int = 250, overlap_words: int = 50) -> list[str]:
    """Split text into passages of about ``target_words`` words on sentence boundaries.

    Consecutive passages share whole trailing sentences worth up to ``overlap_words``.
    A single sentence longer than ``target_words`` is split into word windows.
    """
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    if not 0 <= overlap_words < target_words:
        raise ValueError("overlap_words must be >= 0 and < target_words")

    units: list[list[str]] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        for start in range(0, len(words), target_words):
            units.append(words[start : start + target_words])

    chunks: list[str] = []
    current: list[list[str]] = []  # units in the passage being built (includes carried overlap)
    current_words = 0
    new_units = 0  # units added since the last emit; a passage of only carried units is never emitted

    for unit in units:
        if new_units and current_words + len(unit) > target_words:
            chunks.append(" ".join(word for u in current for word in u))
            carry: list[list[str]] = []
            carry_words = 0
            for u in reversed(current):
                if carry_words + len(u) > overlap_words:
                    break
                carry.insert(0, u)
                carry_words += len(u)
            while carry and carry_words + len(unit) > target_words:
                carry_words -= len(carry.pop(0))
            current, current_words, new_units = carry, carry_words, 0
        current.append(unit)
        current_words += len(unit)
        new_units += 1

    if new_units:
        chunks.append(" ".join(word for u in current for word in u))
    return chunks


def _as_str(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def prepare_comments(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    meta_cols=(),
    date_col: str | None = None,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Return one row per comment with ``comment_id``, cleaned ``text`` and metadata.

    Empty comments are dropped (count in ``.attrs["dropped_empty"]``). ``date_col`` is parsed
    to datetime (unparseable values become NaT). ``group_col`` is kept as metadata and can
    be used by :func:`commentsearch.index.CommentIndex.build` to reuse existing duplicate groups.
    """
    meta_cols = [c for c in meta_cols]
    needed = [id_col, text_col, *meta_cols, *[c for c in (date_col, group_col) if c]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}. Available: {list(df.columns)}")

    work = df.reset_index(drop=True)
    out = pd.DataFrame({"comment_id": work[id_col].map(_as_str).to_numpy()})
    if out["comment_id"].duplicated().any():
        raise ValueError(f"{id_col} must be unique; found duplicates")
    out["text"] = [clean_text(v) for v in work[text_col]]

    for col in meta_cols:
        if col not in out.columns:
            out[col] = work[col].map(_as_str).to_numpy()
    if group_col and group_col not in out.columns:
        out[group_col] = work[group_col].map(_as_str).to_numpy()
    if date_col:
        raw = work[date_col].map(_as_str)
        raw = raw.where(~raw.str.lower().isin(_MISSING_DATES), "")
        out[date_col] = pd.to_datetime(raw, errors="coerce", format="mixed").to_numpy()

    keep = out["text"].str.len().to_numpy() > 0
    dropped = int((~keep).sum())
    out = out.loc[keep].reset_index(drop=True)
    out["boilerplate"] = [flag_boilerplate(t) for t in out["text"]]
    out.attrs["dropped_empty"] = dropped
    return out


def make_passages(comments: pd.DataFrame, target_words: int = 250, overlap_words: int = 50) -> pd.DataFrame:
    """Chunk every comment. Each passage keeps its comment row, comment_id and chunk index."""
    rows: list[int] = []
    chunk_idx: list[int] = []
    texts: list[str] = []
    for row, text in enumerate(comments["text"].tolist()):
        for j, chunk in enumerate(chunk_text(text, target_words, overlap_words)):
            rows.append(row)
            chunk_idx.append(j)
            texts.append(chunk)
    row_arr = np.asarray(rows, dtype=np.int64)
    return pd.DataFrame(
        {
            "comment_row": row_arr,
            "comment_id": comments["comment_id"].to_numpy()[row_arr] if len(row_arr) else np.array([], dtype=object),
            "chunk_idx": np.asarray(chunk_idx, dtype=np.int64),
            "text": texts,
        }
    )
