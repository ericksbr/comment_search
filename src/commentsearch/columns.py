"""Guess which column holds each field. The notebook shows the guesses; the user can override them."""

from __future__ import annotations

import pandas as pd

_ID_PREFERRED = ["CommentId", "Document ID", "comment_id", "part_id"]


def _mean_length(df: pd.DataFrame, col: str) -> float:
    return df[col].astype(str).str.len().mean()


def detect_text_column(df: pd.DataFrame) -> str:
    """The column with the longest text on average."""
    return max(df.columns, key=lambda c: _mean_length(df, c))


def detect_id_column(df: pd.DataFrame) -> str | None:
    """A column named like an ID. Returns None if nothing looks like one."""
    for name in _ID_PREFERRED:
        if name in df.columns:
            return name
    for col in df.columns:
        low = col.lower()
        if low.endswith("id") and "group" not in low and "family" not in low:
            return col
    return None


def detect_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if "date" in col.lower():
            return col
    return None


def detect_page_column(df: pd.DataFrame) -> str | None:
    """A column that holds the page number, if one looks like it (e.g. 'Page', 'page_number')."""
    for col in df.columns:
        if "page" in col.lower():
            return col
    return None


_NAME_PREFERRED = ["commenter_name", "Commenter Name", "Name", "Full Name"]
_NOT_A_PERSON = ("file", "group", "label", "agency", "organization", "org", "filename", "user")


def detect_name_column(df: pd.DataFrame) -> str | None:
    """A column that holds the commenter's name, if one looks like it. Returns None otherwise."""
    for name in _NAME_PREFERRED:
        if name in df.columns:
            return name
    for col in df.columns:
        low = col.lower()
        if "name" in low and not any(word in low for word in _NOT_A_PERSON):
            return col
    return None


def detect_group_column(df: pd.DataFrame) -> str | None:
    """Grouper's group column, if present."""
    if "group_id" in df.columns:
        return "group_id"
    for col in df.columns:
        if "group" in col.lower() and "label" not in col.lower() and "size" not in col.lower():
            return col
    return None


def suggest_filter_columns(df: pd.DataFrame, exclude: set[str], max_values: int = 100) -> list[str]:
    """Columns with a short list of values (2 to max_values), useful as filters."""
    picked = []
    for col in df.columns:
        if col in exclude:
            continue
        n = df[col].astype(str).nunique()
        if 2 <= n <= max_values:
            picked.append(col)
    return picked
