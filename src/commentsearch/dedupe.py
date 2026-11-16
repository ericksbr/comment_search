"""Near-duplicate grouping: exact (normalized-text hash), cosine on comment embeddings, and existing groups."""

from __future__ import annotations

import hashlib
import re

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text) -> str:
    return _NORM_RE.sub(" ", str(text).lower()).strip()


def exact_edges(texts) -> tuple[np.ndarray, np.ndarray]:
    """Link each comment to the first comment with the same normalized-text hash."""
    first: dict[str, int] = {}
    src: list[int] = []
    dst: list[int] = []
    for i, text in enumerate(texts):
        digest = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()
        if digest in first:
            src.append(first[digest])
            dst.append(i)
        else:
            first[digest] = i
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def near_edges(emb, threshold: float = 0.95, block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (i, j), i < j, with cosine >= threshold. Blocked so the full n x n matrix is never built."""
    emb = np.asarray(emb, dtype=np.float32)
    n = len(emb)
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    for start in range(0, n, block):
        stop = min(n, start + block)
        sims = emb[start:stop] @ emb[start:].T
        rows, cols = np.nonzero(sims >= threshold)
        rows = rows + start
        cols = cols + start
        keep = cols > rows
        src.append(rows[keep].astype(np.int64))
        dst.append(cols[keep].astype(np.int64))
    if not src:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(src), np.concatenate(dst)


def value_edges(values) -> tuple[np.ndarray, np.ndarray]:
    """Link rows that share a value in an existing group column (blank values are ignored)."""
    first: dict[str, int] = {}
    src: list[int] = []
    dst: list[int] = []
    for i, value in enumerate(values):
        key = str(value).strip()
        if key == "" or key.lower() == "nan":
            continue
        if key in first:
            src.append(first[key])
            dst.append(i)
        else:
            first[key] = i
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def build_groups(
    texts,
    emb,
    group_values=None,
    threshold: float = 0.95,
) -> tuple[np.ndarray, dict]:
    """Label every comment with a duplicate-group id.

    Groups are connected components over three kinds of links: identical normalized text,
    cosine >= ``threshold`` on the comment embedding, and equal values in ``group_values``
    (for example Grouper's ``group_id``). Returns (labels, pair counts).
    """
    n = len(texts)
    e_src, e_dst = exact_edges(texts)
    n_src, n_dst = near_edges(emb, threshold)
    parts_src = [e_src, n_src]
    parts_dst = [e_dst, n_dst]
    stats = {"exact_pairs": int(len(e_src)), "near_pairs": int(len(n_src))}
    if group_values is not None:
        g_src, g_dst = value_edges(group_values)
        parts_src.append(g_src)
        parts_dst.append(g_dst)
        stats["group_pairs"] = int(len(g_src))
    src = np.concatenate(parts_src)
    dst = np.concatenate(parts_dst)
    graph = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64), stats


def first_per_group(ranked_rows, labels) -> list[int]:
    """Keep the best-ranked row of each group, preserving order."""
    seen: set[int] = set()
    kept: list[int] = []
    for row in ranked_rows:
        group = int(labels[row])
        if group in seen:
            continue
        seen.add(group)
        kept.append(int(row))
    return kept
