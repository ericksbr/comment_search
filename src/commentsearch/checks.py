"""Validation checks shown after a build and after loading a saved index.

Each check returns a Check with a plain PASS/FAIL result and a short explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'}  {self.name}: {self.detail}"


def check_rows_accounted(raw_rows: int, comments: pd.DataFrame) -> Check:
    dropped = int(comments.attrs.get("dropped_empty", 0))
    kept = len(comments)
    ok = raw_rows == kept + dropped
    return Check(
        "Every row is accounted for",
        ok,
        f"{raw_rows:,} rows in file = {kept:,} kept + {dropped:,} empty dropped",
    )


def check_unique_ids(comments: pd.DataFrame) -> Check:
    dupes = int(comments["comment_id"].duplicated().sum())
    return Check("Comment IDs are unique", dupes == 0, f"{dupes:,} duplicate IDs")


def check_every_comment_has_passage(comments: pd.DataFrame, passages: pd.DataFrame) -> Check:
    covered = int(passages["comment_row"].nunique())
    missing = len(comments) - covered
    return Check(
        "Every comment has at least one passage",
        missing == 0,
        f"{covered:,} of {len(comments):,} comments have passages",
    )


def _without_spaces(text: str) -> str:
    return "".join(str(text).split())


def check_passages_cover_text(comments: pd.DataFrame, passages: pd.DataFrame) -> Check:
    """Every character of each comment (ignoring whitespace) is in at least one of its passages.

    Word tokens are not compared on purpose: how a word splits at punctuation ("wells.The") or in
    citations ("ORS 192.11.023") is a tokenization choice, not a loss. This check only asks whether
    any text went missing.
    """
    passages_by_row: dict[int, list[str]] = {}
    for row, text in zip(passages["comment_row"].to_numpy(), passages["text"].to_numpy()):
        passages_by_row.setdefault(int(row), []).append(_without_spaces(text))
    bad = []
    for row, text in enumerate(comments["text"].to_numpy()):
        full = _without_spaces(text)
        covered = np.zeros(len(full), dtype=bool)
        altered = False
        for piece in passages_by_row.get(row, []):
            if not piece:
                continue
            at = full.find(piece)
            if at < 0:
                altered = True  # a passage that is not in the comment means its text changed
                break
            while at >= 0:
                covered[at : at + len(piece)] = True
                at = full.find(piece, at + 1)
        if altered or not covered.all():
            bad.append(row)
    return Check(
        "Passages cover every character of every comment (whitespace ignored)",
        not bad,
        "all text covered" if not bad else f"{len(bad):,} comments lose text when split",
    )


def check_embeddings_valid(index) -> Check:
    emb = index.passage_emb
    n = len(index.passages)
    shape_ok = emb.shape[0] == n and emb.shape[1] == index.meta["dim"]
    sample = np.asarray(emb[: min(n, 2000)], dtype=np.float32)
    finite_ok = bool(np.isfinite(sample).all())
    norms = np.linalg.norm(sample, axis=1)
    unit_ok = bool(np.allclose(norms, 1.0, atol=1e-3))
    ok = shape_ok and finite_ok and unit_ok
    return Check(
        "Embeddings are valid",
        ok,
        f"{emb.shape[0]:,} vectors of {emb.shape[1]} numbers; "
        f"{'all finite' if finite_ok else 'missing values found'}; "
        f"{'unit length' if unit_ok else 'not unit length'}",
    )


def check_duplicate_groups(index) -> Check:
    labels = index.comments["dup_group"].to_numpy()
    ok = len(labels) == len(index.comments) and labels.min() >= 0
    n_groups = int(len(np.unique(labels)))
    n_in_groups = int((index.comments["dup_group_size"] > 1).sum())
    return Check(
        "Duplicate groups assigned",
        ok,
        f"{n_groups:,} groups; {n_in_groups:,} comments belong to a group of 2 or more",
    )


def check_reload_matches(index, loaded) -> Check:
    ok = (
        len(loaded.comments) == len(index.comments)
        and len(loaded.passages) == len(index.passages)
        and loaded.meta.get("model") == index.meta.get("model")
    )
    return Check(
        "Saved index reloads unchanged",
        ok,
        f"{len(loaded.comments):,} comments and {len(loaded.passages):,} passages after reload",
    )


def self_test(
    searcher,
    n: int = 20,
    top_k: int = 5,
    seed: int = 0,
    min_words: int = 8,
    method: str = "rrf",
    min_share: float = 0.9,
) -> Check:
    """Search one unique sentence from each of up to n random comments.

    The source comment should appear in the top_k. Uses the same default method as the search window
    (hybrid RRF) and passes at min_share or better. On the OWRD test file, keyword search found 37 of
    38 such sentences, hybrid 35 of 38, and dense-only 30 of 38.
    """
    from .ingest import split_sentences

    comments = searcher.index.comments
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(comments["text"].str.split().str.len().to_numpy() >= min_words)
    if candidates.size == 0:
        return Check("Self-test: a sentence finds its own comment", False, "no comments long enough to test")
    all_text = comments["text"].astype(str)
    tested = hits = 0
    for row in rng.permutation(candidates):
        if tested >= n:
            break
        text = str(comments["text"].iloc[row])
        # Use a sentence that appears in no other comment. Shared boilerplate would match other comments
        # first and make the test fail for a reason that has nothing to do with the search.
        query = None
        for sentence in split_sentences(text):
            if len(sentence.split()) < min_words:
                continue
            if int(all_text.str.contains(sentence, regex=False).sum()) == 1:
                query = sentence
                break
        if query is None:
            continue
        tested += 1
        frame = searcher.search(query, top_k=top_k, method=method, collapse=False)
        if comments["comment_id"].iloc[row] in set(frame["comment_id"]):
            hits += 1
    name = "Self-test: a sentence finds its own comment"
    if tested == 0:
        return Check(name, False, "no sentence in these comments is unique enough to test")
    share = hits / tested
    return Check(
        name,
        share >= min_share,
        f"{hits} of {tested} sentences (each unique to one comment) found their comment in the top {top_k}"
        f" with {method}; passes at {min_share:.0%} or better",
    )


def run_build_checks(raw_rows: int, index, searcher=None) -> list[Check]:
    checks = [
        check_rows_accounted(raw_rows, index.comments),
        check_unique_ids(index.comments),
        check_every_comment_has_passage(index.comments, index.passages),
        check_passages_cover_text(index.comments, index.passages),
        check_embeddings_valid(index),
        check_duplicate_groups(index),
    ]
    if searcher is not None:
        checks.append(self_test(searcher))
    return checks
