"""Retrieval.

Two methods serve the search window:
- ``exact``: every word (stemmed) must appear, quoted phrases in order. Engine: ``fts5`` or ``bm25s``.
- ``semantic``: passages whose embedding is at least ``min_similarity`` (cosine) from the query.

The older fusion methods (``rrf``, ``blend``, ``dense``, ``bm25``) are kept for tests and comparisons.
"""

from __future__ import annotations

import html
import re
import time

import numpy as np
import pandas as pd

from .dedupe import first_per_group
from .exact import MODE_ALL, build_fts, get_stemmer, match_fts, parse_query, passage_hits, StemPostings

DEFAULT_RERANKER = "BAAI/bge-reranker-base"
METHODS = ("rrf", "blend", "dense", "bm25", "exact", "semantic", "sentence")
DISPLAY_META = ["commenter_name", "group_label", "family_id", "file", "date"]

_STOP = frozenset(
    "the and for with that this are was were from have has not but you your our their they them "
    "what which who how why when where about into over under than then there these those will "
    "would can could should also just any all".split()
)
_SKIP_META = {"comment_id", "text", "dup_group", "dup_group_size"}


# ---------------------------------------------------------------- text helpers

def query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for word in re.findall(r"[a-z0-9]+", query.lower()):
        if len(word) >= 3 and word not in _STOP and word not in terms:
            terms.append(word)
    return terms


def make_snippet(text: str, terms: list[str], width: int = 400) -> str:
    """A window of about ``width`` characters around the first query term."""
    if len(text) <= width:
        return text
    low = text.lower()
    pos = -1
    for term in terms:
        found = low.find(term)
        if found != -1 and (pos == -1 or found < pos):
            pos = found
    start = 0 if pos == -1 else max(0, pos - width // 4)
    snippet = text[start : start + width].strip()
    prefix = "… " if start > 0 else ""
    suffix = " …" if start + width < len(text) else ""
    return prefix + snippet + suffix


def highlight(snippet: str, terms: list[str]) -> str:
    """HTML-escape the snippet and wrap query terms in <mark>."""
    escaped = html.escape(snippet)
    if not terms:
        return escaped
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)


# ---------------------------------------------------------------- fusion (pure functions)

def minmax_norm(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Min-max scale the valid entries to [0, 1]. Constant scores map to 0."""
    out = np.zeros(scores.shape, dtype=np.float64)
    values = scores[valid]
    if values.size == 0:
        return out
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return out
    out[valid] = (scores[valid] - lo) / (hi - lo)
    return out


def blend_scores(dense: np.ndarray, bm25: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    """alpha * minmax(dense) + (1 - alpha) * minmax(bm25). Invalid entries become -inf."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    fused = alpha * minmax_norm(dense, valid) + (1.0 - alpha) * minmax_norm(bm25, valid)
    fused[~valid] = -np.inf
    return fused


def rrf_fuse(rankings: list[np.ndarray], k: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank), rank starting at 1."""
    contrib: dict[int, float] = {}
    for ranking in rankings:
        for position, idx in enumerate(ranking):
            contrib[int(idx)] = contrib.get(int(idx), 0.0) + 1.0 / (k + position + 1)
    ids = sorted(contrib, key=lambda i: (-contrib[i], i))
    return np.asarray(ids, dtype=np.int64), np.asarray([contrib[i] for i in ids], dtype=np.float64)


def top_indices(scores: np.ndarray, n: int, min_score: float = -np.inf) -> np.ndarray:
    """Indices of the n largest finite scores above ``min_score``, best first."""
    candidates = np.flatnonzero(np.isfinite(scores) & (scores > min_score))
    if candidates.size == 0:
        return candidates.astype(np.int64)
    if candidates.size > n:
        candidates = candidates[np.argpartition(-scores[candidates], n - 1)[:n]]
    return candidates[np.argsort(-scores[candidates], kind="stable")].astype(np.int64)


# ---------------------------------------------------------------- filters

def _is_empty(cond) -> bool:
    if cond is None:
        return True
    if isinstance(cond, str):
        return cond.strip() == ""
    if isinstance(cond, (list, tuple, set)):
        return len(cond) == 0 or all(c is None for c in cond)
    return False


def filter_mask(comments: pd.DataFrame, filters: dict | None = None) -> np.ndarray:
    """Boolean mask over comments.

    ``{col: value}`` or ``{col: [v1, v2]}`` matches equal / listed values (compared as text).
    ``{col: (lo, hi)}`` is an inclusive range for dates; use None for an open end.
    """
    mask = np.ones(len(comments), dtype=bool)
    for col, cond in (filters or {}).items():
        if _is_empty(cond):
            continue
        if col not in comments.columns:
            raise KeyError(f"Filter column not found: {col}")
        series = comments[col]
        if isinstance(cond, tuple) and len(cond) == 2:
            lo, hi = cond
            dates = series if pd.api.types.is_datetime64_any_dtype(series) else pd.to_datetime(series, errors="coerce", format="mixed")
            ok = np.ones(len(comments), dtype=bool)
            if lo is not None:
                ok &= (dates >= pd.Timestamp(lo)).to_numpy()
            if hi is not None:
                ok &= (dates <= pd.Timestamp(hi)).to_numpy()
            mask &= ok
        elif isinstance(cond, (list, set)):
            mask &= series.astype(str).isin([str(c) for c in cond]).to_numpy()
        else:
            mask &= (series.astype(str) == str(cond)).to_numpy()
    return mask


# ---------------------------------------------------------------- searcher

class Searcher:
    """Search a :class:`commentsearch.index.CommentIndex`.

    ``encoder`` defaults to the model named in the index metadata. The reranker is loaded
    the first time ``rerank=True`` is used. Keyword engines are built the first time they are used.
    """

    def __init__(self, index, encoder=None, reranker_name: str = DEFAULT_RERANKER):
        self.index = index
        self._encoder = encoder
        self.reranker_name = reranker_name
        self._reranker = None
        self._p_row = index.passages["comment_row"].to_numpy()
        self._p_text = index.passages["text"].to_numpy()
        self._p_chunk = index.passages["chunk_idx"].to_numpy()
        self._labels = index.comments["dup_group"].to_numpy()
        self._meta_cols = [c for c in index.comments.columns if c not in _SKIP_META]
        self._texts = index.comments["text"].tolist()
        # Passage ids of each comment, in one array per comment.
        order = np.argsort(self._p_row, kind="stable")
        counts = np.bincount(self._p_row, minlength=len(self._texts))
        self._row_pids = np.split(order, np.cumsum(counts)[:-1])
        self._postings: StemPostings | None = None
        self._fts = None
        sents = index.sentences
        self._s_row = sents["comment_row"].to_numpy() if sents is not None else None
        self._s_text = sents["text"].to_numpy() if sents is not None else None
        self._s_emb = index.sent_emb if sents is not None else None
        if sents is not None:
            comment_ids = np.arange(len(self._texts))
            self._s_start = np.searchsorted(self._s_row, comment_ids, side="left")
            self._s_end = np.searchsorted(self._s_row, comment_ids, side="right")
        else:
            self._s_start = self._s_end = None

    @property
    def encoder(self):
        if self._encoder is None:
            from .index import STEncoder

            self._encoder = STEncoder(self.index.meta["model"])
        return self._encoder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_name)
        return self._reranker

    @property
    def has_sentences(self) -> bool:
        return self._s_emb is not None

    def search(
        self,
        query: str,
        top_k: int = 10,
        method: str = "rrf",
        alpha: float = 0.5,
        rerank: bool = False,
        collapse: bool = True,
        filters: dict | None = None,
        rerank_top: int = 50,
        rrf_k: int = 60,
        pool: int | None = None,
        backend: str = "fts5",
        min_similarity: float = 0.0,
        exact_mode: str = MODE_ALL,
    ) -> pd.DataFrame:
        """Return the top comments. Per-step timings are in ``df.attrs["timings"]``.

        Filters are applied before ranking. Each comment is scored by its best passage.
        With ``collapse=True`` only the best-ranked comment of each duplicate group is shown,
        and ``search_group_size`` counts the group members that pass the filters.
        ``method="exact"`` uses ``backend`` ("fts5" or "bm25s"); ``method="semantic"`` keeps passages
        with cosine similarity >= ``min_similarity``.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        def lap(name: str, start: float) -> float:
            now = time.perf_counter()
            timings[name] = round(now - start, 4)
            return now

        cmask = filter_mask(self.index.comments, filters)
        pmask = cmask[self._p_row]
        t = lap("filter", t0)

        if method == "exact":
            return self._search_exact(query, backend, cmask, top_k, collapse, timings, t0, exact_mode)
        if method == "semantic":
            return self._search_semantic(query, min_similarity, pmask, cmask, top_k, collapse, timings, t0)
        if method == "sentence":
            return self._search_sentence(query, min_similarity, cmask, top_k, collapse, timings, t0)

        terms = query_terms(query)
        pool = pool or max(200, top_k * 10)
        empty = self._empty(terms, timings, t0)
        if not pmask.any() or not query.strip():
            return empty

        s_dense = None
        s_bm25 = None
        if method != "bm25":
            q_vec = self.encoder.encode_query(query)
            t = lap("encode_query", t)
            s_dense = np.where(pmask, np.asarray(self.index.passage_emb, dtype=np.float32) @ q_vec, -np.inf)
            t = lap("dense_score", t)
        if method != "dense":
            s_bm25 = self._bm25_scores(query, pmask)
            t = lap("bm25_score", t)

        if method == "dense":
            ids = top_indices(s_dense, pool)
            scores = s_dense[ids]
        elif method == "bm25":
            ids = top_indices(s_bm25, pool, min_score=0.0)
            scores = s_bm25[ids]
        elif method == "blend":
            fused = blend_scores(s_dense, s_bm25, pmask, alpha)
            ids = top_indices(fused, pool)
            scores = fused[ids]
        else:
            r_dense = top_indices(s_dense, pool)
            r_bm25 = top_indices(s_bm25, pool, min_score=0.0)
            ids, scores = rrf_fuse([r_dense, r_bm25], k=rrf_k)
            ids, scores = ids[:pool], scores[:pool]
        t = lap("fuse", t)

        if rerank and len(ids):
            head = ids[:rerank_top]
            ce = np.asarray(
                self.reranker.predict([(query, self._p_text[i]) for i in head], batch_size=16, show_progress_bar=False),
                dtype=np.float64,
            )
            order = np.argsort(-ce, kind="stable")
            # Reranked passages go first. Passages below the top `rerank_top` keep their fused scores.
            ids = np.concatenate([head[order], ids[rerank_top:]])
            scores = np.concatenate([ce[order], np.asarray(scores)[rerank_top:]])
            t = lap("rerank", t)

        best: dict[int, tuple[int, float]] = {}
        ordered: list[int] = []
        for pid, score in zip(ids, scores):
            row = int(self._p_row[pid])
            if row in best:
                continue
            best[row] = (int(pid), float(score))
            ordered.append(row)
        t = lap("roll_up", t)
        return self._frame(ordered, best, terms, cmask, collapse, top_k, timings, t0, method)

    def compare(self, query: str, top_k: int = 10, filters: dict | None = None, collapse: bool = True) -> dict[str, pd.DataFrame]:
        """Dense only, BM25 only, hybrid (RRF) and hybrid + rerank, for side-by-side viewing."""
        return {
            "dense": self.search(query, top_k, method="dense", collapse=collapse, filters=filters),
            "bm25": self.search(query, top_k, method="bm25", collapse=collapse, filters=filters),
            "hybrid": self.search(query, top_k, method="rrf", collapse=collapse, filters=filters),
            "hybrid+rerank": self.search(query, top_k, method="rrf", rerank=True, collapse=collapse, filters=filters),
        }

    def compare_exact(self, query: str, show: int = 50) -> dict:
        """Which comments each exact engine finds for the same query, and the ones only one engine finds."""
        rows_fts = set(self._exact_rows(query, "fts5")[0].tolist())
        rows_bm = set(self._exact_rows(query, "bm25s")[0].tolist())
        ids = self.index.comments["comment_id"].to_numpy()

        def names(rows):
            return [str(ids[r]) for r in sorted(rows)[:show]]

        return {
            "fts5": len(rows_fts),
            "bm25s": len(rows_bm),
            "both": len(rows_fts & rows_bm),
            "only_fts5": names(rows_fts - rows_bm),
            "only_bm25s": names(rows_bm - rows_fts),
            "only_fts5_count": len(rows_fts - rows_bm),
            "only_bm25s_count": len(rows_bm - rows_fts),
        }

    def sentence_similarities(self, query: str, comment_rows) -> dict[int, list[tuple[str, float]]]:
        """Cosine similarity of each sentence of the given comments to the query, in comment order."""
        if self._s_emb is None or len(comment_rows) == 0:
            return {}
        q_vec = self.encoder.encode_query(query)
        idx = np.flatnonzero(np.isin(self._s_row, np.asarray(list(comment_rows), dtype=np.int64)))
        scores = np.asarray(self._s_emb[idx] @ q_vec, dtype=np.float64)
        out: dict[int, list[tuple[str, float]]] = {}
        for i, s in zip(idx, scores):
            out.setdefault(int(self._s_row[i]), []).append((str(self._s_text[i]), float(s)))
        return out

    # ------------------------------------------------------------ exact and semantic

    def _exact_rows(self, query: str, backend: str, mode: str = MODE_ALL) -> tuple[np.ndarray, np.ndarray, dict]:
        """Comment rows that match, their scores (higher is better), and the parsed query."""
        plan = parse_query(query)
        if backend == "fts5":
            if self._fts is None:
                self._fts = build_fts(self._texts)
            rows, scores = match_fts(self._fts, plan, mode)
        elif backend == "bm25s":
            if self._postings is None:
                self._postings = StemPostings(self._texts)
            rows = self._postings.match(plan, mode)
            scores = self._bm25_comment_scores(plan, rows)
        else:
            raise ValueError("backend must be 'fts5' or 'bm25s'")
        return rows, scores, plan

    def _bm25_comment_scores(self, plan: dict, rows: np.ndarray) -> np.ndarray:
        """Best passage BM25 score per comment, for the given rows."""
        words = plan["loose"] + [w for p in plan["phrases"] for w in p]
        if not words or rows.size == 0:
            return np.zeros(len(rows), dtype=np.float64)
        passage_scores = self._bm25_scores(" ".join(words), np.ones(len(self._p_row), dtype=bool))
        per_comment = np.zeros(len(self._texts), dtype=np.float64)
        np.maximum.at(per_comment, self._p_row, passage_scores)
        return per_comment[rows]

    def _search_exact(self, query, backend, cmask, top_k, collapse, timings, t0, mode=MODE_ALL):
        t = time.perf_counter()
        rows, scores, plan = self._exact_rows(query, backend, mode)
        keep = cmask[rows]
        rows, scores = rows[keep], scores[keep]
        order = np.lexsort((rows, -scores))
        rows = rows[order]
        timings[f"{backend}_match"] = round(time.perf_counter() - t, 4)
        terms = plan["loose"] + [w for p in plan["phrases"] for w in p]
        best: dict[int, tuple[int, float]] = {}
        ordered: list[int] = []
        score_of = dict(zip(rows.tolist(), scores[order].tolist()))
        for row in rows.tolist():
            pids = self._row_pids[row]
            if pids.size == 0:
                continue
            hits = [passage_hits(str(self._p_text[p]), plan) for p in pids]
            pid = int(pids[int(np.argmax(hits))])
            best[row] = (pid, float(score_of[row]))
            ordered.append(row)
        frame = self._frame(ordered, best, terms, cmask, collapse, top_k, timings, t0, "exact", plan=plan)
        return frame

    def _search_semantic(self, query, min_similarity, pmask, cmask, top_k, collapse, timings, t0):
        t = time.perf_counter()
        terms = query_terms(query)
        if not query.strip() or not pmask.any():
            return self._empty(terms, timings, t0)
        q_vec = self.encoder.encode_query(query)
        timings["encode_query"] = round(time.perf_counter() - t, 4)
        s_dense = np.where(pmask, np.asarray(self.index.passage_emb, dtype=np.float32) @ q_vec, -np.inf)
        ids = np.flatnonzero(s_dense >= min_similarity)
        ids = ids[np.argsort(-s_dense[ids], kind="stable")]
        best: dict[int, tuple[int, float]] = {}
        ordered: list[int] = []
        for pid in ids:
            row = int(self._p_row[pid])
            if row in best:
                continue
            best[row] = (int(pid), float(s_dense[pid]))
            ordered.append(row)
        return self._frame(ordered, best, terms, cmask, collapse, top_k, timings, t0, "semantic",
                           plan={"min_similarity": float(min_similarity), "query": query})

    def _search_sentence(self, query, min_similarity, cmask, top_k, collapse, timings, t0):
        """Comments with at least one sentence at or above min_similarity. Score = best sentence."""
        terms = query_terms(query)
        if not query.strip() or self._s_emb is None or self._s_row.size == 0:
            return self._empty(terms, timings, t0)
        q_vec = self.encoder.encode_query(query)
        sims = np.asarray(self._s_emb @ q_vec, dtype=np.float64)
        sims = np.where(cmask[self._s_row], sims, -np.inf)
        best = np.full(len(self._texts), -np.inf)
        np.maximum.at(best, self._s_row, sims)
        rows = np.flatnonzero(best >= min_similarity)
        rows = rows[np.argsort(-best[rows], kind="stable")]
        best_map: dict[int, tuple[int, float]] = {}
        ordered: list[int] = []
        for row in rows.tolist():
            lo, hi = int(self._s_start[row]), int(self._s_end[row])
            top_sentence = str(self._s_text[lo + int(np.argmax(sims[lo:hi]))])
            best_map[row] = (self._sentence_passage(row, top_sentence, q_vec), float(best[row]))
            ordered.append(row)
        return self._frame(ordered, best_map, terms, cmask, collapse, top_k, timings, t0, "sentence",
                           plan={"min_similarity": float(min_similarity), "query": query})

    def _sentence_passage(self, row: int, sentence: str, q_vec) -> int:
        """The passage that contains the sentence (so the Passage view shows it). Falls back to the closest passage."""
        pids = self._row_pids[row]
        norm = " ".join(sentence.split())
        for pid in pids.tolist():
            if norm in " ".join(str(self._p_text[pid]).split()):
                return int(pid)
        scores = np.asarray(self.index.passage_emb[pids] @ q_vec)
        return int(pids[int(np.argmax(scores))])

    def _frame(self, ordered, best, terms, cmask, collapse, top_k, timings, t0, method, plan=None):
        """Turn the chosen passage per comment into the results table (one row per comment)."""
        if collapse:
            ordered = first_per_group(ordered, self._labels)
        ordered = ordered[:top_k]
        group_counts = np.bincount(self._labels[cmask], minlength=int(self._labels.max()) + 1)
        if not ordered:
            return self._empty(terms, timings, t0)
        passage_ids = [best[r][0] for r in ordered]
        frame = pd.DataFrame(
            {
                "rank": np.arange(1, len(ordered) + 1),
                "score": [best[r][1] for r in ordered],
                "comment_id": self.index.comments["comment_id"].to_numpy()[ordered],
                "snippet": [make_snippet(self._p_text[p], terms) for p in passage_ids],
                "passage": [self._p_text[p] for p in passage_ids],
                "chunk_idx": self._p_chunk[passage_ids],
                "search_group_size": group_counts[self._labels[ordered]],
            }
        )
        meta = self.index.comments.iloc[ordered][self._meta_cols].reset_index(drop=True)
        frame = pd.concat([frame, meta], axis=1)
        timings["total"] = round(time.perf_counter() - t0, 4)
        frame.attrs.update(
            {
                "timings": timings,
                "query_terms": terms,
                "method": method,
                "rows": np.asarray(ordered, dtype=np.int64),
                "plan": plan or {},
            }
        )
        return frame

    # ------------------------------------------------------------ internals

    def _bm25_scores(self, query: str, pmask: np.ndarray) -> np.ndarray:
        import bm25s

        retriever = self.index.bm25
        tokens = bm25s.tokenize([query], stopwords="en", stemmer=get_stemmer(), return_ids=False, show_progress=False)[0]
        vocab = getattr(retriever, "vocab_dict", None) or {}
        tokens = [tok for tok in tokens if tok in vocab]
        if not tokens:
            return np.zeros(len(pmask), dtype=np.float64)
        scores = np.asarray(retriever.get_scores(tokens), dtype=np.float64)
        return np.where(pmask, scores, 0.0)

    def _empty(self, terms, timings, t0):
        timings["total"] = round(time.perf_counter() - t0, 4)
        cols = ["rank", "score", "comment_id", "snippet", "passage", "chunk_idx", "search_group_size", *self._meta_cols]
        frame = pd.DataFrame(columns=cols)
        frame.attrs.update({"timings": timings, "query_terms": terms, "rows": np.array([], dtype=np.int64), "plan": {}})
        return frame
