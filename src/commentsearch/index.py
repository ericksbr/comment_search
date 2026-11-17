"""Index: passages, dense embeddings (cached to disk, resumable), BM25, duplicate groups. Save/load to a folder."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .dedupe import build_groups
from .ingest import make_passages

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# BGE retrieval models expect this prefix on queries (not on passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMB_FILE = "passage_emb.npy"
SENT_EMB_FILE = "sentence_emb.npy"
PROGRESS_FILE = "passage_emb.progress.json"


class STEncoder:
    """Sentence-transformers encoder. Embeddings are L2-normalized, so dot product = cosine."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, query_prefix: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.query_prefix = BGE_QUERY_PREFIX if query_prefix is None and "bge" in model_name.lower() else (query_prefix or "")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([self.query_prefix + text])[0]


def _texts_key(model_name: str, texts) -> str:
    h = hashlib.sha1(model_name.encode("utf-8"))
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def embed_passages(
    texts: list[str],
    encoder,
    cache_dir=None,
    batch_size: int = 32,
    step: int = 2048,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed passages in steps. With ``cache_dir``, writes a .npy memmap and resumes after an interruption.

    A cache is reused only when the model and the exact passage texts match.
    """
    from tqdm.auto import tqdm

    n, dim = len(texts), encoder.dim
    if n == 0:
        raise ValueError("No passages to embed")
    start = 0
    if cache_dir is None:
        out = np.empty((n, dim), dtype=np.float32)
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path, progress = cache_dir / EMB_FILE, cache_dir / PROGRESS_FILE
        key = _texts_key(encoder.name, texts)
        resumed = False
        if path.exists() and progress.exists():
            try:
                state = json.loads(progress.read_text(encoding="utf-8"))
                resumed = state.get("key") == key and state.get("n") == n and state.get("dim") == dim
                start = int(state["done"]) if resumed else 0
                resumed = resumed and np.load(path, mmap_mode="r").shape == (n, dim)
            except (ValueError, KeyError, OSError):
                resumed = False
        if resumed:
            out = np.load(path, mmap_mode="r+")
        else:
            out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, dim))
            start = 0
            progress.write_text(json.dumps({"key": key, "n": n, "dim": dim, "done": 0}), encoding="utf-8")

    bar = tqdm(total=n, initial=start, desc="Embedding passages", unit="passage", disable=not show_progress)
    for s in range(start, n, step):
        e = min(n, s + step)
        out[s:e] = encoder.encode(texts[s:e], batch_size=batch_size)
        if cache_dir is not None:
            out.flush()
            progress.write_text(json.dumps({"key": key, "n": n, "dim": dim, "done": e}), encoding="utf-8")
        bar.update(e - s)
    bar.close()
    return out


def mean_by_comment(emb, comment_row: np.ndarray, n_comments: int) -> np.ndarray:
    """Mean of each comment's passage embeddings, re-normalized."""
    out = np.zeros((n_comments, np.asarray(emb[:1]).shape[1]), dtype=np.float32)
    np.add.at(out, comment_row, np.asarray(emb, dtype=np.float32))
    counts = np.bincount(comment_row, minlength=n_comments).astype(np.float32)
    counts[counts == 0] = 1.0
    out /= counts[:, None]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def build_bm25(texts: list[str]):
    import bm25s

    from .exact import get_stemmer

    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=get_stemmer(), show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    return retriever


def make_sentences(comments: pd.DataFrame) -> pd.DataFrame:
    """Every sentence of every comment, split with NLTK Punkt. Sentences stay in comment order."""
    from .ingest import split_sentences

    rows, sent_idx, texts = [], [], []
    for row, text in enumerate(comments["text"].tolist()):
        for j, sentence in enumerate(split_sentences(text)):
            rows.append(row)
            sent_idx.append(j)
            texts.append(sentence)
    return pd.DataFrame(
        {
            "comment_row": np.asarray(rows, dtype=np.int64),
            "sent_idx": np.asarray(sent_idx, dtype=np.int64),
            "text": texts,
        }
    )


@dataclass
class CommentIndex:
    comments: pd.DataFrame          # one row per comment; includes dup_group, dup_group_size
    passages: pd.DataFrame          # comment_row, comment_id, chunk_idx, text
    passage_emb: np.ndarray         # (n_passages, dim), normalized; may be a memmap
    comment_emb: np.ndarray         # (n_comments, dim), normalized mean of passage embeddings
    bm25: object                    # bm25s.BM25
    meta: dict
    timings: dict = field(default_factory=dict)
    sentences: pd.DataFrame | None = None   # comment_row, sent_idx, text; None for indexes built before sentences
    sent_emb: np.ndarray | None = None      # (n_sentences, dim), normalized

    @property
    def encoder_name(self) -> str:
        return self.meta["model"]

    # ------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        comments: pd.DataFrame,
        encoder,
        target_words: int = 250,
        overlap_words: int = 50,
        cache_dir=None,
        group_col: str | None = None,
        dup_threshold: float = 0.95,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> "CommentIndex":
        """Chunk, embed, build BM25 and duplicate groups. Per-step times are in ``.timings``."""
        timings: dict[str, float] = {}
        t = time.perf_counter()

        def lap(name: str, start: float) -> float:
            now = time.perf_counter()
            timings[name] = round(now - start, 3)
            return now

        comments = comments.reset_index(drop=True)
        passages = make_passages(comments, target_words, overlap_words)
        t = lap("chunk", t)
        texts = passages["text"].tolist()
        emb = embed_passages(texts, encoder, cache_dir=cache_dir, batch_size=batch_size, show_progress=show_progress)
        t = lap("embed", t)
        bm25 = build_bm25(texts)
        t = lap("bm25_index", t)
        sentences = make_sentences(comments)
        sent_texts = sentences["text"].tolist()
        sent_emb = (
            encoder.encode(sent_texts, batch_size=batch_size, show_progress=show_progress)
            if sent_texts
            else np.zeros((0, encoder.dim), dtype=np.float32)
        )
        t = lap("sentence_embed", t)
        comment_emb = mean_by_comment(emb, passages["comment_row"].to_numpy(), len(comments))
        t = lap("comment_mean_emb", t)
        group_values = comments[group_col].to_numpy() if group_col else None
        labels, pair_stats = build_groups(comments["text"].tolist(), comment_emb, group_values, dup_threshold)
        comments = comments.copy()
        comments["dup_group"] = labels
        comments["dup_group_size"] = np.bincount(labels)[labels]
        lap("dedupe", t)
        meta = {
            "model": encoder.name,
            "dim": int(encoder.dim),
            "target_words": target_words,
            "overlap_words": overlap_words,
            "dup_threshold": dup_threshold,
            "group_col": group_col,
            "n_comments": int(len(comments)),
            "n_passages": int(len(passages)),
            "n_sentences": int(len(sentences)),
            "stemmer": "Porter (original algorithm, NLTK)",
            "sentence_splitter": "NLTK Punkt (punkt_tab), passages and sentences",
            "dedupe_pairs": pair_stats,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return cls(
            comments=comments,
            passages=passages,
            passage_emb=emb,
            comment_emb=comment_emb,
            bm25=bm25,
            meta=meta,
            timings=timings,
            sentences=sentences,
            sent_emb=sent_emb,
        )

    # ------------------------------------------------------------ save / load

    def save(self, folder) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.comments.to_parquet(folder / "comments.parquet", index=False)
        self.passages.to_parquet(folder / "passages.parquet", index=False)
        target = folder / EMB_FILE
        if isinstance(self.passage_emb, np.memmap) and Path(self.passage_emb.filename).resolve() == target.resolve():
            self.passage_emb.flush()
        else:
            np.save(target, np.asarray(self.passage_emb, dtype=np.float32))
        np.save(folder / "comment_emb.npy", np.asarray(self.comment_emb, dtype=np.float32))
        if self.sentences is not None and self.sent_emb is not None:
            self.sentences.to_parquet(folder / "sentences.parquet", index=False)
            np.save(folder / SENT_EMB_FILE, np.asarray(self.sent_emb, dtype=np.float32))
        self.bm25.save(str(folder / "bm25"), show_progress=False)
        (folder / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder) -> "CommentIndex":
        import bm25s

        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        has_sentences = (folder / SENT_EMB_FILE).exists() and (folder / "sentences.parquet").exists()
        return cls(
            comments=pd.read_parquet(folder / "comments.parquet"),
            passages=pd.read_parquet(folder / "passages.parquet"),
            passage_emb=np.load(folder / EMB_FILE, mmap_mode="r"),
            comment_emb=np.load(folder / "comment_emb.npy"),
            bm25=bm25s.BM25.load(str(folder / "bm25"), load_corpus=False, show_progress=False),
            meta=meta,
            sentences=pd.read_parquet(folder / "sentences.parquet") if has_sentences else None,
            sent_emb=np.load(folder / SENT_EMB_FILE) if has_sentences else None,
        )
