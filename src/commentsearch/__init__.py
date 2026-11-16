"""Local hybrid (dense + BM25) search over public comments."""

from .ingest import chunk_text, clean_text, load_table, make_passages, prepare_comments
from .index import CommentIndex, STEncoder
from .search import Searcher

__all__ = [
    "CommentIndex",
    "STEncoder",
    "Searcher",
    "chunk_text",
    "clean_text",
    "load_table",
    "make_passages",
    "prepare_comments",
]

__version__ = "0.1.0"
