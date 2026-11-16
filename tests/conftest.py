import hashlib
import re

import numpy as np
import pandas as pd
import pytest


class FakeEncoder:
    """Hashed bag-of-words encoder. Same interface as STEncoder, no model download."""

    def __init__(self, dim: int = 256, name: str = "fake-hash"):
        self.dim = dim
        self.name = name
        self.query_prefix = ""

    def encode(self, texts, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", str(text).lower()):
                slot = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dim
                out[i, slot] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@pytest.fixture
def fake_encoder():
    return FakeEncoder()


@pytest.fixture
def raw_comments():
    """A small raw table in the shape the user's CSV has (ids, text, metadata, groups)."""
    return pd.DataFrame(
        {
            "part_id": ["A-1", "A-2", "B-1", "B-2", "C-1", "C-2"],
            "commenter_name": ["Ann", "Ann", "Bob", "Bob", "Cy", "Cy"],
            "date": ["2024-01-05", "None", "2024-03-01", "2024-03-02", "None", "None"],
            "group_id": ["7", "7", "8", "9", "10", "11"],
            "body_clean": [
                "<p>Wildlife habitat along the river is <b>critical</b> for elk.</p>",
                "Economic impacts on ranchers would be severe if wells go dry.",
                "The agency lacks legal authority to set these fees under state law.",
                "Notice was posted late and the public hearing was rushed.",
                "",
                "See attached letter for my full comments.",
            ],
        }
    )


@pytest.fixture
def comments(raw_comments):
    from commentsearch.ingest import prepare_comments

    return prepare_comments(
        raw_comments,
        id_col="part_id",
        text_col="body_clean",
        meta_cols=["commenter_name"],
        date_col="date",
        group_col="group_id",
    )
