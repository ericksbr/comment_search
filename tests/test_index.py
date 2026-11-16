import numpy as np

from commentsearch.index import embed_passages


class CountingEncoder:
    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.calls = 0

    def encode(self, texts, batch_size=32, show_progress=False):
        self.calls += len(texts)
        return self.inner.encode(texts, batch_size, show_progress)


def test_embedding_cache_is_reused_when_texts_match(fake_encoder, tmp_path):
    texts = [f"passage number {i} about ranchers" for i in range(10)]
    enc = CountingEncoder(fake_encoder)
    first = np.array(embed_passages(texts, enc, cache_dir=tmp_path, step=4, show_progress=False))
    assert enc.calls == 10
    enc.calls = 0
    second = np.array(embed_passages(texts, enc, cache_dir=tmp_path, step=4, show_progress=False))
    assert enc.calls == 0
    assert np.allclose(first, second)


def test_embedding_cache_resumes_after_interruption(fake_encoder, tmp_path):
    texts = [f"passage {i} wells and streams" for i in range(10)]
    # Simulate an interrupted run: the first 4 rows are cached, the progress file says so.
    full = np.array(embed_passages(texts, fake_encoder, cache_dir=None, show_progress=False))
    enc = CountingEncoder(fake_encoder)
    embed_passages(texts, enc, cache_dir=tmp_path, step=4, show_progress=False)
    import json

    progress = tmp_path / "passage_emb.progress.json"
    state = json.loads(progress.read_text())
    state["done"] = 4
    progress.write_text(json.dumps(state))
    enc.calls = 0
    resumed = np.array(embed_passages(texts, enc, cache_dir=tmp_path, step=4, show_progress=False))
    assert enc.calls == 6  # only the rows after the checkpoint are re-encoded
    assert np.allclose(full, resumed)


def test_changed_texts_invalidate_cache(fake_encoder, tmp_path):
    enc = CountingEncoder(fake_encoder)
    embed_passages(["a b c", "d e f"], enc, cache_dir=tmp_path, show_progress=False)
    enc.calls = 0
    embed_passages(["a b c", "d e g"], enc, cache_dir=tmp_path, show_progress=False)
    assert enc.calls == 2
