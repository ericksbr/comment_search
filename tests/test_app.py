

import pandas as pd

from commentsearch.app import matched_terms, render_cards
from commentsearch.index import CommentIndex
from commentsearch.search import Searcher


def _searcher(comments, fake_encoder):
    index = CommentIndex.build(comments, fake_encoder, show_progress=False)
    return Searcher(index, encoder=fake_encoder)


def test_matched_terms_are_whole_words_only():
    assert matched_terms("Ranchers and wells.", ["ranchers", "well"]) == ["ranchers"]


def test_cards_highlight_terms_and_offer_full_text(comments, fake_encoder):
    s = _searcher(comments, fake_encoder)
    frame = s.search("ranchers economic", top_k=3, method="rrf", collapse=False)
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    out = render_cards(frame, frame.attrs["query_terms"], text_by_id, collapse=False)

    assert out.count("class='cs-card'") == len(frame)
    assert "<mark>" in out
    assert "Read the full passage" in out
    assert "Read the full comment" in out
    assert "+0 similar" not in out  # badge only when there are similar comments


def test_cards_escape_comment_text(fake_encoder):
    from commentsearch.ingest import prepare_comments

    raw = pd.DataFrame({"id": ["X"], "text": ["<script>alert(1)</script> ranchers"]})
    comments = prepare_comments(raw, "id", "text")
    s = _searcher(comments, fake_encoder)
    frame = s.search("ranchers", top_k=1, method="dense", collapse=False)
    out = render_cards(frame, frame.attrs["query_terms"], dict(zip(comments["comment_id"], comments["text"])), False)
    # ingest strips the tag; the card output must not reintroduce it
    assert "<script" not in out
    assert "ranchers" in out


def test_empty_frame_gives_message(comments, fake_encoder):
    s = _searcher(comments, fake_encoder)
    frame = s.search("   ", top_k=3)
    assert "No matching comments" in render_cards(frame, [], {}, False)
