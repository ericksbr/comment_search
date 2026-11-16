

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


def test_hits_table_has_rank_id_metadata_and_short_excerpt(comments, fake_encoder):
    from commentsearch.app import EXCERPT_CHARS, hits_table, meta_columns

    s = _searcher(comments, fake_encoder)
    frame = s.search("ranchers economic", top_k=5, method="rrf", collapse=False)
    cols = meta_columns(comments)
    table = hits_table(frame, cols)
    assert list(table.columns) == ["rank", "comment_id", *cols, "excerpt"]
    assert "commenter_name" in table.columns
    assert "text" not in table.columns and "dup_group" not in table.columns
    assert table["excerpt"].str.len().max() <= EXCERPT_CHARS + 1


def test_hits_table_empty_frame_keeps_columns(comments):
    from commentsearch.app import hits_table, meta_columns

    cols = meta_columns(comments)
    table = hits_table(pd.DataFrame(), cols)
    assert table.empty and "comment_id" in table.columns and "excerpt" in table.columns


def test_detail_shows_selected_hit_in_each_mode(comments, fake_encoder):
    from commentsearch.app import detail_html, meta_columns

    s = _searcher(comments, fake_encoder)
    frame = s.search("ranchers economic", top_k=3, method="rrf", collapse=False)
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    cols = meta_columns(comments)
    terms = frame.attrs["query_terms"]
    row = 0
    passage_html = detail_html(frame, row, "Passage", terms, text_by_id, cols)
    full_html = detail_html(frame, row, "Full comment", terms, text_by_id, cols)
    excerpt_html = detail_html(frame, row, "Excerpt", terms, text_by_id, cols)
    assert "<mark>" in passage_html
    assert frame.iloc[row]["comment_id"] in passage_html
    assert len(full_html) >= len(excerpt_html) >= 0
    assert "Click a row" in detail_html(None, None, "Passage", terms, text_by_id, cols)
