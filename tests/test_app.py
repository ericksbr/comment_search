

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
    assert "cs-full" in out
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


def test_hits_table_has_rank_id_and_metadata_but_no_excerpt(comments, fake_encoder):
    from commentsearch.app import hits_table, meta_columns

    s = _searcher(comments, fake_encoder)
    frame = s.search("ranchers economic", top_k=5, method="rrf", collapse=False)
    cols = meta_columns(comments)
    table = hits_table(frame, cols)
    assert list(table.columns) == ["rank", "comment_id", *cols]
    assert "commenter_name" in table.columns
    assert "text" not in table.columns and "dup_group" not in table.columns
    assert "excerpt" not in table.columns  # the text is shown in the preview, not the list


def test_hits_table_empty_frame_keeps_columns(comments):
    from commentsearch.app import hits_table, meta_columns

    cols = meta_columns(comments)
    table = hits_table(pd.DataFrame(), cols)
    assert table.empty and "comment_id" in table.columns and "rank" in table.columns


def test_detail_shows_selected_hit_in_each_mode(comments, fake_encoder):
    from commentsearch.app import MATCH_EXACT, detail_html, meta_columns, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "economic ranchers", MATCH_EXACT, collapse=False, min_similarity=0.6)
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    cols = meta_columns(comments)
    row = 0
    passage_html = detail_html(frame, row, "Passage", text_by_id, cols)
    full_html = detail_html(frame, row, "Full comment", text_by_id, cols)
    excerpt_html = detail_html(frame, row, "Excerpt", text_by_id, cols)
    assert 'class="cs-exact"' in passage_html
    assert frame.iloc[row]["comment_id"] in passage_html
    assert len(full_html) >= len(excerpt_html) >= 0
    assert "Click a row" in detail_html(None, None, "Passage", text_by_id, cols)


def test_fuzzy_detail_shades_similar_sentences_blue(comments, fake_encoder):
    from commentsearch.app import MATCH_FUZZY, detail_html, meta_columns, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "economic impacts ranchers", MATCH_FUZZY, collapse=False, min_similarity=0.2)
    assert not frame.empty
    row = 0
    sims = s.sentence_similarities(frame.attrs["plan"]["query"], [int(frame.attrs["rows"][row])])
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    html_out = detail_html(frame, row, "Full comment", text_by_id, meta_columns(comments), sims[int(frame.attrs["rows"][row])])
    assert 'class="cs-sent"' in html_out
    assert "similarity" in html_out


def test_show_modes_run_shortest_to_longest():
    from commentsearch.app import SHOW_MODES

    assert SHOW_MODES == ["Excerpt", "Passage", "Full comment"]


def test_list_has_no_long_text_so_rows_stay_short(comments, fake_encoder):
    from commentsearch.app import hits_table, meta_columns

    long_text = " ".join(["ranchers wells groundwater"] * 80)
    raw = pd.DataFrame({"id": ["L"], "text": [long_text]})
    from commentsearch.ingest import prepare_comments

    long_comments = prepare_comments(raw, "id", "text")
    s = _searcher(long_comments, fake_encoder)
    frame = s.search("ranchers wells", top_k=1, method="dense", collapse=False)
    table = hits_table(frame, meta_columns(long_comments))
    assert all(table[c].astype(str).str.len().max() < 200 for c in table.columns)


def test_table_columns_choice_keeps_rank_and_file_order(comments, fake_encoder):
    from commentsearch.app import default_table_columns, hits_table, meta_columns, select_table_columns

    s = _searcher(comments, fake_encoder)
    frame = s.search("ranchers economic", top_k=3, method="rrf", collapse=False)
    table = hits_table(frame, meta_columns(comments))
    defaults = default_table_columns(table)
    assert defaults[0] == "rank" and defaults[1] == "comment_id" and "excerpt" not in defaults
    picked = select_table_columns(table, ["commenter_name"])
    assert list(picked.columns) == ["rank", "commenter_name"]
    everything = select_table_columns(table, list(table.columns))
    assert list(everything.columns) == list(table.columns)


def test_exact_match_returns_every_comment_with_a_query_word(comments, fake_encoder):
    from commentsearch.app import MATCH_EXACT, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "ranchers", MATCH_EXACT, collapse=False, min_similarity=0.6)
    expected = comments["text"].str.contains("ranchers", case=False).sum()
    assert len(frame) == expected
    assert list(frame["rank"]) == list(range(1, len(frame) + 1))


def test_fuzzy_match_keeps_only_results_at_or_above_cutoff(comments, fake_encoder):
    from commentsearch.app import MATCH_FUZZY, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "ranchers economic wells", MATCH_FUZZY, collapse=False, min_similarity=0.3)
    assert (frame["score"] >= 0.3).all()


def test_app_builds_with_exact_and_fuzzy_choice(comments, fake_encoder):
    from commentsearch.app import MATCH_EXACT, MATCH_FUZZY, make_app

    demo = make_app(_searcher(comments, fake_encoder))
    assert demo is not None
    assert MATCH_EXACT != MATCH_FUZZY
