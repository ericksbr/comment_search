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


def test_hits_table_has_id_and_metadata_but_no_rank_or_text(comments, fake_encoder):
    from commentsearch.app import hits_table, meta_columns

    s = _searcher(comments, fake_encoder)
    frame = s.search("ranchers economic", top_k=5, method="rrf", collapse=False)
    cols = meta_columns(comments)
    table = hits_table(frame, cols)
    assert list(table.columns) == ["comment_id", *cols]  # exact-style list: no similarity column
    assert "commenter_name" in table.columns
    assert "text" not in table.columns and "dup_group" not in table.columns
    assert "excerpt" not in table.columns and "rank" not in table.columns


def test_hits_table_empty_frame_keeps_columns(comments):
    from commentsearch.app import hits_table, meta_columns

    cols = meta_columns(comments)
    table = hits_table(pd.DataFrame(), cols)
    assert table.empty and "comment_id" in table.columns


def test_detail_shows_selected_hit_in_each_mode(comments, fake_encoder):
    from commentsearch.app import MATCH_EXACT, LEVEL_PASSAGE, detail_html, meta_columns, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "economic ranchers", MATCH_EXACT, False, 0.6, LEVEL_PASSAGE)
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


def test_sentence_match_shades_similar_sentences_blue(comments, fake_encoder):
    from commentsearch.app import LEVEL_SENTENCE, MATCH_FUZZY, detail_html, meta_columns, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "economic impacts ranchers", MATCH_FUZZY, False, 0.2, LEVEL_SENTENCE)
    assert not frame.empty
    row = 0
    comment_row = int(frame.attrs["rows"][row])
    sims = s.sentence_similarities(frame.attrs["plan"]["query"], [comment_row])
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    html_out = detail_html(frame, row, "Full comment", text_by_id, meta_columns(comments), sims[comment_row])
    assert 'class="cs-sent"' in html_out
    assert "similarity" in html_out


def test_passage_match_shades_the_whole_matching_passage(comments, fake_encoder):
    from commentsearch.app import LEVEL_PASSAGE, MATCH_FUZZY, detail_html, meta_columns, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "economic impacts ranchers", MATCH_FUZZY, False, 0.2, LEVEL_PASSAGE)
    assert not frame.empty
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    passage_html = detail_html(frame, 0, "Passage", text_by_id, meta_columns(comments))
    full_html = detail_html(frame, 0, "Full comment", text_by_id, meta_columns(comments))
    assert 'class="cs-sent"' in passage_html
    assert 'class="cs-sent"' in full_html


def test_sentence_ranges_ignore_line_breaks_in_the_body():
    from commentsearch.app import sentence_ranges

    body = "Costs rise.\nThe wells  are dry."
    ranges = sentence_ranges(body, [("The wells are dry.", 0.9), ("Costs rise.", 0.2)], 0.5)
    assert [body[s:e] for s, e, _ in ranges] == ["The wells  are dry."]


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


def test_table_columns_choice_keeps_file_order(comments, fake_encoder):
    from commentsearch.app import default_table_columns, hits_table, meta_columns, select_table_columns

    s = _searcher(comments, fake_encoder)
    frame = s.search("ranchers economic", top_k=3, method="rrf", collapse=False)
    table = hits_table(frame, meta_columns(comments))
    defaults = default_table_columns(table)
    assert defaults[0] == "comment_id" and "excerpt" not in defaults and "rank" not in defaults
    picked = select_table_columns(table, ["commenter_name"])
    assert list(picked.columns) == ["commenter_name"]
    everything = select_table_columns(table, list(table.columns))
    assert list(everything.columns) == list(table.columns)


def test_similarity_column_appears_for_fuzzy_results(comments, fake_encoder):
    from commentsearch.app import LEVEL_PASSAGE, MATCH_FUZZY, hits_table, meta_columns, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "ranchers economic wells", MATCH_FUZZY, False, 0.3, LEVEL_PASSAGE)
    table = hits_table(frame, meta_columns(comments))
    assert list(table.columns)[:2] == ["similarity", "comment_id"]


def test_sort_by_comment_id_and_similarity(comments, fake_encoder):
    from commentsearch.app import LEVEL_PASSAGE, MATCH_FUZZY, SORT_ID, SORT_SIM, search_comments, sort_frame

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "ranchers economic wells", MATCH_FUZZY, False, 0.0, LEVEL_PASSAGE)
    by_id = sort_frame(frame, SORT_ID, descending=False)
    assert list(by_id["comment_id"]) == sorted(frame["comment_id"].astype(str))
    # the comment rows must move with the hits, so the sentence lookup still points at the right comment
    for i in range(len(by_id)):
        assert by_id.attrs["rows"][i] == comments.index.get_loc(
            comments.index[comments["comment_id"] == by_id.iloc[i]["comment_id"]][0]
        )
    by_sim = sort_frame(frame, SORT_SIM, descending=True)
    assert list(by_sim["score"]) == sorted(frame["score"], reverse=True)


def test_exact_match_returns_every_comment_with_a_query_word(comments, fake_encoder):
    from commentsearch.app import LEVEL_PASSAGE, MATCH_EXACT, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "ranchers", MATCH_EXACT, False, 0.6, LEVEL_PASSAGE)
    expected = comments["text"].str.contains("ranchers", case=False).sum()
    assert len(frame) == expected
    assert list(frame["rank"]) == list(range(1, len(frame) + 1))


def test_passage_match_keeps_only_results_at_or_above_cutoff(comments, fake_encoder):
    from commentsearch.app import LEVEL_PASSAGE, MATCH_FUZZY, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "ranchers economic wells", MATCH_FUZZY, False, 0.3, LEVEL_PASSAGE)
    assert (frame["score"] >= 0.3).all()


def test_sentence_match_keeps_only_comments_with_a_sentence_at_cutoff(comments, fake_encoder):
    from commentsearch.app import LEVEL_SENTENCE, MATCH_FUZZY, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "ranchers economic wells", MATCH_FUZZY, False, 0.3, LEVEL_SENTENCE)
    assert (frame["score"] >= 0.3).all()


def test_app_builds_with_exact_and_fuzzy_choice(comments, fake_encoder):
    from commentsearch.app import MATCH_EXACT, MATCH_FUZZY, make_app

    demo = make_app(_searcher(comments, fake_encoder))
    assert demo is not None
    assert MATCH_EXACT != MATCH_FUZZY


def test_sort_falls_back_to_best_match_when_the_key_is_not_offered():
    from commentsearch.app import SORT_BEST, SORT_SIM, effective_sort

    assert effective_sort(SORT_SIM, "Exact words", None) == SORT_BEST
    assert effective_sort(SORT_SIM, "Fuzzy (by meaning)", None) == SORT_SIM


def test_exact_mode_buttons_map_to_search_modes():
    from commentsearch.app import EXACT_ALL, EXACT_ANY, exact_mode_for
    from commentsearch.exact import MODE_ALL, MODE_ANY

    assert exact_mode_for(EXACT_ALL) == MODE_ALL
    assert exact_mode_for(EXACT_ANY) == MODE_ANY


def test_preview_heading_is_bold_name_and_id_in_parentheses(comments, fake_encoder):
    from commentsearch.app import MATCH_EXACT, LEVEL_PASSAGE, detail_html, meta_columns, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "economic ranchers", MATCH_EXACT, False, 0.6, LEVEL_PASSAGE)
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    row = 0
    name = frame.iloc[row]["commenter_name"]
    cid = frame.iloc[row]["comment_id"]
    out = detail_html(frame, row, "Passage", text_by_id, meta_columns(comments))
    assert f"<b>{name}</b> ({cid})" in out


def test_name_column_detection_and_saved_choice():
    from commentsearch.columns import detect_name_column

    assert detect_name_column(pd.DataFrame(columns=["id", "Commenter Name", "text"])) == "Commenter Name"
    assert detect_name_column(pd.DataFrame(columns=["id", "file_name", "text"])) is None
    assert detect_name_column(pd.DataFrame(columns=["id", "First Name", "text"])) == "First Name"


def test_preview_uses_the_name_column_chosen_at_build(comments, fake_encoder):
    from commentsearch.app import MATCH_EXACT, LEVEL_PASSAGE, detail_html, meta_columns, search_comments

    s = _searcher(comments, fake_encoder)
    frame = search_comments(s, "economic ranchers", MATCH_EXACT, False, 0.6, LEVEL_PASSAGE)
    text_by_id = dict(zip(comments["comment_id"], comments["text"]))
    cid = frame.iloc[0]["comment_id"]
    # the saved choice wins even when the column has another name
    renamed = frame.rename(columns={"commenter_name": "Submitter"})
    out = detail_html(renamed, 0, "Passage", text_by_id, meta_columns(comments), None, "Submitter")
    assert f"({cid})" in out and "<b>" in out
    # the user chose (none): only the ID is shown, in bold
    none_out = detail_html(frame, 0, "Passage", text_by_id, meta_columns(comments), None, None)
    assert f"<b>{cid}</b>" in none_out
