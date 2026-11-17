import numpy as np
import pandas as pd

from commentsearch.exact import highlight_exact, parse_query, passage_hits
from commentsearch.index import CommentIndex, make_sentences
from commentsearch.search import Searcher


def _searcher(comments, fake_encoder):
    index = CommentIndex.build(comments, fake_encoder, show_progress=False)
    return Searcher(index, encoder=fake_encoder)


def test_parse_query_drops_stop_words_and_keeps_phrases():
    plan = parse_query('the "economic impact" ranchers')
    assert plan["loose"] == ["ranchers"]
    assert plan["phrases"] == [["economic", "impact"]]


def test_all_words_must_match_and_stems_count(comments, fake_encoder):
    s = _searcher(comments, fake_encoder)
    # "impact" stems to "impact", matching "impacts" in the fixture; "ranchers" also matches itself
    rows_one, _, _ = s._exact_rows("impact ranchers", "bm25s")
    rows_none, _, _ = s._exact_rows("impact elk", "bm25s")
    assert len(rows_one) == 1
    assert len(rows_none) == 0


def test_phrase_needs_words_in_order(comments, fake_encoder):
    s = _searcher(comments, fake_encoder)
    assert len(s._exact_rows('"economic impacts"', "bm25s")[0]) == 1
    assert len(s._exact_rows('"impacts economic"', "bm25s")[0]) == 0
    assert len(s._exact_rows('"economic impacts"', "fts5")[0]) == 1
    assert len(s._exact_rows('"impacts economic"', "fts5")[0]) == 0


def test_both_engines_agree_on_fixture(comments, fake_encoder):
    s = _searcher(comments, fake_encoder)
    for q in ["ranching", "wells dry", "legal authority fees", '"public hearing"', "elk river"]:
        bm = set(s._exact_rows(q, "bm25s")[0].tolist())
        fts = set(s._exact_rows(q, "fts5")[0].tolist())
        assert bm == fts, q


def test_exact_search_frame_and_plan(comments, fake_encoder):
    s = _searcher(comments, fake_encoder)
    frame = s.search("economic ranchers", method="exact", backend="fts5", top_k=50, collapse=False)
    assert list(frame["comment_id"]) == [comments["comment_id"].iloc[1]]
    assert frame.attrs["plan"]["loose"] == ["economic", "ranchers"]


def test_highlight_exact_marks_stems_and_phrases():
    plan = parse_query('ranchers "economic impacts"')
    html_out = highlight_exact("Ranchers face economic impacts and ranching costs.", plan)
    assert html_out.count('class="cs-exact"') == 3  # Ranchers, economic, impacts (ranching is a different stem)


def test_passage_hits_counts_matched_words():
    plan = parse_query("ranchers wells")
    assert passage_hits("Ranchers lose wells", plan) == 2
    assert passage_hits("Nothing here", plan) == 0


def test_semantic_cutoff_keeps_only_similar_passages(comments, fake_encoder):
    s = _searcher(comments, fake_encoder)
    everything = s.search("economic impacts on ranchers", method="semantic", min_similarity=-1.0, top_k=50, collapse=False)
    strict = s.search("economic impacts on ranchers", method="semantic", min_similarity=0.9, top_k=50, collapse=False)
    assert len(everything) >= len(strict)
    assert (strict["score"] >= 0.9).all()


def test_semantic_empty_when_nothing_passes(comments, fake_encoder):
    s = _searcher(comments, fake_encoder)
    frame = s.search("zebra quartz", method="semantic", min_similarity=0.99, top_k=50, collapse=False)
    assert frame.empty


def test_sentences_split_with_punkt():
    df = pd.DataFrame({"text": ["Dr. Smith went to the U.S. office. It closed! Did it?", ""]})
    sents = make_sentences(df)
    assert list(sents["text"]) == ["Dr. Smith went to the U.S. office.", "It closed!", "Did it?"]
    assert set(sents["comment_row"]) == {0}


def test_sentence_similarities_cover_each_sentence(comments, fake_encoder):
    s = _searcher(comments, fake_encoder)
    assert s.has_sentences
    sims = s.sentence_similarities("economic impacts", [1])
    assert set(sims) == {1}
    assert all(isinstance(score, float) for _, score in sims[1])


def test_compare_exact_reports_both_engines(comments, fake_encoder):
    s = _searcher(comments, fake_encoder)
    report = s.compare_exact("impacts")
    assert report["fts5"] == report["bm25s"] == report["both"]
    assert report["only_fts5_count"] == 0 and report["only_bm25s_count"] == 0
