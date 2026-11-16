import numpy as np
import pandas as pd

from commentsearch.index import CommentIndex
from commentsearch.search import Searcher, filter_mask, highlight, make_snippet, query_terms


def _build(comments, fake_encoder, group_col=None):
    return CommentIndex.build(comments, fake_encoder, target_words=250, overlap_words=50, group_col=group_col, show_progress=False)


def test_query_terms_drop_short_and_stop_words():
    assert query_terms("The economic impacts on ranchers") == ["economic", "impacts", "ranchers"]


def test_highlight_wraps_terms_and_escapes_html():
    out = highlight("Elk <b>habitat</b> and habitat loss", ["habitat"])
    assert out == "Elk &lt;b&gt;<mark>habitat</mark>&lt;/b&gt; and <mark>habitat</mark> loss"


def test_make_snippet_centers_on_term():
    text = "x " * 300 + "TARGET word " + "y " * 300
    snip = make_snippet(text, ["target"], width=100)
    assert "TARGET" in snip and snip.startswith("…")


def test_filter_mask_equality_list_and_date_range(comments):
    assert filter_mask(comments, {"commenter_name": "Ann"}).tolist() == [True, True, False, False, False]
    assert filter_mask(comments, {"commenter_name": ["Bob", "Cy"]}).tolist() == [False, False, True, True, True]
    m = filter_mask(comments, {"date": (pd.Timestamp("2024-03-01"), None)})
    assert m.tolist() == [False, False, True, True, False]  # NaT rows drop out of a bounded range


def test_search_returns_one_row_per_comment_with_metadata(comments, fake_encoder):
    index = _build(comments, fake_encoder)
    frame = Searcher(index, encoder=fake_encoder).search("ranchers economic", top_k=5, method="rrf", collapse=False)
    assert frame["comment_id"].is_unique
    assert frame.iloc[0]["comment_id"] == "A-2"
    assert {"rank", "score", "comment_id", "snippet", "group_size", "commenter_name"} <= set(frame.columns)
    assert "timings" in frame.attrs and "total" in frame.attrs["timings"]


def test_filters_apply_before_ranking(comments, fake_encoder):
    index = _build(comments, fake_encoder)
    s = Searcher(index, encoder=fake_encoder)
    # A-2 is the best match for "ranchers" but is excluded by the filter, so it must not appear at all.
    frame = s.search("ranchers economic", top_k=5, method="dense", filters={"commenter_name": "Bob"})
    assert "A-2" not in set(frame["comment_id"])
    assert set(frame["comment_id"]) <= {"B-1", "B-2"}


def test_bm25_only_returns_only_term_matches(comments, fake_encoder):
    index = _build(comments, fake_encoder)
    frame = Searcher(index, encoder=fake_encoder).search("fees", top_k=5, method="bm25", collapse=False)
    assert frame["comment_id"].tolist() == ["B-1"]


def test_collapse_shows_one_representative_per_group(fake_encoder):
    from commentsearch.ingest import prepare_comments

    raw = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "text": [
                "Ranchers face economic loss from the rule.",
                "Ranchers face economic loss from the rule!",
                "Elk habitat along the river matters most.",
            ],
            "grp": ["g", "g", ""],
        }
    )
    comments = prepare_comments(raw, "id", "text", group_col="grp")
    index = _build(comments, fake_encoder, group_col="grp")
    s = Searcher(index, encoder=fake_encoder)
    shown = s.search("ranchers economic loss", top_k=5, method="rrf", collapse=True)
    assert len(shown) == 2
    rep = shown.iloc[0]
    assert rep["group_size"] == 2
    raw_shown = s.search("ranchers economic loss", top_k=5, method="rrf", collapse=False)
    assert len(raw_shown) == 3


def test_roll_up_gives_each_comment_once_even_with_many_passages(fake_encoder):
    from commentsearch.ingest import prepare_comments

    long_text = " ".join(f"Sentence about ranchers number {i} and wells." for i in range(120))
    raw = pd.DataFrame({"id": ["L", "S"], "text": [long_text, "Ranchers and wells."]})
    comments = prepare_comments(raw, "id", "text")
    index = _build(comments, fake_encoder)
    assert index.meta["n_passages"] > 2
    frame = Searcher(index, encoder=fake_encoder).search("ranchers wells", top_k=10, method="rrf", collapse=False)
    assert frame["comment_id"].is_unique


def test_save_and_load_round_trip_gives_same_results(comments, fake_encoder, tmp_path):
    index = _build(comments, fake_encoder)
    before = Searcher(index, encoder=fake_encoder).search("wildlife habitat elk", top_k=4, method="rrf", collapse=False)
    folder = index.save(tmp_path / "idx")
    loaded = CommentIndex.load(folder)
    after = Searcher(loaded, encoder=fake_encoder).search("wildlife habitat elk", top_k=4, method="rrf", collapse=False)
    assert before["comment_id"].tolist() == after["comment_id"].tolist()
    assert np.allclose(before["score"].to_numpy(), after["score"].to_numpy(), atol=1e-6)
    assert loaded.meta["model"] == fake_encoder.name


def test_empty_query_returns_empty_frame(comments, fake_encoder):
    index = _build(comments, fake_encoder)
    assert Searcher(index, encoder=fake_encoder).search("   ", top_k=3).empty

