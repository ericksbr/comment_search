import numpy as np
import pandas as pd

from commentsearch.checks import (
    check_passages_cover_text,
    check_reload_matches,
    check_rows_accounted,
    check_unique_ids,
    run_build_checks,
    self_test,
)
from commentsearch.columns import (
    detect_date_column,
    detect_group_column,
    detect_id_column,
    detect_text_column,
    suggest_filter_columns,
)
from commentsearch.index import CommentIndex
from commentsearch.search import Searcher


def _build(comments, fake_encoder):
    return CommentIndex.build(comments, fake_encoder, show_progress=False)


def test_detect_columns_from_names_and_content(raw_comments):
    assert detect_text_column(raw_comments) == "body_clean"
    assert detect_id_column(raw_comments) == "part_id"
    assert detect_date_column(raw_comments) == "date"
    assert detect_group_column(raw_comments) == "group_id"


def test_detect_id_ignores_group_and_family_columns():
    df = pd.DataFrame({"group_id": ["1"], "family_id": ["2"], "Comment": ["hello there"]})
    assert detect_id_column(df) is None


def test_suggest_filter_columns_excludes_text_and_unique_columns():
    df = pd.DataFrame({
        "text": [f"comment {i}" for i in range(4)],
        "name": ["a", "a", "b", "b"],
        "single": ["x"] * 4,
    })
    assert suggest_filter_columns(df, exclude={"text"}) == ["name"]


def test_rows_accounted_fails_when_rows_are_lost(comments):
    ok = check_rows_accounted(len(comments) + comments.attrs["dropped_empty"], comments)
    bad = check_rows_accounted(len(comments) + comments.attrs["dropped_empty"] + 1, comments)
    assert ok.passed and not bad.passed


def test_unique_ids_check(comments):
    assert check_unique_ids(comments).passed
    dup = pd.concat([comments, comments.iloc[[0]]], ignore_index=True)
    assert not check_unique_ids(dup).passed


def test_cover_text_check_passes_for_real_passages(comments, fake_encoder):
    index = _build(comments, fake_encoder)
    assert check_passages_cover_text(index.comments, index.passages).passed


def test_cover_text_check_catches_lost_words(comments, fake_encoder):
    index = _build(comments, fake_encoder)
    broken = index.passages.copy()
    broken["text"] = broken["text"].str.replace(r"\bcritical\b", "", regex=True)
    assert not check_passages_cover_text(index.comments, broken).passed


def test_reload_check_and_full_run(comments, fake_encoder, tmp_path):
    index = _build(comments, fake_encoder)
    index.save(tmp_path / "idx")
    loaded = CommentIndex.load(tmp_path / "idx")
    assert check_reload_matches(index, loaded).passed
    searcher = Searcher(index, encoder=fake_encoder)
    checks = run_build_checks(raw_rows=len(comments) + comments.attrs["dropped_empty"], index=index, searcher=searcher)
    assert all(c.passed for c in checks), [c.line() for c in checks if not c.passed]


def test_self_test_finds_source_comment(fake_encoder):
    from commentsearch.ingest import prepare_comments

    texts = [
        "Wildlife habitat along the river is critical for elk and deer across the basin.",
        "The agency lacks legal authority to set these fees under the state statute.",
        "Economic impacts on ranchers would be severe if the wells go dry this summer.",
    ]
    raw = pd.DataFrame({"id": [f"C{i}" for i in range(3)], "text": texts})
    comments = prepare_comments(raw, "id", "text")
    index = _build(comments, fake_encoder)
    result = self_test(Searcher(index, encoder=fake_encoder), n=3, top_k=3)
    assert result.passed, result.line()
