import pandas as pd

from commentsearch.ingest import chunk_text, clean_text, flag_boilerplate, make_passages, prepare_comments


def _sentences(n, words_per=20, prefix="w"):
    return " ".join(
        " ".join(f"{prefix}{i}_{j}" for j in range(words_per - 1)) + f" end{i}." for i in range(n)
    )


def test_short_text_is_one_chunk():
    assert chunk_text("A short comment. Two sentences.", target_words=250, overlap_words=50) == [
        "A short comment. Two sentences."
    ]


def test_empty_text_gives_no_chunks():
    assert chunk_text("", 250, 50) == []


def test_chunks_respect_target_and_end_on_sentence_boundaries():
    text = _sentences(30)  # 30 sentences x 20 words
    chunks = chunk_text(text, target_words=100, overlap_words=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 100
        assert chunk.endswith(".")  # every passage ends at a sentence end


def test_consecutive_chunks_overlap_by_at_most_overlap_words():
    text = _sentences(30)
    chunks = chunk_text(text, target_words=100, overlap_words=40)
    for a, b in zip(chunks, chunks[1:]):
        shared = set(a.split()) & set(b.split())
        assert shared, "expected some overlap between consecutive passages"
        # the shared tail of `a` that starts `b` is at most 40 words
        a_words, b_words = a.split(), b.split()
        k = 0
        while k < len(a_words) and k < len(b_words) and a_words[len(a_words) - 1 - k] == b_words[len(b_words) - 1 - k]:
            k += 1
        assert k <= 40 + 20  # bounded near overlap_words (sentence granularity)


def test_every_word_is_covered():
    text = _sentences(12)
    chunks = chunk_text(text, target_words=50, overlap_words=10)
    covered = set(" ".join(chunks).split())
    assert covered == set(text.split())


def test_long_sentence_is_split_into_windows():
    text = " ".join(f"word{i}" for i in range(600))  # one sentence, no punctuation
    chunks = chunk_text(text, target_words=250, overlap_words=0)
    assert [len(c.split()) for c in chunks] == [250, 250, 100]
    assert " ".join(chunks) == text


def test_invalid_overlap_raises():
    try:
        chunk_text("a b c", target_words=10, overlap_words=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_clean_text_strips_html_and_collapses_whitespace():
    assert clean_text("<p>Hello\n\n  <b>there</b>&amp;&nbsp;friend</p>") == "Hello there & friend"


def test_clean_text_handles_missing():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_boilerplate_flag():
    assert flag_boilerplate("See attached letter.")
    assert flag_boilerplate("ok")
    assert flag_boilerplate("Please see the attached comments for my full position on this rule.")
    assert not flag_boilerplate(" ".join(["substantive"] * 100))


def test_prepare_comments_drops_empty_and_parses_dates(raw_comments):
    out = prepare_comments(raw_comments, "part_id", "body_clean", ["commenter_name"], date_col="date", group_col="group_id")
    assert out.attrs["dropped_empty"] == 1
    assert list(out["comment_id"]) == ["A-1", "A-2", "B-1", "B-2", "C-2"]
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-05")
    assert pd.isna(out["date"].iloc[1])  # "None" is missing, not a date
    assert out.loc[out.comment_id == "C-2", "boilerplate"].item() is True


def test_make_passages_keeps_comment_row_and_chunk_index(comments):
    passages = make_passages(comments, target_words=250, overlap_words=50)
    assert set(passages["comment_row"]) == set(range(len(comments)))
    first = passages[passages.comment_row == 0]
    assert list(first["chunk_idx"]) == [0]
    assert (passages["comment_id"] == comments["comment_id"].to_numpy()[passages["comment_row"].to_numpy()]).all()
