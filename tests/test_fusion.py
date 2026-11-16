import numpy as np
import pytest

from commentsearch.search import blend_scores, minmax_norm, rrf_fuse, top_indices


def test_rrf_scores_match_formula_and_order():
    ids, scores = rrf_fuse([np.array([1, 2, 3]), np.array([2, 1, 4])], k=60)
    expected = {1: 1 / 61 + 1 / 62, 2: 1 / 62 + 1 / 61, 3: 1 / 63, 4: 1 / 63}
    assert set(ids.tolist()) == set(expected)
    for i, s in zip(ids, scores):
        assert s == pytest.approx(expected[int(i)])
    # 1 and 2 tie at the top; 3 and 4 tie below. Ties break by id.
    assert ids.tolist() == [1, 2, 3, 4]


def test_rrf_item_in_both_lists_beats_item_in_one():
    ids, _ = rrf_fuse([np.array([5, 6]), np.array([6, 7])], k=60)
    assert ids[0] == 6


def test_rrf_smaller_k_weights_top_ranks_more():
    _, s_small = rrf_fuse([np.array([0, 1])], k=1)
    _, s_big = rrf_fuse([np.array([0, 1])], k=60)
    assert s_small[0] - s_small[1] > s_big[0] - s_big[1]


def test_minmax_constant_scores_map_to_zero():
    x = np.array([0.3, 0.3, 0.3])
    assert np.allclose(minmax_norm(x, np.ones(3, bool)), 0.0)


def test_minmax_ignores_invalid_entries():
    x = np.array([1.0, 100.0, 3.0])
    valid = np.array([True, False, True])
    out = minmax_norm(x, valid)
    assert out[0] == 0.0 and out[2] == 1.0 and out[1] == 0.0


def test_blend_alpha_one_is_dense_order_and_alpha_zero_is_bm25_order():
    dense = np.array([0.9, 0.1, 0.5, 0.2])
    bm25 = np.array([0.0, 4.0, 2.0, 1.0])
    valid = np.ones(4, bool)
    assert np.argsort(-blend_scores(dense, bm25, valid, 1.0)).tolist() == np.argsort(-dense).tolist()
    assert np.argsort(-blend_scores(dense, bm25, valid, 0.0)).tolist() == np.argsort(-bm25).tolist()


def test_blend_marks_invalid_as_minus_inf():
    dense = np.array([0.9, 0.1])
    bm25 = np.array([1.0, 0.0])
    fused = blend_scores(dense, bm25, np.array([False, True]), 0.5)
    assert fused[0] == -np.inf and np.isfinite(fused[1])


def test_blend_rejects_bad_alpha():
    with pytest.raises(ValueError):
        blend_scores(np.ones(2), np.ones(2), np.ones(2, bool), 1.5)


def test_top_indices_order_and_min_score():
    scores = np.array([0.1, -np.inf, 0.9, 0.0, 0.5])
    assert top_indices(scores, 3).tolist() == [2, 4, 0]
    assert top_indices(scores, 10, min_score=0.0).tolist() == [2, 4, 0]
    assert top_indices(np.array([0.0, 0.0]), 5, min_score=0.0).tolist() == []
