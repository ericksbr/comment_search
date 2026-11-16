import numpy as np

from commentsearch.dedupe import build_groups, exact_edges, first_per_group, near_edges, normalize_text


def test_normalize_text_ignores_case_punctuation_and_spacing():
    assert normalize_text("Hello,   WORLD!!") == normalize_text("hello world")


def test_exact_edges_link_normalized_duplicates():
    src, dst = exact_edges(["Hello, world!", "hello world", "something else", "HELLO   world"])
    assert list(zip(src.tolist(), dst.tolist())) == [(0, 1), (0, 3)]


def test_near_edges_threshold_and_upper_triangle():
    emb = np.array(
        [
            [1.0, 0.0],
            [0.999, 0.0447],  # cosine ~0.999 with row 0
            [0.0, 1.0],       # orthogonal to both
        ],
        dtype=np.float32,
    )
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    src, dst = near_edges(emb, threshold=0.95, block=2)
    assert list(zip(src.tolist(), dst.tolist())) == [(0, 1)]


def test_near_edges_blocked_matches_unblocked():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(20, 8)).astype(np.float32)
    emb = np.vstack([base, base[:5] + 0.001]).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    a = near_edges(emb, 0.95, block=3)
    b = near_edges(emb, 0.95, block=1000)
    assert sorted(zip(*(x.tolist() for x in a))) == sorted(zip(*(x.tolist() for x in b)))


def test_build_groups_connects_exact_near_and_existing_groups():
    texts = ["the same text here", "The same text here.", "alpha beta", "gamma delta", "epsilon zeta"]
    emb = np.eye(5, dtype=np.float32)  # no near pairs
    group_values = ["g1", "", "", "g9", "g9"]
    labels, stats = build_groups(texts, emb, group_values, threshold=0.95)
    assert labels[0] == labels[1]          # exact (normalized) duplicates
    assert labels[3] == labels[4]          # same existing Grouper group
    assert labels[2] not in (labels[0], labels[3])
    assert len(set(labels.tolist())) == 3
    assert stats["exact_pairs"] == 1 and stats["group_pairs"] == 1


def test_build_groups_without_group_column():
    labels, stats = build_groups(["a b", "c d"], np.eye(2, dtype=np.float32))
    assert labels.tolist() == [0, 1]
    assert "group_pairs" not in stats


def test_first_per_group_keeps_best_ranked_member_in_order():
    labels = np.array([0, 1, 0, 2, 1])
    assert first_per_group([4, 0, 2, 3, 1], labels) == [4, 0, 3]
