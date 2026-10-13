"""T-303 pure metric helpers (no model loading)."""

import pytest

from calibration import metrics, pairs

SCORES = [0.95, 0.90, 0.85, 0.80, 0.70]
LABELS = [True, True, False, True, False]


def test_evaluate_counts_and_rates() -> None:
    o = metrics.evaluate(SCORES, LABELS, 0.82)  # predicts 0.95 0.90 0.85
    assert (o.tp, o.fp, o.fn, o.tn) == (2, 1, 1, 1)
    assert o.precision == pytest.approx(2 / 3)
    assert o.recall == pytest.approx(2 / 3)
    assert o.f1 == pytest.approx(2 / 3)
    none = metrics.evaluate(SCORES, LABELS, 0.99)  # no predictions -> precision 0, no crash
    assert (none.precision, none.recall, none.f1) == (0.0, 0.0, 0.0)


def test_candidates_are_midpoints_highest_first() -> None:
    assert metrics.candidate_thresholds([0.5, 0.9, 0.5, 0.7]) == pytest.approx([0.8, 0.6])


def test_best_f1_perfectly_separable_picks_gap_midpoint() -> None:
    o = metrics.best_f1([0.9, 0.95, 0.5, 0.4], [True, True, False, False])
    assert o.f1 == 1.0
    assert o.threshold == pytest.approx(0.7)


def test_best_f1_tie_prefers_higher_threshold() -> None:
    # 0.9 pos, 0.8 neg, 0.7 pos: t=0.85 -> F1 .67 ; t=0.75 -> F1 .5
    o = metrics.best_f1([0.9, 0.8, 0.7], [True, False, True])
    assert o.threshold == pytest.approx(0.85)


def test_max_recall_at_precision() -> None:
    o = metrics.max_recall_at_precision(SCORES, LABELS, 1.0)
    assert o is not None
    assert o.fp == 0
    assert o.recall == pytest.approx(2 / 3)  # 0.95, 0.90 above the 0.85 negative
    assert o.threshold == pytest.approx(0.875)
    assert metrics.max_recall_at_precision([0.9, 0.8], [False, True], 0.9) is None


def test_auc_and_margin() -> None:
    assert metrics.auc([0.9, 0.8], [0.5, 0.4]) == 1.0
    assert metrics.auc([0.5], [0.9]) == 0.0
    assert metrics.auc([0.5, 0.5], [0.5]) == 0.5
    assert metrics.margin([0.9, 0.8], [0.5, 0.4]) == pytest.approx(0.3)
    assert metrics.margin([0.6], [0.7]) == pytest.approx(-0.1)
    with pytest.raises(ValueError):
        metrics.auc([], [0.1])


def test_describe_and_split() -> None:
    assert metrics.describe([3.0, 1.0, 2.0]) == (1.0, 2.0, 3.0)
    s = metrics.assign_split([True, False, True, True, False, False])
    assert s == ["dev", "dev", "test", "dev", "test", "dev"]


def test_dataset_is_balanced_and_split_in_both_halves() -> None:
    rp, dp = pairs.recall_pairs(), pairs.dedupe_pairs()
    assert len(rp) >= 30 and len(dp) >= 15
    for rel in (True, False):
        assert {p.split for p in rp if p.relevant is rel} == {"dev", "test"}
    for lab in ("dup", "distinct"):
        assert {p.split for p in dp if p.label == lab} == {"dev", "test"}
    assert all(p.split == "n/a" for p in dp if p.label == "supersede")
    assert len({(p.query, p.memory) for p in rp}) == len(rp)  # no accidental duplicates
