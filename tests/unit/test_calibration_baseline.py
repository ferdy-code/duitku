"""The T-303 raw scores in the committed report reproduce its own summary numbers."""

from pathlib import Path

import pytest

from calibration.baseline import load_baseline
from calibration.pairs import dedupe_pairs, recall_pairs

REPORT = Path(__file__).parents[2] / "experiments" / "results" / "memory_calibration.md"


def test_baseline_rebuilds_t303_summary() -> None:
    runs = {r.spec.key: r for r in load_baseline(REPORT, recall_pairs(), dedupe_pairs())}
    assert set(runs) == {"e5", "bge"}
    e5, bge = runs["e5"], runs["bge"]
    assert e5.recall and bge.recall and e5.dedupe and bge.dedupe
    # raw scores are rounded to 3 decimals in the report, so ties shift the AUC slightly
    got = (e5.recall.auc, bge.recall.auc, e5.dedupe.auc, bge.dedupe.auc)
    assert got == pytest.approx((0.867, 0.902, 0.654, 0.808), abs=0.01)
    assert len(e5.dedupe_scores) == len(dedupe_pairs())  # supersede pairs kept for reporting
