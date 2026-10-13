"""Pure metric helpers for threshold calibration (no model, no I/O). A score >= t is a positive."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import median


@dataclass(frozen=True)
class Operating:
    """Confusion counts and derived rates at one threshold."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0  # no positives -> 0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> Operating:
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    tp = sum(1 for s, y in zip(scores, labels, strict=True) if y and s >= threshold)
    fp = sum(1 for s, y in zip(scores, labels, strict=True) if not y and s >= threshold)
    pos = sum(labels)
    return Operating(threshold, tp, fp, pos - tp, len(labels) - pos - fp)


def candidate_thresholds(scores: Sequence[float]) -> list[float]:
    """Midpoints between adjacent distinct scores, highest first (ties resolve conservatively)."""
    u = sorted(set(scores), reverse=True)
    return [(a + b) / 2 for a, b in pairwise(u)]


def best_f1(scores: Sequence[float], labels: Sequence[bool]) -> Operating:
    """Threshold maximizing F1; on equal F1 the highest threshold wins."""
    cands = candidate_thresholds(scores)
    if not cands:
        raise ValueError("need at least two distinct scores")
    return max((evaluate(scores, labels, t) for t in cands), key=lambda o: o.f1)  # max keeps first


def max_recall_at_precision(
    scores: Sequence[float], labels: Sequence[bool], min_precision: float
) -> Operating | None:
    """Highest recall among thresholds with precision >= min_precision (None if unreachable)."""
    ops = (evaluate(scores, labels, t) for t in candidate_thresholds(scores))
    ok = [o for o in ops if o.tp and o.precision >= min_precision]
    return max(ok, key=lambda o: o.recall, default=None)  # ties -> highest threshold


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(random positive scores above random negative); ties count half (Mann-Whitney U)."""
    if not pos or not neg:
        raise ValueError("auc needs both classes")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def margin(pos: Sequence[float], neg: Sequence[float]) -> float:
    """min(positive) - max(negative): > 0 means a perfectly separating threshold exists."""
    return min(pos) - max(neg)


def describe(xs: Sequence[float]) -> tuple[float, float, float]:
    return min(xs), float(median(xs)), max(xs)


def assign_split(labels: Sequence[bool]) -> list[str]:
    """Deterministic stratified 50/50 split: alternate dev/test within each label, in order."""
    seen = {True: 0, False: 0}
    out: list[str] = []
    for y in labels:
        out.append("dev" if seen[y] % 2 == 0 else "test")
        seen[y] += 1
    return out
