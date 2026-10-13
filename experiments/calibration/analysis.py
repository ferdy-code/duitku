"""Score pairs with a loaded model and summarize one family (recall or dedupe)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from calibration import metrics
from calibration.models import cosine
from calibration.pairs import DedupePair, RecallPair


class Scorer(Protocol):
    """Local `Runner` and gateway `ApiRunner` both apply the model's own prefixes."""

    def embed_queries(self, texts: Sequence[str]) -> dict[str, list[float]]: ...
    def embed_passages(self, texts: Sequence[str]) -> dict[str, list[float]]: ...


def score_recall(r: Scorer, pairs: Sequence[RecallPair]) -> list[float]:
    q = r.embed_queries([p.query for p in pairs])
    m = r.embed_passages([p.memory for p in pairs])
    return [cosine(q[p.query], m[p.memory]) for p in pairs]


def score_dedupe(r: Scorer, pairs: Sequence[DedupePair]) -> list[float]:
    # `save` embeds both sides as passages, so dedupe is passage-vs-passage.
    v = r.embed_passages([t for p in pairs for t in (p.a, p.b)])
    return [cosine(v[p.a], v[p.b]) for p in pairs]


@dataclass(frozen=True)
class Summary:
    n_pos: int
    n_neg: int
    pos_stats: tuple[float, float, float]  # min / median / max
    neg_stats: tuple[float, float, float]
    auc: float
    margin: float
    f1_full: metrics.Operating
    f1_dev: metrics.Operating  # threshold chosen on dev ...
    f1_dev_on_test: metrics.Operating  # ... applied to held-out test
    p90_full: metrics.Operating | None  # max recall with precision >= 0.9
    p100_full: metrics.Operating | None  # max recall with precision == 1.0 (no false positive)
    p100_dev_on_test: metrics.Operating | None


def summarize(scores: Sequence[float], labels: Sequence[bool], splits: Sequence[str]) -> Summary:
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]

    def part(name: str) -> tuple[list[float], list[bool]]:
        idx = [i for i, s in enumerate(splits) if s == name]
        return [scores[i] for i in idx], [labels[i] for i in idx]

    (ds, dl), (ts, tl) = part("dev"), part("test")
    f1_dev = metrics.best_f1(ds, dl)
    p100_dev = metrics.max_recall_at_precision(ds, dl, 1.0)
    return Summary(
        len(pos),
        len(neg),
        metrics.describe(pos),
        metrics.describe(neg),
        metrics.auc(pos, neg),
        metrics.margin(pos, neg),
        metrics.best_f1(scores, labels),
        f1_dev,
        metrics.evaluate(ts, tl, f1_dev.threshold),
        metrics.max_recall_at_precision(scores, labels, 0.9),
        metrics.max_recall_at_precision(scores, labels, 1.0),
        metrics.evaluate(ts, tl, p100_dev.threshold) if p100_dev else None,
    )
