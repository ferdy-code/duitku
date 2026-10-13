"""Markdown report for the T-303 calibration run."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from calibration.analysis import Summary
from calibration.metrics import Operating
from calibration.models import ModelSpec, Perf
from calibration.pairs import DedupePair, RecallPair


@dataclass
class ModelRun:
    spec: ModelSpec
    perf: Perf | None = None
    recall_scores: list[float] = field(default_factory=lambda: list[float]())
    dedupe_scores: list[float] = field(default_factory=lambda: list[float]())
    recall: Summary | None = None
    dedupe: Summary | None = None
    error: str | None = None


def _op(o: Operating | None) -> str:
    if o is None:
        return "n/a"
    return f"t={o.threshold:.3f} P={o.precision:.2f} R={o.recall:.2f} F1={o.f1:.2f}"


def _stats(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.3f} / {t[1]:.3f} / {t[2]:.3f}"


def _family_table(title: str, runs: Sequence[ModelRun], pick: str, pos: str, neg: str) -> list[str]:
    out = [
        f"### {title}",
        "",
        f"| model | {pos} min/med/max | {neg} min/med/max | AUC | margin | best F1 (all) "
        f"| best F1 on dev -> test | max R @ P>=0.9 | max R @ P=1.0 | P=1.0 on dev -> test |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        s: Summary | None = getattr(r, pick)
        if s is None:
            continue
        dev = f"dev {_op(s.f1_dev)} -> test {_op(s.f1_dev_on_test)}"
        out.append(
            f"| {r.spec.key} | {_stats(s.pos_stats)} | {_stats(s.neg_stats)} | {s.auc:.3f} "
            f"| {s.margin:+.3f} | {_op(s.f1_full)} | {dev} | {_op(s.p90_full)} "
            f"| {_op(s.p100_full)} | {_op(s.p100_dev_on_test)} |"
        )
    return [*out, ""]


def _perf_table(runs: Sequence[ModelRun]) -> list[str]:
    out = [
        "### Cost (CPU)",
        "",
        "| model | dim | load s | RSS base MB | RSS loaded MB | RSS after encode MB "
        "| 1 text ms median / p95 | 32 texts ms | torch threads |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        p = r.perf
        if p is None:
            out.append(f"| {r.spec.key} | {r.spec.dim} | FAILED: {r.error} | | | | | | |")
            continue
        out.append(
            f"| {r.spec.key} | {r.spec.dim} | {p.load_s:.1f} | {p.rss_base_mb:.0f} "
            f"| {p.rss_loaded_mb:.0f} | {p.rss_encoded_mb:.0f} "
            f"| {p.single_ms_median:.1f} / {p.single_ms_p95:.1f} | {p.batch32_ms:.0f} "
            f"| {p.torch_threads} |"
        )
    return [*out, ""]


def _supersede(runs: Sequence[ModelRun], pairs: Sequence[DedupePair]) -> list[str]:
    idx = [i for i, p in enumerate(pairs) if p.label == "supersede"]
    out = ["### Supersede (conflicting update) pairs: cosine, merged at the dup threshold?", ""]
    for r in runs:
        if r.dedupe is None:
            continue
        t = r.dedupe.p100_full.threshold if r.dedupe.p100_full else float("nan")
        sc = [r.dedupe_scores[i] for i in idx]
        merged = sum(s >= t for s in sc)
        out.append(
            f"- {r.spec.key}: scores {', '.join(f'{s:.3f}' for s in sc)}; "
            f"merged {merged}/{len(sc)} at t={t:.3f} (P=1.0 threshold)"
        )
    return [*out, ""]


def _appendix(
    runs: Sequence[ModelRun], rp: Sequence[RecallPair], dp: Sequence[DedupePair]
) -> list[str]:
    ok = [r for r in runs if r.perf is not None]
    head = " | ".join(r.spec.key for r in ok)
    sep = "|".join("---" for _ in range(len(ok) + 5))
    out = [
        "## Appendix: raw scores",
        "",
        "Recall (query -> memory)",
        "",
        f"| split | rel | tag | query | memory | {head} |",
        f"|{sep}|",
    ]
    for i, p in enumerate(rp):
        sc = " | ".join(f"{r.recall_scores[i]:.3f}" for r in ok)
        out.append(f"| {p.split} | {int(p.relevant)} | {p.tag} | {p.query} | {p.memory} | {sc} |")
    out += ["", "Dedupe (a <-> b)", "", f"| split | label | tag | a | b | {head} |", f"|{sep}|"]
    for i, d in enumerate(dp):
        sc = " | ".join(f"{r.dedupe_scores[i]:.3f}" for r in ok)
        out.append(f"| {d.split} | {d.label} | {d.tag} | {d.a} | {d.b} | {sc} |")
    return [*out, ""]


def render(runs: Sequence[ModelRun], rp: Sequence[RecallPair], dp: Sequence[DedupePair]) -> str:
    done = [r for r in runs if r.recall is not None]
    strict = [p for p in dp if p.label != "supersede"]
    lines = [
        "# T-303 memory threshold calibration",
        "",
        f"Dataset: {len(rp)} recall pairs ({sum(p.relevant for p in rp)} relevant), "
        f"{len(strict)} dedupe pairs ({sum(p.label == 'dup' for p in strict)} dup / "
        f"{sum(p.label == 'distinct' for p in strict)} distinct) + "
        f"{len(dp) - len(strict)} supersede pairs reported separately. "
        "INDICATIVE ONLY: tiny, hand-written, 50/50 dev/test split (threshold chosen on dev, "
        "scored on test). A threshold is `score >= t`; t is the midpoint of the gap between "
        "adjacent scores.",
        "",
        *_family_table("Recall (query vs memory)", done, "recall", "relevant", "irrelevant"),
        *_family_table("Dedupe (memory vs memory)", done, "dedupe", "dup", "distinct"),
        *_supersede(done, dp),
        *_perf_table(runs),
        *_appendix(done, rp, dp),
    ]
    return "\n".join(lines)
