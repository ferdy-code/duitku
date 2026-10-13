"""Rebuild local-model results from the raw scores in a committed report (no re-run)."""

from pathlib import Path

from calibration.analysis import summarize
from calibration.models import MODELS
from calibration.pairs import DedupePair, RecallPair
from calibration.report import ModelRun


def _rows(part: str) -> list[list[str]]:
    return [
        [c.strip() for c in ln.strip("|").split("|")]
        for ln in part.splitlines()
        if ln.startswith("| ") and not ln.startswith("| split")
    ]


def _tables(md: str) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Appendix rows as cell lists, plus the model keys (the trailing header cells)."""
    app = md.split("## Appendix")[1]
    recall, dedupe = app.split("Dedupe (a <-> b)")
    header = next(ln for ln in recall.splitlines() if ln.startswith("| split"))
    keys = [c.strip() for c in header.strip("|").split("|")][5:]
    return keys, _rows(recall), _rows(dedupe)


def load_baseline(
    path: Path, rp: list[RecallPair], dp: list[DedupePair], keys: list[str] | None = None
) -> list[ModelRun]:
    """Same pair order as the current dataset is required; checked via the query/memory text."""
    all_keys, rrows, drows = _tables(path.read_text())
    if len(rrows) != len(rp) or len(drows) != len(dp):
        raise ValueError(f"{path}: dataset size changed; re-run the local models")
    for r, p in zip(rrows, rp, strict=True):
        if (r[3], r[4]) != (p.query, p.memory):
            raise ValueError(f"{path}: recall pairs differ from the current dataset")
    for r, d in zip(drows, dp, strict=True):
        if (r[3], r[4]) != (d.a, d.b):
            raise ValueError(f"{path}: dedupe pairs differ from the current dataset")
    strict = [i for i, p in enumerate(dp) if p.label != "supersede"]
    runs: list[ModelRun] = []
    for j, key in enumerate(all_keys):
        if keys is not None and key not in keys:
            continue
        run = ModelRun(MODELS[key], reused=True)
        run.recall_scores = [float(r[5 + j]) for r in rrows]
        run.dedupe_scores = [float(r[5 + j]) for r in drows]
        run.recall = summarize(run.recall_scores, [p.relevant for p in rp], [p.split for p in rp])
        run.dedupe = summarize(
            [run.dedupe_scores[i] for i in strict],
            [dp[i].label == "dup" for i in strict],
            [dp[i].split for i in strict],
        )
        runs.append(run)
    return runs
