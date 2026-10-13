"""T-303: calibrate MEMORY_RECALL_THRESHOLD / MEMORY_DEDUPE_THRESHOLD, compare embedding models.

Run: uv run python experiments/calibrate_memory_threshold.py [--models e5,bge]
Needs network on first run (HuggingFace download: e5-small ~0.5GB, bge-m3 ~2.3GB). No DB, no LLM.
Writes experiments/results/memory_calibration.md. Dataset and label policy: calibration/pairs.py.
"""

import argparse
import sys
from pathlib import Path

from calibration.analysis import score_dedupe, score_recall, summarize
from calibration.models import MODELS, Runner
from calibration.pairs import dedupe_pairs, recall_pairs
from calibration.report import ModelRun, render

OUT = Path(__file__).parent / "results" / "memory_calibration.md"


def run_model(key: str) -> ModelRun:
    run = ModelRun(MODELS[key])
    rp, dp = recall_pairs(), dedupe_pairs()
    try:
        runner = Runner(run.spec)
    except Exception as exc:  # download/load failure: report it, never skip silently
        run.error = f"{type(exc).__name__}: {exc}"
        print(f"[{key}] LOAD FAILED -> {run.error}", file=sys.stderr)
        return run
    run.recall_scores = score_recall(runner, rp)
    run.dedupe_scores = score_dedupe(runner, dp)
    strict = [(s, p) for s, p in zip(run.dedupe_scores, dp, strict=True) if p.label != "supersede"]
    run.recall = summarize(run.recall_scores, [p.relevant for p in rp], [p.split for p in rp])
    run.dedupe = summarize(
        [s for s, _ in strict], [p.label == "dup" for _, p in strict], [p.split for _, p in strict]
    )
    run.perf = runner.measure([p.memory for p in rp])
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="e5,bge", help="comma list of: " + ",".join(MODELS))
    keys = [k.strip() for k in ap.parse_args().models.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        ap.error(f"unknown model(s): {unknown}")
    runs = [run_model(k) for k in keys]
    md = render(runs, recall_pairs(), dedupe_pairs())
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(md)
    print(md.split("## Appendix")[0])
    print(f"saved: {OUT}")
    return 1 if any(r.error for r in runs) else 0


if __name__ == "__main__":
    sys.exit(main())
