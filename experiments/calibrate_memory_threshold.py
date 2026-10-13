"""T-303: calibrate MEMORY_RECALL_THRESHOLD / MEMORY_DEDUPE_THRESHOLD, compare embedding models.

Local:  uv run python experiments/calibrate_memory_threshold.py [--models e5,bge]
        Needs network on first run (HuggingFace: e5-small ~0.5GB, bge-m3 ~2.3GB). No DB, no LLM.
        Writes experiments/results/memory_calibration.md.
API:    uv run python experiments/calibrate_memory_threshold.py \
            --models api-bge-m3,api-qwen3,api-3small,api-3large,api-mistral
        Uses the OpenAI-compatible gateway from .env (LLM_BASE_URL / LLM_API_KEY), ~a few hundred
        short texts per model. Writes experiments/results/memory_calibration_api.md and adds the
        local e5/bge rows rebuilt from the committed T-303 raw scores (never re-run, never edited).
Dataset and label policy: calibration/pairs.py.
"""

import argparse
import sys
from pathlib import Path

from calibration.analysis import Scorer, score_dedupe, score_recall, summarize
from calibration.api_runner import ApiRunner
from calibration.baseline import load_baseline
from calibration.models import MODELS, Runner
from calibration.pairs import dedupe_pairs, recall_pairs
from calibration.report import ModelRun, render
from duitku.memory.embeddings import EmbeddingError

RESULTS = Path(__file__).parent / "results"
LOCAL_OUT = RESULTS / "memory_calibration.md"  # T-303 evidence: only local runs may write here
API_OUT = RESULTS / "memory_calibration_api.md"
API_TITLE = "Embedding model comparison: gateway API models vs T-303 local baseline"


def run_model(key: str) -> ModelRun:
    print(f"[{key}] start", file=sys.stderr, flush=True)
    run = ModelRun(MODELS[key])
    rp, dp = recall_pairs(), dedupe_pairs()
    runner: Runner | ApiRunner | None = None
    try:
        runner = ApiRunner(run.spec) if run.spec.api_model else Runner(run.spec)
        scorer: Scorer = runner
        run.recall_scores = score_recall(scorer, rp)
        run.dedupe_scores = score_dedupe(scorer, dp)
        run.perf = runner.measure([p.memory for p in rp])
    except (EmbeddingError, OSError, ImportError) as exc:  # report it, never skip silently
        run.error = f"{type(exc).__name__}: {exc}"
        print(f"[{key}] FAILED -> {run.error}", file=sys.stderr)
        return run
    finally:
        if isinstance(runner, ApiRunner):
            runner.close()
    strict = [(s, p) for s, p in zip(run.dedupe_scores, dp, strict=True) if p.label != "supersede"]
    run.recall = summarize(run.recall_scores, [p.relevant for p in rp], [p.split for p in rp])
    run.dedupe = summarize(
        [s for s, _ in strict], [p.label == "dup" for _, p in strict], [p.split for _, p in strict]
    )
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", default="e5,bge", help="comma list of: " + ",".join(MODELS))
    ap.add_argument(
        "--out", type=Path, help="default: local-only runs -> T-303 file, else API file"
    )
    ap.add_argument("--no-baseline", action="store_true", help="API runs: skip local e5/bge rows")
    args = ap.parse_args()
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        ap.error(f"unknown model(s): {unknown}")
    uses_api = any(MODELS[k].api_model for k in keys)
    out: Path = args.out or (API_OUT if uses_api else LOCAL_OUT)
    if uses_api and out.resolve() == LOCAL_OUT.resolve():
        ap.error(f"refusing to overwrite the T-303 evidence {LOCAL_OUT} with API results")
    rp, dp = recall_pairs(), dedupe_pairs()
    base = [] if args.no_baseline or not uses_api else load_baseline(LOCAL_OUT, rp, dp)
    runs = [r for r in base if r.spec.key not in keys] + [run_model(k) for k in keys]
    md = render(runs, rp, dp, API_TITLE if uses_api else "T-303 memory threshold calibration")
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    print(md.split("## Appendix")[0])
    print(f"saved: {out}")
    return 1 if any(r.error for r in runs) else 0


if __name__ == "__main__":
    sys.exit(main())
