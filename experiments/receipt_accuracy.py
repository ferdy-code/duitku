"""T-404: measure receipt extraction accuracy (total exact, merchant match) on REAL receipts.

Prepare (once):
  1. Put 20+ real receipt photos (JPEG/PNG: minimarket, restaurant, e-commerce screenshots) in
     data/receipts_eval/   (gitignored: never commit receipts).
  2. Write data/receipts_eval/truth.csv, typed from the paper/screen, NOT from model output:
         filename,merchant,total_idr
         r01.jpg,Indomaret,35000
     (plain integer rupiah actually paid; JSON list of the same keys also works)

Run (needs LLM_BASE_URL / LLM_API_KEY / MODEL_VISION in .env; one vision call per image):
  uv run python experiments/receipt_accuracy.py [--dir DIR] [--truth FILE] [--out FILE]
  (defaults: --dir data/receipts_eval, --truth <dir>/truth.csv)

Output: the summary on stdout (counts and rates only: paste it into docs/DECISIONS.md) and a full
report with per-receipt details at <dir>/accuracy_report.md (private, inside the gitignored dir).
Exit code 0 when >= 85% correct on >= 20 receipts (G3), else 1. Logic: experiments/receipt_eval/.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from duitku.config import ConfigError, get_settings
from duitku.receipts.vision import make_vision_model
from receipt_eval.report import render_detail, render_summary
from receipt_eval.runner import run_all, unlabelled_files
from receipt_eval.scoring import summarize
from receipt_eval.truth import TruthError, load_truth

DEFAULT_DIR = Path("data/receipts_eval")


async def amain(folder: Path, truth_path: Path, out: Path) -> int:
    truths = load_truth(truth_path)
    for name in unlabelled_files(folder, truths):
        print(f"warning: {name} has no truth row, not scored", file=sys.stderr)
    rows = await run_all(folder, truths, make_vision_model())
    summary = summarize(rows)
    text = render_summary(summary, get_settings().model_vision)
    report = f"{text}\n{render_detail(rows)}"
    await asyncio.to_thread(out.write_text, report, encoding="utf-8")
    print(text)
    print(f"details (private): {out}")
    return 0 if summary.meets_target else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="folder with the images")
    ap.add_argument("--truth", type=Path, help="default: <dir>/truth.csv")
    ap.add_argument("--out", type=Path, help="default: <dir>/accuracy_report.md")
    args = ap.parse_args()
    folder: Path = args.dir
    truth_path: Path = args.truth or folder / "truth.csv"
    out: Path = args.out or folder / "accuracy_report.md"
    if not folder.is_dir():
        ap.error(f"{folder} is not a folder (see the docstring: put the receipts there)")
    try:
        return asyncio.run(amain(folder, truth_path, out))
    except (TruthError, FileNotFoundError, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
