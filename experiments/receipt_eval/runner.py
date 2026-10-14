"""Run the real extraction pipeline over a folder of receipt images (no DB, no agent).

Same steps as the `parse_receipt` tool: read the image, vision model with structured output,
`finalize` (adds our warnings). One failing image never stops the run: it becomes an error row.
"""

import datetime as dt
import logging
from pathlib import Path

from duitku.receipts.checks import finalize
from duitku.receipts.storage import MIME, ReceiptFileError, read_image
from duitku.receipts.vision import VisionError, VisionModel
from receipt_eval.scoring import Row, score
from receipt_eval.truth import Truth

logger = logging.getLogger(__name__)

MAX_BYTES = 20 * 1024 * 1024  # Telegram's own download limit; the bot's default is lower


def unlabelled_files(folder: Path, truth: list[Truth]) -> list[str]:
    """Images in the folder that have no truth row (they are NOT scored)."""
    known = {t.filename for t in truth}
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        p.name for p in folder.iterdir() if p.suffix.lower() in exts and p.name not in known
    )


async def run_one(folder: Path, truth: Truth, vision: VisionModel, today: dt.date) -> Row:
    try:
        # The folder doubles as the allowed directory: the guard still rejects `../x` and symlinks.
        image, kind = await read_image(str(folder / truth.filename), folder, MAX_BYTES)
    except ReceiptFileError as e:
        return score(truth, None, f"file: {e}")
    try:
        ex = await vision.extract(image, MIME[kind])
    except VisionError as e:
        return score(truth, None, f"vision: {e}")
    return score(truth, finalize(ex, today))


async def run_all(
    folder: Path, truths: list[Truth], vision: VisionModel, today: dt.date | None = None
) -> list[Row]:
    day = today or dt.date.today()
    rows: list[Row] = []
    for i, t in enumerate(truths, start=1):  # sequential: be gentle with the gateway
        rows.append(await run_one(folder, t, vision, day))
        logger.info("receipt %d/%d done", i, len(truths))
    return rows
