"""Markdown report. Two parts on purpose: the SUMMARY holds only counts and rates (safe to paste
into docs/DECISIONS.md), the DETAIL table holds merchants and totals from real receipts and is
therefore written next to the images (data/ is gitignored), never into a committed folder."""

from receipt_eval.scoring import MIN_RECEIPTS, TARGET, Row, Summary


def _pct(part: int, whole: int) -> str:
    return f"{part}/{whole} = {100 * part / whole:.0f}%" if whole else "n/a"


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def render_summary(s: Summary, model: str) -> str:
    verdict = "MET" if s.meets_target else "NOT MET (or fewer than 20 receipts)"
    lines = [
        "# Receipt extraction accuracy (T-404, G3)",
        "",
        f"- vision model: `{model}`",
        f"- receipts scored: {s.n} (the AC needs >= {MIN_RECEIPTS} real ones)",
        f"- total exact: {_pct(s.total_ok, s.n)}",
        f"- merchant match: {_pct(s.merchant_ok, s.n)}",
        f"- **correct (total AND merchant): {_pct(s.correct, s.n)}**, "
        f"target >= {TARGET:.0%}: {verdict}",
        f"- failures (model/file error): {s.errors}",
        f"- wrong rows that carried a warning, low confidence or error: "
        f"{_pct(s.wrong_flagged, s.wrong)} (the rest reached the approval step silently)",
    ]
    lines += [f"- NOTE: {n}" for n in s.notes]
    return "\n".join(lines) + "\n"


def render_detail(rows: list[Row]) -> str:
    out = [
        "## Per receipt (private: real merchants and totals, keep out of git)",
        "",
        "| file | total | merchant | predicted total / truth | predicted merchant / truth | "
        "conf | warnings | error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        conf = "" if r.confidence is None else f"{r.confidence:.2f}"
        out.append(
            f"| {_cell(r.filename)} | {'ok' if r.total_ok else 'WRONG'} | "
            f"{'ok' if r.merchant_ok else 'WRONG'} | {r.predicted_total} / {r.truth_total} | "
            f"{_cell(str(r.predicted_merchant))} / {_cell(r.truth_merchant)} | {conf} | "
            f"{_cell('; '.join(r.warnings))} | {_cell(r.error or '')} |"
        )
    return "\n".join(out) + "\n"
