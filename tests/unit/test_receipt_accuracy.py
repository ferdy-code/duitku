"""T-404: the accuracy experiment's logic (truth loading, matching, scoring, runner, report).
Pure: no network. Builds NO claim about real accuracy: that needs real receipts."""

import datetime as dt
import json
from pathlib import Path

import pytest
from tests.fake_vision import FakeVision, extraction

from receipt_eval.report import render_detail, render_summary
from receipt_eval.runner import run_all, unlabelled_files
from receipt_eval.scoring import MIN_RECEIPTS, Row, merchant_match, score, summarize
from receipt_eval.truth import Truth, TruthError, load_truth

JPEG = b"\xff\xd8\xff\xe0" + b"j" * 50
TODAY = dt.date(2026, 10, 14)


# --- truth -------------------------------------------------------------------------------------


def test_csv_truth_loads(tmp_path: Path) -> None:
    f = tmp_path / "truth.csv"
    f.write_text(
        'filename,merchant,total_idr\nr1.jpg,Indomaret,35000\nr2.png,"Warung, Bu Ani",12500\n'
    )
    assert load_truth(f) == [
        Truth("r1.jpg", "Indomaret", 35_000),
        Truth("r2.png", "Warung, Bu Ani", 12_500),
    ]


def test_json_truth_loads(tmp_path: Path) -> None:
    f = tmp_path / "truth.json"
    f.write_text(json.dumps([{"filename": "a.jpg", "merchant": "X", "total_idr": 1000}]))
    assert load_truth(f) == [Truth("a.jpg", "X", 1000)]


@pytest.mark.parametrize(
    "body",
    [
        "filename,merchant,total_idr\n",  # no rows
        "filename,merchant\nr1.jpg,X\n",  # missing column
        "filename,merchant,total_idr\nr1.jpg,X,Rp35.000\n",  # not a plain integer
        "filename,merchant,total_idr\nr1.jpg,X,0\n",
        "filename,merchant,total_idr\nr1.jpg,,5000\n",  # empty merchant
        "filename,merchant,total_idr\nr1.jpg,X,5000\nr1.jpg,Y,6000\n",  # duplicate file
    ],
)
def test_bad_truth_is_an_error_not_a_silent_skip(tmp_path: Path, body: str) -> None:
    f = tmp_path / "truth.csv"
    f.write_text(body)
    with pytest.raises(TruthError):
        load_truth(f)


def test_json_truth_must_be_a_list_of_objects(tmp_path: Path) -> None:
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"filename": "a"}))
    with pytest.raises(TruthError):
        load_truth(f)


# --- merchant matching ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("predicted", "truth", "expected"),
    [
        ("Indomaret", "INDOMARET", True),
        ("INDOMARET PLAZA BARU", "Indomaret", True),  # branch name appended
        ("Alfa Mart", "Alfamart", True),  # spacing
        ("Alfarnart", "Alfamart", True),  # one misread letter
        ("PT Sumber Alfaria Trijaya Tbk", "Sumber Alfaria Trijaya", True),  # legal form
        ("Kopi Kenangan", "Kopi-Kenangan!", True),
        ("Warung Sate", "Sate Khas Senayan", False),
        ("Indomaret", "Alfamart", False),
        ("", "Indomaret", False),
        (None, "Indomaret", False),
        ("Ab", "Abc Store", False),  # too short to count as containment
    ],
)
def test_merchant_match(predicted: str | None, truth: str, expected: bool) -> None:
    assert merchant_match(predicted, truth) is expected


# --- scoring -------------------------------------------------------------------------------------

T = Truth("r.jpg", "Indomaret", 35_000)


def test_exact_total_and_merchant_is_correct() -> None:
    r = score(T, extraction())
    assert (r.total_ok, r.merchant_ok, r.correct, r.flagged) == (True, True, True, False)


def test_off_by_one_rupiah_is_wrong_total() -> None:
    r = score(T, extraction(total_idr=35_001, items=[]))
    assert not r.total_ok and r.merchant_ok and not r.correct


def test_wrong_merchant_is_not_correct_even_with_the_right_total() -> None:
    r = score(T, extraction(merchant="Alfamart"))
    assert r.total_ok and not r.merchant_ok and not r.correct


def test_error_rows_are_wrong_everywhere_and_flagged() -> None:
    r = score(T, None, "vision: ValidationError")
    assert not (r.total_ok or r.merchant_ok or r.correct) and r.flagged


def test_warnings_or_low_confidence_flag_a_row() -> None:
    assert score(T, extraction(warnings=["blurry"])).flagged
    assert score(T, extraction(confidence=0.4)).flagged


def rows(correct: int, wrong_flagged: int, wrong_silent: int, errors: int = 0) -> list[Row]:
    out = [score(T, extraction()) for _ in range(correct)]
    out += [score(T, extraction(total_idr=1, warnings=["x"])) for _ in range(wrong_flagged)]
    out += [score(T, extraction(total_idr=1)) for _ in range(wrong_silent)]
    out += [score(T, None, "boom") for _ in range(errors)]
    return out


def test_summary_counts() -> None:
    s = summarize(rows(correct=17, wrong_flagged=2, wrong_silent=1, errors=1))
    assert (s.n, s.correct, s.wrong, s.wrong_flagged, s.errors) == (21, 17, 4, 3, 1)
    assert s.total_ok == 17 and s.merchant_ok == 20  # the three wrong totals kept the merchant
    assert s.rate(s.correct) == pytest.approx(17 / 21)


def test_target_needs_85_percent_AND_enough_receipts() -> None:
    assert summarize(rows(17, 0, 3)).meets_target  # 17/20 = 85%
    assert not summarize(rows(16, 0, 4)).meets_target  # 80%
    small = summarize(rows(10, 0, 0))  # 100% but only 10 receipts
    assert not small.meets_target and any(str(MIN_RECEIPTS) in n for n in small.notes)


# --- runner + report -----------------------------------------------------------------------------


async def test_run_all_scores_a_folder_and_survives_failures(tmp_path: Path) -> None:
    (tmp_path / "ok.jpg").write_bytes(JPEG)
    (tmp_path / "fake.jpg").write_bytes(b"<html>")
    (tmp_path / "extra.png").write_bytes(JPEG)
    truths = [
        Truth("ok.jpg", "Indomaret", 35_000),
        Truth("fake.jpg", "X", 1000),  # not an image
        Truth("gone.jpg", "X", 1000),  # missing
    ]
    result = await run_all(tmp_path, truths, FakeVision(), TODAY)
    assert [r.correct for r in result] == [True, False, False]
    assert result[1].error is not None and result[1].error.startswith("file:")
    assert result[2].error is not None
    assert unlabelled_files(tmp_path, truths) == ["extra.png"]


async def test_vision_failure_becomes_an_error_row(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(JPEG)
    (r,) = await run_all(tmp_path, [Truth("a.jpg", "X", 1)], FakeVision(error="APIError"), TODAY)
    assert r.error == "vision: APIError" and not r.correct


async def test_runner_cannot_be_pointed_outside_the_folder(tmp_path: Path) -> None:
    inside = tmp_path / "in"
    inside.mkdir()
    (tmp_path / "secret.jpg").write_bytes(JPEG)
    (r,) = await run_all(inside, [Truth("../secret.jpg", "X", 1)], FakeVision(), TODAY)
    assert r.error is not None and r.error.startswith("file:")


def test_summary_report_has_rates_but_no_merchants_or_filenames() -> None:
    rs = rows(17, 2, 1)
    text = render_summary(summarize(rs), "vision-x")
    assert "17/20 = 85%" in text and "MET" in text and "vision-x" in text
    assert "Indomaret" not in text and "r.jpg" not in text  # safe to paste into DECISIONS.md


def test_detail_report_lists_every_row_and_escapes_table_pipes() -> None:
    r = score(
        Truth("a|b.jpg", "Toko | Baru", 5000),
        extraction(merchant="Toko | Baru", total_idr=5000, items=[]),
    )
    text = render_detail([r])
    assert text.count("\n| a/b.jpg") == 1 and "Toko / Baru" in text
