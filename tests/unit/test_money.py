import pytest

from duitku.domain.money import MAX_IDR, AmountError, parse_idr_amount


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("35rb", 35_000),
        ("35k", 35_000),
        ("35 ribu", 35_000),
        ("35 RB", 35_000),
        ("35K", 35_000),
        ("1,5 juta", 1_500_000),
        ("2.5jt", 2_500_000),
        ("1jt", 1_000_000),
        ("2 Juta", 2_000_000),
        ("1,5k", 1_500),
        ("0,5 juta", 500_000),
        ("1,25jt", 1_250_000),
        ("1 miliar", 1_000_000_000),
        ("2,5 milyar", 2_500_000_000),
        ("Rp 1.250.000", 1_250_000),
        ("Rp1.250.000", 1_250_000),
        ("rp35rb", 35_000),
        ("Rp. 35.000", 35_000),
        ("IDR 35000", 35_000),
        ("idr35rb", 35_000),
        ("35.000", 35_000),
        ("35,000", 35_000),
        ("1,250,000", 1_250_000),
        ("35000", 35_000),
        ("  35000  ", 35_000),
        ("1", 1),
        ("007", 7),
        (f"{MAX_IDR}", MAX_IDR),
        ("10 miliar", MAX_IDR),
        ("9.999.999.999", 9_999_999_999),
    ],
)
def test_parses(text: str, expected: int) -> None:
    result = parse_idr_amount(text)
    assert result == expected
    assert type(result) is int


@pytest.mark.parametrize(
    "text",
    [
        "1.5",  # decimal or thousands? never guess
        "1,5",
        "1.25",
        "35.0000",
        "1.250,000",  # mixed separators
        "1,000.000",
        "1.250.00",
        "5m",  # million or billion
        "5 M",
        "5x",
        "35 ribuan",
        "1.250.000jt",  # thousands separators + multiplier
        "1,2345rb",  # fractional rupiah
    ],
)
def test_ambiguous(text: str) -> None:
    with pytest.raises(AmountError) as exc:
        parse_idr_amount(text)
    assert exc.value.code == "AMBIGUOUS_AMOUNT"
    assert exc.value.hint


@pytest.mark.parametrize(
    "text",
    [
        "0",
        "0rb",
        "0.000",
        "-5rb",
        "-35000",
        "Rp -5rb",
        "10000000001",
        "11 miliar",
        "",
        "   ",
        "abc",
        "rb",
        "Rp",
        "35 rb 10",
        "1 000",
        "١٢٣",  # non-ASCII digits
        "9" * 100,
    ],
)
def test_invalid(text: str) -> None:
    with pytest.raises(AmountError) as exc:
        parse_idr_amount(text)
    assert exc.value.code == "INVALID_AMOUNT"
    assert exc.value.hint
