from duitku.tools.compare_amounts import compare


def test_compare_difference_direction_percent() -> None:
    assert compare(1_200_000, 900_000) == {
        "difference_idr": 300_000,
        "direction": "higher",
        "percent_change": "33.3",
    }
    assert compare(450_000, 520_000) == {
        "difference_idr": -70_000,
        "direction": "lower",
        "percent_change": "-13.5",
    }
    assert compare(5, 5)["direction"] == "same"
    assert compare(5, 0)["percent_change"] is None
