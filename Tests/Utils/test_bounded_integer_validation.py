"""Integer form values preserve spelling flexibility without numeric coercion."""

import pytest

from tldw_chatbook.Utils import input_validation


@pytest.mark.parametrize("value", [27, "27", "027", "+27", " 27 "])
def test_bounded_integer_returns_normalized_integer(value):
    assert (
        input_validation.validate_bounded_integer(value, minimum=1, maximum=1000) == 27
    )


@pytest.mark.parametrize(
    "value, expected", [(1, 1), ("1", 1), (1000, 1000), ("1000", 1000)]
)
def test_bounded_integer_accepts_inclusive_endpoints(value, expected):
    assert (
        input_validation.validate_bounded_integer(value, minimum=1, maximum=1000)
        == expected
    )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        27.0,
        27.5,
        float("nan"),
        float("inf"),
        None,
        "",
        "27.0",
        "NaN",
        0,
        "1001",
    ],
)
def test_bounded_integer_rejects_invalid_types_spellings_and_ranges(value):
    with pytest.raises(ValueError):
        input_validation.validate_bounded_integer(value, minimum=1, maximum=1000)
