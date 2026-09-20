"""Shared clone input validation preserves the server's normalized admission identity."""

import pytest

from tldw_chatbook.Utils.input_validation import validate_sharing_clone_input


@pytest.mark.parametrize(
    "share_id,name,expected",
    [
        (" 7 ", " My\t copy  ", (7, "My copy")),
        (7, " ", (7, None)),
        (7, None, (7, None)),
        (7, "x" * 255, (7, "x" * 255)),
    ],
)
def test_clone_input_normalizes_valid_values(share_id, name, expected):
    assert validate_sharing_clone_input(share_id, name) == expected


@pytest.mark.parametrize(
    "share_id,name",
    [
        ("0", "Copy"),
        (-1, "Copy"),
        (True, "Copy"),
        ("7.5", "Copy"),
        ("invalid", "Copy"),
        (7, 8),
        (7, "x" * 256),
    ],
)
def test_clone_input_rejects_invalid_values(share_id, name):
    with pytest.raises(ValueError):
        validate_sharing_clone_input(share_id, name)
