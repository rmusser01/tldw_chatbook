"""Shared native QA identifiers preserve exact names and reject tmux syntax."""

import pytest


@pytest.mark.parametrize("value", ["a", "0", "A0_b-c", "x" * 64])
def test_tmux_identifier_preserves_valid_names(value: str) -> None:
    from tldw_chatbook.Utils.input_validation import validate_tmux_identifier

    assert validate_tmux_identifier(value) == value


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        True,
        b"name",
        "",
        "x" * 65,
        "_name",
        "-name",
        "../name",
        "name:window",
        "space name",
        "name\n",
        "name\r",
        "éname",
        "name\x00",
    ],
)
def test_tmux_identifier_rejects_unsafe_or_coerced_names(value: object) -> None:
    from tldw_chatbook.Utils.input_validation import validate_tmux_identifier

    with pytest.raises(ValueError):
        validate_tmux_identifier(value)
