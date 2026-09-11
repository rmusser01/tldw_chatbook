"""Shared input boundary for Console reasoning-history selectors."""

import pytest

from tldw_chatbook.Utils.input_validation import (
    validate_reasoning_history_selector,
)


@pytest.mark.parametrize("value", ["auto", "current", "all", "off"])
def test_reasoning_history_selector_accepts_exact_modes(value: str) -> None:
    assert validate_reasoning_history_selector(value) == value


def test_reasoning_history_override_accepts_inherit_only_when_requested() -> None:
    assert (
        validate_reasoning_history_selector("inherit", allow_inherit=True)
        == "inherit"
    )
    with pytest.raises(ValueError, match="selector value is invalid"):
        validate_reasoning_history_selector("inherit")


@pytest.mark.parametrize("value", [None, 1, True, "", "server_default"])
def test_reasoning_history_selector_rejects_unlisted_or_coerced_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="selector value is invalid"):
        validate_reasoning_history_selector(value, allow_inherit=True)
