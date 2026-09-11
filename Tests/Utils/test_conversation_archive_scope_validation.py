"""Shared finite lifecycle scope validation at UI and storage boundaries."""

import pytest

from tldw_chatbook.Utils.input_validation import validate_conversation_archive_scope


@pytest.mark.parametrize("scope", ["active", "archived", "all"])
def test_archive_scope_preserves_exact_supported_values(scope):
    assert validate_conversation_archive_scope(scope) == scope


@pytest.mark.parametrize(
    "scope",
    [
        None,
        True,
        1,
        b"active",
        [],
        {},
        "",
        "Active",
        " active",
        "all ",
        "deleted",
        "archived\x00",
    ],
)
def test_archive_scope_rejects_unknown_or_coerced_values(scope):
    with pytest.raises(
        ValueError, match="archive_scope must be active, archived, or all"
    ):
        validate_conversation_archive_scope(scope)
