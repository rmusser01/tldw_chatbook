"""Pure validation contracts for external MCP tool-result flags."""

from typing import Any

import pytest
from pydantic import ValidationError

from tldw_chatbook.Utils.input_validation import MCPToolResultInput

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("flags", "expected"),
    [({}, False), ({"isError": False}, False), ({"isError": True}, True)],
)
def test_tool_result_accepts_boolean_flags_without_interpreting_content(
    flags: dict[str, bool], expected: bool
) -> None:
    """Absent flags default to success while content remains opaque."""
    content = {"uninterpreted": [None, 1, {"type": "unknown"}]}
    result = MCPToolResultInput.model_validate(
        {**flags, "content": content, "extra_server_metadata": "ignored"}
    )

    assert result.is_error is expected
    assert result.content is content
    assert result.model_dump() == {"is_error": expected, "content": content}


@pytest.mark.parametrize(
    "flag", [1, 0, "true", "false", None, [], {}, {"secret": "private-flag-sentinel"}]
)
def test_tool_result_rejects_non_boolean_error_flags(flag: Any) -> None:
    """Lookalike and malformed flags cannot be coerced into success or failure."""
    with pytest.raises(ValidationError) as error:
        MCPToolResultInput.model_validate({"isError": flag})

    assert [
        (item["loc"], item["type"])
        for item in error.value.errors(include_input=False, include_url=False)
    ] == [(("isError",), "bool_type")]


def test_tool_result_missing_content_has_an_independent_empty_default() -> None:
    """A result with no content cannot share mutable content with another call."""
    first = MCPToolResultInput.model_validate({})
    second = MCPToolResultInput.model_validate({})

    assert first.is_error is False
    assert first.content == []
    first.content.append({"type": "text", "text": "first result only"})
    assert second.content == []
