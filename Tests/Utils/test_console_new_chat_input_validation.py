"""Strict chat-creation projection and safe shared-boundary failures."""

import pytest

from tldw_chatbook.Agents.agent_models import (
    CHAT_CREATE_PAYLOAD_MAX,
    CHAT_CREATE_TITLE_MAX,
)
from tldw_chatbook.Chat.console_agent_bridge import validate_new_chat_arguments
from tldw_chatbook.Utils import input_validation


def _shared_validate(arguments):
    validator = getattr(input_validation, "validate_console_new_chat_arguments", None)
    assert callable(validator), (
        "chat creation needs the shared Pydantic validation boundary"
    )
    return validator(arguments)


@pytest.mark.parametrize("validate", [_shared_validate, validate_new_chat_arguments])
def test_defaults_discard_authority_and_preserve_literal_fields(validate):
    assert validate({"source_run_id": "forged", "workspace_id": "secret"}) == {
        "title": "",
        "opening_prompt": "",
        "instructions": "",
        "destination": "same_workspace",
        "mode": "draft",
        "provider": "",
        "model": "",
        "preset": "",
    }
    literal = "  /new @helper\n\t exact bytes  "
    validated = validate(
        {
            "title": "  title  ",
            "opening_prompt": literal,
            "instructions": literal,
            "mode": "start",
            "destination": "casual",
            "provider": " cloud ",
            "model": " m ",
            "preset": " p ",
        }
    )
    assert validated["title"] == "title"
    assert validated["opening_prompt"] == validated["instructions"] == literal
    assert validated["provider"] == " cloud "
    assert validated["model"] == " m "
    assert validated["preset"] == " p "


@pytest.mark.parametrize("validate", [_shared_validate, validate_new_chat_arguments])
@pytest.mark.parametrize(
    "field",
    [
        "title",
        "opening_prompt",
        "instructions",
        "destination",
        "mode",
        "provider",
        "model",
        "preset",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        True,
        b"secret-payload",
        ["secret-payload"],
        {"secret-payload": "private"},
    ],
)
def test_every_public_field_requires_a_string_without_diagnostic_leakage(
    validate, field, value
):
    with pytest.raises(
        ValueError, match=f"^invalid_args: {field} must be a string$"
    ) as error:
        validate({field: value})
    assert "secret-payload" not in str(error.value)
    assert "private" not in str(error.value)


@pytest.mark.parametrize("validate", [_shared_validate, validate_new_chat_arguments])
@pytest.mark.parametrize(
    ("field", "limit"),
    [
        ("title", CHAT_CREATE_TITLE_MAX),
        ("opening_prompt", CHAT_CREATE_PAYLOAD_MAX),
        ("instructions", CHAT_CREATE_PAYLOAD_MAX),
    ],
)
def test_exact_limits_and_length_check_before_title_trim(validate, field, limit):
    assert validate({field: "x" * limit})[field] == "x" * limit
    with pytest.raises(ValueError, match="^payload_too_large:"):
        validate({field: "x" * limit + " "})


@pytest.mark.parametrize("validate", [_shared_validate, validate_new_chat_arguments])
@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"destination": "secret-payload"}, "invalid_args: destination"),
        ({"mode": "secret-payload"}, "invalid_args: mode"),
        (
            {"mode": "start", "opening_prompt": " \n\t"},
            "invalid_args: start requires a nonblank opening_prompt",
        ),
        (
            {"title": "x" * (CHAT_CREATE_TITLE_MAX + 1), "provider": None},
            "invalid_args: provider must be a string",
        ),
    ],
)
def test_closed_choices_nonblank_start_and_stable_type_error_precedence(
    validate, arguments, error
):
    with pytest.raises(ValueError) as captured:
        validate(arguments)
    assert str(captured.value) == error
