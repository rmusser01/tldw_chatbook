"""Versioned Console roleplay conversation metadata contracts."""

import json

import pytest

from tldw_chatbook.Chat.console_roleplay_metadata import (
    ConsoleRoleplayContext,
    RoleplayContextVersionError,
    merge_console_roleplay_context,
    parse_console_roleplay_context,
)


def test_parse_version_one_context():
    raw = json.dumps(
        {
            "console_roleplay_context": {
                "version": 1,
                "user_name_override": "Captain Rowan",
                "character_system_template": "Speak with {{user}}.",
            }
        }
    )

    assert parse_console_roleplay_context(raw) == ConsoleRoleplayContext(
        user_name_override="Captain Rowan",
        character_system_template="Speak with {{user}}.",
    )


@pytest.mark.parametrize("payload", [None, "not json", "[]"])
def test_invalid_metadata_degrades_to_empty_context(payload):
    assert parse_console_roleplay_context(payload) == ConsoleRoleplayContext()


def test_future_version_degrades_without_guessing():
    raw = json.dumps(
        {
            "console_roleplay_context": {
                "version": 2,
                "user_name_override": "Do not trust this build",
            }
        }
    )

    assert parse_console_roleplay_context(raw) == ConsoleRoleplayContext()


def test_boolean_version_is_not_a_trusted_integer_version():
    raw = {
        "console_roleplay_context": {
            "version": True,
            "user_name_override": "Do not trust this build",
        }
    }

    assert parse_console_roleplay_context(raw) == ConsoleRoleplayContext()


def test_write_refuses_to_clobber_future_owned_version():
    raw = json.dumps(
        {
            "sibling": {"kept": True},
            "console_roleplay_context": {"version": 2, "future_field": "keep"},
        }
    )

    with pytest.raises(RoleplayContextVersionError, match="version 2"):
        merge_console_roleplay_context(
            raw, ConsoleRoleplayContext(user_name_override="Rowan")
        )


def test_merge_preserves_siblings_and_removes_empty_owned_object():
    raw = json.dumps(
        {"active_dictionaries": [4], "pinned_response_prefill": "Yes"}
    )

    merged = json.loads(
        merge_console_roleplay_context(
            raw, ConsoleRoleplayContext(user_name_override="Rowan")
        )
    )

    assert merged["active_dictionaries"] == [4]
    assert merged["pinned_response_prefill"] == "Yes"

    cleared = json.loads(
        merge_console_roleplay_context(json.dumps(merged), ConsoleRoleplayContext())
    )

    assert "console_roleplay_context" not in cleared


def test_parse_rejects_an_invalid_name_and_blank_system_template():
    raw = {
        "console_roleplay_context": {
            "version": 1,
            "user_name_override": "Rowan\n",
            "character_system_template": "  ",
        }
    }

    assert parse_console_roleplay_context(raw) == ConsoleRoleplayContext()


def test_merge_writes_only_nonblank_owned_fields():
    merged = json.loads(
        merge_console_roleplay_context(
            {},
            ConsoleRoleplayContext(
                user_name_override="  Rowan  ",
                character_system_template="  Speak plainly.  ",
            ),
        )
    )

    assert merged == {
        "console_roleplay_context": {
            "version": 1,
            "user_name_override": "Rowan",
            "character_system_template": "  Speak plainly.  ",
        }
    }
