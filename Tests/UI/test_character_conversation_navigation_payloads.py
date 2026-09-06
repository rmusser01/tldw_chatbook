from __future__ import annotations

import pytest

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    ResolvedLocalCharacterKey,
    UnresolvedConversationKey,
)
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    LibraryCharacterRepairContext,
    RoleplayCharacterConversationLink,
    RoleplayReturnTarget,
    deserialize_library_character_repair_context,
    deserialize_roleplay_character_conversation_link,
    serialize_library_character_repair_context,
    serialize_roleplay_character_conversation_link,
)


def test_roleplay_payload_round_trip_preserves_exact_typed_key_and_safe_fields() -> (
    None
):
    link = RoleplayCharacterConversationLink(
        character=ResolvedLocalCharacterKey("Authority-A", 7),
        conversation_id="Conversation-X",
        query="silver rain",
        data_revision=12,
        return_target=RoleplayReturnTarget.console_context_character(),
    )

    payload = serialize_roleplay_character_conversation_link(link)

    assert deserialize_roleplay_character_conversation_link(payload) == link
    assert payload == {
        "version": 1,
        "source": "local",
        "character": {
            "version": 1,
            "tag": "resolved_local_character",
            "data_authority_id": "Authority-A",
            "character_id": 7,
        },
        "conversation_id": "Conversation-X",
        "query": "silver rain",
        "data_revision": 12,
        "return_target": {
            "screen_id": "chat",
            "focus_id": "console-context-character",
        },
    }
    assert not ({"prompt", "transcript", "credential", "card", "path"} & payload.keys())


def test_library_repair_payload_round_trip_preserves_exact_unresolved_key() -> None:
    context = LibraryCharacterRepairContext(
        unresolved=UnresolvedConversationKey("Authority-A", "Conversation-X"),
        expected_conversation_version=4,
        historical_display_snapshot="Historical Ada",
        return_target=RoleplayReturnTarget.personas_conversations(),
    )

    payload = serialize_library_character_repair_context(context)

    assert deserialize_library_character_repair_context(payload) == context
    assert payload["unresolved"]["tag"] == "unresolved_conversation"
    assert not (
        {"prompt", "transcript", "credential", "card_body", "filesystem_path"}
        & payload.keys()
    )


@pytest.mark.parametrize("version", [0, 2, "1", None])
def test_payloads_reject_unknown_versions(version: object) -> None:
    payload = serialize_roleplay_character_conversation_link(
        RoleplayCharacterConversationLink(ResolvedLocalCharacterKey("authority", 1))
    )
    payload["version"] = version

    with pytest.raises(ValueError, match="version"):
        deserialize_roleplay_character_conversation_link(payload)


@pytest.mark.parametrize("conversation_id", ["", " ", "x" * 257])
def test_roleplay_payload_rejects_blank_or_overlong_conversation_ids(
    conversation_id: str,
) -> None:
    payload = serialize_roleplay_character_conversation_link(
        RoleplayCharacterConversationLink(ResolvedLocalCharacterKey("authority", 1))
    )
    payload["conversation_id"] = conversation_id

    with pytest.raises(ValueError, match="conversation_id"):
        deserialize_roleplay_character_conversation_link(payload)


def test_payloads_reject_nonlocal_sources_and_mismatched_authority_components() -> None:
    roleplay = serialize_roleplay_character_conversation_link(
        RoleplayCharacterConversationLink(ResolvedLocalCharacterKey("authority", 1))
    )
    roleplay["source"] = "server"
    with pytest.raises(ValueError, match="local"):
        deserialize_roleplay_character_conversation_link(roleplay)

    repair = serialize_library_character_repair_context(
        LibraryCharacterRepairContext(
            unresolved=UnresolvedConversationKey("authority", "conversation"),
            expected_conversation_version=1,
            historical_display_snapshot="Historical",
            return_target=RoleplayReturnTarget.personas_filter(),
        )
    )
    repair["data_authority_id"] = "other-authority"
    with pytest.raises(ValueError, match="authority"):
        deserialize_library_character_repair_context(repair)


@pytest.mark.parametrize(
    "focus_id", ["", " ", "#already-prefixed", "bad focus", "x" * 129, "../path"]
)
def test_return_target_rejects_invalid_focus_ids(focus_id: str) -> None:
    with pytest.raises(ValueError, match="focus_id"):
        RoleplayReturnTarget("chat", focus_id)


@pytest.mark.parametrize(
    ("payload_name", "nested_field"),
    (("roleplay", "character"), ("repair", "unresolved")),
)
def test_payloads_reject_unexpected_nested_identity_fields(
    payload_name: str, nested_field: str
) -> None:
    """A nested identity parser must not accept authority-adjacent data."""

    roleplay = serialize_roleplay_character_conversation_link(
        RoleplayCharacterConversationLink(ResolvedLocalCharacterKey("authority", 1))
    )
    repair = serialize_library_character_repair_context(
        LibraryCharacterRepairContext(
            unresolved=UnresolvedConversationKey("authority", "conversation"),
            expected_conversation_version=1,
            historical_display_snapshot="Historical",
            return_target=RoleplayReturnTarget.personas_conversations(),
        )
    )
    payload = roleplay if payload_name == "roleplay" else repair
    payload[nested_field]["unexpected"] = "must-not-be-accepted"

    parser = (
        deserialize_roleplay_character_conversation_link
        if payload_name == "roleplay"
        else deserialize_library_character_repair_context
    )
    with pytest.raises(ValueError, match="identity fields"):
        parser(payload)


@pytest.mark.parametrize(
    ("screen_id", "focus_id"),
    (("settings", "arbitrary-focus"), ("chat", "personas-filter")),
)
def test_return_target_rejects_routes_outside_closed_character_flow(
    screen_id: str, focus_id: str
) -> None:
    """Character navigation cannot smuggle an arbitrary app destination."""

    with pytest.raises(ValueError, match="return target"):
        RoleplayReturnTarget(screen_id, focus_id)


def test_exact_conversation_link_requires_captured_query_revision() -> None:
    """Resume payloads bind the immutable list snapshot they came from."""

    with pytest.raises(ValueError, match="data_revision"):
        RoleplayCharacterConversationLink(
            ResolvedLocalCharacterKey("authority", 1),
            conversation_id="conversation",
        )
