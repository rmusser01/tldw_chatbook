from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
    CharacterConversationRow,
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
    UnavailableCharacterReason,
    UnresolvedConversationKey,
    deserialize_character_conversation_key,
    serialize_character_conversation_key,
)


def test_navigation_service_is_exposed_from_application_contract_module() -> None:
    assert CharacterConversationNavigationService.__module__ == (
        "tldw_chatbook.Character_Chat.character_conversation_navigation"
    )


def test_resolved_and_unresolved_row_keys_cannot_collide() -> None:
    resolved = CharacterConversationRow.resolved(
        LocalCharacterConversationTarget(
            character=ResolvedLocalCharacterKey("authority-A", 7),
            conversation_id="same-conversation",
        ),
        character_label="Ada",
        title="Resolved",
        last_modified="2026-09-03T12:00:00Z",
        created_at="2026-09-01T10:00:00Z",
    )
    unresolved = CharacterConversationRow.unavailable(
        UnresolvedConversationKey("authority-A", "same-conversation"),
        reason=UnavailableCharacterReason.MISSING_CARD,
        character_label="Ada",
        title="Unavailable",
        last_modified="2026-09-03T12:00:00Z",
        created_at="2026-09-01T10:00:00Z",
    )

    assert resolved.row_key != unresolved.row_key
    assert resolved.row_key.startswith("resolved_local_character:")
    assert unresolved.row_key.startswith("unresolved_conversation:")


def test_identity_rejects_blank_overlong_casefolded_or_path_derived_values() -> None:
    with pytest.raises(ValueError):
        ResolvedLocalCharacterKey(" ", 1)
    with pytest.raises(ValueError):
        UnresolvedConversationKey("authority", "x" * 257)
    with pytest.raises(ValueError):
        ResolvedLocalCharacterKey("authority", 0)
    with pytest.raises(ValueError):
        ResolvedLocalCharacterKey("authority", 2**63)
    with pytest.raises(TypeError):
        UnresolvedConversationKey(Path("/tmp/profile.sqlite"), "conversation")  # type: ignore[arg-type]

    mixed_case = UnresolvedConversationKey("Authority-A", "Conversation-X")
    payload = serialize_character_conversation_key(mixed_case)

    assert payload == {
        "version": 1,
        "tag": "unresolved_conversation",
        "data_authority_id": "Authority-A",
        "conversation_id": "Conversation-X",
    }
    assert deserialize_character_conversation_key(payload) == mixed_case
    with pytest.raises(ValueError):
        deserialize_character_conversation_key({**payload, "version": 2})
    with pytest.raises(ValueError):
        deserialize_character_conversation_key({**payload, "tag": "persona"})


def test_unavailable_reason_changes_without_changing_unresolved_identity() -> None:
    key = UnresolvedConversationKey("authority-A", "conversation-1")
    missing = CharacterConversationRow.unavailable(
        key,
        reason=UnavailableCharacterReason.MISSING_CARD,
        character_label="Historical Ada",
        title="Old chat",
        last_modified="2026-09-03T12:00:00Z",
        created_at="2026-09-01T10:00:00Z",
    )
    changed = CharacterConversationRow.unavailable(
        key,
        reason=UnavailableCharacterReason.DELETED_CARD,
        character_label="Historical Ada",
        title="Old chat",
        last_modified="2026-09-03T12:00:00Z",
        created_at="2026-09-01T10:00:00Z",
    )

    assert missing.row_key == changed.row_key
    assert missing.unresolved == changed.unresolved == key
    assert missing.unavailable_reason != changed.unavailable_reason
    with pytest.raises(FrozenInstanceError):
        missing.title = "mutation"  # type: ignore[misc]


def test_persona_rows_are_never_character_conversation_targets() -> None:
    with pytest.raises(TypeError):
        LocalCharacterConversationTarget(  # type: ignore[arg-type]
            character={"assistant_kind": "persona", "assistant_id": "7"},
            conversation_id="conversation-1",
        )
