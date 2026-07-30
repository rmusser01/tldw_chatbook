"""Trusted Console speech snapshot contracts."""

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_speech import (
    ConsoleSpeechSnapshotRejected,
    ConsoleSpeechSnapshotRejectionCode,
    TTSMessageSpeechSnapshot,
)
from tldw_chatbook.TTS.profile_types import CharacterRef


def _snapshot(
    *,
    raw_content: str = "private response",
    character_ref: CharacterRef | None = None,
) -> TTSMessageSpeechSnapshot:
    return TTSMessageSpeechSnapshot(
        session_id="session-1",
        message_id="message-1",
        persisted_conversation_id=None,
        persisted_message_id=None,
        raw_content=raw_content,
        selected_variant_id="message-1",
        speech_revision=0,
        persisted_message_version=None,
        role=ConsoleMessageRole.ASSISTANT,
        status="complete",
        assistant_kind="character" if character_ref is not None else "generic",
        character_ref=character_ref,
    )


def test_snapshot_is_frozen_and_redacts_content_and_authority_from_repr():
    character_ref = CharacterRef(
        source="local",
        authority_id="authority-secret",
        character_id="17",
    )
    snapshot = _snapshot(
        raw_content="do not log this response",
        character_ref=character_ref,
    )

    rendered = repr(snapshot)

    assert "do not log this response" not in rendered
    assert "authority-secret" not in rendered
    assert snapshot.raw_content == "do not log this response"
    assert snapshot.character_ref == character_ref
    with pytest.raises(FrozenInstanceError):
        snapshot.raw_content = "changed"  # type: ignore[misc]


def test_snapshot_rejects_invalid_structural_values():
    with pytest.raises(ValueError, match="speech_revision"):
        TTSMessageSpeechSnapshot(
            session_id="session-1",
            message_id="message-1",
            persisted_conversation_id=None,
            persisted_message_id=None,
            raw_content="response",
            selected_variant_id="message-1",
            speech_revision=-1,
            persisted_message_version=None,
            role=ConsoleMessageRole.ASSISTANT,
            status="complete",
            assistant_kind="generic",
            character_ref=None,
        )


def test_snapshot_rejection_exposes_only_bounded_code_and_safe_retry_copy():
    error = ConsoleSpeechSnapshotRejected(
        ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
    )

    assert error.code is ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
    assert str(error) == "Message changed before speech started; select Speak again."
    assert "private response" not in repr(error)

    with pytest.raises(ValueError, match="rejection code"):
        ConsoleSpeechSnapshotRejected("unbounded-detail")  # type: ignore[arg-type]
