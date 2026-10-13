"""Speech snapshot authority delegated by the native Chat store."""

from __future__ import annotations

from tldw_chatbook.Chat.console_chat_store import (
    ConsoleMessageRole,
    ConsolePresentationContext,
    ConsoleSpeechSnapshotRejected,
    ConsoleSpeechSnapshotRejectionCode,
    TTSMessageSpeechSnapshot,
    resolve_console_message_presentation,
)


def issue_tts_message_speech_snapshot(
    self,
    message_id: str,
    *,
    presentation_context: ConsolePresentationContext | None = None,
    owner_session_id: str | None = None,
) -> TTSMessageSpeechSnapshot:
    """Issue a trusted snapshot for one speakable active-path message.

    Args:
        message_id: Native Console message selected by the user.
        presentation_context: Optional live identity used to resolve trusted
            character-template content. ``None`` preserves neutral callers.
        owner_session_id: Explicit local owner for app-owned Buddy speech.
            Omission retains the visible-session requirement.

    Returns:
        An immutable snapshot bound to the exact selected text and
        current trusted session authorship.

    Raises:
        ConsoleSpeechSnapshotRejected: If the message is missing,
            inactive, incomplete, non-assistant, blank, or cannot be
            durably version-fenced.
    """
    session_id = self._message_session_index.get(message_id)
    if session_id is None:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MISSING_MESSAGE
        )
    if session_id != (
        self.active_session_id if owner_session_id is None else owner_session_id
    ):
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.SESSION_CHANGED
        )
    session = self._sessions.get(session_id)
    message = self._nodes_by_session.get(session_id, {}).get(message_id)
    if session is None or message is None:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MISSING_MESSAGE
        )
    if owner_session_id is not None and session.runtime_backend != "local":
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.SESSION_CHANGED
        )
    if message.generation_projection_quarantined:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
        )
    if (
        message.persisted_message_id is not None
        and session.persisted_conversation_id is None
    ):
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
        )
    if message_id not in self.active_path_message_ids(session_id):
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
        )
    raw_content, selected_variant_id = self._speech_selection(message)
    if presentation_context is not None:
        if not isinstance(presentation_context, ConsolePresentationContext):
            raise ValueError(
                "presentation_context must be ConsolePresentationContext or None"
            )
        raw_content = resolve_console_message_presentation(
            self._snapshot(message), presentation_context
        ).content
    if (
        message.role is not ConsoleMessageRole.ASSISTANT
        or message.status != "complete"
        or not raw_content.strip()
    ):
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_NOT_SPEAKABLE
        )
    speech_revision = self._message_speech_revisions.get(message_id)
    if type(speech_revision) is not int or speech_revision < 0:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
        )
    if session.assistant_kind is not None and type(session.assistant_kind) is not str:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.AUTHORSHIP_CHANGED
        )
    persisted_version = self._persisted_message_version_or_reject(message)
    return TTSMessageSpeechSnapshot(
        session_id=session_id,
        message_id=message.id,
        persisted_conversation_id=session.persisted_conversation_id,
        persisted_message_id=message.persisted_message_id,
        raw_content=raw_content,
        selected_variant_id=selected_variant_id,
        speech_revision=speech_revision,
        persisted_message_version=persisted_version,
        role=message.role,
        status=message.status,
        assistant_kind=session.assistant_kind,
        character_ref=session.character_ref(),
    )


def validate_tts_message_speech_snapshot(
    self,
    snapshot: TTSMessageSpeechSnapshot,
    *,
    presentation_context: ConsolePresentationContext | None = None,
    owner_session_id: str | None = None,
) -> str:
    """Revalidate an issued Console speech snapshot against live state.

    Args:
        snapshot: Immutable snapshot previously issued by this store.
        presentation_context: Optional fresh identity used to re-resolve
            trusted character-template content before comparison.
        owner_session_id: Exact local owner, matching the issue-time opt-in.
            The caller also validates its captured Buddy binding.

    Returns:
        The captured exact raw content after every identity, state,
        authorship, and durable-version check succeeds.

    Raises:
        ConsoleSpeechSnapshotRejected: If any captured fact is stale or
            cannot be re-established.
    """
    if type(snapshot) is not TTSMessageSpeechSnapshot:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
        )
    if snapshot.session_id != (
        self.active_session_id if owner_session_id is None else owner_session_id
    ):
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.SESSION_CHANGED
        )
    session = self._sessions.get(snapshot.session_id)
    if session is None:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.SESSION_CHANGED
        )
    if owner_session_id is not None and session.runtime_backend != "local":
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.SESSION_CHANGED
        )
    owner_session_id = self._message_session_index.get(snapshot.message_id)
    if owner_session_id is None:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MISSING_MESSAGE
        )
    if owner_session_id != snapshot.session_id:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
        )
    message = self._nodes_by_session.get(snapshot.session_id, {}).get(
        snapshot.message_id
    )
    if message is None:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MISSING_MESSAGE
        )
    if message.generation_projection_quarantined:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
        )
    if (
        message.persisted_message_id is not None
        and session.persisted_conversation_id is None
    ):
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
        )
    if snapshot.message_id not in self.active_path_message_ids(snapshot.session_id):
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
        )
    raw_content, selected_variant_id = self._speech_selection(message)
    if presentation_context is not None:
        if not isinstance(presentation_context, ConsolePresentationContext):
            raise ValueError(
                "presentation_context must be ConsolePresentationContext or None"
            )
        raw_content = resolve_console_message_presentation(
            self._snapshot(message), presentation_context
        ).content
    if (
        message.role is not ConsoleMessageRole.ASSISTANT
        or message.status != "complete"
        or not raw_content.strip()
    ):
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_NOT_SPEAKABLE
        )
    if (
        message.id != snapshot.message_id
        or message.role is not snapshot.role
        or message.status != snapshot.status
        or selected_variant_id != snapshot.selected_variant_id
        or raw_content != snapshot.raw_content
        or self._message_speech_revisions.get(message.id) != snapshot.speech_revision
        or session.persisted_conversation_id != snapshot.persisted_conversation_id
        or message.persisted_message_id != snapshot.persisted_message_id
    ):
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
        )
    if (
        session.assistant_kind != snapshot.assistant_kind
        or session.character_ref() != snapshot.character_ref
    ):
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.AUTHORSHIP_CHANGED
        )
    current_version = self._persisted_message_version_or_reject(message)
    if current_version != snapshot.persisted_message_version:
        raise ConsoleSpeechSnapshotRejected(
            ConsoleSpeechSnapshotRejectionCode.PERSISTED_VERSION_CHANGED
        )
    return snapshot.raw_content
