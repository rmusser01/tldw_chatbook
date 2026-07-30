"""Pre-cooldown admission tests for trusted Console speech snapshots."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest
from loguru import logger

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_speech import (
    ConsoleSpeechSnapshotRejected,
    ConsoleSpeechSnapshotRejectionCode,
    TTSMessageSpeechSnapshot,
)
from tldw_chatbook.Event_Handlers.TTS_Events import tts_events as tts_events_module
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSEventHandler,
    TTSMessageSpeechRequestEvent,
    TTSRequestEvent,
)


class _RecordingHandler(TTSEventHandler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[object] = []
        self.generated: list[tuple[str, str | None, str | None]] = []
        self._request_cooldown = {}

    async def post_message(self, message: object) -> None:
        self.messages.append(message)

    async def _generate_tts_with_rate_limit(
        self,
        text: str,
        message_id: str | None,
        voice: str | None,
    ) -> None:
        self.generated.append((text, message_id, voice))


def _issued_snapshot() -> tuple[
    ConsoleChatStore,
    TTSMessageSpeechSnapshot,
]:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="  Exact   Console response.\n",
    )
    return store, store.issue_tts_message_speech_snapshot(message.id)


def test_message_speech_event_carries_snapshot_and_validator_without_text_field():
    store, snapshot = _issued_snapshot()
    validator = store.validate_tts_message_speech_snapshot

    event = TTSMessageSpeechRequestEvent(snapshot, validator)

    assert event.snapshot is snapshot
    assert event.validator is validator
    assert event.message_id == snapshot.message_id
    assert not hasattr(event, "text")
    assert not hasattr(event, "voice")


@pytest.mark.asyncio
async def test_bounded_rejection_happens_before_clock_service_or_cooldown(
    monkeypatch,
):
    store, snapshot = _issued_snapshot()
    store.update_message_content(snapshot.message_id, "changed")
    handler = _RecordingHandler()
    handler._tts_service = None
    handler._request_cooldown = {"old-message": -1000.0}
    handler._last_cooldown_cleanup = -1000.0
    cooldown_before = deepcopy(handler._request_cooldown)
    cleanup_before = handler._last_cooldown_cleanup

    def forbidden_clock():
        raise AssertionError("clock must not be read before snapshot admission")

    def forbidden_mutation(*_args, **_kwargs):
        raise AssertionError("cooldown must not mutate before snapshot admission")

    monkeypatch.setattr(tts_events_module.asyncio, "get_event_loop", forbidden_clock)
    monkeypatch.setattr(handler, "_cleanup_cooldown_dict", forbidden_mutation)
    monkeypatch.setattr(handler, "_enforce_cooldown_limit", forbidden_mutation)

    await handler.handle_tts_request(
        TTSMessageSpeechRequestEvent(
            snapshot,
            store.validate_tts_message_speech_snapshot,
        )
    )

    completions = [
        message for message in handler.messages if isinstance(message, TTSCompleteEvent)
    ]
    assert len(completions) == 1
    assert completions[0].message_id == snapshot.message_id
    assert (
        completions[0].error
        == "Message changed before speech started; select Speak again."
    )
    assert handler._request_cooldown == cooldown_before
    assert handler._last_cooldown_cleanup == cleanup_before
    assert handler.generated == []
    assert handler._active_tasks == set()


@pytest.mark.asyncio
async def test_unexpected_validator_failure_is_generic_and_privacy_safe(monkeypatch):
    _store, snapshot = _issued_snapshot()
    handler = _RecordingHandler()
    handler._tts_service = object()
    private_values = (
        snapshot.raw_content,
        "PRIVATE_AUTHORITY_COMPONENT",
        "PRIVATE_EXCEPTION_DETAIL",
    )
    log_messages: list[str] = []

    def fail_closed(_snapshot: TTSMessageSpeechSnapshot) -> str:
        raise RuntimeError(" ".join(private_values))

    monkeypatch.setattr(
        tts_events_module.asyncio,
        "get_event_loop",
        lambda: pytest.fail("clock read before validator rejection"),
    )
    sink_id = logger.add(log_messages.append, level="DEBUG", format="{message}")
    try:
        await handler.handle_tts_request(
            TTSMessageSpeechRequestEvent(snapshot, fail_closed)
        )
    finally:
        logger.remove(sink_id)

    completions = [
        message for message in handler.messages if isinstance(message, TTSCompleteEvent)
    ]
    assert len(completions) == 1
    assert (
        completions[0].error
        == "Message changed before speech started; select Speak again."
    )
    rendered = "\n".join(log_messages) + repr(completions)
    for private_value in private_values:
        assert private_value not in rendered
    assert handler._request_cooldown == {}
    assert handler.generated == []
    assert handler._active_tasks == set()


@pytest.mark.asyncio
async def test_valid_snapshot_uses_existing_normalization_and_global_voice_path():
    store, snapshot = _issued_snapshot()
    handler = _RecordingHandler()
    handler._tts_service = object()

    await handler.handle_tts_request(
        TTSMessageSpeechRequestEvent(
            snapshot,
            store.validate_tts_message_speech_snapshot,
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert handler.generated == [("Exact Console response.", snapshot.message_id, None)]
    assert snapshot.message_id in handler._request_cooldown


@pytest.mark.asyncio
async def test_explicit_global_request_path_remains_available():
    handler = _RecordingHandler()
    handler._tts_service = object()

    await handler.handle_tts_request(
        TTSRequestEvent(
            text="  Trusted   global speech. ",
            message_id="global-message",
            voice="alloy",
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert handler.generated == [("Trusted global speech.", "global-message", "alloy")]


def test_snapshot_event_rejects_non_snapshot_or_non_callable_validator():
    store, snapshot = _issued_snapshot()

    with pytest.raises(ValueError, match="snapshot"):
        TTSMessageSpeechRequestEvent(
            object(),  # type: ignore[arg-type]
            store.validate_tts_message_speech_snapshot,
        )
    with pytest.raises(ValueError, match="validator"):
        TTSMessageSpeechRequestEvent(
            snapshot,
            object(),  # type: ignore[arg-type]
        )


def test_bounded_snapshot_rejection_has_no_unbounded_exception_data():
    error = ConsoleSpeechSnapshotRejected(
        ConsoleSpeechSnapshotRejectionCode.MESSAGE_CHANGED
    )

    assert str(error) == "Message changed before speech started; select Speak again."
