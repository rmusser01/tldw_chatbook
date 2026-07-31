"""Pre-cooldown admission tests for trusted Console speech snapshots."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import cast

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
    TTSGlobalOverrideDecisionEvent,
    TTSMessageSpeechRequestEvent,
    TTSRequestEvent,
)
from tldw_chatbook.TTS.character_request_resolver import (
    CharacterTTSRequestResolution,
)
from tldw_chatbook.TTS.profile_service import LoadedCharacterTTSAssignment
from tldw_chatbook.TTS.profile_types import CharacterRef


class _RecordingHandler(TTSEventHandler):
    def __init__(self, profile_service_loader=None) -> None:
        super().__init__(profile_service_loader=profile_service_loader)
        self.messages: list[object] = []
        self.generated: list[tuple[str, str | None, str | None]] = []
        self.resolutions: list[CharacterTTSRequestResolution | None] = []
        self._request_cooldown = {}

    async def post_message(self, message: object) -> None:
        self.messages.append(message)

    async def _generate_tts_with_rate_limit(
        self,
        text: str,
        message_id: str | None,
        voice: str | None,
        resolution: CharacterTTSRequestResolution | None = None,
    ) -> None:
        self.generated.append((text, message_id, voice))
        self.resolutions.append(resolution)


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


def _issued_character_snapshot() -> tuple[
    ConsoleChatStore,
    TTSMessageSpeechSnapshot,
]:
    store = ConsoleChatStore()
    session = store.create_session(
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="local-authority",
        character_id=7,
    )
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="  Character   response.\n",
    )
    return store, store.issue_tts_message_speech_snapshot(message.id)


class _UnassignedProfileService:
    def __init__(self) -> None:
        self.calls: list[CharacterRef] = []

    async def get_assigned_profile(
        self,
        character_ref: CharacterRef,
    ) -> LoadedCharacterTTSAssignment:
        self.calls.append(character_ref)
        return LoadedCharacterTTSAssignment(
            repository_generation=4,
            snapshot=None,
        )


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
    assert handler.resolutions[0] is not None
    assert handler.resolutions[0].source == "global"
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
    assert handler.resolutions == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("message_id", (None, ""))
async def test_explicit_global_request_uses_one_fallback_message_id(
    message_id: str | None,
) -> None:
    rejected_handler = _RecordingHandler()
    rejected_handler._tts_service = object()

    await rejected_handler.handle_tts_request(
        TTSRequestEvent(text="", message_id=message_id)
    )

    completions = [
        message
        for message in rejected_handler.messages
        if isinstance(message, TTSCompleteEvent)
    ]
    assert len(completions) == 1
    assert completions[0].message_id == "adhoc"

    admitted_handler = _RecordingHandler()
    admitted_handler._tts_service = object()
    await admitted_handler.handle_tts_request(
        TTSRequestEvent(text="Speak this.", message_id=message_id)
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert admitted_handler.generated == [("Speak this.", "adhoc", None)]
    assert set(admitted_handler._request_cooldown) == {"adhoc"}


@pytest.mark.asyncio
async def test_unassigned_character_reads_once_then_uses_global_resolution() -> None:
    store, snapshot = _issued_character_snapshot()
    profile_service = _UnassignedProfileService()

    async def load_profile_service() -> _UnassignedProfileService:
        return profile_service

    handler = _RecordingHandler(load_profile_service)
    handler._tts_service = object()

    await handler.handle_tts_request(
        TTSMessageSpeechRequestEvent(
            snapshot,
            store.validate_tts_message_speech_snapshot,
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert profile_service.calls == [cast(CharacterRef, snapshot.character_ref)]
    assert handler.generated == [("Character response.", snapshot.message_id, None)]
    assert handler.resolutions[0] is not None
    assert handler.resolutions[0].source == "global"


@pytest.mark.asyncio
async def test_resolution_failure_offers_single_use_override_without_cooldown() -> None:
    store, snapshot = _issued_character_snapshot()

    async def unavailable_profile_service():
        return None

    handler = _RecordingHandler(unavailable_profile_service)
    handler._tts_service = object()

    await handler.handle_tts_request(
        TTSMessageSpeechRequestEvent(
            snapshot,
            store.validate_tts_message_speech_snapshot,
        )
    )

    completion = next(
        message for message in handler.messages if isinstance(message, TTSCompleteEvent)
    )
    assert completion.error
    assert completion.global_override_token is not None
    token = completion.global_override_token
    assert handler.generated == []
    assert handler._request_cooldown == {}

    await handler.handle_tts_global_override_decision(
        TTSGlobalOverrideDecisionEvent(token, accepted=True)
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert handler.generated == [("Character response.", snapshot.message_id, None)]
    assert handler.resolutions[0] is not None
    assert handler.resolutions[0].source == "explicit_override"
    assert snapshot.message_id in handler._request_cooldown

    await handler.handle_tts_global_override_decision(
        TTSGlobalOverrideDecisionEvent(token, accepted=True)
    )
    await handler.handle_tts_global_override_decision(
        TTSGlobalOverrideDecisionEvent("0" * 32, accepted=True)
    )
    await asyncio.sleep(0)

    assert len(handler.generated) == 1


@pytest.mark.asyncio
async def test_override_revalidates_snapshot_and_decline_performs_no_work() -> None:
    store, snapshot = _issued_character_snapshot()

    async def unavailable_profile_service():
        return None

    stale_handler = _RecordingHandler(unavailable_profile_service)
    stale_handler._tts_service = object()
    await stale_handler.handle_tts_request(
        TTSMessageSpeechRequestEvent(
            snapshot,
            store.validate_tts_message_speech_snapshot,
        )
    )
    stale_completion = next(
        message
        for message in stale_handler.messages
        if isinstance(message, TTSCompleteEvent)
    )
    assert stale_completion.global_override_token is not None
    store.update_message_content(snapshot.message_id, "changed")

    await stale_handler.handle_tts_global_override_decision(
        TTSGlobalOverrideDecisionEvent(
            stale_completion.global_override_token,
            accepted=True,
        )
    )

    assert stale_handler.generated == []
    assert stale_handler._request_cooldown == {}
    assert any(
        isinstance(message, TTSCompleteEvent)
        and message.error
        == "Message changed before speech started; select Speak again."
        for message in stale_handler.messages[1:]
    )

    fresh_store, fresh_snapshot = _issued_character_snapshot()
    decline_handler = _RecordingHandler(unavailable_profile_service)
    decline_handler._tts_service = object()
    await decline_handler.handle_tts_request(
        TTSMessageSpeechRequestEvent(
            fresh_snapshot,
            fresh_store.validate_tts_message_speech_snapshot,
        )
    )
    decline_completion = next(
        message
        for message in decline_handler.messages
        if isinstance(message, TTSCompleteEvent)
    )
    assert decline_completion.global_override_token is not None

    await decline_handler.handle_tts_global_override_decision(
        TTSGlobalOverrideDecisionEvent(
            decline_completion.global_override_token,
            accepted=False,
        )
    )

    assert decline_handler.generated == []
    assert decline_handler._request_cooldown == {}


@pytest.mark.asyncio
async def test_unknown_and_expired_override_tokens_do_no_admission_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, snapshot = _issued_character_snapshot()
    validator_calls = 0
    loader_calls = 0

    def validator(candidate: TTSMessageSpeechSnapshot) -> str:
        nonlocal validator_calls
        validator_calls += 1
        return store.validate_tts_message_speech_snapshot(candidate)

    async def unavailable_profile_service():
        nonlocal loader_calls
        loader_calls += 1
        return None

    handler = _RecordingHandler(unavailable_profile_service)
    handler._tts_service = object()
    await handler.handle_tts_request(TTSMessageSpeechRequestEvent(snapshot, validator))
    completion = next(
        message for message in handler.messages if isinstance(message, TTSCompleteEvent)
    )
    assert completion.global_override_token is not None
    pending = handler._pending_global_overrides[completion.global_override_token]

    with monkeypatch.context() as context:
        context.setattr(
            tts_events_module.asyncio,
            "get_event_loop",
            lambda: pytest.fail("unknown token must not read the clock"),
        )
        await handler.handle_tts_global_override_decision(
            TTSGlobalOverrideDecisionEvent("0" * 32, accepted=True)
        )

    with monkeypatch.context() as context:
        context.setattr(
            tts_events_module.asyncio,
            "get_event_loop",
            lambda: SimpleNamespace(
                time=lambda: (
                    pending.created_at + handler.GLOBAL_OVERRIDE_TTL_SECONDS + 1.0
                )
            ),
        )
        await handler.handle_tts_global_override_decision(
            TTSGlobalOverrideDecisionEvent(
                completion.global_override_token,
                accepted=True,
            )
        )

    assert validator_calls == 1
    assert loader_calls == 1
    assert handler.generated == []
    assert handler._request_cooldown == {}
    assert handler._pending_global_overrides == {}


@pytest.mark.asyncio
async def test_override_capabilities_are_bounded_and_cleared_on_cleanup() -> None:
    store, snapshot = _issued_character_snapshot()
    handler = _RecordingHandler()

    tokens = [
        handler._issue_global_override(
            snapshot,
            store.validate_tts_message_speech_snapshot,
        )
        for _ in range(handler.MAX_PENDING_GLOBAL_OVERRIDES + 1)
    ]

    assert len(handler._pending_global_overrides) == (
        handler.MAX_PENDING_GLOBAL_OVERRIDES
    )
    assert tokens[0] not in handler._pending_global_overrides
    assert tokens[-1] in handler._pending_global_overrides

    await handler.cleanup_tts_resources()

    assert handler._pending_global_overrides == {}


def test_override_decision_rejects_malformed_public_values() -> None:
    with pytest.raises(ValueError, match="token"):
        TTSGlobalOverrideDecisionEvent("not-a-token", accepted=True)
    with pytest.raises(ValueError, match="accepted"):
        TTSGlobalOverrideDecisionEvent("0" * 32, accepted=1)  # type: ignore[arg-type]


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
