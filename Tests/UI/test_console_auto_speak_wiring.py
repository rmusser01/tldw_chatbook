from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSMessageSpeechRequestEvent,
)
from tldw_chatbook.UI.Console_Modules.wiring import build_console_controllers
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import ConsoleTTSDestination
from tldw_chatbook.Widgets.Console.console_auto_speak_consent import (
    AutoSpeakConsentModal,
    ConsoleAutoSpeakCoordinator,
    sanitize_auto_speak_destination,
)

DEST_A = "sha256:" + "a" * 64
DEST_B = "sha256:" + "b" * 64


class AutoSpeakHarness:
    def __init__(self) -> None:
        self.store = ConsoleChatStore()
        self.session = self.store.create_session()
        self.greeting = self.store.append_message(
            self.session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Existing greeting.",
        )
        self.destination = ConsoleTTSDestination(
            fingerprint=DEST_A,
            provider_label="PocketChat TTS",
            sanitized_destination="http://127.0.0.1:8765",
            charges_may_apply=False,
        )
        self.hands_free = False
        self.issue_error = False
        self.open_error = False
        self.destination_gate: asyncio.Event | None = None
        self.destination_resolutions = 0
        self.schedule_failures = 0
        self.opened: list[
            tuple[AutoSpeakConsentModal, Callable[[bool], None]]
        ] = []
        self.spoken: list[str] = []
        self.expected_destinations: list[str | None] = []
        self.outcomes: list[Callable[[bool], None]] = []
        self.retry_requests: list[bool] = []
        self.synced: list[tuple[bool, bool, bool]] = []
        self.notices: list[tuple[str, str]] = []
        self.tasks: list[asyncio.Task[Any]] = []

        async def resolve_destination(
            _assistant_kind: str | None,
            _character_ref: object | None,
        ) -> ConsoleTTSDestination:
            self.destination_resolutions += 1
            if self.destination_gate is not None:
                await self.destination_gate.wait()
            return self.destination

        async def issue_speech(
            message_id: str,
            outcome_callback: Callable[[bool], None],
            expected_destination_fingerprint: str | None,
            retry_failed_auto: bool = False,
        ) -> bool:
            if self.issue_error:
                raise RuntimeError("bounded dispatch failure")
            self.spoken.append(message_id)
            self.expected_destinations.append(expected_destination_fingerprint)
            self.outcomes.append(outcome_callback)
            self.retry_requests.append(retry_failed_auto)
            return True

        def open_consent(
            modal: AutoSpeakConsentModal,
            callback: Callable[[bool], None],
        ) -> None:
            if self.open_error:
                raise RuntimeError("screen no longer accepts modals")
            self.opened.append((modal, callback))

        def schedule(coroutine: Coroutine[Any, Any, Any]) -> None:
            if self.schedule_failures:
                self.schedule_failures -= 1
                raise RuntimeError("screen scheduler rejected work")
            self.tasks.append(asyncio.create_task(coroutine))

        self.coordinator = ConsoleAutoSpeakCoordinator(
            store_accessor=lambda: self.store,
            resolve_destination=resolve_destination,
            issue_message_speech=issue_speech,
            open_consent=open_consent,
            hands_free_active=lambda: self.hands_free,
            sync_controls=lambda enabled, paused, retry_available: self.synced.append(
                (enabled, paused, retry_available)
            ),
            notify=lambda copy, severity: self.notices.append((copy, severity)),
            schedule=schedule,
        )
        self.coordinator.mount()

    async def drain(self) -> None:
        while self.tasks:
            tasks, self.tasks = self.tasks, []
            await asyncio.gather(*tasks)

    async def enable(self) -> None:
        self.coordinator.request_enabled(True)
        await self.drain()
        assert len(self.opened) == 1
        self.opened.pop()[1](True)
        await self.drain()

    def begin_reply(self, content: str = ""):
        return self.store.append_message(
            self.session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=content,
        )

    async def complete_reply(self, text: str):
        message = self.begin_reply()
        self.store.append_stream_chunk(message.id, text)
        self.store.mark_message_complete(message.id)
        await self.drain()
        return message


def test_destination_display_removes_credentials_path_query_and_controls() -> None:
    raw = "https://user:secret@example.com:8443/v1/audio/speech?key=abc#fragment"

    shown = sanitize_auto_speak_destination(raw)
    modal = AutoSpeakConsentModal(
        "Pocket\nChat\x00",
        raw,
        charges_may_apply=True,
    )

    assert shown == "https://example.com:8443"
    assert modal.sanitized_destination == shown
    assert "user" not in repr(modal)
    assert "secret" not in repr(modal)
    assert "key=abc" not in repr(modal)
    assert modal.provider_label == "Pocket Chat?"


def test_console_wiring_opens_auto_speak_consent_on_owning_app() -> None:
    screen = MagicMock()
    screen.app = MagicMock()
    screen.app_instance = MagicMock()
    del screen.push_screen
    build_console_controllers(
        screen,
        rag_source_types_accessor=lambda: (),
        rag_top_k_accessor=lambda: 10,
    )
    modal = AutoSpeakConsentModal(
        "PocketChat TTS",
        "http://127.0.0.1:8765/v1/audio/speech",
        charges_may_apply=False,
    )
    callback = MagicMock()

    screen._console_auto_speak._open_consent_fn(modal, callback)

    screen.app.push_screen.assert_called_once_with(modal, callback=callback)


@pytest.mark.asyncio
async def test_enabling_auto_speak_confirms_destination_without_replaying_greeting() -> None:
    harness = AutoSpeakHarness()

    harness.coordinator.request_enabled(True)
    await harness.drain()

    modal, accept = harness.opened.pop()
    assert modal.sanitized_destination == "http://127.0.0.1:8765"
    assert harness.spoken == []
    accept(True)
    await harness.drain()
    assert harness.session.speech_preferences.auto_speak is True
    assert harness.session.speech_preferences.consent_destination == DEST_A
    assert harness.spoken == []


@pytest.mark.asyncio
async def test_opted_out_completion_does_not_resolve_or_initialize_tts() -> None:
    harness = AutoSpeakHarness()

    await harness.complete_reply("Remain silent.")

    assert harness.destination_resolutions == 0
    assert harness.opened == []
    assert harness.spoken == []


@pytest.mark.asyncio
async def test_new_active_reply_dispatches_exactly_once() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()

    message = await harness.complete_reply("Welcome back.")
    harness.store._publish_message_completed(harness.session.id, message.id)
    await harness.drain()

    assert harness.spoken == [message.id]


@pytest.mark.asyncio
async def test_destination_change_requires_one_reconfirmation_and_drops_extra_reply() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    harness.destination = ConsoleTTSDestination(
        fingerprint=DEST_B,
        provider_label="PocketChat TTS",
        sanitized_destination="https://voice.example.test",
        charges_may_apply=True,
    )

    first = await harness.complete_reply("This must wait.")
    second = await harness.complete_reply("Do not queue this.")

    assert harness.spoken == []
    assert len(harness.opened) == 1
    modal, accept = harness.opened.pop()
    assert modal.sanitized_destination == "https://voice.example.test"
    accept(True)
    accept(True)
    await harness.drain()

    assert harness.session.speech_preferences.consent_destination == DEST_B
    assert harness.spoken == [first.id]
    assert second.id not in harness.spoken


@pytest.mark.asyncio
async def test_background_completion_never_prompts_or_plays() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    background = harness.session
    message = harness.begin_reply()
    harness.store.create_session(title="Foreground")

    harness.store.append_stream_chunk(message.id, "Background reply.")
    harness.store.mark_message_complete(message.id)
    await harness.drain()

    assert harness.store.active_session_id != background.id
    assert harness.opened == []
    assert harness.spoken == []


@pytest.mark.asyncio
async def test_hands_free_owns_new_reply_without_prompt_or_play() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    harness.hands_free = True

    await harness.complete_reply("Hands-free owns this.")

    assert harness.opened == []
    assert harness.spoken == []


@pytest.mark.asyncio
async def test_dismissal_and_stale_completion_drop_retained_token() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    harness.destination = ConsoleTTSDestination(
        fingerprint=DEST_B,
        provider_label="PocketChat TTS",
        sanitized_destination="https://voice.example.test",
        charges_may_apply=True,
    )

    message = await harness.complete_reply("Changed destination.")
    _modal, dismiss = harness.opened.pop()
    dismiss(False)
    await harness.drain()
    assert harness.spoken == []

    harness.store.begin_variant_stream(message.id)
    harness.store.append_stream_chunk(message.id, "Changed destination again.")
    harness.store.finalize_variant_stream(message.id)
    await harness.drain()
    _modal, accept = harness.opened.pop()
    harness.store.begin_variant_stream(message.id)
    accept(True)
    await harness.drain()

    assert harness.spoken == []


@pytest.mark.asyncio
async def test_persistence_failure_keeps_switch_truthful(monkeypatch) -> None:
    harness = AutoSpeakHarness()
    original = harness.store.set_auto_speak

    def fail_enable(session_id: str, enabled: bool):
        if enabled:
            return harness.session, False
        return original(session_id, enabled)

    monkeypatch.setattr(harness.store, "set_auto_speak", fail_enable)
    harness.coordinator.request_enabled(True)
    await harness.drain()
    harness.opened.pop()[1](True)
    await harness.drain()

    assert harness.session.speech_preferences.auto_speak is False
    assert harness.synced[-1] == (False, False, False)
    assert harness.notices[-1][1] == "error"


@pytest.mark.asyncio
async def test_disable_persistence_failure_keeps_switch_enabled(monkeypatch) -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    original = harness.store.set_auto_speak

    def fail_disable(session_id: str, enabled: bool):
        if not enabled:
            return harness.session, False
        return original(session_id, enabled)

    monkeypatch.setattr(harness.store, "set_auto_speak", fail_disable)
    harness.coordinator.request_enabled(False)
    await harness.drain()

    assert harness.session.speech_preferences.auto_speak is True
    assert harness.synced[-1] == (True, False, False)
    assert harness.notices[-1][1] == "error"


@pytest.mark.asyncio
async def test_dispatch_exception_persists_paused_state() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    harness.issue_error = True

    await harness.complete_reply("Fail before admission.")

    assert harness.session.speech_preferences.paused is True
    assert harness.notices[-1][1] == "warning"


@pytest.mark.asyncio
async def test_tts_failure_persists_pause_and_resume_clears_it() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    await harness.complete_reply("Try me.")

    harness.outcomes.pop()(False)
    await harness.drain()

    assert harness.session.speech_preferences.paused is True
    assert "Retry speech" in harness.notices[-1][0]
    harness.coordinator.request_resume()
    await harness.drain()
    assert harness.session.speech_preferences.paused is False


@pytest.mark.asyncio
async def test_unmount_unsubscribes_and_stale_callbacks_are_noops() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    harness.coordinator.unmount()

    await harness.complete_reply("No longer mounted.")
    harness.coordinator.request_enabled(False)
    await harness.drain()

    assert harness.spoken == []
    assert harness.session.speech_preferences.auto_speak is True


@pytest.mark.asyncio
async def test_concurrent_enable_requests_share_one_destination_lookup_and_modal() -> None:
    harness = AutoSpeakHarness()
    harness.destination_gate = asyncio.Event()

    harness.coordinator.request_enabled(True)
    harness.coordinator.request_enabled(True)
    await asyncio.sleep(0)

    assert harness.destination_resolutions == 1
    harness.destination_gate.set()
    await harness.drain()
    assert len(harness.opened) == 1


@pytest.mark.asyncio
async def test_modal_open_failure_releases_enable_reservation_and_keeps_state_truthful() -> None:
    harness = AutoSpeakHarness()
    harness.open_error = True

    harness.coordinator.request_enabled(True)
    await harness.drain()

    assert harness.session.speech_preferences.auto_speak is False
    assert harness.synced[-1] == (False, False, False)
    harness.open_error = False
    harness.coordinator.request_enabled(True)
    await harness.drain()
    assert len(harness.opened) == 1


@pytest.mark.asyncio
async def test_unmount_while_destination_lookup_is_blocked_never_prompts_or_dispatches() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    harness.destination = ConsoleTTSDestination(
        fingerprint=DEST_B,
        provider_label="PocketChat TTS",
        sanitized_destination="https://voice.example.test",
        charges_may_apply=True,
    )
    harness.destination_gate = asyncio.Event()
    message = harness.begin_reply()
    harness.store.append_stream_chunk(message.id, "Wait for destination.")
    harness.store.mark_message_complete(message.id)
    await asyncio.sleep(0)

    harness.coordinator.unmount()
    harness.destination_gate.set()
    await harness.drain()

    assert harness.opened == []
    assert harness.spoken == []


@pytest.mark.asyncio
async def test_delayed_completion_drops_after_active_session_a_b_a_cycle() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    harness.destination_gate = asyncio.Event()
    message = harness.begin_reply()
    harness.store.append_stream_chunk(message.id, "Delayed A reply.")
    harness.store.mark_message_complete(message.id)
    await asyncio.sleep(0)

    other = harness.store.create_session(title="B")
    harness.store.switch_session(harness.session.id)
    assert other.id != harness.store.active_session_id
    harness.destination_gate.set()
    await harness.drain()

    assert harness.spoken == []


@pytest.mark.asyncio
async def test_enable_modal_acceptance_drops_after_active_session_a_b_a_cycle() -> None:
    harness = AutoSpeakHarness()
    harness.coordinator.request_enabled(True)
    await harness.drain()
    _modal, accept = harness.opened.pop()

    harness.store.create_session(title="B")
    harness.store.switch_session(harness.session.id)
    accept(True)
    await harness.drain()

    assert harness.session.speech_preferences.auto_speak is False


@pytest.mark.asyncio
async def test_failure_after_unmount_still_persists_pause_for_same_opt_in_epoch() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("Speech starts before unmount.")
    outcome = harness.outcomes.pop()

    harness.coordinator.unmount()
    outcome(False)

    assert harness.session.speech_preferences.paused is True
    assert harness.coordinator.failed_message_ids[harness.session.id] == message.id


@pytest.mark.asyncio
async def test_old_failure_after_disable_reenable_does_not_pause_new_opt_in() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    await harness.complete_reply("Old request.")
    old_outcome = harness.outcomes.pop()
    harness.store.set_auto_speak(harness.session.id, False)
    harness.store.set_auto_speak(harness.session.id, True)

    old_outcome(False)

    assert harness.session.speech_preferences.auto_speak is True
    assert harness.session.speech_preferences.paused is False


@pytest.mark.asyncio
async def test_retry_uses_trusted_path_for_failed_message_without_resuming() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("Retry this exact reply.")
    harness.outcomes.pop()(False)
    assert harness.session.speech_preferences.paused is True

    harness.coordinator.request_retry()
    await harness.drain()

    assert harness.spoken == [message.id, message.id]
    assert harness.expected_destinations == [DEST_A, DEST_A]
    assert harness.retry_requests == [False, True]
    assert harness.session.speech_preferences.paused is True


@pytest.mark.asyncio
async def test_retry_failure_after_resume_does_not_notify_stale_paused_state() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("Resume before retry settles.")
    harness.outcomes.pop()(False)

    harness.coordinator.request_retry()
    await harness.drain()
    retry_outcome = harness.outcomes.pop()
    notice_count = len(harness.notices)

    harness.coordinator.request_resume()
    await harness.drain()
    retry_outcome(False)

    assert harness.session.speech_preferences.paused is False
    assert message.id not in harness.coordinator.failed_message_ids.values()
    assert len(harness.notices) == notice_count


@pytest.mark.asyncio
async def test_retry_drops_when_auto_speak_resumes_during_destination_lookup() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("Do not retry after resume.")
    harness.outcomes.pop()(False)
    harness.destination_gate = asyncio.Event()

    harness.coordinator.request_retry()
    await asyncio.sleep(0)
    harness.coordinator.request_resume()
    await asyncio.sleep(0)
    assert harness.session.speech_preferences.paused is False

    harness.destination_gate.set()
    await harness.drain()

    assert harness.spoken == [message.id]


@pytest.mark.asyncio
async def test_retry_drops_when_hands_free_starts_during_destination_lookup() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("Hands-free owns this reply.")
    harness.outcomes.pop()(False)
    harness.destination_gate = asyncio.Event()

    harness.coordinator.request_retry()
    await asyncio.sleep(0)
    harness.hands_free = True
    harness.destination_gate.set()
    await harness.drain()

    assert harness.spoken == [message.id]


@pytest.mark.asyncio
async def test_retry_failed_message_from_wrong_active_session_fails_closed() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    await harness.complete_reply("Do not cross sessions.")
    harness.outcomes.pop()(False)
    harness.store.create_session(title="Other")

    harness.coordinator.request_retry()
    await harness.drain()

    assert len(harness.spoken) == 1


@pytest.mark.asyncio
async def test_later_successful_completion_for_same_message_dispatches_again() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("First answer.")
    harness.outcomes.pop()(True)
    harness.store.begin_variant_stream(message.id)
    harness.store.append_stream_chunk(message.id, "Regenerated answer.")
    harness.store.finalize_variant_stream(message.id)
    await harness.drain()

    assert harness.spoken == [message.id, message.id]
    assert harness.expected_destinations == [DEST_A, DEST_A]


@pytest.mark.asyncio
async def test_regeneration_waits_for_prior_speech_then_dispatches_once() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("First answer.")
    first_outcome = harness.outcomes.pop()
    harness.store.begin_variant_stream(message.id)
    harness.store.append_stream_chunk(message.id, "Second answer.")
    harness.store.finalize_variant_stream(message.id)
    await harness.drain()

    assert harness.spoken == [message.id]
    first_outcome(True)
    await harness.drain()
    assert harness.spoken == [message.id, message.id]


@pytest.mark.asyncio
async def test_pending_regeneration_reconfirms_changed_destination_before_speech() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("First destination.")
    first_outcome = harness.outcomes.pop()
    harness.store.begin_variant_stream(message.id)
    harness.store.append_stream_chunk(message.id, "Second destination.")
    harness.store.finalize_variant_stream(message.id)
    await harness.drain()
    assert harness.spoken == [message.id]
    assert harness.opened == []

    harness.destination = ConsoleTTSDestination(
        fingerprint=DEST_B,
        provider_label="PocketChat TTS",
        sanitized_destination="https://voice-b.example.test",
        charges_may_apply=False,
    )

    first_outcome(True)
    await harness.drain()

    assert harness.spoken == [message.id]
    assert len(harness.opened) == 1
    harness.opened.pop()[1](True)
    await harness.drain()

    assert harness.spoken == [message.id, message.id]
    assert harness.expected_destinations == [DEST_A, DEST_B]


@pytest.mark.asyncio
async def test_regeneration_pending_behind_failed_speech_stays_paused() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("First answer.")
    first_outcome = harness.outcomes.pop()
    harness.store.begin_variant_stream(message.id)
    harness.store.append_stream_chunk(message.id, "Second answer.")
    harness.store.finalize_variant_stream(message.id)
    await harness.drain()

    first_outcome(False)
    await harness.drain()
    assert harness.spoken == [message.id]
    assert harness.session.speech_preferences.paused is True


@pytest.mark.asyncio
async def test_regeneration_destination_await_rechecks_failure_pause_before_dispatch(
) -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("First answer.")
    first_outcome = harness.outcomes.pop()
    harness.destination_gate = asyncio.Event()

    harness.store.begin_variant_stream(message.id)
    harness.store.append_stream_chunk(message.id, "Second answer.")
    harness.store.finalize_variant_stream(message.id)
    await asyncio.sleep(0)
    assert harness.destination_resolutions >= 3

    first_outcome(False)
    assert harness.session.speech_preferences.paused is True
    harness.destination_gate.set()
    await harness.drain()

    assert harness.spoken == [message.id]


@pytest.mark.asyncio
async def test_same_id_restore_accepts_new_completion_and_drops_old_generation(
) -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("Original answer.")
    harness.outcomes.pop()(True)
    restored_session = replace(harness.session)
    restored_messages = harness.store.messages_for_session(harness.session.id)
    harness.destination_gate = asyncio.Event()

    harness.store.begin_variant_stream(message.id)
    harness.store.append_stream_chunk(message.id, "Pre-restore answer.")
    harness.store.finalize_variant_stream(message.id)
    await asyncio.sleep(0)
    harness.store.restore_state(
        sessions=[restored_session],
        messages_by_session={restored_session.id: restored_messages},
        active_session_id=restored_session.id,
    )
    harness.destination_gate.set()
    await harness.drain()

    harness.store.begin_variant_stream(message.id)
    harness.store.append_stream_chunk(message.id, "Post-restore answer.")
    harness.store.finalize_variant_stream(message.id)
    await harness.drain()

    assert harness.spoken == [message.id, message.id]


@pytest.mark.asyncio
async def test_regeneration_pending_behind_speech_drops_after_active_change() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("First answer.")
    first_outcome = harness.outcomes.pop()
    harness.store.begin_variant_stream(message.id)
    harness.store.append_stream_chunk(message.id, "Second answer.")
    harness.store.finalize_variant_stream(message.id)
    await harness.drain()

    harness.store.create_session(title="B")
    first_outcome(True)
    await harness.drain()
    assert harness.spoken == [message.id]


@pytest.mark.asyncio
async def test_duplicate_completion_callback_after_settlement_does_not_replay() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("One completed answer.")
    harness.outcomes.pop()(True)

    harness.store._publish_message_completed(harness.session.id, message.id)
    await harness.drain()

    assert harness.spoken == [message.id]


@pytest.mark.asyncio
async def test_distinct_completions_during_destination_lookup_both_dispatch() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    harness.destination_gate = asyncio.Event()
    first = harness.begin_reply()
    second = harness.begin_reply()
    for message, text in ((first, "First."), (second, "Second.")):
        harness.store.append_stream_chunk(message.id, text)
        harness.store.mark_message_complete(message.id)
    await asyncio.sleep(0)

    harness.destination_gate.set()
    await harness.drain()

    assert harness.spoken == [first.id, second.id]


@pytest.mark.asyncio
async def test_unavailable_app_handler_callback_durably_pauses_auto_speak() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("Handler is unavailable.")
    outcome = harness.outcomes.pop()
    event = TTSMessageSpeechRequestEvent(
        harness.store.issue_tts_message_speech_snapshot(message.id),
        harness.store.validate_tts_message_speech_snapshot,
        outcome_callback=outcome,
        expected_destination_fingerprint=DEST_A,
    )
    app = MagicMock()
    app.loguru_logger = MagicMock()
    app._ensure_tts_handler = AsyncMock(return_value=None)
    app.post_message = MagicMock(return_value=True)

    await TldwCli.handle_tts_message_speech_request_event(app, event)

    assert harness.session.speech_preferences.paused is True
    assert harness.coordinator.failed_message_ids[harness.session.id] == message.id


@pytest.mark.asyncio
async def test_retry_deleted_failed_message_fails_closed() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    message = await harness.complete_reply("Delete before retry.")
    harness.outcomes.pop()(False)
    harness.store.delete_message(message.id)

    harness.coordinator.request_retry()
    await harness.drain()

    assert harness.spoken == [message.id]


@pytest.mark.asyncio
async def test_retry_stale_failed_message_id_fails_closed() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    await harness.complete_reply("Stale token.")
    harness.outcomes.pop()(False)
    harness.coordinator.failed_message_ids[harness.session.id] = "missing-message"

    harness.coordinator.request_retry()
    await harness.drain()

    assert len(harness.spoken) == 1


@pytest.mark.asyncio
async def test_retry_ownership_is_independent_per_session_and_resume_is_scoped() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    first = await harness.complete_reply("A failed reply.")
    harness.outcomes.pop()(False)

    second_session = harness.store.create_session(title="B")
    harness.store.confirm_auto_speak_destination(second_session.id, DEST_A)
    harness.store.set_auto_speak(second_session.id, True)
    second = harness.store.append_message(
        second_session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    harness.store.append_stream_chunk(second.id, "B failed reply.")
    harness.store.mark_message_complete(second.id)
    await harness.drain()
    harness.outcomes.pop()(False)

    harness.store.switch_session(harness.session.id)
    harness.coordinator.sync_controls()
    harness.coordinator.request_retry()
    await harness.drain()
    harness.store.switch_session(second_session.id)
    harness.coordinator.request_resume()
    await harness.drain()

    assert harness.spoken == [first.id, second.id, first.id]
    assert harness.session.speech_preferences.paused is True
    assert second_session.speech_preferences.paused is False
    assert harness.coordinator.failed_message_ids[harness.session.id] == first.id
    assert second_session.id not in harness.coordinator.failed_message_ids


@pytest.mark.asyncio
async def test_old_failure_cannot_pause_same_id_restored_session() -> None:
    harness = AutoSpeakHarness()
    await harness.enable()
    await harness.complete_reply("Old state request.")
    old_outcome = harness.outcomes.pop()
    restored = replace(harness.session)
    restored_messages = harness.store.messages_for_session(harness.session.id)

    harness.store.restore_state(
        sessions=[restored],
        messages_by_session={restored.id: restored_messages},
        active_session_id=restored.id,
    )
    old_outcome(False)

    active = harness.store.sessions()[0]
    assert active.speech_preferences.auto_speak is True
    assert active.speech_preferences.paused is False
    assert restored.id not in harness.coordinator.failed_message_ids


@pytest.mark.asyncio
async def test_scheduler_rejection_closes_work_and_allows_retry() -> None:
    harness = AutoSpeakHarness()
    harness.schedule_failures = 1

    harness.coordinator.request_enabled(True)
    await harness.drain()
    assert harness.opened == []
    assert harness.session.speech_preferences.auto_speak is False

    harness.coordinator.request_enabled(True)
    await harness.drain()
    assert len(harness.opened) == 1


@pytest.mark.asyncio
async def test_modal_callback_scheduler_rejection_unwinds_modal() -> None:
    harness = AutoSpeakHarness()
    harness.coordinator.request_enabled(True)
    await harness.drain()
    _modal, accept = harness.opened.pop()
    harness.schedule_failures = 1

    accept(True)
    await harness.drain()
    harness.coordinator.request_enabled(True)
    await harness.drain()

    assert harness.session.speech_preferences.auto_speak is False
    assert len(harness.opened) == 1
