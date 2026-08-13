from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Widgets.Console.console_auto_speak_consent import (
    AutoSpeakConsentModal,
    ConsoleAutoSpeakCoordinator,
    ConsoleTTSDestination,
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
        self.destination_resolutions = 0
        self.opened: list[
            tuple[AutoSpeakConsentModal, Callable[[bool], None]]
        ] = []
        self.spoken: list[str] = []
        self.outcomes: list[Callable[[bool], None]] = []
        self.synced: list[tuple[bool, bool]] = []
        self.notices: list[tuple[str, str]] = []
        self.tasks: list[asyncio.Task[Any]] = []

        async def resolve_destination(
            _assistant_kind: str | None,
            _character_ref: object | None,
        ) -> ConsoleTTSDestination:
            self.destination_resolutions += 1
            return self.destination

        async def issue_speech(
            message_id: str,
            outcome_callback: Callable[[bool], None],
        ) -> bool:
            if self.issue_error:
                raise RuntimeError("bounded dispatch failure")
            self.spoken.append(message_id)
            self.outcomes.append(outcome_callback)
            return True

        def schedule(coroutine: Coroutine[Any, Any, Any]) -> None:
            self.tasks.append(asyncio.create_task(coroutine))

        self.coordinator = ConsoleAutoSpeakCoordinator(
            store_accessor=lambda: self.store,
            resolve_destination=resolve_destination,
            issue_message_speech=issue_speech,
            open_consent=lambda modal, callback: self.opened.append(
                (modal, callback)
            ),
            hands_free_active=lambda: self.hands_free,
            sync_controls=lambda enabled, paused: self.synced.append(
                (enabled, paused)
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

    harness.store._message_completion_emitted_ids.discard(message.id)
    harness.store._publish_message_completed(harness.session.id, message.id)
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
    assert harness.synced[-1] == (False, False)
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
    assert harness.synced[-1] == (True, False)
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
