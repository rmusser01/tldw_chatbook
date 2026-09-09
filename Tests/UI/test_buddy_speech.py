"""App-owned named speech follows trusted sources without activating Console."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost
from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding
from tldw_chatbook.UI.Navigation.buddy_speech import BuddySpeechCoordinator

DESTINATION = "sha256:" + "b" * 64


class Handler:
    def __init__(self):
        self.spoken = []
        self.destination = SimpleNamespace(
            fingerprint=DESTINATION,
            provider_label="Configured TTS",
            sanitized_destination="https://speech.example",
            charges_may_apply=True,
        )
        self.resolving = None

    async def resolve_console_speech_destination(self, *_args):
        if self.resolving is not None:
            await self.resolving.wait()
        return self.destination

    async def speak_guarded_utterance(self, text, **kwargs):
        assert kwargs["validator"]()
        assert kwargs["expected_destination_fingerprint"] == DESTINATION
        self.spoken.append(text)
        return True


def setup():
    store = ConsoleChatStore()
    session = store.create_session(title="Research", ephemeral=True)
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Owned response"
    )
    other = store.create_session(ephemeral=True)
    handler = Handler()
    host = InterruptRoundHost(SimpleNamespace(store=store))
    app = SimpleNamespace(
        app_config={},
        console_runtime=SimpleNamespace(
            chat_store=store,
            chat_controller=SimpleNamespace(_interrupt_host=host),
            alive=True,
        ),
    )

    async def ensure():
        return handler

    app._ensure_tts_handler = ensure
    return app, store, session, other, message, handler


@pytest.mark.asyncio
async def test_hidden_named_reply_uses_consent_and_does_not_select_or_acknowledge():
    app, store, owner, other, _, handler = setup()
    store.confirm_auto_speak_destination(owner.id, DESTINATION)
    speech = BuddySpeechCoordinator(app)
    speech.configure(BuddyBinding.for_session(owner), True)
    try:
        await speech.refresh()
        await speech.queue.wait_idle()
        await speech.refresh()
        await speech.queue.wait_idle()
        assert handler.spoken == ["Research. Owned response"]
        assert store.active_session_id == other.id
        assert not owner.speech_preferences.auto_speak
    finally:
        await speech.aclose()


@pytest.mark.asyncio
async def test_no_destination_consent_means_no_synthesis_or_background_modal():
    app, _store, owner, _, _, handler = setup()
    app.push_screen = lambda *_args: pytest.fail(
        "Background speech must not steal focus"
    )
    speech = BuddySpeechCoordinator(app)
    speech.configure(BuddyBinding.for_session(owner), True)
    try:
        await speech.refresh()
        await speech.queue.wait_idle()
        assert handler.spoken == []
        assert speech.needs_consent
        assert owner.speech_preferences.consent_destination is None
    finally:
        await speech.aclose()


@pytest.mark.asyncio
async def test_explicit_consent_revalidates_changed_destination_before_speech():
    app, _, owner, _, _, handler = setup()
    callbacks = []
    app.push_screen = lambda screen, callback: callbacks.append(callback)
    speech = BuddySpeechCoordinator(app)
    speech.configure(BuddyBinding.for_session(owner), True)
    try:
        await speech.refresh()
        await speech.queue.wait_idle()
        speech.request_consent()
        await asyncio.sleep(0)
        handler.destination = SimpleNamespace(fingerprint="sha256:" + "c" * 64)
        callbacks[0](True)
        await speech._consent_task
        assert handler.spoken == []
        assert "changed" in speech.notice
    finally:
        await speech.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["assistant", "close", "profile", "binding", "content"]
)
async def test_owner_revalidated_after_destination_await(change):
    app, store, owner, other, message, handler = setup()
    store.confirm_auto_speak_destination(owner.id, DESTINATION)
    handler.resolving = asyncio.Event()
    speech = BuddySpeechCoordinator(app)
    speech.configure(BuddyBinding.for_session(owner), True)
    try:
        await speech.refresh()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if change == "assistant":
            owner.assistant_id = "different"
        elif change == "close":
            store.close_session(owner.id)
        elif change == "profile":
            app.console_runtime = None
        elif change == "content":
            store.update_message_content(message.id, "Edited while preparing speech")
        else:
            speech.configure(BuddyBinding.for_session(other), True)
        handler.resolving.set()
        await speech.queue.wait_idle()
        assert handler.spoken == []
    finally:
        await speech.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["version", "is_active", "deleted"])
async def test_persona_revision_and_retirement_revoke_queued_speech(change):
    app, store, owner, _, _, handler = setup()
    record = {"id": "persona", "version": 1, "is_active": True}
    owner.assistant_kind, owner.assistant_id = "persona", "persona"
    app.local_character_persona_service = SimpleNamespace(
        get_persona_profile=lambda _id: dict(record)
    )
    store.confirm_auto_speak_destination(owner.id, DESTINATION)
    handler.resolving = asyncio.Event()
    speech = BuddySpeechCoordinator(app)
    speech.configure(BuddyBinding.for_session(owner), True)
    try:
        await speech.refresh()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        record[change] = {"version": 2, "is_active": False, "deleted": True}[change]
        handler.resolving.set()
        await speech.queue.wait_idle()
        assert handler.spoken == []
    finally:
        await speech.aclose()


@pytest.mark.asyncio
async def test_speech_controls_only_change_presentation():
    from textual.app import App
    from textual.widgets import Button

    from tldw_chatbook.Widgets.Persona_Widgets.buddy_speech_controls import (
        BuddySpeechControls,
    )

    app, _, owner, _, _, _ = setup()
    speech = BuddySpeechCoordinator(app)
    speech.enabled = True
    ui = App()
    async with ui.run_test(size=(80, 24)) as pilot:
        await ui.screen.mount(BuddySpeechControls(speech))
        ui.query_one("#buddy-speech-pause", Button).press()
        await pilot.pause()
        assert speech.queue.state.paused
        ui.query_one("#buddy-speech-mute", Button).press()
        await pilot.pause()
        assert speech.queue.state.muted
        assert not owner.speech_preferences.auto_speak
    await speech.aclose()


@pytest.mark.asyncio
async def test_rebind_dismisses_owned_destination_confirmation_without_consent():
    from textual.app import App

    from tldw_chatbook.Widgets.Console.console_auto_speak_consent import (
        AutoSpeakConsentModal,
    )

    app, _, owner, other, _, handler = setup()
    ui = App()
    ui.app_config = app.app_config
    ui.console_runtime = app.console_runtime
    ui._ensure_tts_handler = app._ensure_tts_handler
    speech = BuddySpeechCoordinator(ui)
    async with ui.run_test() as pilot:
        speech.configure(BuddyBinding.for_session(owner), True)
        await speech.refresh()
        await speech.queue.wait_idle()
        speech.request_consent()
        await pilot.pause()
        assert isinstance(ui.screen, AutoSpeakConsentModal)
        speech.configure(BuddyBinding.for_session(other), True)
        await pilot.pause()
        assert not isinstance(ui.screen, AutoSpeakConsentModal)
        assert speech._consents == set() and handler.spoken == []
        await speech.aclose()


@pytest.mark.asyncio
async def test_workspace_questions_precede_named_results_and_never_acknowledge(
    monkeypatch,
):
    from tldw_chatbook.Persona_Buddy.inbox import BuddyInboxEntry
    from tldw_chatbook.UI.Navigation.buddy_workspace import BuddyWorkspaceCoordinator

    app, store, owner, other, message, handler = setup()
    owner.workspace_id = other.workspace_id = "workspace"
    other.title = "Questions"
    for session in (owner, other):
        store.confirm_auto_speak_destination(session.id, DESTINATION)
    receipt = SimpleNamespace(activity_id="receipt", assistant_message_id=message.id)
    receipts = [receipt]
    app.console_runtime.activity_receipts = SimpleNamespace(
        unseen_snapshot=lambda: tuple(receipts)
    )
    host = app.console_runtime.chat_controller._interrupt_host
    event = threading.Event()
    host.registries["question"]["round"] = {"event": event}
    host.park_round_payload(
        "question",
        "round",
        {
            "round_id": "round",
            "session_id": other.id,
            "questions": [{"question": "Which source should I use?"}],
        },
    )
    rows = (
        BuddyInboxEntry(
            "result",
            owner.title,
            "results",
            "Done",
            BuddyBinding.for_session(owner),
            ("receipt",),
        ),
        BuddyInboxEntry(
            "question",
            other.title,
            "needs_you",
            "Question",
            BuddyBinding.for_session(other),
        ),
    )

    async def snapshot(_self):
        await asyncio.sleep(0)
        return "Workspace", rows

    monkeypatch.setattr(BuddyWorkspaceCoordinator, "snapshot", snapshot)
    speech = BuddySpeechCoordinator(app)
    speech.queue.pause()
    speech.configure(BuddyBinding("workspace", "workspace"), True)
    try:
        await speech.refresh()
        speech.queue.resume()
        await speech.queue.wait_idle()
        assert handler.spoken == [
            "Questions. Your response is needed. Which source should I use?",
            "Research. Owned response",
        ]
        assert receipts == [receipt] and not event.is_set()
        assert store.active_session_id == other.id
    finally:
        await speech.aclose()
