"""Console hands-free conversation loop -- screen wiring (task 5).

`Chat/console_hands_free.py` (`HandsFreeController`) and `Chat/reply_sentence_
sequencer.py` (`SentenceSequencer`) are pure/headless and fully covered by
their own test suites (`Tests/Chat/test_console_hands_free.py`,
`Tests/Chat/test_reply_sentence_sequencer.py`). This module covers the thin
`ChatScreen` wiring that composes them with the existing dictation/TTS/V2-send
machinery: grammar entry, chip states, keypress barge-in/exit, the real
two-stage send, the reply-identity guard, and the typed-Enter hazard. See
`Docs/superpowers/specs/2026-08-02-hands-free-loop-design.md`.
"""

from __future__ import annotations

import time
from typing import Any, Callable
from unittest.mock import Mock

import pytest
from textual.widgets import Button

from Tests.UI.test_console_dictation import (
    FakeDictationSession,
    _mounted_console,
    _ready_host,
    _wait_for_mic_label,
)
from Tests.UI.test_console_dictation_streaming import (
    FakeDictationService,
    _install_streaming_session,
    _patch_availability,
)
from Tests.UI.test_console_native_chat_flow import (
    _ReadyResolutionGateway,
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)

from tldw_chatbook.Chat.console_hands_free import ExitLoop
from tldw_chatbook.Chat.console_voice_input import (
    VoiceCommand,
    VoiceVadUnavailable,
    classify_segment,
    handsfree_send_delay_seconds as real_handsfree_send_delay_seconds,
    acoustic_barge_in_enabled as real_acoustic_barge_in_enabled,
)
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module

_ASYNC_SETTLE_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Grammar (pure, no app)
# ---------------------------------------------------------------------------


def test_classify_hands_free_grammar_phrase():
    result = classify_segment("Console, hands free.")
    assert isinstance(result, VoiceCommand)
    assert result.name == "hands-free"


def test_classify_hands_free_grammar_phrase_is_whole_segment_only():
    # Fail-open rule: dictated text that merely CONTAINS the phrase is not a
    # command -- the same discipline every other grammar entry follows.
    result = classify_segment("Console, hands free to leave whenever you like.")
    assert not isinstance(result, VoiceCommand)


# ---------------------------------------------------------------------------
# Config readers (pure, no app)
# ---------------------------------------------------------------------------


def test_handsfree_send_delay_seconds_reader_default(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_voice_input.get_cli_setting",
        lambda *a, **k: 1.5,
    )
    assert real_handsfree_send_delay_seconds() == 1.5


def test_handsfree_send_delay_seconds_reader_rejects_non_numeric(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_voice_input.get_cli_setting",
        lambda *a, **k: "not-a-number",
    )
    assert real_handsfree_send_delay_seconds() == 1.5


def test_handsfree_send_delay_seconds_reader_rejects_non_positive(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_voice_input.get_cli_setting",
        lambda *a, **k: -3,
    )
    assert real_handsfree_send_delay_seconds() == 1.5


def test_handsfree_send_delay_seconds_reader_accepts_configured_value(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_voice_input.get_cli_setting",
        lambda *a, **k: 2.5,
    )
    assert real_handsfree_send_delay_seconds() == 2.5


def test_acoustic_barge_in_enabled_reader_default_false(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_voice_input.get_cli_setting",
        lambda *a, **k: False,
    )
    assert real_acoustic_barge_in_enabled() is False


def test_acoustic_barge_in_enabled_reader_accepts_truthy_string(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_voice_input.get_cli_setting",
        lambda *a, **k: "true",
    )
    assert real_acoustic_barge_in_enabled() is True


def test_acoustic_barge_in_enabled_reader_accepts_falsy_string(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_voice_input.get_cli_setting",
        lambda *a, **k: "off",
    )
    assert real_acoustic_barge_in_enabled() is False


# ---------------------------------------------------------------------------
# Fakes shared by the mounted-app wiring tests
# ---------------------------------------------------------------------------


class _FakeHandsFreeTtsHandler:
    """Stands in for `TTSEventHandler`'s `speak_utterance` entry.

    Records every call; `on_finished` fires per `mode`:
      - "immediate": synchronously, with `ok=True` (default -- keeps a
        multi-sentence reply's utterances draining without a test having to
        manually resolve each one).
      - "fail": synchronously, with `ok=False`.
    """

    def __init__(self, mode: str = "immediate") -> None:
        self.mode = mode
        self.calls: list[tuple[str, bool]] = []

    async def speak_utterance(
        self, text: str, *, on_finished: Callable[[bool], None], quiet: bool = False
    ) -> None:
        self.calls.append((text, quiet))
        on_finished(self.mode != "fail")


def _install_fake_tts_handler(app, mode: str = "immediate") -> _FakeHandsFreeTtsHandler:
    handler = _FakeHandsFreeTtsHandler(mode=mode)

    async def _ensure_tts_handler():
        return handler

    app._ensure_tts_handler = _ensure_tts_handler
    return handler


class _HandsFreeReplyGateway(_ReadyResolutionGateway):
    """A ready gateway that streams a fixed reply and records what it sent."""

    def __init__(
        self, reply_text: str = "Reply sentence one. Reply sentence two. "
    ) -> None:
        self.reply_text = reply_text
        self.sent_messages: list[list[dict]] = []

    async def stream_chat(self, resolution, messages):
        self.sent_messages.append(list(messages))
        yield self.reply_text


def _fast_countdown(monkeypatch, seconds: float = 0.3) -> None:
    """Speed up the hands-free countdown so wiring tests don't wait 1.5s+."""
    monkeypatch.setattr(
        chat_screen_module, "handsfree_send_delay_seconds", lambda: seconds
    )


# ---------------------------------------------------------------------------
# Entry: from idle (key binding) and from a live capture (spoken command)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_key_binding_starts_loop_from_idle_and_opens_capture(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        assert console._console_hands_free is None

        console.action_toggle_console_hands_free()
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        assert console._console_hands_free is not None
        assert console._console_hands_free.controller.state == "listening"
        assert fake.start_calls == 1


@pytest.mark.asyncio
async def test_spoken_hands_free_command_adopts_live_capture_as_first_turn(
    monkeypatch,
):
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        assert console._console_hands_free is None

        service.emit_final("Console, hands free.")
        await pilot.pause()

        assert console._console_hands_free is not None
        assert console._console_hands_free.controller.state == "listening"
        # The capture that was already open is adopted, not restarted.
        assert service.stop_calls == 0
        assert str(composer.query_one("#console-dictation", Button).label) == "Rec ●"


# ---------------------------------------------------------------------------
# Countdown chip + the real two-stage send
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_countdown_chip_painted_and_two_stage_send_drives_real_flow(
    monkeypatch,
):
    """The load-bearing end-to-end wiring test.

    Speak a segment -> countdown chip paints "sending in ...s" -> the
    countdown expires -> the REAL V2 send flow ships the dictated text
    through a stub gateway -> the reply streams into the store -> the
    sequencer speaks it through the faked TTS entry -> the reply completes
    -> the loop returns to `listening` and reopens the microphone.
    """
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _fast_countdown(monkeypatch, seconds=0.3)
    gateway = _HandsFreeReplyGateway("First sentence here. Second one too. ")
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    tts = _install_fake_tts_handler(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        store = console._ensure_console_chat_store()

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        console.action_toggle_console_hands_free()
        await pilot.pause()
        assert console._console_hands_free is not None

        service.emit_final("hello from hands free")
        await pilot.pause()
        assert console._console_hands_free.controller.state == "countdown"

        deadline = time.monotonic() + 4
        painted = False
        while time.monotonic() < deadline:
            if "sending in" in _visible_text(console):
                painted = True
                break
            await pilot.pause(0.02)
        assert painted, _visible_text(console)

        # Let the countdown expire and the real send flow run.
        deadline = time.monotonic() + _ASYNC_SETTLE_TIMEOUT
        while time.monotonic() < deadline and not gateway.sent_messages:
            await pilot.pause(0.02)
        assert gateway.sent_messages, "the countdown never drove a real send"
        sent_user_turns = [
            m["content"]
            for turn in gateway.sent_messages
            for m in turn
            if m.get("role") == "user"
        ]
        assert any("hello from hands free" in text for text in sent_user_turns)

        # The reply streamed -> the sequencer spoke it via the faked entry.
        deadline = time.monotonic() + _ASYNC_SETTLE_TIMEOUT
        while time.monotonic() < deadline and not tts.calls:
            await pilot.pause(0.02)
        assert tts.calls, "the reply never reached speak_utterance"
        assert any("First sentence" in text for text, _quiet in tts.calls)

        # Drained -> back to listening, mic reopened.
        deadline = time.monotonic() + _ASYNC_SETTLE_TIMEOUT
        while (
            time.monotonic() < deadline
            and console._console_hands_free.controller.state != "listening"
        ):
            await pilot.pause(0.02)
        assert console._console_hands_free.controller.state == "listening"
        await _wait_for_mic_label(composer, pilot, "Rec ●")

        # A completed assistant message actually landed in the store.
        session_id = store.active_session_id
        messages = store.messages_for_session(session_id)
        assert any(
            m.role == "assistant" and "First sentence" in m.content
            for m in messages
        )


@pytest.mark.asyncio
async def test_spoken_feedback_false_still_speaks_reply(monkeypatch):
    """Reply speech is intrinsic to hands-free -- it must not read
    `dictation.spoken_feedback` (which governs status acks only)."""

    def _fake_cli_setting(section, key=None, default=None):
        if (section, key) == ("dictation", "spoken_feedback"):
            return False
        return default

    monkeypatch.setattr(
        chat_screen_module, "get_cli_setting", _fake_cli_setting
    )
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _fast_countdown(monkeypatch, seconds=0.2)
    gateway = _HandsFreeReplyGateway("Spoken despite the flag. ")
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    tts = _install_fake_tts_handler(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        console.action_toggle_console_hands_free()
        await pilot.pause()

        service.emit_final("please reply")
        await pilot.pause()

        deadline = time.monotonic() + _ASYNC_SETTLE_TIMEOUT
        while time.monotonic() < deadline and not tts.calls:
            await pilot.pause(0.02)
        assert tts.calls
        assert any("Spoken despite" in text for text, _quiet in tts.calls)


# ---------------------------------------------------------------------------
# Keypress barge-in / exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keypress_in_speaking_silences_and_reopens_capture(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )

        console.action_toggle_console_hands_free()
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        session = console._console_hands_free
        assert session is not None

        # Drive straight to `speaking` without a real send -- the FSM/
        # sequencer transition logic is Task 2/3's own, fully unit-tested
        # territory; this test only pins the WIRING reaction to a barge-in
        # keypress. `RequestStopAndSend`'s own wiring effect is stubbed so
        # this stays a pure unit test of the barge-in reaction, not a live
        # network send.
        console._console_hands_free_request_stop_and_send = lambda: None
        session.controller._begin_awaiting_reply()
        session.controller.on_reply_started()
        session.controller.on_first_utterance()
        assert session.controller.state == "speaking"

        post_message = Mock()
        console.app_instance.post_message = post_message

        await pilot.press("x")
        await pilot.pause()

        assert session.controller.state == "listening"
        posted_actions = [
            getattr(call.args[0], "action", None) for call in post_message.call_args_list
        ]
        assert "stop" in posted_actions


@pytest.mark.asyncio
async def test_esc_exits_loop_and_restores_normal_esc_semantics(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )

        console.action_toggle_console_hands_free()
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        assert console._console_hands_free is not None

        await pilot.press("escape")
        await pilot.pause()
        await _wait_for_mic_label(composer, pilot, "Mic")

        assert console._console_hands_free is None
        assert fake.stop_calls == 1

        # The ordinary (non-priority) Esc binding fires normally again --
        # `focus_console_composer_home` -- once the loop is not running.
        called = []
        monkeypatch.setattr(
            chat_screen_module.ChatScreen,
            "_focus_console_composer_if_needed",
            lambda self, force=False: called.append(force),
        )
        await pilot.press("escape")
        await pilot.pause()
        assert called == [True]


# ---------------------------------------------------------------------------
# on_key byte-identity outside the loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_key_outside_hands_free_loop_behaves_exactly_as_before(monkeypatch):
    """With no hands-free session, `on_key` must do exactly what it did
    before task 5: type a printable key, then Enter sends it -- the new
    hands-free branch (gated on `self._console_hands_free is not None`)
    must never run, and Esc must still hit the ordinary composer-focus
    binding rather than any hands-free exit path."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        assert console._console_hands_free is None

        await pilot.press("h", "i")
        assert composer.draft_text() == "hi"

        called = []
        monkeypatch.setattr(
            chat_screen_module.ChatScreen,
            "_focus_console_composer_if_needed",
            lambda self, force=False: called.append(force),
        )
        await pilot.press("escape")
        await pilot.pause()
        assert called == [True]


# ---------------------------------------------------------------------------
# Typed-Enter hazard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typed_enter_during_listening_sends_normally_once(monkeypatch):
    """Mic open in hands-free `listening`: typing a draft and pressing
    Enter must send the TYPED draft via the normal path exactly once, and
    must not also fire hands-free's own voice-triggered send."""
    fake = FakeDictationSession()
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    send_calls: list[str] = []

    async def _fake_send(self, event) -> bool:
        # TASK-340: a keyboard send stashes the draft (and clears it from
        # the composer) at the Enter KEYPRESS, before the `Button.Pressed`
        # this fake intercepts even fires -- `composer.draft_text()` reads
        # empty by this point; the stash is where the sent text actually is.
        stash = self._console_pending_send_stash
        send_calls.append(stash.text if stash is not None else "")
        event.stop()
        return True

    monkeypatch.setattr(
        chat_screen_module.ChatScreen, "handle_console_send_message", _fake_send
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )

        console.action_toggle_console_hands_free()
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        assert console._console_hands_free.controller.state == "listening"

        await pilot.press("t", "y", "p", "e", "d")
        assert composer.draft_text() == "typed"
        await pilot.press("enter")
        await pilot.pause()

        assert send_calls == ["typed"]
        # Still listening -- an ordinary typed send is not this loop's
        # business, and `on_composer_key()` is a documented no-op there.
        assert console._console_hands_free.controller.state == "listening"


@pytest.mark.asyncio
async def test_typed_enter_cancels_an_armed_countdown_first(monkeypatch):
    """The SAME keypress that would send the typed draft also reaches
    `on_composer_key()` first (this wiring's `on_key` ordering) -- a
    countdown armed from a spoken segment is cancelled before Enter's own
    send logic runs, so the auto-send never fires alongside the typed one."""
    fake = FakeDictationSession()
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    send_calls: list[str] = []

    async def _fake_send(self, event) -> bool:
        # TASK-340: a keyboard send stashes the draft (and clears it from
        # the composer) at the Enter KEYPRESS, before the `Button.Pressed`
        # this fake intercepts even fires -- `composer.draft_text()` reads
        # empty by this point; the stash is where the sent text actually is.
        stash = self._console_pending_send_stash
        send_calls.append(stash.text if stash is not None else "")
        event.stop()
        return True

    monkeypatch.setattr(
        chat_screen_module.ChatScreen, "handle_console_send_message", _fake_send
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )

        console.action_toggle_console_hands_free()
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        session = console._console_hands_free
        session.controller.on_voice_final()
        assert session.controller.state == "countdown"

        await pilot.press("t")
        await pilot.pause()

        assert session.controller.state == "listening"
        await pilot.press("y", "p")
        await pilot.press("enter")
        await pilot.pause()

        assert send_calls == ["typ"]


# ---------------------------------------------------------------------------
# Reply identity (binding carrier)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_identity_drops_deltas_and_completion_for_a_different_id():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)

        # `capture_live=True` -- adopts an "already open" capture without
        # touching real dictation machinery at all (no `OpenCapture` is
        # emitted from `idle`), which is all these tests need: they drive
        # the controller/sequencer/tap wiring directly, never a real mic.
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()
        session = console._console_hands_free
        assert session is not None

        # Force the controller into `awaiting_reply` without a real send.
        session.controller._begin_awaiting_reply()
        assert session.controller.state == "awaiting_reply"

        fed: list[str] = []
        session.sequencer.feed = fed.append

        # The FIRST delta observed while `awaiting_reply` claims `reply_id`
        # -- there is no independent ground truth to validate it against
        # (see `_on_console_hands_free_delta`'s own docstring); that is
        # exactly what makes every LATER delta for a DIFFERENT id
        # identifiable as stale/foreign and droppable.
        console._on_console_hands_free_delta("real-id", "first chunk. ")
        assert session.reply_id == "real-id"
        assert fed == ["first chunk. "]

        # A delta for a DIFFERENT id, arriving mid-reply, must be dropped.
        console._on_console_hands_free_delta("stale-id", "must not feed")
        assert fed == ["first chunk. "]

        # ...while a delta for the SAME id keeps feeding normally.
        console._on_console_hands_free_delta("real-id", "second chunk. ")
        assert fed == ["first chunk. ", "second chunk. "]

        finished: list[Any] = []
        session.sequencer.reply_completed = lambda: finished.append("sequencer")
        session.controller.on_reply_finished = lambda: finished.append("controller")

        console._on_console_hands_free_terminal("stale-id", failed=False)
        assert finished == []

        console._on_console_hands_free_terminal("real-id", failed=False)
        assert finished == ["sequencer", "controller"]


@pytest.mark.asyncio
async def test_zero_content_reply_completion_still_completes_the_turn():
    """A reply that streams ZERO chunks (empty/pure-tool-call) must still
    reach `on_reply_finished()` via its completion tap, or the loop hangs
    in `awaiting_reply` until the 30s watchdog gives up."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        # `capture_live=True` -- adopts an "already open" capture without
        # touching real dictation machinery at all (no `OpenCapture` is
        # emitted from `idle`), which is all these tests need: they drive
        # the controller/sequencer/tap wiring directly, never a real mic.
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()
        session = console._console_hands_free
        session.controller._begin_awaiting_reply()

        finished: list[Any] = []
        session.sequencer.reply_completed = lambda: finished.append("sequencer")
        session.controller.on_reply_finished = lambda: finished.append("controller")

        console._on_console_hands_free_terminal("zero-content-id", failed=False)

        assert session.reply_id == "zero-content-id"
        assert finished == ["sequencer", "controller"]


# ---------------------------------------------------------------------------
# ExitLoop teardown / idempotent Open+CloseCapture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exit_loop_intent_emits_silence_and_close_capture_itself(monkeypatch):
    """`ExitLoop` reaching the FSM does not itself carry `SilenceSpeech`/
    `CloseCapture` -- the wiring's own `_console_hands_free_exit_loop` must
    perform both (binding carrier)."""
    fake = FakeDictationSession()
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        console.action_toggle_console_hands_free()
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        session = console._console_hands_free

        # Drive straight to `speaking`, stubbing `RequestStopAndSend`'s own
        # wiring effect (a real V2 send) -- this test targets ONLY the
        # `ExitLoop` handler's own teardown behavior.
        console._console_hands_free_request_stop_and_send = lambda: None
        session.controller._begin_awaiting_reply()
        session.controller.on_reply_started()
        session.controller.on_first_utterance()
        assert session.controller.state == "speaking"
        # An utterance in flight (without a real TTS round trip -- Task 2's
        # own suite already covers `flush()`'s internal behavior
        # exhaustively; this only needs `flush()` to have something to act
        # on so its `stop_speech()` call -- the both-ways TTS stop routine
        # -- actually fires).
        session.sequencer._inflight = True

        post_message = Mock()
        console.app_instance.post_message = post_message

        console._handle_console_hands_free_intent(ExitLoop())

        assert session.sequencer._inflight is False
        posted_actions = [
            getattr(call.args[0], "action", None) for call in post_message.call_args_list
        ]
        assert "stop" in posted_actions
        assert console._console_hands_free is None
        await _wait_for_mic_label(composer, pilot, "Mic")
        assert fake.stop_calls == 1


@pytest.mark.asyncio
async def test_open_and_close_capture_handlers_are_idempotent_no_ops(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        assert console._console_dictation_state == "idle"

        # OpenCapture while idle starts; while ALREADY recording, no-op.
        console._console_hands_free_open_capture()
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        console._console_hands_free_open_capture()
        await pilot.pause()
        assert fake.start_calls == 1

        # CloseCapture is a no-op unless genuinely `recording`.
        console._console_dictation_state = "idle"
        console._console_hands_free_close_capture()
        await pilot.pause()
        assert fake.stop_calls == 0


# ---------------------------------------------------------------------------
# Acoustic barge-in reopens capture during `speaking`
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acoustic_barge_in_opens_capture_on_reply_started(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    monkeypatch.setattr(
        chat_screen_module, "acoustic_barge_in_enabled", lambda: True
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )

        console.action_toggle_console_hands_free()
        await _wait_for_mic_label(composer, pilot, "Rec ●")
        session = console._console_hands_free
        assert session.controller._acoustic_barge_in is True

        # `RequestStopAndSend`'s own wiring effect is stubbed -- this test
        # targets ONLY acoustic mode's reopen-on-`on_reply_started` effect,
        # not a real V2 send. `CloseCapture` (real) still closes the mic.
        console._console_hands_free_request_stop_and_send = lambda: None
        session.controller._begin_awaiting_reply()

        deadline = time.monotonic() + _ASYNC_SETTLE_TIMEOUT
        while (
            time.monotonic() < deadline
            and console._console_dictation_state != "idle"
        ):
            await pilot.pause(0.02)
        assert console._console_dictation_state == "idle"

        session.controller.on_reply_started()
        await pilot.pause()

        # Acoustic mode reopens the mic the instant generation starts,
        # rather than waiting for the reply to drain.
        assert fake.start_calls == 2
        await _wait_for_mic_label(composer, pilot, "Rec ●")


# ---------------------------------------------------------------------------
# VAD-degraded honesty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vad_degraded_entry_warns_instead_of_promising_auto_send(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        chat_screen_module.ChatScreen,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        notifications: list[tuple[str, dict]] = []
        console.app_instance.notify = lambda message, **kwargs: notifications.append(
            (message, kwargs)
        )
        console._handle_console_dictation_event(
            chat_screen_module.ConsoleDictationEvent(
                console._console_dictation_session, VoiceVadUnavailable()
            )
        )
        assert console._console_hands_free_vad_degraded is True

        console.action_toggle_console_hands_free()
        await pilot.pause()

        assert any(
            "auto-send" in message and kwargs.get("severity") == "warning"
            for message, kwargs in notifications
        )
