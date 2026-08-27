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

import asyncio
import threading
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

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_hands_free import ExitLoop, HandsFreeController
from tldw_chatbook.Chat.console_voice_input import (
    VoiceCommand,
    VoiceVadUnavailable,
    classify_segment,
    handsfree_send_delay_seconds as real_handsfree_send_delay_seconds,
    acoustic_barge_in_enabled as real_acoustic_barge_in_enabled,
)
from tldw_chatbook.UI.Console_Modules import dictation as dictation_module
from tldw_chatbook.UI.Console_Modules import hands_free as hands_free_module
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

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


def _spy_get_cli_setting(monkeypatch, value):
    """Task-5 review M5: record the EXACT call args `get_cli_setting` was
    invoked with, so a reader misreading a typo'd/wrong key cannot pass
    silently the way a bare `lambda *a, **k: value` would."""
    calls: list[tuple] = []

    def _fake(*args, **kwargs):
        calls.append((args, kwargs))
        return value

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_voice_input.get_cli_setting", _fake
    )
    return calls


def test_handsfree_send_delay_seconds_reader_reads_the_exact_key(monkeypatch):
    calls = _spy_get_cli_setting(monkeypatch, 1.5)
    assert real_handsfree_send_delay_seconds() == 1.5
    assert calls == [
        (
            ("dictation", "handsfree_send_delay_seconds", 1.5),
            {},
        )
    ]


def test_handsfree_send_delay_seconds_reader_rejects_non_numeric(monkeypatch):
    _spy_get_cli_setting(monkeypatch, "not-a-number")
    assert real_handsfree_send_delay_seconds() == 1.5


def test_handsfree_send_delay_seconds_reader_rejects_non_positive(monkeypatch):
    _spy_get_cli_setting(monkeypatch, -3)
    assert real_handsfree_send_delay_seconds() == 1.5


def test_handsfree_send_delay_seconds_reader_accepts_configured_value(monkeypatch):
    _spy_get_cli_setting(monkeypatch, 2.5)
    assert real_handsfree_send_delay_seconds() == 2.5


def test_acoustic_barge_in_enabled_reader_reads_the_exact_key(monkeypatch):
    calls = _spy_get_cli_setting(monkeypatch, False)
    assert real_acoustic_barge_in_enabled() is False
    assert calls == [(("dictation.acoustic_barge_in", False), {})]


def test_acoustic_barge_in_enabled_reader_accepts_truthy_string(monkeypatch):
    _spy_get_cli_setting(monkeypatch, "true")
    assert real_acoustic_barge_in_enabled() is True


def test_acoustic_barge_in_enabled_reader_accepts_falsy_string(monkeypatch):
    _spy_get_cli_setting(monkeypatch, "off")
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

    async def stream_chat(self, resolution, messages, **kwargs):
        # `**kwargs`: dev's gateway interface grew keyword arguments (e.g.
        # `signal`) after this suite was written -- absorb them the same way
        # the donor module's gateways do, or every send dies with a TypeError
        # that the loop's failure path then politely swallows.
        self.sent_messages.append(list(messages))
        yield self.reply_text


def _make_active_conversation_temporary(console) -> None:
    """Allow real sends in the database-free screen harness."""

    store = console._ensure_console_chat_store()
    active_id = store.active_session_id
    next(session for session in store.sessions() if session.id == active_id).ephemeral = (
        True
    )


def _fast_countdown(monkeypatch, seconds: float = 0.3) -> None:
    """Speed up the hands-free countdown so wiring tests don't wait 1.5s+."""
    monkeypatch.setattr(
        hands_free_module, "handsfree_send_delay_seconds", lambda: seconds
    )


# ---------------------------------------------------------------------------
# Entry: from idle (key binding) and from a live capture (spoken command)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_key_binding_starts_loop_from_idle_and_opens_capture(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
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
        await _wait_for_mic_label(composer, pilot, "Dictating")

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
        await _wait_for_mic_label(composer, pilot, "Dictating")
        assert console._console_hands_free is None

        service.emit_final("Console, hands free.")
        await pilot.pause()

        assert console._console_hands_free is not None
        assert console._console_hands_free.controller.state == "listening"
        # The capture that was already open is adopted, not restarted.
        assert service.stop_calls == 0
        assert str(composer.query_one("#console-dictation", Button).label) == "Dictating"


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
        _make_active_conversation_temporary(console)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        store = console._ensure_console_chat_store()

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        console.action_toggle_console_hands_free()
        await pilot.pause()
        assert console._console_hands_free is not None

        service.emit_final("hello from hands free")
        await pilot.pause()
        session = console._console_hands_free
        assert session.controller.state == "countdown"
        # The mode change seeds the configured delay, then the real 0.1 s
        # timer may decrement it before this posted event has settled.
        assert 0 < session.countdown_remaining <= 0.3

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
        await _wait_for_mic_label(composer, pilot, "Dictating")

        # A completed assistant message actually landed in the store.
        session_id = store.active_session_id
        messages = store.messages_for_session(session_id)
        assert any(
            m.role == "assistant" and "First sentence" in m.content
            for m in messages
        )


@pytest.mark.asyncio
async def test_countdown_cancel_restores_the_chip(monkeypatch):
    """Task-5 final review I1 (PROBE A): cancelling an armed countdown
    (`on_speech_resumed()`, the primary cancel route) must restore the
    ordinary listening chip immediately -- not leave "sending in ...s…"
    on screen advertising a send that was already cancelled, which used
    to persist through the user's entire next utterance."""
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _fast_countdown(monkeypatch, seconds=5.0)  # generous -- must not expire
    _, host = _ready_host()

    async with host.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        console.action_toggle_console_hands_free()
        await pilot.pause()
        session = console._console_hands_free

        service.emit_final("hello from hands free")
        await pilot.pause()
        assert session.controller.state == "countdown"
        await _wait_for(lambda: "sending in" in _visible_text(console), pilot)

        session.controller.on_speech_resumed()
        assert session.controller.state == "listening"
        await pilot.pause()

        assert "sending in" not in _visible_text(console), _visible_text(
            console
        )


@pytest.mark.asyncio
async def test_segment_no_final_consumes_latch_so_the_next_real_final_still_arms_countdown(
    monkeypatch,
):
    """Qodo review (task-5 follow-up): wiring-level pin for `HandsFree
    Controller.on_segment_no_final` -- proves the SCREEN actually routes a
    real `VoiceSegmentNoFinal` event to it (through `ConsoleDictationEvent`
    -> `_handle_console_dictation_event`'s `_console_dictation_state ==
    "recording"` guard, the same one `VoiceFinal`/`VoiceSpeechResumed` use),
    not just that the FSM method is correct in isolation (see `Tests/Chat/
    test_console_hands_free.py`'s own pin for that half).

    Without this wiring, a resume latched via `VoiceSpeechResumed` while a
    segment was transcribing to nothing would sit armed forever (no
    `VoiceFinal` ever arrives for a blank segment to consume it) and
    incorrectly swallow the NEXT real segment's final -- dropping a whole
    turn's countdown silently.
    """
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _fast_countdown(monkeypatch, seconds=5.0)  # generous -- must not expire
    _, host = _ready_host()

    async with host.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        console.action_toggle_console_hands_free()
        await pilot.pause()
        session = console._console_hands_free

        service.emit_speech_resumed()  # latches, before this segment's own outcome
        await pilot.pause()
        assert session.controller.state == "listening"

        service.emit_segment_no_final()  # THIS segment transcribed to nothing
        await pilot.pause()

        service.emit_final("hello from hands free")  # a genuine final: next segment
        await pilot.pause()

        assert session.controller.state == "countdown"


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
        _make_active_conversation_temporary(console)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
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
# At-most-one-failure-toast-per-reply (task-5 review M2) and the
# multi-reply hazard (task-5 review M3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_at_most_one_failure_toast_per_reply():
    """Task-5 review M2: the first failed utterance in a reply passes
    `quiet=False` (a toast is allowed); every LATER failed utterance in
    the SAME reply passes `quiet=True` (log only)."""
    app = _build_test_app()
    tts = _install_fake_tts_handler(app, mode="fail")
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()
        session = console._console_hands_free

        session.sequencer.feed("First sentence fails. Second sentence too. ")
        await _wait_for(lambda: len(tts.calls) >= 2, pilot)

        assert [quiet for _text, quiet in tts.calls[:2]] == [False, True]
        assert session.toast_shown_for_reply is True


@pytest.mark.asyncio
async def test_two_sequential_replies_both_drain_through_the_real_wiring(
    monkeypatch,
):
    """Task-5 review M3: `begin_reply()` must run at EACH reply start
    ("a reused sequencer without it never drains reply 2", per the brief)
    -- pinned end to end by driving TWO full turns through the real
    wiring, not just inspecting the call site."""
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _fast_countdown(monkeypatch, seconds=0.2)
    gateway = _HandsFreeReplyGateway("Reply one sentence. ")
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    tts = _install_fake_tts_handler(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(host, pilot)
        _make_active_conversation_temporary(console)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        console.action_toggle_console_hands_free()
        await pilot.pause()

        service.emit_final("turn one")
        await _wait_for(lambda: len(gateway.sent_messages) >= 1, pilot)
        await _wait_for(lambda: len(tts.calls) >= 1, pilot)
        assert any("Reply one sentence" in text for text, _q in tts.calls)
        await _wait_for(
            lambda: console._console_hands_free is not None
            and console._console_hands_free.controller.state == "listening",
            pilot,
        )
        await _wait_for_mic_label(composer, pilot, "Dictating")

        gateway.reply_text = "Reply two sentence. "
        service.emit_final("turn two")
        await _wait_for(lambda: len(gateway.sent_messages) >= 2, pilot)
        await _wait_for(
            lambda: any("Reply two sentence" in text for text, _q in tts.calls),
            pilot,
        )
        await _wait_for(
            lambda: console._console_hands_free is not None
            and console._console_hands_free.controller.state == "listening",
            pilot,
        )
        await _wait_for_mic_label(composer, pilot, "Dictating")
        assert console._console_hands_free.sequencer.drained


# ---------------------------------------------------------------------------
# Capture-ending spoken commands other than "stop" (task-5 review I3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spoken_send_mid_loop_drives_a_real_send_and_speaks_the_reply(
    monkeypatch,
):
    """Task-5 review I3: spoken "Console, send." mid-loop must drive the
    SAME semantics as a countdown expiry -- `awaiting_reply` is entered and
    the reply is actually spoken -- not just end the capture and leave the
    FSM in `listening` silently dropping the reply the user just asked
    hands-free to send."""
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    gateway = _HandsFreeReplyGateway("Spoken send reply. ")
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    tts = _install_fake_tts_handler(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(host, pilot)
        _make_active_conversation_temporary(console)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        console.action_toggle_console_hands_free()
        await pilot.pause()
        assert console._console_hands_free.controller.state == "listening"

        service.emit_final("hello from spoken send")
        await pilot.pause()
        service.emit_final("Console, send.")
        await pilot.pause()

        await _wait_for(lambda: bool(gateway.sent_messages), pilot)
        sent_user_turns = [
            m["content"]
            for turn in gateway.sent_messages
            for m in turn
            if m.get("role") == "user"
        ]
        assert any("hello from spoken send" in t for t in sent_user_turns)

        await _wait_for(lambda: bool(tts.calls), pilot)
        assert any("Spoken send reply" in text for text, _q in tts.calls)

        # Drains back to listening once the reply completes.
        await _wait_for(
            lambda: console._console_hands_free is not None
            and console._console_hands_free.controller.state == "listening",
            pilot,
        )


@pytest.mark.asyncio
async def test_discard_mid_loop_exits_cleanly_instead_of_desyncing(monkeypatch):
    """Task-5 review I3: spoken "Console, discard." mid-loop must not
    leave the FSM `listening` with `capture_open=True` while the real mic
    is closed (alive-but-inert forever) -- it must end the loop cleanly."""
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
        await _wait_for_mic_label(composer, pilot, "Dictating")
        console.action_toggle_console_hands_free()
        await pilot.pause()
        assert console._console_hands_free is not None

        service.emit_final("some words to discard")
        await pilot.pause()
        service.emit_final("Console, discard.")
        await pilot.pause()

        await _wait_for(lambda: console._console_hands_free is None, pilot)
        await _wait_for_mic_label(composer, pilot, "Dictate")
        # The loop is genuinely gone, not desynced -- pressing the mic
        # button again does an ordinary one-shot start, proving the
        # dictation state machine (and the hands-free bookkeeping) agree.
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        assert console._console_hands_free is None


@pytest.mark.asyncio
async def test_spoken_send_mid_reply_acoustic_mode_ends_capture_and_exits(
    monkeypatch,
):
    """Task-5 review round 2, D3: spoken "Console, send." while a reply is
    already outstanding (only reachable in acoustic mode -- the only mode
    with the mic open mid-reply) must not be a silent no-op -- it must end
    the capture and exit the loop cleanly, the same honest choice already
    made for discard/new-session/read-that-back."""
    fake = FakeDictationSession()
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    monkeypatch.setattr(
        hands_free_module, "acoustic_barge_in_enabled", lambda: True
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )
        console.action_toggle_console_hands_free()
        await _wait_for_mic_label(composer, pilot, "Dictating")
        session = console._console_hands_free
        assert session.controller._acoustic_barge_in is True

        # Drive straight to `awaiting_reply` with the mic reopened
        # (acoustic mode's own effect of `on_reply_started`), stubbing
        # `RequestStopAndSend`'s own wiring effect (a real V2 send) --
        # this test targets ONLY the mid-reply spoken-"send" reaction.
        console._hands_free._console_hands_free_request_stop_and_send = lambda: None
        session.controller._begin_awaiting_reply()
        # `_begin_awaiting_reply()`'s own `CloseCapture` is real and async
        # -- `on_reply_started()`'s reopen must wait for it to actually
        # land on `idle`, or the reopen's own `state == "idle"` guard
        # no-ops (same ordering B2 already established).
        await _wait_for(lambda: console._console_dictation_state == "idle", pilot)
        session.controller.on_reply_started()
        await _wait_for_mic_label(composer, pilot, "Dictating")
        assert session.controller.state == "awaiting_reply"

        console._handle_console_dictation_event(
            chat_screen_module.ConsoleDictationEvent(
                console._console_dictation_session, VoiceCommand("send")
            )
        )
        await pilot.pause()

        await _wait_for(lambda: console._console_hands_free is None, pilot)
        await _wait_for_mic_label(composer, pilot, "Dictate")


def test_silence_speech_posts_stop_unconditionally_even_with_nothing_in_flight():
    """Task-5 review M9: a `_speak_status` ack ("Sent.", "Discarded.", ...)
    bypasses the sequencer entirely -- `SilenceSpeech`'s handler must post
    the both-ways stop UNCONDITIONALLY, not only via `flush()`'s own
    conditional `stop_speech()` (which only fires when an utterance is
    actually in flight)."""
    app = _build_test_app()
    console = chat_screen_module.ChatScreen(app)
    posted = Mock()
    app.post_message = posted

    class _FakeSequencer:
        def __init__(self) -> None:
            self.flush_calls = 0

        def flush(self) -> None:
            self.flush_calls += 1  # nothing in flight -> stop_speech() never called

    fake_sequencer = _FakeSequencer()
    console._console_hands_free = chat_screen_module.ConsoleHandsFreeSession(
        controller=object(), sequencer=fake_sequencer
    )

    console._hands_free._console_hands_free_silence_speech()

    assert fake_sequencer.flush_calls == 1
    posted.assert_called_once()
    (event,), _kwargs = posted.call_args
    assert getattr(event, "action", None) == "stop"


# ---------------------------------------------------------------------------
# Keypress barge-in / exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keypress_in_speaking_silences_and_reopens_capture(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
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
        await _wait_for_mic_label(composer, pilot, "Dictating")
        session = console._console_hands_free
        assert session is not None

        # Drive straight to `speaking` without a real send -- the FSM/
        # sequencer transition logic is Task 2/3's own, fully unit-tested
        # territory; this test only pins the WIRING reaction to a barge-in
        # keypress. `RequestStopAndSend`'s own wiring effect is stubbed so
        # this stays a pure unit test of the barge-in reaction, not a live
        # network send.
        console._hands_free._console_hands_free_request_stop_and_send = lambda: None
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
        dictation_module.ConsoleDictationController,
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
        await _wait_for_mic_label(composer, pilot, "Dictating")
        assert console._console_hands_free is not None

        await pilot.press("escape")
        await pilot.pause()
        await _wait_for_mic_label(composer, pilot, "Dictate")

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


@pytest.mark.asyncio
async def test_barge_in_and_esc_work_with_focus_off_the_composer(monkeypatch):
    """Task-5 review I2: keyboard barge-in and Esc are the loop's PRIMARY
    interruption/exit mechanism (the docs say "press any key"/"Esc from
    any point in the loop") -- both must keep working when focus has moved
    away from the composer (clicking/scrolling the transcript), not only
    when it happens to still be focused."""
    fake = FakeDictationSession()
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
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
        await _wait_for_mic_label(composer, pilot, "Dictating")
        session = console._console_hands_free
        console._hands_free._console_hands_free_request_stop_and_send = lambda: None

        def _drive_to_speaking() -> None:
            session.controller._begin_awaiting_reply()
            session.controller.on_reply_started()
            session.controller.on_first_utterance()
            assert session.controller.state == "speaking"

        transcript = console.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        transcript.focus()
        await pilot.pause()
        assert console.app.focused is transcript
        assert console._should_capture_console_input(composer) is False

        _drive_to_speaking()
        await pilot.press("x")
        await pilot.pause()
        assert session.controller.state == "listening", (
            "barge-in did nothing with focus off the composer"
        )

        _drive_to_speaking()
        transcript.focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert console._console_hands_free is None, (
            "Esc did not exit the loop with focus off the composer"
        )


# ---------------------------------------------------------------------------
# On the DEFAULT production send path: worker-thread marshal (task-5
# review I1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hands_free_marshal_routes_off_thread_calls_through_call_from_thread():
    """On `[console] agent_runtime`'s DEFAULT production send path, the
    delta/completion tap runs on a worker thread with its own event loop
    (`ConsoleChatController._run_agent_reply` ->
    `asyncio.to_thread(bridge.run_reply)` -> `console_agent_bridge.py`'s
    streaming adapter -> `store.append_stream_chunk`/etc, all on that
    thread). The harness cannot build the real agent bridge (in-memory DB
    -> None per `_ensure_console_agent_bridge`), so this pins the tap
    boundary directly: call `_console_hands_free_marshal` from a REAL
    background thread and assert it routes through `app_instance.call_
    from_thread` rather than running the callback in place. Requires the
    loop to be running -- see the fast-path test below for the loop-off
    case (task-5 review round 2, D1)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()
        marshal_calls = Mock()
        console.app_instance.call_from_thread = marshal_calls
        direct_calls: list[tuple] = []
        off_thread_id: dict[str, int] = {}

        def _callback(a, b):
            direct_calls.append((a, b))

        def _from_background_thread() -> None:
            off_thread_id["id"] = threading.get_ident()
            console._hands_free._console_hands_free_marshal(_callback, "x", "y")

        worker = threading.Thread(target=_from_background_thread)
        worker.start()
        worker.join(timeout=5)

        assert off_thread_id.get("id") is not None
        assert off_thread_id["id"] != console.app_instance._thread_id
        marshal_calls.assert_called_once_with(_callback, "x", "y")
        assert direct_calls == []  # never invoked in place off-thread


@pytest.mark.asyncio
async def test_hands_free_marshal_calls_directly_when_already_on_ui_thread():
    """The same-thread half of I1's fix: the async, on-the-app-loop
    direct-provider send path calls the tap from the UI thread already --
    the callback must run immediately there, with no `call_from_thread`
    round trip."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()
        marshal_calls = Mock()
        console.app_instance.call_from_thread = marshal_calls
        direct_calls: list[tuple] = []

        console._hands_free._console_hands_free_marshal(
            lambda a, b: direct_calls.append((a, b)), "x", "y"
        )

        assert direct_calls == [("x", "y")]
        marshal_calls.assert_not_called()


@pytest.mark.asyncio
async def test_hands_free_marshal_fast_path_when_loop_is_off():
    """Task-5 review round 2, D1: the tap is installed once and never
    uninstalled, so EVERY chunk of EVERY message pays for this call
    whether hands-free is running or not. With the loop off
    (`_console_hands_free is None`), the marshal must bail out BEFORE the
    thread-identity check, let alone a real cross-thread call -- neither
    the callback nor `call_from_thread` may run at all."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        assert console._console_hands_free is None
        marshal_calls = Mock()
        console.app_instance.call_from_thread = marshal_calls
        direct_calls: list[tuple] = []

        console._hands_free._console_hands_free_marshal(
            lambda a, b: direct_calls.append((a, b)), "x", "y"
        )

        assert direct_calls == []
        marshal_calls.assert_not_called()


@pytest.mark.asyncio
async def test_hands_free_marshal_swallows_call_from_thread_failure():
    """Task-5 review round 2, D1: `App.call_from_thread` raises
    `RuntimeError("App is not running")` when the app has no running event
    loop -- reachable from a real worker thread whenever `app_instance`
    is not (yet, or no longer) the running app, e.g. the standard test
    harness. The tap sits INSIDE `store.append_stream_chunk`/`mark_
    message_*`, which every reply streams through -- a marshal failure
    must never escape into the caller."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()

        def _raise_not_running(callback, *args):
            raise RuntimeError("App is not running")

        console.app_instance.call_from_thread = _raise_not_running
        raised: list[BaseException] = []

        def _from_background_thread() -> None:
            try:
                console._hands_free._console_hands_free_marshal(lambda: None)
            except BaseException as exc:  # noqa: BLE001 - this IS the pin
                raised.append(exc)

        worker = threading.Thread(target=_from_background_thread)
        worker.start()
        worker.join(timeout=5)

        assert raised == []


@pytest.mark.asyncio
async def test_hands_free_marshal_swallows_a_raising_callback_on_the_ui_thread():
    """Task-5 final review I2: the docstring claims a hands-free plumbing
    failure can NEVER escape into `store.append_message`/`append_stream_
    chunk`/`mark_message_*` -- but only the OFF-thread branch was wrapped;
    the UI-thread branch (the whole direct-provider, non-agent-runtime
    send path -- a supported configuration, not hypothetical) called
    `callback(*args)` bare. A raising callback on THAT branch must be
    swallowed too, making the "NEVER" claim actually true."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()

        def _raising_callback() -> None:
            raise RuntimeError("tap callback blew up")

        raised: list[BaseException] = []
        try:
            # Called directly, on the UI thread (the test's own async
            # body -- no background thread involved), exercising the
            # UI-thread branch specifically.
            console._hands_free._console_hands_free_marshal(_raising_callback)
        except BaseException as exc:  # noqa: BLE001 - this IS the pin
            raised.append(exc)

        assert raised == []


@pytest.mark.asyncio
async def test_append_stream_chunk_never_raises_off_thread_even_when_call_from_thread_fails():
    """The end-to-end shape of D1's exception concern: a background-thread
    call into the WRAPPED `store.append_stream_chunk` (the exact seam the
    tap wraps -- not just the marshal helper in isolation) must return
    normally even when `call_from_thread` fails, and the underlying store
    write must still have happened (the tap is read-only)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()
        store = console._ensure_console_chat_store()
        message = store.append_message(
            store.active_session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        )

        def _raise_not_running(callback, *args):
            raise RuntimeError("App is not running")

        console.app_instance.call_from_thread = _raise_not_running
        raised: list[BaseException] = []

        def _from_background_thread() -> None:
            try:
                store.append_stream_chunk(message.id, "hello")
            except BaseException as exc:  # noqa: BLE001 - this IS the pin
                raised.append(exc)

        worker = threading.Thread(target=_from_background_thread)
        worker.start()
        worker.join(timeout=5)

        assert raised == []
        assert "hello" in store.get_message(message.id).content


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
        dictation_module.ConsoleDictationController,
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
        await _wait_for_mic_label(composer, pilot, "Dictating")
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
        dictation_module.ConsoleDictationController,
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
        await _wait_for_mic_label(composer, pilot, "Dictating")
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
# Reply identity (binding carrier) -- task-5 review B1/M7
# ---------------------------------------------------------------------------


def _prime_hands_free_send(
    console, session, *, existing_assistant_ids: frozenset[str] = frozenset()
) -> str:
    """Set up `session` as if `_console_hands_free_request_stop_and_send`
    had just recorded a real send into the CURRENTLY active session, then
    force the controller into `awaiting_reply` -- without a real dictation
    stop or a real network send. Returns the sending session's id.
    """
    store = console._ensure_console_chat_store()
    sending_session_id = store.active_session_id
    console._console_dictation_origin_session_id = sending_session_id
    session.pending_session_id = sending_session_id
    session.pending_existing_assistant_ids = existing_assistant_ids
    session.controller._begin_awaiting_reply()
    assert session.controller.state == "awaiting_reply"
    return sending_session_id


@pytest.mark.asyncio
async def test_reply_identity_same_session_new_id_claims_and_feeds():
    """Baseline happy path: a brand-new assistant id, in the recorded
    sending session, claims and feeds; a later delta for any OTHER id is
    dropped; the same real id keeps feeding."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()
        session = console._console_hands_free
        assert session is not None
        sending_session_id = _prime_hands_free_send(console, session)
        store = console._ensure_console_chat_store()
        real = store.append_message(
            sending_session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        )

        fed: list[str] = []
        session.sequencer.feed = fed.append

        console._hands_free._on_console_hands_free_delta(real.id, "first chunk. ")
        assert session.reply_id == real.id
        assert fed == ["first chunk. "]

        console._hands_free._on_console_hands_free_delta("stale-id", "must not feed")
        assert fed == ["first chunk. "]

        console._hands_free._on_console_hands_free_delta(real.id, "second chunk. ")
        assert fed == ["first chunk. ", "second chunk. "]

        finished: list[Any] = []
        session.sequencer.reply_completed = lambda: finished.append("sequencer")
        session.controller.on_reply_finished = lambda: finished.append("controller")

        console._hands_free._on_console_hands_free_terminal("stale-id", False)
        assert finished == []

        console._hands_free._on_console_hands_free_terminal(real.id, False)
        assert finished == ["sequencer", "controller"]


@pytest.mark.asyncio
async def test_reply_identity_rejects_a_concurrent_background_session_reply():
    """Task-5 review B1 probe 1 (cross-session): parallel per-session runs
    are a first-class supported feature -- a DIFFERENT session's reply
    streaming concurrently must never be claimed, spoken, or completed as
    THIS turn's reply; the real, same-session reply must still work."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        sending_session_id = store.active_session_id

        # A second, background session -- `create_session` activates it,
        # so switch back to the sending session afterward.
        background = store.create_session()
        background_reply = store.append_message(
            background.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        store.switch_session(sending_session_id)

        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()
        session = console._console_hands_free
        _prime_hands_free_send(console, session)

        fed: list[str] = []
        session.sequencer.feed = fed.append

        # The background session's reply id must not claim the slot.
        console._hands_free._on_console_hands_free_delta(
            background_reply.id, "Background tab reply here."
        )
        assert session.reply_id is None
        assert fed == []

        # The turn's OWN reply, once it actually streams, is claimed fine.
        real = store.append_message(
            sending_session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        console._hands_free._on_console_hands_free_delta(real.id, "Real reply here.")
        assert session.reply_id == real.id
        assert fed == ["Real reply here."]

        # The background reply's own completion must not resolve THIS turn.
        finished: list[Any] = []
        session.controller.on_reply_finished = lambda: finished.append("controller")
        console._hands_free._on_console_hands_free_terminal(background_reply.id, False)
        assert finished == []
        console._hands_free._on_console_hands_free_terminal(real.id, False)
        assert finished == ["controller"]


@pytest.mark.asyncio
async def test_reply_identity_rejects_a_stale_same_session_reply():
    """Task-5 review B1 probe 2 (same session, no concurrency needed):
    keyboard barge-in during `awaiting_reply` suppresses speech but never
    cancels generation, so the suppressed reply's own message id already
    exists (and keeps streaming) by the time the NEXT turn's send fires.
    That id must never be re-claimed by the new turn."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()
        session = console._console_hands_free
        store = console._ensure_console_chat_store()
        sending_session_id = store.active_session_id

        # Turn 1: a reply starts, speaks one sentence, then the user
        # barges in with a keypress -- suppressed, but generation (and the
        # store append_stream_chunk tap calls) keeps going regardless.
        _prime_hands_free_send(console, session)
        old_reply = store.append_message(
            sending_session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        spoken: list[str] = []
        session.sequencer.feed = lambda text: spoken.append(text)
        console._hands_free._on_console_hands_free_delta(old_reply.id, "Old reply first sentence.")
        assert session.reply_id == old_reply.id
        assert spoken == ["Old reply first sentence."]

        session.controller.on_composer_key()  # barge-in during awaiting_reply
        assert session.controller.state == "listening"

        # Turn 2: a fresh send. The pending-send snapshot now includes the
        # OLD reply's id (it already existed in the store by this point).
        _prime_hands_free_send(
            console, session, existing_assistant_ids=frozenset({old_reply.id})
        )
        assert session.reply_id is None  # begin_reply() cleared the claim

        # The OLD reply's generation is still streaming its next sentence
        # -- this must NOT be claimed as turn 2's reply.
        console._hands_free._on_console_hands_free_delta(old_reply.id, "Old reply second sentence.")
        assert session.reply_id is None
        assert spoken == ["Old reply first sentence."]

        # Turn 2's OWN new reply claims correctly.
        new_reply = store.append_message(
            sending_session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        console._hands_free._on_console_hands_free_delta(new_reply.id, "New reply sentence.")
        assert session.reply_id == new_reply.id
        assert spoken == ["Old reply first sentence.", "New reply sentence."]


@pytest.mark.asyncio
async def test_awaiting_reply_watchdog_disarms_at_row_creation_not_first_token():
    """Task-5 final review I3: the `awaiting_reply` watchdog's own
    docstring says it guards only the send -> `on_reply_started()` gap,
    never generation already in progress -- but before this fix the ONLY
    `on_reply_started()` call sites were the first streamed delta and the
    terminal tap, both downstream of the model producing VISIBLE output.
    A reply that tool-round-trips first, opens with a fenced code block
    the sequencer skips by design, or is a sealed/non-streaming turn could
    blow `AWAITING_REPLY_DEADLINE_SECONDS` before a single visible token
    ever arrived, even though generation had started immediately -- the
    FSM's own docstring calls that "routine," not exceptional.

    `store.append_message`'s ASSISTANT-role tap (`_on_console_hands_free_
    assistant_row_created`) now fires `on_reply_started()` at row
    CREATION -- synchronously, before any streaming, on both the agent-
    runtime and direct-provider paths -- so the watchdog disarms long
    before the first delta. This pins that: the watchdog must already be
    disarmed the instant the (real, wrapped) `append_message` call
    returns, well past the OLD 30s mark with zero visible content must
    NOT abandon the reply, and the eventually-late first token must still
    feed and complete the turn normally.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()
        session = console._console_hands_free
        sending_session_id = _prime_hands_free_send(console, session)
        assert session.controller._awaiting_watchdog_disarmed is False

        # The real send path appends the assistant row up front, before
        # any streaming begins -- going through the REAL wrapped `append_
        # message` (not calling the handler directly) exercises the
        # actual tap installed by `_install_console_hands_free_store_tap`.
        store = console._ensure_console_chat_store()
        real = store.append_message(
            sending_session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        await pilot.pause()
        assert session.reply_id == real.id
        assert session.controller._awaiting_watchdog_disarmed is True

        # The model takes its time producing anything VISIBLE -- a tool
        # round-trip, a fenced-code opener the sequencer skips, a sealed
        # turn -- well past the OLD 30s deadline, with zero deltas
        # delivered yet. A real `tick()` call (the wiring's own 0.1s
        # timer would eventually make one) must be a no-op now: the
        # watchdog is disarmed, so elapsed time no longer matters.
        session.controller.tick(time.monotonic() + 120.0)
        assert session.controller.state == "awaiting_reply"
        assert session.controller._reply_abandoned_by_watchdog is False

        # The first VISIBLE token, arriving "late", still feeds and
        # completes the turn normally -- nothing was abandoned.
        fed: list[str] = []
        session.sequencer.feed = fed.append
        console._hands_free._on_console_hands_free_delta(real.id, "Finally, some text. ")
        assert fed == ["Finally, some text. "]

        finished: list[Any] = []
        session.sequencer.reply_completed = lambda: finished.append("sequencer")
        session.controller.on_reply_finished = lambda: finished.append("controller")
        console._hands_free._on_console_hands_free_terminal(real.id, False)
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
        console._enter_console_hands_free_loop(capture_live=True)
        await pilot.pause()
        session = console._console_hands_free
        sending_session_id = _prime_hands_free_send(console, session)
        store = console._ensure_console_chat_store()
        real = store.append_message(
            sending_session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        )

        finished: list[Any] = []
        session.sequencer.reply_completed = lambda: finished.append("sequencer")
        session.controller.on_reply_finished = lambda: finished.append("controller")

        console._hands_free._on_console_hands_free_terminal(real.id, False)

        assert session.reply_id == real.id
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
        dictation_module.ConsoleDictationController,
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
        await _wait_for_mic_label(composer, pilot, "Dictating")
        session = console._console_hands_free

        # Drive straight to `speaking`, stubbing `RequestStopAndSend`'s own
        # wiring effect (a real V2 send) -- this test targets ONLY the
        # `ExitLoop` handler's own teardown behavior.
        console._hands_free._console_hands_free_request_stop_and_send = lambda: None
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

        console._hands_free._handle_console_hands_free_intent(ExitLoop())

        assert session.sequencer._inflight is False
        posted_actions = [
            getattr(call.args[0], "action", None) for call in post_message.call_args_list
        ]
        assert "stop" in posted_actions
        assert console._console_hands_free is None
        await _wait_for_mic_label(composer, pilot, "Dictate")
        assert fake.stop_calls == 1


@pytest.mark.asyncio
async def test_open_and_close_capture_handlers_are_idempotent_no_ops(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
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
        console._hands_free._console_hands_free_open_capture()
        await _wait_for_mic_label(composer, pilot, "Dictating")
        console._hands_free._console_hands_free_open_capture()
        await pilot.pause()
        assert fake.start_calls == 1

        # CloseCapture is a no-op unless genuinely `recording`.
        console._console_dictation_state = "idle"
        console._hands_free._console_hands_free_close_capture()
        await pilot.pause()
        assert fake.stop_calls == 0


# ---------------------------------------------------------------------------
# Empty-capture service limit: real reopen, then a real exit (task-5
# review B2)
# ---------------------------------------------------------------------------


async def _wait_for(condition, pilot, *, timeout: float = _ASYNC_SETTLE_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"condition never became true: {condition!r}")


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_hands_free_limit_exits_without_reopen_until_a_physical_mic_press(
    monkeypatch,
):
    """A bounded ending inserts normally and requires an explicit resume."""
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
        await _wait_for_mic_label(composer, pilot, "Dictating")
        console.action_toggle_console_hands_free()
        await pilot.pause()
        assert console._console_hands_free is not None
        assert service.start_calls == 1

        service.emit_final("retained bounded text")
        await pilot.pause()
        console._dictation._handle_console_dictation_limit()
        await _wait_for(lambda: console._console_hands_free is None, pilot)
        await _wait_for_mic_label(composer, pilot, "Dictate")

        await pilot.pause(0.2)
        assert composer.draft_text() == "retained bounded text"
        assert service.start_calls == 1

        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictating")
        assert service.start_calls == 2
        service.emit_partial("resumed physically")
        await _wait_for(
            lambda: console._console_dictation_partial == "resumed physically",
            pilot,
        )
        service.emit_final("resumed physically")
        await _wait_for(lambda: console._console_dictation_partial == "", pilot)
        await pilot.pause()
        await pilot.click("#console-dictation")
        await _wait_for_mic_label(composer, pilot, "Dictate")


@pytest.mark.asyncio
async def test_hands_free_limit_never_auto_sends_the_retained_text(
    monkeypatch,
):
    """Limit recovery follows ordinary insertion, never the send path."""
    service = FakeDictationService()
    _patch_availability(monkeypatch)
    _install_streaming_session(monkeypatch, service)
    _fast_countdown(monkeypatch, seconds=0.2)
    gateway = _HandsFreeReplyGateway("Limit triggered reply. ")
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
        await _wait_for_mic_label(composer, pilot, "Dictating")
        console.action_toggle_console_hands_free()
        await pilot.pause()

        service.emit_final("dictated before the limit hit")
        await _wait_for(
            lambda: (
                console._console_hands_free is not None
                and console._console_hands_free.controller.state == "countdown"
            ),
            pilot,
        )
        console._dictation._handle_console_dictation_limit()

        await _wait_for(lambda: console._console_hands_free is None, pilot)
        await _wait_for_mic_label(composer, pilot, "Dictate")
        await pilot.pause(0.3)

        assert composer.draft_text() == "dictated before the limit hit"
        assert gateway.sent_messages == []
        assert not any("Limit triggered reply" in text for text, _q in tts.calls)
        assert service.start_calls == 1


@pytest.mark.asyncio
async def test_deferred_capture_ended_is_dropped_for_a_replaced_loop():
    """Task-5 review round 2, D4: if the user exits and re-enters hands-free
    WHILE a limit-triggered `on_capture_ended` delivery is still waiting
    for `idle`, the stale delivery must never reach the NEW loop -- it
    would silently burn the new loop's own one-time reopen ceiling for an
    ending that has nothing to do with it.

    Drives `_deliver_console_hands_free_capture_ended` directly with full
    control over the `idle` transition's timing, rather than racing the
    method's own real 0.05s poll loop against the rest of the test body
    (a first version of this test raced that poll and was not reliably
    mutation-sensitive -- caught by running the required mutation check
    before trusting it)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)

        session_a = chat_screen_module.ConsoleHandsFreeSession(
            controller=HandsFreeController(emit=lambda intent: None),
            sequencer=object(),
        )
        session_b = chat_screen_module.ConsoleHandsFreeSession(
            controller=HandsFreeController(emit=lambda intent: None),
            sequencer=object(),
        )
        b_calls: list[dict] = []
        session_b.controller.on_capture_ended = lambda **kw: b_calls.append(kw)

        console._console_hands_free = session_a
        console._console_dictation_state = "recording"  # not idle -- blocks the poll

        task = asyncio.create_task(
            console._hands_free._deliver_console_hands_free_capture_ended(session_a, False)
        )
        await asyncio.sleep(0)  # let the poll loop start and observe "not idle"

        # Exit loop A, enter loop B -- WHILE the delivery is still waiting.
        console._console_hands_free = session_b

        console._console_dictation_state = "idle"  # unblocks the poll
        await asyncio.wait_for(task, timeout=5)

        assert b_calls == []


# ---------------------------------------------------------------------------
# Acoustic barge-in reopens capture during `speaking`
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acoustic_barge_in_opens_capture_on_reply_started(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: fake,
    )
    monkeypatch.setattr(
        hands_free_module, "acoustic_barge_in_enabled", lambda: True
    )
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one(
            "#console-native-composer", chat_screen_module.ConsoleComposerBar
        )

        console.action_toggle_console_hands_free()
        await _wait_for_mic_label(composer, pilot, "Dictating")
        session = console._console_hands_free
        assert session.controller._acoustic_barge_in is True

        # `RequestStopAndSend`'s own wiring effect is stubbed -- this test
        # targets ONLY acoustic mode's reopen-on-`on_reply_started` effect,
        # not a real V2 send. `CloseCapture` (real) still closes the mic.
        console._hands_free._console_hands_free_request_stop_and_send = lambda: None
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
        await _wait_for_mic_label(composer, pilot, "Dictating")


# ---------------------------------------------------------------------------
# VAD-degraded honesty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vad_degraded_entry_warns_instead_of_promising_auto_send(monkeypatch):
    fake = FakeDictationSession()
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
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
