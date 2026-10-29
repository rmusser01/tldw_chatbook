"""Console header Speak replies / Hands-free controls say their state.

TASK-33626.1. Live on textual-dark the header row read
``Speak replies ▌  Hands-free ▌  Ready`` for both Off and On: each control was
a Textual Switch squeezed to one row, its Off track 1.42:1 on the header and
its knob 1.22:1. Focus replaced the knob with the app's ``*:focus`` outline
(``┌───┐``), so focused-Off and focused-On painted identically. Pressing Enter
on Hands-free started dictation, audio and the speech-model load at once,
with no consent, although Hands-free speaks replies through the same TTS
provider Speak replies asks about. Esc ended the loop but not the model
preparation: the chip still read "Preparing speech model…" 80 s later.

Every test here mounts the real ChatScreen with the production stylesheets
(ConsoleHarness) and uses real key routing and the real dictation state
machine. The dictation *session* is a fake: it stands in for the microphone
and the speech model, and nothing on the consent or cancel path is stubbed.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager

import pytest
from rich.cells import cell_len
from textual.color import Color
from textual.widget import Widget
from textual.widgets import Button

from Tests.UI.hands_free_consent_support import accept_hands_free_consent
from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Chat import console_voice_input as voice_module
from tldw_chatbook.css.Themes.themes import _contrast_ratio
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import ConsoleTTSDestination
from tldw_chatbook.UI.Console_Modules import dictation as dictation_module
from tldw_chatbook.UI.Console_Modules import hands_free as hands_free_module

#: These tests mount the real ChatScreen (Tests/conftest.py keeps the
#: collection-time profile for them; the per-test redirect fails admission).
pytestmark = pytest.mark.bootstrap_profile

AA_TEXT = 4.5
NON_TEXT = 3.0
SIZES = ((235, 52), (80, 24))
#: Literal ids, so the PR-gate census can name one case (scripts/check_ui_pr_gate_census.py).
SIZE_IDS = ("235x52", "80x24")
THEMES = ("textual-dark", "textual-light")

DESTINATION = ConsoleTTSDestination(
    fingerprint="sha256:" + "b" * 64,
    provider_label="Test Voice Cloud",
    sanitized_destination="https://voice.example",
    charges_may_apply=True,
)


class _BlockingStartSession:
    """A dictation session whose start blocks like a first-run model load."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.discard_calls = 0
        self.stop_calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.retry_available = False

    def start(self, *, on_buffer_limit=None) -> None:
        del on_buffer_limit
        self.start_calls += 1
        self.started.set()
        # Bounded so a broken cancel path fails the test instead of hanging.
        self.release.wait(timeout=10)

    def stop_and_transcribe(self) -> str:
        self.stop_calls += 1
        return ""

    def discard(self) -> None:
        self.discard_calls += 1

    def clear_retry(self) -> None:
        return None


def _install_tts_destination(app) -> None:
    """Give the app the TTS destination Speak replies would consent to."""

    class _Handler:
        async def resolve_console_speech_destination(self, *_args, **_kwargs):
            return DESTINATION

    async def _ensure_tts_handler():
        return _Handler()

    app._ensure_tts_handler = _ensure_tts_handler


def _row_text(screen, widget: Widget) -> str:
    """The plain text the compositor paints across ``widget``'s row."""
    strip = screen._compositor.render_strips()[widget.region.y]
    return "".join(segment.text for segment in strip)


def _cells(screen, y: int) -> list[tuple[str, Color | None, Color | None]]:
    """(glyph, ink, fill) for every cell of painted row ``y``."""
    cells: list[tuple[str, Color | None, Color | None]] = []
    for segment in screen._compositor.render_strips()[y]:
        style = segment.style
        ink = Color.from_rich_color(style.color) if style and style.color else None
        fill = Color.from_rich_color(style.bgcolor) if style and style.bgcolor else None
        if style is not None and style.reverse:
            ink, fill = fill, ink
        for glyph in segment.text:
            cells.append((glyph, ink, fill))
            if cell_len(glyph) == 2:
                cells.append(("", ink, fill))
    return cells


def _control_cells(screen, widget: Widget):
    row = _cells(screen, widget.region.y)
    return row[widget.region.x : widget.region.right]


def _first_glyph(cells):
    return next(cell for cell in cells if cell[0].strip())


async def _wait_for(pilot, predicate, *, timeout: float = 6.0, step: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(step)
    return predicate()


async def _console(host, pilot, *, theme: str = "textual-dark"):
    console = await _mounted_console(host, pilot)
    host.theme = theme
    await _wait_for(pilot, lambda: console.query("#console-hands-free-switch"))
    await pilot.pause()
    return console


def _header_controls(console):
    return (
        console.query_one("#console-speech-controls"),
        console.query_one("#console-auto-speak"),
        console.query_one("#console-hands-free-switch"),
    )


# ---------------------------------------------------------------------------
# AC#1 / AC#5: the state is a word, and it reads on the header
# ---------------------------------------------------------------------------


async def _turn_hands_free_on(console, host, pilot, session) -> None:
    """Hands-free On through the real path, pausing at "Starting…".

    The user's gesture (`request_console_hands_free_toggle`, what the control
    and Ctrl+Shift+H call), the consent's real Enable button, then the model
    load: the loop's capture reads `starting` until ``session.release`` is
    set, and the header reads from that state, not from a pushed value.
    """
    console._hands_free.request_console_hands_free_toggle()
    await accept_hands_free_consent(host, pilot)
    assert await _wait_for(pilot, lambda: session.started.is_set())
    assert console._console_hands_free is not None
    assert console._console_dictation_state == "starting"


async def _finish_model_load(console, pilot, session) -> None:
    session.release.set()
    assert await _wait_for(
        pilot, lambda: console._console_dictation_state == "recording"
    ), console._console_dictation_state
    await pilot.pause()


def _word_contrast(console, widget: Widget, theme: str, state: str) -> None:
    """The state word reads at AA against the header it is painted on."""
    glyph, ink, fill = _first_glyph(_control_cells(console, widget))
    ratio = _contrast_ratio(ink, fill)
    assert ratio >= AA_TEXT, (theme, widget.id, state, glyph, ink, fill, ratio)


@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("theme", THEMES)
async def test_header_prints_each_state_as_a_readable_word(
    monkeypatch, theme, size
) -> None:
    speech = _patch_speech(monkeypatch)
    app, host = _ready_host()
    _install_tts_destination(app)
    async with _running(host, speech, size=size) as pilot:
        console = await _console(host, pilot, theme=theme)
        controls, auto_speak, hands_free = _header_controls(console)
        coordinator = console._console_auto_speak
        store = coordinator._store_accessor()
        session = coordinator._active_session()
        assert session is not None

        off = _row_text(console, controls)
        assert "Speak replies: Off" in off and "Hands-free: Off" in off, off
        _word_contrast(console, auto_speak, theme, "off")
        _word_contrast(console, hands_free, theme, "off")

        # Speak replies' state comes from the conversation's persisted
        # preference, through the real coordinator (a later re-sync of the
        # active conversation repaints the same state, not Off).
        store.set_auto_speak(session.id, True)
        coordinator.sync_controls()
        # Hands-free's state comes from the loop and its capture.
        await _turn_hands_free_on(console, host, pilot, speech)
        await pilot.pause()
        assert "Hands-free: Starting…" in _row_text(console, controls)
        _word_contrast(console, hands_free, theme, "starting")

        await _finish_model_load(console, pilot, speech)
        on = _row_text(console, controls)
        assert "Speak replies: On" in on and "Hands-free: On" in on, on
        assert on != off
        _word_contrast(console, auto_speak, theme, "on")
        _word_contrast(console, hands_free, theme, "on")

        # AC#5: an auto-speak failure pauses it persistently; the control
        # itself says so instead of staying On.
        store.pause_auto_speak(session.id)
        coordinator.sync_controls()
        await pilot.pause()
        assert "Speak replies: Paused" in _row_text(console, controls)
        _word_contrast(console, auto_speak, theme, "paused")


# ---------------------------------------------------------------------------
# AC#2: focus keeps the value visible and is itself a 3:1 cue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theme", THEMES)
async def test_focused_control_keeps_its_value_and_shows_a_3_to_1_cue(
    monkeypatch, theme
) -> None:
    speech = _patch_speech(monkeypatch)
    app, host = _ready_host()
    _install_tts_destination(app)
    async with _running(host, speech, size=(235, 52)) as pilot:
        console = await _console(host, pilot, theme=theme)
        _controls, _auto_speak, hands_free = _header_controls(console)
        resting = _control_cells(console, hands_free)

        hands_free.focus()
        await pilot.pause()
        assert console.app.focused is hands_free
        focused_off = _control_cells(console, hands_free)
        await _turn_hands_free_on(console, host, pilot, speech)
        await _finish_model_load(console, pilot, speech)
        hands_free.focus()
        await pilot.pause()
        assert console.app.focused is hands_free
        focused_on = _control_cells(console, hands_free)

        off_text = "".join(cell[0] for cell in focused_off)
        on_text = "".join(cell[0] for cell in focused_on)
        assert "Off" in off_text and "On" in on_text, (off_text, on_text)
        assert "┌" not in off_text + on_text, "the focus outline hides the value"
        assert focused_off != focused_on

        # The cue: the focused fill against the header beside the control.
        header_fill = resting[0][2]
        _glyph, ink, focus_fill = _first_glyph(focused_off)
        assert focus_fill is not None and header_fill is not None
        assert _contrast_ratio(focus_fill, header_fill) >= NON_TEXT, (
            theme,
            focus_fill,
            header_fill,
        )
        assert _contrast_ratio(ink, focus_fill) >= AA_TEXT, (theme, ink, focus_fill)


# ---------------------------------------------------------------------------
# AC#3 / AC#6: consent before any speech pipeline starts
# ---------------------------------------------------------------------------


def _patch_speech(monkeypatch) -> _BlockingStartSession:
    session = _BlockingStartSession()
    monkeypatch.setattr(voice_module, "probe", lambda: voice_module.Availability(ok=True))
    monkeypatch.setattr(hands_free_module, "speculative_voice_qualified", lambda: False)
    monkeypatch.setattr(
        dictation_module.ConsoleDictationController,
        "_create_console_dictation_session",
        lambda self: session,
    )
    return session


@asynccontextmanager
async def _running(host, session: _BlockingStartSession, size=(160, 48)):
    """Run ``host``; whatever happens, release the blocked model load.

    A failed assertion (the pre-fix code never asks for consent) must not
    leave the start worker blocked for the app's teardown to time out on.
    """
    async with host.run_test(size=size) as pilot:
        try:
            yield pilot
        finally:
            session.release.set()


def _top(host) -> str:
    return type(host.screen_stack[-1]).__name__


async def _press_enter_on_hands_free(console, pilot) -> None:
    toggle = console.query_one("#console-hands-free-switch")
    toggle.focus()
    await pilot.pause()
    assert console.app.focused is toggle
    await pilot.press("enter")


async def test_enter_on_hands_free_asks_consent_and_declining_starts_nothing(
    monkeypatch,
) -> None:
    session = _patch_speech(monkeypatch)
    app, host = _ready_host()
    _install_tts_destination(app)
    async with _running(host, session) as pilot:
        console = await _console(host, pilot)
        await _press_enter_on_hands_free(console, pilot)

        assert await _wait_for(pilot, lambda: _top(host) == "AutoSpeakConsentModal"), (
            "Hands-free started without asking: "
            f"dictation={console._console_dictation_state!r} "
            f"session_starts={session.start_calls}"
        )
        modal = host.screen_stack[-1]
        copy = " ".join(
            str(static.render()) for static in modal.query("Static")
        )
        assert "Provider: Test Voice Cloud" in copy
        assert "Destination: https://voice.example" in copy
        assert "Provider charges may apply." in copy
        # Nothing of the speech pipeline has started behind the modal.
        assert session.start_calls == 0
        assert console._console_dictation_state == "idle"
        assert console._console_hands_free is None

        await pilot.press("escape")
        assert await _wait_for(pilot, lambda: _top(host) == "ChatScreen")
        await pilot.pause(0.3)
        assert session.start_calls == 0
        assert console._console_dictation_state == "idle"
        assert console._console_hands_free is None
        controls = console.query_one("#console-speech-controls")
        assert "Hands-free: Off" in _row_text(console, controls)


def _modal_copy(modal) -> str:
    return " ".join(str(static.render()) for static in modal.query("Static"))


async def _open_hands_free_consent(console, host, pilot):
    await _press_enter_on_hands_free(console, pilot)
    assert await _wait_for(pilot, lambda: _top(host) == "AutoSpeakConsentModal")
    modal = host.screen_stack[-1]
    assert modal.hands_free is True
    return modal


async def _decline(console, host, pilot, session) -> None:
    await pilot.press("escape")
    assert await _wait_for(pilot, lambda: _top(host) == "ChatScreen")
    await pilot.pause(0.2)
    assert session.start_calls == 0
    assert console._console_hands_free is None
    assert console._console_realtime is None


async def test_hands_free_consent_without_a_tts_destination_says_replies_go_unspoken(
    monkeypatch,
) -> None:
    """Review follow-up: the no-destination variant had no rendered case."""
    session = _patch_speech(monkeypatch)
    app, host = _ready_host()

    class _NoDestination:
        async def resolve_console_speech_destination(self, *_args, **_kwargs):
            return None

    async def _ensure_tts_handler():
        return _NoDestination()

    app._ensure_tts_handler = _ensure_tts_handler
    async with _running(host, session) as pilot:
        console = await _console(host, pilot)
        copy = _modal_copy(await _open_hands_free_consent(console, host, pilot))
        assert "Turn on Hands-free?" in copy
        assert (
            "Replies will not be spoken: no TTS destination is available." in copy
        ), copy
        assert "Provider:" not in copy and "Destination:" not in copy, copy
        assert "Provider charges may apply." not in copy, copy
        await _decline(console, host, pilot, session)


async def test_hands_free_consent_for_the_realtime_engine_names_that_provider(
    monkeypatch,
) -> None:
    """Review follow-up: with the realtime engine, replies come from the
    realtime provider, so the consent names it and shows no TTS destination
    even when one is configured."""
    session = _patch_speech(monkeypatch)
    monkeypatch.setattr(hands_free_module, "resolve_handsfree_engine", lambda: "realtime")
    monkeypatch.setattr(hands_free_module, "realtime_enabled", lambda: True)
    monkeypatch.setattr(voice_module, "realtime_provider", lambda config=None: "openai")
    monkeypatch.setattr(voice_module, "realtime_model", lambda config=None: "gpt-realtime")
    app, host = _ready_host()
    _install_tts_destination(app)
    async with _running(host, session) as pilot:
        console = await _console(host, pilot)
        copy = _modal_copy(await _open_hands_free_consent(console, host, pilot))
        assert "Provider: openai realtime voice (gpt-realtime)" in copy, copy
        assert "Destination:" not in copy, copy
        assert "Test Voice Cloud" not in copy, copy
        assert "Provider charges may apply." in copy, copy
        await _decline(console, host, pilot, session)


async def test_hands_free_stays_off_when_its_consent_cannot_be_asked(
    monkeypatch,
) -> None:
    """While a Speak replies enable is still resolving its destination the
    coordinator cannot raise a second question. Hands-free stays Off, starts
    nothing, and its notice does not claim a question is open (none is yet:
    review follow-up on the old "Close the open question" copy)."""
    session = _patch_speech(monkeypatch)
    app, host = _ready_host()
    resolve = asyncio.Event()

    class _SlowHandler:
        async def resolve_console_speech_destination(self, *_args, **_kwargs):
            await resolve.wait()
            return DESTINATION

    async def _ensure_tts_handler():
        return _SlowHandler()

    app._ensure_tts_handler = _ensure_tts_handler
    notices: list[str] = []
    notify = app.notify

    def _record_notice(message, *args, **kwargs):
        notices.append(str(message))
        return notify(message, *args, **kwargs)

    monkeypatch.setattr(app, "notify", _record_notice)
    async with _running(host, session) as pilot:
        console = await _console(host, pilot)
        controls, auto_speak, _hands_free = _header_controls(console)
        auto_speak.focus()
        await pilot.pause()
        await pilot.press("enter")
        coordinator = console._console_auto_speak
        assert await _wait_for(
            pilot, lambda: coordinator._enable_operation_generation is not None
        )

        await _press_enter_on_hands_free(console, pilot)
        notice = "Hands-free couldn't ask for the speech consent just now"
        assert await _wait_for(
            pilot, lambda: any(notice in message for message in notices)
        ), notices
        assert not any("open question" in message for message in notices)
        assert _top(host) == "ChatScreen"
        assert session.start_calls == 0
        assert console._console_dictation_state == "idle"
        assert console._console_hands_free is None
        assert "Hands-free: Off" in _row_text(console, controls)

        # The Speak replies question it was waiting on still opens.
        resolve.set()
        assert await _wait_for(pilot, lambda: _top(host) == "AutoSpeakConsentModal")
        assert host.screen_stack[-1].hands_free is False
        await pilot.press("escape")
        assert await _wait_for(pilot, lambda: _top(host) == "ChatScreen")


async def _enter_hands_free_with_consent(console, host, pilot, session) -> None:
    await _press_enter_on_hands_free(console, pilot)
    assert await _wait_for(pilot, lambda: _top(host) == "AutoSpeakConsentModal")
    await pilot.click("#console-auto-speak-consent-confirm")
    assert await _wait_for(pilot, lambda: session.started.is_set()), (
        "accepting consent did not start the capture"
    )
    assert await _wait_for(pilot, lambda: _top(host) == "ChatScreen")
    assert console._console_dictation_state == "starting"
    assert console._console_hands_free is not None


async def _exit_by_escape(console, pilot) -> None:
    await pilot.press("escape")


async def _exit_by_toggle(console, pilot) -> None:
    await _press_enter_on_hands_free(console, pilot)


async def _exit_by_shortcut(console, pilot) -> None:
    await pilot.press("ctrl+shift+h")


@pytest.mark.parametrize(
    "exit_hands_free",
    [_exit_by_escape, _exit_by_toggle, _exit_by_shortcut],
    ids=["escape", "toggle", "ctrl+shift+h"],
)
async def test_exit_during_model_preparation_cancels_and_a_late_start_never_records(
    monkeypatch, exit_hands_free
) -> None:
    session = _patch_speech(monkeypatch)
    app, host = _ready_host()
    _install_tts_destination(app)
    async with _running(host, session) as pilot:
        console = await _console(host, pilot)
        await _enter_hands_free_with_consent(console, host, pilot, session)
        controls = console.query_one("#console-speech-controls")
        assert "Hands-free: Starting…" in _row_text(console, controls)
        # AC#4: the preparation shows a visible way out, not a tooltip.
        mic = console.query_one("#console-dictation", Button)
        assert str(mic.label) == "Cancel", str(mic.label)

        await exit_hands_free(console, pilot)
        assert await _wait_for(pilot, lambda: console._console_hands_free is None)
        assert await _wait_for(
            pilot, lambda: console._console_dictation_state == "idle"
        ), console._console_dictation_state
        composer_text = " ".join(
            str(static.render())
            for static in console.query_one("#console-native-composer").query("Static")
        )
        assert "Preparing speech model" not in composer_text

        # The model load finishes after the user left: the capture is
        # released, never promoted to recording.
        session.release.set()
        assert await _wait_for(pilot, lambda: session.discard_calls >= 1)
        await pilot.pause(0.3)
        assert console._console_dictation_state == "idle"
        assert "Hands-free: Off" in _row_text(console, controls)


# ---------------------------------------------------------------------------
# AC#5: Ctrl+Shift+H is in F1 help
# ---------------------------------------------------------------------------


async def test_f1_help_lists_the_hands_free_shortcut() -> None:
    app, host = _ready_host()
    async with host.run_test(size=(160, 48)) as pilot:
        console = await _console(host, pilot)
        console.query_one("#console-native-composer").focus()
        await pilot.press("f1")
        assert await _wait_for(pilot, lambda: _top(host) == "WorkbenchHelpPanel")
        panel = host.screen_stack[-1]
        text = " ".join(str(static.render()) for static in panel.query("Static"))
        assert "Ctrl+Shift+H" in text, text[:400]
