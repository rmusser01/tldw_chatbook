"""Console key ownership: bare letters, type-to-compose and Esc from the transcript.

Console UX review 2026-09-29, umbrella TASK-33622 ("keys reach the control
the user aimed at"), two P1 children fixed together because they share one
contract:

* TASK-33622.3 (G4-12) -- a printable key typed while a status-strip, rail or
  header control had focus was dropped, and the screen's bare ``y`` binding
  opened the full-screen Trace from any of them (the rest of the sentence then
  ran as Trace commands). Owner ruling 2026-10-27, recorded as an ADR-031
  refinement: single-letter screen actions fire only from the transcript (the
  announced list mode); elsewhere a printable key that no binding claims types
  into the composer.
* TASK-33622.4 (G4-13) -- ``ConsoleTranscript.on_key`` stopped every Esc, so the
  screen's ``escape -> focus_console_composer_home`` fallback never ran from a
  focused transcript, and prose typed there fired Edit/Copy/Regenerate.

Every case mounts the REAL ChatScreen (app stylesheet, real store, real key
routing through the Pilot). The file carries ``bootstrap_profile`` so it runs
under the plain ``pytest <file>`` command (`lessons-testing-evidence.md`:
"Tests/UI `RecoveryRequired` at setup is a profile-selection trip").
"""

from __future__ import annotations

import time

import pytest

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Console import ConsoleComposerBar, ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleTranscriptActionButton,
)

pytestmark = pytest.mark.bootstrap_profile

#: The transcript's announcement while it holds focus (TASK-33622.4 AC#3).
_TRANSCRIPT_ANNOUNCEMENT = "Transcript keys · Esc to type"


def _host() -> ConsoleHarness:
    app = _build_test_app()
    _configure_native_ready_console(app)
    return ConsoleHarness(app)


async def _wait_for(pilot, predicate, what, timeout: float = 5.0) -> None:
    """Poll ``predicate``; ``what`` may be a callable for late diagnostics."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.05)
    assert predicate(), what() if callable(what) else what


async def _type(pilot, text: str) -> None:
    """Type ``text`` key by key through the real key routing."""
    for character in text:
        await pilot.press("space" if character == " " else character)


async def _mounted(host, pilot, *, messages: bool = True):
    """Mount the Console, let its activation focus restore land, seed a chat.

    ``ChatScreen`` arms ``set_timer(0.2, _restore_console_workbench_focus)`` on
    mount; a case that moves focus before it runs can have the composer
    refocused under it (`lessons-testing-evidence.md`, TASK-33620.15.8). The
    wait is bounded on the composer taking focus, which is what the restore
    does when nothing owns focus.
    """
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    restore = console._restore_console_workbench_focus

    def restore_pending() -> bool:
        return any(
            timer._callback == restore
            and timer._task is not None
            and not timer._task.done()
            for timer in list(console._timers)
        )

    await _wait_for(pilot, lambda: not restore_pending(), "focus restore never ran")
    await pilot.pause()
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    if messages:
        store = console._ensure_console_chat_store()
        store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.USER,
            content="first question",
        )
        store.append_message(
            store.active_session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="an answer",
        )
        await console._sync_native_console_chat_ui()
        await _wait_for(
            pilot,
            lambda: len(transcript._visible_messages()) == 2,
            "the seeded messages never reached the transcript",
        )
    composer.focus()
    await _wait_for(pilot, lambda: composer.draft_has_focus, "composer never focused")
    return console, composer, transcript


def _spy_trace(console) -> list[str]:
    """Record every Trace launch (the ``y`` binding's only effect)."""
    calls: list[str] = []
    console._review_selection.open_trajectory_view = lambda: calls.append("trace")
    return calls


def _spy_transcript_actions(transcript) -> list[tuple]:
    """Record every transcript single-letter command that would run."""
    calls: list[tuple] = []
    transcript.action_invoke_selected_action = lambda action_id: calls.append(
        ("action", action_id)
    )
    transcript.action_open_review_notes = lambda: calls.append(("notes",))
    transcript.action_enter_text_selection = lambda: calls.append(("select-text",))
    return calls


async def _focus_transcript_with_f6(console, pilot, transcript) -> None:
    """Reach the transcript the way a keyboard user does: F6 between panes."""
    for _ in range(5):
        if console.focused is transcript:
            return
        await pilot.press("f6")
        await pilot.pause()
    assert console.focused is transcript, f"F6 never reached it: {console.focused!r}"


def _footer_text(console) -> str:
    return console.query_one(AppFooterStatus).shortcut_text


# ---------------------------------------------------------------------------
# TASK-33622.4: Esc from the transcript returns to the composer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_f6_j_esc_esc_returns_to_the_composer_and_prose_types_there():
    """AC#5 (and AC#1, AC#4): F6 to the transcript, j, Esc, Esc, 'look around'.

    The first Esc clears the selection, the second has nothing left to clear
    and must reach the screen's return-to-composer fallback. On the pre-fix
    code the transcript stopped every Esc: focus stayed on it, and 'look
    around' re-selected a message with 'k' and fired Regenerate with 'r'.
    """
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, transcript = await _mounted(host, pilot)
        trace = _spy_trace(console)
        actions = _spy_transcript_actions(transcript)
        stack = list(host.screen_stack)

        await _focus_transcript_with_f6(console, pilot, transcript)
        await pilot.press("j")
        await _wait_for(
            pilot,
            lambda: transcript.selected_message_id is not None,
            "j selected nothing",
        )

        await pilot.press("escape")
        await pilot.pause()
        assert transcript.selected_message_id is None
        assert console.focused is transcript, "the first Esc only clears"

        await pilot.press("escape")
        await pilot.pause()
        assert composer.draft_has_focus, (
            f"the second Esc left focus on {console.focused!r}"
        )

        await _type(pilot, "look around")
        assert composer.draft_text() == "look around"
        assert actions == []
        assert trace == []
        assert transcript.selected_message_id is None
        assert list(host.screen_stack) == stack


@pytest.mark.asyncio
async def test_esc_clears_text_selection_then_message_then_returns_to_composer():
    """AC#2: each layer is cleared, in order, before Esc returns to typing."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, transcript = await _mounted(host, pilot)
        await _focus_transcript_with_f6(console, pilot, transcript)
        await pilot.press("j")
        await _wait_for(
            pilot,
            lambda: transcript.selected_message_id is not None,
            "j selected nothing",
        )
        selected = transcript.selected_message_id
        await pilot.press("s")
        await _wait_for(
            pilot,
            lambda: transcript._kb_selection_row is not None,
            "s did not arm keyboard text selection",
        )

        await pilot.press("escape")
        await pilot.pause()
        assert transcript._kb_selection_row is None
        assert transcript.selected_message_id == selected
        assert console.focused is transcript

        await pilot.press("escape")
        await pilot.pause()
        assert transcript.selected_message_id is None
        assert console.focused is transcript

        await pilot.press("escape")
        await pilot.pause()
        assert composer.draft_has_focus, (
            f"Esc with nothing left to clear left focus on {console.focused!r}"
        )


@pytest.mark.asyncio
async def test_esc_on_a_focused_message_action_clears_then_returns_to_composer():
    """AC#2 from the action row: the first Esc clears the selection and keeps
    keyboard focus on the transcript, the next one returns to typing."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, transcript = await _mounted(host, pilot)
        await _focus_transcript_with_f6(console, pilot, transcript)
        await pilot.press("j")
        await _wait_for(
            pilot,
            lambda: bool(transcript.query(ConsoleTranscriptActionButton)),
            "the selected message never showed its action row",
        )
        action = next(
            button
            for button in transcript.query(ConsoleTranscriptActionButton)
            if not button.disabled
        )
        action.focus()
        await pilot.pause()
        assert console.focused is action

        await pilot.press("escape")
        await pilot.pause()
        assert transcript.selected_message_id is None
        assert console.focused is transcript, (
            f"Esc on an action left focus on {console.focused!r}"
        )

        await pilot.press("escape")
        await pilot.pause()
        assert composer.draft_has_focus


@pytest.mark.asyncio
async def test_a_focused_transcript_announces_its_keys_and_the_way_back():
    """AC#3: while the transcript has focus the footer leads with the
    announcement, and it goes away with the focus."""
    host = _host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer, transcript = await _mounted(host, pilot)
        assert _TRANSCRIPT_ANNOUNCEMENT not in _footer_text(console)

        await _focus_transcript_with_f6(console, pilot, transcript)
        await _wait_for(
            pilot,
            lambda: _footer_text(console).startswith(_TRANSCRIPT_ANNOUNCEMENT),
            lambda: f"footer while the transcript has focus: {_footer_text(console)!r}",
        )
        # The keys the announcement means: list navigation and Trace.
        assert "j/k select" in _footer_text(console)
        assert "Y trace" in _footer_text(console)
        # Enter shows the message actions here; it does not send.
        assert "Enter send" not in _footer_text(console)

        await pilot.press("escape")
        await _wait_for(pilot, lambda: composer.draft_has_focus, "Esc did not return")
        await _wait_for(
            pilot,
            lambda: _TRANSCRIPT_ANNOUNCEMENT not in _footer_text(console),
            lambda: f"footer after Esc: {_footer_text(console)!r}",
        )
