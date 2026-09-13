"""task-3311: the Clear button's refocus must survive the relayout it triggers.

Live finding (3300-3305 arc verification, 2026-08-08): 2 of 4 Clear clicks
left focus adrift -- once the typed path's tail landed in the rail search
box, once a leading "/" from an unfocused state triggered the global
focus-search binding and the path vanished entirely.

Actual mechanism (from Textual 8.2.8 source, not the warning-relayout
hypothesis): with a pre-flight staged, clearing shrinks the type-group set,
so ``_update_library_ingest_dynamic_regions`` takes the STRUCTURAL branch
-- ``_refresh_library_ingest_canvas_preserving_context``. That helper
captures ``app.focused`` to restore after the full recompose, but
``Widget.focus()`` (the old refocus) is DEFERRED via ``app.call_later``, so
at capture time ``app.focused`` is still the just-clicked Clear button. The
restore then targets the NEW Clear button -- hidden (empty path) -- and
``Screen.set_focus`` silently no-ops on a non-focusable widget, leaving
focus wherever the recompose's prune dropped it (the rail search box, the
nearest non-pruned focus-chain neighbor, or nothing). The fix focuses the
path field SYNCHRONOUSLY (``Screen.set_focus``) before the update, so the
capture/restore round-trips ``#library-ingest-path`` deterministically.

The reproduction is LOOPED (AC#1: "not a single pass"): every iteration
stages a warning-bearing pre-flight (the ⚠ block that relayouts during the
clear), clicks Clear, immediately types "/", and requires the path field to
own both the focus and the character.
"""

import pytest
from textual.widgets import Button, Collapsible, Input

from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)

#: Enough passes that the pre-fix capture-the-wrong-widget path fails on the
#: first iteration and the post-fix path proves determinism, not luck.
CLEAR_LOOP_ITERATIONS = 8

_STAGED_PATH = "/tmp/task-3311-talk.mp3"


def _staged_preflight(path: str) -> PreflightResult:
    """A pre-flight whose group set differs from the cleared state's.

    ``audio_video`` (plus the always-appended ``generic``) versus the
    post-clear ``["generic"]`` is exactly the type-group change that makes
    the Clear press take the structural full-recompose branch -- the race
    window the live walk hit. The warning dict puts the ⚠ tooling-warning
    block on screen so the clear also tears down that region (the task's
    original suspicion).
    """
    return PreflightResult(
        type_groups={"audio_video": [path]},
        warnings=[
            {
                "feature": "audio_processing",
                "label": "Audio processing",
                "hint": "audio transcription",
                "command": "pip install faster-whisper",
            }
        ],
        errors=[],
        total_size=2048,
        truncated=False,
        total_files=1,
    )


def _neutralize_background_preflight(monkeypatch) -> None:
    """Keep the crafted pre-flight authoritative for the whole loop.

    The debounce timer (typing) and the blur trigger (clicking Clear blurs
    the path field) would otherwise launch REAL analysis workers against
    the fake staged path mid-loop and overwrite the staged state at a
    nondeterministic moment. The Clear handler itself uses neither seam.
    """
    monkeypatch.setattr(
        LibraryScreen,
        "_run_debounced_library_ingest_preflight",
        lambda self: None,
    )
    monkeypatch.setattr(
        LibraryScreen,
        "_trigger_library_ingest_preflight",
        lambda self, path: None,
    )


async def _enter_ingest_mode(screen, pilot):
    await screen._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)
    await _wait_for_selector(screen, pilot, "#library-ingest-path")
    await pilot.pause()


async def _stage_warning_preflight(screen, pilot) -> None:
    """Put the canvas in the live walk's pre-Clear shape and settle it."""
    path_input = screen.query_one("#library-ingest-path", Input)
    path_input.value = _STAGED_PATH
    form = screen._ingest_state.form
    form.path = _STAGED_PATH
    form.preflight = _staged_preflight(_STAGED_PATH)
    form.preflight_checking = False
    screen._update_library_ingest_dynamic_regions()
    tooling = await _wait_for_selector(
        screen, pilot, "#ingest-preflight-tooling-detail"
    )
    if getattr(tooling, "collapsed", False):
        tooling.collapsed = False
    await _wait_for_selector(screen, pilot, "#ingest-preflight-warning-0")
    await pilot.pause()


@pytest.mark.asyncio
async def test_clear_then_slash_lands_in_the_path_field_looped(monkeypatch):
    """AC#1: after Clear, focus deterministically lands on the path field
    even while the pre-flight/warning region relayouts -- looped, so a pass
    is determinism, not one lucky interleaving."""
    _neutralize_background_preflight(monkeypatch)
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_ingest_mode(screen, pilot)

        for iteration in range(CLEAR_LOOP_ITERATIONS):
            await _stage_warning_preflight(screen, pilot)

            await pilot.click("#library-ingest-clear-path")
            await pilot.press("/")
            await pilot.pause()

            # The structural recompose replaces the Input instance -- the
            # contract is about whichever widget now holds the id.
            path_input = screen.query_one("#library-ingest-path", Input)
            assert path_input.has_focus, (
                f"iteration {iteration}: focus adrift after Clear "
                f"(focused={screen.app.focused!r})"
            )
            assert path_input.value == "/", (
                f"iteration {iteration}: typed '/' never reached the path "
                f"field (path={path_input.value!r})"
            )
            search = screen.query_one("#library-search-input", Input)
            assert search.value == "", (
                f"iteration {iteration}: keystrokes leaked into the rail "
                f"search box ({search.value!r})"
            )

            # Reset for the next pass without leaving Ingest mode.
            path_input.value = ""
            screen._ingest_state.form.path = ""
            await pilot.pause()


@pytest.mark.asyncio
async def test_leading_slash_after_clear_edits_path_never_focus_search(
    monkeypatch,
):
    """AC#2: a typed leading "/" immediately after Clear edits the path --
    it must never fire the global focus-search binding (the live walk's
    vanished-path symptom)."""
    _neutralize_background_preflight(monkeypatch)
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_ingest_mode(screen, pilot)
        await _stage_warning_preflight(screen, pilot)

        await pilot.click("#library-ingest-clear-path")
        await pilot.press("/", "t", "m", "p")
        await pilot.pause()

        path_input = screen.query_one("#library-ingest-path", Input)
        assert path_input.value == "/tmp"
        assert path_input.has_focus
        search = screen.query_one("#library-search-input", Input)
        assert not search.has_focus, (
            "the leading '/' fired the global focus-search binding"
        )
        assert search.value == ""


#: The type-immediately-after-Clear race, looped for the same reason as
#: ``CLEAR_LOOP_ITERATIONS``: one pass could be a lucky interleaving.
CLEAR_TYPE_RACE_ITERATIONS = 6


@pytest.mark.asyncio
async def test_typing_into_the_pre_recompose_field_after_clear_survives(
    monkeypatch,
):
    """task-3311's other half: the keystroke SWALLOW window.

    Misrouting was fixed; loss was not. Live measurement (2026-08-08): 5/5
    characters typed within ~150ms of Clear vanished, 3/3 landed at
    >=400ms. ``handle_library_ingest_clear_path`` focuses the path field
    synchronously and then runs the STRUCTURAL branch, which rebuilds the
    Input from ``_library_ingest_form.path`` ("") -- so anything typed
    into the still-mounted pre-recompose widget dies with it, and the
    user is left staring at an empty field they just typed into.

    The reproduction types into the LIVE widget in the window the pilot's
    own ``click``+``press`` pair cannot reach (both settle the message
    pump first, which is why the existing tests above pass either way).
    """
    _neutralize_background_preflight(monkeypatch)
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_ingest_mode(screen, pilot)

        for iteration in range(CLEAR_TYPE_RACE_ITERATIONS):
            await _stage_warning_preflight(screen, pilot)

            # Run the Clear handler to completion, then type BEFORE the
            # message pump gets a turn -- the recompose it scheduled is
            # deferred, so the field it clears is still the mounted one.
            # (``pilot.click`` cannot express this: it settles the pump on
            # the way out, which is precisely why the two tests above pass
            # with or without the fix.)
            clear_button = screen.query_one("#library-ingest-clear-path", Button)
            screen.handle_library_ingest_clear_path(
                Button.Pressed(clear_button)
            )

            # ...and type into the widget that is still mounted right now,
            # exactly as a fast typist does.
            racing_input = screen.query_one("#library-ingest-path", Input)
            racing_input.value = "/e"
            await pilot.pause()
            await pilot.pause()

            path_input = screen.query_one("#library-ingest-path", Input)
            assert path_input.value == "/e", (
                f"iteration {iteration}: keystrokes typed right after "
                f"Clear were swallowed by the recompose "
                f"(field={path_input.value!r})"
            )
            assert screen._ingest_state.form.path == "/e", (
                f"iteration {iteration}: the form echo lost the typed text"
            )

            path_input.value = ""
            screen._ingest_state.form.path = ""
            await pilot.pause()


@pytest.mark.asyncio
async def test_clear_keeps_generic_options_reachable_for_next_generic_path(
    monkeypatch,
):
    """Clear must not strand the always-valid generic options panel hidden.

    A generic-only result after Clear has the same structural group set as
    the cleared state, so no later recompose will repair a panel that Clear
    incorrectly hid.
    """
    _neutralize_background_preflight(monkeypatch)
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_ingest_mode(screen, pilot)
        await _stage_warning_preflight(screen, pilot)

        await pilot.click("#library-ingest-clear-path")
        await pilot.pause()

        generic_path = "/tmp/next-note.txt"
        form = screen._ingest_state.form
        form.path = generic_path
        form.preflight = PreflightResult(
            type_groups={"generic": [generic_path]},
            warnings=[],
            errors=[],
            total_size=128,
            truncated=False,
            total_files=1,
        )
        form.preflight_checking = False
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()

        panel = screen.query_one("#type-group-generic", Collapsible)
        assert panel.display is True
        title = screen.query_one("#type-group-generic CollapsibleTitle")
        assert title.can_focus
        title.focus()
        await pilot.pause()
        assert title.has_focus
