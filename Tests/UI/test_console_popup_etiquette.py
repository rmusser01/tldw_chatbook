"""TASK-24416: slash-popup etiquette on the real Console screen.

Three behaviors from the 2026-08-29 `/`-trigger review, all confirmed live
before the fix:

1. **Sticky Escape dismissal** -- Escape closed the popup, but the next
   keystroke re-opened it (every ``DraftChanged`` ran the un-gated popup
   sync), so there was no way to compose a slash-prefixed draft with the
   popup out of the way.
2. **Bare-``/`` Enter guard** -- with the popup open on a bare ``/``, Enter
   silently staged the first listed command (``/prompt ``) instead of
   falling through to send.
3. **Undo-safe accept** -- accepting a suggestion routed through
   ``load_draft``, which wipes the composer's undo stacks (TASK-1281 scope
   semantics), so an accidental accept could not be Ctrl+Z'd.

All three drive REAL key presses through the mounted Console, never a
method call (same discipline as test_console_composer_draft_changed.py).
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_command_popup import ConsoleCommandPopup

APP_SIZE = (160, 48)


async def _console_with_popup_parts():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    return host


async def _mounted_parts(host, pilot):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    popup = console.query_one("#console-command-popup", ConsoleCommandPopup)
    composer.focus()
    await pilot.pause()
    return console, composer, popup


@pytest.mark.asyncio
async def test_escape_dismissal_survives_typing():
    """Typing after an Escape dismissal must not re-open the popup.

    The user said "go away" with Escape; each further keystroke used to
    walk the un-gated sync straight back into show_suggestions.
    """
    host = await _console_with_popup_parts()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer, popup = await _mounted_parts(host, pilot)

        await pilot.press("/")
        await pilot.pause()
        assert popup.is_open

        await pilot.press("escape")
        await pilot.pause()
        assert not popup.is_open

        await pilot.press("p")
        await pilot.pause()
        assert not popup.is_open, "a typed edit re-opened the popup the user dismissed"
        assert composer.draft_text() == "/p"


@pytest.mark.asyncio
async def test_dismissal_rearms_after_leaving_completion_context():
    """Leaving slash context (a space, a clear) re-arms the trigger.

    Sticky does not mean dead: a NEW completion context (typing ``/`` again
    after the draft stopped being a bare command token) must open the popup
    again.
    """
    host = await _console_with_popup_parts()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer, popup = await _mounted_parts(host, pilot)

        await pilot.press("/")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not popup.is_open

        # Leave the completion context entirely, then come back to a fresh
        # bare slash.
        await pilot.press("ctrl+u")
        await pilot.pause()
        assert composer.draft_text() == ""
        assert not popup.is_open

        await pilot.press("/")
        await pilot.pause()
        assert popup.is_open, "a fresh completion context did not re-arm the trigger"


@pytest.mark.asyncio
async def test_bare_slash_enter_does_not_stage_the_first_command():
    """Enter on an unfiltered (empty-prefix) list falls through to send.

    A user probing the trigger types ``/``, sees the list, presses Enter --
    and used to get ``/prompt `` silently staged. The unknown-command
    escape (send-path) owns this draft instead.
    """
    host = await _console_with_popup_parts()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer, popup = await _mounted_parts(host, pilot)

        await pilot.press("/")
        await pilot.pause()
        assert popup.is_open

        await pilot.press("enter")
        await pilot.pause()
        assert composer.draft_text() == "/", (
            "Enter on a bare-slash list staged a command instead of falling "
            f"through to send (draft became {composer.draft_text()!r})"
        )
        assert not popup.is_open


@pytest.mark.asyncio
async def test_filtered_prefix_enter_still_accepts():
    """The guard must not over-reach: a non-empty prefix keeps Enter-accept."""
    host = await _console_with_popup_parts()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer, popup = await _mounted_parts(host, pilot)

        await pilot.press("/", "s", "y", "s")
        await pilot.pause()
        assert popup.is_open

        await pilot.press("enter")
        await pilot.pause()
        assert composer.draft_text() == "/system "


@pytest.mark.asyncio
async def test_tab_accept_is_undoable():
    """Accepting a suggestion must leave the pre-accept draft on Ctrl+Z."""
    host = await _console_with_popup_parts()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer, popup = await _mounted_parts(host, pilot)

        await pilot.press("/", "s", "y")
        await pilot.pause()
        assert popup.is_open

        await pilot.press("tab")
        await pilot.pause()
        assert composer.draft_text() == "/system "

        await pilot.press("ctrl+z")
        await pilot.pause()
        assert composer.draft_text() == "/sy", (
            "undo after an accept did not restore the pre-accept draft "
            f"(got {composer.draft_text()!r})"
        )


@pytest.mark.asyncio
async def test_enter_accept_is_undoable():
    """The Enter-accept path shares the undo-safe replacement."""
    host = await _console_with_popup_parts()
    async with host.run_test(size=APP_SIZE) as pilot:
        console, composer, popup = await _mounted_parts(host, pilot)

        await pilot.press("/", "p", "r")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert composer.draft_text() == "/prompt "

        await pilot.press("ctrl+z")
        await pilot.pause()
        assert composer.draft_text() == "/pr"
