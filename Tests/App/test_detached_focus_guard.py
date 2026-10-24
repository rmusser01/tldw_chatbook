# test_detached_focus_guard.py
# Description: TASK-34000.26 -- a keyboard focus left on a removed widget
# must not swallow the next key; the app-level guard heals it.
"""
Review S-10 (``Docs/QA/notes-library-ux-review-2026-10-02/verify/nl-v-s-10/
textual-dead-focus-mechanism.out``): once during the keyboard-only journey
Tab, F1 and Ctrl+P all stopped working until a mouse click. Mechanism, on
Textual 8.2.8: ``App._prune`` marks a subtree ``_pruning`` synchronously and
only resets focus if the focused widget is in that subtree AT prune time; a
``Screen.set_focus`` write that lands AFTER the mark is accepted (``focusable``
reads ``visible``, never ``_pruning``) and nothing resets it when the detach
completes. ``App.on_event`` then forwards every non-priority key to
``self.focused._forward_event`` -- a closed message pump -- so F1 and Ctrl+P
(non-priority bindings) and Tab (a Screen binding) all die.

A bare ``textual.app.App`` harness, not the real app (no ``tldw_chatbook.app``
import, no Hypothesis): this file runs in the FAST_LANE.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Input

from tldw_chatbook.Utils.text_selection_crash_guard import DetachedFocusGuard

pytestmark = pytest.mark.asyncio


class _Probe:
    hits: list[str]
    pressed: list[str]

    def compose(self) -> ComposeResult:
        yield Input(id="field")
        yield Button("Go", id="go")

    def action_help(self) -> None:
        self.hits.append("f1")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.pressed.append(event.button.id or "")


class _VanillaApp(_Probe, App[None]):
    """Plain Textual app: exhibits the upstream dead-focus state unguarded."""

    BINDINGS = [Binding("f1", "help", "Help")]

    def __init__(self) -> None:
        super().__init__()
        self.hits = []
        self.pressed = []


class _GuardedApp(DetachedFocusGuard, _Probe, App[None]):
    """Same app with the shipped guard mixin, mirroring TldwCli's base list."""

    BINDINGS = [Binding("f1", "help", "Help")]

    def __init__(self) -> None:
        super().__init__()
        self.hits = []
        self.pressed = []


async def _focus_during_removal(app: App[None], pilot) -> Input:
    """The review's repro: a focus write that lands after the prune mark.

    Precondition asserts pin the attribution: the write is ACCEPTED while the
    widget is pruning, and after the removal ``screen.focused`` still names
    the detached widget -- exactly the state the live keyboard died in.
    """
    await pilot.pause()
    field = app.query_one("#field", Input)
    removal = field.remove()  # App._prune marks _pruning synchronously
    assert field._pruning is True
    app.screen.set_focus(field)  # Textual 8.2.8 accepts it
    assert app.screen.focused is field
    await removal
    assert app.screen.focused is field and not field.is_attached
    return field


# --------------------------------------------------------------------------
# Negative control: the upstream mechanism, unguarded.
# --------------------------------------------------------------------------


async def test_a_key_after_a_dead_focus_reaches_nothing_without_the_guard() -> None:
    """Textual 8.2.8 forwards the key into the dead widget's closed pump."""
    app = _VanillaApp()
    async with app.run_test() as pilot:
        await _focus_during_removal(app, pilot)
        await pilot.press("f1")
        assert app.hits == [], (
            "the negative control no longer reproduces the dead-focus state; "
            "re-read Textual's App.on_event forward rule before trusting the guard"
        )


# --------------------------------------------------------------------------
# The guard.
# --------------------------------------------------------------------------


async def test_the_guard_clears_a_dead_focus_and_the_same_key_reaches_the_app() -> None:
    """AC#2: the key that found the corpse is the key that works."""
    app = _GuardedApp()
    async with app.run_test() as pilot:
        await _focus_during_removal(app, pilot)
        await pilot.press("f1")
        assert app.hits == ["f1"]
        assert app.screen.focused is None or app.screen.focused.is_attached


async def test_a_healed_enter_presses_nothing() -> None:
    """Review Focus 5: a healed key is never delivered to a widget the user
    did not choose -- the heal clears focus, it never picks a Button."""
    app = _GuardedApp()
    async with app.run_test() as pilot:
        await _focus_during_removal(app, pilot)
        await pilot.press("enter")
        assert app.pressed == []
        # Review Focus 4: ``None`` is a working state -- Tab walks the chain.
        await pilot.press("tab")
        assert app.screen.focused is not None and app.screen.focused.is_attached


async def test_a_live_focus_is_left_alone() -> None:
    """A key typed into an attached Input reaches that Input unchanged."""
    app = _GuardedApp()
    async with app.run_test() as pilot:
        field = app.query_one("#field", Input)
        field.focus()
        await pilot.pause()
        await pilot.press("a")
        assert field.value == "a" and app.screen.focused is field


async def test_a_key_after_exit_logs_nothing() -> None:
    """Fix round 1, M-1: Textual's ``is_attached`` answers False for every node
    once ``App.exit()`` has set ``_exit``, so without the exit check each key
    processed during shutdown would log a false "removed widget" warning to
    the persistent sink (and pollute AC#5's defect counter)."""
    import io

    from loguru import logger

    app = _GuardedApp()
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        async with app.run_test() as pilot:
            field = app.query_one("#field", Input)
            field.focus()
            await pilot.pause()
            assert app.screen.focused is field
            app.exit()
            assert app._exit is True and not field.is_attached
            from textual import events

            await app.on_event(events.Key("f1", None))
    finally:
        logger.remove(handler_id)
    assert "removed widget" not in sink.getvalue(), sink.getvalue()
