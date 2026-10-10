"""TASK-34000.27, the slower arms (not in the PR-gate census).

The 80x24 repro of the vetoed switch (AC#5's second size), and the
CONFLICTED kind: a note changed elsewhere since it was opened, so the flush
cannot save it and the nav-bar click must say so with the destination
named, while Library stays highlighted. The lean core is
`test_nav_veto_keeps_highlight.py`.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, TextArea

from Tests.UI.test_library_shell import (
    _bump_note_version_externally,
    _open_note_editor,
)
from Tests.UI.test_nav_veto_keeps_highlight import (
    _VETO_HEAD,
    _WHITESPACE_REASON,
    _activate_destination,
    _library,
    _library_app,
    _record_toasts,
    _set_title_and_wait,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

NARROW = (80, 24)


async def test_vetoed_switch_keeps_library_highlighted_at_80x24(monkeypatch):
    app = _library_app(monkeypatch)
    toasts = _record_toasts(monkeypatch, app)

    async with app.run_test(size=NARROW) as pilot:
        screen = await _library(app, pilot)
        await _open_note_editor(screen, pilot)
        await _set_title_and_wait(screen, pilot, "Draft ", _WHITESPACE_REASON)
        toasts.clear()

        activated, bar = await _activate_destination(app, pilot, "console")

        assert activated is True
        assert app.screen is screen
        assert bar.active_destination_id == "library"
        assert bar.query_one("#nav-library").has_class("is-active")
        assert [t for t in toasts if _VETO_HEAD in t and _WHITESPACE_REASON in t], (
            toasts
        )
        assert screen.query_one("#library-note-title", Input).value == "Draft "


async def test_conflicted_note_veto_names_the_conflict_and_the_destination(
    monkeypatch,
):
    """Another writer bumped the note's version after it was opened; the
    flush's save conflicts, and the toast says so in N-25's own words with
    the destination head -- never the title-blaming sentence."""
    app = _library_app(monkeypatch)
    toasts = _record_toasts(monkeypatch, app)

    async with app.run_test(size=(160, 45)) as pilot:
        screen = await _library(app, pilot)
        await _open_note_editor(screen, pilot)
        _bump_note_version_externally(app.notes_scope_service, "n-1")
        body = screen.query_one("#library-note-body", TextArea)
        body.text = "alpha budget line, edited after the external bump"
        await pilot.pause()
        toasts.clear()

        activated, bar = await _activate_destination(app, pilot, "console")

        assert activated is True
        assert app.screen is screen
        assert bar.active_destination_id == "library"
        conflict_toasts = [
            t for t in toasts if _VETO_HEAD in t and "changed elsewhere" in t
        ]
        assert conflict_toasts, f"no conflict reason was shown: {toasts!r}"
        assert not [t for t in conflict_toasts if "title" in t.lower()], (
            conflict_toasts
        )
        assert body.text == "alpha budget line, edited after the external bump"
