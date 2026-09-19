"""Scope recovery must follow the current panel through rebuilds and snapshots."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _snapshot(screen, count, *, schedule_reconcile=True):
    screen._apply_local_source_snapshot(
        {
            "notes": ({"id": "note-1", "title": "Research note"},) if count else (),
            "media": (),
            "conversations": (),
        },
        {"notes": count, "media": 0, "conversations": 0},
        {"notes": True, "media": True, "conversations": True},
        schedule_reconcile=schedule_reconcile,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_count", [0, 1])
@pytest.mark.parametrize("rebuild", ["navigate", "screen", "panel"])
async def test_rebuilt_search_scope_does_not_reuse_previous_visit_recovery(
    initial_count,
    rebuild,
    monkeypatch,
):
    """A new scope DOM must not inherit the previous visit's change gate."""
    app = _build_test_app()
    _seed_conversations(app, [])
    screen = LibraryScreen(app)
    screen.apply_navigation_context({"mode": "search"})
    screen._refresh_local_source_snapshot = lambda: None
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_library_screen(host)
        _snapshot(screen, initial_count)
        await _wait_for_library_shell(screen, pilot)
        await screen.workers.wait_for_complete()
        await pilot.pause()
        original = screen.query_one("#library-rag-source-scope")
        assert original.has_class("has-recovery") is (initial_count == 0)

        if rebuild == "navigate":
            screen.query_one("#library-row-browse-notes", Button).press()
            await _wait_for_selector(screen, pilot, "#library-notes-canvas")
            _snapshot(screen, 1 - initial_count)
            await pilot.pause()
            screen.query_one("#library-row-browse-search", Button).press()
        else:
            # Publish counts before the already-requested rebuild, without
            # letting an incremental sync repair the old DOM first.
            _snapshot(screen, 1 - initial_count, schedule_reconcile=False)
            if rebuild == "screen":
                await screen.recompose()
            else:
                panel = screen.query_one("#library-search-rag-panel")
                panel.sync_state(screen._library_rag_panel_state())
            await pilot.pause()
        await _wait_for_selector(screen, pilot, "#library-rag-source-scope")
        scope = screen.query_one("#library-rag-source-scope")
        assert scope is not original
        assert scope.has_class("has-recovery") is (initial_count == 1)

        _snapshot(screen, initial_count)
        await screen.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()

        assert scope.has_class("has-recovery") is (initial_count == 0)
        assert bool(screen.query("#library-rag-scope-recovery")) is (initial_count == 0)
        assert bool(screen.query("#library-rag-open-import-export")) is (
            initial_count == 0
        )
        toggle = screen.query_one("#library-rag-scope-toggle-notes", Button)
        assert toggle.disabled is (initial_count == 0)

        # Exercise the change gate itself: applying an identical snapshot
        # would return earlier from the screen's whole-snapshot equality guard.
        children = tuple(scope.children)
        mirror = Mock(wraps=screen._mirror_library_rag_scope_recovery)
        monkeypatch.setattr(screen, "_mirror_library_rag_scope_recovery", mirror)
        screen._sync_library_rag_scope_toggle_and_run_gate_widgets()
        screen._sync_library_rag_scope_toggle_and_run_gate_widgets()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert mirror.call_count == 0
        assert tuple(scope.children) == children
