"""The tray must not recompose for state fields its DOM was not built from.

TASK-26836 (TASK-26834 fix target 1). Probe-observed on 2026-09-01: a tree
click pushed ``delta=conversation_browser`` into the WORKSPACES tray -- whose
``content="workspace"`` compose never reads that field -- and it recomposed
anyway, opening one of the 2-3 nested App batches that held every paint for
250-400ms. ``_can_skip_recompose`` condition 5 was whole-state value
equality, so any field delta forced a rebuild regardless of whether the
mounted DOM depends on it.

The fix records which state fields ``compose`` actually read (the same
pattern as the composed row signature) and skips when only unread fields
changed, adopting the new state. Everything else about the guard -- the
fresh-instance healing push, the DOM signature check, the recompose latch --
is untouched; these tests lean on the TASK-15454 suite's harness for exactly
that reason.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from Tests.UI.test_console_workspace_tray_recompose_guard import (
    APP_SIZE,
    _RecomposeCounter,
    _build_test_app,
    _settled_tray,
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)

WORKSPACES_TRAY_SELECTOR = "#console-workspaces-context"


async def _settled_workspaces_tray(host, pilot):
    """Return the rail's workspace-content tray, fully settled."""
    console, _conversations_tray = await _settled_tray(host, pilot)
    tray = console.query_one(WORKSPACES_TRAY_SELECTOR, ConsoleWorkspaceContextTray)
    assert tray.content == "workspace"
    return console, tray


# ---------------------------------------------------------------------------
# The probe-observed waste: browser deltas on the workspaces tray
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browser_only_delta_does_not_recompose_the_workspaces_tray():
    """The exact wasted recompose the in-terminal probe recorded."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        _console, tray = await _settled_workspaces_tray(host, pilot)
        assert getattr(tray, "_console_workspace_context_synced", False) is True
        browser = tray.state.conversation_browser
        assert browser is not None

        changed = replace(
            tray.state,
            conversation_browser=replace(
                browser, selected_summary="A different active conversation"
            ),
        )
        children_before = tuple(tray.children)
        with _RecomposeCounter() as counter:
            tray.sync_state(changed)
            assert counter.calls == 0, (
                "content='workspace' never renders conversation_browser; a "
                "browser-only delta must not rebuild this tray"
            )
        assert tuple(tray.children) == children_before
        # The state is adopted, so the delta is not re-diffed forever.
        assert tray.state == changed


@pytest.mark.asyncio
async def test_unrendered_field_delta_does_not_recompose_the_conversations_tray():
    """Mirror direction: workspace-section fields are unread at
    content='conversations'."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        _console, tray = await _settled_tray(host, pilot)

        changed = replace(tray.state, workspace_name="Renamed Workspace")
        with _RecomposeCounter() as counter:
            tray.sync_state(changed)
            assert counter.calls == 0, (
                "content='conversations' never renders workspace_name; its "
                "delta must not rebuild this tray"
            )
        assert tray.state == changed


# ---------------------------------------------------------------------------
# Read fields still recompose, and the guard's other proofs still gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_read_field_delta_still_recomposes():
    """workspace_name IS rendered at content='workspace' -- rebuild."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        _console, tray = await _settled_workspaces_tray(host, pilot)

        changed = replace(tray.state, workspace_name="Renamed Workspace")
        with _RecomposeCounter() as counter:
            tray.sync_state(changed)
            assert counter.calls == 1


@pytest.mark.asyncio
async def test_unread_delta_on_a_fresh_instance_still_heals():
    """The TASK-344/349 one-time healing push outranks read-awareness."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=APP_SIZE) as pilot:
        _console, tray = await _settled_workspaces_tray(host, pilot)
        if hasattr(tray, "_console_workspace_context_synced"):
            del tray._console_workspace_context_synced
        browser = tray.state.conversation_browser
        assert browser is not None

        changed = replace(
            tray.state,
            conversation_browser=replace(browser, selected_summary="x"),
        )
        with _RecomposeCounter() as counter:
            tray.sync_state(changed)
            assert counter.calls == 1
