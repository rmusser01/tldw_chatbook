"""Details tray label/value legibility regressions (TASK-715).

Live UAT (workspace-settings review, 2026-07-26) caught four defects in the
Session rail's Details tray at its real ~37-column width: the "Server handoff"
label wrapped into an orphaned lowercase "handoff" line, two different rows
were both labeled "Handoff", long values truncated in their default state, and
jargon-dense server/sync/ACP rows rendered for features that cannot be
configured anywhere in the UI yet.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tldw_chatbook.Widgets.Console.console_workspace_details import (
    ConsoleWorkspaceDetailsTray,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState

# The StatusPair label column is 12 cells; anything longer wraps into the
# orphaned-word defect this task fixes.
LABEL_COLUMN_WIDTH = 12


def _state(**overrides) -> ConsoleWorkspaceContextState:
    base = dict(
        heading="Convos & Workspaces",
        workspace_label="Workspace: Default",
        workspace_name="Default",
        authority_label="Authority: local registry ready",
        sync_label="Sync: not configured",
        runtime_label="Runtime: none, file tools disabled",
        conversation_rows=(),
        conversation_empty_copy="No active workspace conversations.",
        change_workspace_enabled=False,
        change_workspace_recovery="",
        new_conversation_enabled=True,
        new_conversation_recovery="",
        recovery_copy="",
    )
    base.update(overrides)
    return ConsoleWorkspaceContextState(**base)


class TrayApp(App[None]):
    def __init__(self, state: ConsoleWorkspaceContextState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        rail = Vertical(id="rail")
        rail.styles.width = 41
        with rail:
            yield ConsoleWorkspaceDetailsTray(self._state, id="tray")


@pytest.mark.asyncio
async def test_unconfigured_server_features_collapse_to_one_line() -> None:
    """Sync/Server/ACP rows are aspirational until something can configure
    them - the default state must show one plain line instead of five rows of
    unreachable-status jargon."""
    app = TrayApp(_state())
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        collapsed = app.query_one(
            "#console-workspace-server-features-collapsed", Static
        )
        text = str(collapsed.renderable)
        assert "not configured" in text.lower()
        # The three collapsed features are named so users know what's off.
        assert "sync" in text.lower()
        assert "handoff" in text.lower()
        # None of the aspirational per-feature rows render.
        assert not app.query("#console-workspace-server-readiness-label")
        assert not app.query("#console-workspace-handoff-label")
        assert not app.query("#console-workspace-acp-handoff-detail")
        # Real rows stay: Storage, File tools, and the Handoff package section.
        assert app.query("#console-workspace-authority-label")
        assert app.query("#console-workspace-runtime-label")
        assert app.query("#console-workspace-handoff-title")


@pytest.mark.asyncio
async def test_configured_server_rows_use_unique_fitting_labels() -> None:
    """When server features ARE configured the rows come back - with labels
    that fit the 12-cell column (no orphan wrap) and no duplicate 'Handoff'."""
    app = TrayApp(
        _state(
            sync_label="Sync: syncing",
            server_readiness_label="Server: adapter ready",
            server_readiness_detail="Server adapter is ready.",
            acp_handoff_label="ACP task/run: ready",
            acp_handoff_detail="ACP runtime is configured.",
        )
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert not app.query("#console-workspace-server-features-collapsed")

        labels = {}
        for selector in (
            "#console-workspace-authority-label",
            "#console-workspace-sync-label",
            "#console-workspace-runtime-label",
            "#console-workspace-server-readiness-label",
            "#console-workspace-handoff-label",
        ):
            widget = app.query_one(selector, Static)
            labels[selector] = str(widget.renderable)

        for selector, text in labels.items():
            assert len(text) <= LABEL_COLUMN_WIDTH, (
                f"{selector} label {text!r} exceeds the {LABEL_COLUMN_WIDTH}-cell "
                "column and would wrap into an orphaned word"
            )
        # The ACP status row must not reuse the Handoff section's label.
        assert labels["#console-workspace-handoff-label"] != "Handoff"


@pytest.mark.asyncio
async def test_default_workspace_file_tools_value_fits_without_truncation() -> None:
    """'Off in Default workspace' ellipsized in its own default state; the
    value must fit the ~23-cell value column outright."""
    app = TrayApp(_state())
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        value = str(
            app.query_one("#console-workspace-runtime-value", Static).renderable
        )
        assert "off" in value.lower()
        assert len(value) <= 20, (
            f"file-tools value {value!r} is long enough to truncate in the rail"
        )
