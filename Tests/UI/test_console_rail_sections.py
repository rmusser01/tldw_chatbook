"""Console rail section header widget contracts."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_QUIET_EMPTY_COPY,
    CONSOLE_READY_EMPTY_COPY,
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.Widgets.Console.console_rail_section import (
    CONSOLE_RAIL_SECTION_TOGGLE_PREFIX,
    ConsoleRailSectionHeader,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscriptEmptyPanel
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Widgets.Console.console_workspace_details import (
    ConsoleWorkspaceDetailsTray,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState


class _HeaderApp(App):
    def compose(self):
        yield ConsoleRailSectionHeader(
            "Details",
            section_id="details",
            open=False,
            id="header-under-test",
        )


@pytest.mark.asyncio
async def test_rail_section_header_renders_title_and_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        title = app.query_one("#console-rail-section-title-details", Static)
        assert str(getattr(title.renderable, "plain", title.renderable)) == "Details"
        toggle = app.query_one(f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "+"
        assert toggle.tooltip == "Expand Details"


@pytest.mark.asyncio
async def test_rail_section_header_sync_open_flips_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        header = app.query_one("#header-under-test", ConsoleRailSectionHeader)
        header.sync_open(True)
        toggle = app.query_one(f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "-"
        assert toggle.tooltip == "Collapse Details"


def _workspace_state() -> ConsoleWorkspaceContextState:
    return ConsoleWorkspaceContextState(
        heading="Convos & Workspaces",
        workspace_label="Workspace: Default",
        authority_label="Authority: local registry ready",
        sync_label="Sync: not configured",
        runtime_label="Runtime: none, file tools disabled",
        conversation_rows=(),
        conversation_empty_copy="No conversations yet.",
        change_workspace_enabled=False,
        change_workspace_recovery="",
        new_conversation_enabled=False,
        new_conversation_recovery="",
        recovery_copy="",
    )


class _DetailsApp(App):
    def compose(self):
        yield ConsoleWorkspaceDetailsTray(_workspace_state(), id="details-tray")


@pytest.mark.asyncio
async def test_details_tray_renders_status_and_handoff_rows():
    app = _DetailsApp()
    async with app.run_test(size=(60, 30)):
        assert app.query_one("#console-workspace-authority-label")
        assert app.query_one("#console-workspace-sync-label")
        assert app.query_one("#console-workspace-runtime-label")
        assert app.query_one("#console-workspace-server-readiness-label")
        assert app.query_one("#console-workspace-handoff-title")
        assert app.query_one("#console-workspace-acp-handoff-audit")


class _ContextTrayApp(App):
    def compose(self):
        yield ConsoleWorkspaceContextTray(
            _workspace_state(),
            show_heading=False,
            id="context-tray",
        )


@pytest.mark.asyncio
async def test_context_tray_without_heading_omits_status_rows():
    app = _ContextTrayApp()
    async with app.run_test(size=(60, 30)):
        assert not list(app.query("#console-workspace-context-title"))
        assert not list(app.query("#console-workspace-authority-label"))
        assert not list(app.query("#console-workspace-handoff-title"))
        assert app.query_one("#console-workspace-selected-conversation")


def _card_state() -> ConsoleSetupCardState:
    return ConsoleSetupCardState(
        mode="card",
        steps=(
            ConsoleSetupStep(state="active", label="Add an API key"),
            ConsoleSetupStep(state="done", label="Pick a model"),
            ConsoleSetupStep(
                state="pending",
                label="Send your first message",
                detail="Type below, Enter to send",
            ),
        ),
    )


class _SetupPanelApp(App):
    def __init__(self, state: ConsoleSetupCardState) -> None:
        super().__init__()
        self._state = state

    def compose(self):
        yield ConsoleTranscriptEmptyPanel(
            self._state,
            provider_action_label="Configure API",
            provider_action_tooltip="Open provider settings.",
        )


@pytest.mark.asyncio
async def test_setup_panel_card_mode_renders_steps_and_actions():
    app = _SetupPanelApp(_card_state())
    async with app.run_test(size=(100, 30)):
        title = app.query_one("#console-empty-title", Static)
        assert "Get started" in str(getattr(title.renderable, "plain", title.renderable))
        step1 = app.query_one("#console-setup-step-1", Static)
        text1 = str(getattr(step1.renderable, "plain", step1.renderable))
        assert "1. ● Add an API key" in text1
        step2 = app.query_one("#console-setup-step-2", Static)
        assert "2. ✓ Pick a model" in str(getattr(step2.renderable, "plain", step2.renderable))
        step3 = app.query_one("#console-setup-step-3", Static)
        text3 = str(getattr(step3.renderable, "plain", step3.renderable))
        assert "3. ○ Send your first message" in text3
        assert "Type below, Enter to send" in text3
        assert app.query_one("#console-empty-action-row").styles.display != "none"
        assert app.query_one("#console-empty-body").styles.display == "none"


@pytest.mark.asyncio
async def test_setup_panel_ready_line_hides_steps_and_actions():
    app = _SetupPanelApp(
        ConsoleSetupCardState(mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY)
    )
    async with app.run_test(size=(100, 30)):
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_READY_EMPTY_COPY in str(getattr(body.renderable, "plain", body.renderable))
        assert not list(app.query("#console-setup-step-1"))
        assert app.query_one("#console-empty-action-row").styles.display == "none"
        assert app.query_one("#console-empty-title").styles.display == "none"


@pytest.mark.asyncio
async def test_setup_panel_quiet_mode_shows_only_quiet_copy():
    app = _SetupPanelApp(
        ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)
    )
    async with app.run_test(size=(100, 30)):
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_QUIET_EMPTY_COPY in str(getattr(body.renderable, "plain", body.renderable))
        assert not list(app.query("#console-setup-step-1"))
        assert app.query_one("#console-empty-action-row").styles.display == "none"


@pytest.mark.asyncio
async def test_setup_panel_sync_card_state_transitions_modes():
    app = _SetupPanelApp(_card_state())
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one(ConsoleTranscriptEmptyPanel)
        panel.sync_card_state(
            ConsoleSetupCardState(mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY),
            provider_action_label="Choose model",
            provider_action_tooltip="Pick a model.",
        )
        await pilot.pause()
        assert not list(app.query("#console-setup-step-1"))
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_READY_EMPTY_COPY in str(getattr(body.renderable, "plain", body.renderable))
