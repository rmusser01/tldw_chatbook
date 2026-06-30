import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.Workbench.workbench_state import (
    RecoveryState,
    WorkbenchAction,
    WorkbenchHeaderState,
    WorkbenchMode,
    WorkbenchPaneState,
    WorkbenchState,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import (
    WorkbenchActionRequested,
    WorkbenchFrame,
)


def _state(
    *,
    subtitle: str = "",
    action_label: str = "Settings",
    recovery_body: str = "Choose a provider to continue.",
) -> WorkbenchState:
    return WorkbenchState(
        header=WorkbenchHeaderState(
            title="Console",
            subtitle=subtitle,
            status="ready",
        ),
        modes=(
            WorkbenchMode(id="chat", label="Chat", active=True, status="ready"),
            WorkbenchMode(id="rag", label="RAG", status="empty"),
        ),
        actions=(
            WorkbenchAction(
                id="provider-recovery",
                label=action_label,
                tooltip="Open provider settings",
                primary=True,
            ),
        ),
        recovery=RecoveryState(
            title="Provider required",
            body=recovery_body,
            action=WorkbenchAction(
                id="provider-recovery",
                label=action_label,
                tooltip="Open provider settings",
                primary=True,
            ),
        ),
        route_id="console",
    )


class _WorkbenchFrameApp(App):
    def __init__(self, state: WorkbenchState) -> None:
        super().__init__()
        self.state = state
        self.requested_actions: list[str] = []

    def compose(self) -> ComposeResult:
        yield WorkbenchFrame(self.state, id="frame")

    @on(WorkbenchActionRequested)
    def on_workbench_action_requested(
        self,
        event: WorkbenchActionRequested,
    ) -> None:
        self.requested_actions.append(event.action_id)


@pytest.mark.asyncio
async def test_workbench_frame_sync_state_keeps_direct_child_ids_stable():
    app = _WorkbenchFrameApp(_state(subtitle="Ready"))

    async with app.run_test() as pilot:
        await pilot.pause()
        frame = app.query_one("#frame", WorkbenchFrame)
        original_child_ids = tuple(child.id for child in frame.children)

        frame.sync_state(
            _state(
                subtitle="Provider setup needed",
                action_label="Choose model",
                recovery_body="Choose a model before running Search/RAG.",
            )
        )
        await pilot.pause()

        recovery = frame.query_one("#workbench-recovery")
        updated_child_ids = tuple(child.id for child in frame.children)
        recovery_text = recovery.renderable.plain

        assert updated_child_ids == original_child_ids
        assert "Choose a model" in recovery_text


@pytest.mark.asyncio
async def test_recovery_callout_action_emits_workbench_action_requested():
    app = _WorkbenchFrameApp(_state())

    async with app.run_test() as pilot:
        await pilot.pause()

        action = app.query_one("#workbench-recovery-action", Button)
        assert action.label.plain == "Settings"

        await pilot.click("#workbench-recovery-action")
        await pilot.pause()

    assert app.requested_actions == ["provider-recovery"]


@pytest.mark.asyncio
async def test_workbench_frame_sync_state_reorders_actions_modes_and_panes():
    initial = WorkbenchState(
        header=WorkbenchHeaderState(title="Console"),
        modes=(
            WorkbenchMode(id="chat", label="Chat"),
            WorkbenchMode(id="rag", label="RAG"),
        ),
        actions=(
            WorkbenchAction(id="settings", label="Settings"),
            WorkbenchAction(id="send", label="Send"),
        ),
        panes=(
            WorkbenchPaneState(id="context", title="Context"),
            WorkbenchPaneState(id="transcript", title="Transcript"),
        ),
    )
    app = _WorkbenchFrameApp(initial)

    async with app.run_test() as pilot:
        await pilot.pause()
        frame = app.query_one("#frame", WorkbenchFrame)

        frame.sync_state(
            WorkbenchState(
                header=WorkbenchHeaderState(title="Console"),
                modes=(
                    WorkbenchMode(id="rag", label="RAG"),
                    WorkbenchMode(id="chat", label="Chat"),
                ),
                actions=(
                    WorkbenchAction(id="send", label="Send"),
                    WorkbenchAction(id="settings", label="Settings"),
                ),
                panes=(
                    WorkbenchPaneState(id="transcript", title="Transcript"),
                    WorkbenchPaneState(id="context", title="Context"),
                ),
            )
        )
        await pilot.pause()

        action_ids = [
            getattr(child, "_workbench_action_id", None)
            for child in frame.query_one("#workbench-command-strip").children
        ]
        mode_ids = [
            getattr(child, "_workbench_mode_id", None)
            for child in frame.query_one("#workbench-mode-strip").children
        ]
        pane_ids = [
            getattr(child, "_workbench_pane_id", None)
            for child in frame.query_one("#workbench-pane-region").children
        ]

    assert action_ids == ["send", "settings"]
    assert mode_ids == ["rag", "chat"]
    assert pane_ids == ["transcript", "context"]
