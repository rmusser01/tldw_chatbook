"""Detail and inspector widgets for the Scheduling workbench."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from ....Scheduling.models import ReminderTask, TaskStatus
from ..destination_recovery import DestinationRecoveryState


SCHEDULES_EMPTY_CONSOLE_RECOVERY = DestinationRecoveryState(
    status_label="Select an active run",
    unavailable_what="Console follow for Schedules",
    why="no active schedule run or reading digest output is available",
    next_action="Start or select a schedule run to enable Console follow.",
    recovery_action="Create a scheduled job",
    authority_owner="local",
    stable_selector="schedules-follow-in-console",
    disabled_tooltip="Start or select a schedule run to enable Console follow.",
)


def _format_next_run(task: ReminderTask | None) -> str:
    """Format a task's next run time for display."""
    if task is None or task.next_run_at is None:
        return "-"
    return task.next_run_at.strftime("%Y-%m-%d %H:%M")


class TaskDetail(Vertical):
    """Render the selected reminder task's core details and actions."""

    def compose(self) -> ComposeResult:
        yield Static("Task Detail", id="scheduling-task-detail-header")
        yield Static("Title: -", id="scheduling-task-detail-title")
        yield Static("Type: -", id="scheduling-task-detail-type")
        yield Static("Status: -", id="scheduling-task-detail-status")
        yield Static("Schedule: -", id="scheduling-task-detail-schedule")
        yield Static("Next Run: -", id="scheduling-task-detail-next-run")
        yield Horizontal(
            Button(
                "Enable",
                id="scheduling-enable-task",
                variant="success",
                tooltip="Enable this scheduled task.",
            ),
            Button(
                "Disable",
                id="scheduling-disable-task",
                variant="warning",
                tooltip="Disable this scheduled task.",
            ),
            Button(
                "Delete",
                id="scheduling-delete-task",
                variant="error",
                tooltip="Delete this scheduled task.",
            ),
            id="scheduling-task-detail-lifecycle",
        )
        yield Button(
            "Follow in Console",
            id="schedules-follow-in-console",
            tooltip=SCHEDULES_EMPTY_CONSOLE_RECOVERY.disabled_tooltip,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle lifecycle actions (console follow is handled by the workbench)."""
        button_id = event.button.id
        if button_id in {
            "scheduling-enable-task",
            "scheduling-disable-task",
            "scheduling-delete-task",
        }:
            event.stop()
        if button_id == "scheduling-enable-task":
            self.log.debug("Enable task requested (not yet implemented)")
        elif button_id == "scheduling-disable-task":
            self.log.debug("Disable task requested (not yet implemented)")
        elif button_id == "scheduling-delete-task":
            self.log.debug("Delete task requested (not yet implemented)")

    def set_task(self, task: ReminderTask | None) -> None:
        """Update the detail view for the given task (or clear it)."""
        if task is None:
            self._update_static("scheduling-task-detail-title", "Title: -")
            self._update_static("scheduling-task-detail-type", "Type: -")
            self._update_static("scheduling-task-detail-status", "Status: -")
            self._update_static("scheduling-task-detail-schedule", "Schedule: -")
            self._update_static("scheduling-task-detail-next-run", "Next Run: -")
            return

        schedule = task.cron if task.schedule_kind.value == "recurring" else "One-time"

        self._update_static("scheduling-task-detail-title", f"Title: {task.title}")
        self._update_static("scheduling-task-detail-type", f"Type: {task.schedule_kind.value}")
        self._update_static("scheduling-task-detail-status", f"Status: {task.last_status.value}")
        self._update_static("scheduling-task-detail-schedule", f"Schedule: {schedule}")
        self._update_static("scheduling-task-detail-next-run", f"Next Run: {_format_next_run(task)}")

    def set_follow_available(self, available: bool) -> None:
        """Enable or disable the Console-follow button and set its tooltip."""
        button = self.query_one("#schedules-follow-in-console", Button)
        button.disabled = not available
        button.tooltip = (
            "Open the active schedule run in Console."
            if available
            else SCHEDULES_EMPTY_CONSOLE_RECOVERY.disabled_tooltip
        )

    def _update_static(self, widget_id: str, content: str) -> None:
        """Update a child Static widget by id."""
        static = self.query_one(f"#{widget_id}", Static)
        static.update(content)


class TaskInspector(Vertical):
    """Render status, sync, and conflict metadata for a task."""

    DEFAULT_CSS = """
    #scheduling-conflict-card {
        padding: 0;
    }
    #scheduling-conflict-card.conflict {
        border: solid $error;
        padding: 1;
        color: $error;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Inspector", id="scheduling-task-inspector-header")
        yield Static("Status: -", id="scheduling-inspector-status")
        yield Static("Sync: -", id="scheduling-inspector-sync")
        yield Vertical(
            Static("No conflict", id="scheduling-conflict-text"),
            id="scheduling-conflict-card",
        )

    def set_task(self, task: ReminderTask | None) -> None:
        """Update the inspector view for the given task (or clear it)."""
        if task is None:
            self._update_static("scheduling-inspector-status", "Status: -")
            self._update_static("scheduling-inspector-sync", "Sync: -")
            self._update_conflict_card(None)
            return

        sync_status = f"version {task.sync_version}"
        if task.server_id:
            sync_status += f" (server {task.server_id})"
        else:
            sync_status += " (local)"

        summary = f"{task.last_status.value}; next run {_format_next_run(task)}"

        self._update_static("scheduling-inspector-status", f"Status: {summary}")
        self._update_static("scheduling-inspector-sync", f"Sync: {sync_status}")
        self._update_conflict_card(task)

    def _update_conflict_card(self, task: ReminderTask | None) -> None:
        """Update the conflict card for the current task state."""
        card = self.query_one("#scheduling-conflict-card", Vertical)
        text = self.query_one("#scheduling-conflict-text", Static)
        if task is not None and task.last_status == TaskStatus.CONFLICT:
            text.update(f"Conflict detected\n{task.title}")
            card.add_class("conflict")
        else:
            text.update("No conflict")
            card.remove_class("conflict")

    def _update_static(self, widget_id: str, content: str) -> None:
        """Update a child Static widget by id."""
        static = self.query_one(f"#{widget_id}", Static)
        static.update(content)
