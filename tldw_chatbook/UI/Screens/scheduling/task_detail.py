"""Detail and inspector widgets for the Scheduling workbench."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from ....Scheduling.events import (
    DeleteTaskRequested,
    DisableTaskRequested,
    EditTaskRequested,
    EnableTaskRequested,
    RunReminderNowRequested,
)
from ....Scheduling.models import ReminderTask, ScheduledTask, ScheduleKind, TaskStatus
from ....Widgets.delete_confirmation_dialog import DeleteConfirmationDialog
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


_WEEKDAYS = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]

_STATUS_LABELS: dict[TaskStatus, str] = {
    TaskStatus.WAITING: "Waiting",
    TaskStatus.RUNNING: "Running",
    TaskStatus.PAUSED: "Paused",
    TaskStatus.NEEDS_ATTENTION: "Needs Attention",
    TaskStatus.BLOCKED: "Blocked",
    TaskStatus.DISABLED: "Disabled",
    TaskStatus.ARCHIVED: "Archived",
    TaskStatus.COMPLETED: "Completed",
    TaskStatus.FOUND_RESULTS: "Found Results",
    TaskStatus.MISSED: "Missed",
    TaskStatus.TIMED_OUT: "Timed out",
    TaskStatus.CONFLICT: "Conflict",
}

_STATUS_BADGE_CLASSES: dict[TaskStatus, str] = {
    TaskStatus.WAITING: "waiting",
    TaskStatus.RUNNING: "running",
    TaskStatus.PAUSED: "paused",
    TaskStatus.NEEDS_ATTENTION: "needs-attention",
    TaskStatus.BLOCKED: "blocked",
    TaskStatus.DISABLED: "disabled",
    TaskStatus.ARCHIVED: "archived",
    TaskStatus.COMPLETED: "completed",
    TaskStatus.FOUND_RESULTS: "found-results",
    TaskStatus.MISSED: "missed",
    TaskStatus.TIMED_OUT: "timed-out",
    TaskStatus.CONFLICT: "conflict",
}

# Rich color/styles for DataTable cell badges. These map to the design-system
# semantics (success/warning/error/muted/primary) using standard Rich colors.
_STATUS_TABLE_STYLES: dict[TaskStatus, str] = {
    TaskStatus.WAITING: "bold white on blue",
    TaskStatus.RUNNING: "bold white on green",
    TaskStatus.PAUSED: "bold black on yellow",
    TaskStatus.NEEDS_ATTENTION: "bold black on yellow",
    TaskStatus.BLOCKED: "bold white on red",
    TaskStatus.DISABLED: "bold white on grey50",
    TaskStatus.ARCHIVED: "bold white on grey50",
    TaskStatus.COMPLETED: "bold white on green",
    TaskStatus.FOUND_RESULTS: "bold white on green",
    TaskStatus.MISSED: "bold black on yellow",
    TaskStatus.TIMED_OUT: "bold black on yellow",
    TaskStatus.CONFLICT: "bold white on red",
}


def _humanize_status(status: TaskStatus) -> str:
    """Return a human-readable, capitalized status label."""
    return _STATUS_LABELS.get(status, status.value.replace("_", " ").title())


def _humanize_schedule_kind(kind: ScheduleKind) -> str:
    """Return 'Recurring' or 'One-time' for a schedule kind."""
    return "Recurring" if kind == ScheduleKind.RECURRING else "One-time"


def _format_timezone(dt) -> str:
    """Return a timezone label for a datetime, defaulting to UTC."""
    if dt.tzinfo is None:
        return "UTC"
    return dt.tzname() or "UTC"


def _format_next_run(task: ReminderTask | ScheduledTask | None) -> str:
    """Format a task's next run time with timezone."""
    if task is None or task.next_run_at is None:
        return "-"
    return f"{task.next_run_at.strftime('%Y-%m-%d %H:%M')} {_format_timezone(task.next_run_at)}"


def _format_last_run(task: ReminderTask | ScheduledTask | None) -> str:
    """Format a task's last run time, or 'Never run' / '-' for projections."""
    if task is None:
        return "Never run"
    if isinstance(task, ScheduledTask):
        return "-"
    if task.last_run_at is None:
        return "Never run"
    return f"{task.last_run_at.strftime('%Y-%m-%d %H:%M')} {_format_timezone(task.last_run_at)}"


def _humanize_cron(cron: str | None, timezone: str | None = None) -> str:
    """Summarize a cron expression in plain English."""
    if not cron:
        return "-"
    parts = cron.split()
    if len(parts) != 5:
        return cron
    minute, hour, dom, month, dow = parts
    tz = f" {timezone}" if timezone else " UTC"

    def _is_wildcard(value: str) -> bool:
        return value == "*"

    def _is_digit(value: str) -> bool:
        return value.isdigit()

    if (
        _is_digit(minute)
        and _is_digit(hour)
        and _is_wildcard(dom)
        and _is_wildcard(month)
        and _is_wildcard(dow)
    ):
        return f"Daily at {int(hour):02d}:{int(minute):02d}{tz}"

    if (
        _is_digit(minute)
        and _is_digit(hour)
        and _is_wildcard(dom)
        and _is_wildcard(month)
        and _is_digit(dow)
    ):
        day_index = int(dow)
        if 0 <= day_index <= 6:
            return f"Weekly on {_WEEKDAYS[day_index]} at {int(hour):02d}:{int(minute):02d}{tz}"

    if (
        _is_digit(minute)
        and _is_digit(hour)
        and _is_digit(dom)
        and _is_wildcard(month)
        and _is_wildcard(dow)
    ):
        return f"Monthly on the {int(dom)} at {int(hour):02d}:{int(minute):02d}{tz}"

    return f"cron: {cron}{tz}"


def _humanize_schedule(task: ReminderTask) -> str:
    """Return a human-readable schedule summary for the task."""
    if task.schedule_kind == ScheduleKind.ONE_TIME:
        if task.run_at is None:
            return "One-time"
        return f"One-time at {task.run_at.strftime('%Y-%m-%d %H:%M')} {_format_timezone(task.run_at)}"
    return _humanize_cron(task.cron, task.timezone)


def _task_status(task: ReminderTask | ScheduledTask) -> TaskStatus:
    """Return the current status for either a reminder or a projected task."""
    if isinstance(task, ReminderTask):
        return task.last_status
    return task.status


def _was_missed_while_away(task: ReminderTask | ScheduledTask) -> bool:
    """Return True when the task's last dispatch was late (task-18937).

    This is deliberately NOT a ``TaskStatus``: 'missed' as a status means the
    dispatch ran and the handler raised. A late dispatch is orthogonal to
    that -- it can complete successfully yet still have been missed-while-
    away -- so it derives from the recorded missed-fire state instead of
    overloading the status enum.
    """
    return isinstance(task, ReminderTask) and task.missed_at is not None


def _task_type_label(task: ReminderTask | ScheduledTask) -> str:
    """Return a readable type label for the task."""
    if isinstance(task, ReminderTask):
        return _humanize_schedule_kind(task.schedule_kind)
    return task.type.replace("_", " ").title()


def _task_schedule_label(task: ReminderTask | ScheduledTask) -> str:
    """Return a human-readable schedule summary for the task."""
    if isinstance(task, ReminderTask):
        return _humanize_schedule(task)
    return task.schedule_summary or "-"


def _task_sync_label(task: ReminderTask | ScheduledTask) -> str:
    """Return a sync description for the task."""
    if isinstance(task, ReminderTask):
        sync_status = f"version {task.sync_version}"
        if task.server_id:
            sync_status += f" (server {task.server_id})"
        else:
            sync_status += " (local)"
        return sync_status
    return "local (read-only projection)"


def _task_owner_label(task: ReminderTask | ScheduledTask) -> str:
    """Return an owner label for the task."""
    owner = task.owner_id or "local"
    if isinstance(task, ReminderTask) and task.server_id:
        owner += f" / server {task.server_id}"
    return owner


def status_badge_text(status: TaskStatus) -> Text:
    """Return a styled Rich Text badge for use in a DataTable cell."""
    label = _humanize_status(status)
    style = _STATUS_TABLE_STYLES.get(status, "bold white on grey50")
    return Text(f" {label} ", style=style)


def status_badge_class(status: TaskStatus) -> str:
    """Return the CSS class suffix for a status badge."""
    return _STATUS_BADGE_CLASSES.get(status, "waiting")


class TaskDetail(Vertical):
    """Render the selected reminder task's core details and actions."""

    BUNDLED_CSS = """
    #scheduling-task-detail-metadata {
        height: auto;
        padding: 0;
    }

    #scheduling-task-detail-metadata Horizontal {
        height: auto;
        padding: 0;
        margin: 0;
    }

    .scheduling-detail-label {
        color: $text-muted;
        padding: 0 1 0 0;
        width: 10;
    }

    .scheduling-detail-value {
        color: $text;
    }

    .scheduling-detail-missed {
        color: $warning;
        margin: 0 0 0 11;
        height: auto;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_task: ReminderTask | ScheduledTask | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Task Detail",
            id="scheduling-task-detail-header",
            classes="scheduling-column-title",
        )
        yield Static(
            "Select a task from the queue, or press c to schedule one.",
            id="scheduling-task-detail-empty-state",
        )
        with Vertical(id="scheduling-task-detail-metadata"):
            yield Horizontal(
                Static("Title:", classes="scheduling-detail-label"),
                Static(
                    "-",
                    id="scheduling-task-detail-title",
                    classes="scheduling-detail-value",
                ),
            )
            yield Horizontal(
                Static("Type:", classes="scheduling-detail-label"),
                Static(
                    "-",
                    id="scheduling-task-detail-type",
                    classes="scheduling-detail-value",
                ),
            )
            yield Horizontal(
                Static("Schedule:", classes="scheduling-detail-label"),
                Static(
                    "-",
                    id="scheduling-task-detail-schedule",
                    classes="scheduling-detail-value",
                ),
            )
            yield Horizontal(
                Static("Status:", classes="scheduling-detail-label"),
                Static("-", id="scheduling-task-status-badge"),
            )
            yield Horizontal(
                Static("Next Run:", classes="scheduling-detail-label"),
                Static(
                    "-",
                    id="scheduling-task-detail-next-run",
                    classes="scheduling-detail-value",
                ),
            )
            # task-18937: "missed while away" is its own state -- distinct from
            # a failed dispatch (which ran and raised). Hidden unless the last
            # dispatch was late; plain text, no markup (titles are untrusted).
            yield Static(
                "",
                id="scheduling-task-detail-missed",
                classes="scheduling-detail-missed",
            )
        yield Horizontal(
            Button(
                "Edit",
                id="scheduling-edit-task",
                variant="primary",
                tooltip="Edit this reminder.",
            ),
            Button(
                "Run now",
                id="scheduling-run-now",
                variant="primary",
                tooltip="Dispatch this reminder immediately, without waiting "
                "for its schedule. A real dispatch: a recurring reminder's "
                "next occurrence is computed from now, a one-time reminder "
                "is consumed. Works on disabled reminders without enabling "
                "them.",
            ),
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
            disabled=True,
            tooltip=SCHEDULES_EMPTY_CONSOLE_RECOVERY.disabled_tooltip,
        )
        # Visible when the action is disabled: keyboard users can't see
        # hover tooltips, so the reason must live in text (UX-073).
        yield Static("", id="schedules-follow-why", classes="follow-why")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle lifecycle actions (console follow is handled by the workbench)."""
        button_id = event.button.id
        if button_id in {
            "scheduling-edit-task",
            "scheduling-run-now",
            "scheduling-enable-task",
            "scheduling-disable-task",
            "scheduling-delete-task",
        }:
            event.stop()
        if button_id == "scheduling-edit-task":
            self._request_edit()
        elif button_id == "scheduling-run-now":
            self._request_run_now()
        elif button_id == "scheduling-enable-task":
            self._request_enable()
        elif button_id == "scheduling-disable-task":
            self._request_disable()
        elif button_id == "scheduling-delete-task":
            self.request_delete()

    def _request_edit(self) -> None:
        """Post an edit request for the current reminder."""
        if isinstance(self._current_task, ReminderTask):
            self.post_message(EditTaskRequested(self._current_task))

    def _request_enable(self) -> None:
        """Post an enable request for the current reminder."""
        if isinstance(self._current_task, ReminderTask):
            self.post_message(EnableTaskRequested(self._current_task))

    def _request_disable(self) -> None:
        """Post a disable request for the current reminder."""
        if isinstance(self._current_task, ReminderTask):
            self.post_message(DisableTaskRequested(self._current_task))

    def _request_run_now(self) -> None:
        """Post a run-now request for the current reminder (task-18938)."""
        if isinstance(self._current_task, ReminderTask):
            self.post_message(RunReminderNowRequested(self._current_task))

    def request_delete(self) -> None:
        """Open the delete confirmation modal for the current task."""
        if self._current_task is None:
            return
        self.app.push_screen(
            DeleteConfirmationDialog(
                item_type="Scheduled task",
                item_name=self._current_task.title,
                permanent=True,
            ),
            callback=self._on_delete_confirmed,
        )

    def _on_delete_confirmed(self, confirmed: Any | None) -> None:
        """Post a delete request when the user confirms the modal."""
        if confirmed and isinstance(self._current_task, ReminderTask):
            self.post_message(DeleteTaskRequested(self._current_task))

    def set_task(
        self, task: ReminderTask | ScheduledTask | None, *, queue_empty: bool = False
    ) -> None:
        """Update the detail view for the given task (or clear it)."""
        self._current_task = task
        metadata = self.query_one("#scheduling-task-detail-metadata", Vertical)
        lifecycle = self.query_one("#scheduling-task-detail-lifecycle", Horizontal)
        self.query_one("#schedules-follow-in-console", Button)
        empty_state = self.query_one("#scheduling-task-detail-empty-state", Static)

        if task is None:
            empty_copy = (
                "No scheduled tasks yet. Press c to schedule your first task."
                if queue_empty
                else "Select a task from the queue, or press c to schedule one."
            )
            empty_state.update(empty_copy)
            empty_state.display = True
            metadata.display = False
            lifecycle.display = False
            missed_notice = self.query_one("#scheduling-task-detail-missed", Static)
            missed_notice.update("")
            missed_notice.display = False
            self.query_one("#schedules-follow-in-console", Button).label = (
                "Follow in Console"
            )
            return

        empty_state.display = False
        metadata.display = True
        lifecycle.display = isinstance(task, ReminderTask)

        follow_button = self.query_one("#schedules-follow-in-console", Button)
        short_title = task.title if len(task.title) <= 24 else f"{task.title[:23]}…"
        follow_button.label = f"Follow '{short_title}' in Console"

        # Only the action that would change the current state stays enabled;
        # the other is visibly disabled instead of silently no-oping (UX-059).
        if isinstance(task, ReminderTask):
            enabled = bool(getattr(task, "enabled", True))
            self.query_one("#scheduling-enable-task", Button).disabled = enabled
            self.query_one("#scheduling-disable-task", Button).disabled = not enabled
            # The retry affordance for a failed dispatch (task-18938): a
            # reminder whose last dispatch ran and raised (or was cancelled
            # at its execution deadline, task-18939) offers Run now as its
            # retry -- the never-wired "Retry run" concept, now real.
            run_now_button = self.query_one("#scheduling-run-now", Button)
            if _task_status(task) in {TaskStatus.MISSED, TaskStatus.TIMED_OUT}:
                run_now_button.label = "Run now (retry)"
                run_now_button.tooltip = (
                    "Retry this reminder now: its last dispatch ran and "
                    "failed. Dispatches immediately through the same path "
                    "the scheduler uses."
                )
            else:
                run_now_button.label = "Run now"

        self._update_static("scheduling-task-detail-title", task.title)
        self._update_static("scheduling-task-detail-type", _task_type_label(task))
        self._update_static(
            "scheduling-task-detail-schedule", _task_schedule_label(task)
        )
        self._update_static("scheduling-task-detail-next-run", _format_next_run(task))
        self._update_missed_notice(task)

        status = _task_status(task)
        badge = self.query_one("#scheduling-task-status-badge", Static)
        badge.update(_humanize_status(status))
        badge.remove_class(*_STATUS_BADGE_CLASSES.values())
        badge.add_class(status_badge_class(status))

    def _update_missed_notice(
        self, task: ReminderTask | ScheduledTask | None
    ) -> None:
        """Render the "missed while away" notice for a late last dispatch.

        Distinct from failed: failed means the dispatch ran and the handler
        raised; missed-while-away means the scheduled time elapsed with the
        scheduler not running (app closed, or the occurrence predates the
        task reaching the queue). The notice describes the last dispatch and
        self-heals: the next on-time dispatch clears it. Plain text, no
        markup -- titles are untrusted and never interpolated into markup.
        """
        notice = self.query_one("#scheduling-task-detail-missed", Static)
        if not isinstance(task, ReminderTask):
            notice.update("")
            notice.display = False
            return
        missed_at = getattr(task, "missed_at", None)
        if missed_at is None:
            notice.update("")
            notice.display = False
            return
        scheduled = missed_at.strftime("%Y-%m-%d %H:%M")
        missed_count = int(getattr(task, "missed_count", 0) or 0)
        if missed_count < 0:
            # Sentinel from the counting cap: more occurrences elapsed than
            # the counter will enumerate. Rendered as an explicit "more than
            # N", never as a false exact number.
            from tldw_chatbook.Scheduling.db.scheduled_tasks_db import (
                ScheduledTasksDB,
            )

            copy = (
                f"Missed while away: ran late for the {scheduled} occurrence; "
                f"more than {ScheduledTasksDB._MISSED_COUNT_CAP:,} earlier "
                "occurrence(s) were skipped, not replayed."
            )
        elif missed_count > 0:
            copy = (
                f"Missed while away: ran late for the {scheduled} occurrence; "
                f"{missed_count} earlier occurrence(s) were skipped, not replayed."
            )
        else:
            copy = (
                f"Missed while away: the {scheduled} occurrence ran late "
                "(the scheduler was not running at the scheduled time)."
            )
        notice.update(copy)
        notice.display = True

    def set_follow_available(self, available: bool) -> None:
        """Enable or disable the Console-follow button and set its tooltip."""
        button = self.query_one("#schedules-follow-in-console", Button)
        button.disabled = not available
        button.tooltip = (
            "Open the active schedule run in Console."
            if available
            else SCHEDULES_EMPTY_CONSOLE_RECOVERY.disabled_tooltip
        )
        # Keyboard users can't see the tooltip; the reason goes in text.
        try:
            why = self.query_one("#schedules-follow-why", Static)
            why.update(
                ""
                if available
                else SCHEDULES_EMPTY_CONSOLE_RECOVERY.disabled_tooltip
            )
        except Exception:  # noqa: BLE001 - widget not mounted yet
            pass

    def _update_static(self, widget_id: str, content: str) -> None:
        """Update a child Static widget by id."""
        static = self.query_one(f"#{widget_id}", Static)
        static.update(content)


class TaskInspector(Vertical):
    """Render sync, conflict, and last-run metadata for a task."""

    def compose(self) -> ComposeResult:
        yield Static(
            "Inspector",
            id="scheduling-task-inspector-header",
            classes="scheduling-column-title",
        )
        with Vertical(id="scheduling-inspector-metadata"):
            yield Horizontal(
                Static("Sync:", classes="scheduling-inspector-label"),
                Static(
                    "-",
                    id="scheduling-inspector-sync",
                    classes="scheduling-inspector-value",
                ),
            )
            yield Horizontal(
                Static("Last Run:", classes="scheduling-inspector-label"),
                Static(
                    "-",
                    id="scheduling-inspector-last-run",
                    classes="scheduling-inspector-value",
                ),
            )
            yield Horizontal(
                Static("Owner:", classes="scheduling-inspector-label"),
                Static(
                    "-",
                    id="scheduling-inspector-owner",
                    classes="scheduling-inspector-value",
                ),
            )
        yield Vertical(
            Static("No conflict", id="scheduling-conflict-text"),
            id="scheduling-conflict-card",
        )

    def set_task(self, task: ReminderTask | ScheduledTask | None) -> None:
        """Update the inspector view for the given task (or clear it)."""
        if task is None:
            self._update_static("scheduling-inspector-sync", "-")
            self._update_static("scheduling-inspector-last-run", "-")
            self._update_static("scheduling-inspector-owner", "-")
            self._update_conflict_card(None)
            return

        self._update_static("scheduling-inspector-sync", _task_sync_label(task))
        self._update_static("scheduling-inspector-last-run", _format_last_run(task))
        self._update_static("scheduling-inspector-owner", _task_owner_label(task))
        self._update_conflict_card(task)

    def _update_conflict_card(self, task: ReminderTask | ScheduledTask | None) -> None:
        """Update the conflict card for the current task state."""
        card = self.query_one("#scheduling-conflict-card", Vertical)
        text = self.query_one("#scheduling-conflict-text", Static)
        if task is not None and _task_status(task) == TaskStatus.CONFLICT:
            text.update(f"Conflict detected\n{task.title}")
            card.add_class("conflict")
        else:
            text.update("No conflict")
            card.remove_class("conflict")

    def _update_static(self, widget_id: str, content: str) -> None:
        """Update a child Static widget by id."""
        static = self.query_one(f"#{widget_id}", Static)
        static.update(content)
