"""Detail and inspector widgets for the Scheduling workbench."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Select, Static

from ....Scheduling.events import (
    DeleteTaskRequested,
    DisableTaskRequested,
    AcknowledgeIncidentRequested,
    EditTaskRequested,
    EnableTaskRequested,
    ReminderFieldEditRequested,
    ReminderOwnerActionRequested,
    RunReminderNowRequested,
)
from ....Scheduling.models import ReminderTask, ScheduledTask, ScheduleKind, TaskStatus
# PR-3 task 5: the owner-row dropdown's own lock/failed-state gating reads
# the SAME state tuple `SchedulingService.transfer_lock_reason` keys off
# (survey §3) -- a plain constant import, not a fresh boot-cost tier
# (`scheduled_tasks_db` is already loaded via `scheduling_service.py`
# by the time this lazy-loaded Scheduling screen is up, same reasoning
# Task 3's own `reminder_form`/`schedule_input_parsing` imports rely on).
from ....Scheduling.db.scheduled_tasks_db import IN_FLIGHT_TRANSFER_STATES
from ....Widgets.delete_confirmation_dialog import DeleteConfirmationDialog
from ....Widgets.detail_value_row import DetailGroup, DetailValueRow
from ..destination_recovery import DestinationRecoveryState
# PR-3 task 3: the Frequency row editors reuse the create/edit modal's own
# preset<->cron mapping and timezone-option builder verbatim (never
# re-derived) -- `reminder_form.py` is already part of this same lazy-
# loaded Scheduling-screen import chain (`schedules_workbench.py` imports
# it at module level too), so this adds no new boot-cost tier (ADR-097).
from .forms.reminder_form import (
    DEFAULT_TIME_OF_DAY,
    ReminderForm,
    cron_to_preset,
    preset_to_cron,
    timezone_options,
)
# Hoisted to `unified_rows.py` (redesign PR-2 Task 1) so that pure module can
# reuse them without pulling Textual in as an import side effect; re-exported
# here unchanged so every existing call site/test import keeps working.
from .unified_rows import (
    _format_timezone,
    _humanize_cron,  # noqa: F401  (re-export: definition_detail.py + tests import this from here)
    _humanize_schedule,
    definition_cron_expression,  # noqa: F401  (re-export: definition_detail.py imports this from here)
    owner_display_label,
)


#: Stable ids for the Frequency row editors (PR-3 task 3), so the
#: `on_select_changed`/`on_input_submitted` handlers can route by id --
#: same "filter on `.id`" idiom `on_button_pressed` already uses in this
#: file for its own action buttons.
_REPEAT_EDITOR_ID = "scheduling-detail-repeat-editor"
_AT_EDITOR_ID = "scheduling-detail-at-editor"
_TIMEZONE_EDITOR_ID = "scheduling-detail-timezone-editor"

#: Repeat's "custom" preset has no single-value edit target here (the raw
#: cron field only exists in the full modal) -- shown so the row's current
#: value round-trips (Select requires the initial value be among its
#: options), but selecting it as a NEW target is refused with this copy
#: rather than silently doing nothing (ruling 2: never silent).
_REPEAT_CUSTOM_REFUSAL = "Use the full Edit form to set a custom cron expression."

#: Stable ids for the owner-row transfer dropdown (PR-3 task 5) and its
#: proactively-shown Cancel/Retry mini-bar -- same "filter on `.id`" idiom
#: the Frequency editors above already use.
_RUNS_ON_EDITOR_ID = "scheduling-detail-runs-on-editor"
_RUNS_ON_CANCEL_ID = "scheduling-detail-runs-on-cancel"
_RUNS_ON_RETRY_ID = "scheduling-detail-runs-on-retry"


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


def _format_relative(next_run_at: datetime, now: datetime) -> str:
    """Render the distance to ``next_run_at`` as plain prose (task-23111).

    Naive datetimes are treated as UTC, matching ``_format_timezone``'s
    labeling of naive values.
    """
    if next_run_at.tzinfo is None:
        next_run_at = next_run_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    seconds = (next_run_at - now).total_seconds()
    overdue = seconds < 0
    seconds = abs(seconds)
    if seconds < 60:
        return "due now"
    if seconds < 3600:
        amount = f"{int(seconds // 60)}m"
    elif seconds < 2 * 86400:
        amount = f"{int(seconds // 3600)}h"
    else:
        amount = f"{int(seconds // 86400)}d"
    return f"overdue {amount}" if overdue else f"in {amount}"


def _format_next_run(
    task: ReminderTask | ScheduledTask | None,
    *,
    now: datetime | None = None,
    compact: bool = False,
) -> str:
    """Format a task's next run time with a relative form alongside.

    A disabled reminder will not run at its stored ``next_run_at``, so a
    concrete future time would be a false promise (task-23101). The
    detail pane uses the full form ("2026-08-28 09:00 UTC (in 14h)");
    the queue column passes ``compact=True`` to drop the timezone token
    and keep the column width sane (task-23111). ``now`` is injectable
    for deterministic tests.
    """
    if task is None:
        return "-"
    # Suppression covers projections too (review F6): a watchlist row
    # whose projection maps is_active False -> DISABLED still carries a
    # computed next_run_at it will not honor.
    #
    # This runs BEFORE the empty-next_run_at check on purpose: dispatching
    # a one-time reminder disables it AND clears next_run_at, so testing
    # the timestamp first made a completed task read "-" while its status
    # badge said "Disabled" (Qodo review). Suppression is a property of the
    # status, not of whether a stale timestamp happens to survive.
    display_status = _task_status(task)
    if display_status == TaskStatus.DISABLED:
        return "— (disabled)"
    if display_status == TaskStatus.PAUSED:
        return "— (paused)"
    # An ENABLED task with nothing scheduled genuinely has no next run.
    if task.next_run_at is None:
        return "-"
    reference = now if now is not None else datetime.now(timezone.utc)
    absolute = task.next_run_at.strftime("%Y-%m-%d %H:%M")
    relative = _format_relative(task.next_run_at, reference)
    if compact:
        return f"{absolute} ({relative})"
    return f"{absolute} {_format_timezone(task.next_run_at)} ({relative})"


def _format_last_run(task: ReminderTask | ScheduledTask | None) -> str:
    """Format a task's last run time, or 'Never run' / '-' for projections."""
    if task is None:
        return "Never run"
    if isinstance(task, ScheduledTask):
        return "-"
    if task.last_run_at is None:
        return "Never run"
    return f"{task.last_run_at.strftime('%Y-%m-%d %H:%M')} {_format_timezone(task.last_run_at)}"


def format_run_history(runs) -> str:
    """TASK-26026: a compact multi-line run history, newest first.

    ``runs`` is the ``list_task_runs`` shape (dicts with status/started_at/
    error_msg). Empty/None reads as "No runs recorded yet" -- distinct from
    a task that has a last_status but no ledger rows (a pre-ledger task).
    """
    if not runs:
        return "No runs recorded yet"
    lines = []
    for run in runs[:8]:
        started = str(run.get("started_at") or "?")[:16].replace("T", " ")
        status = str(run.get("status") or "?")
        error = run.get("error_msg")
        suffix = f" — {str(error)[:80]}" if error else ""
        lines.append(f"{started}  {status}{suffix}")
    return "\n".join(lines)


def format_incidents(incidents) -> str:
    """TASK-26027: a compact incident list, newest first.

    Shows only OPEN incidents (alerting/acknowledged) -- a closed incident
    is resolved and not actionable. Empty reads as "No open incidents".
    """
    if not incidents:
        return "No open incidents"
    open_rows = [
        row for row in incidents if str(row.get("status")) != "closed"
    ]
    if not open_rows:
        return "No open incidents"
    lines = []
    for row in open_rows[:5]:
        status = str(row.get("status") or "?")
        count = row.get("occurrence_count") or 1
        sig = str(row.get("signature") or "")[:80]
        lines.append(f"[{status} ×{count}] {sig}")
    return "\n".join(lines)


def _underlying_status(task: ReminderTask | ScheduledTask) -> TaskStatus:
    """The recorded dispatch status, without the enabled-state overlay.

    Behavior checks (retry affordance, conflict card, text filter) must
    consult this where the recorded outcome is the honest answer for a
    disabled row too (task-23101 review F5).
    """
    if isinstance(task, ReminderTask):
        return task.last_status
    return task.status


def _task_status(task: ReminderTask | ScheduledTask) -> TaskStatus:
    """Return the DISPLAY status for either a reminder or a projected task.

    A disabled reminder reads as Disabled regardless of its last dispatch
    outcome: disabling never touches ``last_status``, so deriving from it
    left disabled rows showing "Waiting" (task-23101). Enabling restores
    the recorded last outcome. Consumers that need the recorded outcome
    itself use ``_underlying_status`` (review F5).
    """
    if isinstance(task, ReminderTask) and not task.enabled:
        return TaskStatus.DISABLED
    return _underlying_status(task)


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


#: Which screen owns each read-only projection row (task-23106). Briefings
#: are configured from the Watchlists screen, so both point there.
_PROJECTION_MANAGERS: dict[str, str] = {
    "watchlist_job": "Watchlists",
    "briefing_job": "Watchlists",
}


def _managed_elsewhere_notice(
    task: ScheduledTask, verb: str = "edit"
) -> str:
    """Copy for rows managed by another system (task-23106).

    Schedules shows these rows read-only; the copy names the owning
    screen instead of exposing the internal reminder/projection split.
    """
    manager = _PROJECTION_MANAGERS.get(task.type)
    if manager:
        return f"Managed by {manager} — {verb} it there."
    return (
        f"Managed by another screen — this row is read-only here; "
        f"{verb} it where it was created."
    )


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
    """Return the RAW owner label for the task (TaskInspector's Owner row).

    Prose-rendered surfaces use `owner_display_label` instead.
    """
    owner = task.owner_id or "local"
    if isinstance(task, ReminderTask) and task.server_id:
        owner += f" / server {task.server_id}"
    return owner


def transfer_row_dict(task: ReminderTask) -> dict[str, Any]:
    """Build the raw-row dict `SchedulingService.transfer_refusal`/
    `transfer_warnings` expect from an already-loaded `ReminderTask`
    (schedules-handoff spec §6.4, PR-5 task 7).

    Both facade functions were written against real DB rows (ISO date
    strings, plain enum values) -- this reshapes the in-memory Pydantic
    model to match without a DB round-trip, since every field they read
    (`owner_id`, `server_id`, `transfer_state`, `schedule_kind`, `run_at`,
    `timeout_seconds`) is already on the model.
    """
    return {
        "owner_id": task.owner_id,
        "server_id": task.server_id,
        "transfer_state": task.transfer_state,
        "schedule_kind": task.schedule_kind.value,
        "run_at": task.run_at.isoformat() if task.run_at else None,
        "timeout_seconds": task.timeout_seconds,
    }


#: Minimal queue-row signal that a transfer is in flight (spec §9's badge
#: language, pulled forward from PR-6 only far enough to keep PR-5's state
#: machine from being silently inert -- plan ruling 1 keeps full badge/
#: owner-column polish out of scope here).
_TRANSFER_STATE_ROW_LABELS: dict[str, str] = {
    "to_server_pending": "Moving to server…",
    "to_server_sent": "Moving to server…",
    "from_server_pending": "Waiting for server release",
    "to_server_failed": "Transfer failed — retry/cancel",
}


def _transfer_row_suffix(task: ReminderTask | ScheduledTask) -> str:
    """Return a queue-row title suffix for an in-flight transfer, or ``""``."""
    if not isinstance(task, ReminderTask):
        return ""
    label = _TRANSFER_STATE_ROW_LABELS.get(task.transfer_state or "")
    return f" ({label})" if label else ""


def _queue_owner_suffix(task: ReminderTask | ScheduledTask, *, compact: bool) -> str:
    """Return a queue-row title owner suffix, or ``""`` (plan ruling 4).

    Same append-a-parenthetical idiom as `_transfer_row_suffix` above and
    the same wording as `results_tab._result_owner_suffix` (schedules-
    handoff PR-6 task 3) -- a local row says nothing, a server-scoped row
    gets ``" (server: <id>)"``. Hidden at compact width: the compact
    layout (`SCHEDULES_COMPACT_WORKBENCH_MAX_WIDTH`) already trims panes
    to fit narrow terminals, and this suffix would be the next thing to
    overflow the row.
    """
    if compact:
        return ""
    # ADR-097: scheduler.queue stays off the boot census -- imported
    # function-locally everywhere else this helper is needed too
    # (schedules_workbench.py, scheduling_service.py, results_tab.py).
    from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

    owner_id = task.owner_id
    if not is_server_scoped_owner(owner_id):
        return ""
    owner_id = str(owner_id)
    label = owner_id.split(":", 1)[1] if ":" in owner_id else owner_id
    return f" (server: {label})"


#: spec §5's Frequency-group "Notifications" row (schedules-redesign PR-1,
#: task 3). A reminder dispatch always writes an inbox notification and
#: attempts a transient toast (`ReminderHandler.handle` ->
#: `NotificationDispatchService.dispatch`, `Scheduling/scheduler/handlers/
#: reminder_handler.py`) -- there is no per-reminder notification-channel
#: config to read (survey §4), so this is a fixed label, never derived.
_REMINDER_NOTIFICATIONS_LABEL = "Inbox + toast"


def _reminder_runs_on_label(task: ReminderTask) -> str:
    """'Runs on' row value (spec §5 Details group): the shared prose owner
    label plus the existing in-flight transfer badge text, when a transfer
    is running.

    Final review F6/F7: this used to render `_task_owner_label`'s raw
    metadata string (``local``), which neither matched the definitions
    pane's ``This device`` nor the User Guide's own description of this
    very row. Both panes now go through `owner_display_label`.
    """
    return owner_display_label(task.owner_id) + _transfer_row_suffix(task)


def _reminder_repeat_label(task: ReminderTask) -> str:
    """'Repeat' row value (Frequency group): the schedule kind -- the same
    content the old Type row showed for a reminder, via the same helper
    (`_humanize_schedule_kind`, unchanged).
    """
    return _humanize_schedule_kind(task.schedule_kind)


def _reminder_at_label(task: ReminderTask) -> str:
    """'At' row value (Frequency group): the full schedule summary -- the
    same content the old Schedule row showed, via the same helper
    (`_humanize_schedule`, unchanged). The task-3 brief requires reusing
    the current field-formatting helpers verbatim rather than re-deriving
    a cadence-only or time-only string from the cron.
    """
    return _humanize_schedule(task)


def _reminder_timezone_label(task: ReminderTask) -> str:
    """'Timezone' row value (Frequency group), reusing the same per-kind
    timezone source `_humanize_schedule`'s own formatting already reads
    from: `run_at`'s zone for one-time, the stored cron timezone for
    recurring.
    """
    if task.schedule_kind == ScheduleKind.ONE_TIME:
        return _format_timezone(task.run_at) if task.run_at is not None else "UTC"
    return task.timezone or "UTC"


def _reminder_last_fire_label(task: ReminderTask) -> str:
    """'Last fire' row value (History group): last_run_at plus the recorded
    outcome. Uses the underlying status (task-23101 review F5), not the
    enabled-overlaid display status, so a disabled reminder's last real
    outcome still shows here -- same discipline as `_update_missed_notice`.
    """
    if task.last_run_at is None:
        return _format_last_run(task)  # "Never run"
    return f"{_format_last_run(task)} — {_humanize_status(_underlying_status(task))}"


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
        # schedules-redesign PR-1, task 3: reminder-only DetailValueRow
        # refs, captured in compose() and refreshed in place by set_task
        # (no recompose). None until first compose().
        self._runs_on_row: DetailValueRow | None = None
        self._repeat_row: DetailValueRow | None = None
        self._at_row: DetailValueRow | None = None
        self._timezone_row: DetailValueRow | None = None
        self._last_fire_row: DetailValueRow | None = None
        self._body_card: Static | None = None
        # PR-3 task 3: cached from `set_lifecycle_lock` (never re-derived
        # here, same "one source of truth" rule survey §8 names) so the
        # Frequency rows' `Activated` handler can tell a locked row apart
        # from an editable one without a second `transfer_lock_reason`
        # call -- the workbench already computed this once per selection.
        self._lifecycle_lock_reason: str | None = None
        #: Zones already used by other tasks (task-23102), threaded in by
        #: the workbench's own `set_task` call so the Timezone row editor
        #: offers the same option set the create/edit modal does. Empty
        #: by default: every other `set_task` caller (including every
        #: pre-task-3 test) keeps working unchanged.
        self._known_timezones: Sequence[str] = ()
        #: PR-3 task 5: the workbench's own `_runs_on_options()` result
        #: (hoisted the SAME way `known_timezones` already is -- the
        #: workbench holds the service/server state needed to compute it,
        #: this widget does not). Empty by default: every pre-task-5
        #: `set_task` caller/test keeps working unchanged.
        self._runs_on_options: Sequence[tuple[str, str]] = ()
        #: PR-3 task 5: the Runs-on row's proactive Cancel/Retry
        #: affordances -- plain always-mounted `Button`s toggled via
        #: `.display` (never `begin_edit`-swapped INTO the row: that
        #: would hide `#scheduling-detail-runs-on`'s own value Static,
        #: which the existing transfer-badge rendering pins verbatim --
        #: see `_configure_runs_on_row`). `None` until first `compose()`.
        self._runs_on_cancel_button: Button | None = None
        self._runs_on_retry_button: Button | None = None
        #: PR-3 task 5 fix round 1 (finding 2): a `to_server_failed`
        #: row's stored `transfer_errors`, threaded in by the workbench
        #: (`set_runs_on_transfer_errors`) the SAME way it already feeds
        #: the legacy Retry button's own reason line -- read by
        #: `_runs_on_failure_reason` when the row is activated.
        self._runs_on_transfer_errors: list[str] = []
        #: Final review I2: the id of the reminder an open row editor was
        #: opened AGAINST, captured at `begin_edit` time and validated at
        #: commit time by `_editing_task`. Never reset by a repaint -- a
        #: commit that crossed a repaint in flight has to still see the
        #: id it was opened for, so it can be discarded instead of
        #: writing one row's typed value onto another.
        self._editing_task_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Task Detail",
            id="scheduling-task-detail-header",
            classes="scheduling-column-title",
        )
        yield Static(
            "Select a task from the queue, or press n to schedule one.",
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
            # task-3 brief: the old combined Type/Schedule rows only apply
            # to a `ScheduledTask` projection (watchlist_job/briefing_job)
            # now -- reminders render Repeat/At/Timezone/Notifications
            # through the Frequency `DetailGroup` below instead. `set_task`
            # toggles this container's `.display` per task type.
            with Vertical(id="scheduling-task-detail-legacy-fields"):
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
            # schedules-redesign PR-1, task 3 (spec §5 reminder column):
            # Details/Frequency/History groups. Rows are read-only in this
            # PR (plan ruling 1: `affordance` stays at its False default,
            # no new bindings). `set_task` shows this container only for a
            # `ReminderTask`; a `ScheduledTask` projection keeps the
            # legacy Type/Schedule rows above instead.
            with Vertical(id="scheduling-task-detail-groups"):
                # Spec §5 wants the body text in a rounded card above the
                # groups, exactly like the definitions pane's question
                # card -- final review F10: it was never built, and
                # `ReminderTask.body` was rendered nowhere. Same
                # `markup=False` escape discipline (reminder bodies are
                # user text); hidden entirely for a body-less reminder
                # rather than showing an empty bordered box.
                self._body_card = Static(
                    "", id="scheduling-task-detail-body-card", markup=False
                )
                yield self._body_card
                # `row_key` (final review M12): the per-row identity the
                # workbench's edit worker groups its commit under, so two
                # quick edits on DIFFERENT rows stop cancelling each
                # other. Set on every editable row, nowhere else.
                self._runs_on_row = DetailValueRow(
                    "Runs on",
                    "-",
                    value_id="scheduling-detail-runs-on",
                    row_key="runs_on",
                )
                # PR-3 task 5: the Runs-on row's own proactive Cancel/
                # Retry affordances for an in-flight/failed transfer --
                # a plain sibling `Horizontal`, hidden by default
                # (`_configure_runs_on_row` toggles `.display`), NOT
                # `begin_edit`-mounted into the row itself (that would
                # hide the row's own value Static, which the existing
                # transfer-badge rendering pins verbatim).
                self._runs_on_cancel_button = Button(
                    "Cancel transfer", id=_RUNS_ON_CANCEL_ID, variant="warning", classes="detail-owner-action-button"
                )
                self._runs_on_retry_button = Button(
                    "Retry transfer", id=_RUNS_ON_RETRY_ID, variant="warning", classes="detail-owner-action-button"
                )
                runs_on_actions = Horizontal(
                    self._runs_on_cancel_button,
                    self._runs_on_retry_button,
                    classes="detail-value-row-owner-actions",
                )
                yield DetailGroup(
                    self._runs_on_row,
                    runs_on_actions,
                    title="Details",
                    id="scheduling-detail-group-details",
                )
                self._repeat_row = DetailValueRow(
                    "Repeat",
                    "-",
                    value_id="scheduling-detail-repeat",
                    row_key="cron",
                )
                self._at_row = DetailValueRow(
                    "At", "-", value_id="scheduling-detail-at", row_key="run_at"
                )
                self._timezone_row = DetailValueRow(
                    "Timezone",
                    "-",
                    value_id="scheduling-detail-timezone",
                    row_key="timezone",
                )
                notifications_row = DetailValueRow(
                    "Notifications",
                    _REMINDER_NOTIFICATIONS_LABEL,
                    value_id="scheduling-detail-notifications",
                )
                yield DetailGroup(
                    self._repeat_row,
                    self._at_row,
                    self._timezone_row,
                    notifications_row,
                    title="Frequency",
                    id="scheduling-detail-group-frequency",
                )
                self._last_fire_row = DetailValueRow(
                    "Last fire", "-", value_id="scheduling-detail-last-fire"
                )
                # Final review N2: labelled "Recent runs" pointing at a
                # section whose own label is also "Recent runs" -- one of
                # the two had to be renamed.
                history_link_row = DetailValueRow(
                    "Run history",
                    "See list below",
                    value_id="scheduling-detail-history-link",
                )
                yield DetailGroup(
                    self._last_fire_row,
                    history_link_row,
                    title="History",
                    collapsed=True,
                    id="scheduling-detail-group-history",
                )
            # TASK-26026: durable per-dispatch run history -- the whole
            # point is that run N-1 is recoverable, not just the latest.
            yield Static(
                "Recent runs:", classes="scheduling-detail-label"
            )
            yield Static(
                "No runs recorded yet",
                id="scheduling-task-detail-run-history",
                classes="scheduling-detail-value",
            )
            # TASK-26027: open failure incidents + an acknowledge action.
            yield Static(
                "Open incidents:", classes="scheduling-detail-label"
            )
            yield Static(
                "No open incidents",
                id="scheduling-task-detail-incidents",
                classes="scheduling-detail-value",
            )
        yield Horizontal(
            Button(
                "Edit",
                id="scheduling-edit-task",
                variant="primary",
                tooltip="Edit this scheduled task.",
            ),
            Button(
                "Acknowledge incident",
                id="scheduling-ack-incident",
                tooltip="Silence notifications for the current failure "
                "incident until it recurs after a success. Does not disable "
                "the task.",
            ),
            Button(
                "Run now",
                id="scheduling-run-now",
                variant="primary",
                tooltip="Dispatch this scheduled task immediately, without "
                "waiting for its schedule. A real dispatch: a recurring "
                "task's next occurrence is computed from now, a one-time "
                "task is consumed. Works on disabled tasks without enabling "
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
        # redesign PR-4, task 4 (ruling 2): the legacy Move/Retry/Cancel
        # button row that lived here (schedules-handoff spec §6, PR-5
        # task 7) is RETIRED -- the Runs-on row's own dropdown + mini-bar
        # (`_runs_on_cancel_button`/`_runs_on_retry_button` above) is now
        # the one transfer surface (PR-3 task 5's "coexistence pinned"
        # window is over). `#scheduling-transfer-why` stays: `set_
        # lifecycle_lock` still writes the Edit/Enable/Disable/Delete
        # read-only reason into it (UX-073 -- keyboard users can't see
        # hover tooltips).
        yield Static("", id="scheduling-transfer-why", classes="follow-why")
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
            "scheduling-ack-incident",
            _RUNS_ON_CANCEL_ID,
            _RUNS_ON_RETRY_ID,
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
        elif button_id == "scheduling-ack-incident":
            self._request_acknowledge()
        elif button_id == _RUNS_ON_CANCEL_ID:
            self._request_runs_on_cancel()
        elif button_id == _RUNS_ON_RETRY_ID:
            self._request_runs_on_retry()

    def _sync_acknowledge_button(self) -> None:
        """Enable the acknowledge button only when an alerting incident exists."""
        try:
            button = self.query_one("#scheduling-ack-incident", Button)
        except Exception:  # noqa: BLE001 -- absent before mount
            return
        incidents = getattr(self, "_current_incidents", []) or []
        alerting = [
            row for row in incidents if str(row.get("status")) == "alerting"
        ]
        button.disabled = not alerting
        button.display = bool(alerting)

    def _request_acknowledge(self) -> None:
        """Post an acknowledge request for the newest alerting incident."""
        incidents = getattr(self, "_current_incidents", []) or []
        alerting = [
            row for row in incidents if str(row.get("status")) == "alerting"
        ]
        if not alerting:
            return
        incident_id = alerting[0].get("id")
        if incident_id is not None:
            self.post_message(AcknowledgeIncidentRequested(int(incident_id)))

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

    def _request_runs_on_cancel(self) -> None:
        """Post a cancel request from the Runs-on row's own mini-bar
        (PR-3 task 5) -- the ONE transfer surface now (redesign PR-4 task
        4 retired the legacy `_request_cancel_transfer` this docstring
        used to contrast with): renders a refusal via `row.show_error`,
        not a toast."""
        task = self._current_task
        row = self._runs_on_row
        if isinstance(task, ReminderTask) and row is not None:
            self.post_message(ReminderOwnerActionRequested(task, "cancel", row))

    def _request_runs_on_retry(self) -> None:
        """Post a retry request from the Runs-on row's own mini-bar
        (PR-3 task 5) -- the PR-5 retry leg (re-begin)."""
        task = self._current_task
        row = self._runs_on_row
        if isinstance(task, ReminderTask) and row is not None:
            self.post_message(ReminderOwnerActionRequested(task, "retry", row))

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

    # -- Frequency row in-pane editing (PR-3 task 3) -------------------------

    def _configure_frequency_editability(self, task: ReminderTask) -> None:
        """Wire each Frequency row's affordance to whether it has a real
        single-field edit target for the task's CURRENT schedule kind.

        Repeat (regenerates `cron`) and Timezone (`timezone`) only apply
        to a recurring schedule; At (`run_at`) only applies to a one-time
        one. Editing the "wrong" one for the current kind is not merely
        unhelpful -- `update_reminder`'s own recompute step (survey §2)
        silently clobbers it back (a `cron`/`timezone` write on a
        one-time row is reset to `None`; a `run_at` write on a recurring
        row is reset to `None` too), so offering the affordance there
        would be a guaranteed-to-silently-fail control. Notifications has
        no backing field at all (survey §2 -- a reminder dispatch always
        writes the same fixed inbox+toast, nothing per-row to persist)
        and stays permanently read-only; `Runs on` is out of this task's
        row list entirely.

        Locked rows keep their affordance ON (not off): ruling 2 requires
        activation to still respond with the lock reason via `show_error`
        rather than going silent, and `on_detail_value_row_activated`
        checks `self._lifecycle_lock_reason` before ever opening an
        editor -- so the affordance glyph staying lit is what makes that
        reachable at all, not a bug.
        """
        recurring = task.schedule_kind == ScheduleKind.RECURRING
        for row, editable in (
            (self._repeat_row, recurring),
            (self._at_row, not recurring),
            (self._timezone_row, recurring),
        ):
            assert row is not None
            row.affordance = editable
            row.can_focus = editable

    def _configure_runs_on_row(self, task: ReminderTask) -> None:
        """Wire the Runs-on row's Cancel/Retry button visibility (PR-3
        task 5).

        `row.affordance`/`can_focus` stay ON unconditionally (ruling 3:
        "dropdown always renders") -- unlike the Frequency rows' kind
        gating, nothing about the owner row ever makes activation itself
        unreachable; `on_detail_value_row_activated` is what decides
        whether activating it opens the dropdown or shows why not (fix
        round 1 finding 2: a locked/failed row must not go silently
        inert, same idiom the Frequency rows already use for their own
        lock reason).

        The Cancel[/Retry] buttons are plain always-mounted siblings of
        the row, toggled via `.display` -- deliberately NOT `begin_edit`-
        mounted INTO the row: `begin_edit` hides the row's own value
        `Static` (`#scheduling-detail-runs-on`), and the existing
        transfer-badge rendering (`_reminder_runs_on_label`'s suffix,
        painted into that SAME Static by `set_task` above) is pinned
        verbatim by an existing test -- this row's value must stay
        readable while the actions show, not be replaced by them.
        `to_server_failed` shows BOTH (`cancel_refusal`/the PR-5 retry
        leg both allow this substate, mirroring the existing buttons'
        own Retry-alongside-Cancel visibility rule); any other in-flight
        state shows Cancel only.
        """
        row = self._runs_on_row
        assert row is not None
        row.affordance = True
        row.can_focus = True
        state = task.transfer_state
        failed = state == "to_server_failed"
        locked = failed or state in IN_FLIGHT_TRANSFER_STATES
        assert self._runs_on_cancel_button is not None
        assert self._runs_on_retry_button is not None
        self._runs_on_cancel_button.display = locked
        self._runs_on_retry_button.display = failed

    def _runs_on_failure_reason(self, task: ReminderTask) -> str:
        """The Runs-on row's own `to_server_failed` explanation (fix
        round 1 finding 2): the SAME "Last transfer error: …" copy
        `set_transfer_reasons` already renders for the legacy Retry
        button, reusing the SAME stored `transfer_errors` (threaded in
        by `set_runs_on_transfer_errors`) -- falls back to the plain
        state label when no stored errors exist (e.g. a row that failed
        before this field existed)."""
        errors = self._runs_on_transfer_errors
        if errors:
            return "Last transfer error: " + "; ".join(errors)
        return _TRANSFER_STATE_ROW_LABELS.get(
            task.transfer_state or "", "This transfer failed."
        )

    @property
    def runs_on_row(self) -> DetailValueRow | None:
        """The Runs-on row, for the workbench's `m` keybinding (redesign
        PR-4 task 4) to activate programmatically -- posting `DetailValueRow.
        Activated(row)` on it drives the exact same `on_detail_value_row_
        activated` path (honest lock/failed-transfer refusal, then the
        dropdown) a real Enter/click already does, so `m` needs no new
        activation logic, only this reference."""
        return self._runs_on_row

    def _editable_rows(self) -> tuple[DetailValueRow, ...]:
        """Every row this pane can open an editor on (mounted ones only).

        One list, read by the `Activated` router below AND by
        `_reset_row_editing` -- a row that can open an editor is exactly a
        row whose editor a repaint has to be able to close again.
        """
        return tuple(
            row
            for row in (
                self._repeat_row,
                self._at_row,
                self._timezone_row,
                self._runs_on_row,
            )
            if row is not None
        )

    def _reset_row_editing(self) -> None:
        """Close every open row editor and clear every row error.

        Final review I2/I3: an open editor and an inline error both belong
        to the ROW THEY WERE OPENED ON. Left standing across a repaint
        that swaps in a DIFFERENT reminder, the editor commits its typed
        value onto the new task and the error accuses a value that is
        perfectly valid. `set_task` calls this on a row-identity change
        only -- never on a same-row tick repaint, which this pane does
        continuously and which would otherwise make typing impossible.

        `_editing_task_id` is deliberately NOT cleared here: it is the
        commit-time evidence `_editing_task` needs to discard an edit that
        crossed this repaint in flight.
        """
        for row in self._editable_rows():
            row.end_edit(restore_focus=False)
            row.clear_error()

    def _editing_task(self) -> ReminderTask | None:
        """The reminder an open editor was opened against, or ``None``.

        Final review I2's belt. `set_task` closes the editors on a
        row-identity change, so a commit that still arrives for a row the
        pane no longer shows means the `Changed`/`Submitted` message and
        the repaint crossed in flight. Writing it would land the typed
        value on WHATEVER reminder is painted now -- discard it instead.
        """
        task = self._current_task
        if not isinstance(task, ReminderTask):
            return None
        if self._editing_task_id is not None and self._editing_task_id != task.id:
            logger.debug(
                "Discarding a reminder row edit opened for {} while the "
                "detail pane now shows {}",
                self._editing_task_id,
                task.id,
            )
            return None
        return task

    def on_detail_value_row_activated(self, event: DetailValueRow.Activated) -> None:
        """Open the activated Frequency/Runs-on row's editor, or -- locked
        -- show why editing is refused instead of doing nothing (ruling 2).
        """
        row = event.row
        if row not in self._editable_rows():
            return
        event.stop()
        if self._lifecycle_lock_reason is not None:
            row.show_error(self._lifecycle_lock_reason)
            return
        task = self._current_task
        if not isinstance(task, ReminderTask):
            return
        if row is self._runs_on_row and task.transfer_state == "to_server_failed":
            # fix round 1 finding 2: `_lifecycle_lock_reason` does not
            # cover `to_server_failed` (it is not "locked" for the OTHER
            # rows -- editing before a retry is meant to work there), but
            # the Runs-on row's own dropdown has nothing sensible to
            # offer a failed row either -- show why instead of opening it.
            row.show_error(self._runs_on_failure_reason(task))
            return
        row.clear_error()
        # Final review I2: capture the identity this editor is being
        # opened against, for `_editing_task` to validate at commit time.
        self._editing_task_id = task.id
        if row is self._runs_on_row:
            # A normal, unlocked owner pick (spec §7 flow) -- an
            # in-flight row never reaches here at all: the top-of-
            # function `_lifecycle_lock_reason` check above already
            # intercepted it (that reason IS `transfer_lock_reason`,
            # which covers every `IN_FLIGHT_TRANSFER_STATES` member).
            current_owner = task.owner_id or "local"
            options = list(self._runs_on_options)
            if current_owner not in {value for _, value in options}:
                # Same "the row's real owner always round-trips" fallback
                # `_edit_selected_automation`'s own options-building
                # already uses (survey §7) -- a `Select`'s initial value
                # must be among its options.
                options = [*options, (owner_display_label(current_owner), current_owner)]
            row.begin_edit(
                Select(
                    options,
                    allow_blank=False,
                    value=current_owner,
                    id=_RUNS_ON_EDITOR_ID,
                )
            )
        elif row is self._repeat_row:
            current_preset, _time_text = cron_to_preset(task.cron or "")
            row.begin_edit(
                Select(
                    ReminderForm._preset_options(),
                    allow_blank=False,
                    value=current_preset,
                    id=_REPEAT_EDITOR_ID,
                )
            )
        elif row is self._at_row:
            initial = task.run_at.isoformat() if task.run_at is not None else ""
            row.begin_edit(Input(value=initial, id=_AT_EDITOR_ID))
        elif row is self._timezone_row:
            # task.timezone is guaranteed set here: this row's affordance
            # is only ever True while `task.schedule_kind` is RECURRING,
            # and `ReminderTask`'s own model validator requires a
            # timezone for every recurring row.
            row.begin_edit(
                Select(
                    timezone_options(task.timezone, self._known_timezones),
                    allow_blank=False,
                    value=task.timezone,
                    id=_TIMEZONE_EDITOR_ID,
                )
            )

    def on_select_changed(self, event: Select.Changed) -> None:
        """Route a Frequency/Runs-on Select editor's commit by its stable id."""
        if event.select.id == _REPEAT_EDITOR_ID:
            self._commit_repeat_edit(event)
        elif event.select.id == _TIMEZONE_EDITOR_ID:
            self._commit_timezone_edit(event)
        elif event.select.id == _RUNS_ON_EDITOR_ID:
            self._commit_runs_on_edit(event)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Route the At row's Input editor commit (Enter submits)."""
        if event.input.id == _AT_EDITOR_ID:
            self._commit_at_edit(event)

    def _commit_repeat_edit(self, event: Select.Changed) -> None:
        task = self._editing_task()
        row = self._repeat_row
        if task is None or row is None:
            return
        event.stop()
        new_preset = str(event.value)
        current_preset, current_time_text = cron_to_preset(task.cron or "")
        if new_preset == current_preset:
            # `Select` posts a synthetic `Changed` the moment `begin_edit`
            # mounts it with its CURRENT value preselected (Textual's own
            # `_init_selected_option` assigns `self.value` on `_on_mount`,
            # which its `_watch_value` always turns into a `Changed` post)
            # -- indistinguishable from a real commit except by comparing
            # against the stored value. Also correctly no-ops a genuine
            # reselect of the unchanged value: nothing to persist either
            # way, so the editor stays open for a real pick.
            return
        row.end_edit()
        if new_preset == "custom":
            row.show_error(_REPEAT_CUSTOM_REFUSAL)
            return
        new_cron = preset_to_cron(
            new_preset, current_time_text or DEFAULT_TIME_OF_DAY
        )
        assert new_cron is not None, (
            "every _preset_options() value besides 'custom' always yields a cron"
        )
        self.post_message(ReminderFieldEditRequested(task, {"cron": new_cron}, row))

    def _commit_timezone_edit(self, event: Select.Changed) -> None:
        task = self._editing_task()
        row = self._timezone_row
        if task is None or row is None:
            return
        event.stop()
        new_zone = str(event.value)
        if new_zone == (task.timezone or ""):
            # Same mount-time-synthetic-`Changed` guard as the Repeat
            # editor above.
            return
        row.end_edit()
        self.post_message(
            ReminderFieldEditRequested(task, {"timezone": new_zone}, row)
        )

    def _commit_at_edit(self, event: Input.Submitted) -> None:
        task = self._editing_task()
        row = self._at_row
        if task is None or row is None:
            return
        event.stop()
        row.end_edit()
        raw = event.value.strip()
        # Validated by the bridge (`SchedulingService.edit_reminder_fields`
        # reuses the exact same `parse_forgiving_datetime` this pane's
        # `At` display already implies) -- junk/empty text comes back as
        # a `run_at`-addressed field error, rendered via `row.show_error`
        # by the workbench's own outcome handler.
        self.post_message(ReminderFieldEditRequested(task, {"run_at": raw}, row))

    def _commit_runs_on_edit(self, event: Select.Changed) -> None:
        """Commit the owner-picker Select (PR-3 task 5, spec §7 flow).

        Same-owner selection is a no-op that leaves the dropdown OPEN,
        and it has to be (final review F8/M8 -- this docstring used to
        claim `end_edit()` closed it "right back to the read-only
        display", which is not what the code, its pinning test, or
        Textual allow). Two facts, both re-probed during the final fix
        wave against Textual 8.2.8:

        1. `begin_edit` mounting a `Select` with the row's current owner
           preselected posts a synthetic `Changed` immediately -- the
           reactive `value` var starts at `NULL`, so `_on_mount`'s
           `_init_selected_option` assignment is a real change. Probed:
           this handler fires with the current owner before the user
           touches anything, and closing on it leaves the dropdown shut
           the instant it opens.
        2. A genuine re-pick of the SAME option posts nothing at all:
           `Select._update_selection` assigns only `if value !=
           self.value`. So this branch is reachable ONLY from (1).

        Cancelling without picking is Escape, which `DetailValueRow._on_
        key` already handles. Otherwise the target
        owner decides the transfer direction (a server-shaped value is
        `to_server`, anything else is `to_local`) and the workbench (via
        `ReminderOwnerActionRequested`) runs `transfer_refusal` FIRST --
        refusal renders inline via `row.show_error`.
        """
        task = self._editing_task()
        row = self._runs_on_row
        if task is None or row is None:
            return
        event.stop()
        current_owner = task.owner_id or "local"
        new_owner = str(event.value)
        if new_owner == current_owner:
            return
        row.end_edit()
        direction = "to_server" if new_owner.startswith("server:") else "to_local"
        self.post_message(ReminderOwnerActionRequested(task, direction, row))

    def set_task(
        self,
        task: ReminderTask | ScheduledTask | None,
        *,
        queue_empty: bool = False,
        run_history=None,
        incidents=None,
        known_timezones: Sequence[str] = (),
        runs_on_options: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Update the detail view for the given task (or clear it).

        Args:
            task: The row to paint, or ``None`` for the empty state. A
                `ReminderTask` fills every group; a bare `ScheduledTask`
                (the legacy shape) paints only what it carries.
            queue_empty: True when the queue has no rows at all, which
                picks the "schedule your first task" empty copy over the
                "select a task" one. Read only when ``task`` is ``None``.
            run_history: Pre-fetched run rows for the History group,
                already read off the event loop by the caller (this
                method performs no I/O of its own). ``None``/empty
                renders `format_run_history`'s own no-history copy.
            incidents: Pre-fetched incident rows -- rendered by
                `format_incidents` and retained for the acknowledge
                action's own open-incident lookup. Same caller-fetched
                contract as ``run_history``.
            known_timezones: PR-3 task 3 -- zones already used by other
                tasks, passed through to the Timezone row editor's option
                source (`timezone_options`) so it offers the same choices
                the create/edit modal does. Every pre-task-3 caller omits
                this and is unaffected.
            runs_on_options: PR-3 task 5 -- the workbench's own
                `_runs_on_options()[0]`, passed through to the Runs-on
                row's owner-picker `Select` (same threaded-in-by-the-
                workbench shape `known_timezones` already uses). Every
                pre-task-5 caller/test omits this and is unaffected.
        """
        # Final review I2/I3: a repaint that swaps in a DIFFERENT reminder
        # (a filter keystroke, a chip switch, a row aging out of the
        # filter -- `_update_detail_for_index` falls back to index 0)
        # takes the open editor and the inline error with it. A same-row
        # tick repaint, which is what this pane mostly does, must not.
        if getattr(task, "id", None) != getattr(self._current_task, "id", None):
            self._reset_row_editing()
        self._current_task = task
        self._current_incidents = list(incidents or [])
        self._known_timezones = known_timezones
        self._runs_on_options = runs_on_options
        metadata = self.query_one("#scheduling-task-detail-metadata", Vertical)
        lifecycle = self.query_one("#scheduling-task-detail-lifecycle", Horizontal)
        self.query_one("#schedules-follow-in-console", Button)
        empty_state = self.query_one("#scheduling-task-detail-empty-state", Static)

        if task is None:
            empty_copy = (
                "No scheduled tasks yet. Press n to schedule your first task."
                if queue_empty
                else "Select a task from the queue, or press n to schedule one."
            )
            empty_state.update(empty_copy)
            empty_state.display = True
            metadata.display = False
            lifecycle.display = False
            self.query_one("#scheduling-transfer-why", Static).update("")
            missed_notice = self.query_one("#scheduling-task-detail-missed", Static)
            missed_notice.update("")
            missed_notice.display = False
            self.query_one("#schedules-follow-in-console", Button).label = (
                "Follow in Console"
            )
            # PR-3 task 3: stale lock state from a PREVIOUS selection must
            # not survive into a cleared pane (harmless today since the
            # groups are hidden either way, but `on_detail_value_row_
            # activated` reads this directly -- keep it honest).
            self._lifecycle_lock_reason = None
            # PR-3 task 5: the Runs-on row's Cancel/Retry buttons from a
            # PREVIOUS selection must not stay visible in a cleared pane.
            if self._runs_on_cancel_button is not None:
                self._runs_on_cancel_button.display = False
            if self._runs_on_retry_button is not None:
                self._runs_on_retry_button.display = False
            self._runs_on_transfer_errors = []
            return

        empty_state.display = False
        metadata.display = True
        lifecycle.display = isinstance(task, ReminderTask)

        # schedules-redesign PR-1, task 3: the Details/Frequency/History
        # groups are reminder-only (spec §5's reminder column); a
        # `ScheduledTask` projection (watchlist_job/briefing_job) keeps the
        # legacy Type/Schedule rows instead, since this regrammar does not
        # touch projection rendering.
        is_reminder = isinstance(task, ReminderTask)
        self.query_one("#scheduling-task-detail-legacy-fields", Vertical).display = (
            not is_reminder
        )
        self.query_one("#scheduling-task-detail-groups", Vertical).display = (
            is_reminder
        )
        if is_reminder:
            assert self._runs_on_row is not None, "set_task called before mount"
            body = (task.body or "").strip()
            self._body_card.update(body)
            self._body_card.display = bool(body)
            self._runs_on_row.update_value(_reminder_runs_on_label(task))
            self._repeat_row.update_value(_reminder_repeat_label(task))
            self._at_row.update_value(_reminder_at_label(task))
            self._timezone_row.update_value(_reminder_timezone_label(task))
            self._last_fire_row.update_value(_reminder_last_fire_label(task))
            self._configure_frequency_editability(task)
            self._configure_runs_on_row(task)

        # redesign PR-4, task 4: the legacy Move/Retry/Cancel button
        # display-toggling that used to live here is retired along with
        # the buttons themselves (ruling 2) -- the caller's `_update_
        # transfer_actions` -> `set_lifecycle_lock` (which runs right
        # after every `set_task`, for every task type) already owns
        # `#scheduling-transfer-why`'s content, so no explicit reset is
        # needed here either.

        # redesign PR-4 task 5 (ruling 5): the task-23106 ownership line
        # ("#scheduling-task-detail-managed") and its `else` branch are
        # DELETED -- provably unreachable, not merely unused. `TaskDetail(`
        # is constructed in exactly TWO places in the repo, both in
        # `schedules_workbench.py` (the docked detail pane, and task 6's
        # fresh per-push overlay instance -- final review F5 corrected the
        # original "exactly once" wording, which task 6 had made false).
        # Both are fed through the SAME single seam, `_update_detail_for_
        # index` (`_detail_panes` is one list, not a second data path),
        # whose data comes from `load_tasks` -> `list_tasks(owner_id=None,
        # include_projections=False)` filtered to `ReminderTask`, and which
        # asserts `isinstance(task, ReminderTask)` before every call. So
        # `task` here is
        # never anything but a `ReminderTask`, the `else` never ran, and
        # the empty Static it painted into was pure weight. The copy
        # generator itself (`_managed_elsewhere_notice`) STAYS: the
        # workbench's own edit/mark/enable action guards still call it,
        # and `test_managed_elsewhere_notice_names_the_owning_screen`
        # covers it directly.

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
            # Underlying status, not display status (review F5): Run now
            # explicitly works on disabled tasks, so a disabled task whose
            # last dispatch failed keeps its retry affordance.
            run_now_button = self.query_one("#scheduling-run-now", Button)
            if _underlying_status(task) in {TaskStatus.MISSED, TaskStatus.TIMED_OUT}:
                run_now_button.label = "Run now (retry)"
                run_now_button.tooltip = (
                    "Retry this scheduled task now: its last dispatch ran "
                    "and failed. Dispatches immediately through the same "
                    "path the scheduler uses."
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
        self._update_static(
            "scheduling-task-detail-run-history", format_run_history(run_history)
        )
        self._update_static(
            "scheduling-task-detail-incidents", format_incidents(incidents)
        )
        self._sync_acknowledge_button()

        status = _task_status(task)
        badge = self.query_one("#scheduling-task-status-badge", Static)
        badge.update(_humanize_status(status))
        badge.remove_class(*_STATUS_BADGE_CLASSES.values())
        badge.add_class(status_badge_class(status))

    def _update_missed_notice(
        self, task: ReminderTask | ScheduledTask | None
    ) -> None:
        """Render the late-dispatch notice for the last dispatch.

        Distinct from failed: failed means the dispatch ran and the handler
        raised; this means the dispatch happened well after its scheduled
        time. The notice describes the last dispatch and self-heals: the next
        on-time dispatch clears it. Plain text, no markup -- titles are
        untrusted and never interpolated into markup.

        task-19562: this used to say "Missed while away ... (the scheduler
        was not running at the scheduled time)", which the app cannot know
        from the row. `SchedulerLoop.tick` awaits every due handler serially
        and inline, so one slow handler (a watchlist check may run to its
        300 s execution timeout, against a 60 s missed-fire grace) pushes
        every task behind it past the grace and produces exactly this row
        while the scheduler is running the whole time. `missed_at` and
        `missed_count` remain true either way -- the occurrence WAS owed
        late, and earlier ones really were skipped -- so the facts stay and
        only the invented cause goes. Which cause it actually was is
        recorded where the loop can still tell (see
        `SchedulerLoop._report_lateness_cause`).
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
                f"Ran late: dispatched well after the {scheduled} occurrence; "
                f"more than {ScheduledTasksDB._MISSED_COUNT_CAP:,} earlier "
                "occurrence(s) were skipped, not replayed."
            )
        elif missed_count > 0:
            copy = (
                f"Ran late: dispatched well after the {scheduled} occurrence; "
                f"{missed_count} earlier occurrence(s) were skipped, not replayed."
            )
        else:
            copy = (
                f"Ran late: the {scheduled} occurrence dispatched well after "
                "its scheduled time (for example the app was closed or "
                "asleep, or the scheduler was busy with an earlier task)."
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

    def set_runs_on_transfer_errors(self, errors: list[str]) -> None:
        """Cache a `to_server_failed` row's stored `transfer_errors` (PR-3
        task 5 fix round 1, finding 2) for `_runs_on_failure_reason` --
        fed from the same `retry_errors` the workbench computes (redesign
        PR-4 task 4: the legacy Retry button that USED to also read this
        list, via the now-retired `set_transfer_reasons`, is gone; this
        stays the Runs-on row's own single source)."""
        self._runs_on_transfer_errors = list(errors)

    def set_lifecycle_lock(self, reason: str | None) -> None:
        """Freeze Edit/Enable/Disable/Delete while a transfer is in flight.

        Spec §6.3's "dormant and in-flight rows are read-only except
        cancel" (final review I7): the transfer snapshotted this row's
        payload at begin time, so an edit made now ships the PRE-edit
        content to the server and is then overwritten locally by the
        first mirror pull -- the user's edit vanishes with no warning.
        ``reason`` comes from `SchedulingService.transfer_lock_reason`
        (never re-derived here) and is both each button's tooltip and a
        line in the always-visible Static, since keyboard users cannot
        see tooltips (UX-073). ``None`` restores the row's normal
        enabled/disabled logic, which `set_task` has already applied.

        PR-3 task 3: also cached for the Frequency rows' `Activated`
        handler (`self._lifecycle_lock_reason`) -- the SAME reason, not a
        second `transfer_lock_reason` call, per survey §8's one-source-
        of-truth rule.
        """
        self._lifecycle_lock_reason = reason
        locked = reason is not None
        for button_id, tooltip in (
            ("scheduling-edit-task", "Edit this scheduled task."),
            ("scheduling-delete-task", "Delete this scheduled task."),
        ):
            button = self.query_one(f"#{button_id}", Button)
            button.disabled = locked
            button.tooltip = reason or tooltip
        enable_btn = self.query_one("#scheduling-enable-task", Button)
        disable_btn = self.query_one("#scheduling-disable-task", Button)
        if locked:
            enable_btn.disabled = True
            disable_btn.disabled = True
            enable_btn.tooltip = reason
            disable_btn.tooltip = reason
        else:
            enable_btn.tooltip = "Enable this scheduled task."
            disable_btn.tooltip = "Disable this scheduled task."

        why = self.query_one("#scheduling-transfer-why", Static)
        line = f"Edit/Enable/Disable/Delete: {reason}" if reason else ""
        existing = [
            text
            for text in str(why.renderable).split("\n")
            if text and not text.startswith("Edit/Enable/Disable/Delete:")
        ]
        if line:
            existing.append(line)
        why.update("\n".join(existing))

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
        """Update the conflict card for the current task state.

        Underlying status (review F5): a disabled task's conflict is
        still a conflict -- the conflicts view lists it, so this card
        must not claim "No conflict".
        """
        card = self.query_one("#scheduling-conflict-card", Vertical)
        text = self.query_one("#scheduling-conflict-text", Static)
        if task is not None and _underlying_status(task) == TaskStatus.CONFLICT:
            text.update(f"Conflict detected\n{task.title}")
            card.add_class("conflict")
        else:
            text.update("No conflict")
            card.remove_class("conflict")

    def _update_static(self, widget_id: str, content: str) -> None:
        """Update a child Static widget by id."""
        static = self.query_one(f"#{widget_id}", Static)
        static.update(content)
