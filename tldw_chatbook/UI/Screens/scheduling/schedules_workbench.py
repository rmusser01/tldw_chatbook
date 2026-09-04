"""Schedules workbench shell for run timing, triggers, and recovery."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import Button, DataTable, Input, Static, TabbedContent, TabPane

from ...Navigation.base_app_screen import BaseAppScreen
from ...Navigation.screen_state_store import RuntimeIdentity
from ...Workbench.workbench_state import (
    RecoveryState,
    WorkbenchHeaderState,
    WorkbenchStatus,
)
from ...Workbench.workbench_widgets import DestinationHeader, RecoveryCallout
from ....runtime_policy.bootstrap import set_authoritative_runtime_source
from ....Scheduling.automation_health import compute_local_health
from ....Scheduling.events import (
    CancelTransferRequested,
    DefinitionFieldEditRequested,
    DefinitionLifecycleToggleRequested,
    DefinitionOwnerActionRequested,
    DeleteTaskRequested,
    DisableTaskRequested,
    AcknowledgeIncidentRequested,
    EditTaskRequested,
    EnableTaskRequested,
    ReminderFieldEditRequested,
    ReminderOwnerActionRequested,
    RetryTransferRequested,
    RunReminderNowRequested,
    SyncCompleted,
    SyncFailed,
    TransferToLocalRequested,
    TransferToServerRequested,
)
from ....Scheduling.models import ReminderTask, ScheduledTask
from ....Scheduling.services.server_client import (
    ServerClientError,
    ServerClientValidationError,
)
from ....Scheduling.services.sync_engine import (
    _RESULTS_PAGE_SIZE,
    _SYNC_MAX_PAGES,
)
from ....UI.Screens.scheduling.conflicts_tab import ConflictsTab
from ....UI.Screens.scheduling.results_tab import (
    ResultsTab,
    definition_for_result,
    # NOT `rich.markup.escape`: this dialog copy is rendered by a
    # Textual `Label` -> `Content.from_markup`, whose tokenizer eats ANY
    # `[...]`, while rich's escape only covers `[a-z#/@]...` tags (task 6
    # round 1). Same parser, same escape as the results detail pane.
    escape_markup,
    index_definitions_by_id,
    solved_eligibility,
)
from ....UI.Screens.scheduling.sync_status_widget import SyncStatusWidget
from ....Widgets.confirmation_dialog import ConfirmationDialog
from ....Widgets.detail_value_row import DetailValueRow
# schedules-redesign PR-1, Task 4: `automation_execution_target_label`/
# `automation_name_cell` moved to `definition_detail.py` (that leaf
# module's own "Model"/"Runs on" row formatters need them and importing
# them back from here would be circular) -- re-exported via this import
# so the DataTable render call site below and the pre-existing
# `test_execution_target_label_matrix` test (which imports
# `automation_execution_target_label` from THIS module) keep working.
from .definition_detail import (
    DefinitionDetail,
    _definition_transfer_suffix,
    _LIFECYCLE_TOGGLE_RESULTS,
    automation_execution_target_label,
    automation_name_cell,
)
from .forms.automation_definition_form import AutomationDefinitionForm
from .forms.new_task_choice_modal import NewTaskChoiceModal
from .forms.reminder_form import ReminderForm
from .task_detail import (
    SCHEDULES_EMPTY_CONSOLE_RECOVERY,
    TaskDetail,
    TaskInspector,
    _format_next_run,
    _format_relative,
    _managed_elsewhere_notice,
    _queue_owner_suffix,
    _transfer_row_suffix,
    _was_missed_while_away,
    transfer_row_dict,
)
# schedules-redesign PR-2, Task 2: the pure row adapter -- see that
# module's docstring for why it is a standalone, Textual-free file.
from .unified_rows import (
    Chip,
    RowKind,
    UnifiedRow,
    build_unified_rows,
    filter_rows,
    sort_rows,
)

if TYPE_CHECKING:
    from tldw_chatbook.Scheduling.services.scheduling_service import (
        SchedulingService,
        TransferOutcome,
    )
    from tldw_chatbook.app import TldwCli


logger = logger.bind(module="SchedulesWorkbench")

SCHEDULES_COMPACT_WORKBENCH_MAX_WIDTH = 120

#: Debounce for the queue filter `Input` -- mirrors the console picker
#: family's 0.2 s shape (`console_prompt_picker_modal.py`). A full render
#: pass clears and rebuilds the whole `DataTable` (task-15476).
QUEUE_FILTER_DEBOUNCE_SECONDS = 0.2

#: Cadence for re-rendering the relative next-run column ("in 25m" goes
#: stale otherwise -- task-23111 review F9). Paused while the screen is
#: not current, per the hidden-progress-clock rule (TASK-23022).
NEXT_RUN_REFRESH_SECONDS = 60.0

#: Defensive cap for the Automations tab's follow-the-pages load -- the loop
#: exists so the tail of a large definition list is never silently hidden,
#: not to render unbounded rows.
AUTOMATIONS_LOAD_MAX_ROWS = 500

#: Debounce before acting on a notification-triggered results pull
#: (schedules-handoff PR-6 task 4) -- a burst of `automation_run_*` events
#: collapses into ONE pull (plan ruling 3), same stop-and-restart timer
#: shape as the queue filter's own debounce above.
RESULTS_PULL_DEBOUNCE_SECONDS = 0.3

#: How many results the inbox lists. The DB default (50) silently hid older
#: rows while the badge counted EVERY unread one, so the tab could read
#: "Results (120)" over 50 rows. This is the sync-mirrored window --
#: exactly the newest-pages walk `SyncEngine._pull_results` performs -- so
#: the inbox shows everything a pull could have brought down and nothing it
#: could not. Beyond the cap the tab says so out loud (`ResultsTab.
#: populate`'s `total`); deliberately no pagination machinery.
RESULTS_INBOX_LIMIT = _RESULTS_PAGE_SIZE * _SYNC_MAX_PAGES

#: How many transport reconnects `EventObserver.run()` absorbs internally
#: (its own built-in exponential backoff, capped at 5s per attempt) before
#: giving up and raising back to `_run_server_notification_observer`. That
#: outer loop is the real "never give up for the life of this screen"
#: layer -- see its docstring for the full failure-mode read.
_NOTIFICATION_OBSERVER_MAX_RECONNECTS = 5

#: Flat delay before `_run_server_notification_observer` restarts a fresh
#: `observe()` call after one gave up (matches the inner backoff's own
#: 5s cap -- no need to reimplement exponential backoff a second time for
#: a rare outer-level restart). Interrupted immediately by unmount via
#: `cancel_event`, never a blind sleep.
_NOTIFICATION_OBSERVER_RESTART_DELAY_SECONDS = 5.0


def _cancel_toast_text(name: str) -> str:
    """Honest cancel confirmation (final review L13).

    The old copy ("Transfer cancelled for 'X'.") and the docs beside it
    promised "no server-side effect", which spec §6.3's own table does
    not support: `from_server_pending` also covers a release whose
    server-side delete already landed but whose ack was lost, and that
    delete cannot be undone from here. What IS true in every cancellable
    state is that the queued mutation is gone, so nothing further goes
    out -- which is what this now says.
    """
    return (
        f"Transfer cancelled for '{name}' — nothing further will be sent "
        "to the server."
    )


def _pending_transfer_errors(
    service: "SchedulingService", table_kind: str, row_id: str
) -> list[str]:
    """The stored `transfer_errors` from `row_id`'s queued transfer
    mutation, or `[]` when none exist -- shared by `_update_transfer_
    actions`'s existing Retry-button computation (reminders) and both
    panes' Runs-on row (fix round 1, finding 2: same source `set_
    transfer_reasons`'s own "Last transfer error: …" line already reads,
    fed here rather than re-derived, one lookup per row)."""
    mutation = service.db.get_pending_mutation_for_local_id(row_id, table_kind)
    if mutation is None:
        return []
    errors = (mutation.get("payload") or {}).get("transfer_errors")
    return list(errors) if errors else []


def _definition_transfer_errors(
    service: "SchedulingService", definition: dict[str, Any]
) -> list[str]:
    """`_pending_transfer_errors` for a definition dict, only when it is
    actually `to_server_failed` -- a `to_server_failed` row is always
    local-owned (a failed to_server transfer never became server-owned),
    so `definition["id"]` is already the local id, no `_resolve_local_
    definition_id` round trip needed (fix round 1, finding 2)."""
    if definition.get("transfer_state") != "to_server_failed":
        return []
    return _pending_transfer_errors(
        service, "automation_definition", str(definition.get("id"))
    )


#: Delayed second fetch of the run-history pane after a Run-now dispatch:
#: the terminal audit event lands only after the server finishes executing
#: the run, so an immediate fetch alone would usually miss it.
AUTOMATION_HISTORY_FOLLOWUP_SECONDS = 5.0


def _row_title_cell(
    row: UnifiedRow, *, marked_ids: set[str], compact_owner_suffix: bool
) -> Text:
    """Queue-row title cell for one `UnifiedRow` (redesign PR-2, spec S4).

    Each primitive keeps its OWN existing title-suffix/prefix rendering
    verbatim -- a reminder row is byte-identical to the pre-redesign
    Title column (`_transfer_row_suffix`/`_queue_owner_suffix`), a
    definition row reuses the Automations tab's own Name-cell rendering
    (`automation_name_cell`/`_definition_transfer_suffix`) -- rather than
    inventing one shared format neither primitive used before.
    """
    if row.kind == "reminder":
        task = row.source_row
        assert isinstance(task, ReminderTask)
        text = Text(
            ("● " if task.id in marked_ids else "")
            + ("◇ " if _was_missed_while_away(task) else "")
            + task.title
            + _transfer_row_suffix(task)
            + _queue_owner_suffix(task, compact=compact_owner_suffix)
        )
    else:
        definition = row.source_row
        assert isinstance(definition, dict)
        text = Text(
            automation_name_cell(definition) + _definition_transfer_suffix(definition)
        )
    if row.unread_count > 0:
        # Same "bold leading dot" idiom `results_tab._review_state_cell`
        # uses for an unread result -- reused rather than inventing a
        # second unread affordance.
        text.append(" ●", style="bold")
    return text


def _row_subtitle(row: UnifiedRow, now: datetime) -> str:
    """Queue-row subtitle: schedule summary + relative next-run (spec S4).

    A reminder row reuses `_format_next_run` verbatim (the exact text the
    pre-redesign Next-Run column showed). A definition row has no
    existing per-row relative-time formatter to reuse (`_format_next_run`
    is typed for `ReminderTask | ScheduledTask`), so this derives the
    same "absolute (relative)" shape from `UnifiedRow`'s own
    already-normalized `next_run_at` + `bucket`.
    """
    if row.kind == "reminder":
        next_text = _format_next_run(row.source_row, now=now, compact=True)
    elif row.bucket == "paused":
        next_text = "— (paused)"
    elif row.next_run_at is None:
        next_text = "-"
    else:
        absolute = row.next_run_at.strftime("%Y-%m-%d %H:%M")
        next_text = f"{absolute} ({_format_relative(row.next_run_at, now)})"
    return f"{row.schedule_summary} · {next_text}"


class SchedulesWorkbench(BaseAppScreen):
    """Main workbench for managing scheduled runs, reminders, and jobs."""

    BINDINGS = [
        Binding("c", "create_reminder", "Create"),
        Binding("e", "edit_task", "Edit"),
        Binding("r", "run_task_now", "Run now"),
        Binding("space", "toggle_enabled", "Enable/Disable"),
        Binding("d", "delete", "Delete"),
        Binding("x", "mark_task", "Mark"),
        Binding("escape", "clear_marks", "Clear marks"),
        Binding("s", "sync_now", "Sync"),
        # Automations-tab-only (schedules-handoff PR-5 task 7 fix round):
        # the tab has no per-row detail widget, so its actions are
        # keybindings routed by active tab -- same idiom as r/e above,
        # not new buttons (mirrors _edit_selected_automation/
        # _run_automation_now, the tab's existing action grammar).
        Binding("m", "move_automation_to_local", "Move to local"),
        Binding("M", "move_automation_to_server", "Move to server"),
        Binding("y", "retry_automation_transfer", "Retry transfer"),
        Binding("k", "cancel_automation_transfer", "Cancel transfer"),
        # Results-tab-only (schedules-handoff PR-6 task 3): read/dismiss
        # reuse r/d via the SAME active-tab routing action_run_task_now/
        # action_delete already do for Automations -- r=Read and d=Dismiss
        # are natural readings of those same keys on this tab. Mark
        # solved/Mark all read have no existing-key mnemonic to reuse, so
        # they get fresh letters (o/a), guarded the same way m/M/y/k are.
        Binding("o", "mark_result_solved", "Mark solved"),
        Binding("a", "mark_all_results_read", "Mark all read"),
    ]

    # Footer hints must stay 1:1 with BINDINGS and only advertise implemented
    # actions (ADR-031). Single letters are safe: focused inputs consume
    # printable keys before screen bindings fire.
    SCHEDULES_SHORTCUTS = (
        ("c", "create"),
        ("e", "edit"),
        ("r", "run now"),
        ("space", "toggle"),
        ("d", "delete"),
        ("x", "mark"),
        ("s", "sync"),
        ("m", "move to local"),
        ("M", "move to server"),
        ("y", "retry transfer"),
        ("k", "cancel transfer"),
        ("o", "mark solved"),
        ("a", "mark all read"),
    )

    def __init__(
        self, app_instance: "TldwCli", screen_name: str = "schedules", **kwargs
    ):
        super().__init__(app_instance, screen_name, **kwargs)
        self._scheduling_service = getattr(app_instance, "scheduling_service", None)
        # redesign PR-2, Task 2: `self._tasks`/`self._visible_tasks` stay
        # reminder-only (never watchlist/briefing `ScheduledTask`
        # projections -- spec S2 locked decision 2, Task 1's report) so
        # EVERY existing reminder action (`_selected_task`, mark/toggle/
        # edit/delete, the timezone/marks-legend helpers below) keeps
        # reading them unchanged. `self._all_rows`/`self._visible_rows`
        # are the NEW unified source of truth the DataTable itself
        # renders from (reminders + automation definitions, spans owners).
        self._tasks: list[ReminderTask] = []
        self._visible_tasks: list[ReminderTask] = []
        self._all_rows: list[UnifiedRow] = []
        self._visible_rows: list[UnifiedRow] = []
        # Final review F2: the UNFILTERED local definitions table from the
        # last full definitions fetch, for unread-count resolution only
        # (never a source of rows). The display merge above excludes every
        # row that has a `server_id`, so a transferred definition's
        # pre-transfer results have no key to resolve through without it.
        self._queue_local_definitions: list[dict[str, Any]] = []
        # redesign PR-2, Task 2 review round 2: any Automations-tab
        # mutation of a definition (create/edit save, run-now, transfer
        # begin/cancel -- everything that funnels through
        # `_request_automations_refresh`) sets this; a reminder-only
        # refresh (`refresh_definitions=False`) upgrades to a full one
        # and clears it, and switching TO the Queue tab while stale does
        # the same. Without this, the Queue's cached definition rows
        # could go stale for the whole session -- reminder-only actions
        # deliberately stopped self-healing it (finding 3's own fix).
        self._definitions_stale = False
        self._chip: Chip = "all"
        self._filter_text = ""
        self._filter_debounce_timer: Timer | None = None
        self._next_run_refresh_timer: Timer | None = None
        # task-15476: the task id currently shown in the detail/inspector
        # panes, tracked independently of row index so a filter keystroke
        # can restore the same selection instead of always jumping to row 0.
        self._selected_task_id: str | None = None
        # redesign PR-2, Task 2: the highlighted UnifiedRow's own id
        # (`"reminder:<id>"`/`"definition:<id>"`), the general cursor-
        # restoration key `_render_table` uses across BOTH kinds --
        # `_selected_task_id` above stays reminder-only for the existing
        # action code that reads it directly.
        self._selected_row_id: str | None = None
        self._marked_ids: set[str] = set()
        #: The current hidden-panes notice from on_resize; combined with
        #: the marks/glyph legend in _update_pane_notice (task-23107).
        self._resize_notice = ""
        self._sync_running = False
        # ADR-077: server-owned automation definitions shown in the
        # Automations tab. Kept as the raw dicts the server client returns
        # (model_dump(mode="json")) -- the server owns the enum vocabularies
        # and the tab must not break when a new lifecycle/health value ships.
        self._automations: list[dict[str, Any]] = []
        self._selected_automation_id: str | None = None
        self._current_console_follow_item = None
        self._latest_console_follow_item_id: str | None = None
        self._latest_console_launch_kwargs: dict[str, Any] | None = None
        self._latest_console_context_loaded = False
        # schedules-handoff PR-6 task 4: notification-triggered results
        # pull (single-flight -- `_results_pull_running` guards concurrent
        # pulls, `_results_pull_rerun_requested` queues at most one
        # follow-up) and the SSE observer's stop signal, workbench-scoped
        # per plan ruling 3 (started in on_mount, cancelled in on_unmount).
        self._results_pull_debounce_timer: Timer | None = None
        self._results_pull_running = False
        self._results_pull_rerun_requested = False
        self._notification_cancel_event: asyncio.Event | None = None

    def _active_server_id(self) -> str | None:
        runtime_policy = getattr(self.app_instance, "runtime_policy", None)
        runtime_state = runtime_policy.state if runtime_policy is not None else None
        return getattr(runtime_state, "active_server_id", None)

    @staticmethod
    def _server_available(service: Any, active_server_id: str | None) -> bool:
        """Return whether Schedules can switch ownership to a live server."""
        return (
            service is not None
            and bool(active_server_id)
            and service.server_client.notifications_service is not None
        )

    @staticmethod
    def _update_static_content(target: Static, content: str) -> None:
        """Preserve layout-aware updates while skipping identical timer copy."""
        if target.content != content:
            target.update(content)

    def compose_content(self) -> ComposeResult:
        """Build the three-pane scheduling workbench layout."""
        service = self._service()
        owner_id = service.owner_id if service else "local"
        active_server_id = self._active_server_id()
        server_available = self._server_available(service, active_server_id)
        yield DestinationHeader(
            WorkbenchHeaderState(
                title="Schedules",
                subtitle="When jobs, watchlists, and workflows run.",
                status="loading",
                status_label="Checking sync status…",
            ),
            id="schedules-destination-header",
        )
        if service is None:
            # Visible recovery instead of a silently empty workbench (UX-043).
            yield RecoveryCallout(
                RecoveryState(
                    title="Scheduling service unavailable",
                    body=(
                        "The scheduling service did not start, so the queue and "
                        "sync are offline. Check the scheduling configuration, "
                        "then restart the app."
                    ),
                    action=None,
                    visible=True,
                ),
                id="scheduling-recovery",
            )
        with Vertical(id="schedules-shell"):
            # TASK-26025: scheduler liveness -- a stale heartbeat reads
            # distinctly from an empty queue and a never-started loop.
            yield Static("", id="scheduling-liveness")
            with TabbedContent(id="scheduling-tabs"):
                with TabPane("Queue", id="scheduling-queue-tab"):
                    with Horizontal(id="scheduling-workbench"):
                        with Vertical(id="scheduling-list-pane"):
                            # redesign PR-2, Task 3: the rail header --
                            # `Create ▾` is the pre-existing 2-choice
                            # chooser button (`scheduling-new-task`,
                            # unchanged id/handler, relabeled/repositioned
                            # here); `Mark all read` is new, and only shown
                            # once `_update_mark_all_read_visibility` finds
                            # unread rows (default hidden below, mirrors
                            # `SyncStatusWidget`'s own Clear-button idiom).
                            with Horizontal(id="scheduling-list-header"):
                                yield Static(
                                    "Schedule Queue",
                                    id="scheduling-list-title",
                                    classes="scheduling-column-title",
                                )
                                yield Button(
                                    "Create ▾",
                                    id="scheduling-new-task",
                                    variant="primary",
                                    tooltip="Schedule a new task (c).",
                                )
                                yield Button(
                                    "Mark all read",
                                    id="scheduling-mark-all-read",
                                    tooltip="Mark every unread automation result read (a).",
                                )
                            # redesign PR-2, Task 2: the chip row (spec S3)
                            # -- one of the four buttons always carries
                            # variant="primary" for "current", the same
                            # idiom `SyncStatusWidget`'s owner toggle uses.
                            with Horizontal(id="scheduling-queue-chips"):
                                yield Button(
                                    "All",
                                    id="scheduling-chip-all",
                                    variant="primary",
                                    classes="scheduling-queue-chip",
                                )
                                yield Button(
                                    "Active",
                                    id="scheduling-chip-active",
                                    classes="scheduling-queue-chip",
                                )
                                yield Button(
                                    "Paused",
                                    id="scheduling-chip-paused",
                                    classes="scheduling-queue-chip",
                                )
                                yield Button(
                                    "Completed",
                                    id="scheduling-chip-completed",
                                    classes="scheduling-queue-chip",
                                )
                            yield Input(
                                # Says what ruling 5's search actually
                                # matches -- title + question/body (final
                                # review F6: the Type/Status columns are
                                # gone and status words like "missed" no
                                # longer match, which the user guide
                                # already documents).
                                placeholder="Filter: title or question…",
                                id="scheduling-queue-filter",
                            )
                            yield DataTable(
                                id="scheduling-task-table", cursor_type="row"
                            )
                            yield Static("", id="scheduling-pane-notice")
                        with Vertical(id="scheduling-detail-pane"):
                            yield TaskDetail(id="scheduling-task-detail")
                            # redesign PR-2, Task 2: a sibling of TaskDetail
                            # in the SAME pane -- highlight routes between
                            # the two via the `pane-hidden` class (survey
                            # section 3's recipe), never both visible.
                            # Definition rows are viewable + detail-only
                            # here (no actions until PR-4).
                            yield DefinitionDetail(
                                id="scheduling-queue-definition-detail",
                                classes="pane-hidden",
                            )
                        with Vertical(id="scheduling-inspector-pane"):
                            yield TaskInspector(id="scheduling-task-inspector")
                with TabPane("Automations", id="scheduling-automations-tab"):
                    with Horizontal(id="scheduling-automations-split"):
                        with Vertical(id="scheduling-automations-pane"):
                            with Horizontal(id="scheduling-automations-header"):
                                yield Static(
                                    "Server Automations",
                                    id="scheduling-automations-title",
                                    classes="scheduling-column-title",
                                )
                                yield Button(
                                    "+ New",
                                    id="scheduling-new-automation",
                                    variant="primary",
                                    tooltip="Schedule a new recurring question.",
                                )
                            yield DataTable(
                                id="scheduling-automations-table", cursor_type="row"
                            )
                            yield Static("", id="scheduling-automations-notice")
                        # schedules-redesign PR-1, Task 4: the definitions
                        # detail pane -- the first per-row detail widget the
                        # Automations tab has had (see redesign-pr1-survey.md
                        # section 1's "no per-row detail widget" finding).
                        # Third pane alongside list|history, matching the
                        # Queue tab's list|detail|inspector idiom.
                        with Vertical(id="scheduling-automations-detail-pane"):
                            yield DefinitionDetail(id="scheduling-automation-detail")
                        with Vertical(id="scheduling-automation-history-pane"):
                            yield Static(
                                "Run history",
                                id="scheduling-automation-history-title",
                                classes="scheduling-column-title",
                            )
                            yield DataTable(
                                id="scheduling-automation-history-table",
                                cursor_type="row",
                            )
                            yield Static(
                                "",
                                id="scheduling-automation-history-notice",
                            )
                with TabPane("Conflicts", id="scheduling-conflicts-tab"):
                    yield ConflictsTab(
                        id="scheduling-conflicts",
                        sync_engine=service.sync_engine if service else None,
                    )
                with TabPane("Results", id="scheduling-results-tab"):
                    yield ResultsTab(id="scheduling-results")
            # redesign PR-2, Task 3: the bottom status strip (plan ruling
            # 4) -- shared across every tab (not Queue-specific, so it
            # sits below the TabbedContent rather than inside one
            # TabPane), hosting the existing `SyncStatusWidget` (owner
            # indicator + sync health, now with a width-compact styling
            # path -- see `_sync_responsive_workbench`) and a conflicts
            # badge chip that switches to the Conflicts tab.
            with Horizontal(id="scheduling-status-strip"):
                yield SyncStatusWidget(
                    id="scheduling-sync-status",
                    current_owner=owner_id,
                    active_server_id=active_server_id,
                    server_available=server_available,
                )
                yield Button(
                    "Conflicts",
                    id="scheduling-conflicts-badge",
                    classes="scheduling-queue-chip",
                    tooltip=(
                        "Sync conflicts for the current owner's scheduled "
                        "tasks only. Click to open the Conflicts tab."
                    ),
                )

    def _service(self) -> "SchedulingService | None":
        """Return the app's scheduling service, if available."""
        return self._scheduling_service

    def _register_footer_shortcuts(self) -> None:
        """Register Scheduling shortcuts via BaseAppScreen's persisting API."""
        self.register_footer_shortcuts(
            source="schedules", shortcuts=self.SCHEDULES_SHORTCUTS
        )

    def on_mount(self) -> None:
        # No super().on_mount(): the dispatcher already invokes
        # BaseAppScreen.on_mount separately for this Mount event.
        self._sync_responsive_workbench()
        self._register_footer_shortcuts()
        self._refresh_owner_select()
        self._refresh_conflicts_tab()
        self._refresh_results_tab()
        # redesign PR-2, Task 3: hidden until `load_tasks` finds unread
        # rows (mirrors `SyncStatusWidget`'s own Clear-button idiom of
        # starting hidden rather than flashing visible-then-hidden).
        self.query_one("#scheduling-mark-all-read", Button).display = False
        # redesign PR-2, Task 2: glyph/title/subtitle (spec S4) replaces
        # the old Title/Type/Status/Next-Run shape -- a single primitive's
        # column set no longer fits a mixed reminder+definition list.
        table = self.query_one("#scheduling-task-table", DataTable)
        table.add_columns("", "Title", "Details")
        automations_table = self.query_one("#scheduling-automations-table", DataTable)
        automations_table.add_columns(
            "Name", "Family", "Lifecycle", "Health", "Model"
        )
        history_table = self.query_one(
            "#scheduling-automation-history-table", DataTable
        )
        history_table.add_columns("Time", "Event", "Summary")
        # task-23111 review F9: the relative next-run column ("in 25m")
        # is render-time text; refresh it periodically while visible.
        self._next_run_refresh_timer = self.set_interval(
            NEXT_RUN_REFRESH_SECONDS, self._refresh_next_run_rendering
        )
        self._request_tasks_refresh()
        self._request_automations_refresh()
        self._refresh_scheduler_liveness()
        self._start_server_notification_observer()
        self._schedule_catch_up_results_pull()

    def on_unmount(self) -> None:
        """Stop the notification observer + any pending pull debounce.

        Workbench-scoped lifecycle (plan ruling 3): neither must outlive
        this screen -- setting the cancel event is the observer's own
        documented stop signal (`EventObserver.run` checks it at every
        await point), so the background worker unwinds on its own instead
        of being force-cancelled mid-stream.
        """
        if self._notification_cancel_event is not None:
            self._notification_cancel_event.set()
        self._notification_cancel_event = None
        if self._results_pull_debounce_timer is not None:
            self._results_pull_debounce_timer.stop()
            self._results_pull_debounce_timer = None
        super().on_unmount()

    def _start_server_notification_observer(self) -> None:
        """Start the SSE notification observer (schedules-handoff PR-6
        task 4) -- workbench-scoped per plan ruling 3: begins here when a
        server connection is configured; `on_unmount` above stops it.

        This is the first real caller of
        `ServerNotificationEventObserver.observe()` --
        `NotificationsScopeService.observe_server_feed_events` has existed
        since 18940 slice 3 with nothing invoking it (survey §3).
        """
        service = self._service()
        scope_service = getattr(self.app_instance, "notifications_scope_service", None)
        if (
            service is None
            or scope_service is None
            or not self._server_available(service, self._active_server_id())
        ):
            return
        self._notification_cancel_event = asyncio.Event()
        self.run_worker(
            self._run_server_notification_observer,
            exclusive=True,
            group="schedules-notification-observer",
            exit_on_error=False,
        )

    async def _run_server_notification_observer(self) -> None:
        """Supervise the SSE notification observer for this screen's life.

        Reading `EventObserver.run()` end to end (event_observer.py),
        `observe()` can leave us in exactly four ways:

        1. **Cancelled** -- `cancel_event` was set (our `on_unmount`
           signal). `run()` returns cleanly, `result.cancelled is True`,
           checked at every await point (mid-stream-read AND mid-backoff-
           sleep), so this is near-immediate. We return -- no retry.
        2. **Clean stream end** -- the transport's async generator raises
           `StopAsyncIteration` (the server closed the SSE connection
           without error). `run()` returns normally, `cancelled=False`.
           Not fatal for a long-lived feed; we restart `observe()` after
           the flat backoff below.
        3. **Stale/unsupported cursor exhausted** -- `StaleCursorError`/
           `UnsupportedCursorError` are retried internally (with reset +
           backoff) up to `max_reconnects` times; once exhausted, `run()`
           STILL returns normally (`cancelled=False`) rather than raising.
           Same handling as case 2: restart.
        4. **Sustained transport failure** -- any other `Exception` from
           the transport is retried internally the same way, but once
           `max_reconnects` is exhausted `run()` RE-RAISES (after
           `observe()` records status="error" via `_record_status`). This
           is the one case that reaches our `except` below; we log and
           restart after the same backoff, so a down/flaky server
           degrades to "resumes once reachable again", never a
           permanently dead observer for the rest of this screen's life.

        A handler exception (case 5, hypothetical -- `_on_server_
        notification_event` performs no I/O and cannot realistically
        raise) would surface identically to case 4: `run()`'s inner loop
        has no handler-specific try/except, so it falls into the same
        generic `except Exception` path.

        `exit_on_error=False` on the `run_worker` call that invokes this
        coroutine is the backstop against a bug here still crashing the
        app (the established run_worker(exit_on_error) trap).

        Log discipline (fix round 1 -- a sustained failure used to dump a
        full ERROR-level traceback every ~5s restart, worst in the
        accepted profile-vanished edge case where `_resolve_server_
        event_scope` raises synchronously and the inner 5-reconnect
        absorption in `EventObserver.run()` never even engages): the
        FIRST failure of a given exception class logs one `warning` with
        just the exception summary (no traceback); identical-class
        repeats log at `debug`; a class change re-warns (a
        `ServerEventScopeRequiredError` outage turning into a genuine
        network error, say, is worth a fresh heads-up); a subsequent
        success logs one `info` and clears the remembered class.
        """
        scope_service = getattr(self.app_instance, "notifications_scope_service", None)
        cancel_event = self._notification_cancel_event
        if scope_service is None or cancel_event is None:
            return
        last_failure_class: type[BaseException] | None = None
        while not cancel_event.is_set():
            try:
                result = await scope_service.observe_server_feed_events(
                    handler=self._on_server_notification_event,
                    cancel_event=cancel_event,
                    max_reconnects=_NOTIFICATION_OBSERVER_MAX_RECONNECTS,
                )
            except Exception as exc:  # noqa: BLE001 - case 4/5 above, never fatal here
                if type(exc) is last_failure_class:
                    logger.debug(
                        f"Schedules notification observer still failing "
                        f"({exc.__class__.__name__}: {exc}); retrying"
                    )
                else:
                    logger.warning(
                        f"Schedules notification observer connection failed "
                        f"({exc.__class__.__name__}: {exc}); retrying"
                    )
                    last_failure_class = type(exc)
            else:
                if result.cancelled:
                    return
                if last_failure_class is not None:
                    logger.info("Schedules notification observer reconnected")
                    last_failure_class = None
            if cancel_event.is_set():
                return
            try:
                await asyncio.wait_for(
                    cancel_event.wait(),
                    timeout=_NOTIFICATION_OBSERVER_RESTART_DELAY_SECONDS,
                )
            except TimeoutError:
                pass

    async def _on_server_notification_event(self, event: Any) -> bool:
        """ACK every server notification; `automation_run_*` kinds
        schedule a debounced results pull (plan ruling 3).

        Always returns True (ack): a kind this screen doesn't act on must
        still advance the observer's durable cursor, or an unrelated
        notification stream would replay forever (`EventObserver.run`'s
        ack-then-advance contract). The server puts the automation_run_*
        vocabulary at `payload["data"]["kind"]` (ADR-077 phase-1
        pass-back, survey §3) -- `event_kind`/`payload_kind` are the SSE
        envelope's own generic fields, not this.
        """
        payload = getattr(event, "payload", None)
        data = payload.get("data") if isinstance(payload, Mapping) else None
        kind = data.get("kind") if isinstance(data, Mapping) else None
        if isinstance(kind, str) and kind.startswith("automation_run_"):
            self._schedule_results_pull()
        return True

    def _schedule_catch_up_results_pull(self) -> None:
        """Recover results whose notification was acked while unmounted.

        `_on_server_notification_event` acks EVERY event before the pull
        it schedules has run (it must -- the observer's ack-then-advance
        contract is what stops an unrelated stream replaying forever), so
        an event acked just before this screen went away advances the
        durable cursor and is never redelivered. Re-timing the ack is the
        wrong lever: the pull is a newest-window walk keyed on nothing but
        "what does the server have now", i.e. idempotent, so one pull at
        mount recovers whatever the lost event would have fetched -- and
        any number of lost events, not just one.

        Goes through the same debounced single-flight path as a live
        event, so mounting into an event burst still costs exactly one
        pull. Gated on a configured server for the same reason
        `_start_server_notification_observer` is: with no server there is
        nothing to pull and `_run_phase` would only record a sync error.
        """
        service = self._service()
        if service is None or not self._server_available(
            service, self._active_server_id()
        ):
            return
        self._schedule_results_pull()

    def _schedule_results_pull(self) -> None:
        """Debounce a notification-triggered results pull (plan ruling
        3): a burst of `automation_run_*` events collapses into ONE
        pull, same stop-and-restart timer shape as the queue filter's
        own debounce.
        """
        if self._results_pull_debounce_timer is not None:
            self._results_pull_debounce_timer.stop()
        self._results_pull_debounce_timer = self.set_timer(
            RESULTS_PULL_DEBOUNCE_SECONDS, self._start_results_pull
        )

    def _start_results_pull(self) -> None:
        self._results_pull_debounce_timer = None
        if self._results_pull_running:
            # A pull from an earlier debounce window is still in flight --
            # absorb this trigger into a single follow-up pull instead of
            # running two pulls concurrently (single-flight, no pile-up;
            # the worker-collision lesson).
            self._results_pull_rerun_requested = True
            return
        self._results_pull_running = True
        self.run_worker(
            self._pull_results_worker,
            exclusive=True,
            group="schedules-results-pull",
            exit_on_error=False,
        )

    async def _pull_results_worker(self) -> None:
        """Pull automation results once, then once more if a trigger
        landed while the first pull was running (rerun flag) -- never
        more than one queued follow-up.

        Reuses `SyncEngine._run_phase` -- the same containment `sync_now`
        uses for this exact phase (survey §2) -- rather than a full sync
        (the brief: "the narrowest callable, not a full sync").
        `_run_phase` never raises: a failure is recorded via
        `_record_sync_error` onto the persisted sync-error state
        `_refresh_owner_select` already renders, so a pull failure here
        surfaces exactly like a failed "s" sync (the existing sync-error
        path) -- and can never reach the observer's own coroutine, which
        is a wholly separate worker/group.
        """
        try:
            while True:
                service = self._service()
                if service is not None:
                    await service.sync_engine._run_phase(
                        service.owner_id,
                        "Automation results pull (notification)",
                        service.sync_engine._pull_results,
                    )
                    self._refresh_owner_select()
                    # A pull is exactly the event the Queue's unread dots
                    # and rail button exist for (final review F5).
                    self._refresh_results_surfaces()
                if not self._results_pull_rerun_requested:
                    return
                self._results_pull_rerun_requested = False
        finally:
            self._results_pull_running = False

    def _refresh_scheduler_liveness(self) -> None:
        """Update the scheduler-liveness line from the durable heartbeat
        (TASK-26025). Never raises -- a diagnostics read must not break the
        screen."""
        try:
            from datetime import datetime, timezone

            from ....config import get_cli_setting
            from ....Scheduling.constants import SCHEDULER_POLL_INTERVAL_SECONDS
            from ....Scheduling.scheduler_heartbeat import (
                default_heartbeat_path,
                read_heartbeat,
                scheduler_liveness_line,
            )

            poll = get_cli_setting(
                "scheduling",
                "scheduler_poll_interval_seconds",
                SCHEDULER_POLL_INTERVAL_SECONDS,
            )
            line = scheduler_liveness_line(
                read_heartbeat(default_heartbeat_path()),
                now=datetime.now(timezone.utc),
                poll_interval=float(poll or SCHEDULER_POLL_INTERVAL_SECONDS),
            )
            self._update_static_content(
                self.query_one("#scheduling-liveness", Static), line
            )
        except Exception:  # noqa: BLE001 -- liveness display never breaks the screen
            pass

    def _refresh_next_run_rendering(self) -> None:
        """Re-render the queue so relative next-run text stays honest.

        Also refreshes the scheduler-liveness line (TASK-26025) so a loop
        that dies while the screen is open turns visibly stale.

        Skips unless this screen is the top of the stack. (Textual's
        ``is_current`` also counts screens behind the top one --
        ``_background_screens`` always includes the screen directly
        beneath the top regardless of opacity -- so it cannot express
        "covered"; the suspend/resume handlers pause the timer while
        covered and refresh on uncover.) Also skips an empty queue:
        nothing to refresh, and the no-service path must keep its own
        detail-pane copy.
        """
        if self.app.screen is not self:
            return
        # TASK-26025: refresh liveness even on an empty queue -- a stall
        # with nothing queued is exactly the case AC#2 must distinguish.
        self._refresh_scheduler_liveness()
        # redesign PR-2, Task 2: `_visible_rows`, not the reminder-only
        # `_visible_tasks` -- a Queue showing only definition rows still
        # has relative next-run text that must not go stale.
        if not self._visible_rows:
            return
        self._render_table(tick=True)

    def on_screen_suspend(self) -> None:
        """Stop the relative-time refresh while another screen covers this.

        Hidden clocks must not tick unseen (TASK-23022); the resume
        handler refreshes immediately so no stale text is ever shown.
        """
        if self._next_run_refresh_timer is not None:
            self._next_run_refresh_timer.pause()

    def on_screen_resume(self) -> None:
        """Refresh relative times and restart the cadence when uncovered.

        No ``super().on_screen_resume()``: Textual's dispatcher invokes
        every handler along the MRO for one event (see BaseAppScreen's
        MRO contract), so the base handler runs regardless.
        """
        if self._next_run_refresh_timer is not None:
            self._next_run_refresh_timer.resume()
        self._refresh_next_run_rendering()

    def _sync_responsive_workbench(self) -> None:
        """Keep the primary queue and detail action visible at narrow widths."""
        compact = self.size.width <= SCHEDULES_COMPACT_WORKBENCH_MAX_WIDTH
        self.set_class(compact, "schedules-workbench-compact")
        # redesign PR-2, Task 3: the strip's own width-triggered compact
        # path -- additive, independent of `_apply_collapse`'s owner/
        # server-based collapse; reuses this same threshold rather than
        # inventing a second one.
        try:
            self.query_one("#scheduling-sync-status", SyncStatusWidget).set_compact(
                compact
            )
        except Exception:  # noqa: BLE001 - not mounted yet
            pass

    def _request_tasks_refresh(self, *, refresh_definitions: bool = True) -> None:
        """Schedule the task loader through its exclusive worker group.

        Args:
            refresh_definitions: Passed through to `load_tasks`. Default
                ``True`` preserves every call site's prior behavior (full
                reload). Reminder-only actions (delete/edit/mark/toggle/
                transfer/bulk-*, owner switch) pass ``False`` -- redesign
                PR-2 Task 2 review, finding 3: none of them can change
                which automation definitions exist, so re-running the
                full local+server definitions fetch on each one was pure
                waste on the hot refresh path. Mount and sync-completed/
                failed keep the default: sync can genuinely pull/push
                definitions, and mount has no prior snapshot to reuse.
        """

        async def _load() -> None:
            await self.load_tasks(refresh_definitions=refresh_definitions)

        self.run_worker(
            _load,
            exclusive=True,
            group="schedules-load-tasks",
        )  # type: ignore[arg-type]

    def _current_definitions(self) -> list[dict[str, Any]]:
        """The automation-definition dicts already sitting in the last-
        built unified rows -- reused by a reminder-only refresh instead
        of re-fetching them (redesign PR-2 Task 2 review, finding 3).
        """
        return [row.source_row for row in self._all_rows if row.kind == "definition"]

    def _update_mark_all_read_visibility(self) -> None:
        """Rail `Mark all read` visibility (redesign PR-2, Task 3): shown
        only when the unified rows carry unread automation results in
        total. `UnifiedRow.unread_count` is always 0 for a reminder row
        (reminders never produce results) so this sum is effectively the
        definitions' own unread total -- no separate query."""
        try:
            button = self.query_one("#scheduling-mark-all-read", Button)
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        button.display = sum(row.unread_count for row in self._all_rows) > 0

    @on(TabbedContent.TabActivated, pane="#scheduling-queue-tab")
    def _on_queue_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Arriving on the Queue tab while definitions are stale (review
        round 2) upgrades the next refresh to a full one, rather than
        waiting for a reminder action to (maybe never) trigger one."""
        if self._definitions_stale:
            self._request_tasks_refresh()

    async def load_tasks(self, *, refresh_definitions: bool = True) -> None:
        """Fetch reminders + automation definitions and build the unified
        Queue rows (redesign PR-2, Task 2).

        Three listings feed Task 1's `build_unified_rows`: reminders
        spans-owners (`SchedulingService.list_tasks(owner_id=None,
        include_projections=False)` -- watchlist/briefing projections
        stay out per spec S2 locked decision 2 and Task 1's own report,
        and `include_projections=False` (Task 2 review finding 2) stops
        `list_tasks` from building AND sorting them in the first place,
        since this call site discarded every one of them anyway), both
        definition halves (the Automations tab's existing local+server
        merge, reused verbatim -- or, when ``refresh_definitions`` is
        False AND nothing marked the cache stale, the definitions
        already in `self._all_rows` from the last full load, review
        finding 3), and one all-owners results listing (unread-count
        derivation only). The results read + row build are pushed off
        the event loop (`asyncio.to_thread`), the same "local DB read,
        off-thread" discipline `_load_local_automations` already uses.

        `self._definitions_stale` (review round 2): an Automations-tab
        mutation upgrades the NEXT call here to a full definitions fetch
        even when the caller only asked for `refresh_definitions=False`
        -- a reminder-only refresh must not keep painting a definitions
        snapshot a since-edited/transferred/run automation has outgrown.
        """
        service = self._scheduling_service
        if service is None:
            logger.debug("No scheduling_service available; cannot load tasks")
            await self._refresh_console_context()
            return

        try:
            combined = await service.list_tasks(
                owner_id=None, include_projections=False
            )
            # Defensive, not load-bearing: `include_projections=False`
            # already guarantees every row is a `ReminderTask`.
            reminders = [task for task in combined if isinstance(task, ReminderTask)]
            do_full_definitions_fetch = refresh_definitions or self._definitions_stale
            definitions = (
                await self._load_queue_definitions(service)
                if do_full_definitions_fetch
                else self._current_definitions()
            )
            if do_full_definitions_fetch:
                self._definitions_stale = False

            def _build_rows() -> list[UnifiedRow]:
                results = service.db.list_automation_results(
                    owner_id=None, limit=RESULTS_INBOX_LIMIT
                )
                return build_unified_rows(
                    reminders,
                    definitions,
                    results,
                    self._queue_local_definitions,
                )

            all_rows = await asyncio.to_thread(_build_rows)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load tasks")
            self.app_instance.notify(
                "Could not load tasks. Check the scheduling service and retry.",
                severity="error",
            )
            self._tasks = []
            self._all_rows = []
            self._update_mark_all_read_visibility()
            table = self.query_one("#scheduling-task-table", DataTable)
            table.clear()
            self.query_one("#scheduling-task-detail", TaskDetail).set_task(
                None, queue_empty=True
            )
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)
            self.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            ).set_definition(None)
            await self._refresh_console_context()
            return

        self._tasks = reminders
        # Marks must always refer to rows that still exist (task-23107
        # review F1): a task deleted or filtered out of existence must not
        # linger as an invisible mark a bulk verb would act on.
        self._marked_ids.intersection_update({task.id for task in self._tasks})
        self._all_rows = all_rows
        self._update_mark_all_read_visibility()
        self._render_table()
        await self._refresh_console_context()

    async def _load_queue_definitions(
        self, service: "SchedulingService"
    ) -> list[dict[str, Any]]:
        """Local + server automation-definition rows for the unified list.

        Reuses the Automations tab's own both-owners merge precedent
        (`_load_local_automations` + `_load_server_automations`) rather
        than a third fetch shape -- this is a SEPARATE fetch from
        `load_automations`'s own cadence (own tab, own refresh triggers),
        not a shared cache.

        Also stashes the UNFILTERED local listing in
        `self._queue_local_definitions` for `build_unified_rows`'s
        unread-count resolution (final review F2): the display merge
        above drops every local row that has a `server_id`, so a
        transferred definition's pre-transfer results resolved to
        nothing. Same single `list_automation_definitions` call -- the
        rows were already read and then filtered away.
        """
        try:
            local_rows = await asyncio.to_thread(service.db.list_automation_definitions)
        except Exception:  # noqa: BLE001
            # `owner_id`/"Queue list" context only (Qodo LOW) -- never
            # payload content. This is the sibling of the identical
            # message `_load_local_automations` logs for the Automations
            # tab's own refresh; without a call-site tag the two were
            # indistinguishable in the log.
            logger.exception(
                "Failed to load local automation definitions for the "
                "Queue list (owner_id={})",
                service.owner_id,
            )
            local_rows = []
        self._queue_local_definitions = local_rows
        local_items = self._device_only_automations(local_rows)
        server_client = getattr(service, "server_client", None)
        server_available = server_client is not None and self._server_available(
            service, self._active_server_id()
        )
        if not server_available:
            return local_items
        try:
            server_items, _total = await self._load_server_automations(server_client)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to load server automations for the Queue list (server_id={})",
                self._active_server_id(),
            )
            return local_items
        return local_items + server_items

    def _render_table(self, now: datetime | None = None, *, tick: bool = False) -> None:
        """Rebuild the unified queue rows from `self._all_rows` + the
        current chip + filter text (redesign PR-2, Task 2).

        ``tick`` marks the relative-time ticker's own re-render (the only
        caller that re-renders WITHOUT new data): it suppresses the
        definition-detail re-read for an unchanged selection (final
        review F12). Every other caller leaves it False, because data
        genuinely can change without the selection changing -- the
        PR-1 F4 lesson: a refresh-driven re-feed must still re-feed.

        Chip + search narrowing and ordering are Task 1's own pure
        functions (`filter_rows`/`sort_rows`) -- never re-derived here.
        `self._visible_rows` becomes the new 1:1 source of truth for the
        `DataTable`'s row index; `self._visible_tasks` is DERIVED from it
        (reminder rows only, in table order) purely so every existing
        reminder-action helper below keeps reading an unchanged shape.

        Restores the previously selected row (by its `UnifiedRow.row_id`,
        stable across BOTH kinds) when it is still visible after the
        chip/filter narrows, instead of always jumping back to row 0
        (task-15476): a filter keystroke or chip switch must not discard
        what the user was looking at.

        ``now`` is one shared reference for every row's relative
        next-run rendering (review F9: per-row ``datetime.now()`` let a
        single frame straddle a bucket boundary); injectable for
        deterministic tests.
        """
        render_now = now if now is not None else datetime.now(timezone.utc)
        previous_selected_row_id = self._selected_row_id
        self._visible_rows = sort_rows(
            filter_rows(self._all_rows, chip=self._chip, query=self._filter_text),
            self._chip,
        )
        self._visible_tasks = [
            row.source_row for row in self._visible_rows if row.kind == "reminder"
        ]
        # Owner suffix (plan ruling 4): hidden at compact width, evaluated
        # once per render pass -- `_sync_responsive_workbench` (on_mount/
        # on_resize) always runs before this, so `self.size` is current.
        compact_owner_suffix = self.size.width <= SCHEDULES_COMPACT_WORKBENCH_MAX_WIDTH

        table = self.query_one("#scheduling-task-table", DataTable)
        table.clear()
        for row in self._visible_rows:
            # Every cell is `Text`, never `str` (D8: `DataTable` runs a
            # `str` cell through `Text.from_markup`, which eats
            # user-authored `[...]` tokens -- the reminder title's own
            # bracket-safety precedent, reused for definition rows too).
            table.add_row(
                Text(row.glyph),
                _row_title_cell(
                    row,
                    marked_ids=self._marked_ids,
                    compact_owner_suffix=compact_owner_suffix,
                ),
                Text(_row_subtitle(row, render_now)),
                key=row.row_id,
            )
        self._update_pane_notice()

        if self._visible_rows:
            target_index = 0
            if previous_selected_row_id is not None:
                for index, row in enumerate(self._visible_rows):
                    if row.row_id == previous_selected_row_id:
                        target_index = index
                        break
            if table.row_count:
                table.move_cursor(row=target_index)
            self._update_detail_for_index(target_index, from_tick=tick)
        else:
            self._selected_row_id = None
            self._selected_task_id = None
            self.query_one("#scheduling-task-detail", TaskDetail).set_task(
                None, queue_empty=not self._tasks
            )
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)
            self.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            ).set_definition(None)
            self._show_queue_detail_pane("reminder")
            if self._all_rows and (self._filter_text.strip() or self._chip != "all"):
                # Everything filtered out: say so instead of "select a task".
                filter_text = self._filter_text.strip()
                if filter_text:
                    notice = (
                        f"No tasks match '{filter_text}'. "
                        "Clear the filter to see the queue."
                    )
                else:
                    # The chip row is hidden below width 84 (the `compact`
                    # class this same threshold sets), while the selected
                    # chip persists -- so don't point at a control that is
                    # not on screen (final review F10).
                    notice = "No tasks in this view."
                    if not self._chips_hidden():
                        notice += " Choose a different chip to see the queue."
                self._update_static_content(
                    self.query_one("#scheduling-task-detail-empty-state", Static),
                    notice,
                )

    def _chips_hidden(self) -> bool:
        """True when the chip row is off screen (`_scheduling.tcss`:
        `#scheduling-workbench.compact #scheduling-queue-chips { display:
        none }`). Reads the class `on_resize` actually set rather than
        re-deriving its width threshold here."""
        try:
            return self.query_one("#scheduling-workbench").has_class("compact")
        except Exception:  # noqa: BLE001 - not mounted yet
            return False

    def _show_queue_detail_pane(self, kind: RowKind) -> None:
        """Toggle which Queue detail widget is visible (redesign PR-2,
        Task 2): `TaskDetail` for a reminder row, `DefinitionDetail` for a
        definition row -- via the `pane-hidden` class, the SAME mechanism
        `on_resize` already uses for width-based hiding of independent
        panes (survey section 3's routing recipe). Orthogonal to that
        width-based hide: this only ever runs while `#scheduling-detail-
        pane` itself is shown.
        """
        is_reminder = kind == "reminder"
        self.query_one("#scheduling-task-detail", TaskDetail).set_class(
            not is_reminder, "pane-hidden"
        )
        self.query_one(
            "#scheduling-queue-definition-detail", DefinitionDetail
        ).set_class(is_reminder, "pane-hidden")

    @on(Input.Changed, "#scheduling-queue-filter")
    def _on_queue_filter_changed(self, event: Input.Changed) -> None:
        """Filter the queue rows by title substring (debounced).

        A settled render clears and rebuilds the whole `DataTable`, so it
        must not run on every keystroke (task-15476).
        """
        self._filter_text = event.value
        if self._filter_debounce_timer is not None:
            self._filter_debounce_timer.stop()
        self._filter_debounce_timer = self.set_timer(
            QUEUE_FILTER_DEBOUNCE_SECONDS, self._apply_queue_filter_debounced
        )

    def _apply_queue_filter_debounced(self) -> None:
        self._filter_debounce_timer = None
        self._render_table()

    #: redesign PR-2, Task 2: chip button id -> `Chip` value.
    _CHIP_BY_BUTTON_ID: dict[str, Chip] = {
        "scheduling-chip-all": "all",
        "scheduling-chip-active": "active",
        "scheduling-chip-paused": "paused",
        "scheduling-chip-completed": "completed",
    }

    @on(Button.Pressed, ".scheduling-queue-chip")
    def _on_queue_chip_pressed(self, event: Button.Pressed) -> None:
        """Switch the Queue's active chip (spec S3: All/Active/Paused/
        Completed) -- one handler for all four buttons, matched by id."""
        event.stop()
        chip = self._CHIP_BY_BUTTON_ID.get(event.button.id or "")
        if chip is not None:
            self._set_queue_chip(chip)

    def _set_queue_chip(self, chip: Chip) -> None:
        if chip == self._chip:
            return
        self._chip = chip
        for button_id, candidate in self._CHIP_BY_BUTTON_ID.items():
            self.query_one(f"#{button_id}", Button).variant = (
                "primary" if candidate == chip else "default"
            )
        self._render_table()

    @on(DataTable.RowHighlighted)
    def _on_task_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Update the detail pane when the user highlights a task row.

        Both guards below suppress `_render_table`'s own ECHOES, not any
        real cursor move (final review F12 -- the same unchanged-selection
        discipline `_on_automations_row_highlighted` has):

        1. `DataTable.clear()` posts a RowHighlighted for row 0 before
           the rows are re-added, so every render re-rendered row 0's
           detail (clobbering `_selected_row_id` on the way) before the
           restored row won it back. That event is stale by the time it
           is processed -- the table's live cursor has already moved on.
        2. `move_cursor` back to the restored row posts an echo for a row
           `_render_table` just called `_update_detail_for_index` with
           directly. A refresh's re-feed is that DIRECT call; repeating
           it here is pure duplication.
        """
        if event.cursor_row != event.data_table.cursor_row:
            return
        if not (0 <= event.cursor_row < len(self._visible_rows)):
            return
        if self._visible_rows[event.cursor_row].row_id == self._selected_row_id:
            return
        self._update_detail_for_index(event.cursor_row)

    def _incidents_for(self, task_id) -> list:
        """TASK-26027: the selected task's failure incidents (fail-safe).

        Reminders key their incidents by the raw task id; briefings (whose
        failures the briefing handler records) key by "briefing:<id>", so
        both spellings are queried.
        """
        db = getattr(self._scheduling_service, "db", None)
        reader = getattr(db, "list_task_incidents", None)
        if not callable(reader):
            return []
        rows: list = []
        for key in (str(task_id), f"briefing:{task_id}"):
            try:
                rows.extend(reader(key, limit=10))
            except Exception:  # noqa: BLE001 -- never breaks the pane
                pass
        return rows

    def _run_history_for(self, task_id) -> list:
        """TASK-26026: the selected task's durable run history, newest first.

        Fail-safe: a missing service/method or a read error yields an empty
        history rather than breaking the detail pane.
        """
        service = self._scheduling_service
        db = getattr(service, "db", None)
        reader = getattr(db, "list_task_runs", None)
        if not callable(reader):
            return []
        try:
            return list(reader(str(task_id), limit=8))
        except Exception:  # noqa: BLE001 -- history read never breaks the pane
            return []

    def _update_detail_for_index(self, index: int, *, from_tick: bool = False) -> None:
        """Render the highlighted Queue row's detail, routed by kind
        (redesign PR-2, Task 2). ``index`` is a `self._visible_rows`
        index (the table's own row index), NOT a `self._visible_tasks`
        one -- the two lists diverge whenever a definition row precedes
        the highlighted one.

        ``from_tick`` is set only by the relative-time ticker's own
        re-render: it skips the definition-detail worker when the
        selection has not changed (final review F12 -- 3 DB reads a
        minute for a row that did not move, against the ticker's own
        "no reload/DB on tick" contract). Deliberately NOT applied to
        refresh-driven calls: those re-feed on purpose, because the
        DATA can change while the selection stands still.
        """
        previously_selected_row_id = self._selected_row_id
        if not (0 <= index < len(self._visible_rows)):
            self._selected_row_id = None
            self._selected_task_id = None
            self.query_one("#scheduling-task-detail", TaskDetail).set_task(
                None, queue_empty=not self._tasks
            )
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)
            self.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            ).set_definition(None)
            return

        row = self._visible_rows[index]
        self._selected_row_id = row.row_id
        if row.kind == "reminder":
            task = row.source_row
            assert isinstance(task, ReminderTask)
            self._selected_task_id = task.id
            self._show_queue_detail_pane("reminder")
            task_detail = self.query_one("#scheduling-task-detail", TaskDetail)
            task_detail.set_task(
                task,
                run_history=self._run_history_for(task.id),
                incidents=self._incidents_for(task.id),
                # PR-3 task 3: same option source the create/edit modal's
                # own Timezone selector reads (`_task_timezones`), so the
                # pane's inline Timezone row editor offers the same zones.
                known_timezones=self._task_timezones(),
                # PR-3 task 5: same option source the create/edit forms'
                # own owner selector reads, so the Runs-on row's dropdown
                # offers the same choices.
                runs_on_options=self._runs_on_options()[0],
            )
            self._update_transfer_actions(task_detail, task)
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(task)
        else:
            # Definition rows expose no actions in this PR (viewable +
            # detail only, plan ruling 1) -- `_selected_task_id = None`
            # means every existing reminder action (`_selected_task`,
            # edit/mark/toggle/delete) already no-ops gracefully here,
            # with no new guard branches needed.
            self._selected_task_id = None
            self._show_queue_detail_pane("definition")
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(None)
            definition = row.source_row
            assert isinstance(definition, dict)
            if from_tick and row.row_id == previously_selected_row_id:
                # Same row, no new data (F12): the pane already shows
                # this definition's counts. Mirrors `_on_automations_row_
                # highlighted`'s own unchanged-selection guard.
                return
            self._request_queue_definition_detail(row.row_id, definition)

    def _update_transfer_actions(
        self, task_detail: TaskDetail, task: ReminderTask | ScheduledTask
    ) -> None:
        """Compute Move/Retry/Cancel disabled-reasons for the detail pane.

        Reasons come straight from `SchedulingService.transfer_refusal`/
        `cancel_refusal` (spec §6.4/§6.3) so the UI never re-derives the
        refusal rules itself (health quoting included, and -- fix round
        finding 1 -- cancel's own state branching too) -- each reason is
        only computed for the button `TaskDetail.set_task` already
        decided is structurally relevant (e.g. `to_local_reason` is never
        computed for a local row, which would always trivially refuse
        with "not server-owned" noise).
        """
        service = self._scheduling_service
        if service is None or not isinstance(task, ReminderTask):
            task_detail.set_transfer_reasons(
                to_server_reason=None,
                to_local_reason=None,
                retry_reason=None,
                cancel_reason=None,
                retry_errors=[],
            )
            task_detail.set_lifecycle_lock(None)
            task_detail.set_runs_on_transfer_errors([])
            return

        row = transfer_row_dict(task)
        is_server_owned = str(task.owner_id or "").startswith("server:")
        transfer_state = task.transfer_state

        # `transfer_refusal` already returns None for `to_server_failed`
        # (Task 6's retry leg deliberately excludes it from "already in
        # progress"), so this needs no extra state carve-out -- the same
        # call also backs the Retry button below.
        to_server_reason = (
            service.transfer_refusal(row, "to_server") if not is_server_owned else None
        )
        to_local_reason = (
            service.transfer_refusal(row, "to_local") if is_server_owned else None
        )
        cancel_reason = service.cancel_refusal(row)

        # A `to_server_failed` row is never server-owned, so this is the
        # exact same call `to_server_reason` already made above -- reused
        # rather than repeated.
        retry_reason: str | None = None
        retry_errors: list[str] = []
        if transfer_state == "to_server_failed":
            retry_reason = to_server_reason
            # fix round finding 3: keyed off the mutation's OWN owner_id
            # column, not a guess via "today's active server" -- a
            # `to_server_failed` row's mutation was recorded under
            # whatever server was active at the time of the failed
            # attempt, which silently stops matching after a server
            # switch if guessed instead of read.
            retry_errors = _pending_transfer_errors(service, "reminder_task", task.id)

        task_detail.set_transfer_reasons(
            to_server_reason=to_server_reason,
            to_local_reason=to_local_reason,
            retry_reason=retry_reason,
            cancel_reason=cancel_reason,
            retry_errors=retry_errors,
        )
        # spec §6.3 read-only-except-cancel (final review I7). Applied
        # AFTER set_transfer_reasons, which owns the same reason Static.
        task_detail.set_lifecycle_lock(service.transfer_lock_reason(row))
        # PR-3 task 5 fix round 1 (finding 2): the SAME `retry_errors`
        # feeds the Runs-on row's own failure text -- one source, not a
        # second derivation.
        task_detail.set_runs_on_transfer_errors(retry_errors)

    def _refuse_if_transfer_locked(self, task: Any, verb: str) -> bool:
        """Notify and return True when ``task`` is read-only mid-transfer.

        Spec §6.3 (final review I7): the key-bound Edit / Delete /
        Enable-Disable verbs share the detail pane's lock, so pressing
        `e`/`d`/`space` on an in-flight row says why instead of silently
        no-oping against the facade's own guard. The reason string comes
        from `SchedulingService.transfer_lock_reason` -- never re-derived
        here.
        """
        service = self._scheduling_service
        if service is None or not isinstance(task, ReminderTask):
            return False
        reason = service.transfer_lock_reason(transfer_row_dict(task))
        if reason is None:
            return False
        self.app_instance.notify(f"Cannot {verb}: {reason}", severity="warning")
        return True

    async def _refresh_console_context(self) -> None:
        """Load the latest Schedules Console-follow context."""
        latest_console_item = await self._latest_console_follow_item_from_adapter()
        latest_console_launch = None
        if latest_console_item is None:
            latest_console_launch = await self._latest_reading_digest_console_launch()
        self._apply_console_context(latest_console_item, latest_console_launch)

    async def _latest_console_follow_item_from_adapter(self) -> Any | None:
        adapter = getattr(self.app_instance, "home_active_work_adapter", None)
        build_dashboard_input = getattr(adapter, "build_dashboard_input", None)
        if not callable(build_dashboard_input):
            return None
        try:
            providers = getattr(self.app_instance, "providers_models", {}) or {}
            runtime_identity = RuntimeIdentity.from_state(
                self.app_instance.runtime_policy.state
            )
            has_recent_work = self.app_instance.screen_state_store.has_snapshots(
                runtime_identity
            )
            dashboard_input = build_dashboard_input(
                providers_models=providers,
                has_recent_work=has_recent_work,
            )
            if inspect.isawaitable(dashboard_input):
                dashboard_input = await dashboard_input
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to load Schedules Console follow item from Home active-work adapter.",
            )
            return None
        for item in tuple(getattr(dashboard_input, "active_work_items", ()) or ()):
            if (
                getattr(item, "source", None) == "Schedules"
                and bool(getattr(item, "console_available", False))
                and getattr(item, "item_id", None)
            ):
                return item
        return None

    async def _latest_reading_digest_console_launch(self) -> dict[str, Any] | None:
        service = getattr(self.app_instance, "local_media_reading_service", None)
        list_outputs = getattr(service, "list_reading_digest_outputs", None)
        if not callable(list_outputs):
            return None
        try:
            output_listing = list_outputs(schedule_id=None, limit=1, offset=0)
            if inspect.isawaitable(output_listing):
                output_listing = await output_listing
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to load Schedules Console launch context from local reading digest outputs.",
            )
            return None
        items = (
            output_listing.get("items") if isinstance(output_listing, Mapping) else None
        )
        latest_output = next(iter(tuple(items or ())), None)
        if not isinstance(latest_output, Mapping):
            return None

        output_id = latest_output.get("output_id") or latest_output.get("id")
        if output_id in (None, ""):
            return None

        metadata = latest_output.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        schedule_name = str(
            metadata.get("schedule_name")
            or latest_output.get("schedule_name")
            or latest_output.get("schedule_id")
            or ""
        ).strip()
        title = str(
            latest_output.get("title") or schedule_name or "Reading digest output"
        ).strip()
        item_count = metadata.get("item_count", latest_output.get("item_count"))
        payload = {
            "target_id": f"local:reading_digest_output:{output_id}",
            "output_id": output_id,
            "schedule_id": latest_output.get("schedule_id"),
            "schedule_name": schedule_name or None,
            "download_url": latest_output.get("download_url")
            or latest_output.get("storage_path"),
            "created_at": latest_output.get("created_at"),
            "item_count": item_count,
        }
        return {
            "source": "schedules",
            "title": title,
            "payload": payload,
            "status": "ready",
            "recovery": "Review this reading digest output from Schedules or return to Library.",
            "action_label": "Open schedule output",
        }

    def _apply_console_context(
        self,
        latest_console_item: Any | None,
        latest_console_launch: dict[str, Any] | None,
    ) -> None:
        self._current_console_follow_item = latest_console_item
        self._latest_console_follow_item_id = (
            getattr(latest_console_item, "item_id", None)
            if latest_console_item is not None
            else None
        )
        self._latest_console_launch_kwargs = latest_console_launch
        self._latest_console_context_loaded = True
        self._update_follow_button_state()

    def _update_follow_button_state(self) -> None:
        task_detail = self.query_one("#scheduling-task-detail", TaskDetail)
        available = (
            self._latest_console_follow_item_id is not None
            or self._latest_console_launch_kwargs is not None
        )
        task_detail.set_follow_available(available)

    @on(DeleteTaskRequested)
    def _on_delete_task_requested(self, event: DeleteTaskRequested) -> None:
        """Delete the requested task and refresh the queue."""
        event.stop()
        self._marked_ids.discard(event.task.id)
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot delete task.",
                severity="warning",
            )
            return

        async def _delete_and_refresh() -> None:
            try:
                # The ROW's owner (final review F4): deleting a `server:`
                # row while "This device" is active took the local-only
                # branch -- no server call, no tombstone -- and the row
                # came back on the next pull.
                await service.delete_reminder(
                    event.task.id, owner_id=event.task.owner_id
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to delete reminder {}", event.task.id)
                self.app_instance.notify(
                    f"Failed to delete '{event.task.title}'.",
                    severity="error",
                )
            else:
                self.app_instance.notify(
                    f"Deleted '{event.task.title}'.",
                    severity="information",
                )
            self._request_tasks_refresh(refresh_definitions=False)

        self.run_worker(
            _delete_and_refresh,
            exclusive=True,
            group="schedules-delete-task",
        )  # type: ignore[arg-type]

    @on(TransferToServerRequested)
    def _on_transfer_to_server_requested(
        self, event: TransferToServerRequested
    ) -> None:
        event.stop()
        self._begin_transfer(event.task, "to_server")

    @on(TransferToLocalRequested)
    def _on_transfer_to_local_requested(
        self, event: TransferToLocalRequested
    ) -> None:
        event.stop()
        self._begin_transfer(event.task, "to_local")

    @on(RetryTransferRequested)
    def _on_retry_transfer_requested(self, event: RetryTransferRequested) -> None:
        # Obligation (f)/spec §6.1.5: retrying a `to_server_failed` row is
        # the SAME facade call as a first-time begin -- `transfer_refusal`
        # deliberately narrows its "already in progress" check to exclude
        # `to_server_failed`, and `begin_transfer_to_server`'s CAS accepts
        # both `None` and `to_server_failed` as its starting state.
        event.stop()
        self._begin_transfer(event.task, "to_server")

    @on(CancelTransferRequested)
    def _on_cancel_transfer_requested(self, event: CancelTransferRequested) -> None:
        event.stop()
        self._cancel_transfer(event.task)

    @staticmethod
    def _transfer_confirm_dialog(
        name: str, direction: str, warnings: list[str]
    ) -> ConfirmationDialog:
        """Build the Move confirm dialog (spec §6.1/§6.2/§6.4) -- shared by
        the reminder (`_begin_transfer`) and Automations-tab
        (`_begin_automation_transfer`) flows (Task 7 fix round item 1: two
        near-identical call sites, one copy of the confirm-dialog copy)."""
        destination_label = "the server" if direction == "to_server" else "this device"
        lines = [f'Move "{escape_markup(name)}" to {destination_label}?']
        if warnings:
            lines.append("")
            lines.extend(f"- {escape_markup(warning)}" for warning in warnings)
        if direction == "to_server":
            lines.append("")
            lines.append(
                "It keeps running on this device until the server accepts "
                "the transfer -- nothing goes dark while this is only "
                "queued."
            )
        return ConfirmationDialog(
            title="Move to server" if direction == "to_server" else "Move to local",
            message="\n".join(lines),
            confirm_label="Move",
            cancel_label="Cancel",
        )

    @staticmethod
    def _transfer_pending_toast_text(name: str, direction: str) -> str:
        """Honest §6.1.1 "still runs here" / dormant-copy copy, shared by
        the reminder and Automations-tab transfer flows."""
        if direction == "to_server":
            return (
                f"'{name}' is queued to move to the server -- it still "
                "runs on this device until the server accepts it."
            )
        return (
            f"'{name}' is queued to move to this device -- a dormant copy "
            "is ready and will arm once the server releases it."
        )

    def _begin_transfer(self, task: ReminderTask, direction: str) -> None:
        """Confirm, then start a transfer for ``task`` (spec §6.1/§6.2).

        Always confirms first -- Move is not a casual action, and the
        dialog is the one place `transfer_warnings` (imminent one-time
        `run_at`, non-transferring `timeout_seconds`) actually reaches the
        user before they commit, not only when something happens to be
        wrong. `transfer_refusal` is checked again defensively (the
        button should already be disabled when refused) before ever
        opening the dialog.
        """
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot start a transfer.",
                severity="warning",
            )
            return

        row = transfer_row_dict(task)
        reason = service.transfer_refusal(row, direction)
        if reason is not None:
            self.app_instance.notify(reason, severity="warning")
            return
        warnings = service.transfer_warnings(row, direction)
        dialog = self._transfer_confirm_dialog(task.title, direction, warnings)

        async def _confirm_and_begin() -> None:
            confirmed = await self.app.push_screen_wait(dialog)
            if not confirmed:
                return
            try:
                if direction == "to_server":
                    outcome = await service.begin_transfer_to_server(
                        "reminder_task", task.id
                    )
                else:
                    outcome = await service.begin_transfer_to_local(
                        "reminder_task", task.id
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to begin transfer for {}", task.id)
                self.app_instance.notify(
                    f"Failed to start the transfer for '{task.title}'.",
                    severity="error",
                )
                self._request_tasks_refresh(refresh_definitions=False)
                return
            self._notify_transfer_outcome(task, direction, outcome)
            self._request_tasks_refresh(refresh_definitions=False)

        self.run_worker(
            _confirm_and_begin,
            exclusive=True,
            group="schedules-transfer",
        )  # type: ignore[arg-type]

    def _notify_transfer_outcome(
        self, task: ReminderTask, direction: str, outcome: "TransferOutcome"
    ) -> None:
        """Toast a transfer's result -- honest about what actually happened
        (spec §6.1.1: a queued-not-sent transfer never claims the task
        stopped running here)."""
        if outcome.status == "pending":
            self.app_instance.notify(
                self._transfer_pending_toast_text(task.title, direction),
                severity="information",
            )
        elif outcome.status == "refused":
            self.app_instance.notify(
                outcome.reason or f"Could not move '{task.title}'.",
                severity="warning",
            )
        elif outcome.status == "not_found":
            self.app_instance.notify(
                f"'{task.title}' no longer exists.", severity="warning"
            )

    def _cancel_transfer(self, task: ReminderTask) -> None:
        """Cancel ``task``'s in-progress transfer immediately.

        No confirm dialog: cancel is the escape hatch spec §6.3 exists
        for, and gating the escape hatch behind its own confirmation
        fights the point of it. `task.id` is whichever row is currently
        selected -- for a release's dormant local copy that is already
        the copy's OWN id (it is a first-class queue row, not reached
        through the mirror), which is exactly the id `cancel_transfer`
        needs for that leg (Task 6 handoff note).
        """
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot cancel the "
                "transfer.",
                severity="warning",
            )
            return

        async def _cancel_and_refresh() -> None:
            try:
                outcome = await service.cancel_transfer("reminder_task", task.id)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to cancel transfer for {}", task.id)
                self.app_instance.notify(
                    f"Failed to cancel the transfer for '{task.title}'.",
                    severity="error",
                )
                self._request_tasks_refresh(refresh_definitions=False)
                return
            if outcome.status == "cancelled":
                self.app_instance.notify(
                    _cancel_toast_text(task.title),
                    severity="information",
                )
            else:
                self.app_instance.notify(
                    outcome.reason
                    or f"Could not cancel the transfer for '{task.title}'.",
                    severity="warning",
                )
            self._request_tasks_refresh(refresh_definitions=False)

        self.run_worker(
            _cancel_and_refresh,
            exclusive=True,
            group="schedules-transfer",
        )  # type: ignore[arg-type]

    @on(Button.Pressed, "#scheduling-new-task")
    def _on_new_task_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_new_task_chooser()

    @on(Button.Pressed, "#scheduling-new-automation")
    def _on_new_automation_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_create_automation()

    @on(Button.Pressed, "#scheduling-mark-all-read")
    def _on_mark_all_read_pressed(self, event: Button.Pressed) -> None:
        """Rail `Mark all read` (redesign PR-2, Task 3) -- reachable from
        the Queue tab, unlike the `a` keybinding which requires the
        Results tab active."""
        event.stop()
        self._rail_mark_all_read()

    @on(Button.Pressed, "#scheduling-conflicts-badge")
    def _on_conflicts_badge_pressed(self, event: Button.Pressed) -> None:
        """Status-strip conflicts badge (redesign PR-2, Task 3, plan
        ruling 4) -- switches to the Conflicts tab, no overlay."""
        event.stop()
        self.query_one("#scheduling-tabs", TabbedContent).active = (
            "scheduling-conflicts-tab"
        )

    @on(Button.Pressed, "#schedules-follow-in-console")
    def follow_latest_schedule_run_in_console(self, event: Button.Pressed) -> None:
        """Hand off the active schedule run or digest output to the Console."""
        event.stop()
        if event.button.disabled:
            return
        target_id = self._latest_console_follow_item_id
        if target_id:
            open_active_item_in_console = getattr(
                self.app_instance, "open_active_home_item_in_console", None
            )
            if not callable(open_active_item_in_console):
                self.app_instance.notify(
                    "Console follow is unavailable for Schedules in this runtime.",
                    severity="warning",
                )
                return
            open_active_item_in_console(
                target_id=target_id,
                target_route="chat",
            )
            return

        launch_kwargs = self._latest_console_launch_kwargs
        if launch_kwargs is not None:
            open_in_console = getattr(
                self.app_instance, "open_console_for_live_work", None
            )
            if not callable(open_in_console):
                self.app_instance.notify(
                    "Console launch is unavailable for Schedules in this runtime.",
                    severity="warning",
                )
                return
            open_in_console(**launch_kwargs)
            return

        self.app_instance.notify(
            SCHEDULES_EMPTY_CONSOLE_RECOVERY.disabled_tooltip,
            severity="warning",
        )

    def _task_timezones(self) -> list[str]:
        """Zones already used by tasks, offered in the form's selector."""
        zones: list[str] = []
        for task in self._tasks:
            zone = getattr(task, "timezone", None)
            if zone and zone not in zones:
                zones.append(zone)
        return zones

    def _runs_on_options(self) -> tuple[list[tuple[str, str]], str]:
        """Runs-on choices for the reminder/automation forms.

        The current screen owner leads and is always the default (spec
        sec 8); "Server (<id>)" is offered only when a server owner is
        actually connected (mirrors `_server_available`'s own gate). The
        owner might not literally be "local" or match the offered server
        option (e.g. it drifted to a different server id) -- appended as
        a labeled fallback so the Select never receives a value outside
        its own options (same precedent as the reminder form's timezone
        selector, review F4).
        """
        service = self._service()
        owner_id = service.owner_id if service else "local"
        options: list[tuple[str, str]] = [("This device", "local")]
        active_server_id = self._active_server_id()
        if self._server_available(service, active_server_id):
            options.append((f"Server ({active_server_id})", f"server:{active_server_id}"))
        if owner_id not in {value for _, value in options}:
            options.append((owner_id, owner_id))
        return options, owner_id

    def action_create_reminder(self) -> None:
        """Open the create-reminder form."""
        options, default_owner = self._runs_on_options()
        self.app.push_screen(
            ReminderForm(
                known_timezones=self._task_timezones(),
                available_owners=options,
                default_owner=default_owner,
            ),
            callback=self._on_reminder_form_result,
        )

    def action_create_automation(self) -> None:
        """Open the create-recurring-question-automation form (task-5)."""
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot create an automation.",
                severity="warning",
            )
            return
        options, default_owner = self._runs_on_options()
        self.app.push_screen(
            AutomationDefinitionForm(
                service, available_owners=options, default_owner=default_owner
            ),
            callback=self._on_automation_form_result,
        )

    def _open_new_task_chooser(self) -> None:
        """Ask which kind of scheduled task to create (task-5)."""
        self.app.push_screen(NewTaskChoiceModal(), callback=self._on_new_task_choice)

    def _on_new_task_choice(self, choice: str | None) -> None:
        if choice == "reminder":
            self.action_create_reminder()
        elif choice == "recurring_question":
            self.action_create_automation()

    def _on_automation_form_result(
        self, outcome: Any | None, *, was_edit: bool = False
    ) -> None:
        """Notify and refresh the Automations list after a definition save.

        `was_edit` only changes the toast wording ("updated" vs
        "created") -- an edit reusing the create-mode "created" copy
        would misreport what actually happened.

        Refreshes the QUEUE too, not just the Automations list (final
        review F1): Task 3 moved the create entry point onto the Queue
        rail (`Create ▾` -> "Recurring question…"), so this save's
        flagship path never leaves the Queue tab -- `TabActivated`, the
        only consumer of `_definitions_stale`, never fires, and the
        automation the user just created stayed invisible on the surface
        it was created from. Symmetric with the reminder half
        (`_on_reminder_form_result`), which has always refreshed here.
        """
        if outcome is None:
            return
        # redesign PR-2 Task 2 review round 2: a real definition save --
        # the Queue's cached definitions rows may now be outdated.
        self._definitions_stale = True
        status = getattr(outcome, "status", None)
        verb = "updated" if was_edit else "created"
        if status == "saved":
            self.app_instance.notify(
                f"Automation {verb}.", severity="information"
            )
        elif status == "queued":
            self.app_instance.notify(
                f"Automation {verb} locally — it will sync to the server.",
                severity="information",
            )
        self._request_automations_refresh()
        # Full refresh (definitions included): `_definitions_stale` is set
        # above, so a `refresh_definitions=False` call would upgrade
        # itself anyway -- ask for what this actually needs.
        self._request_tasks_refresh()

    def _on_reminder_form_result(
        self, form_data: dict[str, Any] | None, task_id: str | None = None
    ) -> None:
        """Create or update a reminder from the form and refresh the queue."""
        if form_data is None:
            return

        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot save the scheduled task.",
                severity="warning",
            )
            return

        # Routing only (task-5): which owner the reminder is written under,
        # not a ReminderTask field. Defaults to the service's current owner
        # so a form built before this selector existed (or a caller that
        # omits it) behaves exactly as before.
        target_owner = form_data.pop("owner_id", None) or service.owner_id
        # Name the owner it was created for (label, not raw id: same
        # vocabulary as the "Runs on" selector the owner was picked from).
        # The old copy also told the user to "switch to that owner to see
        # it" -- true when this list was owner-scoped, wrong since Task 1
        # made it span owners (final review F7, spec §4: the cross-owner
        # list "dissolves" that wart). The row appears here immediately
        # whatever owner it was created under.
        owner_label = {
            value: label for label, value in self._runs_on_options()[0]
        }.get(target_owner, target_owner)
        created_message = (
            "Scheduled task created."
            if target_owner == service.owner_id
            else f"Scheduled task created for {owner_label}."
        )

        async def _save_and_refresh() -> None:
            # The owner is threaded through the call rather than flipped onto
            # the service around it: `service.owner_id` is shared mutable
            # state that the sync/refresh/run-now workers also read, and a
            # flip held across an awaited network round-trip is visible to
            # every one of them.
            try:
                if task_id is None:
                    await service.create_reminder(form_data, owner_id=target_owner)
                    self.app_instance.notify(created_message, severity="information")
                else:
                    await service.update_reminder(
                        task_id, form_data, owner_id=target_owner
                    )
                    self.app_instance.notify(
                        "Scheduled task updated.", severity="information"
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to save reminder")
                self.app_instance.notify(
                    "Failed to save the scheduled task. Check the form values and try again.",
                    severity="error",
                )
            self._request_tasks_refresh(refresh_definitions=False)

        self.run_worker(
            _save_and_refresh,
            exclusive=True,
            group="schedules-save-reminder",
        )  # type: ignore[arg-type]

    @on(AcknowledgeIncidentRequested)
    def _on_acknowledge_incident(self, event) -> None:
        """TASK-26027: acknowledge one incident, then refresh the detail."""
        event.stop()
        db = getattr(self._scheduling_service, "db", None)
        ack = getattr(db, "acknowledge_incident", None)
        if not callable(ack):
            return
        try:
            from datetime import datetime, timezone

            ack(int(event.incident_id), datetime.now(timezone.utc))
        except Exception:  # noqa: BLE001 -- ack failure never breaks the screen
            logger.debug("acknowledge_incident failed")
            return
        # re-render the detail so the acked incident drops out of the
        # alerting set and the button hides. Indexed over `_visible_rows`,
        # NOT `_visible_tasks` (final review F3): `_update_detail_for_
        # index` takes the table's own row index, and the two lists
        # diverge the moment a definition row sorts above the highlighted
        # reminder -- the old code then rendered a DIFFERENT row's detail
        # and moved the selection with it, silently, while the cursor
        # stayed put.
        if self._selected_row_id is not None:
            for index, row in enumerate(self._visible_rows):
                if row.row_id == self._selected_row_id:
                    self._update_detail_for_index(index)
                    break

    @on(EditTaskRequested)
    def _on_edit_task_requested(self, event: EditTaskRequested) -> None:
        """Open the reminder form pre-filled for editing."""
        event.stop()
        options, default_owner = self._runs_on_options()
        self.app.push_screen(
            ReminderForm(
                event.task,
                known_timezones=self._task_timezones(),
                available_owners=options,
                default_owner=default_owner,
            ),
            callback=lambda result: self._on_reminder_form_result(
                result, event.task.id
            ),
        )

    @on(EnableTaskRequested)
    def _on_enable_task_requested(self, event: EnableTaskRequested) -> None:
        """Enable the requested reminder and refresh the queue."""
        event.stop()
        self._set_reminder_enabled(event.task, True)

    @on(DisableTaskRequested)
    def _on_disable_task_requested(self, event: DisableTaskRequested) -> None:
        """Disable the requested reminder and refresh the queue."""
        event.stop()
        self._set_reminder_enabled(event.task, False)

    @on(ReminderFieldEditRequested)
    def _on_reminder_field_edit_requested(
        self, event: ReminderFieldEditRequested
    ) -> None:
        """A Frequency row's inline editor committed a value (PR-3 task 3)."""
        event.stop()
        self._edit_reminder_field(event.task, event.payload, event.row)

    def _edit_reminder_field(
        self,
        task: ReminderTask,
        payload: dict[str, Any],
        row: DetailValueRow,
    ) -> None:
        """Persist one Frequency row's edit via Task 2's validation bridge.

        `TaskDetail` has already closed the row's editor (`end_edit`,
        restoring the OLD display) before posting the request -- a
        failure needs no separate "restore" step here, only `show_error`.
        Success repaints authoritatively from a fresh read: the SAME
        reminder-only refresh (`refresh_definitions=False`) every other
        reminder mutation in this file uses, which re-selects the row by
        id and re-feeds `TaskDetail.set_task` -- so the row shows the
        value the bridge actually persisted, not a locally-guessed one.
        """
        service = self._scheduling_service
        if service is None:
            row.show_error(
                "Scheduling service is unavailable; cannot save this edit."
            )
            return

        async def _edit_and_refresh() -> None:
            try:
                outcome = await service.edit_reminder_fields(task.id, payload)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to edit reminder field")
                row.show_error("Failed to save this edit.")
                return
            if outcome.status != "saved":
                message = "; ".join(
                    str(err.get("message") or "")
                    for err in outcome.errors
                    if err.get("message")
                ) or "This edit could not be saved."
                row.show_error(message)
                return
            row.clear_error()
            self._request_tasks_refresh(refresh_definitions=False)

        self.run_worker(
            _edit_and_refresh,
            exclusive=True,
            # Per-ROW group (final review M12): one group for the whole
            # KIND meant committing a second row's editor cancelled the
            # first commit mid-flight -- possibly after its write landed,
            # so the success repaint never ran. `exclusive=True` still
            # holds, now within one row.
            group=f"schedules-edit-reminder-field-{row.row_key}",
        )  # type: ignore[arg-type]

    @on(DefinitionFieldEditRequested)
    def _on_definition_field_edit_requested(
        self, event: DefinitionFieldEditRequested
    ) -> None:
        """A Details/Frequency row's inline editor committed a value
        (PR-3 task 4)."""
        event.stop()
        self._edit_definition_field(event.definition, event.payload, event.row)

    def _edit_definition_field(
        self,
        definition: dict[str, Any],
        payload: dict[str, Any],
        row: DetailValueRow,
    ) -> None:
        """Persist one Details/Frequency row's edit via `save_definition`.

        `DefinitionDetail` has already closed the row's editor (`end_
        edit`, restoring the OLD display) before posting the request --
        a failure needs no separate "restore" step here, only `show_
        error`. `definition["id"]` may be a LOCAL row id or, for a row
        shown from a pure server fetch with no local shadow yet, the
        SERVER's id -- resolved to a real local id the same way the
        existing full-modal Edit action already does (`_resolve_local_
        definition_id`, `_edit_selected_automation`'s own precedent).
        Success repaints authoritatively via the SAME staleness-plus-
        refresh seam every other definition mutation in this file uses
        (run-now, transfer begin/cancel) -- `_definitions_stale = True`
        + `_request_automations_refresh()`, which reloads the Automations
        tab's own table+detail immediately and marks the Queue's unified
        list stale for its own next (lazy, tab-activation-gated) refresh
        -- PLUS `_repaint_queue_definition_detail`, because that lazy
        Queue refresh is not a repaint of the Queue tab's own SECOND
        `DefinitionDetail` instance (final review F4/I4).
        """
        service = self._scheduling_service
        if service is None:
            row.show_error(
                "Scheduling service is unavailable; cannot save this edit."
            )
            return

        async def _edit_and_refresh() -> None:
            local_id = await self._resolve_local_definition_id(service, definition)
            if local_id is None:
                row.show_error(
                    "Could not prepare this automation for editing — see "
                    "the log."
                )
                return
            owner_id = str(definition.get("owner_id") or "local")
            try:
                outcome = await service.save_definition(
                    payload, owner_id, definition_id=local_id
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to edit automation definition field for {}", local_id
                )
                row.show_error("Failed to save this edit.")
                return
            if outcome.status not in ("saved", "queued"):
                message = "; ".join(
                    str(err.get("message") or "")
                    for err in outcome.errors
                    if err.get("message")
                ) or "This edit could not be saved."
                row.show_error(message)
                return
            row.clear_error()
            self._definitions_stale = True
            self._request_automations_refresh()
            await self._repaint_queue_definition_detail(
                service, local_id, definition
            )

        self.run_worker(
            _edit_and_refresh,
            exclusive=True,
            # Per-ROW group (final review M12) -- see `_edit_reminder_
            # field`'s own group for why.
            group=f"schedules-edit-definition-field-{row.row_key}",
        )  # type: ignore[arg-type]

    async def _repaint_queue_definition_detail(
        self,
        service: "SchedulingService",
        local_id: str,
        definition: dict[str, Any],
    ) -> None:
        """Repaint the QUEUE tab's `DefinitionDetail` for ``local_id``.

        `DefinitionDetail` is mounted TWICE -- `#scheduling-automation-
        detail` (Automations tab) and `#scheduling-queue-definition-
        detail` (Queue tab) -- and `_request_automations_refresh` only
        ever repaints the first. The Queue one is painted from
        `_update_detail_for_index`, which early-returns for the same row
        on a tick, so nothing repainted it after an in-pane edit: the
        editor closed, the row restored the OLD value, and stayed that
        way indefinitely even though the edit had persisted (final review
        F4/I4). `_toggle_definition_lifecycle` got this right by looping
        over both widget ids via `apply_lifecycle`; this is the same
        both-homes discipline for a field edit, which needs the
        authoritative re-read `apply_lifecycle`'s single known column
        does not.

        Only paints when the Queue's selected row IS this definition --
        which is also what makes the widget guaranteed-mounted here. The
        row id is matched against BOTH ids: `build_unified_rows` keys the
        row on whatever id the merged listing carried (the SERVER id for
        a pure server fetch with no local shadow yet), while the edit
        itself went through `_resolve_local_definition_id`.
        """
        if self._selected_row_id not in {
            f"definition:{local_id}",
            f"definition:{definition.get('id') or ''}",
        }:
            return
        fresh = await asyncio.to_thread(
            service.db.get_automation_definition, local_id
        )
        if fresh is None:
            return
        await self._load_queue_definition_detail(self._selected_row_id, fresh)

    @on(DefinitionLifecycleToggleRequested)
    def _on_definition_lifecycle_toggle_requested(
        self, event: DefinitionLifecycleToggleRequested
    ) -> None:
        """The header Pause/Resume button was pressed (PR-3 task 4 --
        `set_definition_lifecycle`'s first UI caller)."""
        event.stop()
        self._toggle_definition_lifecycle(event.definition, event.action)

    def _toggle_definition_lifecycle(
        self, definition: dict[str, Any], action: str
    ) -> None:
        """Pause/resume one automation via `set_definition_lifecycle`.

        Optimistic repaint (task-4 brief): on success, every mounted
        `DefinitionDetail` instance currently showing this definition is
        patched + repainted in place (`apply_lifecycle`) BEFORE the
        slower background refresh below runs, so neither pane shows a
        stale label while that worker is still fetching. Task 2's own
        DB-level pull-guard (`ScheduledTasksDB.upsert_automation_
        definitions_from_server`'s lifecycle skip) is what then keeps a
        sync pull racing that background refresh from reverting the
        value it eventually reads back -- this method does not need to
        know about that guard, only rely on it already existing.
        """
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot update the "
                "automation.",
                severity="warning",
            )
            return
        name = str(definition.get("name") or definition.get("id") or "")
        definition_id = str(definition.get("id") or "")

        async def _toggle_and_refresh() -> None:
            local_id = await self._resolve_local_definition_id(service, definition)
            if local_id is None:
                self.app_instance.notify(
                    f"Could not prepare '{name}' — see the log.",
                    severity="error",
                )
                return
            try:
                outcome = await service.set_definition_lifecycle(local_id, action)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to toggle lifecycle for automation {}", local_id
                )
                self.app_instance.notify(
                    f"Failed to update '{name}'.", severity="error"
                )
                return
            if outcome.status != "saved":
                message = (
                    outcome.errors[0]["message"]
                    if outcome.errors
                    else f"Could not update '{name}'."
                )
                self.app_instance.notify(message, severity="warning")
                return
            new_lifecycle = _LIFECYCLE_TOGGLE_RESULTS[action]
            definition["lifecycle"] = new_lifecycle
            for widget_id in (
                "#scheduling-automation-detail",
                "#scheduling-queue-definition-detail",
            ):
                try:
                    detail = self.query_one(widget_id, DefinitionDetail)
                except Exception:  # noqa: BLE001 - not mounted yet
                    continue
                detail.apply_lifecycle(definition_id, new_lifecycle)
            self._definitions_stale = True
            self._request_automations_refresh()

        self.run_worker(
            _toggle_and_refresh,
            exclusive=True,
            group="schedules-definition-lifecycle",
        )  # type: ignore[arg-type]

    # -- Owner-row transfer dropdown (PR-3 task 5, spec §7 flow) -------------
    #
    # A SECOND, row-scoped surface onto the SAME PR-5 transfer facade the
    # legacy Move/Retry/Cancel buttons (`_begin_transfer`/`_cancel_
    # transfer`/`_begin_automation_transfer`/`_cancel_automation_transfer`
    # above/below) already drive -- deliberately independent end to end
    # (own events, own helpers), not a refactor of them: coexistence is a
    # pinned requirement (task-5 brief), and the two surfaces differ in
    # how a refusal renders (this one uses `DetailValueRow.show_error`,
    # the legacy ones toast/write the tab's shared inline notice).

    async def _run_owner_transfer(
        self,
        *,
        table_kind: str,
        row_id: str,
        row_dict: dict[str, Any],
        name: str,
        direction: str,
        row: DetailValueRow,
        refresh,
    ) -> None:
        """Shared move/retry body for both panes' Runs-on row.

        `transfer_refusal` runs FIRST (health-quoting preserved, since it
        is the SAME call `_begin_transfer`/`_begin_automation_transfer`
        make) -- a refusal renders inline via `row.show_error`, never the
        legacy toast/notice. Allowed -> the SAME `ConfirmationDialog` +
        `transfer_warnings` + honest-toast shape those flows already use;
        confirmed -> `begin_transfer_to_server`/`to_local` by `direction`
        with `row_id` (the CURRENTLY DISPLAYED row's own local id for
        both directions -- a `to_local` release's returned dormant-copy
        id is a DIFFERENT id, `outcome.row_id`, relevant to a later
        Cancel on that new row, not to this call).
        """
        service = self._scheduling_service
        if service is None:
            row.show_error(
                "Scheduling service is unavailable; cannot start a transfer."
            )
            return
        reason = service.transfer_refusal(row_dict, direction)
        if reason is not None:
            row.show_error(reason)
            return
        warnings = service.transfer_warnings(row_dict, direction)
        dialog = self._transfer_confirm_dialog(name, direction, warnings)
        confirmed = await self.app.push_screen_wait(dialog)
        if not confirmed:
            return
        try:
            if direction == "to_server":
                outcome = await service.begin_transfer_to_server(table_kind, row_id)
            else:
                outcome = await service.begin_transfer_to_local(table_kind, row_id)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to begin owner-row transfer for {}", row_id)
            row.show_error(f"Failed to start the transfer for '{name}'.")
            return
        if outcome.status == "refused":
            row.show_error(outcome.reason or f"Could not move '{name}'.")
            return
        if outcome.status == "not_found":
            row.show_error(f"'{name}' no longer exists.")
            return
        row.clear_error()
        self.app_instance.notify(
            self._transfer_pending_toast_text(name, direction),
            severity="information",
        )
        refresh()

    async def _run_owner_cancel(
        self,
        *,
        table_kind: str,
        row_id: str,
        name: str,
        row: DetailValueRow,
        refresh,
    ) -> None:
        """Shared cancel body for both panes' Runs-on row mini-bar.

        No confirm dialog (same rationale `_cancel_transfer`'s own
        docstring gives: cancel is the escape hatch). `row_id` is
        whichever row the pane is CURRENTLY DISPLAYING -- for a release
        leg (`from_server_pending`) that is already the DORMANT COPY's
        own id, never the mirror's: the mirror's own `transfer_state`
        stays untouched by `create_local_copy_from_mirror` (survey §3),
        so this row-level Cancel affordance is only ever proactively
        shown (`_configure_runs_on_row`) on a row whose OWN state is
        in-flight -- which, for a release, can only be the copy itself.
        """
        service = self._scheduling_service
        if service is None:
            row.show_error(
                "Scheduling service is unavailable; cannot cancel the transfer."
            )
            return
        try:
            outcome = await service.cancel_transfer(table_kind, row_id)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to cancel owner-row transfer for {}", row_id)
            row.show_error(f"Failed to cancel the transfer for '{name}'.")
            return
        if outcome.status != "cancelled":
            row.show_error(
                outcome.reason or f"Could not cancel the transfer for '{name}'."
            )
            return
        row.clear_error()
        self.app_instance.notify(_cancel_toast_text(name), severity="information")
        refresh()

    @on(ReminderOwnerActionRequested)
    def _on_reminder_owner_action_requested(
        self, event: ReminderOwnerActionRequested
    ) -> None:
        """The reminder pane's Runs-on row dropdown/mini-bar fired
        (PR-3 task 5)."""
        event.stop()
        self._reminder_owner_action(event.task, event.action, event.row)

    def _reminder_owner_action(
        self, task: ReminderTask, action: str, row: DetailValueRow
    ) -> None:
        def _refresh() -> None:
            self._request_tasks_refresh(refresh_definitions=False)

        if action == "cancel":
            async def _do() -> None:
                await self._run_owner_cancel(
                    table_kind="reminder_task",
                    row_id=task.id,
                    name=task.title,
                    row=row,
                    refresh=_refresh,
                )

            self.run_worker(_do, exclusive=True, group="schedules-transfer")  # type: ignore[arg-type]
            return

        direction = "to_server" if action == "retry" else action
        row_dict = transfer_row_dict(task)

        async def _do() -> None:
            await self._run_owner_transfer(
                table_kind="reminder_task",
                row_id=task.id,
                row_dict=row_dict,
                name=task.title,
                direction=direction,
                row=row,
                refresh=_refresh,
            )

        self.run_worker(_do, exclusive=True, group="schedules-transfer")  # type: ignore[arg-type]

    @on(DefinitionOwnerActionRequested)
    def _on_definition_owner_action_requested(
        self, event: DefinitionOwnerActionRequested
    ) -> None:
        """A definition pane's Runs-on row dropdown/mini-bar fired
        (PR-3 task 5)."""
        event.stop()
        self._definition_owner_action(event.definition, event.action, event.row)

    def _definition_owner_action(
        self, definition: dict[str, Any], action: str, row: DetailValueRow
    ) -> None:
        service = self._scheduling_service
        if service is None:
            row.show_error(
                "Scheduling service is unavailable; cannot start a transfer."
            )
            return
        name = str(definition.get("name") or definition.get("id") or "")

        def _refresh() -> None:
            self._request_automations_refresh()

        async def _do() -> None:
            # fix round 1 finding 1: unconditional, BEFORE resolving --
            # same rule `_begin_automation_transfer`/`_cancel_automation_
            # transfer` follow (schedules_workbench.py:2766-2778).
            # `_resolve_local_definition_id` can itself mirror a brand
            # new local row (`upsert_automation_definitions_from_server`)
            # the first time a pure server-fetch definition is touched --
            # regardless of which branch below this lands on (refused,
            # failed, `local_id is None`, or a genuine success), the
            # Automations tab's cached list may now be outdated.
            self._definitions_stale = True
            local_id = await self._resolve_local_definition_id(service, definition)
            if local_id is None:
                row.show_error(
                    "Could not prepare this automation for transfer — see "
                    "the log."
                )
                return
            if action == "cancel":
                await self._run_owner_cancel(
                    table_kind="automation_definition",
                    row_id=local_id,
                    name=name,
                    row=row,
                    refresh=_refresh,
                )
                return
            direction = "to_server" if action == "retry" else action
            # A fresh read, like `_begin_automation_transfer`'s own
            # precedent -- `self._definition` may be a raw server-fetch
            # dict without the local row's `transfer_state`/`lifecycle`.
            db_row = await asyncio.to_thread(
                service.db.get_automation_definition, local_id
            )
            row_dict = db_row if db_row is not None else definition
            await self._run_owner_transfer(
                table_kind="automation_definition",
                row_id=local_id,
                row_dict=row_dict,
                name=name,
                direction=direction,
                row=row,
                refresh=_refresh,
            )

        self.run_worker(_do, exclusive=True, group="schedules-automation-transfer")  # type: ignore[arg-type]

    @on(RunReminderNowRequested)
    def _on_run_reminder_now_requested(self, event: RunReminderNowRequested) -> None:
        """Dispatch the requested reminder immediately."""
        event.stop()
        self._run_reminder_now(event.task)

    def action_run_task_now(self) -> None:
        """Run the highlighted task immediately (``r`` key).

        Routes by active tab: the Automations tab's ``r`` dispatches a
        server-side run (ADR-077 -- the server owns execution); the
        Results tab's ``r`` marks the selected result read instead
        (schedules-handoff PR-6 task 3 -- a natural reading of the same
        key); everywhere else it is the local reminder Run-now
        (task-18938).
        """
        try:
            active_pane = self.query_one("#scheduling-tabs", TabbedContent).active
        except Exception:  # noqa: BLE001
            active_pane = None
        if active_pane == "scheduling-automations-tab":
            definition = self._selected_automation()
            if definition is not None:
                self._run_automation_now(definition)
            return
        if active_pane == "scheduling-results-tab":
            self._review_selected_result("read")
            return
        task = self._selected_reminder_task()
        if task is None:
            # Never swallow the key silently (final review F8): on a
            # definition row `r` did nothing at all, with no message.
            self.app_instance.notify(
                self._no_task_notice("run"), severity="warning"
            )
            return
        self._run_reminder_now(task)

    def _selected_reminder_task(self) -> ReminderTask | None:
        """Return the highlighted task when it is a reminder (not a projection)."""
        for task in self._visible_tasks:
            if task.id == self._selected_task_id and isinstance(task, ReminderTask):
                return task
        return None

    def _run_reminder_now(self, task: ReminderTask) -> None:
        """Dispatch one reminder through the scheduler's own path (task-18938)."""
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot run the scheduled task.",
                severity="warning",
            )
            return
        loop = getattr(self.app_instance, "scheduler_loop", None)
        if loop is None:
            self.app_instance.notify(
                "The scheduler is not running; cannot run scheduled tasks manually.",
                severity="warning",
            )
            return

        # ADR-077 decision 1: server-scoped reminders are executed by the
        # server; a local Run-now would be a dispatch on the wrong side.
        # The refusal is precise rather than the generic did-not-run copy.
        # The predicate is the shared source of truth (queue.py) so the
        # UI can never drift from the scheduler/service refusal behavior.
        from tldw_chatbook.Scheduling.scheduler.queue import (
            is_server_scoped_owner,
        )

        if is_server_scoped_owner(getattr(task, "owner_id", None)):
            self.app_instance.notify(
                f"'{task.title}' is server-scheduled: the server runs it and "
                "delivers the notification -- it cannot be run from here.",
                severity="warning",
            )
            return

        was_disabled = not bool(getattr(task, "enabled", True))

        async def _run_and_refresh() -> None:
            try:
                result = await service.run_reminder_now(task.id, loop=loop)
                if result is None:
                    self.app_instance.notify(
                        f"'{task.title}' did not run -- it is missing, the "
                        "handler for it is unavailable, or its handler "
                        "failed (the task's status shows which).",
                        severity="warning",
                    )
                else:
                    suffix = " (still disabled)" if was_disabled else ""
                    self.app_instance.notify(
                        f"'{task.title}' ran now{suffix}.",
                        severity="information",
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to run reminder now")
                self.app_instance.notify(
                    f"Failed to run '{task.title}'.",
                    severity="error",
                )
            self._request_tasks_refresh(refresh_definitions=False)

        self.run_worker(
            _run_and_refresh,
            exclusive=True,
            group="schedules-run-reminder-now",
        )  # type: ignore[arg-type]

    def _request_automations_refresh(self) -> None:
        """Schedule the automations loader through its exclusive worker group.

        NOTE: this alone must NOT set `self._definitions_stale` -- it is
        also called at mount and from sync-completed/failed
        (`schedules-load-automations` fires concurrently with the
        Queue's own mount-time full refresh), and doing so here caused a
        real regression (round 2 fix-round-1 draft): the Queue tab's
        `TabbedContent.TabActivated` fires for the INITIAL default-active
        tab too, so a stale flag set by mount's own automations refresh
        was immediately consumed by the tab-activation handler, forcing
        a SECOND full Queue reload on every mount. Staleness is instead
        marked at each genuine mutation call site (create/edit save,
        run-now, transfer begin/cancel) -- see their own `self.
        _definitions_stale = True` lines.
        """
        self.run_worker(
            self.load_automations,
            exclusive=True,
            group="schedules-load-automations",
        )  # type: ignore[arg-type]

    async def load_automations(self) -> None:
        """Fetch and merge local + server automation definitions (task-5 fix round).

        This tab used to be server-only: a locally-saved recurring
        question had no on-screen home. Local rows are now merged in
        (owner distinguished via `automation_name_cell`'s Name-cell
        prefix) so a local save's refresh actually shows the new row.
        """
        notice = self.query_one("#scheduling-automations-notice", Static)
        table = self.query_one("#scheduling-automations-table", DataTable)
        service = self._scheduling_service
        if service is None:
            self._automations = []
            self._selected_automation_id = None
            table.clear()
            self._update_static_content(
                notice, "Server automations need a connected server."
            )
            self._clear_automation_history(
                "Run history needs a connected server."
            )
            self.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            ).set_definition(None)
            return

        # task-15476 discipline: a rebuild must reconcile the selection by
        # id -- keep the cursor on the same definition when it survives the
        # refresh, and clear it when it does not, so `r`/`e` can never act
        # on a row the user is no longer looking at.
        previous_selection = self._selected_automation_id

        local_items = await self._load_local_automations(service)

        server_client = getattr(service, "server_client", None)
        server_available = server_client is not None and self._server_available(
            service, self._active_server_id()
        )
        server_items: list[dict[str, Any]] = []
        total_server = 0
        server_error = False
        if server_available:
            try:
                server_items, total_server = await self._load_server_automations(
                    server_client
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to load server automations (server_id={})",
                    self._active_server_id(),
                )
                server_error = True

        items = local_items + server_items
        self._automations = items
        table.clear()
        for definition in items:
            # Every cell goes in as `Text`, never `str` (task 6 round 2,
            # D8). `DataTable` runs string cells through `rich.text.Text.
            # from_markup`, whose tag regex matches `\[[a-z#/@]...]` -- so
            # a server row's own owner prefix, `[http://127.0.0.1:8020]
            # ...`, was consumed whole and server automations rendered
            # with NO ownership prefix while the pane's own count line
            # still said "1 automation on the server". (The old
            # `server:42` fixture value would have been eaten identically,
            # so no fixture shape could have caught this.) The cell
            # formatter returns a `Text` untouched, so nothing re-parses
            # content that came from outside this module -- structural,
            # rather than depending on picking the right escape for
            # whichever parser consumes the cell, which is the mistake
            # this saga has now made three times.
            table.add_row(
                Text(automation_name_cell(definition)),
                Text(str(definition.get("family", "?"))),
                Text(str(definition.get("lifecycle", "?"))),
                Text(str(definition.get("health", "?"))),
                Text(automation_execution_target_label(definition)),
                key=str(definition.get("id")),
            )
        row_keys = [str(definition.get("id")) for definition in items]
        if previous_selection in row_keys:
            # Restoring the cursor fires RowHighlighted, which re-records
            # the same id -- belt and braces, set both explicitly.
            table.cursor_coordinate = (row_keys.index(previous_selection), 0)
            # ...and that RowHighlighted hits the unchanged-id early
            # return, so the pane would keep painting the PRE-refresh row
            # (final review F4: edit a definition's model/cron/sources and
            # save -- the table cell updated, the detail pane beside it
            # did not, until the user selected another row and came back).
            # The row DATA changed even though its id did not, so re-feed
            # the pane explicitly. The detail worker is exclusive within
            # its own group, so repeated refreshes are latest-wins, never
            # a pile-up.
            self._request_automation_detail(previous_selection)
        else:
            self._selected_automation_id = None
            self._clear_automation_history("Select an automation to see its history.")
            self.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            ).set_definition(None)

        notice_text = self._automations_notice_text(
            local_items, server_items, server_available, total_server, server_error
        )
        self._update_static_content(notice, notice_text)

    async def _load_local_automations(
        self, service: "SchedulingService"
    ) -> list[dict[str, Any]]:
        """Every definition that exists ONLY on this device, off the event loop.

        That is not the same as `owner_id="local"`: an automation authored
        offline with "Runs on: Server" is stored under `owner_id=
        "server:<id>"` with `server_id IS NULL` until a sync pushes it, so
        filtering on the local owner hid it from this half while the
        server half could not know about it either -- it appeared in
        NEITHER list (final review I5). Any row with no `server_id`
        belongs here, whoever owns it; rows that HAVE a `server_id` are the
        server half's by construction, so the two halves cannot duplicate.

        Rows in that pending state are stamped `pending_sync` so the
        renderer can say so (`automation_name_cell`) and the actions that
        need a real server id can refuse honestly rather than calling the
        server with a local uuid.

        Health is never persisted (`automation_health.py`'s own docstring)
        -- it is computed fresh here the same way `run_automation_now`
        computes it before dispatching, so the column never shows the
        create-time placeholder (`execution_unavailable`) as if it were
        live.
        """
        try:
            all_rows = await asyncio.to_thread(service.db.list_automation_definitions)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load local automation definitions")
            return []
        return self._device_only_automations(all_rows)

    def _device_only_automations(
        self, all_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """The device-only half of a local definitions listing, decorated.

        Split out of `_load_local_automations` (final review F2) so the
        Queue loader can keep the UNFILTERED listing for its unread-count
        resolution while still deriving the same display half from it --
        one `list_automation_definitions` call per refresh either way.
        """
        from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

        rows = [row for row in all_rows if not row.get("server_id")]
        for row in rows:
            if is_server_scoped_owner(row.get("owner_id")):
                row["pending_sync"] = True
            health, _reason = compute_local_health(self.app_instance, row)
            row["health"] = health
        return rows

    async def _load_server_automations(
        self, server_client: Any
    ) -> tuple[list[dict[str, Any]], int]:
        """Follow `has_more` pages so the tab never silently hides the tail
        of a large definition list; the cap is a defensive bound, not an
        expected cliff.

        Every item's `owner_id` is OVERWRITTEN with this connection's
        `server:<active-server-id>` scope -- never read from the payload.
        These rows are server-scoped by construction (this IS the server
        fetch), and a server's self-reported `owner_id` is its own raw
        user id, which has no reason to match our scoping convention:
        live verification against a real tldw_server (task 6, D1) got
        `"owner_id": "1"`, so the older stamp-only-when-absent guard
        passed a present, non-prefixed value straight through. Every
        downstream consumer (`is_server_scoped_owner`, the Name-cell
        prefix, run-now routing, `_resolve_local_definition_id`'s mirror
        lookup, transfer refusals) then read the row as LOCAL: server
        automations rendered `[This device]`, `r` refused with the
        local-health message, and `m` refused with "This automation no
        longer exists."

        This is the only ingestion boundary that needed the fix: both DB
        mirror upserts (`upsert_automation_definitions_from_server`,
        `upsert_automation_results_from_server`) already exclude the
        payload's `owner_id` and stamp the caller's owner scope instead.
        """
        items: list[dict[str, Any]] = []
        total = 0
        offset = 0
        while True:
            response = await server_client.list_automation_definitions(
                limit=50, offset=offset
            )
            page = list(response.get("items", []))
            items.extend(page)
            total = int(response.get("total", len(items)) or 0)
            offset += len(page)
            if (
                not page
                or not response.get("has_more")
                or len(items) >= AUTOMATIONS_LOAD_MAX_ROWS
            ):
                break
        owner_id = f"server:{self._active_server_id()}"
        for item in items:
            item["owner_id"] = owner_id
        return items, total

    @staticmethod
    def _automations_notice_text(
        local_items: list[dict[str, Any]],
        server_items: list[dict[str, Any]],
        server_available: bool,
        total_server: int,
        server_error: bool,
    ) -> str:
        """Compose the Automations-pane notice honestly from what actually loaded.

        A server failure never hides local rows that DID load -- the two
        sources degrade independently, so the server segment reports its
        own outcome and a local-count addendum is appended only when there
        is one to report.
        """
        if server_error:
            base = "Could not load server automations — see the log."
        elif server_available:
            shown = len(server_items)
            suffix = f" (showing {shown} of {total_server})" if total_server > shown else ""
            base = (
                f"{shown} automation{'' if shown == 1 else 's'} on the server{suffix}."
                if shown
                else "No automations on the server yet."
            )
        else:
            base = "Server automations need a connected server."
        if local_items:
            base += f" {len(local_items)} on this device."
        return base

    def _clear_automation_history(self, notice_text: str) -> None:
        """Reset the run-history pane to an explanatory notice."""
        table = self.query_one("#scheduling-automation-history-table", DataTable)
        notice = self.query_one("#scheduling-automation-history-notice", Static)
        title = self.query_one("#scheduling-automation-history-title", Static)
        table.clear()
        self._update_static_content(notice, notice_text)
        self._update_static_content(title, "Run history")

    def _request_automation_history(self, definition_id: str) -> None:
        """Schedule the audit-trail loader through its exclusive worker group."""
        # run_worker takes no worker arguments in Textual 8.x -- bind the id
        # in a closure (same shape as _run_automation_now's _run).
        async def _load() -> None:
            await self._load_automation_history(definition_id)

        self.run_worker(
            _load,
            exclusive=True,
            group="schedules-load-automation-history",
        )  # type: ignore[arg-type]

    async def _load_automation_history(self, definition_id: str) -> None:
        """Fetch and render one definition's durable execution-audit trail."""
        table = self.query_one("#scheduling-automation-history-table", DataTable)
        notice = self.query_one("#scheduling-automation-history-notice", Static)
        title = self.query_one("#scheduling-automation-history-title", Static)
        # A newer selection may have won the race with this worker; render
        # nothing for a stale definition id.
        if definition_id != self._selected_automation_id:
            return
        definition = self._selected_automation()
        name = str((definition or {}).get("name") or definition_id)
        self._update_static_content(title, f"Run history — {name}")
        # Drop the previous definition's rows BEFORE awaiting: a slow fetch
        # must never leave another definition's trail under the new title.
        table.clear()
        self._update_static_content(notice, "Loading run history…")

        from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

        owner_id = (definition or {}).get("owner_id")
        if (definition or {}).get("pending_sync"):
            # Never synced: the audit endpoint has no id to look up, and
            # asking it with a local uuid would render a bare error.
            table.clear()
            self._update_static_content(
                notice,
                "This automation hasn't synced to the server yet, so it has "
                "no run history.",
            )
            return
        if not is_server_scoped_owner(owner_id):
            # Honest gap (task-5 fix round): local automation runs are not
            # tracked in a durable audit trail yet -- only the server side
            # is. Never claim a server-shaped history for a local row.
            table.clear()
            self._update_static_content(
                notice, "Local automation history isn't available yet."
            )
            return

        service = self._scheduling_service
        server_client = getattr(service, "server_client", None) if service else None
        if server_client is None:
            table.clear()
            self._update_static_content(
                notice, "Run history needs a connected server."
            )
            return
        try:
            response = await server_client.list_automation_definition_audit(
                definition_id
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to load automation audit trail (definition_id={})",
                definition_id,
            )
            table.clear()
            self._update_static_content(
                notice, "Could not load the run history — see the log."
            )
            return
        if definition_id != self._selected_automation_id:
            return
        items = list(response.get("items", []))
        total = int(response.get("total", len(items)) or 0)
        table.clear()
        for event in items:
            created = str(event.get("created_at") or "")
            # Keep the timestamp compact: date and minute-level time,
            # no microseconds or timezone noise in a table cell.
            stamp = created[:16].replace("T", " ") if created else "?"
            summary = str(event.get("summary") or "")
            # `Text`, not `str` -- same D8 rule as the definitions table
            # above. `summary` is free-form server text ("Run failed:
            # ChatConfigurationError: ..."), the likeliest of all these
            # cells to carry a bracket token.
            table.add_row(
                Text(stamp),
                Text(str(event.get("event_type") or "?")),
                Text(summary),
            )
        suffix = f" of {total}" if total > len(items) else ""
        self._update_static_content(
            notice,
            f"{len(items)} event{'' if len(items) == 1 else 's'}{suffix}."
            if items
            else "No recorded events for this automation yet.",
        )

    def _request_automation_detail(self, definition_id: str) -> None:
        """Schedule the definitions detail pane's DB reads through their
        own exclusive worker group (schedules-redesign PR-1, Task 4): the
        pane's counts are local-only sqlite reads, taken off the event
        loop the same way `_load_local_automations` already reads
        `service.db.*` from inside a worker coroutine."""
        # run_worker takes no worker arguments in Textual 8.x -- bind the id
        # in a closure (same shape as _request_automation_history's _load).
        async def _load() -> None:
            await self._load_automation_detail(definition_id)

        self.run_worker(
            _load,
            exclusive=True,
            group="schedules-load-automation-detail",
        )  # type: ignore[arg-type]

    async def _load_automation_detail(self, definition_id: str) -> None:
        """Paint the definitions detail pane for `definition_id`.

        `DefinitionDetail.set_definition` performs no I/O itself; this
        method fetches the Task 2 count seams off the event loop
        (`asyncio.to_thread`) and passes the results in.
        """
        detail = self.query_one("#scheduling-automation-detail", DefinitionDetail)
        definition = self._selected_automation()
        # A newer selection may have won the race with this worker; render
        # nothing for a stale definition id (same guard `_load_automation_
        # history` uses).
        if definition is None or definition_id != self._selected_automation_id:
            return
        service = self._scheduling_service
        if service is None:
            detail.set_definition(
                definition, run_count=0, last_run=None, unread_count=0
            )
            # Same fallback `_update_transfer_actions` uses for `TaskDetail`
            # when there is no service to derive a real reason from.
            detail.set_lifecycle_lock(None)
            detail.set_runs_on_transfer_errors([])
            return

        run_count, last_run, unread_count, history_error = (
            await self._fetch_definition_detail_counts(
                service, definition, definition_id
            )
        )
        if definition_id != self._selected_automation_id:
            return
        detail.set_definition(
            definition,
            run_count=run_count,
            last_run=last_run,
            unread_count=unread_count,
            history_error=history_error,
            known_timezones=self._task_timezones(),
            # PR-3 task 5: same option source the Queue reminder pane's
            # own Runs-on row reads (`_update_detail_for_index`).
            runs_on_options=self._runs_on_options()[0],
        )
        # PR-3 task 4: `DefinitionDetail` gains the SAME transfer-lock
        # wiring `TaskDetail` has (survey point 10) -- `reason` comes from
        # `SchedulingService.transfer_lock_reason` (never re-derived in
        # the widget), fed right after `set_definition` per that
        # method's own docstring, same as `_update_transfer_actions` does
        # for the reminder pane.
        detail.set_lifecycle_lock(service.transfer_lock_reason(definition))
        # PR-3 task 5 fix round 1 (finding 2): the Runs-on row's own
        # failure text, same source the legacy Retry button would use.
        detail.set_runs_on_transfer_errors(
            _definition_transfer_errors(service, definition)
        )

    async def _fetch_definition_detail_counts(
        self,
        service: "SchedulingService",
        definition: dict[str, Any],
        definition_id: str,
    ) -> tuple[int, dict[str, Any] | None, int, bool]:
        """Off-thread run_count/last_run/unread_count read for one
        definition -- shared by the Automations tab's own detail pane
        (`_load_automation_detail`) and the Queue tab's definition-row
        routing (redesign PR-2, Task 2: `_load_queue_definition_detail`),
        same DB reads, same owner-scoping (final review F11), same
        never-paint-0-off-a-failed-read guard (F14).
        """

        def _read_counts() -> tuple[int, dict[str, Any] | None, int]:
            # Both run reads are owner-scoped (final review F11): they sit
            # in one group on screen, and `convert_row_to_server_mirror`'s
            # "converted" path rewrites `owner_id` while keeping the local
            # id -- an unscoped count beside a scoped last-run rendered
            # "Run count: 3" next to "Last run: Never run".
            owner_id = str(definition.get("owner_id") or "local")
            run_count = service.db.count_automation_runs(definition_id, owner_id)
            runs = service.db.list_automation_runs(owner_id, definition_id, limit=1)
            unread_count = service.db.count_unread_results(
                owner_id=None, definition_id=definition_id
            )
            return run_count, (runs[0] if runs else None), unread_count

        try:
            run_count, last_run, unread_count = await asyncio.to_thread(_read_counts)
            return run_count, last_run, unread_count, False
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to load automation detail counts (definition_id={})",
                definition_id,
            )
            # Never paint 0/Never run off a read that blew up (F14): the
            # pane says the read failed, matching how `_load_automation_
            # history` reports its own read failure.
            return 0, None, 0, True

    def _request_queue_definition_detail(
        self, row_id: str, definition: dict[str, Any]
    ) -> None:
        """Schedule the Queue tab's definition-detail counts through their
        own exclusive worker group (redesign PR-2, Task 2) -- mirrors
        `_request_automation_detail`'s shape, separate group so a Queue
        selection and an Automations-tab selection can never contend."""

        async def _load() -> None:
            await self._load_queue_definition_detail(row_id, definition)

        self.run_worker(
            _load,
            exclusive=True,
            group="schedules-load-queue-definition-detail",
        )  # type: ignore[arg-type]

    async def _load_queue_definition_detail(
        self, row_id: str, definition: dict[str, Any]
    ) -> None:
        """Paint the Queue tab's `DefinitionDetail` sibling for the
        highlighted definition row (redesign PR-2, Task 2). Reuses how
        the Automations tab loads its own detail pane's counts, off the
        event loop.
        """
        detail = self.query_one(
            "#scheduling-queue-definition-detail", DefinitionDetail
        )
        definition_id = str(definition.get("id") or "")
        service = self._scheduling_service
        if service is None:
            if row_id == self._selected_row_id:
                detail.set_definition(
                    definition, run_count=0, last_run=None, unread_count=0
                )
                detail.set_lifecycle_lock(None)
                detail.set_runs_on_transfer_errors([])
            return
        run_count, last_run, unread_count, history_error = (
            await self._fetch_definition_detail_counts(
                service, definition, definition_id
            )
        )
        # A newer selection may have won the race with this worker; render
        # nothing for a stale row (same guard `_load_automation_detail` uses).
        if row_id != self._selected_row_id:
            return
        detail.set_definition(
            definition,
            run_count=run_count,
            last_run=last_run,
            unread_count=unread_count,
            history_error=history_error,
            known_timezones=self._task_timezones(),
            # PR-3 task 5: same option source the Automations-tab sibling
            # instance's own Runs-on row reads (`_load_automation_detail`).
            runs_on_options=self._runs_on_options()[0],
        )
        # PR-3 task 4: same transfer-lock wiring as `_load_automation_
        # detail`'s Automations-tab pane -- this is the Queue tab's own
        # sibling instance of the SAME widget class, so it needs the
        # same call, independently (each `DefinitionDetail` instance
        # locks itself; there's no shared state between them).
        detail.set_lifecycle_lock(service.transfer_lock_reason(definition))
        # PR-3 task 5 fix round 1 (finding 2): same as the Automations-tab
        # sibling instance above.
        detail.set_runs_on_transfer_errors(
            _definition_transfer_errors(service, definition)
        )

    @on(DataTable.RowHighlighted, "#scheduling-automations-table")
    def _on_automations_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        """Track the highlighted definition for Run-now, its history pane,
        and its detail pane (schedules-redesign PR-1, Task 4)."""
        new_id = (
            str(event.row_key.value) if event.row_key and event.row_key.value else None
        )
        if new_id == self._selected_automation_id:
            return
        self._selected_automation_id = new_id
        if new_id is None:
            self._clear_automation_history("Select an automation to see its history.")
            self.query_one(
                "#scheduling-automation-detail", DefinitionDetail
            ).set_definition(None)
        else:
            self._request_automation_history(new_id)
            self._request_automation_detail(new_id)

    def _selected_automation(self) -> dict[str, Any] | None:
        """Return the highlighted automation definition, if any."""
        for definition in self._automations:
            if str(definition.get("id")) == self._selected_automation_id:
                return definition
        return None

    def _run_automation_now(self, definition: dict[str, Any]) -> None:
        """Dispatch one automation definition now, routed by its owner.

        Local rows use the existing PR-2 `SchedulingService.run_automation_
        now` seam (the same claim/spawn machinery the scheduled dispatch
        path uses); server rows keep the existing control-plane dispatch.
        Never both -- ADR-077 decision 1, one executor per owner.
        """
        from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

        if definition.get("pending_sync"):
            # Server-owned but never pushed: the server has no id for it,
            # and this side must not run a server-owned definition (§3).
            self.app_instance.notify(
                "This automation hasn't synced to the server yet — it will "
                "run there after the next successful sync.",
                severity="warning",
            )
            return
        if is_server_scoped_owner(definition.get("owner_id")):
            self._run_automation_now_server(definition)
        else:
            self._run_automation_now_local(definition)

    def _run_automation_now_local(self, definition: dict[str, Any]) -> None:
        """Dispatch a local automation through the existing run-now seam."""
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot run the automation.",
                severity="warning",
            )
            return
        definition_id = str(definition.get("id"))
        name = str(definition.get("name") or definition_id)

        async def _run() -> None:
            try:
                result = await service.run_automation_now(definition_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Local automation run-now failed for definition {}",
                    definition_id,
                )
                self.app_instance.notify(
                    f"Failed to run '{name}'.", severity="error"
                )
                return
            if result is None:
                self.app_instance.notify(
                    f"'{name}' did not run — it is missing, paused/"
                    "archived, mid-transfer, or its health is not ready "
                    "(the definition's own state shows which).",
                    severity="warning",
                )
                return
            deduped = (
                " — deduped, a run was already in flight"
                if result.get("deduped")
                else ""
            )
            self.app_instance.notify(
                f"'{name}' ran now{deduped}.", severity="information"
            )
            # redesign PR-2 Task 2 review round 2: a real local run --
            # the Queue's cached definitions rows may now be outdated.
            self._definitions_stale = True
            self._request_automations_refresh()

        self.run_worker(
            _run,
            exclusive=True,
            group="schedules-run-automation-now",
        )  # type: ignore[arg-type]

    def _run_automation_now_server(self, definition: dict[str, Any]) -> None:
        """Dispatch one server automation through the control-plane run endpoint."""
        service = self._scheduling_service
        server_client = getattr(service, "server_client", None) if service else None
        if server_client is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot run the automation.",
                severity="warning",
            )
            return
        definition_id = str(definition.get("id"))
        name = str(definition.get("name") or definition_id)

        async def _run() -> None:
            try:
                result = await server_client.run_automation_definition_now(
                    definition_id
                )
            except ServerClientValidationError as exc:
                # Lifecycle refusals (paused/archived) and policy denials
                # arrive here with the server's own reason text.
                self.app_instance.notify(
                    f"'{name}' refused: {exc}", severity="warning"
                )
                return
            except ServerClientError as exc:
                logger.opt(exception=True).warning(
                    "Server run-now failed for definition {}",
                    definition_id,
                )
                self.app_instance.notify(
                    f"Failed to run '{name}': {exc}", severity="error"
                )
                return
            deduped = (
                " — deduped, a run for this slot was already queued"
                if result.get("deduped")
                else ""
            )
            run_slot = str(result.get("run_slot_utc") or "unknown slot")
            self.app_instance.notify(
                f"'{name}' dispatched to the server (slot {run_slot}){deduped}. "
                "The result arrives as a notification.",
                severity="information",
            )
            # Same rule as the local twin (final review F11): mark
            # staleness at each genuine mutation call site. A server
            # run-now can change what the next `_load_server_automations`
            # returns (next_run_at, health, last-run state).
            self._definitions_stale = True
            # The dispatch returns when the run is ENQUEUED, not finished:
            # the terminal audit event lands only after the server executes
            # (an LLM call -- seconds). One immediate fetch catches the
            # dispatch-time events; a delayed fetch catches quick terminal
            # outcomes. Long runs stay stale until the next selection or
            # sync -- honest, and the notification still reports the result.
            self._request_automation_history(definition_id)
            self.set_timer(
                AUTOMATION_HISTORY_FOLLOWUP_SECONDS,
                lambda: self._request_automation_history(definition_id)
                if self._selected_automation_id == definition_id
                else None,
            )

        self.run_worker(
            _run,
            exclusive=True,
            group="schedules-run-automation-now",
        )  # type: ignore[arg-type]

    def _edit_selected_automation(self) -> None:
        """Open the selected automation definition for editing (e key).

        `agent_task` rows are excluded -- only `recurring_question`
        authoring exists (the same v1 scope guard `save_definition`
        itself enforces via `_reject_unsupported_family`).
        """
        definition = self._selected_automation()
        if definition is None:
            self.app_instance.notify(
                "Nothing to edit — select an automation first.",
                severity="warning",
            )
            return
        if definition.get("family") != "recurring_question":
            self.app_instance.notify(
                "Only recurring-question automations can be edited here "
                "(agent-task authoring is not yet available).",
                severity="warning",
            )
            return
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot edit the automation.",
                severity="warning",
            )
            return

        async def _open() -> None:
            definition_id = await self._resolve_local_definition_id(
                service, definition
            )
            if definition_id is None:
                self.app_instance.notify(
                    "Could not prepare this automation for editing — see "
                    "the log.",
                    severity="error",
                )
                return
            owner_id = str(definition.get("owner_id") or "local")
            options, _default = self._runs_on_options()
            if owner_id not in {value for _, value in options}:
                # Same defensive fallback as the reminder form's timezone/
                # owner selectors: the row's real owner always round-trips
                # even when it is not among the currently offered choices
                # (e.g. a server owner other than the active one).
                options = [*options, (owner_id, owner_id)]
            self.app.push_screen(
                AutomationDefinitionForm(
                    service,
                    definition_row=definition,
                    definition_id=definition_id,
                    available_owners=options,
                    default_owner=owner_id,
                ),
                callback=lambda outcome: self._on_automation_form_result(
                    outcome, was_edit=True
                ),
            )

        self.run_worker(_open, exclusive=True, group="schedules-edit-automation")

    async def _resolve_local_definition_id(
        self, service: "SchedulingService", definition: dict[str, Any]
    ) -> str | None:
        """Return the LOCAL row id `save_definition`'s `definition_id` needs.

        `save_definition`'s `definition_id` parameter is always a LOCAL
        id (Task 4's own contract). A row shown here from a pure server
        fetch may have no local shadow yet (nothing has synced or saved
        it locally before) -- editing it still needs one, so this mirrors
        it in place via the same `upsert_automation_definitions_from_
        server` the sync pull and `save_definition`'s own online-create
        path already use, rather than inventing a second mirroring path.
        """
        from tldw_chatbook.Scheduling.scheduler.queue import is_server_scoped_owner

        owner_id = str(definition.get("owner_id") or "local")
        if definition.get("pending_sync") or not is_server_scoped_owner(owner_id):
            # A `pending_sync` row IS a local row (server-owned, never
            # synced): its `id` is already the local one, and treating it
            # as a server id here would mirror it back as a SECOND row
            # keyed by that uuid.
            local_id = definition.get("id")
            return str(local_id) if local_id else None

        server_id = str(definition.get("id"))
        existing = await asyncio.to_thread(
            service.db.get_automation_definition_by_server_id, owner_id, server_id
        )
        if existing is not None:
            return existing.get("id")
        try:
            await asyncio.to_thread(
                service.db.upsert_automation_definitions_from_server,
                owner_id,
                [definition],
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to mirror server automation {} before editing",
                server_id,
            )
            return None
        mirrored = await asyncio.to_thread(
            service.db.get_automation_definition_by_server_id, owner_id, server_id
        )
        return mirrored.get("id") if mirrored else None

    # -- Automations-tab transfer actions (schedules-handoff spec §6,
    # PR-5 task 7 fix round item 1) -----------------------------------
    #
    # With PR-5 as first shipped, `begin_transfer_to_server`/`begin_
    # transfer_to_local`/`cancel_transfer` had no UI call site at all for
    # `table_kind="automation_definition"` -- the facade fully supports
    # it (Task 6), but nothing let a user ever start, cancel, or retry a
    # definition's transfer. The tab has no per-row detail widget (no
    # `TaskDetail` equivalent), so these are keybindings routed by active
    # tab -- the SAME idiom `action_run_task_now`/`action_edit_task`
    # already use for this tab's Run-now/Edit, not new buttons. A
    # refusal renders inline in the tab's own notice Static (its only
    # existing status affordance -- reused rather than adding a second
    # one); an allowed Move reuses the same `ConfirmationDialog` +
    # honest-toast shape the Queue tab's reminder flow already
    # established. Move to server (`M`) and Retry (`y`) share one call:
    # a retry IS a re-begin, so the difference is the label and which
    # state each is offered in, not the code path.

    def action_move_automation_to_local(self) -> None:
        """m key: queue a server-owned automation mirror to move here
        (spec §6.2), Automations-tab only."""
        self._begin_automation_transfer("to_local")

    def action_move_automation_to_server(self) -> None:
        """M key: queue a LOCAL automation definition to move to the
        server (spec §6.1), Automations-tab only.

        The missing half of the definitions transfer UI (final review
        M8): `y`/Retry already reached `begin_transfer_to_server`, but
        only ever as a retry beside a failed transfer, so a plain local
        definition had no way to start one. Same facade call -- a retry
        IS a re-begin (`transfer_refusal` excludes `to_server_failed`
        from "in progress", and the CAS accepts both `None` and
        `to_server_failed`) -- given its own honest label and key, rather
        than teaching users that "Retry" means "Move".
        """
        self._begin_automation_transfer("to_server")

    def action_retry_automation_transfer(self) -> None:
        """y key: retry a definitively-failed local -> server automation
        transfer (spec §6.1.5), Automations-tab only."""
        self._begin_automation_transfer("to_server")

    def action_cancel_automation_transfer(self) -> None:
        """k key: cancel the selected automation's in-progress transfer
        (spec §6.3), Automations-tab only."""
        self._cancel_automation_transfer()

    def _is_automations_tab_active(self) -> bool:
        try:
            return (
                self.query_one("#scheduling-tabs", TabbedContent).active
                == "scheduling-automations-tab"
            )
        except Exception:  # noqa: BLE001
            return False

    def _show_automations_inline_reason(self, reason: str) -> None:
        """Surface a transfer refusal/failure inline (fix round item 1's
        "refusal -> inline reason" flow) -- the Automations pane's notice
        Static is the tab's only existing status affordance
        (`_automations_notice_text`'s own home), reused rather than
        adding a second one. The next `load_automations()` (any other
        action, or a completed transfer's own reload) naturally replaces
        it with the aggregate summary again.
        """
        try:
            notice = self.query_one("#scheduling-automations-notice", Static)
        except Exception:  # noqa: BLE001
            return
        self._update_static_content(notice, reason)

    def _begin_automation_transfer(self, direction: str) -> None:
        """Move-to-local / Retry for the selected automation (spec
        §6.1/§6.2), Automations-tab only. Same flow as the Queue tab's
        `_begin_transfer`: refusal -> inline reason; allowed ->
        `ConfirmationDialog` listing `transfer_warnings`; honest toast on
        completion; reload via the tab's existing `_request_automations_
        refresh` wiring.
        """
        if not self._is_automations_tab_active():
            self.app_instance.notify(
                "Switch to the Automations tab to move or retry an "
                "automation's transfer.",
                severity="warning",
            )
            return
        definition = self._selected_automation()
        if definition is None:
            self.app_instance.notify(
                "Nothing selected — select an automation first.",
                severity="warning",
            )
            return
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot start a transfer.",
                severity="warning",
            )
            return

        async def _resolve_and_begin() -> None:
            # redesign PR-2 Task 2 review round 2: a genuine user-
            # triggered transfer attempt (m/M/y) -- the Queue's cached
            # definitions rows may now be outdated regardless of which
            # branch below this lands on (refused/pending/not_found all
            # still touch the local mirror row via `_resolve_local_
            # definition_id`).
            self._definitions_stale = True
            local_id = await self._resolve_local_definition_id(service, definition)
            if local_id is None:
                self.app_instance.notify(
                    "Could not prepare this automation for transfer — see "
                    "the log.",
                    severity="error",
                )
                return
            row = await asyncio.to_thread(
                service.db.get_automation_definition, local_id
            )
            if row is None:
                self.app_instance.notify(
                    "This automation no longer exists.", severity="warning"
                )
                self._request_automations_refresh()
                return
            name = str(row.get("name") or definition.get("name") or local_id)
            reason = service.transfer_refusal(row, direction)
            if reason is not None:
                self._show_automations_inline_reason(reason)
                return
            warnings = service.transfer_warnings(row, direction)
            dialog = self._transfer_confirm_dialog(name, direction, warnings)
            confirmed = await self.app.push_screen_wait(dialog)
            if not confirmed:
                return
            try:
                if direction == "to_server":
                    outcome = await service.begin_transfer_to_server(
                        "automation_definition", local_id
                    )
                else:
                    outcome = await service.begin_transfer_to_local(
                        "automation_definition", local_id
                    )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to begin automation transfer for {}", local_id
                )
                self.app_instance.notify(
                    f"Failed to start the transfer for '{name}'.",
                    severity="error",
                )
                self._request_automations_refresh()
                return
            if outcome.status == "pending":
                self.app_instance.notify(
                    self._transfer_pending_toast_text(name, direction),
                    severity="information",
                )
            elif outcome.status == "refused":
                self._show_automations_inline_reason(
                    outcome.reason or f"Could not move '{name}'."
                )
            elif outcome.status == "not_found":
                self.app_instance.notify(
                    f"'{name}' no longer exists.", severity="warning"
                )
            self._request_automations_refresh()

        self.run_worker(
            _resolve_and_begin,
            exclusive=True,
            group="schedules-automation-transfer",
        )  # type: ignore[arg-type]

    def _cancel_automation_transfer(self) -> None:
        """Cancel the selected automation's in-progress transfer (spec
        §6.3), Automations-tab only. No confirm dialog -- same rationale
        as the Queue tab's cancel: it is the escape hatch, gating it
        behind its own confirmation fights the point. The resolved LOCAL
        row id is already the dormant copy's own id for a release in
        progress -- `_load_local_automations` shows that copy directly (a
        `server_id`-less local row), never through the mirror, so
        resolution never routes to the mirror's id for that leg (same
        construction as the reminder side's cancel).
        """
        if not self._is_automations_tab_active():
            self.app_instance.notify(
                "Switch to the Automations tab to cancel an automation's "
                "transfer.",
                severity="warning",
            )
            return
        definition = self._selected_automation()
        if definition is None:
            self.app_instance.notify(
                "Nothing selected — select an automation first.",
                severity="warning",
            )
            return
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot cancel the "
                "transfer.",
                severity="warning",
            )
            return

        async def _resolve_and_cancel() -> None:
            # redesign PR-2 Task 2 review round 2: a genuine user-
            # triggered cancel (k) -- see `_resolve_and_begin`'s matching
            # comment for why this is marked unconditionally.
            self._definitions_stale = True
            local_id = await self._resolve_local_definition_id(service, definition)
            if local_id is None:
                self.app_instance.notify(
                    "Could not resolve this automation locally — see the "
                    "log.",
                    severity="error",
                )
                return
            name = str(definition.get("name") or local_id)
            try:
                outcome = await service.cancel_transfer(
                    "automation_definition", local_id
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to cancel automation transfer for {}", local_id
                )
                self.app_instance.notify(
                    f"Failed to cancel the transfer for '{name}'.",
                    severity="error",
                )
                self._request_automations_refresh()
                return
            if outcome.status == "cancelled":
                self.app_instance.notify(
                    _cancel_toast_text(name), severity="information"
                )
            else:
                self._show_automations_inline_reason(
                    outcome.reason
                    or f"Could not cancel the transfer for '{name}'."
                )
            self._request_automations_refresh()

        self.run_worker(
            _resolve_and_cancel,
            exclusive=True,
            group="schedules-automation-transfer",
        )  # type: ignore[arg-type]

    # -- Results-tab actions (schedules-handoff PR-6 task 3) ---------------
    #
    # Read/dismiss reuse r/d via action_run_task_now/action_delete's own
    # tab routing above. Mark-solved/Mark-all-read get fresh keys (o/a),
    # guarded the same way m/M/y/k refuse off the Automations tab.

    def _is_results_tab_active(self) -> bool:
        try:
            return (
                self.query_one("#scheduling-tabs", TabbedContent).active
                == "scheduling-results-tab"
            )
        except Exception:  # noqa: BLE001
            return False

    def _review_selected_result(self, review_state: str) -> None:
        """r/d on the Results tab: read/dismiss the selected result.

        `SchedulingService.review_automation_result` writes the local row
        and, for a server mirror, queues the PR-3 pushback mutation in the
        SAME DB transaction -- nothing extra to do here for that half.
        """
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable.", severity="warning"
            )
            return
        results_tab = self.query_one("#scheduling-results", ResultsTab)
        result = results_tab.selected_result()
        if result is None:
            self.app_instance.notify("Select a result first.", severity="warning")
            return

        async def _review() -> None:
            updated = await service.review_automation_result(
                result["id"], review_state
            )
            if not updated:
                self.app_instance.notify(
                    "Could not update this result — see the log.",
                    severity="error",
                )
            self._refresh_results_surfaces()

        self.run_worker(
            _review, exclusive=True, group="schedules-review-result"
        )  # type: ignore[arg-type]

    def action_mark_result_solved(self) -> None:
        """o key: mark the selected finding's definition solved (Task 2's
        facade), Results-tab only. Refused client-side for a row `solved_
        eligibility` already rules out (wrong kind, already solved, or an
        unknown definition); a still-eligible row can still be refused by
        the facade itself (transfer lock, offline+server-owned -- UX-073),
        surfaced from `ResolveOutcome.reason`.
        """
        if not self._is_results_tab_active():
            self.app_instance.notify(
                "Switch to the Results tab to mark a result solved.",
                severity="warning",
            )
            return
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable.", severity="warning"
            )
            return
        results_tab = self.query_one("#scheduling-results", ResultsTab)
        result = results_tab.selected_result()
        if result is None:
            self.app_instance.notify("Select a result first.", severity="warning")
            return
        eligible, reason = solved_eligibility(result, results_tab.definitions_by_id)
        if not eligible:
            self.app_instance.notify(
                reason or "This result cannot be marked solved.",
                severity="warning",
            )
            return
        # `resolve_definition` takes a LOCAL definition id (its own
        # contract), but a synced result's `definition_id` is the SERVER's
        # id -- passing it through unresolved refused with "Automation
        # definition <server id> was not found" on exactly the rows the
        # action exists for (live verification task 6, D3). The gate above
        # already resolved the row across both id spaces; reuse it rather
        # than re-deriving the id a second, divergent way. Non-None here
        # by construction: `solved_eligibility` returns ineligible when
        # the same lookup misses.
        definition = definition_for_result(result, results_tab.definitions_by_id)
        local_definition_id = str((definition or {}).get("id") or "")

        async def _mark_solved() -> None:
            outcome = await service.resolve_definition(
                local_definition_id, solved=True, result_id=result["id"]
            )
            if outcome.status == "saved":
                self.app_instance.notify("Marked solved.", severity="information")
            else:
                self.app_instance.notify(
                    outcome.reason or "Could not mark this result solved.",
                    severity="warning",
                )
            self._refresh_results_surfaces()

        self.run_worker(
            _mark_solved, exclusive=True, group="schedules-mark-solved"
        )  # type: ignore[arg-type]

    def _unread_result_ids(self, service: "SchedulingService") -> list[str]:
        """Every unread result id across the FULL table, read straight
        from the DB -- not the Results tab's own listing, which is capped
        at `RESULTS_INBOX_LIMIT` (200) rows. The rail button's visibility
        already sums the full-table unread count (`_refresh_results_tab`'s
        `count_unread_results`); mark-all-read has to clear that same set
        or an unread row older than the loaded window survives while the
        button hides itself as if nothing were left (Qodo HIGH).
        """
        unread_total = service.db.count_unread_results(owner_id=None)
        if not unread_total:
            return []
        results = service.db.list_automation_results(
            owner_id=None, review_state="unread", limit=unread_total
        )
        return [result["id"] for result in results]

    async def _dispatch_mark_all_results_read(
        self, service: "SchedulingService", unread_ids: list[str]
    ) -> None:
        """Per-row `review_automation_result` fan-out for a batch of
        result ids -- there is no bulk DB primitive for this (spec's
        documented fan-out, mirroring `_on_bulk_delete_confirmed`'s
        loop-and-count shape). Shared by the Results tab's `a` keybinding
        and the rail's `Mark all read` button (redesign PR-2, Task 3) --
        the Queue refresh included, which used to sit in the rail
        button's own wrapper as if it were a rail-specific nicety (final
        review F5): pressing `a` on the Results tab left the Queue's
        unread dots painted and its rail button visible, and pressing
        that button then reported "Nothing unread."
        """
        errors = 0
        for result_id in unread_ids:
            if not await service.review_automation_result(result_id, "read"):
                errors += 1
        count = len(unread_ids) - errors
        self.app_instance.notify(
            f"Marked {count} result{'s' if count != 1 else ''} read"
            + (f" ({errors} failed)" if errors else "")
            + ".",
            severity="information" if not errors else "warning",
        )
        self._refresh_results_surfaces()

    def action_mark_all_results_read(self) -> None:
        """a key: mark every currently-loaded unread result read,
        Results-tab only (the rail's `Mark all read` button reaches the
        same fan-out from the Queue tab, see `_rail_mark_all_read`).
        """
        if not self._is_results_tab_active():
            self.app_instance.notify(
                "Switch to the Results tab to mark all results read.",
                severity="warning",
            )
            return
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable.", severity="warning"
            )
            return
        unread_ids = self._unread_result_ids(service)
        if not unread_ids:
            self.app_instance.notify("Nothing unread.", severity="information")
            return

        async def _mark_all() -> None:
            await self._dispatch_mark_all_results_read(service, unread_ids)

        self.run_worker(
            _mark_all, exclusive=True, group="schedules-mark-all-read"
        )  # type: ignore[arg-type]

    def _rail_mark_all_read(self) -> None:
        """Rail `Mark all read` (redesign PR-2, Task 3): reuses the exact
        same per-row fan-out as the `a` keybinding above, but reachable
        without switching to the Results tab first -- the whole point of
        a rail-level affordance for it. The Queue refresh that used to
        live here moved INTO that shared fan-out (final review F5): it
        was never rail-specific.
        """
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable.", severity="warning"
            )
            return
        unread_ids = self._unread_result_ids(service)
        if not unread_ids:
            self.app_instance.notify("Nothing unread.", severity="information")
            return

        async def _mark_all() -> None:
            await self._dispatch_mark_all_results_read(service, unread_ids)

        self.run_worker(
            _mark_all, exclusive=True, group="schedules-mark-all-read"
        )  # type: ignore[arg-type]

    def _set_reminder_enabled(self, task: ReminderTask, enabled: bool) -> None:
        """Update a reminder's enabled state and refresh the queue."""
        service = self._scheduling_service
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot update the scheduled task.",
                severity="warning",
            )
            return

        async def _update_and_refresh() -> None:
            try:
                # The ROW's owner, never the service's active one (final
                # review F4): this list spans owners since Task 1, so
                # toggling a `server:` row while "This device" is active
                # used to write the local mirror with NO pending mutation
                # -- the server kept firing it and the next pull undid the
                # toggle. Same `owner_id=` thread PR-5 established for
                # `create_reminder`.
                await service.update_reminder(
                    task.id, {"enabled": enabled}, owner_id=task.owner_id
                )
                status = "enabled" if enabled else "disabled"
                self.app_instance.notify(
                    f"'{task.title}' {status}.", severity="information"
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to update reminder enabled state")
                self.app_instance.notify(
                    f"Failed to update '{task.title}'.",
                    severity="error",
                )
            self._request_tasks_refresh(refresh_definitions=False)

        self.run_worker(
            _update_and_refresh,
            exclusive=True,
            group="schedules-set-reminder-enabled",
        )  # type: ignore[arg-type]

    def _refresh_owner_select(self) -> None:
        status = self.query_one("#scheduling-sync-status", SyncStatusWidget)
        service = self._service()
        if service is None:
            status.set_owner_state("local", None, False)
            status.update_status(None, None, [])
            self._sync_header_status("blocked", "Scheduling unavailable")
            return
        active_server_id = self._active_server_id()
        server_available = self._server_available(service, active_server_id)
        status.set_owner_state(service.owner_id, active_server_id, server_available)
        state = service.db.get_sync_state(service.owner_id) or {}
        sync_errors = state.get("sync_errors") or []
        # A runtime-mode refusal is "sync not applicable", never a failure.
        # New refusals are no longer recorded (task-2722, SyncEngine), but
        # profiles that synced on older builds still carry persisted ones —
        # keep them off the error surface instead of badging local-only
        # profiles with an error the user did nothing to cause.
        sync_errors = [
            entry
            for entry in sync_errors
            if "requires server mode" not in str(entry.get("message", ""))
        ]
        status.update_status(
            last_pull_at=state.get("last_pull_at"),
            last_push_at=state.get("last_push_at"),
            sync_errors=sync_errors,
        )
        if sync_errors:
            count = len(sync_errors)
            self._sync_header_status(
                "error", f"{count} sync error{'s' if count != 1 else ''}"
            )
        elif not server_available:
            self._sync_header_status("empty", "Local only — no server connection")
        elif service.owner_id.startswith("server:"):
            self._sync_header_status("ready", "Synced with server")
        else:
            self._sync_header_status("ready", "Local schedules")

    def _sync_header_status(self, status: WorkbenchStatus, label: str) -> None:
        """Reflect real sync health in the destination header chip."""
        try:
            header = self.query_one("#schedules-destination-header", DestinationHeader)
        except Exception:  # noqa: BLE001 - header not mounted yet
            return
        header.sync_state(
            WorkbenchHeaderState(
                title="Schedules",
                subtitle="When jobs, watchlists, and workflows run.",
                status=status,
                status_label=label,
            )
        )

    def on_resize(self) -> None:
        """Hide side panes (with a notice) instead of clipping them."""
        self._sync_responsive_workbench()
        try:
            width = self.size.width
            inspector = self.query_one("#scheduling-inspector-pane")
            detail = self.query_one("#scheduling-detail-pane")
        except Exception:  # noqa: BLE001 - panes not mounted yet
            return
        hide_inspector = 0 < width < 118
        hide_detail = 0 < width < 84
        inspector.set_class(hide_inspector, "pane-hidden")
        detail.set_class(hide_detail, "pane-hidden")
        # At detail-hiding widths the pane chrome also gets too tall to fit:
        # the Queue tab label already names this pane, so the in-pane title
        # yields its row to the table + notice (see _scheduling.tcss).
        self.query_one("#scheduling-workbench").set_class(hide_detail, "compact")
        if hide_detail:
            # The create CTA normally lives in the (now hidden) detail pane;
            # keep it reachable at compact widths when the queue is empty.
            base = "Detail and inspector hidden — widen the window to see them."
            if not self._tasks:
                base += " Press c to schedule your first task."
            self._resize_notice = base
        elif hide_inspector:
            self._resize_notice = "Inspector hidden — widen the window to see it."
        else:
            self._resize_notice = ""
        self._update_pane_notice()
        # schedules-redesign PR-1, Task 4: the Automations tab's detail
        # pane hides at the same width the Queue tab's own detail pane
        # does -- same mechanism (`pane-hidden`), separate try/except so a
        # missing pane here (e.g. before the Automations TabPane mounts)
        # never short-circuits the Queue-pane handling above.
        try:
            automations_detail = self.query_one("#scheduling-automations-detail-pane")
        except Exception:  # noqa: BLE001 - pane not mounted yet
            return
        automations_detail.set_class(hide_detail, "pane-hidden")

    def _update_pane_notice(self) -> None:
        """Compose the queue-pane notice: hidden panes, marks, glyph legend.

        task-23107: while rows are marked, visible text states the count,
        the keys that act on all marked rows, and how to clear the marks;
        the ◇ missed-while-away glyph gets an on-screen explanation
        whenever a visible row carries it.
        """
        try:
            notice = self.query_one("#scheduling-pane-notice", Static)
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        parts: list[str] = []
        if self._resize_notice:
            parts.append(self._resize_notice)
        # Marking is reminder-only and marks are pruned on load, so the
        # legend count IS the count the bulk verbs act on (review F1).
        marked_count = len(self._marked_reminder_tasks())
        if marked_count:
            visible_ids = {task.id for task in self._visible_tasks}
            hidden = sum(
                1 for task_id in self._marked_ids if task_id not in visible_ids
            )
            hidden_note = f" ({hidden} hidden by the filter)" if hidden else ""
            parts.append(
                f"{marked_count} marked{hidden_note} — space toggles all "
                "· d deletes all · esc clears"
            )
        if any(_was_missed_while_away(task) for task in self._visible_tasks):
            parts.append("◇ = ran late (dispatched after its scheduled time)")
        self._update_static_content(notice, "\n".join(parts))

    @on(Button.Pressed, "#scheduling-owner-local")
    def _on_owner_local(self) -> None:
        self._set_owner("local")

    @on(Button.Pressed, "#scheduling-owner-server")
    def _on_owner_server(self) -> None:
        service = self._service()
        if service is None:
            return
        active_server_id = self._active_server_id()
        if not self._server_available(service, active_server_id):
            self.app_instance.notify("No server connection", severity="warning")
            return
        self._set_owner(f"server:{active_server_id}")

    def _set_owner(self, new_owner: str) -> None:
        service = self._service()
        if service is None:
            return
        service.set_owner(new_owner)
        runtime_source = "server" if new_owner.startswith("server:") else "local"
        set_authoritative_runtime_source(
            self.app_instance.runtime_policy,
            runtime_source,
            app_config=self.app_instance.app_config,
        )
        self._refresh_owner_select()
        self._request_tasks_refresh(refresh_definitions=False)
        self._refresh_conflicts_tab()

    @on(Button.Pressed, "#scheduling-clear-error")
    def _on_clear_sync_errors(self) -> None:
        service = self._service()
        if service is None:
            return
        service.db.update_sync_state(service.owner_id, sync_errors=[])
        self._refresh_owner_select()

    @on(SyncCompleted)
    def _on_sync_completed(self, event: SyncCompleted) -> None:
        self._sync_running = False
        outcome = event.outcome
        status = getattr(outcome, "status", None)
        pulled = int(getattr(outcome, "pulled", 0) or 0)
        pushed = int(getattr(outcome, "pushed", 0) or 0)
        if outcome is None:
            # Legacy sender without an outcome.
            message = "Sync completed."
        elif status == "not_applicable":
            message = (
                "Sync skipped — not applicable in this mode; nothing was "
                "pulled or pushed."
            )
        elif pulled or pushed:
            message = f"Sync completed — pulled {pulled}, pushed {pushed}."
        else:
            message = "Sync finished — nothing to pull or push."
        self.app_instance.notify(message, severity="information")
        self._refresh_owner_select()
        self._request_tasks_refresh()
        self._request_automations_refresh()
        self._refresh_conflicts_tab()
        self._refresh_results_tab()

    @on(SyncFailed)
    def _on_sync_failed(self, event: SyncFailed) -> None:
        self._sync_running = False
        self.app_instance.notify(f"Sync failed: {event.error}", severity="error")
        self._refresh_owner_select()
        self._request_tasks_refresh()
        self._request_automations_refresh()
        self._refresh_conflicts_tab()
        self._refresh_results_tab()

    @on(ConflictsTab.ConflictResolved)
    def _on_conflict_resolved(self, event: ConflictsTab.ConflictResolved) -> None:
        self._request_tasks_refresh(refresh_definitions=False)
        self._refresh_conflicts_tab()

    def _refresh_conflicts_tab(self) -> None:
        service = self._service()
        if service is None:
            return
        conflicts_tab = self.query_one("#scheduling-conflicts", ConflictsTab)
        conflicts = service.db.get_conflicts(
            service.owner_id, primitive="reminder_task"
        )
        conflicts_tab.populate(conflicts)
        label = f"Conflicts ({len(conflicts)})" if conflicts else "Conflicts"
        # Surface the conflict count on the tab label itself (UX-063).
        self._set_tab_label("scheduling-conflicts-tab", label)
        # redesign PR-2, Task 3: same count, mirrored onto the status
        # strip's badge (plan ruling 4) -- no new query.
        try:
            self.query_one("#scheduling-conflicts-badge", Button).label = label
        except Exception:  # noqa: BLE001 - strip not mounted yet
            pass

    def _set_tab_label(self, pane_id: str, label: str) -> None:
        """Relabel one tab in the workbench's `TabbedContent`.

        `TabPane.label` does not exist in Textual 8.x -- a pane stores its
        title in `_title` and exposes no `label` reactive, so the previous
        `pane.label = ...` assignment silently created an inert Python
        attribute and the rendered tab text never changed (live
        verification task 6, D2: both the Results unread badge and the
        Conflicts badge it was copied from were no-ops on screen while
        their tests passed by reading the attribute back). The real seam
        is `TabbedContent.get_tab(pane_id)` -> the `Tab` widget's own
        `label` setter, which calls `Tab.update` and repaints.
        """
        try:
            tab = self.query_one("#scheduling-tabs", TabbedContent).get_tab(pane_id)
        except Exception:  # noqa: BLE001 - tabs/pane not mounted yet
            return
        tab.label = label

    def _refresh_results_tab(self) -> None:
        """Reload the Results tab and its unread badge (schedules-handoff
        PR-6 task 3). Mirrors `_refresh_conflicts_tab`'s shape: direct
        `service.db.*` calls (list_automation_results/count_unread_
        results span every owner -- Task 1), no worker -- this is a local
        DB-only read, same cost class as `get_conflicts`. Also called
        after Task 4's notification-triggered pull and after every
        read/dismiss/mark-solved/mark-all-read action below.

        Lists `RESULTS_INBOX_LIMIT` rows, not the DB's own default 50: the
        badge counts EVERY unread result, so a 50-row listing made the two
        numbers disagree and quietly hid the rest. `total` is passed so the
        tab can say "showing newest N of M" once the cap bites.
        """
        service = self._service()
        if service is None:
            return
        results_tab = self.query_one("#scheduling-results", ResultsTab)
        results = service.db.list_automation_results(
            owner_id=None, limit=RESULTS_INBOX_LIMIT
        )
        unread = service.db.count_unread_results(owner_id=None)
        total = service.db.count_automation_results(owner_id=None)
        definitions_by_id = index_definitions_by_id(
            service.db.list_automation_definitions(owner_id=None)
        )
        results_tab.populate(results, definitions_by_id, total=total)
        # Surface the unread count on the tab label itself (spec §4's
        # inbox badge, same UX-063 idiom as the Conflicts tab above).
        self._set_tab_label(
            "scheduling-results-tab",
            f"Results ({unread})" if unread else "Results",
        )

    def _refresh_results_surfaces(self) -> None:
        """Every surface a results MUTATION moves: the Results tab and the
        Queue's own unread dots + rail `Mark all read` visibility.

        Final review F5: the unread affordances Task 3 added to the Queue
        derive from `UnifiedRow.unread_count`, which is only recomputed by
        `load_tasks` -- so an SSE-triggered pull, a read/dismiss, a
        mark-solved or a mark-all-read updated the Results tab and left
        the Queue's dots (and the rail button, gated on
        `sum(row.unread_count) > 0`) stale in both directions: hidden
        while unread work existed, or visible with nothing left to mark.
        `refresh_definitions=False` -- results never change which
        definitions exist, and results are re-read on every load anyway.
        Called from the mutation paths only, never from the mount/reload
        paths that already run their own `_request_tasks_refresh`.
        """
        self._refresh_results_tab()
        self._request_tasks_refresh(refresh_definitions=False)

    def action_delete(self) -> None:
        """Delete marked tasks in bulk, else the selected one (confirmed).

        While ANY mark exists, d never falls through to the highlighted,
        unmarked row (task-23107 review F1): acting on a row the user
        never marked is worse than refusing. On the Results tab, ``d``
        dismisses the selected result instead (schedules-handoff PR-6
        task 3) -- same key, the tab-appropriate "remove from view" verb.
        """
        try:
            active_pane = self.query_one("#scheduling-tabs", TabbedContent).active
        except Exception:  # noqa: BLE001
            active_pane = None
        if active_pane == "scheduling-results-tab":
            self._review_selected_result("dismissed")
            return
        if self._marked_ids:
            marked = self._marked_reminder_tasks()
            if not marked:
                # Defensive: marking is reminder-only and marks are pruned
                # on every load, so this means the marked rows vanished
                # between renders. Refuse instead of falling through.
                self._marked_ids.clear()
                self._render_table()
                self.app_instance.notify(
                    "The marked rows are no longer in the queue — marks "
                    "cleared; nothing was deleted.",
                    severity="warning",
                )
                return
            from ....Widgets.delete_confirmation_dialog import (
                DeleteConfirmationDialog,
            )

            self.app.push_screen(
                DeleteConfirmationDialog(
                    item_type="Scheduled tasks",
                    item_name=f"{len(marked)} marked tasks",
                    permanent=True,
                ),
                callback=lambda confirmed: self._on_bulk_delete_confirmed(
                    confirmed, marked
                ),
            )
            return
        if not self._tasks:
            self.app_instance.notify(
                "Nothing to delete — the queue is empty.",
                severity="warning",
            )
            return
        task = self._selected_task()
        if task is None:
            # `TaskDetail.request_delete` deletes the task the pane last
            # HELD, and the definition branch of `_update_detail_for_
            # index` only hides that pane -- it never clears it. Pressing
            # `d` on a definition row therefore opened a confirmation for
            # whichever reminder was highlighted before it (final review
            # F8's silent-refusal finding, in its sharpest form).
            self.app_instance.notify(
                self._no_task_notice("delete"), severity="warning"
            )
            return
        if self._refuse_if_transfer_locked(task, "delete this task"):
            return
        self.query_one("#scheduling-task-detail", TaskDetail).request_delete()

    def _on_bulk_delete_confirmed(self, confirmed, marked: list[ReminderTask]) -> None:
        """Delete all marked tasks after the confirmation dialog."""
        if not confirmed:
            return
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot delete tasks.",
                severity="warning",
            )
            return

        async def _bulk_delete() -> None:
            errors = 0
            for task in marked:
                try:
                    # A False return is a refusal, not a crash (e.g. the
                    # transfer read-only guard, final review I7) -- count
                    # it, so the toast never claims a delete that the
                    # facade declined. `owner_id` is the ROW's own (final
                    # review F4), same as the single-row delete.
                    if not await service.delete_reminder(
                        task.id, owner_id=task.owner_id
                    ):
                        errors += 1
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to delete reminder {}", task.id)
                    errors += 1
            count = len(marked) - errors
            self.app_instance.notify(
                f"Deleted {count} marked task{'s' if count != 1 else ''}"
                + (f" ({errors} failed)" if errors else "")
                + ".",
                severity="information" if not errors else "warning",
            )
            self._marked_ids.clear()
            self._request_tasks_refresh(refresh_definitions=False)

        self.run_worker(
            _bulk_delete,
            exclusive=True,
            group="schedules-bulk-delete",
        )  # type: ignore[arg-type]

    def _selected_task(self) -> ReminderTask | None:
        """Return the reminder under the queue cursor, if any.

        redesign PR-2, Task 2: routes through `self._visible_rows` (the
        table's own 1:1 row index) rather than `self._visible_tasks`
        directly -- the two diverge whenever a definition row precedes
        the cursor. A definition row under the cursor returns ``None``,
        the same "nothing to act on" result every caller (`action_edit_
        task`/`action_mark_task`/`action_toggle_enabled`/`action_delete`)
        already handles -- definition rows expose no actions in this PR.
        """
        if not self._visible_rows:
            return None
        table = self.query_one("#scheduling-task-table", DataTable)
        row_index = table.cursor_row
        if row_index is None or not (0 <= row_index < len(self._visible_rows)):
            return None
        row = self._visible_rows[row_index]
        if row.kind != "reminder":
            return None
        return row.source_row

    def _no_task_notice(self, verb: str) -> str:
        """Copy for a reminder verb that found no reminder to act on.

        A definition row IS selected and highlighted when the cursor sits
        on one, so "select a task first" reads as a bug (final review
        F8). Say where the action lives instead -- the same "managed
        elsewhere" vocabulary `_managed_elsewhere_notice` established for
        projection rows. Definition rows stay action-free here per plan
        ruling 1; this only fixes how the refusal is REPORTED.
        """
        if (self._selected_row_id or "").startswith("definition:"):
            return (
                "This automation is managed on the Automations tab for "
                f"now — {verb} it there."
            )
        return f"Nothing to {verb} — select a task first."

    def action_edit_task(self) -> None:
        """Open the highlighted task/definition in its edit form (e key).

        Routes by active tab, same shape as `action_run_task_now`: the
        Automations tab's `e` opens `AutomationDefinitionForm` pre-filled
        for a `recurring_question` row (either owner); everywhere else it
        is the existing reminder edit flow.
        """
        try:
            active_pane = self.query_one("#scheduling-tabs", TabbedContent).active
        except Exception:  # noqa: BLE001
            active_pane = None
        if active_pane == "scheduling-automations-tab":
            self._edit_selected_automation()
            return
        task = self._selected_task()
        if task is None:
            self.app_instance.notify(
                self._no_task_notice("edit"),
                severity="warning",
            )
            return
        if not isinstance(task, ReminderTask):
            # task-23106: say who owns the row instead of exposing the
            # internal reminder/projection split.
            self.app_instance.notify(
                _managed_elsewhere_notice(task, verb="edit"),
                severity="warning",
            )
            return
        if self._refuse_if_transfer_locked(task, "edit this task"):
            return
        self.post_message(EditTaskRequested(task))

    def action_mark_task(self) -> None:
        """Mark/unmark the highlighted task for bulk actions (x key).

        Only rows the bulk verbs can act on are markable (task-23107
        review F1): marking a read-only projection row would either be
        silently ignored by the bulk actions or, worse, let them fall
        through to an unmarked row.
        """
        task = self._selected_task()
        if task is None:
            self.app_instance.notify(
                self._no_task_notice("mark"),
                severity="warning",
            )
            return
        if not isinstance(task, ReminderTask):
            self.app_instance.notify(
                _managed_elsewhere_notice(task, verb="manage"),
                severity="warning",
            )
            return
        if task.id in self._marked_ids:
            self._marked_ids.discard(task.id)
        else:
            self._marked_ids.add(task.id)
        self._render_table()

    def action_clear_marks(self) -> None:
        """Clear all bulk marks (escape key)."""
        if self._marked_ids:
            self._marked_ids.clear()
            self._render_table()

    def _marked_reminder_tasks(self) -> list[ReminderTask]:
        """Marked tasks that support bulk operations."""
        return [
            task
            for task in self._tasks
            if task.id in self._marked_ids and isinstance(task, ReminderTask)
        ]

    def action_toggle_enabled(self) -> None:
        """Enable/disable marked tasks in bulk, else the highlighted one.

        While ANY mark exists, space never falls through to the
        highlighted, unmarked row (task-23107 review F1).
        """
        if self._marked_ids:
            marked = self._marked_reminder_tasks()
            if not marked:
                self._marked_ids.clear()
                self._render_table()
                self.app_instance.notify(
                    "The marked rows are no longer in the queue — marks "
                    "cleared; nothing was toggled.",
                    severity="warning",
                )
                return
            self._bulk_toggle_marked(marked)
            return

        task = self._selected_task()
        if task is None:
            self.app_instance.notify(
                self._no_task_notice("enable or disable"),
                severity="warning",
            )
            return
        if not isinstance(task, ReminderTask):
            # task-23106: say who owns the row instead of exposing the
            # internal reminder/projection split.
            self.app_instance.notify(
                _managed_elsewhere_notice(task, verb="enable or disable"),
                severity="warning",
            )
            return
        if self._refuse_if_transfer_locked(task, "enable or disable this task"):
            return
        self._set_reminder_enabled(task, not task.enabled)

    def _bulk_toggle_marked(self, marked: list[ReminderTask]) -> None:
        """Toggle every marked task's enabled state (space with marks)."""
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot update the scheduled tasks.",
                severity="warning",
            )
            return

        async def _bulk_toggle() -> None:
            errors = 0
            for task in marked:
                try:
                    # A None return is a refusal, not a crash -- same
                    # reasoning as the bulk delete above. `owner_id` is
                    # the ROW's own (final review F4), same as the
                    # single-row toggle.
                    if await service.update_reminder(
                        task.id, {"enabled": not task.enabled}, owner_id=task.owner_id
                    ) is None:
                        errors += 1
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to toggle reminder {}", task.id)
                    errors += 1
            count = len(marked) - errors
            self.app_instance.notify(
                f"Toggled {count} marked task{'s' if count != 1 else ''}"
                + (f" ({errors} failed)" if errors else "")
                + ".",
                severity="information" if not errors else "warning",
            )
            self._marked_ids.clear()
            self._request_tasks_refresh(refresh_definitions=False)

        self.run_worker(
            _bulk_toggle,
            exclusive=True,
            group="schedules-bulk-toggle",
        )  # type: ignore[arg-type]

    def action_sync_now(self) -> None:
        """Sync schedule state now."""
        if self._sync_running:
            self.app_instance.notify("Sync already in progress", severity="warning")
            return
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot sync.",
                severity="warning",
            )
            return
        if not self._server_available(service, self._active_server_id()):
            # Honest no-op: never claim "Sync completed" when nothing can
            # sync. Same predicate as the sync bar's collapse (review F10):
            # the bar and the s key must agree on whether sync is possible.
            self.app_instance.notify(
                "Local only — nothing to sync (no server connection).",
                severity="information",
            )
            return
        self._sync_running = True
        self.run_worker(self._run_sync, exclusive=True, group="schedules-sync-now")

    async def _run_sync(self) -> None:
        service = self._service()
        if service is None:
            self._sync_running = False
            return
        for btn_id in ("#scheduling-owner-local", "#scheduling-owner-server"):
            self.query_one(btn_id, Button).disabled = True
        try:
            owner_id = service.owner_id
            # task-23105 review F3: the engine swallows server errors into
            # persisted sync-error state, so its returned SyncOutcome is
            # the only honest report of what the attempt did -- a failed
            # sync must not surface as an info-severity no-op.
            outcome = await service.sync_now(owner_id)
            if outcome is not None and getattr(outcome, "status", None) == "error":
                self.post_message(
                    SyncFailed(
                        owner_id, getattr(outcome, "error", None) or "sync error"
                    )
                )
                return
            conflicts = service.db.get_conflicts(owner_id, primitive="reminder_task")
            self.post_message(
                SyncCompleted(
                    owner_id,
                    conflict_count=len(conflicts),
                    outcome=outcome,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Sync failed")
            self.post_message(SyncFailed(service.owner_id, str(exc)))
        finally:
            for btn_id in ("#scheduling-owner-local", "#scheduling-owner-server"):
                self.query_one(btn_id, Button).disabled = False
            self._refresh_owner_select()
            self._sync_running = False
