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
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Button, DataTable, Input, Static

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
    DefinitionFieldEditRequested,
    DefinitionLifecycleToggleRequested,
    DefinitionOwnerActionRequested,
    DefinitionRunNowRequested,
    DeleteTaskRequested,
    DisableTaskRequested,
    AcknowledgeIncidentRequested,
    EditTaskRequested,
    EnableTaskRequested,
    ReminderFieldEditRequested,
    ReminderOwnerActionRequested,
    RunReminderNowRequested,
    SyncCompleted,
    SyncFailed,
    ViewDefinitionAuditRequested,
    ViewDefinitionResultsRequested,
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
    RESULTS_HEADING,
    ResultsHostScreen,
    ResultsTab,
    # NOT `rich.markup.escape`: this dialog copy is rendered by a
    # Textual `Label` -> `Content.from_markup`, whose tokenizer eats ANY
    # `[...]`, while rich's escape only covers `[a-z#/@]...` tags (task 6
    # round 1). Same parser, same escape as the results detail pane.
    escape_markup,
    index_definitions_by_id,
    mark_results_read,
    _result_sort_key,
)
from ....UI.Screens.scheduling.sync_status_widget import SyncStatusWidget
from ....UI.Screens.scheduling.workbench_host_screen import WorkbenchHostScreen
from ....Widgets.confirmation_dialog import ConfirmationDialog
from ....Widgets.detail_value_row import DetailValueRow
from .definition_audit_view import DefinitionAuditView
from .definition_detail import (
    DefinitionDetail,
    _definition_transfer_suffix,
    _LIFECYCLE_TOGGLE_RESULTS,
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

#: Cadence for the scheduler-liveness strip (UAT finding 3d). It used to
#: be repainted only as a side effect of the 60s ticker above, so a
#: 0-30s "last tick Ns ago" value was visibly static for up to a full
#: minute between samples. `scheduler_liveness_line` is one JSON-file
#: read plus a string compare -- cheap enough for its own faster,
#: independent cadence. Also paused while the screen is not current
#: (same rule, TASK-23022).
SCHEDULER_LIVENESS_REFRESH_SECONDS = 5.0

#: Defensive cap for the server definitions' follow-the-pages load -- the
#: loop exists so the tail of a large definition list is never silently
#: hidden, not to render unbounded rows.
AUTOMATIONS_LOAD_MAX_ROWS = 500

#: Debounce before acting on a notification-triggered results pull
#: (schedules-handoff PR-6 task 4) -- a burst of `automation_run_*` events
#: collapses into ONE pull (plan ruling 3), same stop-and-restart timer
#: shape as the queue filter's own debounce above.
RESULTS_PULL_DEBOUNCE_SECONDS = 0.3

#: How many results the inbox lists. The DB default (50) silently hid older
#: rows while the badge counted EVERY unread one, so the badge could read
#: "Results (120)" over 50 rows. This is the sync-mirrored window --
#: exactly the newest-pages walk `SyncEngine._pull_results` performs -- so
#: the inbox shows everything a pull could have brought down and nothing it
#: could not. Beyond the cap the view says so out loud (`ResultsTab.
#: populate`'s `total`); deliberately no pagination machinery.
RESULTS_INBOX_LIMIT = _RESULTS_PAGE_SIZE * _SYNC_MAX_PAGES

#: redesign PR-4, task 6 (spec §11): the messages a PUSHED detail pane can
#: post whose handler lives on THIS screen rather than on the
#: `WorkbenchHostScreen` hosting it. Textual bubbles a message
#: widget -> parent -> Screen -> App, and the workbench is a different
#: screen, so without relaying these the pushed pane would render
#: correctly and do nothing (`WorkbenchHostScreen(route_message=...)`).
#: An allowlist rather than "relay everything": Textual's own internal
#: `messages.Update`/`Layout` bubble too and carry a widget belonging to
#: another screen.
_PUSHED_DETAIL_MESSAGES: tuple[type[Message], ...] = (
    AcknowledgeIncidentRequested,
    DefinitionFieldEditRequested,
    DefinitionLifecycleToggleRequested,
    DefinitionOwnerActionRequested,
    DefinitionRunNowRequested,
    DeleteTaskRequested,
    DisableTaskRequested,
    EditTaskRequested,
    EnableTaskRequested,
    ReminderFieldEditRequested,
    ReminderOwnerActionRequested,
    RunReminderNowRequested,
    ViewDefinitionAuditRequested,
    ViewDefinitionResultsRequested,
    #: `#schedules-follow-in-console` sits INSIDE `TaskDetail` while its
    #: handler is on this screen. Every other `Button.Pressed` handler
    #: here is id/class-scoped to a control the panes do not contain, so
    #: relaying the type wholesale cannot misfire.
    Button.Pressed,
)

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


def _row_title_cell(
    row: UnifiedRow, *, marked_ids: set[str], compact_owner_suffix: bool
) -> Text:
    """Queue-row title cell for one `UnifiedRow` (redesign PR-2, spec S4).

    Each primitive keeps its OWN existing title-suffix/prefix rendering
    verbatim -- a reminder row is byte-identical to the pre-redesign
    Title column (`_transfer_row_suffix`/`_queue_owner_suffix`), a
    definition row reuses `definition_detail.py`'s own Name-cell
    rendering (`automation_name_cell`/`_definition_transfer_suffix`) --
    rather than
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


class _QueueFilterInput(Input):
    """The rail search box -- `escape` blurs it (UAT Minor 21) instead of
    falling through to `SchedulesWorkbench`'s own `escape` -> clear_marks
    binding. Textual checks a focused widget's own BINDINGS before its
    ancestors' (`App._check_bindings` walks the focus chain closest-first),
    so this in-area binding wins without touching the screen's binding at
    all; unblurred, the filter kept focus and swallowed the single-letter
    chip keys (`f`, `1`-`4`) as literal text instead of routing them."""

    BINDINGS = [Binding("escape", "blur_filter", "Unfocus filter", show=False)]

    def action_blur_filter(self) -> None:
        self.blur()


class SchedulesWorkbench(BaseAppScreen):
    """Main workbench for managing scheduled runs, reminders, and jobs.

    TASK-24459: this screen's `scheduling-*`/`schedules-*` rules live in the
    generated ``css/screen_feature_scheduling.tcss``, loaded by the APP on
    first navigation to the route (``TldwCli._ensure_screen_owned_css``) --
    deliberately NOT via ``CSS_PATH``. Textual loads a screen's ``CSS_PATH``
    under ANY app, including the UI-test harnesses that model the unstyled
    tier (``ConsolidatedCSSApp`` loads no app bundle); a ``CSS_PATH`` here
    styled harness-mounted workbenches with only the MOVED half of the
    module -- a hybrid of the two tiers that flipped three
    destination-shell geometry tests (found 2026-09-04, paired arms).
    """

    # redesign PR-4, task 4: the spec §12 keyboard map. `1`-`4`/`f` cycle
    # the Queue chips, `/` focuses the filter, `n` opens the create
    # chooser (renamed from `c` -- straight rebind, survey §3), `p`
    # pauses/resumes the selected row (routed by kind), `m` opens the
    # selected row's Runs-on dropdown (the SAME activation Enter/click on
    # the row already drives), `r` marks a selected definition row's
    # unread results read. The legacy Automations-tab-only m/M/y/k
    # transfer keybindings (schedules-handoff PR-5 task 7 fix round) are
    # RETIRED here -- the Runs-on dropdown is the one transfer surface
    # (ruling 2; `_begin_automation_transfer`/`_cancel_automation_
    # transfer` and their action_* wrappers are deleted, zero remaining
    # consumers verified). `escape` keeps its existing "clear marks"
    # semantics (spec §12: "Esc back/close" -- WorkbenchHostScreen's own
    # `escape` pop handles the pushed-view half of that; this screen has
    # no back/close state beyond marks). `space`/`e`/`d`/`x`/`s`/`a`
    # are not named in spec §12 -- the survey's own reading ("stay as-is,
    # the spec text doesn't rule") keeps them unchanged. `r`'s OLD
    # meaning (tab-routed run-now/mark-read, `action_run_task_now`) is no
    # longer reachable via a key (ruling 2: "Run-now is NOT a global key
    # -- a detail-pane button" -- `DefinitionDetail`/`TaskDetail` both
    # already have one, task 3); redesign PR-4 task 5 then deleted that
    # method's tab routing with the tabs (one row-kind route left) and
    # retired the Results-tab-only `o` alongside it.
    BINDINGS = [
        Binding("1", "chip_all", "All", show=False),
        Binding("2", "chip_active", "Active", show=False),
        Binding("3", "chip_paused", "Paused", show=False),
        Binding("4", "chip_completed", "Completed", show=False),
        Binding("f", "cycle_chip", "Cycle filter"),
        Binding("/", "focus_search", "Search"),
        Binding("n", "create", "Create"),
        Binding("p", "pause_resume", "Pause/Resume"),
        Binding("m", "move_owner", "Move owner"),
        Binding("r", "mark_read", "Mark read"),
        Binding("e", "edit_task", "Edit"),
        Binding("space", "toggle_enabled", "Enable/Disable"),
        Binding("d", "delete", "Delete"),
        Binding("x", "mark_task", "Mark"),
        Binding("escape", "clear_marks", "Clear marks"),
        Binding("s", "sync_now", "Sync"),
        # `a` mirrors the rail's `Mark all read` button (schedules-handoff
        # PR-6 task 3), unaffected by the §12 remap. redesign PR-4 task 5
        # retired `o`/mark-solved here with the Results tab: it acted only
        # on that tab's selected row, and the pushed results view owns the
        # key now (`ResultsHostScreen`).
        Binding("a", "mark_all_results_read", "Mark all read"),
    ]

    # Footer hints must stay 1:1 with BINDINGS (minus escape, which needs
    # no advertising) and only advertise implemented actions (ADR-031).
    # Single letters are safe: focused inputs consume printable keys
    # before screen bindings fire.
    SCHEDULES_SHORTCUTS = (
        ("1", "all"),
        ("2", "active"),
        ("3", "paused"),
        ("4", "completed"),
        ("f", "cycle filter"),
        ("/", "search"),
        ("n", "create"),
        ("p", "pause/resume"),
        ("m", "move owner"),
        ("r", "mark read"),
        ("e", "edit"),
        ("space", "toggle"),
        ("d", "delete"),
        ("x", "mark"),
        ("s", "sync"),
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
        # redesign PR-2, Task 2 review round 2: any definition mutation
        # (create/edit save, run-now, transfer begin/cancel) sets this; a
        # reminder-only refresh (`refresh_definitions=False`) upgrades to
        # a full one and clears it, and returning to this screen while
        # stale does the same (`_consume_definitions_stale` -- redesign
        # PR-4 task 5 re-homed that trigger off the retired Queue-tab
        # `TabActivated`). Without this, the Queue's cached definition
        # rows could go stale for the whole session -- reminder-only
        # actions deliberately stopped self-healing it (finding 3's fix).
        self._definitions_stale = False
        self._chip: Chip = "all"
        self._filter_text = ""
        self._filter_debounce_timer: Timer | None = None
        self._next_run_refresh_timer: Timer | None = None
        self._liveness_refresh_timer: Timer | None = None
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
        #: redesign PR-4, task 6 (spec §11): the FRESH `TaskDetail`/
        #: `DefinitionDetail` instance currently hosted by a pushed
        #: `WorkbenchHostScreen` at narrow widths, or None. Held only so
        #: the docked panes' own paint seams can feed it the same data
        #: (`_detail_panes`) -- never to reparent or reuse it; the next
        #: push builds another instance.
        self._pushed_detail: TaskDetail | DefinitionDetail | None = None
        #: The `UnifiedRow.row_id` `_pushed_detail` was pushed FOR (fix
        #: wave F2). The pushed pane is pinned to this identity, never to
        #: a row index: a background refresh that drops the open row used
        #: to fall through `_render_table`'s `target_index = 0` and re-feed
        #: the overlay with a DIFFERENT row's data while its Header still
        #: named the original -- a pane whose Delete button then targeted
        #: the wrong reminder. `_detail_panes` now only feeds the pushed
        #: instance for THIS row_id, and `_pop_pushed_detail_if_gone`
        #: closes it (with a notice) when the row leaves the queue.
        self._pushed_row_id: str | None = None
        #: The `WorkbenchHostScreen` hosting `_pushed_detail`, held only so
        #: the gone-row check above can pop it.
        self._pushed_host: WorkbenchHostScreen | None = None
        #: One `asyncio.Lock` per definition an in-pane row edit has
        #: touched, serializing that definition's read-merge-write saves
        #: (Qodo finding 7) -- see `_edit_definition_field`.
        self._definition_edit_locks: dict[str, asyncio.Lock] = {}
        self._marked_ids: set[str] = set()
        #: The current hidden-panes notice from on_resize; combined with
        #: the marks/glyph legend in _update_pane_notice (task-23107).
        self._resize_notice = ""
        self._sync_running = False
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
                            tooltip="Schedule a new task (n).",
                        )
                        yield Button(
                            "Mark all read",
                            id="scheduling-mark-all-read",
                            tooltip="Mark every unread automation result read (a).",
                        )
                        # redesign PR-4, task 2: the Results
                        # relocation's rail affordance -- always
                        # visible (unlike Mark all read, which
                        # hides at zero unread: browsing READ
                        # results is still useful), mirroring the
                        # status strip's Conflicts badge's own
                        # "(N)" idiom.
                        yield Button(
                            "Results",
                            id="scheduling-results-badge",
                            tooltip=(
                                "Browse every automation result "
                                "(view/read/dismiss/mark solved)."
                            ),
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
                        # redesign PR-4, task 6 (spec §11): below the
                        # 84-column threshold the four chips above
                        # collapse into this one cycling control rather
                        # than vanishing, so a mouse user keeps a filter
                        # affordance at the 80x24 floor. Purely
                        # CSS-driven (`.compact`, `_scheduling.tcss`) and
                        # deliberately NOT `.scheduling-queue-chip`: that
                        # class is what the compact rule hides, and this
                        # button drives `action_cycle_chip` rather than
                        # selecting one named chip.
                        yield Button(
                            "Filter: All",
                            id="scheduling-chip-cycle",
                            tooltip="Cycle the queue filter (f).",
                        )
                    yield _QueueFilterInput(
                        # Says what ruling 5's search actually
                        # matches -- title + question/body (final
                        # review F6: the Type/Status columns are
                        # gone and status words like "missed" no
                        # longer match, which the user guide
                        # already documents).
                        placeholder="Filter: title or question…",
                        id="scheduling-queue-filter",
                        # UAT display blocker (finding 1b): a 1-row Input
                        # from app CSS height never carries Textual's own
                        # `.-textual-compact` class, so the existing
                        # `Input.-textual-compact:focus` outline opt-out
                        # (TASK-17961) missed it -- `*:focus`'s outline
                        # painted over the only content row, hiding
                        # whatever was typed. `compact=True` joins that
                        # family instead of adding a bespoke opt-out.
                        compact=True,
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
                    # redesign PR-4 task 5: with the Automations tab
                    # retired this is the ONLY `DefinitionDetail`
                    # instance left, and definition rows carry their
                    # own run-now/edit/pause actions (task 3).
                    yield DefinitionDetail(
                        id="scheduling-queue-definition-detail",
                        classes="pane-hidden",
                    )
                with Vertical(id="scheduling-inspector-pane"):
                    yield TaskInspector(id="scheduling-task-inspector")
            # redesign PR-2, Task 3: the bottom status strip (plan ruling
            # 4) -- sits below the list/detail/inspector row, hosting the
            # existing `SyncStatusWidget` (owner indicator + sync health,
            # with a width-compact styling path -- see `_sync_responsive_
            # workbench`) and a conflicts badge chip. redesign PR-4 task
            # 5: the tab bar it used to span is retired, so the badge
            # pushes the hosted Conflicts view (task 1) instead of
            # flipping to a tab.
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
                        "tasks only. Click to open the conflicts view."
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
        self._refresh_conflicts_badge()
        self._refresh_results_badge()
        self._sync_chip_cycle_label()
        # redesign PR-2, Task 3: hidden until `load_tasks` finds unread
        # rows (mirrors `SyncStatusWidget`'s own Clear-button idiom of
        # starting hidden rather than flashing visible-then-hidden).
        self.query_one("#scheduling-mark-all-read", Button).display = False
        # redesign PR-2, Task 2: glyph/title/subtitle (spec S4) replaces
        # the old Title/Type/Status/Next-Run shape -- a single primitive's
        # column set no longer fits a mixed reminder+definition list.
        table = self.query_one("#scheduling-task-table", DataTable)
        table.add_columns("", "Title", "Details")
        # task-23111 review F9: the relative next-run column ("in 25m")
        # is render-time text; refresh it periodically while visible.
        self._next_run_refresh_timer = self.set_interval(
            NEXT_RUN_REFRESH_SECONDS, self._refresh_next_run_rendering
        )
        # UAT finding 3d: its own faster, independent cadence -- it used
        # to be repainted only as a side effect of the timer above, so a
        # 0-30s value was static for up to a full minute between samples.
        self._liveness_refresh_timer = self.set_interval(
            SCHEDULER_LIVENESS_REFRESH_SECONDS, self._refresh_scheduler_liveness
        )
        self._request_tasks_refresh()
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

        UAT finding 3d: the scheduler-liveness line used to be refreshed
        only as a side effect of this 60s ticker -- it now has its own
        faster `_liveness_refresh_timer` (started alongside this one in
        `on_mount`), so this method no longer touches it.

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
        # redesign PR-2, Task 2: `_visible_rows`, not the reminder-only
        # `_visible_tasks` -- a Queue showing only definition rows still
        # has relative next-run text that must not go stale.
        if not self._visible_rows:
            return
        self._render_table(tick=True)

    def on_screen_suspend(self) -> None:
        """Stop the relative-time and liveness refreshes while covered.

        Hidden clocks must not tick unseen (TASK-23022); the resume
        handler refreshes immediately so no stale text is ever shown.
        """
        if self._next_run_refresh_timer is not None:
            self._next_run_refresh_timer.pause()
        if self._liveness_refresh_timer is not None:
            self._liveness_refresh_timer.pause()

    def on_screen_resume(self) -> None:
        """Refresh relative times/liveness and restart both cadences.

        No ``super().on_screen_resume()``: Textual's dispatcher invokes
        every handler along the MRO for one event (see BaseAppScreen's
        MRO contract), so the base handler runs regardless.
        """
        if self._next_run_refresh_timer is not None:
            self._next_run_refresh_timer.resume()
        if self._liveness_refresh_timer is not None:
            self._liveness_refresh_timer.resume()
        self._refresh_next_run_rendering()
        # UAT finding 3d: `_refresh_next_run_rendering` no longer covers
        # this (its own decoupled timer does) -- refresh it immediately
        # on uncover too, same "no stale text is ever shown" rule.
        self._refresh_scheduler_liveness()
        # redesign PR-4 task 5: the retired Queue-tab `TabActivated`
        # staleness consumer's new home -- see `_consume_definitions_stale`.
        self._consume_definitions_stale()

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

    def _consume_definitions_stale(self) -> None:
        """Upgrade the next Queue refresh when a definition mutation left
        the snapshot stale, rather than waiting for a reminder action to
        (maybe never) trigger one (review round 2).

        redesign PR-4 task 5: this used to be a `TabbedContent.
        TabActivated` handler for the Queue tab -- "the user just arrived
        back on the Queue". With the tab bar retired, the equivalent
        moment is the push/pop lifecycle (ruling 6): every surface this
        screen pushes (the hosted Conflicts/Results/audit views, and
        every modal form) suspends this screen and resumes it on pop, so
        `on_screen_resume` is the ONE seam covering all of them --
        `WorkbenchHostScreen`'s `dismissed` hooks fire alongside it for
        their own surface-specific refreshes.

        Safe at mount for the same reason the retired handler was not:
        `ScreenResume` also fires when this screen first becomes active,
        but the flag is only ever set by a genuine definition mutation,
        so nothing is pending yet at that point. (The old handler's
        regression came from `_request_automations_refresh` marking
        staleness at mount; that call site is retired with the tab.)
        """
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
        definition halves (the local+server merge in `_load_queue_
        definitions` -- or, when ``refresh_definitions`` is
        False AND nothing marked the cache stale, the definitions
        already in `self._all_rows` from the last full load, review
        finding 3), and one all-owners results listing (unread-count
        derivation only). The results read + row build are pushed off
        the event loop (`asyncio.to_thread`), the same "local DB read,
        off-thread" discipline every `service.db.*` read here uses.

        `self._definitions_stale` (review round 2): a definition
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

        # UAT Major 5 (unpinned): names WHICH step raised, since a bare
        # `logger.exception("Failed to load tasks")` covering three
        # distinct calls (reminders listing, definitions listing, row
        # build) left no way to pin the raiser without reproducing it
        # live. Updated before each step so the except below always
        # names the one in flight when it failed.
        stage = "listing tasks"
        try:
            combined = await service.list_tasks(
                owner_id=None, include_projections=False
            )
            # Defensive, not load-bearing: `include_projections=False`
            # already guarantees every row is a `ReminderTask`.
            reminders = [task for task in combined if isinstance(task, ReminderTask)]
            stage = "loading automation definitions"
            do_full_definitions_fetch = refresh_definitions or self._definitions_stale
            definitions = (
                await self._load_queue_definitions(service)
                if do_full_definitions_fetch
                else self._current_definitions()
            )
            if do_full_definitions_fetch:
                self._definitions_stale = False

            stage = "building unified rows"

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
        except Exception as exc:  # noqa: BLE001
            # UAT Major 5: a read failure is not evidence the queue is
            # empty. This used to reset `_tasks`/`_all_rows`, clear the
            # table, and blank every detail/inspector pane on ANY
            # exception here -- destroying the last-good display over a
            # transient read error, with nothing short of a fresh mount
            # able to restore it. Now it keeps whatever was already on
            # screen and only reports the failure.
            logger.exception(
                "Failed to load tasks while {stage} (owner_id={owner_id}, "
                "refresh_definitions={refresh_definitions}): {exc_type}: {exc}",
                stage=stage,
                owner_id=getattr(service, "owner_id", None),
                refresh_definitions=refresh_definitions,
                exc_type=type(exc).__name__,
                exc=exc,
            )
            self.app_instance.notify(
                "Could not refresh tasks (showing the last-loaded queue). "
                "Check the scheduling service and retry.",
                severity="error",
            )
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

        The both-owners merge the retired Automations tab's own loader
        established (redesign PR-4 task 5 deleted that loader; this is
        the only definitions fetch left): the device-only half of the
        local listing plus the paged server listing.

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
            # payload content.
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
        # Fix wave F2: before anything re-feeds the panes, close a pushed
        # overlay whose row is no longer in `_all_rows` -- otherwise the
        # restore/empty branches below would hand it another row's data.
        self._pop_pushed_detail_if_gone()
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
                    # Point at the control that is actually on screen
                    # (final review F10): the four chips below width 84
                    # are collapsed into the cycling control, which is
                    # what the user has to reach for there (redesign
                    # PR-4, task 6 -- the copy used to go silent because
                    # the row vanished entirely).
                    notice = "No tasks in this view."
                    notice += (
                        " Cycle the filter (f) to see the queue."
                        if self._chips_hidden()
                        else " Choose a different chip to see the queue."
                    )
                self._update_static_content(
                    self.query_one("#scheduling-task-detail-empty-state", Static),
                    notice,
                )

    def _chips_hidden(self) -> bool:
        """True when the four named chips are off screen (`_scheduling.
        tcss`: `#scheduling-workbench.compact #scheduling-queue-chips
        .scheduling-queue-chip { display: none }`), i.e. when the
        collapsed cycling control is standing in for them (redesign PR-4,
        task 6). Reads the class `on_resize` actually set rather than
        re-deriving its width threshold here."""
        try:
            return self.query_one("#scheduling-workbench").has_class("compact")
        except Exception:  # noqa: BLE001 - not mounted yet
            return False

    def _detail_hidden(self) -> bool:
        """True when `on_resize` has hidden the docked detail region.

        The same read-the-class-not-the-width discipline `_chips_hidden`
        uses: the threshold lives in `on_resize` alone, and this is what
        routes `Enter` to the pushed detail (redesign PR-4, task 6).
        """
        try:
            return self.query_one("#scheduling-detail-pane").has_class("pane-hidden")
        except Exception:  # noqa: BLE001 - not mounted yet
            return False

    def _detail_panes(
        self,
        pane_id: str,
        pane_type: type[TaskDetail | DefinitionDetail],
        *,
        row_id: str | None = None,
    ) -> list[Any]:
        """Every live instance of one detail pane class to paint.

        The docked pane always, plus the pushed instance whenever the
        narrow layout has one of the SAME class open (redesign PR-4, task
        6). Routing both through one list is what makes "the pushed pane
        is fed by the same service-backed loads the docked pane is" true
        by construction rather than by a parallel code path -- a mutation
        that repaints the pane behind repaints the one on screen too.

        ``row_id`` is the identity gate (fix wave F2). A row-scoped feed
        passes the `UnifiedRow.row_id` whose data it is about to write,
        and the pushed instance joins the list only when that is the row
        it was PUSHED for: the docked pane follows the queue cursor (and
        `_render_table`'s row-0 fallback when the selection vanishes),
        the pushed pane must not, or its Header names one reminder while
        its body -- and its Delete button -- belong to another. Feeds
        that carry no row identity at all (`_update_follow_button_state`'s
        availability flag) pass ``None`` and reach every instance, as
        before.
        """
        panes: list[Any] = [self.query_one(pane_id, pane_type)]
        pushed = self._pushed_detail
        if (
            isinstance(pushed, pane_type)
            and pushed.is_mounted
            and (row_id is None or row_id == self._pushed_row_id)
        ):
            panes.append(pushed)
        return panes

    def _pop_pushed_detail_if_gone(self) -> None:
        """Close a pushed detail whose row has left the queue (fix wave F2).

        The chosen honest gone-state is **auto-pop with a notice**, not a
        blanked overlay: a full-screen pane titled after a reminder that no
        longer exists has nothing true left to show, and popping is the
        one outcome that also removes its Delete/Disable/Run-now buttons
        from reach. Keyed off `_all_rows` (what EXISTS), never
        `_visible_rows` (what the current chip/filter shows) -- a filter
        narrowing must not close an open pane.

        Clears the pushed state inline rather than routing through
        `_pushed_detail_dismissed`: this runs from inside `_render_table`,
        i.e. inside the `schedules-load-tasks` worker, and that hook's own
        `_request_tasks_refresh()` would cancel the very load we are in
        (same exclusive group). If the host is not the active screen (a
        modal opened over it), the pop is skipped this round; the identity
        gate above still keeps the stale pane from being re-fed.

        Only `load_tasks`'s SUCCESS path reaches `_render_table` -- its
        `except` branch empties `_all_rows` and returns early -- so a
        transient service failure never closes an open pane.
        """
        host = self._pushed_host
        row_id = self._pushed_row_id
        if host is None or row_id is None:
            return
        if any(row.row_id == row_id for row in self._all_rows):
            return
        self._pushed_detail = None
        self._pushed_row_id = None
        self._pushed_host = None
        if self.app.screen is not host:
            return
        self.app_instance.notify(
            f"'{host.title}' is no longer in the queue.", severity="warning"
        )
        host.dismiss()

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

    @on(Button.Pressed, "#scheduling-chip-cycle")
    def _on_queue_chip_cycle_pressed(self, event: Button.Pressed) -> None:
        """The collapsed-chip control (redesign PR-4, task 6): the mouse
        route into the SAME `f` cycle, so the narrow layout loses no
        filter capability -- only the four-button row."""
        event.stop()
        self.action_cycle_chip()

    def _sync_chip_cycle_label(self) -> None:
        """Name the current chip on the collapsed control.

        Reads the selected chip button's own label rather than restating
        the four names here -- one source for the chip vocabulary.
        """
        try:
            chip_button = self.query_one(
                f"#scheduling-chip-{self._chip}", Button
            )
            cycle = self.query_one("#scheduling-chip-cycle", Button)
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        cycle.label = f"Filter: {chip_button.label}"
        # `Button.label` refreshes the widget but not the LAYOUT, and this
        # button is `width: auto` -- without this the box keeps the width
        # it was measured at ("Filter: All") and every longer chip name
        # paints clipped.
        cycle.refresh(layout=True)

    def _set_queue_chip(self, chip: Chip) -> None:
        if chip == self._chip:
            return
        self._chip = chip
        for button_id, candidate in self._CHIP_BY_BUTTON_ID.items():
            self.query_one(f"#{button_id}", Button).variant = (
                "primary" if candidate == chip else "default"
            )
        self._sync_chip_cycle_label()
        self._render_table()

    # -- redesign PR-4, task 4: spec §12 keyboard map ------------------------

    def action_chip_all(self) -> None:
        """`1`: switch the Queue chip to All."""
        self._set_queue_chip("all")

    def action_chip_active(self) -> None:
        """`2`: switch the Queue chip to Active."""
        self._set_queue_chip("active")

    def action_chip_paused(self) -> None:
        """`3`: switch the Queue chip to Paused."""
        self._set_queue_chip("paused")

    def action_chip_completed(self) -> None:
        """`4`: switch the Queue chip to Completed."""
        self._set_queue_chip("completed")

    def action_cycle_chip(self) -> None:
        """`f`: cycle to the next Queue chip, wrapping at the end.

        Works "at every width" (task-4 brief: don't couple to chip
        visibility) -- the chip row hides below 84 cols (`.compact`,
        `_chips_hidden`), but `_set_queue_chip` only ever touches
        `self._chip` and the still-MOUNTED button widgets' `.variant`,
        never their `.display`, so cycling never depends on the row
        being shown. `_CHIP_BY_BUTTON_ID`'s own insertion order
        (all/active/paused/completed, matching the chip row's visual
        left-to-right order) is reused as the cycle order rather than a
        second hard-coded tuple.
        """
        order = list(self._CHIP_BY_BUTTON_ID.values())
        next_chip = order[(order.index(self._chip) + 1) % len(order)]
        self._set_queue_chip(next_chip)

    def action_focus_search(self) -> None:
        """`/`: focus the Queue filter input."""
        search = self.query_one("#scheduling-queue-filter", Input)
        if search.display:
            search.focus()

    def action_create(self) -> None:
        """`n`: open the create-task chooser.

        Renamed from `c` (survey §3: "straight rename/rebind, no
        functional conflict") -- previously `c` skipped the chooser and
        went straight to the reminder form (`action_create_reminder`),
        which had drifted from the rail's own `Create ▾` button (always
        the chooser); `n` now matches the button.
        """
        self._open_new_task_chooser()

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

    @on(DataTable.RowSelected, "#scheduling-task-table")
    async def _on_task_row_selected(self, event: DataTable.RowSelected) -> None:
        """`Enter` on a queue row: at the narrow floor, push the detail.

        redesign PR-4, task 6 (spec §11, ruling 6). Above the threshold
        the detail pane is docked beside the queue and already shows this
        row, so `Enter` keeps being the no-op it has always been -- the
        push exists to replace the blank-hide, not to add a second way of
        seeing what is already on screen.
        """
        event.stop()
        if not self._detail_hidden():
            return
        await self._push_row_detail(event.cursor_row)

    async def _push_row_detail(self, index: int) -> None:
        """Host a FRESH detail pane for one queue row, full-screen.

        The same widget CLASS the docked pane uses, built by a factory
        per push (Task 1's contract -- never the docked instance
        reparented), then fed through `_update_detail_for_index`: the
        SAME service-backed loads the docked pane gets, so there is no
        second data path to keep in step. The push is awaited because
        that is what guarantees the hosted widget is mounted (and has
        composed its own children) before it is painted -- `App.push_
        screen`'s awaitable "waits for the screen to be mounted", and
        `TaskDetail.set_task`/`DefinitionDetail.set_definition` both
        query their children.
        """
        if not (0 <= index < len(self._visible_rows)):
            return
        row = self._visible_rows[index]
        built: list[TaskDetail | DefinitionDetail] = []

        def _factory() -> TaskDetail | DefinitionDetail:
            pane: TaskDetail | DefinitionDetail = (
                TaskDetail(id="scheduling-task-detail-overlay")
                if row.kind == "reminder"
                else DefinitionDetail(id="scheduling-definition-detail-overlay")
            )
            built.append(pane)
            return pane

        host = WorkbenchHostScreen(
            _factory,
            title=row.title,
            dismissed=self._pushed_detail_dismissed,
            route_message=self._route_pushed_detail_message,
        )
        await self.app.push_screen(host)
        if not built:
            return
        self._pushed_detail = built[0]
        # Fix wave F2: the pane is pinned to the row it was pushed for,
        # and the Header's `title` above is that same row's -- so the two
        # can no longer disagree. (A later rename of the pinned row does
        # leave the Header stale until the pane is reopened; the identity
        # it names stays correct, which is what the wrong-target-delete
        # defect turned on.)
        self._pushed_row_id = row.row_id
        self._pushed_host = host
        # Fix wave (finding 7): `index` is a POSITION captured before the
        # `await` above -- the queue can reorder, refresh, or filter while
        # the screen mounts, so replaying it here would repaint whatever
        # row now happens to sit at that position, not the one this pane
        # was pushed for. Resolve by IDENTITY instead (`row.row_id`,
        # captured before the await same as `_pushed_row_id`), the same
        # discipline F2 already applies to RE-feeds via the `_detail_
        # panes` gate. A row that vanished from `_all_rows` between the
        # `m`/Enter press and the mount completing takes the SAME
        # auto-pop-with-notice path `_pop_pushed_detail_if_gone` builds
        # for that case -- never a wrong-row paint.
        self._pop_pushed_detail_if_gone()
        if self._pushed_row_id is None:
            return
        for fresh_index, visible_row in enumerate(self._visible_rows):
            if visible_row.row_id == row.row_id:
                self._update_detail_for_index(fresh_index)
                break

    def _pushed_detail_dismissed(self) -> None:
        """Drop the pushed pane and re-read the queue behind it.

        Anything the overlay changed (an in-pane edit, a pause, a
        transfer) already refreshed through its own mutation path; this
        covers the rest -- and, more importantly, clears the reference so
        `_detail_panes` stops feeding a widget that is about to be
        pruned.
        """
        self._pushed_detail = None
        self._pushed_row_id = None
        self._pushed_host = None
        self._request_tasks_refresh()

    def _route_pushed_detail_message(self, message: Message) -> None:
        """Relay a pushed pane's request messages back to this screen.

        See `_PUSHED_DETAIL_MESSAGES`: a pushed pane's messages bubble to
        the host screen and then to `App`, never sideways to the screen
        underneath, so without this the pane would paint correctly and do
        nothing.
        """
        if isinstance(message, _PUSHED_DETAIL_MESSAGES):
            self.post_message(message)

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
            if from_tick and row.row_id == previously_selected_row_id:
                # redesign PR-4 task 5: the definition branch's F12 guard
                # below, mirrored here -- it was only ever applied to one
                # of the two branches. `_run_history_for` and `_incidents_
                # for` are SYNCHRONOUS `service.db` reads (list_task_runs,
                # plus list_task_incidents twice for both id spellings),
                # so a selected reminder row cost the ticker three DB
                # reads a minute for a row that did not move -- against
                # the ticker's own "no reload/DB on tick" contract (PR-2
                # plan Task 4). Refresh-driven calls (`from_tick=False`)
                # still re-feed unconditionally: the DATA can change
                # while the selection stands still.
                return
            # One read of each source, fed to every live instance of this
            # pane (redesign PR-4 task 6: the docked one, plus a pushed
            # one at narrow widths).
            run_history = self._run_history_for(task.id)
            incidents = self._incidents_for(task.id)
            known_timezones = self._task_timezones()
            runs_on_options = self._runs_on_options()[0]
            for task_detail in self._detail_panes(
                "#scheduling-task-detail", TaskDetail, row_id=row.row_id
            ):
                task_detail.set_task(
                    task,
                    run_history=run_history,
                    incidents=incidents,
                    # PR-3 task 3: same option source the create/edit
                    # modal's own Timezone selector reads
                    # (`_task_timezones`), so the pane's inline Timezone
                    # row editor offers the same zones.
                    known_timezones=known_timezones,
                    # PR-3 task 5: same option source the create/edit
                    # forms' own owner selector reads, so the Runs-on
                    # row's dropdown offers the same choices.
                    runs_on_options=runs_on_options,
                )
                self._update_transfer_actions(task_detail, task)
            self.query_one("#scheduling-task-inspector", TaskInspector).set_task(task)
        else:
            # `_selected_task_id = None` keeps every reminder-only
            # action (`_selected_task`, mark/toggle/delete) no-oping
            # gracefully on a definition row; the actions definitions DO
            # have (run-now/edit/pause/move/mark-read, PR-4 tasks 3/4)
            # route through `_selected_queue_definition` instead.
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
        """Cache the Runs-on row's lock/failure state for the detail pane.

        redesign PR-4 task 4 (ruling 2): this used to ALSO compute the
        legacy Move/Retry/Cancel buttons' disabled-reasons via `set_
        transfer_reasons` -- retired along with those buttons (the Runs-on
        dropdown is the one transfer surface now). What is left is what
        the dropdown/mini-bar still need: `transfer_lock_reason` (spec
        §6.3 read-only-except-cancel, final review I7) and the stored
        `to_server_failed` mutation errors (PR-3 task 5 fix round 1,
        finding 2), both never re-derived elsewhere.
        """
        service = self._scheduling_service
        if service is None or not isinstance(task, ReminderTask):
            task_detail.set_lifecycle_lock(None)
            task_detail.set_runs_on_transfer_errors([])
            return

        row = transfer_row_dict(task)
        retry_errors: list[str] = []
        if task.transfer_state == "to_server_failed":
            # fix round finding 3: keyed off the mutation's OWN owner_id
            # column, not a guess via "today's active server" -- a
            # `to_server_failed` row's mutation was recorded under
            # whatever server was active at the time of the failed
            # attempt, which silently stops matching after a server
            # switch if guessed instead of read.
            retry_errors = _pending_transfer_errors(service, "reminder_task", task.id)

        # spec §6.3 read-only-except-cancel (final review I7).
        task_detail.set_lifecycle_lock(service.transfer_lock_reason(row))
        # PR-3 task 5 fix round 1 (finding 2): feeds the Runs-on row's own
        # failure text -- one source, not a second derivation.
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
        available = (
            self._latest_console_follow_item_id is not None
            or self._latest_console_launch_kwargs is not None
        )
        for task_detail in self._detail_panes("#scheduling-task-detail", TaskDetail):
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

    @staticmethod
    def _transfer_confirm_dialog(
        name: str, direction: str, warnings: list[str]
    ) -> ConfirmationDialog:
        """Build the Move confirm dialog (spec §6.1/§6.2/§6.4) -- shared by
        the reminder and definition Runs-on dropdown flows (`_run_owner_
        transfer`, PR-3 task 5). The legacy button-driven `_begin_
        transfer`/`_begin_automation_transfer` call sites this originally
        served (Task 7 fix round item 1's "two near-identical call
        sites") were retired in redesign PR-4 task 4 -- this stayed
        because the dropdown is now the surface using it."""
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
        the reminder and definition transfer flows."""
        if direction == "to_server":
            return (
                f"'{name}' is queued to move to the server -- it still "
                "runs on this device until the server accepts it."
            )
        return (
            f"'{name}' is queued to move to this device -- a dormant copy "
            "is ready and will arm once the server releases it."
        )

    @on(Button.Pressed, "#scheduling-new-task")
    def _on_new_task_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_new_task_chooser()

    @on(Button.Pressed, "#scheduling-mark-all-read")
    def _on_mark_all_read_pressed(self, event: Button.Pressed) -> None:
        """Rail `Mark all read` (redesign PR-2, Task 3) -- the same
        fan-out `a` runs; redesign PR-4 task 5 collapsed the rail's own
        ungated twin into `action_mark_all_results_read` when the tab
        gate that forced the duplication was retired."""
        event.stop()
        self.action_mark_all_results_read()

    @on(Button.Pressed, "#scheduling-results-badge")
    def _on_results_badge_pressed(self, event: Button.Pressed) -> None:
        """Rail `Results (N)` (redesign PR-4, task 2) -- pushes the whole
        (unfiltered) inbox as a fresh `ResultsHostScreen` overlay, beside
        `Mark all read`. Task 5 retired the Results tab, so this (and the
        definition panes' own unread row) is the only route to it."""
        event.stop()
        self._push_results_overlay()

    @on(ViewDefinitionResultsRequested)
    def _on_view_definition_results_requested(
        self, event: ViewDefinitionResultsRequested
    ) -> None:
        """A definition pane's `Unread results` row was activated
        (redesign PR-4, task 2) -- the live replacement for the retired
        "See Results tab" pointer."""
        event.stop()
        self._push_results_overlay(definition=event.definition)

    def _push_results_overlay(
        self, *, definition: dict[str, Any] | None = None
    ) -> None:
        """Push a standalone `ResultsTab` instance via `ResultsHostScreen`
        (redesign PR-4, task 2's Results relocation).

        Mirrors `_push_conflicts_overlay`'s shape (pre-read once, hand
        the fresh instance its own data) but the pushed view ALSO needs
        its own read/dismiss/mark-solved/mark-all binding surface
        (`ResultsHostScreen`, not the plain `WorkbenchHostScreen`): a
        screen underneath never receives a key event, so the workbench's
        own bindings cannot serve this view (task 5 then retired the
        workbench's tab-gated r/d/o copies outright). `query`/`unread_
        ids` are re-run after every mutation to repaint the SAME scope;
        the workbench's own rail/Queue state re-syncs once, on pop, via
        `dismissed` (brief: "refresh the rail + unified rows on
        dismissed").

        `definition=None` is the rail's global inbox; otherwise the
        listing (and its cap line) is scoped to that one definition,
        across both its local/server id spaces.
        """
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable.", severity="warning"
            )
            return

        def _query() -> tuple[
            list[dict[str, Any]], dict[str, dict[str, Any]], int
        ]:
            if definition is None:
                results = service.db.list_automation_results(
                    owner_id=None, limit=RESULTS_INBOX_LIMIT
                )
                total = service.db.count_automation_results(owner_id=None)
            else:
                results, total = self._definition_results_query(service, definition)
            definitions_by_id = index_definitions_by_id(
                service.db.list_automation_definitions(owner_id=None)
            )
            return results, definitions_by_id, total

        def _unread_ids() -> list[str]:
            if definition is None:
                return self._unread_result_ids(service)
            return self._definition_unread_result_ids(service, definition)

        results, definitions_by_id, total = _query()
        heading = RESULTS_HEADING
        if definition is not None:
            name = str(
                definition.get("name") or definition.get("id") or "Untitled automation"
            )
            heading = f"{RESULTS_HEADING} — {escape_markup(name)}"

        def _factory() -> ResultsTab:
            return ResultsTab(
                id="scheduling-results-overlay",
                initial_results=results,
                initial_definitions_by_id=definitions_by_id,
                initial_total=total,
                heading=heading,
            )

        self.app.push_screen(
            ResultsHostScreen(
                _factory,
                title="Results",
                service=service,
                query=_query,
                unread_ids=_unread_ids,
                dismissed=self._refresh_results_surfaces,
            )
        )

    @on(Button.Pressed, "#scheduling-conflicts-badge")
    def _on_conflicts_badge_pressed(self, event: Button.Pressed) -> None:
        """Status-strip conflicts badge (redesign PR-4, Task 1) -- pushes
        the hosted Conflicts view as a fresh overlay instance instead of
        flipping tabs (the tab flip could not survive the tab bar). Task
        5 retired the Conflicts tab, so this badge is the only route to
        the view."""
        event.stop()
        self._push_conflicts_overlay()

    def _push_conflicts_overlay(self) -> None:
        """Push a standalone `ConflictsTab` instance via `WorkbenchHostScreen`.

        Pre-reads the same `get_conflicts` shape `_refresh_conflicts_badge`
        uses (no new query) and hands the result to the fresh instance as
        `initial_conflicts` so it paints immediately on mount -- a pushed
        instance has no external `.populate()` driver (task 5 retired the
        mounted tab instance that did). `dismissed=self._refresh_
        conflicts_badge` re-syncs the badge count on pop, in case the
        overlay resolved anything while it was open.

        Fix wave F3: the push also relays `ConflictsTab.ConflictResolved`
        back here (task 6's `route_message` pattern). Without it the
        message bubbled tab -> host screen -> App and never reached
        `_on_conflict_resolved`, so resolving a conflict updated the badge
        on pop but never reloaded the queue -- the row kept showing the
        LOSING side's title/schedule until some unrelated refresh
        happened to fire. Relaying restores the LIVE reload the mounted
        tab used to get, while the overlay is still open.
        """
        service = self._service()
        conflicts = (
            service.db.get_conflicts(service.owner_id, primitive="reminder_task")
            if service is not None
            else []
        )
        sync_engine = service.sync_engine if service is not None else None

        def _factory() -> ConflictsTab:
            return ConflictsTab(
                sync_engine=sync_engine,
                initial_conflicts=conflicts,
                id="scheduling-conflicts-overlay",
            )

        self.app.push_screen(
            WorkbenchHostScreen(
                _factory,
                title="Conflicts",
                dismissed=self._refresh_conflicts_badge,
                route_message=self._route_conflicts_message,
            )
        )

    def _route_conflicts_message(self, message: Message) -> None:
        """Relay a pushed `ConflictsTab`'s resolution back to this screen
        (fix wave F3) -- the allowlisted counterpart of `_route_pushed_
        detail_message`, kept separate because the two pushes carry
        entirely different message vocabularies.
        """
        if isinstance(message, ConflictsTab.ConflictResolved):
            self.post_message(message)

    @on(DefinitionRunNowRequested)
    def _on_definition_run_now_requested(
        self, event: DefinitionRunNowRequested
    ) -> None:
        """A definition pane's `Run now` button was pressed (redesign
        PR-4, task 3 -- the retired Automations-tab `r` key's live
        replacement, ruling 2); routed through the existing owner-routed
        dispatch unchanged."""
        event.stop()
        self._run_automation_now(event.definition)

    @on(ViewDefinitionAuditRequested)
    def _on_view_definition_audit_requested(
        self, event: ViewDefinitionAuditRequested
    ) -> None:
        """A definition pane's `Last run` row was activated (redesign
        PR-4, task 3 -- the retired Automations-tab third pane's live
        replacement)."""
        event.stop()
        self._push_definition_audit_overlay(event.definition)

    def _push_definition_audit_overlay(self, definition: dict[str, Any]) -> None:
        """Push a standalone `DefinitionAuditView` via `WorkbenchHostScreen`.

        Unlike `_push_conflicts_overlay`/`_push_results_overlay`, the
        fetch here is unavoidably async (`server_client.list_automation_
        definition_audit`) -- there is no synchronous value to pre-read
        before pushing, so the widget fetches its OWN data on mount
        (`DefinitionAuditView.on_mount`) rather than being handed one.
        Read-only (no r/d/o/a-shaped mutation surface), so the plain
        `WorkbenchHostScreen` hosts it directly -- no dedicated Screen
        subclass, no `dismissed` hook needed (nothing here can go stale
        by being viewed).
        """
        service = self._service()
        name = str(
            definition.get("name") or definition.get("id") or "Untitled automation"
        )

        def _factory() -> DefinitionAuditView:
            return DefinitionAuditView(
                service, dict(definition), id="scheduling-audit-view-overlay"
            )

        # The `Screen.title` -> `Header` route renders through `Content
        # (title)` (a LITERAL constructor, unlike `Static.update(str)` ->
        # `Content.from_markup`) -- verified against Textual's own
        # `App.format_title`/`Content.__init__` -- so the definition's
        # name needs no `escape_markup` here, unlike `ResultsTab`'s own
        # `heading` Static.
        self.app.push_screen(
            WorkbenchHostScreen(_factory, title=f"Run history — {name}")
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

        Refreshes the QUEUE eagerly rather than leaving it to the lazy
        staleness consumer (final review F1): Task 3 moved the create
        entry point onto the Queue rail (`Create ▾` -> "Recurring
        question…"), so this save's flagship path never leaves the
        screen -- nothing pops back onto it, and the automation the user
        just created stayed invisible on the surface it was created
        from. Symmetric with the reminder half
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
        # Full refresh (definitions included): `_definitions_stale` is set
        # above, so a `refresh_definitions=False` call would upgrade
        # itself anyway -- ask for what this actually needs. redesign PR-4
        # task 5: the paired Automations-list reload retired with the tab;
        # the Queue IS the definitions list now.
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
                logger.exception(
                    # Qodo finding 2: an unattributed "it failed" line is
                    # unusable in a log with many reminders. Ids only --
                    # the VALUE is the user's own reminder text.
                    "Failed to edit reminder {task_id} field {field}",
                    task_id=task.id,
                    field=row.row_key,
                )
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
        + `_request_tasks_refresh()`, which rebuilds the unified rows --
        PLUS `_repaint_queue_definition_detail`, because a rows rebuild
        is not a repaint of the detail pane beside them (final review
        F4/I4). redesign PR-4 task 5: the refresh call used to be
        `_request_automations_refresh()` (the retired tab's own list).

        Every failure path paints through `_show_definition_row_error`
        rather than `row.show_error` directly (Qodo finding 9): this
        worker holds `row` across an `await`, so a late failure could
        otherwise stamp its message under whatever automation the pane
        moved on to -- the same crossed-identity class the commit path's
        `_editing_definition` belt closes on the way in.
        """
        service = self._scheduling_service
        definition_id = str(definition.get("id") or "")
        if service is None:
            row.show_error(
                "Scheduling service is unavailable; cannot save this edit."
            )
            return

        # Per-DEFINITION serialization (Qodo finding 7, narrowing final
        # review M12): a field-keyed worker group let two edits of the
        # SAME automation run concurrently, and `save_definition` merges
        # each payload onto the row it read at ENTRY -- so the slower one
        # wrote back a snapshot taken before the faster one landed,
        # silently reverting it. The gate is a per-definition lock rather
        # than `exclusive=True` on a per-definition group, because
        # Textual's exclusivity CANCELS the running worker instead of
        # queueing behind it: that turns "two quick edits" into "the
        # first one is discarded", which is the same lost update by
        # another route (M12 hit exactly this across rows). Different
        # automations hold different locks and still run in parallel.
        # The map is never pruned: one `asyncio.Lock` per definition
        # edited this session, which is bounded by what a human clicks.
        lock = self._definition_edit_locks.setdefault(definition_id, asyncio.Lock())

        async def _edit_and_refresh() -> None:
            async with lock:
                await self._save_definition_field(
                    service, definition, definition_id, payload, row
                )

        self.run_worker(
            _edit_and_refresh,
            group=f"schedules-edit-definition-{definition_id}",
        )  # type: ignore[arg-type]

    async def _save_definition_field(
        self,
        service: "SchedulingService",
        definition: dict[str, Any],
        definition_id: str,
        payload: dict[str, Any],
        row: DetailValueRow,
    ) -> None:
        """`_edit_definition_field`'s worker body, under its row lock."""
        local_id = await self._resolve_local_definition_id(service, definition)
        if local_id is None:
            self._show_definition_row_error(
                row,
                {definition_id},
                "Could not prepare this automation for editing — see the log.",
            )
            return
        owner_id = str(definition.get("owner_id") or "local")
        painted = {definition_id, local_id}
        try:
            outcome = await service.save_definition(
                payload, owner_id, definition_id=local_id
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                # Qodo finding 2: which automation, and which of its
                # fields -- never the value, which is the user's own
                # question/model text.
                "Failed to edit automation definition {definition_id} field "
                "{field}",
                definition_id=local_id,
                field=row.row_key,
            )
            self._show_definition_row_error(row, painted, "Failed to save this edit.")
            return
        if outcome.status not in ("saved", "queued"):
            message = "; ".join(
                str(err.get("message") or "")
                for err in outcome.errors
                if err.get("message")
            ) or "This edit could not be saved."
            self._show_definition_row_error(row, painted, message)
            return
        row.clear_error()
        self._definitions_stale = True
        self._request_tasks_refresh()
        await self._repaint_queue_definition_detail(service, local_id, definition)

    def _show_definition_row_error(
        self, row: DetailValueRow, definition_ids: set[str], message: str
    ) -> None:
        """`row.show_error(message)`, unless the row moved on (finding 9).

        The pane that owns ``row`` is asked what it is currently
        painting; if that is no longer one of ``definition_ids`` -- the
        listing id the edit was dispatched for plus the local id it
        resolved to, the same both-shapes pair `_repaint_queue_
        definition_detail` matches on -- the message is dropped with a
        debug line instead of being stamped onto another automation's
        field. An unparented row (never mounted, or already removed)
        paints as before: there is no pane to contradict it.
        """
        pane = next(
            (
                ancestor
                for ancestor in row.ancestors
                if isinstance(ancestor, DefinitionDetail)
            ),
            None,
        )
        if pane is not None and pane.shown_definition_id not in definition_ids:
            logger.debug(
                "Dropping a stale automation edit error for {expected}: the "
                "pane now shows {actual}",
                expected=sorted(definition_ids),
                actual=pane.shown_definition_id,
            )
            return
        row.show_error(message)

    async def _repaint_queue_definition_detail(
        self,
        service: "SchedulingService",
        local_id: str,
        definition: dict[str, Any],
    ) -> None:
        """Repaint the queue's `DefinitionDetail` for ``local_id``.

        The pane is painted from `_update_detail_for_index`, which
        early-returns for the same row on a tick, so nothing repainted it
        after an in-pane edit: the editor closed, the row restored the
        OLD value, and stayed that way indefinitely even though the edit
        had persisted (final review F4/I4). A rows refresh rebuilds the
        TABLE, not the pane beside it, so this stays a separate step --
        it needs the authoritative re-read `apply_lifecycle`'s single
        known column does not.

        Only paints when the selected row IS this definition --
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
            # redesign PR-4 task 5: this used to loop over BOTH
            # `DefinitionDetail` instances (the Automations tab's and the
            # Queue's); the tab's is retired, so one query remains -- the
            # try/except stays because the pane can be unmounted mid-worker.
            try:
                detail = self.query_one(
                    "#scheduling-queue-definition-detail", DefinitionDetail
                )
            except Exception:  # noqa: BLE001 - not mounted yet
                pass
            else:
                detail.apply_lifecycle(definition_id, new_lifecycle)
            self._definitions_stale = True
            self._request_tasks_refresh()

        self.run_worker(
            _toggle_and_refresh,
            exclusive=True,
            group="schedules-definition-lifecycle",
        )  # type: ignore[arg-type]

    # -- Owner-row transfer dropdown (PR-3 task 5, spec §7 flow) -------------
    #
    # Originally a SECOND, row-scoped surface onto the PR-5 transfer
    # facade alongside the legacy Move/Retry/Cancel buttons/keybindings
    # (own events, own helpers, deliberately independent end to end --
    # coexistence was a pinned requirement, task-5 brief). redesign PR-4
    # task 4 retired the legacy side (ruling 2) -- this is now the ONE
    # transfer surface. A refusal renders inline via `row.show_error`.

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

        `transfer_refusal` runs FIRST (health-quoting preserved) -- a
        refusal renders inline via `row.show_error`, never a toast.
        Allowed -> the SAME `ConfirmationDialog` +
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
            self._request_tasks_refresh()

        async def _do() -> None:
            # fix round 1 finding 1: unconditional, BEFORE resolving --
            # same rule `_begin_automation_transfer`/`_cancel_automation_
            # transfer` follow (schedules_workbench.py:2766-2778).
            # `_resolve_local_definition_id` can itself mirror a brand
            # new local row (`upsert_automation_definitions_from_server`)
            # the first time a pure server-fetch definition is touched --
            # regardless of which branch below this lands on (refused,
            # failed, `local_id is None`, or a genuine success), the
            # Queue's cached definitions may now be outdated.
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
        """Run the highlighted Queue row immediately.

        redesign PR-4, task 4: no longer key-bound (ruling 2 -- "Run-now
        is NOT a global key"; `r` is `action_mark_read`). It stays as the
        one run-now entry point the panes' own `Run now` buttons and the
        tests both reach.

        redesign PR-4, task 5: the Automations/Results tab-routing
        branches are gone with the tabs -- there is one surface now, so a
        reminder row runs the local reminder dispatch (task-18938) and a
        definition row runs the owner-routed `_run_automation_now` (task
        3, family-agnostic: every definition family, not just `recurring_
        question`).
        """
        task = self._selected_reminder_task()
        if task is not None:
            self._run_reminder_now(task)
            return
        definition = self._selected_queue_definition()
        if definition is not None:
            self._run_automation_now(definition)
            return
        # Never swallow the key silently (final review F8): nothing at
        # all under the cursor did nothing, with no message.
        self.app_instance.notify(self._no_task_notice("run"), severity="warning")

    def _selected_reminder_task(self) -> ReminderTask | None:
        """Return the highlighted task when it is a reminder (not a projection)."""
        for task in self._visible_tasks:
            if task.id == self._selected_task_id and isinstance(task, ReminderTask):
                return task
        return None

    def _selected_queue_definition(self) -> dict[str, Any] | None:
        """Return the automation definition under the QUEUE cursor, if any.

        redesign PR-4, task 3: the definition-row counterpart of
        `_selected_task` -- its own row-index routing (the two lists diverge
        whenever a definition row precedes the cursor), but returning the
        DEFINITION instead of `None` for a definition row rather than
        `_selected_task`'s "nothing to act on" contract, since Queue
        definition rows now have run-now/edit actions of their own.
        """
        if not self._visible_rows:
            return None
        table = self.query_one("#scheduling-task-table", DataTable)
        row_index = table.cursor_row
        if row_index is None or not (0 <= row_index < len(self._visible_rows)):
            return None
        row = self._visible_rows[row_index]
        if row.kind != "definition":
            return None
        return row.source_row

    def action_pause_resume(self) -> None:
        """`p`: pause/resume the selected row (spec §12), routed by kind.

        A reminder row (or a bulk mark selection) reuses `action_toggle_
        enabled` VERBATIM -- same single-row/bulk-marked seam, same
        transfer-lock refusal (`_refuse_if_transfer_locked`). A
        definition row toggles `lifecycle` via the SAME facade call the
        header Pause/Resume button drives (`_toggle_definition_
        lifecycle`) -- the SAME `transfer_lock_reason` (never re-derived)
        the button's own honest refusal uses, reported here as a
        notification since this is a screen-level key, not a row with
        its own error slot.
        """
        if self._marked_ids or self._selected_task() is not None:
            self.action_toggle_enabled()
            return
        definition = self._selected_queue_definition()
        if definition is not None:
            service = self._scheduling_service
            if service is None:
                self.app_instance.notify(
                    "Scheduling service is unavailable; cannot update the "
                    "automation.",
                    severity="warning",
                )
                return
            name = str(definition.get("name") or definition.get("id") or "")
            lock_reason = service.transfer_lock_reason(definition)
            if lock_reason is not None:
                self.app_instance.notify(
                    f"Cannot pause/resume '{name}': {lock_reason}",
                    severity="warning",
                )
                return
            lifecycle = str(definition.get("lifecycle") or "configured")
            action = "resume" if lifecycle != "configured" else "pause"
            self._toggle_definition_lifecycle(definition, action)
            return
        self.app_instance.notify(
            self._no_task_notice("pause or resume"), severity="warning"
        )

    async def action_move_owner(self) -> None:
        """`m`: open the selected row's Runs-on dropdown (spec §12/§7).

        Posting `DetailValueRow.Activated` on the row directly drives the
        exact same `on_detail_value_row_activated` path a real Enter/
        click already does -- honest lock/family-note refusal (`row.
        show_error`), then the dropdown -- so this needs no new
        activation logic, only a reference to the right pane's row
        (`TaskDetail`/`DefinitionDetail.runs_on_row`).

        Fix wave F1: below the responsive floor the docked detail region
        is `display: none`, and this used to resolve the DOCKED pane's
        row anyway -- mounting a `Select` inside an invisible pane, taking
        focus off the queue table, and painting nothing. The queue's
        arrow keys then stopped working with no notification and no
        visible editor: a silent input trap, only escapable by a guessed
        `Esc`. Narrow widths now take the SAME route `Enter` does
        (`_push_row_detail`, ruling 6) and activate the Runs-on row of the
        pane that is actually on screen -- `m` still means "open the
        transfer dropdown for this row", it just opens it where the user
        can see it. At or above the threshold nothing changes.
        """
        row: DetailValueRow | None
        if self._detail_hidden():
            table = self.query_one("#scheduling-task-table", DataTable)
            index = table.cursor_row
            if index is None or not (0 <= index < len(self._visible_rows)):
                row = None
            else:
                await self._push_row_detail(index)
                pushed = self._pushed_detail
                row = pushed.runs_on_row if pushed is not None else None
        elif self._selected_task() is not None:
            row = self.query_one("#scheduling-task-detail", TaskDetail).runs_on_row
        elif self._selected_queue_definition() is not None:
            row = self.query_one(
                "#scheduling-queue-definition-detail", DefinitionDetail
            ).runs_on_row
        else:
            row = None
        if row is None:
            self.app_instance.notify(
                self._no_task_notice("move"), severity="warning"
            )
            return
        row.post_message(DetailValueRow.Activated(row))

    def action_mark_read(self) -> None:
        """`r`: mark the selected definition row's unread results read
        (spec §12/§8) -- the SAME per-id fan-out `a`/the rail's `Mark all
        read` already drive (`_dispatch_mark_all_results_read`), scoped
        to just this one definition via `_definition_unread_result_ids`
        (both id spaces). A reminder row has no results concept at all
        (honest no-op copy)."""
        if self._selected_task() is not None:
            self.app_instance.notify(
                "This task has no results to mark read.",
                severity="information",
            )
            return
        definition = self._selected_queue_definition()
        if definition is None:
            self.app_instance.notify(
                self._no_task_notice("mark read"), severity="warning"
            )
            return
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable.", severity="warning"
            )
            return
        unread_ids = self._definition_unread_result_ids(service, definition)
        if not unread_ids:
            self.app_instance.notify(
                "Nothing unread for this automation.", severity="information"
            )
            return

        async def _mark() -> None:
            await self._dispatch_mark_all_results_read(service, unread_ids)

        self.run_worker(
            _mark, exclusive=True, group="schedules-mark-all-read"
        )  # type: ignore[arg-type]

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

    def _device_only_automations(
        self, all_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """The device-only half of a local definitions listing, decorated.

        Split out of the retired Automations loader (final review F2) so
        the Queue loader can keep the UNFILTERED listing for its
        unread-count resolution while still deriving the same display
        half from it -- one `list_automation_definitions` call per
        refresh either way.

        Health is never persisted (`automation_health.py`'s own
        docstring) -- it is computed fresh here the same way `run_
        automation_now` computes it before dispatching, so the column
        never shows the create-time placeholder
        (`execution_unavailable`) as if it were live.
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
        """Follow `has_more` pages so the list never silently hides the
        tail of a large definition list; the cap is a defensive bound,
        not an expected cliff.

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

    async def _fetch_definition_detail_counts(
        self,
        service: "SchedulingService",
        definition: dict[str, Any],
        definition_id: str,
    ) -> tuple[int, dict[str, Any] | None, int, bool]:
        """Off-thread run_count/last_run/unread_count read for one
        definition, for `_load_queue_definition_detail` (redesign PR-2,
        Task 2) -- owner-scoped (final review F11), with the
        never-paint-0-off-a-failed-read guard (F14). It used to be shared
        with the Automations tab's own detail pane; task 5 retired that
        second caller, and this stays a separate method because the
        `asyncio.to_thread` read is the part worth naming.
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
        """Schedule the definition-detail counts through their own
        exclusive worker group (redesign PR-2, Task 2) -- latest
        selection wins, and the group is its own so a definition
        selection never contends with the reminder/table loaders."""

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
        """Paint the `DefinitionDetail` pane for the highlighted
        definition row (redesign PR-2, Task 2), reading its counts off
        the event loop.
        """
        # Every live instance of this pane (redesign PR-4 task 6: the
        # docked one, plus a pushed one at narrow widths) -- re-queried
        # after the await, since a push/pop can land while it runs.
        definition_id = str(definition.get("id") or "")
        service = self._scheduling_service
        if service is None:
            if row_id == self._selected_row_id:
                for detail in self._detail_panes(
                    "#scheduling-queue-definition-detail",
                    DefinitionDetail,
                    row_id=row_id,
                ):
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
        known_timezones = self._task_timezones()
        # PR-3 task 5: same option source the reminder pane's own Runs-on
        # row reads (`_update_detail_for_index`).
        runs_on_options = self._runs_on_options()[0]
        lifecycle_lock = service.transfer_lock_reason(definition)
        transfer_errors = _definition_transfer_errors(service, definition)
        for detail in self._detail_panes(
            "#scheduling-queue-definition-detail", DefinitionDetail, row_id=row_id
        ):
            detail.set_definition(
                definition,
                run_count=run_count,
                last_run=last_run,
                unread_count=unread_count,
                history_error=history_error,
                known_timezones=known_timezones,
                runs_on_options=runs_on_options,
            )
            # PR-3 task 4: `reason` comes from `SchedulingService.
            # transfer_lock_reason` (never re-derived in the widget), fed
            # right after `set_definition` per that method's own
            # docstring -- the same discipline `_update_transfer_actions`
            # follows for the reminder pane.
            detail.set_lifecycle_lock(lifecycle_lock)
            # PR-3 task 5 fix round 1 (finding 2): the Runs-on row's own
            # failure text, from the same source.
            detail.set_runs_on_transfer_errors(transfer_errors)

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
            self._request_tasks_refresh()

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
            self._request_tasks_refresh()
            # redesign PR-4 task 5: the immediate + 5s-delayed audit-trail
            # re-fetches that used to follow retired with the Automations
            # tab's history pane. The audit trail's home is now the pushed
            # `DefinitionAuditView`, which fetches on mount -- so it is
            # never showing a stale trail that needs poking. The dispatch
            # still returns when the run is ENQUEUED, not finished, and
            # the notification still reports the result.

        self.run_worker(
            _run,
            exclusive=True,
            group="schedules-run-automation-now",
        )  # type: ignore[arg-type]

    def _edit_selected_automation(
        self, definition: dict[str, Any] | None
    ) -> None:
        """Open an automation definition for editing (e key).

        `agent_task` rows are excluded -- only `recurring_question`
        authoring exists (the same v1 scope guard `save_definition`
        itself enforces via `_reject_unsupported_family`).

        Args:
            definition: The definition to edit, resolved by the caller
                (`_selected_queue_definition()`, redesign PR-4 task 3's
                Queue-row routing) -- ``None`` gets the honest refusal
                below. Task 5 dropped the old `_selected_automation()`
                default with the Automations tab it read from; the one
                remaining caller always passes a value explicitly.
        """
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

    # redesign PR-4, task 4 (ruling 2): the Automations-tab-only
    # m/M/y/k transfer keybindings and their `_begin_automation_
    # transfer`/`_cancel_automation_transfer` flow (schedules-handoff
    # spec §6, PR-5 task 7 fix round item 1) are RETIRED -- the Runs-on
    # row's dropdown (`_run_owner_transfer`/`_run_owner_cancel`, PR-3
    # task 5) is the one transfer surface now. `git grep` verified zero
    # remaining consumers of `action_move_automation_to_local`/`_to_
    # server`, `action_retry_automation_transfer`, `action_cancel_
    # automation_transfer`, `_is_automations_tab_active`, `_show_
    # automations_inline_reason` before deleting them.

    # -- Results actions (schedules-handoff PR-6 task 3) ------------------
    #
    # redesign PR-4, task 5: the Results TAB is retired, and with it the
    # tab-gated `r`/`d`/`o` bindings that only ever acted on its own
    # selected row (`_is_results_tab_active`, `_review_selected_result`,
    # `action_mark_result_solved`). The pushed results view owns those
    # verbs now (`ResultsHostScreen`'s own r/d/o/a, task 2) -- and owns
    # them ALONE, since Textual never routes a key to a screen underneath
    # the active one. `git grep` verified zero remaining consumers of all
    # three before deleting them. `a` survives on this screen because its
    # target survives: the rail's `Mark all read` button, whose fan-out is
    # scoped to the FULL unread set rather than any one listing.

    def _notify(self, message: str, severity: str) -> None:
        self.app_instance.notify(message, severity=severity)

    def _unread_result_ids(self, service: "SchedulingService") -> list[str]:
        """Every unread result id across the FULL table, read straight
        from the DB -- not the pushed view's own listing, which is capped
        at `RESULTS_INBOX_LIMIT` (200) rows. The rail button's visibility
        already sums the full-table unread count (`_refresh_results_badge`'s
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

    @staticmethod
    def _definition_id_space(
        definition: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """`(local_id, server_id)` for `definition`, either half `None`
        when absent -- the two ids `automation_results.definition_id` may
        carry for results BELONGING to this one definition
        (`index_definitions_by_id`'s own caveat: a locally-created result
        carries the local id, a server-mirrored one carries the server's)."""
        local_id = str(definition.get("id") or "") or None
        server_id = definition.get("server_id")
        return local_id, (str(server_id) if server_id else None)

    def _definition_results_query(
        self, service: "SchedulingService", definition: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], int]:
        """`(results, total)` for `definition` alone, across BOTH its id
        spaces -- the definition-filtered pushed Results view's listing
        (redesign PR-4 task 2). Capped at `RESULTS_INBOX_LIMIT`, same as
        the global inbox (the brief: "preserve the cap-line honesty").

        `ScheduledTasksDB.list_automation_results`/`count_automation_
        results` only equality-filter ONE `definition_id` at a time (own
        docstring's caveat), so a definition with results in both spaces
        needs two queries, merged here and re-sorted by REAL instant
        (`_result_sort_key`) rather than trusting either query's own
        newest-first order to interleave correctly across the merge.
        """
        local_id, server_id = self._definition_id_space(definition)
        merged: dict[str, dict[str, Any]] = {}
        total = 0
        for definition_id in (local_id, server_id):
            if not definition_id:
                continue
            total += service.db.count_automation_results(
                owner_id=None, definition_id=definition_id
            )
            for row in service.db.list_automation_results(
                owner_id=None, definition_id=definition_id, limit=RESULTS_INBOX_LIMIT
            ):
                merged[row["id"]] = row
        results = sorted(merged.values(), key=_result_sort_key, reverse=True)
        return results[:RESULTS_INBOX_LIMIT], total

    def _definition_unread_result_ids(
        self, service: "SchedulingService", definition: dict[str, Any]
    ) -> list[str]:
        """Definition-scoped counterpart of `_unread_result_ids` -- every
        unread result id for THIS definition alone (both id spaces),
        uncapped by `RESULTS_INBOX_LIMIT` for the same Qodo-HIGH reason
        that method's own docstring documents."""
        local_id, server_id = self._definition_id_space(definition)
        ids: list[str] = []
        for definition_id in (local_id, server_id):
            if not definition_id:
                continue
            unread_total = service.db.count_unread_results(
                owner_id=None, definition_id=definition_id
            )
            if not unread_total:
                continue
            results = service.db.list_automation_results(
                owner_id=None,
                definition_id=definition_id,
                review_state="unread",
                limit=unread_total,
            )
            ids.extend(result["id"] for result in results)
        return ids

    async def _dispatch_mark_all_results_read(
        self, service: "SchedulingService", unread_ids: list[str]
    ) -> None:
        """Per-row fan-out for a batch of result ids -- the shared
        `mark_results_read` (results_tab.py) does the actual DB calls +
        notify, and also drives the pushed view's own `a` binding
        (`ResultsHostScreen`, redesign PR-4 task 2). The Queue refresh is
        included here rather than in a caller (final review F5): it used
        to sit in the rail button's wrapper as if it were a rail-specific
        nicety, so marking everything read from anywhere else left the
        Queue's unread dots painted and its rail button visible, and
        pressing that button then reported "Nothing unread."
        """
        await mark_results_read(service, unread_ids, self._notify)
        self._refresh_results_surfaces()

    def action_mark_all_results_read(self) -> None:
        """`a` / the rail's `Mark all read` button: mark every unread
        result read, across the FULL table (not just a loaded window --
        `_unread_result_ids`' own Qodo-HIGH contract).

        redesign PR-4, task 5: this used to be Results-tab-gated, with an
        UNGATED byte-identical twin (`_rail_mark_all_read`) behind the
        rail button for exactly the reachability the gate removed. The
        tab is retired, so the gate is retired and the two collapse into
        this one method -- the rail button is now the only on-screen
        target for it, and it is always reachable. The pushed results
        view keeps its own `a` (`ResultsHostScreen`), scoped to whatever
        that push is showing.
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
        """Degrade the side panes instead of clipping them.

        redesign PR-4, task 6 (spec §11, ruling 6): below 84 columns the
        detail region no longer just disappears behind a "widen the
        window" apology -- the queue takes the full width and `Enter` on
        a row PUSHES the same pane class full-screen
        (`_on_task_row_selected`), so every operation stays reachable at
        the 80x24 floor. The inspector, which is a read-only summary of
        what the detail pane already shows, still simply yields.
        """
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
        # the screen header already names this pane, so the in-pane title
        # yields its row to the table + notice, the chip row collapses to
        # its cycling control, and the queue rail takes the whole width
        # (see _scheduling.tcss).
        self.query_one("#scheduling-workbench").set_class(hide_detail, "compact")
        self._sync_resize_notice()
        self._update_pane_notice()

    def _sync_resize_notice(self) -> None:
        """Recompute the width-driven half of the queue-pane notice.

        Reads the classes `on_resize` set rather than the width, and is
        called from `_update_pane_notice` as well as from the resize
        itself: the compact copy branches on whether the queue has any
        tasks, which a resize handler alone cannot keep current (a
        first-run screen mounts, resizes with zero tasks, and only then
        loads them -- redesign PR-4, task 6).
        """
        if self._detail_hidden():
            self._resize_notice = (
                "Press n to schedule your first task."
                if not self._tasks
                else "Press Enter on a row to open its details."
            )
        elif self._inspector_hidden():
            self._resize_notice = "Inspector hidden — widen the window to see it."
        else:
            self._resize_notice = ""

    def _inspector_hidden(self) -> bool:
        """True when `on_resize` has hidden the inspector pane."""
        try:
            return self.query_one("#scheduling-inspector-pane").has_class(
                "pane-hidden"
            )
        except Exception:  # noqa: BLE001 - not mounted yet
            return False

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
        self._sync_resize_notice()
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
        self._refresh_conflicts_badge()

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
        # UAT finding 3c: `outcome.status`/`.error` now describe ONLY the
        # reminder phase -- an automation phase (definition push/pull,
        # results pull) that failed in the same cycle no longer collapses
        # this into a `SyncFailed` "Sync failed: ..." toast (that would
        # lie about a reminder phase, or an unrelated definition push,
        # that genuinely succeeded). Its own truth still has to reach the
        # user, as its own notice rather than silently dropped.
        phase_errors = tuple(getattr(outcome, "phase_errors", ()) or ())
        if phase_errors:
            self.app_instance.notify(
                "Sync completed with issues — " + "; ".join(phase_errors),
                severity="warning",
            )
        self._refresh_owner_select()
        self._request_tasks_refresh()
        self._refresh_conflicts_badge()
        self._refresh_results_badge()

    @on(SyncFailed)
    def _on_sync_failed(self, event: SyncFailed) -> None:
        self._sync_running = False
        self.app_instance.notify(f"Sync failed: {event.error}", severity="error")
        self._refresh_owner_select()
        self._request_tasks_refresh()
        self._refresh_conflicts_badge()
        self._refresh_results_badge()

    @on(ConflictsTab.ConflictResolved)
    def _on_conflict_resolved(self, event: ConflictsTab.ConflictResolved) -> None:
        """Reload the queue when a conflict is resolved.

        The `ConflictsTab` posting this lives on a pushed
        `WorkbenchHostScreen` (task 5 retired the mounted one), so the
        message reaches this handler only through that push's
        `route_message` relay -- see `_route_conflicts_message`.
        """
        self._request_tasks_refresh(refresh_definitions=False)
        self._refresh_conflicts_badge()

    def _refresh_conflicts_badge(self) -> None:
        """Re-count this owner's reminder sync conflicts onto the status
        strip's badge (UX-063's count, plan ruling 4's home).

        redesign PR-4 task 5: this used to ALSO populate the mounted
        Conflicts tab and relabel its tab (`_set_tab_label`, deleted with
        the `TabbedContent`). The conflicts view is pushed now
        (`_push_conflicts_overlay`, task 1), pre-read at push time from
        this same `get_conflicts` shape, so only the badge is left to
        keep current -- and it is the affordance that opens the view.
        """
        service = self._service()
        if service is None:
            return
        conflicts = service.db.get_conflicts(
            service.owner_id, primitive="reminder_task"
        )
        label = f"Conflicts ({len(conflicts)})" if conflicts else "Conflicts"
        try:
            self.query_one("#scheduling-conflicts-badge", Button).label = label
        except Exception:  # noqa: BLE001 - strip not mounted yet
            pass

    def _refresh_results_badge(self) -> None:
        """Re-count unread results onto the rail's `Results` button
        (schedules-handoff PR-6 task 3's badge, redesign PR-4 task 2's
        home). Direct `service.db.*` calls (`count_unread_results` spans
        every owner -- Task 1), no worker: a local DB-only read, same
        cost class as `get_conflicts`. Called after Task 4's
        notification-triggered pull and after every results mutation.

        redesign PR-4 task 5: this used to also list `RESULTS_INBOX_LIMIT`
        rows and populate the mounted Results tab. The pushed results view
        (task 2) runs that listing itself, at push time and after each of
        its own mutations, so the listing here had no consumer left -- the
        count is all that is mirrored now, and it is one query instead of
        four.
        """
        service = self._service()
        if service is None:
            return
        unread = service.db.count_unread_results(owner_id=None)
        label = f"Results ({unread})" if unread else "Results"
        try:
            self.query_one("#scheduling-results-badge", Button).label = label
        except Exception:  # noqa: BLE001 - rail not mounted yet
            pass

    def _refresh_results_surfaces(self) -> None:
        """Every surface a results MUTATION moves: the rail's `Results`
        count and the Queue's own unread dots + `Mark all read` visibility.

        Final review F5: the unread affordances Task 3 added to the Queue
        derive from `UnifiedRow.unread_count`, which is only recomputed by
        `load_tasks` -- so an SSE-triggered pull, a read/dismiss, a
        mark-solved or a mark-all-read updated the results surface and
        left the Queue's dots (and the rail button, gated on
        `sum(row.unread_count) > 0`) stale in both directions: hidden
        while unread work existed, or visible with nothing left to mark.
        `refresh_definitions=False` -- results never change which
        definitions exist, and results are re-read on every load anyway.
        Called from the mutation paths only, never from the mount/reload
        paths that already run their own `_request_tasks_refresh`.
        """
        self._refresh_results_badge()
        self._request_tasks_refresh(refresh_definitions=False)

    def action_delete(self) -> None:
        """Delete marked tasks in bulk, else the selected one (confirmed).

        While ANY mark exists, d never falls through to the highlighted,
        unmarked row (task-23107 review F1): acting on a row the user
        never marked is worse than refusing.

        redesign PR-4, task 5: the Results-tab "dismiss the selected
        result" branch retired with the tab -- the pushed results view
        (`ResultsHostScreen`) owns its own `d`, and a screen underneath
        never receives the key anyway.
        """
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
        F8). Say what the row does not support instead.

        redesign PR-4 task 5: this used to point at the Automations tab
        ("managed on the Automations tab for now"), which is retired --
        and by task 3 the pointer was already wrong for the verbs that
        DO reach a definition row (run/edit/pause/move/mark-read are all
        routed by kind before this is ever consulted). What is left is
        the genuinely reminder-only half -- delete/mark/enable -- plus
        the rare cursor-divergence fallthrough, so the copy names the
        limit rather than a place to go.
        """
        if (self._selected_row_id or "").startswith("definition:"):
            return (
                f"Automations don't support {verb} — use the actions in "
                "this automation's own detail pane."
            )
        return f"Nothing to {verb} — select a task first."

    def action_edit_task(self) -> None:
        """Open the highlighted task/definition in its edit form (e key).

        Routes by row kind (redesign PR-4 task 5 dropped the Automations
        tab branch with the tab): a reminder row opens the existing
        reminder edit flow; a definition row opens `AutomationDefinition
        Form` pre-filled via `_edit_selected_automation` (task 3's
        edit-in-full), which refuses honestly for a
        non-`recurring_question` row.
        """
        task = self._selected_task()
        if task is not None:
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
            return
        definition = self._selected_queue_definition()
        if definition is not None:
            self._edit_selected_automation(definition)
            return
        self.app_instance.notify(
            self._no_task_notice("edit"),
            severity="warning",
        )

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
