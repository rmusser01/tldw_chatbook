"""Pure Home dashboard state and next-best-action selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from tldw_chatbook.Constants import TAB_LLM, TAB_SETTINGS, TAB_STUDY, get_tab_display_label
from tldw_chatbook.UI.Navigation.shell_destinations import (
    get_shell_destination,
    resolve_shell_route,
)
from tldw_chatbook.Workspaces.conversation_browser_state import format_console_relative_age


# C1: human-readable labels for the Home canvas "Opens: <label>" line.
# `shell_destinations` is a leaf module (no imports back into `Home` or
# `UI.Screens`), so this stays a pure, cycle-free lookup. Overrides win over
# the generic shell-destination resolution for routes whose canonical
# destination label would otherwise be misleading (e.g. "study" resolves to
# the Library destination via its legacy-route alias, which would read as
# "Opens: Library" instead of the more specific "Opens: Study").
HOME_ROUTE_LABEL_OVERRIDES = {
    "llm": get_tab_display_label(TAB_LLM),
    TAB_LLM: get_tab_display_label(TAB_LLM),
    "search-rag": "Search/RAG",
    "study": get_tab_display_label(TAB_STUDY),
}

# M1: cap on the failed-item canvas's status-detail (failure reason) line,
# so one runaway/verbose error message can't blow out the canvas.
_STATUS_DETAIL_MAX_CHARS = 140


def _home_route_label(route: str) -> str:
    """Resolve a Home canvas ``detail_route`` to a human-readable label.

    Args:
        route: The raw route/destination id stored on a work item or rail
            row (e.g. ``"chat"``, ``"study"``, ``"watchlists"``).

    Returns:
        A human-facing label suitable for ``f"Opens: {label}"``.
    """
    route = route.strip()
    if route in HOME_ROUTE_LABEL_OVERRIDES:
        return HOME_ROUTE_LABEL_OVERRIDES[route]

    resolved = resolve_shell_route(route)
    try:
        return get_shell_destination(resolved.destination_id).accessible_label
    except KeyError:
        return route.replace("_", " ").replace("-", " ").title()


def _truncate_status_detail(text: str, *, limit: int = _STATUS_DETAIL_MAX_CHARS) -> str:
    """Truncate a canvas status-detail (failure reason) line to ``limit`` chars.

    Args:
        text: The (already-escaped) status detail text.
        limit: Maximum length of the returned string, ellipsis included.

    Returns:
        ``text`` unchanged when it already fits within ``limit``, otherwise
        the text cut short with a trailing "…" so the total length is
        exactly ``limit``.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


APPROVAL_RUN_STATUS = "approval"
FAILED_RUN_STATUS = "failed"
PAUSED_RUN_STATUS = "paused"
RUNNING_RUN_STATUS = "running"
UNKNOWN_RUN_STATUS = "unknown"

RUNTIME_SOURCE_LOCAL = "local"
RUNTIME_SOURCE_SERVER = "server"
SERVER_REACHABILITY_REACHABLE = "reachable"
SERVER_REACHABILITY_UNREACHABLE = "unreachable"
SERVER_AUTH_AUTHENTICATED = "authenticated"
SERVER_AUTH_REQUIRED = "auth_required"
SERVER_AUTH_SESSION_INVALID = "session_invalid"
SERVER_EVENT_STATE_AVAILABLE = "available"
SERVER_EVENT_STATE_EMPTY = "empty"
SERVER_EVENT_STATE_REQUERY_REQUIRED = "requery_required"
SERVER_EVENT_STATE_RECONNECT_REQUIRED = "reconnect_required"
SERVER_EVENT_STATE_UNAVAILABLE = "unavailable"

HOME_FLASHCARDS_DUE_ROW_ID = "home-flashcards-due"
HOME_FLASHCARDS_DUE_STATUS_CATEGORY = "due"

APPROVAL_STATUSES = frozenset({"approval_required", "pending_approval", "pending"})
# "parsing"/"writing" (F3): the Library ingest job registry's two active
# sub-states (replacing its old single "running" state -- see
# IngestJobState in library_ingest_jobs.py) map into this same shared
# "running" bucket, same as every other subsystem's "queued"/"active" here --
# this set is generic across every HomeActiveWorkItem source (workflows,
# watchlists, ACP, Library ingest, ...), not ingest-specific.
RUNNING_STATUSES = frozenset(
    {"running", "queued", "active", "scheduled", "parsing", "writing"}
)
PAUSED_STATUSES = frozenset({"paused"})
FAILED_STATUSES = frozenset({"failed", "error", "errored", "cancelled", "canceled"})

_APPROVAL_STATUSES = APPROVAL_STATUSES
_RUNNING_STATUSES = RUNNING_STATUSES
_PAUSED_STATUSES = PAUSED_STATUSES
_FAILED_STATUSES = FAILED_STATUSES


def categorize_run_status(value: object) -> str:
    """Map an active-work status token into a shared run-control category."""
    status = str(value or "").strip().lower()
    if status in APPROVAL_STATUSES:
        return APPROVAL_RUN_STATUS
    if status in FAILED_STATUSES:
        return FAILED_RUN_STATUS
    if status in PAUSED_STATUSES:
        return PAUSED_RUN_STATUS
    if status in RUNNING_STATUSES:
        return RUNNING_RUN_STATUS
    return UNKNOWN_RUN_STATUS


@dataclass(frozen=True)
class HomeActiveWorkItem:
    item_id: str
    title: str
    source: str
    status: str
    detail_route: str = "chat"
    console_available: bool = False
    updated_at: str = ""
    status_detail: str = ""
    # M4 (fix batch F1b): whether this item's own recovery control (Retry)
    # should be offered at all. ``True`` by default so every pre-existing
    # producer -- which never sets this -- keeps today's always-retryable
    # behavior. Library ingest jobs set it to ``not job.permanent`` for
    # FAILED items: a validation-class failure (unsupported file type,
    # missing source file) fails the same way on every retry, so Home
    # withholds the ``home-retry`` control for it (see
    # ``build_home_controls``) the same way the ingest canvas withholds its
    # own Retry button (``IngestQueueRow.can_retry``).
    retry_available: bool = True


@dataclass(frozen=True)
class HomeDashboardInput:
    model_ready: bool = False
    mcp_ready: bool = True
    acp_ready: bool = True
    rag_ready: bool = False
    runtime_source: str = RUNTIME_SOURCE_LOCAL
    active_server_id: str | None = None
    server_label: str | None = None
    server_configured: bool = False
    server_reachability: str = UNKNOWN_RUN_STATUS
    server_auth_state: str = UNKNOWN_RUN_STATUS
    pending_approval_count: int = 0
    active_run_count: int = 0
    running_run_count: int = 0
    paused_run_count: int = 0
    failed_run_count: int = 0
    failed_schedule_count: int = 0
    notification_count: int = 0
    server_event_count: int = 0
    server_event_state: str = SERVER_EVENT_STATE_UNAVAILABLE
    server_event_recovery: str = "Server event feed is unavailable."
    has_library_content: bool = False
    has_recent_work: bool = False
    active_detail_route: str = "chat"
    active_work_items: tuple[HomeActiveWorkItem, ...] = ()
    recent_work_items: tuple[HomeActiveWorkItem, ...] = ()
    flashcards_due_count: int = 0


@dataclass(frozen=True)
class HomeAction:
    action_id: str
    label: str
    target_route: str
    reason: str


@dataclass(frozen=True)
class HomeSection:
    section_id: str
    title: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class HomeControl:
    control_id: str
    label: str
    target_route: str
    applies_to: str
    target_id: str | None = None


@dataclass(frozen=True)
class HomeDashboard:
    next_action: HomeAction
    sections: tuple[HomeSection, ...]
    controls: tuple[HomeControl, ...]


def choose_next_best_action(
    state: HomeDashboardInput,
    *,
    exclude: frozenset[str] = frozenset(),
) -> HomeAction:
    """Pick the single highest-priority Home "Next" suggestion.

    Args:
        state: Adapter-provided dashboard input.
        exclude: ``action_id`` values to skip even when their branch would
            otherwise win (H3, fix batch F1b) -- used by the triage canvas
            builder to fall through to the next-best suggestion when the
            top one would just repeat the selected item's own recovery
            control (e.g. a failed item's canvas already offers Retry).
            Honored by the ``review_failed_work`` and ``resume_active_work``
            branches (the two the canvas builder needs to suppress: the
            latter's "Live work is already running." copy is false when
            nothing is actually running -- F1b whole-wave review). Empty by
            default, so every other caller is unaffected.
    """
    if not state.model_ready:
        return HomeAction(
            "fix_model_setup",
            "Set up Console model",
            TAB_SETTINGS,
            "Console needs a working model before live AI tasks.",
        )
    if _pending_approval_count(state):
        return HomeAction(
            "review_approvals",
            "Review pending approvals",
            "chat",
            "Agent work is waiting for a decision.",
        )
    if _failed_schedule_count(state):
        return HomeAction(
            "recover_schedules",
            "Review failed schedules",
            "schedules",
            "Scheduled work needs recovery.",
        )
    failed_item = _first_item_for_status(state, _FAILED_STATUSES)
    if _failed_run_count(state) and "review_failed_work" not in exclude:
        return HomeAction(
            "review_failed_work",
            "Review failed work",
            failed_item.detail_route if failed_item else state.active_detail_route,
            "Failed work needs recovery.",
        )
    if _active_run_count(state) and "resume_active_work" not in exclude:
        return HomeAction(
            "resume_active_work",
            "Resume active work",
            "chat",
            "Live work is already running.",
        )
    if state.notification_count:
        return HomeAction(
            "review_notifications",
            "Review notifications",
            "subscriptions",
            "Unread notifications need review.",
        )
    if not state.has_library_content:
        return HomeAction(
            "import_sources",
            "Import Library sources",
            "library",
            "Library content makes Console and RAG more useful.",
        )
    if state.rag_ready:
        return HomeAction(
            "search_library",
            "Search your Library",
            "library",
            "Search/RAG is ready over saved content.",
        )
    return HomeAction("start_console", "Start in Console", "chat", "Console is ready for a task.")


def choose_home_selected_item(state: HomeDashboardInput) -> HomeActiveWorkItem | None:
    """Choose the Home inspector/default details target using control priority."""
    return (
        _first_item_for_status(state, _APPROVAL_STATUSES)
        or _first_item_for_status(state, _FAILED_STATUSES)
        or _first_item_for_status(state, _RUNNING_STATUSES)
        or _first_item_for_status(state, _PAUSED_STATUSES)
        or (state.active_work_items[0] if state.active_work_items else None)
    )


def build_home_controls(
    state: HomeDashboardInput,
    *,
    selected_row_id: str = "",
    selected_item: HomeActiveWorkItem | None = None,
) -> tuple[HomeControl, ...]:
    """Build the Home canvas's control set.

    Args:
        state: Adapter-provided dashboard input.
        selected_row_id: The currently selected Home rail row id, or ""
            when nothing is selected (the count-only fallback path).
            Threaded through only from ``build_home_triage_state`` (H2,
            fix batch F1b) so the global "Review flashcards" shortcut can
            be scoped to "no real item selected": it is a global shortcut,
            not the selected item's own control, so it has no business
            sitting on a real work item's canvas next to that item's own
            controls (Retry, Approve, ...). Callers that don't have a
            selection concept (``summarize_home_dashboard``, and the
            triage builder's own count-only fallback) simply omit this and
            keep today's unconditional-when-due behavior.
        selected_item: The resolved ``HomeActiveWorkItem`` behind
            ``selected_row_id``, when the selection is a real work item
            (M4, fix batch F1b) -- threaded through the same way H2 threads
            ``selected_row_id``. When this item's own status is failed and
            its ``retry_available`` is ``False``, the ``home-retry``
            control is omitted entirely rather than pointing Retry at a
            failure that will just fail the same way again -- and, when
            failed, its route/target also replace whichever failed item
            ``_first_item_for_status`` would otherwise have picked, so
            Retry always reflects the selected item, not just "the first
            failed item in the list". ``None`` (the default) preserves
            today's behavior exactly for every caller without a selection
            concept (``summarize_home_dashboard``, the triage builder's own
            count-only fallback): Retry targets whichever failed item
            ``_first_item_for_status`` finds, unconditionally offered
            whenever any failed run/schedule exists.
    """
    controls: list[HomeControl] = []
    approval_item = _first_item_for_status(state, _APPROVAL_STATUSES)
    running_item = _first_item_for_status(state, _RUNNING_STATUSES)
    paused_item = _first_item_for_status(state, _PAUSED_STATUSES)
    failed_item = _first_item_for_status(state, _FAILED_STATUSES)
    selected_item_is_failed = (
        selected_item is not None and _normalized_status(selected_item) in _FAILED_STATUSES
    )
    if selected_item_is_failed:
        # The selected row's own failure -- not just "the first failed item
        # in the list" -- is what Retry (and its retry_available gate)
        # should reflect when a real item is selected.
        failed_item = selected_item
    selected_item_is_running = (
        selected_item is not None and _normalized_status(selected_item) in _RUNNING_STATUSES
    )
    if selected_item_is_running:
        # Same as the failed-item override above: when a running item is
        # selected, Pause's target should reflect *that* item rather than
        # just "the first running item in the list".
        running_item = selected_item
    # T154: Library ingest jobs (item_id "local:ingest:<job_id>") have no
    # wired pause action -- the ingest job registry has no pause state, only
    # queued/parsing/writing/done/failed (see library_ingest_jobs.py) -- so
    # Home must not offer a control with nothing behind it. Scoped to the
    # *selected* item, mirroring selected_item_is_failed above: with no
    # selection (summarize_home_dashboard, the triage builder's count-only
    # fallback) Pause keeps its unconditional-when-running behavior.
    selected_item_is_ingest_job = selected_item_is_running and _is_local_ingest_item(selected_item)
    chatbook_item = _first_chatbook_artifact_item(state)
    detail_item = choose_home_selected_item(state)

    if _pending_approval_count(state):
        controls.extend(
            (
                HomeControl(
                    "home-approve",
                    "Approve",
                    "chat",
                    "approval",
                    approval_item.item_id if approval_item else None,
                ),
                HomeControl(
                    "home-reject",
                    "Reject",
                    "chat",
                    "approval",
                    approval_item.item_id if approval_item else None,
                ),
            )
        )
    if (
        _running_run_count(state) or (not state.active_work_items and state.active_run_count)
    ) and not selected_item_is_ingest_job:
        controls.append(
            HomeControl(
                "home-pause",
                "Pause",
                "chat",
                "running_work",
                running_item.item_id if running_item else None,
            )
        )
    if _paused_run_count(state):
        controls.append(
            HomeControl(
                "home-resume",
                "Resume",
                "chat",
                "paused_work",
                paused_item.item_id if paused_item else None,
            )
        )
    # M4: only a *selected* failed item's own retry_available can withhold
    # Retry -- callers with no selection concept (summarize_home_dashboard,
    # the triage builder's own count-only fallback) keep today's
    # unconditional-when-failed behavior unchanged.
    retry_withheld = selected_item_is_failed and not selected_item.retry_available
    if (_failed_run_count(state) or _failed_schedule_count(state)) and not retry_withheld:
        failed_route = failed_item.detail_route if failed_item else "schedules"
        controls.append(
            HomeControl(
                "home-retry",
                "Retry",
                failed_route,
                "failed_work",
                failed_item.item_id if failed_item else None,
            )
        )
    if (
        _pending_approval_count(state)
        or _active_run_count(state)
        or _running_run_count(state)
        or _paused_run_count(state)
        or _failed_run_count(state)
        or _failed_schedule_count(state)
    ):
        controls.append(
            HomeControl(
                "home-open-details",
                "Open details",
                detail_item.detail_route if detail_item else state.active_detail_route,
                "work_details",
                detail_item.item_id if detail_item else None,
            )
        )
        if not state.active_work_items or any(item.console_available for item in state.active_work_items):
            console_item = (
                detail_item
                if detail_item is not None and detail_item.console_available
                else next((item for item in state.active_work_items if item.console_available), detail_item)
            )
            controls.append(
                HomeControl(
                    "home-open-in-console",
                    "Open in Console",
                    "chat",
                    "console",
                    console_item.item_id if console_item else None,
                )
            )
            if chatbook_item is not None and chatbook_item != console_item:
                controls.append(
                    HomeControl(
                        "home-open-chatbook-in-console",
                        "Open Chatbook in Console",
                        "chat",
                        "chatbook_console",
                        chatbook_item.item_id,
                    )
                )
        if chatbook_item is not None and chatbook_item != detail_item:
            controls.append(
                HomeControl(
                    "home-open-chatbook-details",
                    "Open Chatbook details",
                    chatbook_item.detail_route,
                    "chatbook_details",
                    chatbook_item.item_id,
                )
            )
    if state.flashcards_due_count > 0 and (
        not selected_row_id or selected_row_id == HOME_FLASHCARDS_DUE_ROW_ID
    ):
        controls.append(
            HomeControl(
                "home-review-flashcards",
                "Review flashcards",
                "study",
                "flashcards_due",
                None,
            )
        )
    return tuple(controls)


def _status_summary_line(state: HomeDashboardInput) -> str:
    return (
        f"Model: {'Ready' if state.model_ready else 'Blocked'} | "
        f"RAG: {'Ready' if state.rag_ready else 'Missing sources'} | "
        f"MCP: {'Ready' if state.mcp_ready else 'Blocked'} | "
        f"ACP: {'Ready' if state.acp_ready else 'Blocked'} | "
        f"Mode: {_runtime_source_label(state.runtime_source)} | "
        f"Server: {_server_status_label(state)} | "
        f"Active: {_active_run_count(state)} | "
        f"Approvals: {_pending_approval_count(state)}"
    )


def summarize_home_dashboard(state: HomeDashboardInput) -> HomeDashboard:
    next_action = choose_next_best_action(state)
    approval_label = "Approval required" if _pending_approval_count(state) else "Ready"
    active_count = _active_run_count(state)
    approval_count = _pending_approval_count(state)
    status_summary = _status_summary_line(state)
    return HomeDashboard(
        next_action=next_action,
        sections=(
            HomeSection("status", "Status", (status_summary,)),
            HomeSection(
                "attention",
                "Attention",
                (
                    approval_label,
                    f"Pending approvals: {approval_count}",
                    f"Unread notifications: {state.notification_count}",
                ),
            ),
            HomeSection(
                "active_work",
                "Active Work",
                _active_work_lines(state),
            ),
            HomeSection(
                "system_status",
                "System Status",
                _system_status_lines(state),
            ),
            HomeSection(
                "next_best_action",
                "Next Best Action",
                _next_action_lines(state, next_action),
            ),
            HomeSection(
                "recent_work",
                "Recent Work",
                (
                    (
                        "Recent work available",
                        "Open Console, Library, or Artifacts to resume.",
                    )
                    if state.has_recent_work
                    else (
                        "No recent work yet",
                        "Runs, chatbooks, imports, and schedules will appear here.",
                    )
                ),
            ),
        ),
        controls=build_home_controls(state),
    )


def _normalized_status(item: HomeActiveWorkItem) -> str:
    return item.status.strip().lower()


def _count_items_for_status(state: HomeDashboardInput, statuses: frozenset[str]) -> int:
    return sum(1 for item in state.active_work_items if _normalized_status(item) in statuses)


def _first_item_for_status(
    state: HomeDashboardInput,
    statuses: frozenset[str],
) -> HomeActiveWorkItem | None:
    return next((item for item in state.active_work_items if _normalized_status(item) in statuses), None)


def _is_local_ingest_item(item: HomeActiveWorkItem) -> bool:
    """True when ``item`` mirrors a Library ingest job.

    Uses the same ``local:ingest:<job_id>`` item_id marker that
    ``active_work_adapter._local_ingest_job_items`` stamps on ingest-mirrored
    ``HomeActiveWorkItem``s (see also that module's
    ``_is_local_ingest_job_id``). Duplicated rather than imported: this
    module is a leaf (``active_work_adapter`` imports *from* it), so it
    cannot import back without a cycle.
    """
    return item.item_id.startswith("local:ingest:")


def _first_chatbook_artifact_item(state: HomeDashboardInput) -> HomeActiveWorkItem | None:
    return next(
        (
            item
            for item in state.active_work_items
            if item.item_id.startswith("local:chatbook:")
        ),
        None,
    )


def _pending_approval_count(state: HomeDashboardInput) -> int:
    return state.pending_approval_count or _count_items_for_status(state, _APPROVAL_STATUSES)


def _running_run_count(state: HomeDashboardInput) -> int:
    return state.running_run_count or _count_items_for_status(state, _RUNNING_STATUSES)


def _paused_run_count(state: HomeDashboardInput) -> int:
    return state.paused_run_count or _count_items_for_status(state, _PAUSED_STATUSES)


def _failed_run_count(state: HomeDashboardInput) -> int:
    return state.failed_run_count or _count_items_for_status(state, _FAILED_STATUSES)


def _failed_schedule_count(state: HomeDashboardInput) -> int:
    return state.failed_schedule_count


def _active_run_count(state: HomeDashboardInput) -> int:
    return state.active_run_count or len(state.active_work_items)


def _active_work_lines(state: HomeDashboardInput) -> tuple[str, ...]:
    if state.active_work_items:
        return tuple(
            f"{item.title} [{item.status}] via {item.source}"
            for item in state.active_work_items
        )
    return (
        f"Running: {state.running_run_count}",
        f"Paused: {state.paused_run_count}",
        f"Failed: {state.failed_run_count}",
    )


def _system_status_lines(state: HomeDashboardInput) -> tuple[str, ...]:
    source_label = _runtime_source_label(state.runtime_source)
    server_label = _server_status_label(state)
    active_count = _active_run_count(state)
    approval_count = _pending_approval_count(state)
    lines = [
        f"Runtime: {source_label}",
        f"Server sync: {server_label}",
        _runtime_explanation_line(state),
        (
            "Agent readiness: "
            f"Model {'ready' if state.model_ready else 'blocked'}, "
            f"RAG {'ready' if state.rag_ready else 'needs sources'}, "
            f"MCP {'ready' if state.mcp_ready else 'blocked'}, "
            f"ACP {'ready' if state.acp_ready else 'blocked'}"
        ),
        _server_event_status_line(state),
        f"Work: {active_count} active, {approval_count} approvals",
    ]
    return tuple(lines)


def _server_event_status_line(state: HomeDashboardInput) -> str:
    event_state = str(state.server_event_state or SERVER_EVENT_STATE_UNAVAILABLE).strip().lower()
    if event_state == SERVER_EVENT_STATE_AVAILABLE:
        return f"Server events: {state.server_event_count} observed via server event feed"
    if event_state == SERVER_EVENT_STATE_EMPTY:
        return "Server events: No observed server events"
    if event_state == SERVER_EVENT_STATE_REQUERY_REQUIRED:
        return "Server events: Replay gap - requery server events"
    if event_state == SERVER_EVENT_STATE_RECONNECT_REQUIRED:
        return "Server events: Reconnect required"
    return "Server events: Unavailable"


def _runtime_source_label(value: object) -> str:
    source = str(value or RUNTIME_SOURCE_LOCAL).strip().lower()
    return "Server" if source == RUNTIME_SOURCE_SERVER else "Local"


def _runtime_explanation_line(state: HomeDashboardInput) -> str:
    source = str(state.runtime_source or RUNTIME_SOURCE_LOCAL).strip().lower()
    reachability = str(state.server_reachability or UNKNOWN_RUN_STATUS).strip().lower()
    auth_state = str(state.server_auth_state or UNKNOWN_RUN_STATUS).strip().lower()
    if source != RUNTIME_SOURCE_SERVER:
        return "Local mode is active. Server sync is optional."
    if not state.server_configured or not state.active_server_id:
        return "Choose or configure a server before server-backed work."
    if reachability == SERVER_REACHABILITY_UNREACHABLE:
        return "Server is unreachable. Check network or server status."
    if auth_state == SERVER_AUTH_REQUIRED:
        return "Authentication is required before server-backed work."
    if auth_state == SERVER_AUTH_SESSION_INVALID:
        return "Authentication expired. Reconnect before server-backed work."
    if reachability == SERVER_REACHABILITY_REACHABLE and auth_state == SERVER_AUTH_AUTHENTICATED:
        return "Server mode is ready for authenticated work."
    return "Checking server readiness."


def _server_status_label(state: HomeDashboardInput) -> str:
    source = str(state.runtime_source or RUNTIME_SOURCE_LOCAL).strip().lower()
    reachability = str(state.server_reachability or UNKNOWN_RUN_STATUS).strip().lower()
    auth_state = str(state.server_auth_state or UNKNOWN_RUN_STATUS).strip().lower()
    if source != RUNTIME_SOURCE_SERVER:
        return "Configured; local mode" if state.server_configured else "Not configured (local mode)"
    if not state.server_configured or not state.active_server_id:
        return "Missing active server"
    if reachability == SERVER_REACHABILITY_UNREACHABLE:
        return "Unreachable"
    if auth_state == SERVER_AUTH_REQUIRED:
        return "Auth required"
    if auth_state == SERVER_AUTH_SESSION_INVALID:
        return "Auth expired"
    if reachability == SERVER_REACHABILITY_REACHABLE and auth_state == SERVER_AUTH_AUTHENTICATED:
        return "Ready"
    return "Checking"


def _next_action_lines(
    state: HomeDashboardInput,
    next_action: HomeAction,
) -> tuple[str, ...]:
    lines = [next_action.label]
    if state.has_recent_work:
        lines.append("Review recent work")
    for label in ("Open Console", "Configure RAG"):
        if label == "Open Console" and "Console" in next_action.label:
            continue
        if label not in lines:
            lines.append(label)
    return tuple(lines[:3])


@dataclass(frozen=True)
class HomeRailRow:
    """One selectable row in the Home triage rail."""

    row_id: str
    section_id: str
    glyph: str
    title: str
    age_label: str
    source: str = ""
    status_category: str = ""
    detail_route: str = "chat"


@dataclass(frozen=True)
class HomeRailSectionState:
    """One Home triage rail section with its rows and empty copy."""

    section_id: str
    title: str
    count: int
    rows: tuple[HomeRailRow, ...]
    empty_copy: str


@dataclass(frozen=True)
class HomeCanvasState:
    """The Home focus canvas for the selected row or the next best action."""

    title: str
    lines: tuple[str, ...]
    actions: tuple[HomeControl, ...]
    next_action: HomeAction
    next_action_is_canvas: bool
    primary_control_id: str = ""


@dataclass(frozen=True)
class HomeTriageState:
    """Full Home triage display state: header, rail sections, canvas."""

    header_line: str
    sections: tuple[HomeRailSectionState, ...]
    details_lines: tuple[str, ...]
    canvas: HomeCanvasState
    selected_row_id: str


_CATEGORY_GLYPHS = {
    APPROVAL_RUN_STATUS: "\u25cf",
    FAILED_RUN_STATUS: "\u25cf",
    PAUSED_RUN_STATUS: "\u25cb",
    RUNNING_RUN_STATUS: "\u25cf",
    UNKNOWN_RUN_STATUS: "\u25cb",
}
_ATTENTION_CATEGORIES = frozenset({APPROVAL_RUN_STATUS, FAILED_RUN_STATUS})
_RUNNING_CATEGORIES = frozenset({RUNNING_RUN_STATUS, PAUSED_RUN_STATUS})


def _item_row(item: HomeActiveWorkItem, section_id: str, now: datetime) -> HomeRailRow:
    category = categorize_run_status(item.status)
    return HomeRailRow(
        row_id=item.item_id,
        section_id=section_id,
        glyph=_CATEGORY_GLYPHS.get(category, "\u25cb"),
        title=item.title,
        age_label=format_console_relative_age(item.updated_at, now=now),
        source=item.source,
        status_category=category,
        detail_route=item.detail_route,
    )


def _header_line(state: HomeDashboardInput) -> str:
    readiness = "Ready" if state.model_ready else "Blocked"
    source = str(state.runtime_source or RUNTIME_SOURCE_LOCAL).strip().lower()
    if source == RUNTIME_SOURCE_SERVER:
        label = str(state.server_label or "").strip()
        runtime = f"Server: {label}" if label else "Server"
    else:
        runtime = "Local"
    return f"Home | {readiness} \u00b7 {runtime}"


def _canvas_primary_control_id(
    category: str,
    controls: tuple[HomeControl, ...],
) -> str:
    """Pick which canvas control (if any) carries primary emphasis.

    Primary emphasis follows the selected row rather than sticking to one
    permanently-accented button: a failed item's Retry, an
    approval-pending item's Approve, or the synthetic flashcards-due row's
    Review flashcards. Anything else (running/paused/no selection) defers
    to Open details when it is present; otherwise nothing is primary.

    Args:
        category: The selected row's status category (``HomeRailRow.
            status_category``), or ``UNKNOWN_RUN_STATUS`` when nothing is
            selected.
        controls: The canvas's rendered controls.

    Returns:
        The control_id that should carry primary styling, or "" for none.
    """
    if category == HOME_FLASHCARDS_DUE_STATUS_CATEGORY:
        candidate = "home-review-flashcards"
    elif category == FAILED_RUN_STATUS:
        candidate = "home-retry"
    elif category == APPROVAL_RUN_STATUS:
        candidate = "home-approve"
    else:
        candidate = "home-open-details"
    return candidate if any(control.control_id == candidate for control in controls) else ""


def build_home_triage_state(
    state: HomeDashboardInput,
    *,
    selected_row_id: str = "",
    now: datetime | None = None,
) -> HomeTriageState:
    """Build the Home triage rail + canvas display state.

    Args:
        state: Adapter-provided dashboard input.
        selected_row_id: Explicit row selection; falls back to control
            priority (approval > failed > running > paused > first).
        now: Reference time for age labels (defaults to UTC now).

    Returns:
        Immutable triage state: header line, rail sections, details lines,
        and the canvas for the selected row (or the next best action when
        nothing is selectable).
    """
    reference_now = now or datetime.now(timezone.utc)
    attention_rows: list[HomeRailRow] = []
    running_rows: list[HomeRailRow] = []
    for item in state.active_work_items:
        category = categorize_run_status(item.status)
        if category in _ATTENTION_CATEGORIES:
            attention_rows.append(_item_row(item, "attention", reference_now))
        else:
            # Running, paused, and unknown/ready items all remain visible
            # as active work rather than silently dropping.
            running_rows.append(_item_row(item, "running", reference_now))
    recent_rows = tuple(
        _item_row(item, "recent", reference_now) for item in state.recent_work_items
    )

    if state.flashcards_due_count > 0:
        attention_rows.append(
            HomeRailRow(
                row_id=HOME_FLASHCARDS_DUE_ROW_ID,
                section_id="attention",
                glyph="●",
                title=f"Flashcards due: {state.flashcards_due_count}",
                age_label="",
                source="Library",
                status_category=HOME_FLASHCARDS_DUE_STATUS_CATEGORY,
                detail_route="study",
            )
        )

    sections = (
        HomeRailSectionState(
            "attention",
            "Needs Attention",
            len(attention_rows),
            tuple(attention_rows),
            "No approvals or failures pending.",
        ),
        HomeRailSectionState(
            "running",
            "Running",
            len(running_rows),
            tuple(running_rows),
            "Nothing running right now.",
        ),
        HomeRailSectionState(
            "recent",
            "Recent",
            len(recent_rows),
            recent_rows,
            "Runs, chatbooks, imports, and schedules will appear here.",
        ),
    )

    all_rows = {row.row_id: row for section in sections for row in section.rows}
    selected = all_rows.get(selected_row_id)
    if selected is None:
        fallback_item = choose_home_selected_item(state)
        selected = all_rows.get(fallback_item.item_id) if fallback_item else None
    next_action = choose_next_best_action(state)
    if selected is not None:
        # H2: thread the selection into build_home_controls so the global
        # "Review flashcards" shortcut is scoped out of a real work item's
        # canvas (kept for the synthetic flashcards row and for "nothing
        # selected", below).
        if selected.row_id == HOME_FLASHCARDS_DUE_ROW_ID:
            # Synthetic row: no backing HomeActiveWorkItem to look up.
            controls = build_home_controls(state, selected_row_id=selected.row_id)
            canvas = HomeCanvasState(
                title=f"Flashcards due: {state.flashcards_due_count}",
                lines=(
                    # L7: "Library" alone reads as a source/destination
                    # mismatch (flashcards live in Study, not Library) --
                    # name the actual feature while still crediting the
                    # Library-sourced due count.
                    f"{selected.glyph} due for review \u00b7 Study decks in Library",
                    f"Opens: {_home_route_label('study')}",
                ),
                actions=controls,
                next_action=next_action,
                next_action_is_canvas=False,
                primary_control_id=_canvas_primary_control_id(
                    HOME_FLASHCARDS_DUE_STATUS_CATEGORY, controls
                ),
            )
        else:
            item = next(
                i
                for i in tuple(state.active_work_items) + tuple(state.recent_work_items)
                if i.item_id == selected.row_id
            )
            # M4: thread the resolved item so build_home_controls can gate
            # home-retry on *this* item's own retry_available rather than
            # just "the first failed item in the list".
            controls = build_home_controls(
                state, selected_row_id=selected.row_id, selected_item=item
            )
            # H3: the Next hint must not repeat the selected item's own
            # recovery control (e.g. a failed item's canvas already offers
            # Retry) -- when the engine's top suggestion is exactly that,
            # for exactly this item's route, recompute with that branch
            # suppressed so it falls through to the next one.
            item_next_action = next_action
            if (
                next_action.action_id == "review_failed_work"
                and _normalized_status(item) in _FAILED_STATUSES
                and item.detail_route == next_action.target_route
            ):
                excluded_actions = {"review_failed_work"}
                if _running_run_count(state) == 0:
                    # (F1b whole-wave review, live QA) The fallthrough
                    # branch, resume_active_work, claims "Live work is
                    # already running." -- but _active_run_count counts
                    # failed/queued attention items too, so with nothing
                    # actually RUNNING that copy is false (the Running rail
                    # section says "Nothing running right now." right beside
                    # it). Exclude it as well and keep falling through.
                    excluded_actions.add("resume_active_work")
                item_next_action = choose_next_best_action(
                    state, exclude=frozenset(excluded_actions)
                )
            status_line = f"{selected.glyph} {item.status} \u00b7 {item.source}"
            if selected.age_label:
                status_line += f" \u00b7 since {selected.age_label}"
            lines = [status_line]
            if item.status_detail:
                # M1: the failure reason, as its own line right after the
                # status line -- truncated so one runaway error message
                # can't blow out the canvas.
                lines.append(_truncate_status_detail(item.status_detail))
            lines.append(f"Opens: {_home_route_label(item.detail_route)}")
            canvas = HomeCanvasState(
                title=item.title,
                lines=tuple(lines),
                actions=controls,
                next_action=item_next_action,
                next_action_is_canvas=False,
                primary_control_id=_canvas_primary_control_id(
                    selected.status_category, controls
                ),
            )
        selected_id = selected.row_id
    else:
        # Count-only inputs (no item list) still expose their controls so
        # approvals/retries remain reachable without a selectable row.
        controls = build_home_controls(state)
        canvas = HomeCanvasState(
            title=next_action.label,
            lines=(next_action.reason,),
            actions=controls,
            next_action=next_action,
            next_action_is_canvas=not controls,
            primary_control_id=_canvas_primary_control_id(UNKNOWN_RUN_STATUS, controls),
        )
        selected_id = ""

    details_lines = (_status_summary_line(state),) + _system_status_lines(state)
    if state.notification_count:
        details_lines = details_lines + (
            f"Notifications: {state.notification_count} unread",
        )
    return HomeTriageState(
        header_line=_header_line(state),
        sections=sections,
        details_lines=details_lines,
        canvas=canvas,
        selected_row_id=selected_id,
    )
