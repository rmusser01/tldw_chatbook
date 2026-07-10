"""Pure Home dashboard state and next-best-action selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from tldw_chatbook.Constants import TAB_SETTINGS
from tldw_chatbook.Workspaces.conversation_browser_state import format_console_relative_age


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

APPROVAL_STATUSES = frozenset({"approval_required", "pending_approval", "pending"})
RUNNING_STATUSES = frozenset({"running", "queued", "active", "scheduled"})
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


def choose_next_best_action(state: HomeDashboardInput) -> HomeAction:
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
    if _failed_run_count(state):
        return HomeAction(
            "review_failed_work",
            "Review failed work",
            failed_item.detail_route if failed_item else state.active_detail_route,
            "Failed work needs recovery.",
        )
    if _active_run_count(state):
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


def build_home_controls(state: HomeDashboardInput) -> tuple[HomeControl, ...]:
    controls: list[HomeControl] = []
    approval_item = _first_item_for_status(state, _APPROVAL_STATUSES)
    running_item = _first_item_for_status(state, _RUNNING_STATUSES)
    paused_item = _first_item_for_status(state, _PAUSED_STATUSES)
    failed_item = _first_item_for_status(state, _FAILED_STATUSES)
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
    if _running_run_count(state) or (not state.active_work_items and state.active_run_count):
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
    if _failed_run_count(state) or _failed_schedule_count(state):
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
    if state.flashcards_due_count > 0:
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
                status_category="due",
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
        if selected.row_id == HOME_FLASHCARDS_DUE_ROW_ID:
            # Synthetic row: no backing HomeActiveWorkItem to look up.
            canvas = HomeCanvasState(
                title=f"Flashcards due: {state.flashcards_due_count}",
                lines=(
                    "Source: Library \u00b7 Status: due for review",
                    "Route: study",
                ),
                actions=build_home_controls(state),
                next_action=next_action,
                next_action_is_canvas=False,
            )
        else:
            item = next(
                i
                for i in tuple(state.active_work_items) + tuple(state.recent_work_items)
                if i.item_id == selected.row_id
            )
            canvas = HomeCanvasState(
                title=item.title,
                lines=(
                    f"Source: {item.source} \u00b7 Status: {item.status}",
                    f"{selected.glyph} {selected.status_category or 'item'}"
                    + (f" since {selected.age_label}" if selected.age_label else ""),
                    f"Route: {item.detail_route}",
                ),
                actions=build_home_controls(state),
                next_action=next_action,
                next_action_is_canvas=False,
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
