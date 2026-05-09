"""Pure Home dashboard state and next-best-action selection."""

from __future__ import annotations

from dataclasses import dataclass


_APPROVAL_STATUSES = frozenset({"approval_required", "pending_approval", "pending"})
_RUNNING_STATUSES = frozenset({"running", "queued", "active"})
_PAUSED_STATUSES = frozenset({"paused"})
_FAILED_STATUSES = frozenset({"failed", "error"})


@dataclass(frozen=True)
class HomeActiveWorkItem:
    item_id: str
    title: str
    source: str
    status: str
    detail_route: str = "chat"
    console_available: bool = False


@dataclass(frozen=True)
class HomeDashboardInput:
    model_ready: bool = False
    mcp_ready: bool = True
    acp_ready: bool = True
    rag_ready: bool = False
    pending_approval_count: int = 0
    active_run_count: int = 0
    running_run_count: int = 0
    paused_run_count: int = 0
    failed_run_count: int = 0
    failed_schedule_count: int = 0
    notification_count: int = 0
    has_library_content: bool = False
    has_recent_work: bool = False
    active_detail_route: str = "chat"
    active_work_items: tuple[HomeActiveWorkItem, ...] = ()


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
            "llm",
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
    return tuple(controls)


def summarize_home_dashboard(state: HomeDashboardInput) -> HomeDashboard:
    next_action = choose_next_best_action(state)
    approval_label = "Approval required" if _pending_approval_count(state) else "Ready"
    active_count = _active_run_count(state)
    approval_count = _pending_approval_count(state)
    status_summary = (
        f"Model: {'Ready' if state.model_ready else 'Blocked'} | "
        f"RAG: {'Ready' if state.rag_ready else 'Missing sources'} | "
        f"MCP: {'Ready' if state.mcp_ready else 'Blocked'} | "
        f"ACP: {'Ready' if state.acp_ready else 'Blocked'} | "
        f"Active: {active_count} | "
        f"Approvals: {approval_count}"
    )
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
