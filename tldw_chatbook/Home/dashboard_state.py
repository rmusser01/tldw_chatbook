"""Pure Home dashboard state and next-best-action selection."""

from __future__ import annotations

from dataclasses import dataclass


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
    has_library_content: bool = False
    has_recent_work: bool = False
    active_detail_route: str = "chat"


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
    if state.pending_approval_count:
        return HomeAction(
            "review_approvals",
            "Review pending approvals",
            "chat",
            "Agent work is waiting for a decision.",
        )
    if state.failed_schedule_count:
        return HomeAction(
            "recover_schedules",
            "Review failed schedules",
            "schedules",
            "Scheduled work needs recovery.",
        )
    if state.active_run_count:
        return HomeAction(
            "resume_active_work",
            "Resume active work",
            "chat",
            "Live work is already running.",
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


def build_home_controls(state: HomeDashboardInput) -> tuple[HomeControl, ...]:
    controls: list[HomeControl] = []
    if state.pending_approval_count:
        controls.extend(
            (
                HomeControl("home-approve", "Approve", "chat", "approval"),
                HomeControl("home-reject", "Reject", "chat", "approval"),
            )
        )
    if state.running_run_count or state.active_run_count:
        controls.append(HomeControl("home-pause", "Pause", "chat", "running_work"))
    if state.paused_run_count:
        controls.append(HomeControl("home-resume", "Resume", "chat", "paused_work"))
    if state.failed_run_count or state.failed_schedule_count:
        controls.append(HomeControl("home-retry", "Retry", "schedules", "failed_work"))
    if (
        state.pending_approval_count
        or state.active_run_count
        or state.running_run_count
        or state.paused_run_count
        or state.failed_run_count
        or state.failed_schedule_count
    ):
        controls.extend(
            (
                HomeControl("home-open-details", "Open details", state.active_detail_route, "work_details"),
                HomeControl("home-open-in-console", "Open in Console", "chat", "console"),
            )
        )
    return tuple(controls)


def summarize_home_dashboard(state: HomeDashboardInput) -> HomeDashboard:
    next_action = choose_next_best_action(state)
    approval_label = "Approval required" if state.pending_approval_count else "Ready"
    return HomeDashboard(
        next_action=next_action,
        sections=(
            HomeSection("status", "Status", (f"Model: {'Ready' if state.model_ready else 'Blocked'}",)),
            HomeSection(
                "attention",
                "Attention",
                (approval_label, f"Pending approvals: {state.pending_approval_count}"),
            ),
            HomeSection(
                "active_work",
                "Active Work",
                (
                    f"Running: {state.running_run_count}",
                    f"Paused: {state.paused_run_count}",
                    f"Failed: {state.failed_run_count}",
                ),
            ),
            HomeSection("next_best_action", "Next Best Action", (next_action.label, next_action.reason)),
            HomeSection(
                "recent_work",
                "Recent Work",
                ("Recent work available" if state.has_recent_work else "No recent work yet",),
            ),
        ),
        controls=build_home_controls(state),
    )
