"""Home active-work adapter contract and default unavailable implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol, runtime_checkable

from .dashboard_state import HomeDashboardInput


class HomeControlAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    PAUSE = "pause"
    RESUME = "resume"
    RETRY = "retry"
    OPEN_DETAILS = "open_details"
    OPEN_IN_CONSOLE = "open_in_console"


class HomeControlResultStatus(StrEnum):
    HANDLED = "handled"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class HomeConsoleLaunch:
    source: str
    title: str
    payload: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class HomeControlResult:
    action: HomeControlAction
    status: HomeControlResultStatus
    message: str
    severity: str = "information"
    recovery_route: str = "chat"
    target_id: str | None = None
    target_route: str | None = None
    console_launch: HomeConsoleLaunch | None = None


@runtime_checkable
class HomeActiveWorkAdapter(Protocol):
    """Adapter boundary for Home dashboard state and lightweight controls."""

    def build_dashboard_input(
        self,
        *,
        providers_models: Mapping[str, Any],
        has_recent_work: bool,
    ) -> HomeDashboardInput:
        """Return dashboard input from the current active-work backend."""
        ...

    def handle_control(
        self,
        action: HomeControlAction,
        *,
        target_id: str | None = None,
        target_route: str | None = None,
    ) -> HomeControlResult:
        """Handle a lightweight Home control action."""
        ...


class UnavailableHomeActiveWorkAdapter(HomeActiveWorkAdapter):
    """Honest default adapter until active-run services are wired."""

    _ACTION_LABELS = {
        HomeControlAction.APPROVE: "Approve",
        HomeControlAction.REJECT: "Reject",
        HomeControlAction.PAUSE: "Pause",
        HomeControlAction.RESUME: "Resume",
        HomeControlAction.RETRY: "Retry",
        HomeControlAction.OPEN_DETAILS: "Open details",
        HomeControlAction.OPEN_IN_CONSOLE: "Open in Console",
    }

    def build_dashboard_input(
        self,
        *,
        providers_models: Mapping[str, Any],
        has_recent_work: bool,
    ) -> HomeDashboardInput:
        return HomeDashboardInput(
            model_ready=bool(providers_models),
            pending_approval_count=0,
            active_run_count=0,
            running_run_count=0,
            paused_run_count=0,
            failed_run_count=0,
            failed_schedule_count=0,
            has_library_content=False,
            has_recent_work=has_recent_work,
            active_detail_route="chat",
        )

    def handle_control(
        self,
        action: HomeControlAction,
        *,
        target_id: str | None = None,
        target_route: str | None = None,
    ) -> HomeControlResult:
        label = self._ACTION_LABELS[action]
        recovery_route = target_route or "chat"
        return HomeControlResult(
            action=action,
            status=HomeControlResultStatus.UNAVAILABLE,
            message=(
                f"{label} is not connected to an active run service yet. "
                "Open details or Console to inspect the work."
            ),
            severity="warning",
            recovery_route=recovery_route,
            target_id=target_id,
        )


class LocalNotificationHomeActiveWorkAdapter(UnavailableHomeActiveWorkAdapter):
    """Home adapter that exposes local notification queue state.

    Active run controls remain unavailable until workflow/run services expose a
    safe synchronous Home contract.
    """

    def __init__(self, *, notification_service: Any | None = None):
        self.notification_service = notification_service

    def build_dashboard_input(
        self,
        *,
        providers_models: Mapping[str, Any],
        has_recent_work: bool,
    ) -> HomeDashboardInput:
        dashboard_input = super().build_dashboard_input(
            providers_models=providers_models,
            has_recent_work=has_recent_work,
        )
        return HomeDashboardInput(
            model_ready=dashboard_input.model_ready,
            mcp_ready=dashboard_input.mcp_ready,
            acp_ready=dashboard_input.acp_ready,
            rag_ready=dashboard_input.rag_ready,
            pending_approval_count=dashboard_input.pending_approval_count,
            active_run_count=dashboard_input.active_run_count,
            running_run_count=dashboard_input.running_run_count,
            paused_run_count=dashboard_input.paused_run_count,
            failed_run_count=dashboard_input.failed_run_count,
            failed_schedule_count=dashboard_input.failed_schedule_count,
            notification_count=self._unread_notification_count(),
            has_library_content=dashboard_input.has_library_content,
            has_recent_work=dashboard_input.has_recent_work,
            active_detail_route=dashboard_input.active_detail_route,
            active_work_items=dashboard_input.active_work_items,
        )

    def _unread_notification_count(self) -> int:
        if self.notification_service is None:
            return 0
        try:
            notifications = self.notification_service.list_queue(
                limit=100,
                include_dismissed=False,
                category=None,
            )
        except Exception:
            return 0
        return sum(1 for notification in notifications if _notification_is_unread(notification))


def _notification_is_unread(notification: Any) -> bool:
    if isinstance(notification, Mapping):
        return not bool(notification.get("is_read"))
    return not bool(getattr(notification, "is_read", False))
