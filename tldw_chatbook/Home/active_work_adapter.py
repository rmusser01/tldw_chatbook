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


class HomeControlResultStatus(StrEnum):
    HANDLED = "handled"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class HomeControlResult:
    action: HomeControlAction
    status: HomeControlResultStatus | str
    message: str
    severity: str = "information"
    recovery_route: str = "chat"


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

    def handle_control(self, action: HomeControlAction) -> HomeControlResult:
        """Handle a lightweight Home control action."""
        ...


class UnavailableHomeActiveWorkAdapter:
    """Honest default adapter until active-run services are wired."""

    _ACTION_LABELS = {
        HomeControlAction.APPROVE: "Approve",
        HomeControlAction.REJECT: "Reject",
        HomeControlAction.PAUSE: "Pause",
        HomeControlAction.RESUME: "Resume",
        HomeControlAction.RETRY: "Retry",
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

    def handle_control(self, action: HomeControlAction) -> HomeControlResult:
        label = self._ACTION_LABELS[action]
        return HomeControlResult(
            action=action,
            status=HomeControlResultStatus.UNAVAILABLE,
            message=(
                f"{label} is not connected to an active run service yet. "
                "Open details or Console to inspect the work."
            ),
            severity="warning",
            recovery_route="chat",
        )
