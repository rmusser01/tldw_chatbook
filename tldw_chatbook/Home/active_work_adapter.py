"""Home active-work adapter contract and default unavailable implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Mapping, Protocol, runtime_checkable

from loguru import logger

from .dashboard_state import HomeActiveWorkItem, HomeDashboardInput


_HOME_WATCHLIST_RUN_STATUSES = frozenset(
    {
        "approval_required",
        "pending_approval",
        "pending",
        "running",
        "queued",
        "active",
        "paused",
        "failed",
        "error",
    }
)


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
    status: str | None = None
    recovery: str | None = None
    action_label: str | None = None


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

    def __init__(
        self,
        *,
        notification_service: Any | None = None,
        watchlist_service: Any | None = None,
    ):
        self.notification_service = notification_service
        self.watchlist_service = watchlist_service

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
        return replace(
            dashboard_input,
            notification_count=self._unread_notification_count(),
            active_work_items=tuple(self._local_watchlist_run_items()),
        )

    def handle_control(
        self,
        action: HomeControlAction,
        *,
        target_id: str | None = None,
        target_route: str | None = None,
    ) -> HomeControlResult:
        if action is HomeControlAction.OPEN_DETAILS and _is_local_watchlist_run_id(target_id):
            run = self._local_watchlist_run_by_id(str(target_id))
            if run is not None:
                title = self._watchlist_run_title(run)
                return HomeControlResult(
                    action=action,
                    status=HomeControlResultStatus.HANDLED,
                    message=f"Opening W+C run details for {title}.",
                    target_route=target_route or "subscriptions",
                    target_id=target_id,
                )
        if action is HomeControlAction.OPEN_IN_CONSOLE and _is_local_watchlist_run_id(target_id):
            run = self._local_watchlist_run_by_id(str(target_id))
            if run is not None:
                title = self._watchlist_run_title(run)
                return HomeControlResult(
                    action=action,
                    status=HomeControlResultStatus.HANDLED,
                    message=f"Opening Console for {title}.",
                    target_id=target_id,
                    console_launch=HomeConsoleLaunch(
                        source="W+C",
                        title=title,
                        payload=self._watchlist_console_payload(run, str(target_id)),
                        status=str(_mapping_value(run, "status") or "pending"),
                        recovery="Review the W+C run details or retry from W+C.",
                        action_label="Open W+C run",
                    ),
                )
        return super().handle_control(
            action,
            target_id=target_id,
            target_route=target_route,
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

    def _local_watchlist_run_items(self) -> list[HomeActiveWorkItem]:
        if self.watchlist_service is None:
            return []
        try:
            runs = self.watchlist_service.list_home_run_snapshot(limit=20)
        except Exception as e:
            logger.warning(f"Failed to fetch local watchlist runs for Home: {e}")
            return []

        items: list[HomeActiveWorkItem] = []
        for run in runs:
            status = str(_mapping_value(run, "status") or "unknown").strip().lower()
            if status not in _HOME_WATCHLIST_RUN_STATUSES:
                continue
            run_id = _mapping_value(run, "run_id")
            item_id = self._local_watchlist_run_item_id(run)
            if not item_id:
                continue
            title = str(
                _mapping_value(run, "title")
                or _mapping_value(run, "source_title")
                or (f"Watchlist run {run_id}" if run_id is not None else "Watchlist run")
            )
            items.append(
                HomeActiveWorkItem(
                    item_id=item_id,
                    title=title,
                    source="W+C",
                    status=status,
                    detail_route="subscriptions",
                    console_available=True,
                )
            )
        return items

    def _local_watchlist_run_by_id(self, target_id: str) -> Any | None:
        if self.watchlist_service is None:
            return None
        try:
            runs = self.watchlist_service.list_home_run_snapshot(limit=20)
        except Exception as e:
            logger.warning(f"Failed to fetch local watchlist run details for Home: {e}")
            return None
        return next(
            (run for run in runs if self._local_watchlist_run_item_id(run) == target_id),
            None,
        )

    @staticmethod
    def _local_watchlist_run_item_id(run: Any) -> str:
        run_id = _mapping_value(run, "run_id")
        return str(
            _mapping_value(run, "id")
            or (f"local:watchlist_run:{run_id}" if run_id is not None else "")
        )

    @staticmethod
    def _watchlist_console_payload(run: Any, target_id: str) -> Mapping[str, Any]:
        payload: dict[str, Any] = {"target_id": target_id}
        for key in ("run_id", "job_id", "source_id"):
            value = _mapping_value(run, key)
            if value is not None:
                payload[key] = value
        return payload

    @staticmethod
    def _watchlist_run_title(run: Any) -> str:
        return str(
            _mapping_value(run, "title")
            or _mapping_value(run, "source_title")
            or f"Watchlist run {_mapping_value(run, 'run_id') or ''}".strip()
            or "Watchlist run"
        )


def _notification_is_unread(notification: Any) -> bool:
    if isinstance(notification, Mapping):
        return not bool(notification.get("is_read"))
    return not bool(getattr(notification, "is_read", False))


def _mapping_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _is_local_watchlist_run_id(value: str | None) -> bool:
    return bool(value and str(value).startswith("local:watchlist_run:"))
