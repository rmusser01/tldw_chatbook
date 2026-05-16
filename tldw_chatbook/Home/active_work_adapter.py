"""Home active-work adapter contract and default unavailable implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum
from html import escape as html_escape
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Protocol, runtime_checkable

from loguru import logger
from rich.markup import escape

from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input
from tldw_chatbook.Utils.path_validation import validate_path
from .dashboard_state import (
    HomeActiveWorkItem,
    HomeDashboardInput,
    SERVER_EVENT_STATE_AVAILABLE,
    SERVER_EVENT_STATE_EMPTY,
    SERVER_EVENT_STATE_RECONNECT_REQUIRED,
    SERVER_EVENT_STATE_REQUERY_REQUIRED,
    SERVER_EVENT_STATE_UNAVAILABLE,
)


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
_MAX_CHATBOOK_ARTIFACT_PREVIEW_CHARS = 1000
_MAX_CHATBOOK_FILE_PATH_CHARS = 2000
_MAX_CHATBOOK_PAYLOAD_TEXT_CHARS = 1000
_MAX_CHATBOOK_METADATA_TEXT_CHARS = 256
_DANGEROUS_TEXT_PATTERNS = ("<script", "</script", "javascript:", "onclick=", "onerror=")


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

    def __init__(self, *, runtime_policy: Any | None = None) -> None:
        self.runtime_policy = runtime_policy

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
            **_runtime_server_status_fields(self.runtime_policy),
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
        chatbook_service: Any | None = None,
        server_event_service: Any | None = None,
        runtime_policy: Any | None = None,
    ) -> None:
        super().__init__(runtime_policy=runtime_policy)
        self.notification_service = notification_service
        self.watchlist_service = watchlist_service
        self.chatbook_service = chatbook_service
        self.server_event_service = server_event_service
        self._chatbook_artifact_snapshot: tuple[Mapping[str, Any], ...] = ()
        self._chatbook_artifact_snapshot_lock = RLock()

    def refresh_chatbook_artifact_snapshot(self, *, limit: int = 20) -> None:
        """Refresh cached local Chatbook artifacts off the Home compose path."""
        if self.chatbook_service is None:
            self._set_chatbook_artifact_snapshot(())
            return
        list_snapshot = getattr(self.chatbook_service, "list_home_artifact_snapshot", None)
        if not callable(list_snapshot):
            self._set_chatbook_artifact_snapshot(())
            return
        try:
            records = list_snapshot(limit=limit)
        except Exception as e:
            logger.warning(f"Failed to fetch local Chatbook artifacts for Home: {e}")
            self._set_chatbook_artifact_snapshot(())
            return
        self._set_chatbook_artifact_snapshot(
            tuple(record for record in records if isinstance(record, Mapping))
        )

    def _set_chatbook_artifact_snapshot(self, records: tuple[Mapping[str, Any], ...]) -> None:
        with self._chatbook_artifact_snapshot_lock:
            self._chatbook_artifact_snapshot = records

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
            **self._server_event_status_fields(),
            active_work_items=tuple(
                [
                    *self._local_watchlist_run_items(),
                    *self._local_chatbook_artifact_items(),
                ]
            ),
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
                    message=f"Opening Watchlists run details for {title}.",
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
                        source="Watchlists",
                        title=title,
                        payload=self._watchlist_console_payload(run, str(target_id)),
                        status=str(_mapping_value(run, "status") or "pending").strip().lower(),
                        recovery="Review the Watchlists run details or retry from Watchlists.",
                        action_label="Open Watchlists run",
                    ),
                )
        if action is HomeControlAction.OPEN_DETAILS and _is_local_chatbook_id(target_id):
            record = self._local_chatbook_artifact_by_id(str(target_id))
            if record is not None:
                title = self._chatbook_title(record)
                return HomeControlResult(
                    action=action,
                    status=HomeControlResultStatus.HANDLED,
                    message=f"Opening Artifacts for {title}.",
                    target_route=target_route or "artifacts",
                    target_id=target_id,
                )
        if action is HomeControlAction.OPEN_IN_CONSOLE and _is_local_chatbook_id(target_id):
            record = self._local_chatbook_artifact_by_id(str(target_id))
            if record is not None:
                title = self._chatbook_title(record)
                return HomeControlResult(
                    action=action,
                    status=HomeControlResultStatus.HANDLED,
                    message=f"Opening Console for {title}.",
                    target_id=target_id,
                    console_launch=HomeConsoleLaunch(
                        source="artifacts",
                        title=title,
                        payload=self._chatbook_console_payload(record, str(target_id)),
                        status="ready",
                        recovery="Review this Chatbook artifact in Console or return to Home.",
                        action_label="Open Chatbook artifact",
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

    def _server_event_status_fields(self) -> dict[str, object]:
        if self.server_event_service is None:
            return {
                "server_event_count": 0,
                "server_event_state": SERVER_EVENT_STATE_UNAVAILABLE,
                "server_event_recovery": "Server event feed is unavailable.",
            }
        list_feed = getattr(self.server_event_service, "list_observed_server_feed", None)
        if not callable(list_feed):
            return {
                "server_event_count": 0,
                "server_event_state": SERVER_EVENT_STATE_UNAVAILABLE,
                "server_event_recovery": "Server event feed is unavailable.",
            }
        try:
            feed = list_feed(limit=20, mark_presented=False)
        except ValueError as exc:
            message = str(exc).lower()
            if "server_profile_id" in message or "active server" in message:
                return {
                    "server_event_count": 0,
                    "server_event_state": SERVER_EVENT_STATE_RECONNECT_REQUIRED,
                    "server_event_recovery": "Reconnect or select an active server.",
                }
            logger.warning(f"Failed to resolve server event feed scope for Home: {exc}")
            return {
                "server_event_count": 0,
                "server_event_state": SERVER_EVENT_STATE_UNAVAILABLE,
                "server_event_recovery": "Server event feed is unavailable.",
            }
        except Exception as exc:
            logger.warning(f"Failed to fetch server event feed for Home: {exc}")
            return {
                "server_event_count": 0,
                "server_event_state": SERVER_EVENT_STATE_UNAVAILABLE,
                "server_event_recovery": "Server event feed is unavailable.",
            }

        if not isinstance(feed, Mapping):
            return {
                "server_event_count": 0,
                "server_event_state": SERVER_EVENT_STATE_UNAVAILABLE,
                "server_event_recovery": "Server event feed is unavailable.",
            }
        items = feed.get("items")
        event_count = _bounded_nonnegative_int(feed.get("total"), default=0)
        if event_count == 0 and isinstance(items, list):
            event_count = len(items)
        replay = feed.get("replay")
        replay_state = ""
        refetch_required = False
        if isinstance(replay, Mapping):
            replay_state = str(replay.get("state") or "").strip().lower()
            refetch_required = bool(replay.get("server_refetch_required"))
        if replay_state == "retention_gap" or refetch_required:
            return {
                "server_event_count": event_count,
                "server_event_state": SERVER_EVENT_STATE_REQUERY_REQUIRED,
                "server_event_recovery": "Requery server events from the active server.",
            }
        return {
            "server_event_count": event_count,
            "server_event_state": (
                SERVER_EVENT_STATE_AVAILABLE
                if event_count > 0
                else SERVER_EVENT_STATE_EMPTY
            ),
            "server_event_recovery": (
                "Review observed server-owned events without changing server read state."
                if event_count > 0
                else "Observe or refresh server events from the active server."
            ),
        }

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
                    source="Watchlists",
                    status=status,
                    detail_route="subscriptions",
                    console_available=True,
                )
            )
        return items

    def _local_chatbook_artifact_items(self) -> list[HomeActiveWorkItem]:
        items: list[HomeActiveWorkItem] = []
        for record in self._local_chatbook_artifacts()[:1]:
            item_id = self._local_chatbook_item_id(record)
            if not item_id:
                continue
            items.append(
                HomeActiveWorkItem(
                    item_id=item_id,
                    title=self._chatbook_title(record),
                    source="Artifacts",
                    status="ready",
                    detail_route="artifacts",
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

    def _local_chatbook_artifacts(self) -> list[Any]:
        with self._chatbook_artifact_snapshot_lock:
            return list(self._chatbook_artifact_snapshot)

    def _local_chatbook_artifact_by_id(self, target_id: str) -> Any | None:
        return next(
            (
                record
                for record in self._local_chatbook_artifacts()
                if self._local_chatbook_item_id(record) == target_id
            ),
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

    @staticmethod
    def _local_chatbook_item_id(record: Any) -> str:
        chatbook_id = _mapping_value(record, "chatbook_id") or _mapping_value(record, "id")
        return f"local:chatbook:{chatbook_id}" if chatbook_id not in (None, "") else ""

    @staticmethod
    def _chatbook_title(record: Any) -> str:
        return _safe_payload_text(
            _mapping_value(record, "name") or _mapping_value(record, "title"),
            fallback="Untitled Chatbook",
            max_length=_MAX_CHATBOOK_PAYLOAD_TEXT_CHARS,
        )

    @classmethod
    def _chatbook_console_payload(cls, record: Any, target_id: str) -> Mapping[str, Any]:
        chatbook_id = _mapping_value(record, "chatbook_id") or _mapping_value(record, "id")
        payload: dict[str, Any] = {
            "target_id": target_id,
            "chatbook_id": chatbook_id,
            "record_id": _safe_payload_text(
                _mapping_value(record, "id"),
                max_length=_MAX_CHATBOOK_METADATA_TEXT_CHARS,
            ),
            "file_path": _safe_file_path(_mapping_value(record, "file_path")),
            "description": _safe_payload_text(
                _mapping_value(record, "description"),
                max_length=_MAX_CHATBOOK_PAYLOAD_TEXT_CHARS,
            ),
            "tags": _csv(_mapping_value(record, "tags")),
            "categories": _csv(_mapping_value(record, "categories")),
            "updated_at": _safe_payload_text(
                _mapping_value(record, "updated_at"),
                max_length=_MAX_CHATBOOK_METADATA_TEXT_CHARS,
            ),
        }
        payload.update(_console_metadata_payload(_mapping_value(record, "metadata")))
        return payload


def _notification_is_unread(notification: Any) -> bool:
    if isinstance(notification, Mapping):
        return not bool(notification.get("is_read"))
    return not bool(getattr(notification, "is_read", False))


def _mapping_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _bounded_nonnegative_int(value: Any, *, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _is_local_watchlist_run_id(value: str | None) -> bool:
    return bool(value and str(value).startswith("local:watchlist_run:"))


def _is_local_chatbook_id(value: str | None) -> bool:
    return bool(value and str(value).startswith("local:chatbook:"))


def _runtime_server_status_fields(runtime_policy: Any | None) -> dict[str, object]:
    state = getattr(runtime_policy, "state", None)
    if not isinstance(state, RuntimeSourceState):
        return {
            "runtime_source": "local",
            "active_server_id": None,
            "server_label": None,
            "server_configured": False,
            "server_reachability": "unknown",
            "server_auth_state": "unknown",
        }
    return {
        "runtime_source": state.active_source,
        "active_server_id": state.active_server_id,
        "server_label": state.last_known_server_label or state.active_server_id,
        "server_configured": state.server_configured,
        "server_reachability": state.server_reachability,
        "server_auth_state": state.server_auth_state,
    }


def _csv(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _safe_payload_text(value, max_length=_MAX_CHATBOOK_PAYLOAD_TEXT_CHARS) or None
    if isinstance(value, (list, tuple)):
        text = ", ".join(
            _safe_payload_text(item, max_length=_MAX_CHATBOOK_PAYLOAD_TEXT_CHARS)
            for item in value
            if str(item).strip()
        )
        text = _safe_payload_text(text, max_length=_MAX_CHATBOOK_PAYLOAD_TEXT_CHARS)
        return text or None
    return _safe_payload_text(value, max_length=_MAX_CHATBOOK_PAYLOAD_TEXT_CHARS) or None


def _safe_payload_text(
    value: Any,
    *,
    fallback: str = "",
    max_length: int = _MAX_CHATBOOK_PAYLOAD_TEXT_CHARS,
    single_line: bool = True,
) -> str:
    text = sanitize_string(str(value or ""), max_length=max_length).strip()
    if single_line:
        text = " ".join(text.split())
    if not text:
        return fallback
    if not validate_text_input(text, max_length=max_length, allow_html=False):
        for pattern in _DANGEROUS_TEXT_PATTERNS:
            replacement = pattern.rstrip(":=").replace("=", "")
            text = re.sub(re.escape(pattern), replacement, text, flags=re.IGNORECASE)
        if not validate_text_input(text, max_length=max_length, allow_html=False):
            return fallback
    return escape(html_escape(text, quote=False))


def _safe_metadata_value(value: Any, *, max_length: int = _MAX_CHATBOOK_METADATA_TEXT_CHARS) -> Any | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = _safe_payload_text(value, max_length=max_length)
    return text or None


def _safe_file_path(value: Any) -> str | None:
    text = sanitize_string(str(value or ""), max_length=_MAX_CHATBOOK_FILE_PATH_CHARS).strip()
    if not text:
        return None
    text = " ".join(text.split())
    try:
        path = Path(text).expanduser()
        base_directory = path.parent if path.is_absolute() else Path.cwd()
        validated = validate_path(path, base_directory)
    except ValueError:
        logger.warning(f"Rejected unsafe Chatbook artifact file path for Home payload: {text!r}")
        return None
    return _safe_payload_text(
        str(validated),
        max_length=_MAX_CHATBOOK_FILE_PATH_CHARS,
    ) or None


def _console_metadata_payload(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    artifact_source = _safe_metadata_value(metadata.get("artifact_source"), max_length=128)
    artifact_kind = _safe_metadata_value(metadata.get("artifact_kind"), max_length=128)
    if str(artifact_source or "").strip().lower() != "console":
        return {}
    if str(artifact_kind or "").strip().lower() != "assistant-response":
        return {}

    payload: dict[str, Any] = {
        "artifact_source": artifact_source,
        "artifact_kind": artifact_kind,
    }
    for key in ("conversation_id", "message_id", "message_role", "provider", "model"):
        value = _safe_metadata_value(metadata.get(key))
        if value is not None:
            payload[key] = value
    if metadata.get("content") is not None:
        content = sanitize_string(
            str(metadata.get("content")),
            max_length=_MAX_CHATBOOK_ARTIFACT_PREVIEW_CHARS,
        )
        payload["content_preview"] = _safe_payload_text(
            content,
            max_length=_MAX_CHATBOOK_ARTIFACT_PREVIEW_CHARS,
            single_line=False,
        )
        payload["content_truncated"] = (
            bool(metadata.get("content_truncated"))
            or len(str(metadata.get("content"))) > _MAX_CHATBOOK_ARTIFACT_PREVIEW_CHARS
        )
    return payload
