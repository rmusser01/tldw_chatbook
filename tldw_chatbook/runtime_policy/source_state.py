from __future__ import annotations

from collections.abc import Mapping
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .types import RuntimeSourceState

POLICY_FRESHNESS_WINDOW = timedelta(minutes=5)

_VALID_ACTIVE_SOURCES = {"local", "server"}
_VALID_SERVER_REACHABILITY = {"unknown", "reachable", "unreachable"}
_VALID_SERVER_AUTH_STATES = {"unknown", "authenticated", "auth_required", "session_invalid"}
_KEEP_VALUE = object()


def _is_fresh(checked_at: datetime | None, *, now: datetime, freshness_window: timedelta) -> bool:
    if checked_at is None:
        return False
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now - checked_at <= freshness_window


def normalize_runtime_source_state(
    state: RuntimeSourceState,
    *,
    now: datetime,
    freshness_window: timedelta,
) -> RuntimeSourceState:
    server_reachability = state.server_reachability
    server_reachability_checked_at = state.server_reachability_checked_at
    if not _is_fresh(server_reachability_checked_at, now=now, freshness_window=freshness_window):
        server_reachability = "unknown"
        server_reachability_checked_at = None

    server_auth_state = state.server_auth_state
    server_auth_checked_at = state.server_auth_checked_at
    if not _is_fresh(server_auth_checked_at, now=now, freshness_window=freshness_window):
        server_auth_state = "unknown"
        server_auth_checked_at = None

    return replace(
        state,
        server_reachability=server_reachability,
        server_reachability_checked_at=server_reachability_checked_at,
        server_auth_state=server_auth_state,
        server_auth_checked_at=server_auth_checked_at,
    )


def runtime_source_state_with_override(
    state: RuntimeSourceState,
    *,
    active_source: str | None = None,
    active_server_id=_KEEP_VALUE,
    server_configured: bool | None = None,
    last_known_server_label=_KEEP_VALUE,
) -> RuntimeSourceState:
    resolved_source = _coerce_choice(
        active_source if active_source is not None else state.active_source,
        valid_values=_VALID_ACTIVE_SOURCES,
        default=state.active_source,
    )
    resolved_server_id = state.active_server_id if active_server_id is _KEEP_VALUE else active_server_id
    resolved_server_configured = (
        state.server_configured if server_configured is None else bool(server_configured)
    )
    resolved_server_label = (
        state.last_known_server_label
        if last_known_server_label is _KEEP_VALUE
        else last_known_server_label
    )

    return replace(
        state,
        active_source=resolved_source,
        active_server_id=resolved_server_id,
        server_configured=resolved_server_configured,
        last_known_server_label=resolved_server_label,
    )


def runtime_source_state_to_dict(state: RuntimeSourceState) -> dict:
    return {
        "active_source": state.active_source,
        "active_server_id": state.active_server_id,
        "server_configured": state.server_configured,
        "server_reachability": state.server_reachability,
        "server_reachability_checked_at": _datetime_to_iso(state.server_reachability_checked_at),
        "server_auth_state": state.server_auth_state,
        "server_auth_checked_at": _datetime_to_iso(state.server_auth_checked_at),
        "last_known_server_label": state.last_known_server_label,
    }


def runtime_source_state_from_dict(data) -> RuntimeSourceState:
    if not isinstance(data, Mapping):
        return RuntimeSourceState()

    return RuntimeSourceState(
        active_source=_coerce_choice(
            data.get("active_source", "local"),
            valid_values=_VALID_ACTIVE_SOURCES,
            default="local",
        ),
        active_server_id=data.get("active_server_id"),
        server_configured=bool(data.get("server_configured", False)),
        server_reachability=_coerce_choice(
            data.get("server_reachability", "unknown"),
            valid_values=_VALID_SERVER_REACHABILITY,
            default="unknown",
        ),
        server_reachability_checked_at=_iso_to_datetime(data.get("server_reachability_checked_at")),
        server_auth_state=_coerce_choice(
            data.get("server_auth_state", "unknown"),
            valid_values=_VALID_SERVER_AUTH_STATES,
            default="unknown",
        ),
        server_auth_checked_at=_iso_to_datetime(data.get("server_auth_checked_at")),
        last_known_server_label=data.get("last_known_server_label"),
    )


def _datetime_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _iso_to_datetime(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _coerce_choice(value, *, valid_values: set[str], default: str) -> str:
    if isinstance(value, str) and value in valid_values:
        return value
    return default


class RuntimeSourceStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RuntimeSourceState:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return RuntimeSourceState()
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return RuntimeSourceState()

        if not isinstance(data, dict):
            return RuntimeSourceState()

        return RuntimeSourceState.from_dict(data)

    def save(self, state: RuntimeSourceState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        payload = runtime_source_state_to_dict(state)

        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

        temp_path.replace(self.path)
