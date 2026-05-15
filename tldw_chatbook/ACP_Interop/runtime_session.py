"""ACP-owned runtime and session readiness state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch


def _text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _mapping_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


@dataclass(frozen=True)
class ACPRuntimeSessionState:
    """Display and Console handoff contract for ACP runtime/session readiness."""

    runtime_id: str = ""
    runtime_label: str = ""
    runtime_version: str = ""
    session_id: str = ""
    session_title: str = ""
    session_status: str = ""
    session_payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any) -> "ACPRuntimeSessionState":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        payload = _mapping_value(value, "session_payload")
        return cls(
            runtime_id=_text(_mapping_value(value, "runtime_id")),
            runtime_label=_text(_mapping_value(value, "runtime_label")),
            runtime_version=_text(_mapping_value(value, "runtime_version")),
            session_id=_text(_mapping_value(value, "session_id")),
            session_title=_text(_mapping_value(value, "session_title")),
            session_status=_text(_mapping_value(value, "session_status"), "pending"),
            session_payload=dict(payload) if isinstance(payload, Mapping) else {},
        )

    @property
    def runtime_configured(self) -> bool:
        return bool(self.runtime_id or self.runtime_label)

    @property
    def runtime_display_name(self) -> str:
        return self.runtime_label or self.runtime_id or "No ACP runtime"

    @property
    def session_display_name(self) -> str:
        return self.session_title or self.session_id or "none"

    @property
    def has_console_session_payload(self) -> bool:
        return bool(self.runtime_configured and self.session_id and self.session_payload)

    def to_console_live_work_launch(self) -> ConsoleLiveWorkLaunch | None:
        if not self.has_console_session_payload:
            return None
        payload = {
            "target_id": f"local:acp_session:{self.session_id}",
            "session_id": self.session_id,
            "runtime_id": self.runtime_id,
            "runtime_label": self.runtime_display_name,
            "session_payload": dict(self.session_payload),
        }
        return ConsoleLiveWorkLaunch.from_values(
            source="ACP",
            title=self.session_display_name,
            payload=payload,
            status=self.session_status,
            recovery="Console can follow this ACP session payload.",
            action_label="Follow ACP session",
        )
