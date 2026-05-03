"""Typed contract for launching live work into Console."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


DEFAULT_RECOVERY = "Console has staged this live-work request."
DEFAULT_ACTION_LABEL = "Open in Console"


def _clean_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _copy_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


@dataclass(frozen=True)
class ConsoleLiveWorkLaunch:
    """Serializable live-work launch context staged for the Console surface."""

    source: str
    title: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    recovery: str = DEFAULT_RECOVERY
    action_label: str = DEFAULT_ACTION_LABEL

    @classmethod
    def from_values(
        cls,
        *,
        source: Any,
        title: Any,
        payload: Mapping[str, Any] | None = None,
        status: Any = None,
        recovery: Any = None,
        action_label: Any = None,
    ) -> "ConsoleLiveWorkLaunch":
        return cls(
            source=_clean_text(source, "unknown"),
            title=_clean_text(title, "Untitled"),
            payload=_copy_payload(payload),
            status=_clean_text(status, "pending"),
            recovery=_clean_text(recovery, DEFAULT_RECOVERY),
            action_label=_clean_text(action_label, DEFAULT_ACTION_LABEL),
        )

    @classmethod
    def from_pending(cls, value: Any) -> "ConsoleLiveWorkLaunch | None":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return None
        return cls.from_values(
            source=value.get("source"),
            title=value.get("title"),
            payload=value.get("payload"),
            status=value.get("status"),
            recovery=value.get("recovery"),
            action_label=value.get("action_label"),
        )

    def to_pending_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "payload": dict(self.payload),
            "status": self.status,
            "recovery": self.recovery,
            "action_label": self.action_label,
        }

    def payload_display_items(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (str(key), self.payload[key])
            for key in sorted(self.payload, key=lambda item: str(item))
        )
