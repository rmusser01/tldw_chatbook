"""Typed contract for launching live work into Console."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


DEFAULT_RECOVERY = "Console has staged this live-work request."
DEFAULT_ACTION_LABEL = "Open in Console"
PENDING_LAUNCH_CARD_ID = "console-pending-launch-card"
LIVE_WORK_CARD_CLASS = "console-live-work-status-card"


def _clean_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _copy_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _safe_widget_suffix(value: Any) -> str:
    suffix = []
    previous_dash = False
    for character in str(value or "").strip().lower():
        if character.isalnum():
            suffix.append(character)
            previous_dash = False
        elif not previous_dash:
            suffix.append("-")
            previous_dash = True
    return "".join(suffix).strip("-") or "item"


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


@dataclass(frozen=True)
class ConsoleLiveWorkStatusCardRow:
    """A stable render row for Console live-work status cards."""

    widget_id: str
    text: str
    classes: str = "destination-section console-live-work-status-row"


@dataclass(frozen=True)
class ConsoleLiveWorkStatusCardState:
    """Reusable display contract for one Console live-work status card."""

    badge_text: str
    rows: tuple[ConsoleLiveWorkStatusCardRow, ...]
    container_id: str = PENDING_LAUNCH_CARD_ID
    container_classes: str = f"ds-panel {LIVE_WORK_CARD_CLASS}"
    badge_id: str = "console-live-work-status-badge"
    badge_classes: str = "ds-status-badge console-live-work-status-badge"

    @classmethod
    def from_launch(cls, launch: ConsoleLiveWorkLaunch) -> "ConsoleLiveWorkStatusCardState":
        rows = [
            ConsoleLiveWorkStatusCardRow(
                widget_id="console-live-work-source",
                text=f"Source: {launch.source}",
            ),
            ConsoleLiveWorkStatusCardRow(
                widget_id="console-live-work-title",
                text=f"Title: {launch.title}",
            ),
            ConsoleLiveWorkStatusCardRow(
                widget_id="console-live-work-status",
                text=f"Status: {launch.status}",
            ),
            ConsoleLiveWorkStatusCardRow(
                widget_id="console-live-work-recovery",
                text=f"Recovery: {launch.recovery}",
            ),
            ConsoleLiveWorkStatusCardRow(
                widget_id="console-live-work-action",
                text=f"Action: {launch.action_label}",
            ),
        ]
        seen_ids = set()
        for key, value in launch.payload_display_items():
            suffix = _safe_widget_suffix(key)
            base_id = f"console-live-work-payload-{suffix}"
            widget_id = base_id
            counter = 1
            while widget_id in seen_ids:
                widget_id = f"{base_id}-{counter}"
                counter += 1
            seen_ids.add(widget_id)
            rows.append(
                ConsoleLiveWorkStatusCardRow(
                    widget_id=widget_id,
                    text=f"{key}: {value}",
                    classes="destination-section console-live-work-status-row console-live-work-payload-row",
                )
            )
        return cls(
            badge_text="Pending Console launch",
            rows=tuple(rows),
        )
