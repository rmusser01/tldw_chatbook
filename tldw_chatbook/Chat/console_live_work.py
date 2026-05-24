"""Typed contract for launching live work into Console."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


DEFAULT_RECOVERY = "Console has staged this live-work request."
DEFAULT_ACTION_LABEL = "Open in Console"
PENDING_LAUNCH_CARD_ID = "console-pending-launch-card"
LIVE_WORK_CARD_CLASS = "console-live-work-status-card"
PRIMARY_ACTION_BUTTON_ID = "console-live-work-primary-action"
SOURCE_READINESS_CARD_ID = "console-live-work-source-readiness"
HIDDEN_PAYLOAD_DISPLAY_KEYS = frozenset({"evidence_bundle"})


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
            if str(key) not in HIDDEN_PAYLOAD_DISPLAY_KEYS
        )


@dataclass(frozen=True)
class ConsoleLiveWorkStatusCardRow:
    """A stable render row for Console live-work status cards."""

    widget_id: str
    text: str
    classes: str = "destination-section console-live-work-status-row"


@dataclass(frozen=True)
class ConsoleLiveWorkPrimaryAction:
    """Action metadata for a supported Console live-work follow-through."""

    label: str
    target_route: str
    target_id: str
    widget_id: str = PRIMARY_ACTION_BUTTON_ID
    classes: str = "destination-action-button console-live-work-primary-action"


def resolve_console_live_work_primary_action(
    launch: ConsoleLiveWorkLaunch,
) -> ConsoleLiveWorkPrimaryAction | None:
    """Resolve launch payloads that can safely route to an existing detail surface."""
    source = launch.source.strip().lower()
    target_id = str(launch.payload.get("target_id") or "").strip()
    if source in {"w+c", "watchlists", "watchlists+collections"} and ":watchlist_run:" in target_id:
        return ConsoleLiveWorkPrimaryAction(
            label=launch.action_label,
            target_route="subscriptions",
            target_id=target_id,
        )
    if source in {"artifacts", "chatbooks"} and ":chatbook:" in target_id:
        return ConsoleLiveWorkPrimaryAction(
            label=launch.action_label,
            target_route="artifacts",
            target_id=target_id,
        )
    if source == "acp" and ":acp_session:" in target_id:
        return ConsoleLiveWorkPrimaryAction(
            label=launch.action_label,
            target_route="acp",
            target_id=target_id,
        )
    return None


@dataclass(frozen=True)
class ConsoleLiveWorkStatusCardState:
    """Reusable display contract for one Console live-work status card."""

    badge_text: str
    rows: tuple[ConsoleLiveWorkStatusCardRow, ...]
    primary_action: ConsoleLiveWorkPrimaryAction | None = None
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
            primary_action=resolve_console_live_work_primary_action(launch),
        )


@dataclass(frozen=True)
class ConsoleLiveWorkSourceReadinessRow:
    """One row in the Console live-work source readiness summary."""

    widget_id: str
    label: str
    status: str
    recovery: str
    classes: str

    @property
    def text(self) -> str:
        return f"{self.label}: {self.status} - {self.recovery}"


@dataclass(frozen=True)
class ConsoleLiveWorkSourceReadinessState:
    """Compact source support summary for Console live-work integrations."""

    rows: tuple[ConsoleLiveWorkSourceReadinessRow, ...]
    title: str = "Live work sources"
    container_id: str = SOURCE_READINESS_CARD_ID
    container_classes: str = "ds-panel console-live-work-source-readiness"
    title_id: str = "console-live-work-source-readiness-title"
    title_classes: str = "ds-status-badge console-live-work-source-readiness-title"

    @classmethod
    def default(cls) -> "ConsoleLiveWorkSourceReadinessState":
        """Build the default Console live-work source readiness summary.

        Returns:
            ConsoleLiveWorkSourceReadinessState: Readiness state that marks Watchlists,
                Schedules, Workflows, RAG, and Artifacts as connected while
                MCP as unavailable until its payload producer is wired.
        """
        connected = "destination-section console-live-work-source-row console-live-work-source-connected"
        unavailable = "destination-section console-live-work-source-row console-live-work-source-unavailable"
        return cls.from_acp_runtime_status("not_configured")

    @classmethod
    def from_acp_runtime_status(
        cls,
        status: str,
    ) -> "ConsoleLiveWorkSourceReadinessState":
        """Build readiness rows with ACP reflecting the shared runtime state."""
        connected = "destination-section console-live-work-source-row console-live-work-source-connected"
        unavailable = "destination-section console-live-work-source-row console-live-work-source-unavailable"
        acp_status = str(status or "").strip().lower()
        acp_label = "Blocked"
        acp_recovery = "Configure ACP runtime."
        acp_classes = unavailable
        if acp_status == "starting":
            acp_label = "Starting"
            acp_recovery = "Waiting for runtime."
        elif acp_status == "running":
            acp_label = "Connected"
            acp_recovery = "Follow ACP session."
            acp_classes = connected
        elif acp_status == "failed":
            acp_label = "Failed"
            acp_recovery = "Review ACP runtime."
        elif acp_status in {"configured", "stopped"}:
            acp_label = "Ready"
            acp_recovery = "Launch ACP runtime."
        return cls(
            rows=(
                ConsoleLiveWorkSourceReadinessRow(
                    widget_id="console-live-work-source-wc",
                    label="Watchlists",
                    status="Connected",
                    recovery="Home run details.",
                    classes=connected,
                ),
                ConsoleLiveWorkSourceReadinessRow(
                    widget_id="console-live-work-source-workflows",
                    label="Workflows",
                    status="Connected",
                    recovery="Stage run context.",
                    classes=connected,
                ),
                ConsoleLiveWorkSourceReadinessRow(
                    widget_id="console-live-work-source-schedules",
                    label="Schedules",
                    status="Connected",
                    recovery="Open job context.",
                    classes=connected,
                ),
                ConsoleLiveWorkSourceReadinessRow(
                    widget_id="console-live-work-source-acp",
                    label="ACP",
                    status=acp_label,
                    recovery=acp_recovery,
                    classes=acp_classes,
                ),
                ConsoleLiveWorkSourceReadinessRow(
                    widget_id="console-live-work-source-mcp",
                    label="MCP",
                    status="Not wired",
                    recovery="Manage servers in MCP.",
                    classes=unavailable,
                ),
                ConsoleLiveWorkSourceReadinessRow(
                    widget_id="console-live-work-source-rag",
                    label="RAG",
                    status="Connected",
                    recovery="Stage search evidence.",
                    classes=connected,
                ),
                ConsoleLiveWorkSourceReadinessRow(
                    widget_id="console-live-work-source-artifacts",
                    label="Artifacts",
                    status="Connected",
                    recovery="Launch Chatbooks.",
                    classes=connected,
                ),
            )
        )
