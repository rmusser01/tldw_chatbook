"""Display-state adapter for Sync v2 profile summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_text(value: Any, fallback: str = "", *, max_length: int = 200) -> str:
    text = sanitize_string(str(value or ""), max_length=max_length).strip()
    text = " ".join(text.split())
    if not text:
        return fallback
    if not validate_text_input(text, max_length=max_length, allow_html=False):
        return fallback
    return text


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _status(value: Any) -> str:
    candidate = _safe_text(value, "not_configured", max_length=80).lower()
    allowed = {
        "not_configured",
        "local_only",
        "server_frontend",
        "pending",
        "attention_required",
        "ready",
    }
    return candidate if candidate in allowed else "not_configured"


@dataclass(frozen=True)
class SyncProfileStatusDisplay:
    """Presentation-safe status derived from a Sync v2 profile summary.

    Attributes:
        status: Normalized Sync v2 profile status.
        severity: UI severity token used by the Library status banner.
        label: Short human-readable banner title.
        detail: Sanitized status detail copy.
        pending_count: Count of local outbox changes waiting to sync.
        dispatched_count: Count of dispatched outbox changes.
        conflict_count: Count of conflicts that need review.
        dataset_label: Sanitized dataset label for the active profile.
        device_label: Sanitized device label for the active profile.
        read_only_notice: Stable copy explaining the banner does not start sync.
    """

    status: str
    severity: str
    label: str
    detail: str
    pending_count: int
    dispatched_count: int
    conflict_count: int
    dataset_label: str
    device_label: str
    read_only_notice: str = "This view only reads sync state; it does not start sync."

    @classmethod
    def from_summary(cls, summary: Mapping[str, Any] | None) -> "SyncProfileStatusDisplay":
        """Build display state from repository/service summary data.

        Args:
            summary: Raw Sync v2 profile summary mapping returned by the repository
                or service layer. Missing or malformed mappings are treated as not
                configured.

        Returns:
            SyncProfileStatusDisplay with normalized counts and sanitized display
            strings.
        """

        record = _mapping(summary)
        status = _status(record.get("status"))
        profile = _mapping(record.get("profile"))
        outbox = _mapping(record.get("outbox"))
        conflicts = _mapping(record.get("conflicts"))
        pending_count = _count(outbox.get("pending"))
        dispatched_count = _count(outbox.get("dispatched"))
        conflict_count = _count(conflicts.get("count"))
        server_profile_id = _safe_text(profile.get("server_profile_id"), "the configured server")
        dataset_id = _safe_text(profile.get("dataset_id"), "")
        device_id = _safe_text(profile.get("device_id"), "")
        last_error = _safe_text(profile.get("last_error"), "unavailable", max_length=240)

        return cls(
            status=status,
            severity=_severity(status),
            label=_label(status),
            detail=_detail(
                status,
                server_profile_id=server_profile_id,
                pending_count=pending_count,
                conflict_count=conflict_count,
                last_error=last_error,
            ),
            pending_count=pending_count,
            dispatched_count=dispatched_count,
            conflict_count=conflict_count,
            dataset_label=f"Dataset {dataset_id}" if dataset_id else "Dataset not assigned",
            device_label=f"Device {device_id}" if device_id else "Device not registered",
        )


def _severity(status: str) -> str:
    severities = {
        "attention_required": "attention",
        "pending": "pending",
        "ready": "ready",
        "server_frontend": "ready",
        "local_only": "neutral",
        "not_configured": "neutral",
    }
    return severities.get(status, "neutral")


def _label(status: str) -> str:
    labels = {
        "attention_required": "Sync profile: needs attention",
        "pending": "Sync profile: pending local changes",
        "ready": "Sync profile: ready",
        "server_frontend": "Sync profile: server front-end",
        "local_only": "Sync profile: local-only",
        "not_configured": "Sync profile: not configured",
    }
    return labels.get(status, "Sync profile: not configured")


def _detail(
    status: str,
    *,
    server_profile_id: str,
    pending_count: int,
    conflict_count: int,
    last_error: str,
) -> str:
    if status == "attention_required":
        if conflict_count:
            suffix = "conflict needs" if conflict_count == 1 else "conflicts need"
            return f"{conflict_count} sync {suffix} review. Last error is {last_error}."
        return f"Sync needs attention. Last error is {last_error}."
    if status == "pending":
        suffix = "change is" if pending_count == 1 else "changes are"
        return f"{pending_count} pending local {suffix} waiting for the next sync pass."
    if status == "ready":
        return "Local-first sync is ready."
    if status == "server_frontend":
        return (
            f"Using {server_profile_id} as a live server front-end. "
            "Offline mirror sync is not configured."
        )
    if status == "local_only":
        return "This profile is local-only. Local Library data stays on this device."
    return "No Sync v2 server profile is configured. Local Library data stays on this device."
