"""Local MCP prompt-reduction recommendation engine.

Pure logic only: no disk, no Textual, no telemetry. Callers provide recent
execution-log records, live tool catalog entries, and resolved permission
states.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState

APPROVED_DECISION = "approved"
AGENT_INITIATOR = "agent"
DEFAULT_MIN_APPROVED_COUNT = 2

_EXCLUSION_LABELS = {
    "below-threshold": "below threshold",
    "already-allowed": "already allowed",
    "denied": "explicitly denied",
    "definition-changed": "definition changed",
    "high-risk-floor": "high-risk safety floor",
    "tool-not-found": "missing from the current catalog",
    "tool-unavailable": "currently unavailable",
    "state-unavailable": "permission state unavailable",
}

PromptKey = tuple[str, str]


@dataclass(frozen=True)
class PermissionPromptRecommendation:
    """One MCP tool-level allow recommendation."""

    server_key: str
    server_label: str
    tool_name: str
    approved_count: int
    first_seen: str
    last_seen: str
    current_state: str
    reason: str

    @property
    def tool_id(self) -> str:
        return f"{self.server_key}::{self.tool_name}"


@dataclass(frozen=True)
class PermissionPromptReport:
    """Prompt-reduction analysis result."""

    recommendations: list[PermissionPromptRecommendation]
    excluded: dict[PromptKey, str]
    total_records: int
    approval_records: int
    min_approved_count: int


def format_permission_prompt_report(
    report: PermissionPromptReport, *, max_recommendations: int = 8
) -> str:
    """Render a compact Console-safe prompt-reduction report."""
    lines = ["MCP prompt recommendations"]
    lines.append(
        "Recent local MCP log: "
        f"{report.total_records} records, "
        f"{report.approval_records} agent approvals, "
        f"threshold {report.min_approved_count}."
    )
    if report.recommendations:
        lines.append("Review candidates:")
        for index, recommendation in enumerate(
            report.recommendations[:max_recommendations], start=1
        ):
            approval_word = (
                "time" if recommendation.approved_count == 1 else "times"
            )
            label = recommendation.server_label or recommendation.server_key
            last_seen = (
                f", last seen {recommendation.last_seen}"
                if recommendation.last_seen
                else ""
            )
            lines.append(
                f"{index}. {label} / {recommendation.tool_name} - "
                f"approved {recommendation.approved_count} {approval_word}"
                f"{last_seen}."
            )
        remaining = len(report.recommendations) - max_recommendations
        if remaining > 0:
            lines.append(f"...and {remaining} more.")
        lines.append("Review/apply through MCP permission APIs; do not edit JSON.")
    else:
        if report.total_records == 0:
            lines.append("No local MCP execution records were found.")
        elif report.approval_records == 0:
            lines.append("No prompted agent approvals were found in those records.")
        else:
            lines.append("No eligible repeated ask-gated MCP approvals were found.")
    if report.excluded:
        reason_counts = Counter(report.excluded.values())
        summaries = [
            f"{label} ({reason_counts[reason]})"
            for reason, label in _EXCLUSION_LABELS.items()
            if reason_counts[reason]
        ]
        unknown_count = sum(
            count for reason, count in reason_counts.items()
            if reason not in _EXCLUSION_LABELS
        )
        if unknown_count:
            summaries.append(f"other safety exclusions ({unknown_count})")
        lines.append("Not recommended: " + "; ".join(summaries) + ".")
    lines.append("Auto Mode and bash allowlisting are deferred.")
    return "\n".join(lines)


@dataclass
class _ApprovalStats:
    count: int = 0
    first_seen: str = ""
    last_seen: str = ""

    def add_seen(self, ts: str) -> None:
        self.count += 1
        if ts and (not self.first_seen or ts < self.first_seen):
            self.first_seen = ts
        if ts and (not self.last_seen or ts > self.last_seen):
            self.last_seen = ts


def build_permission_prompt_report(
    records: Sequence[Mapping[str, Any]],
    tools: Sequence[HubTool],
    states: Mapping[PromptKey, EffectiveToolState],
    *,
    min_approved_count: int = DEFAULT_MIN_APPROVED_COUNT,
) -> PermissionPromptReport:
    """Recommend MCP tools worth reviewing for a tool-level allow.

    Args:
        records: Recent execution-log records, newest or oldest first.
        tools: Live MCP catalog entries available for hash-safe application.
        states: Effective permission state by ``(server_key, tool_name)``.
        min_approved_count: Minimum repeated human approvals required.

    Returns:
        A local, telemetry-free recommendation report.
    """
    threshold = max(1, int(min_approved_count or DEFAULT_MIN_APPROVED_COUNT))
    tool_by_key = {(tool.server_key, tool.name): tool for tool in tools}
    stats_by_key: dict[PromptKey, _ApprovalStats] = {}
    approval_records = 0

    for record in records:
        if record.get("decision") != APPROVED_DECISION:
            continue
        if record.get("initiator") != AGENT_INITIATOR:
            continue
        server_key = str(record.get("server_key") or "").strip()
        tool_name = str(record.get("tool_name") or "").strip()
        if not server_key or not tool_name:
            continue
        approval_records += 1
        stats = stats_by_key.setdefault((server_key, tool_name), _ApprovalStats())
        stats.add_seen(str(record.get("ts") or ""))

    recommendations: list[PermissionPromptRecommendation] = []
    excluded: dict[PromptKey, str] = {}

    for key, stats in stats_by_key.items():
        if stats.count < threshold:
            excluded[key] = "below-threshold"
            continue

        tool = tool_by_key.get(key)
        if tool is None:
            excluded[key] = "tool-not-found"
            continue
        if tool.stale or not tool.executable:
            excluded[key] = "tool-unavailable"
            continue

        state = states.get(key)
        if state is None:
            excluded[key] = "state-unavailable"
            continue
        if state.state == "allow":
            excluded[key] = "already-allowed"
            continue
        if state.state == "deny":
            excluded[key] = "denied"
            continue
        if state.state != "ask":
            excluded[key] = "state-unavailable"
            continue
        if state.config_changed:
            excluded[key] = "definition-changed"
            continue
        if state.risk_floored:
            excluded[key] = "high-risk-floor"
            continue

        recommendations.append(
            PermissionPromptRecommendation(
                server_key=tool.server_key,
                server_label=tool.server_label,
                tool_name=tool.name,
                approved_count=stats.count,
                first_seen=stats.first_seen,
                last_seen=stats.last_seen,
                current_state=state.state,
                reason="Repeatedly approved while still ask-gated.",
            )
        )

    recommendations.sort(
        key=lambda item: (
            -item.approved_count,
            _reverse_sort_text(item.last_seen),
            item.server_label.casefold(),
            item.tool_name.casefold(),
        )
    )
    return PermissionPromptReport(
        recommendations=recommendations,
        excluded=excluded,
        total_records=len(records),
        approval_records=approval_records,
        min_approved_count=threshold,
    )


def _reverse_sort_text(value: str) -> tuple[int, ...]:
    return tuple(-ord(char) for char in value)
