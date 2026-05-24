"""Pure display-state contracts for the native Console workbench."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape as html_escape
from typing import Any, Mapping

from tldw_chatbook.Chat.citation_evidence_models import EvidenceBundle
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch

CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID = "console-inspector-review-approval"
CONSOLE_INSPECTOR_REVIEW_APPROVAL_LABEL = "Review approval"
CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID = "console-inspector-review-tool-call"
CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_LABEL = "Review tool call"
CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID = "console-inspector-save-chatbook"
CONSOLE_INSPECTOR_SAVE_CHATBOOK_LABEL = "Save Chatbook"
CONSOLE_INSPECTOR_NO_APPROVAL_REASON = "No approval is pending."
CONSOLE_INSPECTOR_NO_TOOL_CALLS_REASON = "No tool calls are ready for review."
CONSOLE_INSPECTOR_NO_CHATBOOK_ARTIFACT_REASON = "No Chatbook artifact is available."


def _clean(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _safe_display_text(value: Any, fallback: str = "") -> str:
    """Normalize user/source text before exposing it in Console display rows."""
    return html_escape(_clean(value, fallback), quote=False)


def coerce_non_negative_int(value: Any) -> int:
    """Coerce a loose seam value into a non-negative integer.

    Args:
        value: Value from an app seam, test fixture, or serialized state.

    Returns:
        A non-negative integer, or 0 when the value is missing or invalid.
    """
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _is_blocked_rag_status(value: Any) -> bool:
    text = _clean(value, "").lower()
    return text.startswith("missing") or text in {"blocked", "unavailable"}


@dataclass(frozen=True)
class ConsoleDisplayRow:
    """One user-visible Console display row."""

    label: str
    value: Any
    status: str = "ready"
    recovery: str = ""

    @property
    def text(self) -> str:
        suffix = f" - {self.recovery}" if self.recovery else ""
        return f"{self.label}: {self.value}{suffix}"


@dataclass(frozen=True)
class ConsoleEvidenceDisplayState:
    """Readable Console summary for one staged evidence bundle."""

    summary: str
    authority: str
    status: str
    recovery: str
    available_count: int
    total_count: int
    reference_rows: tuple[ConsoleDisplayRow, ...] = ()


def evidence_bundle_from_launch(launch: ConsoleLiveWorkLaunch | None) -> EvidenceBundle | None:
    """Parse a staged live-work evidence bundle without exposing raw payload text."""
    if launch is None:
        return None
    evidence_payload = launch.payload.get("evidence_bundle")
    if isinstance(evidence_payload, EvidenceBundle):
        return evidence_payload
    if not isinstance(evidence_payload, Mapping):
        return None
    try:
        return EvidenceBundle.from_payload(evidence_payload)
    except (TypeError, ValueError):
        return None


def build_console_evidence_display_state(
    launch: ConsoleLiveWorkLaunch | None,
) -> ConsoleEvidenceDisplayState | None:
    """Build the user-visible evidence summary for Console staged state."""
    bundle = evidence_bundle_from_launch(launch)
    if bundle is None:
        return None

    available_count = 0
    blocked_count = 0
    stale_count = 0
    missing_count = 0
    reference_rows = []
    authority_values: list[str] = []
    seen_authorities: set[str] = set()
    for reference in bundle.references:
        if reference.status == "available":
            available_count += 1
        elif reference.status == "blocked":
            blocked_count += 1
        elif reference.status == "stale":
            stale_count += 1
        elif reference.status == "missing":
            missing_count += 1

        safe_authority = _safe_display_text(reference.authority_label)
        if safe_authority and safe_authority not in seen_authorities:
            authority_values.append(safe_authority)
            seen_authorities.add(safe_authority)

        reference_status = "blocked" if reference.status != "available" else "ready"
        reference_rows.extend(
            (
                ConsoleDisplayRow(
                    "Evidence source",
                    (
                        f"[{_safe_display_text(reference.evidence_id, 'unknown')}] "
                        f"{_safe_display_text(reference.title, 'Untitled source')}"
                    ),
                    status=reference_status,
                ),
                ConsoleDisplayRow(
                    "Evidence authority",
                    safe_authority or "unknown",
                    status=reference_status,
                ),
                ConsoleDisplayRow(
                    "Evidence status",
                    _safe_display_text(reference.status, "unknown"),
                    status=reference_status,
                ),
            )
        )
        if reference.snippet:
            reference_rows.append(
                ConsoleDisplayRow(
                    "Snippet",
                    _safe_display_text(reference.snippet),
                    status=reference_status,
                )
            )

    total_count = len(bundle.references)
    authority = ", ".join(authority_values) or "unknown"
    summary = f"{available_count}/{total_count} available ({bundle.status})"
    recovery = ""
    if total_count == 0:
        recovery = "No evidence references are attached."
    elif available_count == 0:
        recovery = "No available evidence. Review source authority before sending."
    elif blocked_count or stale_count or missing_count:
        warning_parts = []
        if blocked_count:
            warning_parts.append(f"{blocked_count} blocked")
        if stale_count:
            warning_parts.append(f"{stale_count} stale")
        if missing_count:
            warning_parts.append(f"{missing_count} missing")
        recovery = f"Some evidence needs review: {', '.join(warning_parts)}."

    row_status = "blocked" if available_count == 0 else "ready"
    return ConsoleEvidenceDisplayState(
        summary=summary,
        authority=authority,
        status=row_status,
        recovery=recovery,
        available_count=available_count,
        total_count=total_count,
        reference_rows=tuple(reference_rows),
    )


@dataclass(frozen=True)
class ConsoleInspectorAction:
    """One action exposed by the Console run inspector."""

    widget_id: str
    label: str
    enabled: bool
    disabled_reason: str = ""
    classes: str = "destination-action-button console-inspector-action"

    @property
    def tooltip(self) -> str:
        return "" if self.enabled else self.disabled_reason


@dataclass(frozen=True)
class ConsoleControlState:
    """Header/control labels for the Console-native workbench chrome."""

    provider_label: str
    model_label: str
    persona_label: str
    rag_label: str
    sources_label: str
    tools_label: str
    approvals_label: str

    @classmethod
    def from_values(
        cls,
        *,
        provider: Any = None,
        model: Any = None,
        persona: Any = None,
        rag_enabled: bool = False,
        staged_source_count: int = 0,
        tool_count: int = 0,
        approval_count: int = 0,
    ) -> "ConsoleControlState":
        persona_text = _clean(persona, "")
        persona_label = (
            f"Persona: {persona_text}" if persona_text else "Assistant: General"
        )
        return cls(
            provider_label=f"Provider: {_clean(provider, 'not selected')}",
            model_label=f"Model: {_clean(model, 'not selected')}",
            persona_label=persona_label,
            rag_label=f"RAG: {'on' if rag_enabled else 'off'}",
            sources_label=f"Sources: {staged_source_count} staged",
            tools_label=f"Tools: {tool_count} ready",
            approvals_label=f"Approvals: {approval_count} pending",
        )


@dataclass(frozen=True)
class ConsoleStagedContextState:
    """Display state for the Console staged-context tray."""

    heading: str
    summary: str
    rows: tuple[ConsoleDisplayRow, ...] = ()
    recovery: str = ""

    @classmethod
    def from_live_work(
        cls,
        launch: ConsoleLiveWorkLaunch,
    ) -> "ConsoleStagedContextState":
        rows = []
        evidence_state = build_console_evidence_display_state(launch)
        if evidence_state is not None:
            rows.append(
                ConsoleDisplayRow(
                    "Evidence",
                    evidence_state.summary,
                    status=evidence_state.status,
                    recovery=evidence_state.recovery,
                )
            )
            rows.append(
                ConsoleDisplayRow(
                    "Authority",
                    evidence_state.authority,
                    status=evidence_state.status,
                )
            )
            rows.extend(evidence_state.reference_rows)
        rows.extend(
            ConsoleDisplayRow(label=key, value=value)
            for key, value in launch.payload_display_items()
        )
        return cls(
            heading="Staged Context",
            summary=f"{launch.title} ({launch.source}, {launch.status})",
            rows=tuple(rows),
            recovery=launch.recovery,
        )

    @classmethod
    def empty(cls) -> "ConsoleStagedContextState":
        return cls(
            heading="Staged Context",
            summary="No live work item is staged.",
            recovery="Attach Library, runs, Artifacts, or RAG.",
        )


@dataclass(frozen=True)
class ConsoleInspectorState:
    """Display state for Console run/readiness inspection."""

    rows: tuple[ConsoleDisplayRow, ...]
    actions: tuple[ConsoleInspectorAction, ...] = ()
    has_pending_approval: bool = False
    can_save_chatbook: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        live_work_title: Any = None,
        provider_ready: bool = True,
        provider_recovery: Any = None,
        rag_status: Any = None,
        evidence_summary: Any = None,
        evidence_status: Any = None,
        evidence_recovery: Any = None,
        evidence_authority: Any = None,
        artifact_status: Any = None,
        tool_count: int = 0,
        approval_count: int = 0,
        can_save_chatbook: bool = False,
    ) -> "ConsoleInspectorState":
        provider_status = "ready" if provider_ready else "blocked"
        normalized_tool_count = coerce_non_negative_int(tool_count)
        normalized_approval_count = coerce_non_negative_int(approval_count)
        rag_value = _clean(rag_status, "not staged")
        rows = [
            ConsoleDisplayRow("Live work", _clean(live_work_title, "No active work")),
            ConsoleDisplayRow(
                "Provider",
                provider_status,
                status=provider_status,
                recovery=_clean(provider_recovery, "") if not provider_ready else "",
            ),
            ConsoleDisplayRow("Tools", f"{normalized_tool_count} ready"),
            ConsoleDisplayRow(
                "RAG/source",
                rag_value,
                status="blocked" if _is_blocked_rag_status(rag_value) else "ready",
            ),
            ConsoleDisplayRow(
                "Approvals",
                f"{normalized_approval_count} pending",
                status="blocked" if normalized_approval_count > 0 else "ready",
            ),
        ]
        if _clean(evidence_summary, ""):
            rows.append(
                ConsoleDisplayRow(
                    "Evidence",
                    _clean(evidence_summary, ""),
                    status=_clean(evidence_status, "ready"),
                    recovery=_clean(evidence_recovery, ""),
                )
            )
        if _clean(evidence_authority, ""):
            rows.append(
                ConsoleDisplayRow(
                    "Authority",
                    _clean(evidence_authority, ""),
                    status=_clean(evidence_status, "ready"),
                )
            )
        rows.append(ConsoleDisplayRow("Artifacts", _clean(artifact_status, "unavailable")))
        actions = [
            ConsoleInspectorAction(
                widget_id=CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
                label=CONSOLE_INSPECTOR_REVIEW_APPROVAL_LABEL,
                enabled=normalized_approval_count > 0,
                disabled_reason=CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
            ),
            ConsoleInspectorAction(
                widget_id=CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID,
                label=CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_LABEL,
                enabled=normalized_tool_count > 0,
                disabled_reason=CONSOLE_INSPECTOR_NO_TOOL_CALLS_REASON,
            ),
            ConsoleInspectorAction(
                widget_id=CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
                label=CONSOLE_INSPECTOR_SAVE_CHATBOOK_LABEL,
                enabled=can_save_chatbook,
                disabled_reason=CONSOLE_INSPECTOR_NO_CHATBOOK_ARTIFACT_REASON,
            ),
        ]
        return cls(
            rows=tuple(rows),
            actions=tuple(actions),
            has_pending_approval=normalized_approval_count > 0,
            can_save_chatbook=can_save_chatbook,
        )

    def to_plain_text(self) -> str:
        return "\n".join(row.text for row in self.rows)
