"""Pure display-state contracts for the native Console workbench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch


def _clean(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


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
        rows = tuple(
            ConsoleDisplayRow(label=key, value=value)
            for key, value in launch.payload_display_items()
        )
        return cls(
            heading="Staged Context",
            summary=f"{launch.title} ({launch.source}, {launch.status})",
            rows=rows,
            recovery=launch.recovery,
        )

    @classmethod
    def empty(cls) -> "ConsoleStagedContextState":
        return cls(
            heading="Staged Context",
            summary="No live work item is staged.",
            recovery="Attach sources from Library, W+C, Schedules, Artifacts, or RAG.",
        )


@dataclass(frozen=True)
class ConsoleInspectorState:
    """Display state for Console run/readiness inspection."""

    rows: tuple[ConsoleDisplayRow, ...]
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
        artifact_status: Any = None,
        approval_count: int = 0,
        can_save_chatbook: bool = False,
    ) -> "ConsoleInspectorState":
        provider_status = "ready" if provider_ready else "blocked"
        rows = [
            ConsoleDisplayRow("Live work", _clean(live_work_title, "No active work")),
            ConsoleDisplayRow(
                "Provider",
                provider_status,
                status=provider_status,
                recovery=_clean(provider_recovery, "") if not provider_ready else "",
            ),
            ConsoleDisplayRow("RAG", _clean(rag_status, "not staged")),
            ConsoleDisplayRow("Artifacts", _clean(artifact_status, "unavailable")),
            ConsoleDisplayRow("Approvals", f"{approval_count} pending"),
        ]
        return cls(
            rows=tuple(rows),
            has_pending_approval=approval_count > 0,
            can_save_chatbook=can_save_chatbook,
        )

    def to_plain_text(self) -> str:
        return "\n".join(row.text for row in self.rows)
