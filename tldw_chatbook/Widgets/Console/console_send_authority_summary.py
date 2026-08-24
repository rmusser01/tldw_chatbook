"""Pinned projection of the next Console send's authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual.app import ComposeResult
from textual.events import Resize
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import ConsoleInspectorState
from tldw_chatbook.Widgets.Console.console_inspector_ownership import (
    InspectorOwnershipPolicy,
    classify_inspector_content,
)


CONSOLE_AUTHORITY_SUMMARY_ID = "console-send-authority-summary"

_FACTS = (
    ("Where", "where", "console-send-authority-where"),
    ("Scope", "scope", "console-send-authority-scope"),
    ("Run", "run", "console-send-authority-run"),
    ("Sources", "sources", "console-send-authority-sources"),
    ("Approvals", "approvals", "console-send-authority-approvals"),
)


@dataclass(frozen=True, slots=True)
class ConsoleSendAuthorityProjection:
    """Five complete facts shown by the pinned summary."""

    where: str
    scope: str
    run: str
    sources: str
    approvals: str


def project_console_send_authority(
    state: ConsoleInspectorState,
    *,
    ownership_policy: InspectorOwnershipPolicy = InspectorOwnershipPolicy.RESILIENT,
) -> ConsoleSendAuthorityProjection:
    """Project the complete next-send facts from one Inspector snapshot."""

    owned = classify_inspector_content(state, ownership_policy)
    rows = {entry.row.label: entry.row for entry in owned.rows}

    workspace_row = rows.get("Workspace")
    workspace = str(workspace_row.value).strip() if workspace_row else "Default"
    workspace = workspace or "Default"
    conversation_row = rows.get("Selected conversation")
    conversation = str(conversation_row.value).strip() if conversation_row else ""
    if conversation in {"", "No active conversation"}:
        conversation = (
            "Temporary conversation" if state.ephemeral else "No active conversation"
        )

    scope_parts: list[str] = []
    if rows.get("Prefill (next send only)") is not None:
        scope_parts.append("One-shot prefill")
    elif rows.get("Prefill (pinned)") is not None:
        scope_parts.append("Pinned prefill")
    if state.scope_item_count == 0:
        scope_parts.append("No sources")
    elif state.scope_item_count is not None:
        scope_parts.append(f"narrowed to {state.scope_item_count} items")

    provider = rows.get("Provider")
    source = rows.get("Sources") or rows.get("RAG/source")
    recovery_required = any(
        rows.get(label) is not None for label in ("Recovery action", "Next action")
    )
    if owned.incomplete:
        run = "Inspector data incomplete"
    elif recovery_required:
        run = "Recovery required"
    elif state.pending_approval_count > 0:
        run = "Waiting for approval"
    elif (provider is not None and provider.status == "blocked") or (
        source is not None and source.status == "blocked"
    ):
        run = "Blocked"
    elif state.run_active:
        run = "Running"
    else:
        run = "Ready"

    where = f"{workspace} › {conversation}"
    if state.ephemeral:
        where += " · Temporary"
    return ConsoleSendAuthorityProjection(
        where=where,
        scope=" · ".join(scope_parts) or "Everything available",
        run=run,
        sources=(
            f"{state.staged_source_count} staged"
            if state.staged_source_count
            else "None staged"
        ),
        approvals=(
            f"{state.pending_approval_count} pending · action required"
            if state.pending_approval_count
            else "None pending"
        ),
    )


class ConsoleSendAuthoritySummary(Static):
    """One focus stop containing six fixed, single-line physical rows."""

    def __init__(self, state: ConsoleInspectorState, **kwargs: Any) -> None:
        super().__init__(id=CONSOLE_AUTHORITY_SUMMARY_ID, **kwargs)
        self.can_focus = True
        self.last_state = state
        self._projection = project_console_send_authority(state)
        self.recompose_count = 0
        self.styles.height = 6
        self.styles.min_height = 6
        self.styles.max_height = 6

    def compose(self) -> ComposeResult:
        yield self._row(
            "What happens if I send now?",
            "console-send-authority-heading",
        )
        for label, attribute, widget_id in _FACTS:
            value = getattr(self._projection, attribute)
            yield self._row(f"{label}: {value}", widget_id)

    @staticmethod
    def _row(copy: str, widget_id: str) -> Static:
        row = Static(Text(copy), id=widget_id, classes="console-send-authority-row")
        row.styles.height = 1
        row.styles.min_height = 1
        row.styles.max_height = 1
        row.styles.text_wrap = "nowrap"
        row.styles.text_overflow = "ellipsis"
        return row

    def sync_state(self, state: ConsoleInspectorState) -> None:
        """Patch all five facts from one new snapshot without recomposing."""

        if state == self.last_state:
            return
        projection = project_console_send_authority(state)
        self.last_state = state
        self._projection = projection
        if not self.is_mounted:
            return
        for label, attribute, widget_id in _FACTS:
            value = getattr(projection, attribute)
            self.query_one(f"#{widget_id}", Static).update(Text(f"{label}: {value}"))
        self.recompute_tooltips()
        self.refresh()

    def recompute_tooltips(self) -> None:
        """Expose only complete values whose own physical row is clipped."""

        if not self.is_mounted:
            return
        for label, attribute, widget_id in _FACTS:
            row = self.query_one(f"#{widget_id}", Static)
            value = getattr(self._projection, attribute)
            copy = f"{label}: {value}"
            width = max(0, row.content_region.width)
            row.tooltip = Text(value) if width and cell_len(copy) > width else None

    def on_resize(self, _event: Resize) -> None:
        self.recompute_tooltips()

    def contextual_help_rows(self) -> tuple[tuple[str, str], ...]:
        """Return all complete fact values for focused contextual help."""

        return tuple(
            (label, getattr(self._projection, attribute))
            for label, attribute, _widget_id in _FACTS
        )
