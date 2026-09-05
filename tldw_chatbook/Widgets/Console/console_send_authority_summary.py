"""Pinned projection of the next Console send's authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual.app import ComposeResult
from textual.css.query import NoMatches, QueryError
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

#: Rows of the pinned block at its normal density: the heading plus all five
#: facts.
CONSOLE_AUTHORITY_SUMMARY_HEIGHT = 6

#: Rows at constrained heights (TASK-31663 AC#5). Measured at 80x24 before
#: this change: the rail's pinned stack was EIGHT lines -- header 1, project
#: instruction 1, this block 6 -- over a THREE-line scroll body, so the
#: Environment section (seven lines at rest) could never show its four rows.
#: Two lines here turns that 3-line body into a 7-line one.
CONSOLE_AUTHORITY_SUMMARY_COMPACT_HEIGHT = 2

#: The one fact that keeps its own line when compact. `run` is deliberate: it
#: is not one fact among five but the severity-ordered ROLLUP of all of them
#: (incomplete data > recovery required > waiting for approval > blocked
#: provider/retrieval > running > failed > ready), so the line that survives
#: is the one that already answers the block's question. The other four stay
#: reachable, unchanged, through the block's tooltip and through F1 while it
#: has focus -- `ChatScreen.action_show_workbench_help` renders
#: `contextual_help_rows()` under this block's own heading, and that method
#: reads the PROJECTION, not the mounted rows, so hiding rows costs it
#: nothing.
_COMPACT_FACT_ATTRIBUTES = frozenset({"run"})


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
    """Project the complete next-send facts from one Inspector snapshot.

    Args:
        state: The atomic Inspector state to summarize.
        ownership_policy: The ownership policy used to classify Inspector rows.

    Returns:
        The five user-facing authority facts.
    """

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
    # TASK-24610: the run inspector's retrieval row is "Retrieval" now.
    # "Sources" is kept as a fallback because this projection also runs over
    # snapshots produced before the rename (persisted/replayed state), and
    # losing the lookup would leave Run reading "Ready" while retrieval is
    # blocked -- silently, and in the one line pinned above the fold.
    source = (
        rows.get("Retrieval") or rows.get("Sources") or rows.get("RAG/source")
    )
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
    elif state.run_failed:
        # TASK-24602. Ordered BELOW everything above it deliberately: a run in
        # flight, a pending approval and a blocked provider all describe what
        # the NEXT send will do, and that is the question this line asks. A
        # past failure only describes the last one, so it must not mask them.
        # It sits ABOVE "Ready" because "Ready" after a failure is the single
        # most misleading thing this line can say.
        reason = str(state.run_failure_reason or "").strip()
        run = f"Failed — {reason}" if reason else "Failed"
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
    """One focus stop containing six fixed, single-line physical rows.

    At constrained rail heights (TASK-31663 AC#5) it drops to two: the
    heading and the ``Run`` rollup. All six Statics stay MOUNTED either way
    -- ``sync_state`` patches them by id, and consumers query them by id --
    so the compact form costs display, never structure.
    """

    def __init__(self, state: ConsoleInspectorState, **kwargs: Any) -> None:
        super().__init__(id=CONSOLE_AUTHORITY_SUMMARY_ID, **kwargs)
        self.can_focus = True
        self.last_state = state
        self._projection = project_console_send_authority(state)
        self.recompose_count = 0
        self.compact = False
        self._apply_density()

    def compose(self) -> ComposeResult:
        """Compose the heading and five fixed fact rows.

        Returns:
            The child widgets for the six-row summary.
        """

        yield self._row(
            "What happens if I send now?",
            "console-send-authority-heading",
        )
        for label, attribute, widget_id in _FACTS:
            value = getattr(self._projection, attribute)
            # The other half of `_apply_density`'s display pass: a block
            # built ALREADY compact (the rail sizes it before mount at
            # 80x24) has to mount with the right rows hidden, or its first
            # painted frame is six lines tall inside a two-line box.
            yield self._row(
                f"{label}: {value}", widget_id, hidden=self._fact_is_hidden(attribute)
            )

    @staticmethod
    def _row(copy: str, widget_id: str, *, hidden: bool = False) -> Static:
        row = Static(Text(copy), id=widget_id, classes="console-send-authority-row")
        row.styles.height = 1
        row.styles.min_height = 1
        row.styles.max_height = 1
        row.styles.text_wrap = "nowrap"
        row.styles.text_overflow = "ellipsis"
        if hidden:
            row.styles.display = "none"
        return row

    def _fact_is_hidden(self, attribute: str) -> bool:
        """Whether one fact gives up its line at the current density."""

        return self.compact and attribute not in _COMPACT_FACT_ATTRIBUTES

    def set_compact(self, compact: bool) -> None:
        """Switch the block between its six-line and two-line densities.

        Args:
            compact: Whether the rail is too short to spend six rows here.
        """

        if compact == self.compact:
            return
        self.compact = compact
        self._apply_density()

    def _apply_density(self) -> None:
        """Pin the block's height and row visibility to ``self.compact``."""

        height = (
            CONSOLE_AUTHORITY_SUMMARY_COMPACT_HEIGHT
            if self.compact
            else CONSOLE_AUTHORITY_SUMMARY_HEIGHT
        )
        # Inline as well as in the stylesheet: this widget has always pinned
        # its own geometry (a bare harness loads no bundle), and inline wins
        # over CSS in Textual, so the class flip below would be cosmetic on
        # its own. Both halves are kept in step deliberately.
        self.styles.height = height
        self.styles.min_height = height
        self.styles.max_height = height
        self.set_class(self.compact, "-authority-compact")
        if not self.is_mounted:
            return
        for _label, attribute, widget_id in _FACTS:
            try:
                row = self.query_one(f"#{widget_id}", Static)
            except (NoMatches, QueryError):
                continue
            row.styles.display = "none" if self._fact_is_hidden(attribute) else "block"
        self._sync_compact_tooltip()
        self.recompute_tooltips()

    def _sync_compact_tooltip(self) -> None:
        """Carry the facts that gave up their line, on the block itself."""

        if not self.compact:
            self.tooltip = None
            return
        self.tooltip = Text(
            "\n".join(
                f"{label}: {value}" for label, value in self.contextual_help_rows()
            )
        )

    def sync_state(self, state: ConsoleInspectorState) -> None:
        """Patch all five facts from one new snapshot without recomposing.

        Args:
            state: The replacement atomic Inspector snapshot.
        """

        if state == self.last_state:
            return
        projection = project_console_send_authority(state)
        if not self.is_mounted:
            self.last_state = state
            self._projection = projection
            return
        try:
            rows = tuple(
                (label, attribute, self.query_one(f"#{widget_id}", Static))
                for label, attribute, widget_id in _FACTS
            )
        except (NoMatches, QueryError):
            return
        self.last_state = state
        self._projection = projection
        for label, attribute, row in rows:
            value = getattr(projection, attribute)
            row.update(Text(f"{label}: {value}"))
        self._sync_compact_tooltip()
        self.recompute_tooltips()
        self.refresh()

    def recompute_tooltips(self) -> None:
        """Expose only complete values whose own physical row is clipped."""

        if not self.is_mounted:
            return
        try:
            rows = tuple(
                (label, attribute, self.query_one(f"#{widget_id}", Static))
                for label, attribute, widget_id in _FACTS
            )
        except (NoMatches, QueryError):
            return
        for label, attribute, row in rows:
            value = getattr(self._projection, attribute)
            copy = f"{label}: {value}"
            width = max(0, row.content_region.width)
            row.tooltip = Text(value) if width and cell_len(copy) > width else None

    def on_resize(self, _event: Resize) -> None:
        """Recompute clipped-value help after a width change.

        Args:
            _event: The Textual resize event.
        """

        self.recompute_tooltips()

    def contextual_help_rows(self) -> tuple[tuple[str, str], ...]:
        """Return all complete fact values for focused contextual help.

        Returns:
            Ordered label/value pairs for the five authority facts.
        """

        return tuple(
            (label, getattr(self._projection, attribute))
            for label, attribute, _widget_id in _FACTS
        )
