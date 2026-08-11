"""Reusable Inspector-style section: header + rows + optional "View all" tail.

Supervisor fleet PR 2b, Task 3 (spec `Docs/superpowers/specs/2026-08-08-
supervisor-agent-fleet-design.md` section 7): the Console rail is evolving
toward a Claude-Code-desktop-style Inspector -- stacked sections sharing one
grammar: a header (title + optional collapse chevron + optional right-
aligned summary), a body of rows, and an optional "View all" tail. This
module ships that grammar once. The Agents section (Task 4) is the first
consumer; Changes/Sources/Workspace sections are filed follow-ups against
the same classes -- **this module carries no Agents-specific vocabulary**.

The caller supplies rows as a sequence of ``InspectorSectionRow`` value
objects, bundled with the header summary into one atomic
``ConsoleInspectorSectionState`` passed to ``sync_state``; the component
owns layout, DOM ids, and the update discipline -- never the data. A single
state object, not independent kwargs, is deliberate (task-3 review round
1): an earlier ``sync_state(*, rows=(), summary="")`` treated an omitted
argument as "clear this", so the natural "just refresh the rows" call
silently wiped an unrelated summary that was never meant to change. One
state value object makes "what is the section's state" one thing, not two
that can drift -- the same discipline ``ConsoleRunInspector`` already uses
for its own ``ConsoleInspectorState``.

Update discipline follows ``ConsoleRunInspector``
(`console_run_inspector.py:152-179`): a structural key (row identity +
summary presence) decides whether a state change can be patched into the
already-mounted row/summary Statics in place, or whether it must fall back
to a wholesale ``refresh(recompose=True)``. Unlike
``ConsoleWorkspaceContextTray`` (`console_workspace_context.py:537-567`,
which deliberately reverted an equality guard because skipping recompose
broke click targeting on its rows), the in-place path here is safe for
click targeting: each mounted row widget carries its own ``row_id`` as a
plain Python attribute (mirroring ``home_rail.py``'s ``button.row_id``
stamp), and the structural key guarantees the same ``row_id`` stays bound
to the same row *widget instance* across an in-place patch -- only the
row's own text/status Statics are mutated, never the widget carrying the
click identity. ``Tests/UI/test_console_inspector_section.py`` proves this
with a test that clicks a row, patches it in place, and clicks it again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.destination_rail import GLYPH_COLLAPSED, GLYPH_EXPANDED
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


@dataclass(frozen=True)
class InspectorSectionRow:
    """One row rendered in an Inspector section body -- a caller-owned value
    object; the component never mutates or derives from anything but its
    fields.

    Attributes:
        row_id: Stable identity for this row across syncs. Used both as the
            payload of ``ConsoleInspectorSection.RowActivated`` and, via the
            structural key, to decide whether a state change is safe to
            patch in place (same ``row_id``s in the same order) or requires
            a recompose (rows added/removed/reordered).
        primary_text: First (non-dimmed) line, e.g. "glyph + name + elapsed".
        secondary_text: Second, dimmed line, e.g. a truncated last-step
            summary. Empty renders as a blank second line rather than a
            missing one, so a row's mounted shape never depends on whether
            this happens to be set.
        status: Status token driving the row's
            ``console-inspector-section-row-<status>`` CSS class (e.g.
            "running", "done", "error", "blocked"); ``""`` applies none.
        clickable: Whether activating the row (click, Enter, or Space)
            posts ``ConsoleInspectorSection.RowActivated``.
        cancellable: Whether pressing Delete on the row (PR2b Task 5)
            posts ``ConsoleInspectorSection.RowCancelRequested``.
            Independent of ``clickable`` -- a row can be either, both, or
            neither; the two are separate gestures (Enter/Space to drill
            in, Delete to cancel) so they never contend for the same key.
    """

    row_id: str
    primary_text: str
    secondary_text: str = ""
    status: str = ""
    clickable: bool = False
    cancellable: bool = False


@dataclass(frozen=True)
class ConsoleInspectorSectionState:
    """Atomic snapshot of everything ``sync_state`` can change: rows plus
    the header summary.

    Passed as ONE value so the two dimensions can never drift independently
    -- a caller that wants to refresh only the rows still passes the
    summary it already had (its own held state, unchanged), rather than
    the previous kwargs shape where omitting ``summary`` silently meant
    "clear it" (task-3 review round 1 finding). Mirrors
    ``ConsoleInspectorState`` (`console_display_state.py`), the same
    full-snapshot-per-sync pattern ``ConsoleRunInspector`` already uses.

    **Both fields are REQUIRED -- neither has a default** (task-3 review
    round 3 finding, HIGH: round 1's fix kept per-field defaults, which
    meant ``ConsoleInspectorSectionState(rows=updated_rows)`` -- omitting
    ``summary`` -- reproduced the exact same "silently wipes the other
    dimension" symptom the atomic-state redesign existed to eliminate, one
    call-frame later. Removing the defaults makes that construction a
    ``TypeError`` instead of a silent data-loss bug: a caller that means
    "no summary" must write ``summary=""`` explicitly, which costs one
    keyword and makes "I forgot a dimension" impossible to express.

    Attributes:
        rows: Rows to render.
        summary: Right-aligned header summary; ``""`` hides it (an
            explicit, deliberate choice -- not an omitted argument).
    """

    rows: tuple[InspectorSectionRow, ...]
    summary: str


class ConsoleInspectorSection(RecomposeCaptureGuard, Vertical):
    """One Inspector section: header (title + optional chevron + optional
    right-aligned summary), a body of rows, and an optional "View all" tail.

    The section owns its own collapse state (no owning screen needs to
    toggle a body's ``display`` the way ``HomeRail`` does for its section
    Verticals) -- collapsing hides the body but always keeps the header
    painted, so a collapsed section still reads as "N working, M done" on
    the summary line.

    Attributes:
        title: Section title, shown in the header and used to derive the
            chevron tooltip.
        section_id: Stable id fragment folded into every mounted child's
            DOM id and into every posted message -- must be unique among
            sibling sections mounted on the same screen.
        rows: Currently rendered rows (value objects owned by the caller).
        summary: Right-aligned header text (e.g. "3 working, 1 done").
            Empty renders no summary at all.
        collapsible: Whether the header shows a collapse/expand chevron.
            When ``False`` the section is always open and ``open``/
            ``set_open`` have no effect.
        open: Whether the body is currently visible.
        view_all_label: Label for the optional "View all" tail button.
            Empty renders no tail.
        recompose_count: Count of wholesale recomposes taken by
            ``sync_state`` -- test seam mirroring
            ``ConsoleRunInspector.recompose_count`` (Task 5 asserts this
            stays flat across a coalesced burst of non-structural updates).
    """

    class RowActivated(Message):
        """Posted when a clickable row is activated (click, Enter, Space)."""

        def __init__(self, section_id: str, row_id: str) -> None:
            self.section_id = section_id
            self.row_id = row_id
            super().__init__()

    class RowCancelRequested(Message):
        """Posted when a cancellable row's cancel gesture (Delete) fires.

        PR2b Task 5 (per-row cancel). Deliberately a SEPARATE message from
        ``RowActivated`` rather than an overload of it -- a row can be both
        clickable (drill in) and cancellable (stop it) at once, and the two
        must never be conflated into one ambiguous "activate" outcome. This
        module stays generic (see the module docstring): cancelling is a
        plausible action for rows in any future section built on this
        component, not Agents-specific vocabulary.
        """

        def __init__(self, section_id: str, row_id: str) -> None:
            self.section_id = section_id
            self.row_id = row_id
            super().__init__()

    class ViewAllRequested(Message):
        """Posted when the "View all" tail is activated."""

        def __init__(self, section_id: str) -> None:
            self.section_id = section_id
            super().__init__()

    class CollapseToggled(Message):
        """Posted after the section's own open/collapsed state changes.

        The section already applies the new state to its own DOM before
        posting this -- it exists purely so a caller can persist the
        preference (mirroring ``HomeRailPreferences``), not so the caller
        has to apply it.
        """

        def __init__(self, section_id: str, open: bool) -> None:
            self.section_id = section_id
            self.open = open
            super().__init__()

    def __init__(
        self,
        *,
        title: str,
        section_id: str,
        rows: Sequence[InspectorSectionRow] = (),
        summary: str = "",
        collapsible: bool = True,
        open: bool = True,
        view_all_label: str = "",
        **kwargs: Any,
    ) -> None:
        """Create an Inspector section.

        Args:
            title: Section title shown in the header.
            section_id: Stable id fragment for this section's DOM ids and
                posted-message payloads; must be unique among sibling
                sections on the same screen.
            rows: Initial rows to render.
            summary: Initial right-aligned header summary; ``""`` renders
                no summary.
            collapsible: Whether the header shows a collapse/expand
                chevron. ``False`` forces the section permanently open.
            open: Initial open/collapsed state (ignored, forced ``True``,
                when ``collapsible`` is ``False``).
            view_all_label: Label for an optional "View all" tail button;
                ``""`` renders no tail.
            **kwargs: Forwarded to ``Vertical`` (e.g. ``id``, ``classes``).
        """
        super().__init__(**kwargs)
        self.title = title
        self.section_id = section_id
        self.rows: tuple[InspectorSectionRow, ...] = tuple(rows)
        self.summary = summary
        self.collapsible = collapsible
        self.open = True if not collapsible else bool(open)
        self.view_all_label = view_all_label
        self.styles.height = "auto"
        self.styles.min_height = 0
        self.add_class("console-inspector-section")
        self.recompose_count = 0

    # -- DOM id helpers -----------------------------------------------

    @property
    def _header_id(self) -> str:
        return f"console-inspector-section-{self.section_id}-header"

    @property
    def _title_id(self) -> str:
        return f"console-inspector-section-{self.section_id}-title"

    @property
    def _summary_id(self) -> str:
        return f"console-inspector-section-{self.section_id}-summary"

    @property
    def _toggle_id(self) -> str:
        return f"console-inspector-section-{self.section_id}-toggle"

    @property
    def _body_id(self) -> str:
        return f"console-inspector-section-{self.section_id}-body"

    @property
    def _view_all_id(self) -> str:
        return f"console-inspector-section-{self.section_id}-view-all"

    def _row_id_attr(self, index: int) -> str:
        return f"console-inspector-section-{self.section_id}-row-{index}"

    def _row_primary_id(self, index: int) -> str:
        return f"{self._row_id_attr(index)}-primary"

    def _row_secondary_id(self, index: int) -> str:
        return f"{self._row_id_attr(index)}-secondary"

    # -- Compose --------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Render the header, the row body, and the optional tail.

        Returns:
            A ``ComposeResult`` yielding the header ``Horizontal`` (title +
            optional summary + optional chevron), the body ``Vertical`` of
            row widgets, and the optional "View all" ``Button``.
        """
        header = Horizontal(id=self._header_id, classes="console-inspector-section-header")
        header.styles.height = 1
        header.styles.min_height = 1
        with header:
            title = Static(
                self.title,
                id=self._title_id,
                classes="console-inspector-section-title",
                markup=False,
            )
            title.styles.width = "1fr"
            title.styles.min_width = 0
            yield title
            if self.summary:
                summary = Static(
                    self.summary,
                    id=self._summary_id,
                    classes="console-inspector-section-summary",
                    markup=False,
                )
                # Inline, not just the CSS class: geometry must be correct
                # for any host, including a bare test harness that never
                # loads the app's CSS bundle (the CSS class carries the
                # cosmetic color/dim styling only).
                summary.styles.width = "auto"
                summary.styles.min_width = 0
                summary.styles.height = 1
                yield summary
            if self.collapsible:
                toggle = Button(
                    self._toggle_label(),
                    id=self._toggle_id,
                    classes="console-inspector-section-toggle",
                    compact=True,
                )
                toggle.tooltip = self._toggle_tooltip()
                toggle.styles.width = 3
                toggle.styles.min_width = 3
                toggle.styles.max_width = 3
                toggle.styles.height = 1
                yield toggle

        body = Vertical(id=self._body_id, classes="console-inspector-section-body")
        body.styles.height = "auto"
        body.styles.min_height = 0
        if not self.open:
            body.styles.display = "none"
        with body:
            for index, row in enumerate(self.rows):
                yield self._build_row_widget(row, index)

        if self.view_all_label:
            view_all = Button(
                self.view_all_label,
                id=self._view_all_id,
                classes="console-inspector-section-view-all",
                compact=True,
            )
            yield view_all

    def _build_row_widget(
        self, row: InspectorSectionRow, index: int
    ) -> "ConsoleInspectorSectionRow":
        return ConsoleInspectorSectionRow(
            row,
            section_id=self.section_id,
            index=index,
            id=self._row_id_attr(index),
        )

    # -- Collapse ---------------------------------------------------------

    def _toggle_label(self) -> str:
        return resolve_glyph(GLYPH_EXPANDED if self.open else GLYPH_COLLAPSED)

    def _toggle_tooltip(self) -> str:
        return f"Collapse {self.title}" if self.open else f"Expand {self.title}"

    def set_open(self, open: bool) -> None:
        """Show or hide the body without recomposing the section.

        The header (and its summary/chevron) stays painted either way --
        only the row body's ``display`` changes. Posts ``CollapseToggled``
        so a caller can persist the preference; a no-op call (``open``
        already matches) posts nothing.

        Args:
            open: Whether the body should be visible.
        """
        if not self.collapsible:
            return
        if open == self.open:
            return
        self.open = open
        if self.is_mounted:
            try:
                body = self.query_one(f"#{self._body_id}", Vertical)
            except (NoMatches, QueryError):
                pass
            else:
                body.styles.display = "block" if open else "none"
            try:
                toggle = self.query_one(f"#{self._toggle_id}", Button)
            except (NoMatches, QueryError):
                pass
            else:
                toggle.label = self._toggle_label()
                toggle.tooltip = self._toggle_tooltip()
        self.post_message(self.CollapseToggled(self.section_id, open))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the chevron toggle and the "View all" tail button."""
        if self.collapsible and event.button.id == self._toggle_id:
            event.stop()
            self.set_open(not self.open)
        elif self.view_all_label and event.button.id == self._view_all_id:
            event.stop()
            self.post_message(self.ViewAllRequested(self.section_id))

    # -- Sync -------------------------------------------------------------

    @staticmethod
    def _structural_key(
        rows: tuple[InspectorSectionRow, ...], summary: str
    ) -> tuple:
        """Return a key identifying the mounted widget structure for a state.

        Two states with equal keys mount the same row ids, in the same
        order, with the same summary presence (shown vs. hidden) -- they
        differ at most in row/summary text, row status, or row
        clickability, all of which are safe to patch in place.

        ``clickable`` is deliberately EXCLUDED from this key (task-3 review
        round 2 finding, LOW): ``_apply_row_update`` already re-syncs a
        row's ``clickable``/``can_focus`` attributes unconditionally on
        every patch, structural or not, so including it here bought no
        correctness and only forced an avoidable recompose on a plausible
        fleet transition (a row becoming clickable as an agent goes
        queued -> running) -- exactly the recompose churn Task 5 is
        measured on keeping flat.

        Args:
            rows: Row sequence to fingerprint.
            summary: Header summary text to fingerprint (only its presence
                matters, not its content).

        Returns:
            A hashable structure key.
        """
        return (
            tuple(row.row_id for row in rows),
            bool(summary),
        )

    def sync_state(self, state: ConsoleInspectorSectionState) -> None:
        """Refresh the mounted section from a new atomic state snapshot.

        ``state`` is the section's WHOLE state (rows + summary), not a
        delta -- a caller that wants to refresh only the rows still passes
        the summary it already had. This is deliberate (task-3 review
        round 1, HIGH): an earlier ``sync_state(*, rows=(), summary="")``
        kwargs shape treated an omitted argument as "clear this", so the
        natural "just refresh the rows" call silently wiped an unrelated
        summary. See ``ConsoleInspectorSectionState``'s docstring.

        When the new state is structurally compatible with the current one
        (same row ids in the same order, same summary presence), the
        mounted row/summary Statics are patched in place. Any structural
        change -- rows added/removed/reordered, or the summary
        appearing/disappearing -- recomposes the whole section (counted in
        ``recompose_count``).

        Args:
            state: New rows + header summary, as one atomic snapshot.
        """
        rows = tuple(state.rows)
        summary = state.summary
        if rows == self.rows and summary == self.summary:
            return
        previous_rows = self.rows
        previous_summary = self.summary
        same_structure = self._structural_key(rows, summary) == self._structural_key(
            previous_rows, previous_summary
        )
        self.rows = rows
        self.summary = summary
        if (
            not self.is_mounted
            or not same_structure
            or not self._apply_state_updates(previous_rows, previous_summary)
        ):
            self.recompose_count += 1
            self.refresh(recompose=True)

    def _apply_state_updates(
        self,
        previous_rows: tuple[InspectorSectionRow, ...],
        previous_summary: str,
    ) -> bool:
        """Patch changed summary/row Statics in place.

        Args:
            previous_rows: The row sequence that produced the mounted rows.
            previous_summary: The summary text that produced the mounted
                summary Static (or the fact it was absent).

        Returns:
            ``True`` when every changed widget was found and patched;
            ``False`` when a target was missing (caller falls back to a
            recompose).
        """
        try:
            if self.summary and self.summary != previous_summary:
                self.query_one(f"#{self._summary_id}", Static).update(self.summary)
            for index, (row, previous_row) in enumerate(
                zip(self.rows, previous_rows)
            ):
                self._apply_row_update(index, row, previous_row)
        except (NoMatches, QueryError):
            return False
        return True

    def _apply_row_update(
        self, index: int, row: InspectorSectionRow, previous_row: InspectorSectionRow
    ) -> None:
        """Patch one row widget in place; raises NoMatches/QueryError on a
        missing target so the caller can fall back to a recompose."""
        row_widget = self.query_one(
            f"#{self._row_id_attr(index)}", ConsoleInspectorSectionRow
        )
        # Identity is unchanged by construction (same structural key), but
        # kept authoritative rather than assumed.
        row_widget.row_id = row.row_id
        # `clickable` is NOT part of the structural key (see
        # `_structural_key`'s docstring) -- it can change on an in-place
        # patch, so it and the `can_focus` it drives are always re-synced
        # here, unconditionally, whether or not this specific call's
        # `row.clickable` differs from `previous_row.clickable`.
        row_widget.clickable = row.clickable
        row_widget.can_focus = row.clickable or row.cancellable
        # `cancellable` -- like `clickable` immediately above -- is NOT part
        # of the structural key (same reasoning: a row transitioning
        # running -> done flips it without changing row identity/order), so
        # it too is always re-synced here, unconditionally.
        row_widget.cancellable = row.cancellable
        if row.primary_text != previous_row.primary_text:
            self.query_one(f"#{self._row_primary_id(index)}", Static).update(
                row.primary_text
            )
        if row.secondary_text != previous_row.secondary_text:
            self.query_one(f"#{self._row_secondary_id(index)}", Static).update(
                row.secondary_text
            )
        if row.status != previous_row.status:
            if previous_row.status:
                row_widget.remove_class(
                    f"console-inspector-section-row-{previous_row.status}"
                )
            if row.status:
                row_widget.add_class(f"console-inspector-section-row-{row.status}")


class ConsoleInspectorSectionRow(Vertical):
    """One row inside a ``ConsoleInspectorSection`` body.

    Always mounts two Statics (primary + secondary line, the latter
    possibly blank) so a row's mounted shape never depends on whether
    ``secondary_text`` happens to be empty at any given sync -- only row
    identity (the structural key) governs recompose vs. in-place patching.
    ``clickable`` can change across an in-place patch too (re-synced
    unconditionally by ``ConsoleInspectorSection._apply_row_update``).
    """

    BINDINGS = [
        Binding("enter", "activate_row", "Activate row", show=False),
        Binding("space", "activate_row", "Activate row", show=False),
        # PR2b Task 5: Delete, not Enter/Space -- cancelling and drilling in
        # are separate gestures so a row that is both clickable and
        # cancellable never has to arbitrate which one a shared key means.
        Binding("delete", "cancel_row", "Cancel row", show=False),
    ]

    def __init__(
        self,
        row: InspectorSectionRow,
        *,
        section_id: str,
        index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(classes="console-inspector-section-row", **kwargs)
        self.section_id = section_id
        self.row_id = row.row_id
        self.clickable = row.clickable
        self.cancellable = row.cancellable
        self.can_focus = row.clickable or row.cancellable
        self._index = index
        self._primary_text = row.primary_text
        self._secondary_text = row.secondary_text
        self.styles.height = "auto"
        self.styles.min_height = 2
        if row.status:
            self.add_class(f"console-inspector-section-row-{row.status}")

    def compose(self) -> ComposeResult:
        yield Static(
            self._primary_text,
            id=f"console-inspector-section-{self.section_id}-row-{self._index}-primary",
            classes="console-inspector-section-row-primary",
            markup=False,
        )
        yield Static(
            self._secondary_text,
            id=f"console-inspector-section-{self.section_id}-row-{self._index}-secondary",
            classes="console-inspector-section-row-secondary",
            markup=False,
        )

    def _on_click(self, event: events.Click) -> None:
        if not self.clickable:
            return
        self.post_message(ConsoleInspectorSection.RowActivated(self.section_id, self.row_id))

    def action_activate_row(self) -> None:
        if not self.clickable:
            return
        self.post_message(ConsoleInspectorSection.RowActivated(self.section_id, self.row_id))

    def action_cancel_row(self) -> None:
        if not self.cancellable:
            return
        self.post_message(
            ConsoleInspectorSection.RowCancelRequested(self.section_id, self.row_id)
        )
