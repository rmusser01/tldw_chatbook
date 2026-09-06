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

from rich.cells import cell_len
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.destination_rail import GLYPH_COLLAPSED, GLYPH_EXPANDED
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph, resolve_glyph_text
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


#: Content width of the Console Inspect rail's section box at the SMALLEST
#: supported terminal. Measured, not assumed (TASK-31662, TASK-31629 #12):
#: probing the real Console at 80x24 on 2026-09-05 reports
#: ``#console-environment-section`` at ``Size(width=30)``, its body at 29,
#: and each row's own content region at 27; at 200x50 the same three are
#: 36/35/33. Budgets derived from this constant therefore fit EVERY
#: supported width, not just the roomy ones -- the previous 34-column
#: assumption was between the two, so an "Environment" title that fitted in
#: its test painted as "Environm…" on a real 80x24 terminal.
RAIL_CONTENT_WIDTH_MIN = 30

#: Width of a section header's collapse chevron (``compose`` pins all three
#: of the Button's width styles to this).
SECTION_TOGGLE_WIDTH = 3

#: TASK-31664 AC#3: label the "view all" tail button flips to while a
#: caller-driven refresh is in flight. Pressing "Refresh" could go ~12
#: measured seconds with zero visible change when the landed data matched
#: what was already painted -- ``sync_state``'s own equality guard treats
#: that landing as a no-op BY DESIGN (it only patches on a content change),
#: so this acknowledgment has to live outside it. See ``set_view_all_busy``.
VIEW_ALL_BUSY_LABEL = "Refreshing…"

#: Columns a row's own text gets at ``RAIL_CONTENT_WIDTH_MIN``: the section
#: body indents by 1 (``.console-inspector-section-body`` padding) and the
#: row spends 2 more on its own ``padding: 0 1``.
SINGLE_LINE_ROW_BUDGET = RAIL_CONTENT_WIDTH_MIN - 1 - 2


#: Columns one expansion level indents a child row by (TASK-31665 AC#3).
#: Two, not one: a single column reads as a rendering wobble next to the
#: section body's own 1-column padding, and the rail's narrowest content
#: budget (``SINGLE_LINE_ROW_BUDGET`` = 27) still leaves 25 columns for a
#: child row's own text.
ROW_INDENT_COLUMNS = 2


def row_fits_one_line(
    primary_text: str,
    secondary_text: str,
    *,
    budget: int = SINGLE_LINE_ROW_BUDGET,
    indent: int = 0,
) -> bool:
    """Whether a row's primary and secondary can share ONE line.

    Deliberately a pure function of the TEXTS against the smallest
    supported width, not of the widget's measured width: a row's mounted
    shape has to be decided in ``compose`` (before layout has run) and has
    to stay stable across a resize, so deciding it from the narrowest rail
    the app supports is the only answer that is both available in time and
    correct at every width. The cost is that a pair which would fit at
    200x50 but not at 80x24 keeps the two-line form everywhere; the pair is
    never truncated to make one fit.

    TASK-31665 AC#14: measured with ``rich.cells.cell_len``, not ``len``.
    A CJK or emoji title is two terminal cells per character, so ``len``
    under-measured it by up to half -- a pair that ``len`` called a fit
    was mounted in the one-line form and then ellipsized by the primary's
    own ``text-overflow`` at paint time. Branch names and backlog titles
    are ASCII today, but a task title (AC#2 now reads them straight out of
    frontmatter) and a changed file's path are both user data.

    Args:
        primary_text: The row's first-line text.
        secondary_text: The row's dimmed detail text.
        budget: Columns available to the row's own text.
        indent: Columns this row's own indent spends (TASK-31665 AC#3),
            deducted from ``budget`` -- an indented child gets less room,
            and deciding its shape against the un-indented budget would
            reintroduce exactly the truncation this function exists to
            prevent.

    Returns:
        ``True`` when both texts plus one separating column fit ``budget``.
        Always ``False`` for an empty secondary -- that row has no pair to
        put side by side; it renders as its primary line alone.
    """
    if not secondary_text:
        return False
    room = budget - indent
    return cell_len(primary_text) + 1 + cell_len(secondary_text) <= room


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
            Any glyph embedded here (a status marker, a row's own trailing
            "▸"/"▾" affordance marker -- TASK-31664 I3) is resolved through
            ``glyph_fallback.resolve_glyph_text`` at RENDER time
            (``ConsoleInspectorSectionRow``), never here: callers building
            this value (including every pure projection, e.g.
            ``console_environment_state.py``) stay free of the
            ``appearance.ascii_glyphs`` global, so they stay testable
            without an app and produce the same value regardless of the
            live glyph-mode setting. Width/shape decisions
            (``row_fits_one_line``, the structural key) are computed on
            this UNRESOLVED value, so ASCII mode can shift a row's rendered
            width away from the width those decisions assumed. The
            expand/collapse markers the Inspector rows carry are 1-for-1
            substitutes (``▸``→``>``, ``▾``→``v``) and so are exact; the
            STATUS markers in the same table are not (``✓``→``[x]``,
            ``●``→``[*]``, ``◈``→``[s]`` -- one column becoming three), so
            a status-prefixed row resolves up to 2 columns wider than it
            was measured. Harmless except right at the budget edge, where
            it can push a pair that ``row_fits_one_line`` called a fit into
            the primary's own ``text-overflow`` ellipsis.
        secondary_text: Dimmed detail, e.g. a truncated last-step summary.
            Where it renders depends on what it is (TASK-31662): empty
            renders no line at all, a short one shares the primary's line
            right-aligned, and only a pair too wide for
            ``SINGLE_LINE_ROW_BUDGET`` takes a second line. Empty used to
            render as a BLANK second line -- 25% of the Environment
            section's row-lines were that blank.
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
        indent: Expansion depth, in levels (TASK-31665 AC#3). ``0`` is a
            top-level row; ``1`` is a child revealed by expanding the row
            above it. Rendered as ``ROW_INDENT_COLUMNS`` extra columns of
            left padding on the row widget, and deducted from the
            one-line budget so an indented pair is never mounted into a
            shape it cannot fit. Children used to share their parent's
            indent exactly, so the only cue that a block belonged to the
            row above it was the blank line the old two-line row shape
            happened to leave -- a cue TASK-31662 removed when it made
            rows one line tall.
    """

    row_id: str
    primary_text: str
    secondary_text: str = ""
    status: str = ""
    clickable: bool = False
    cancellable: bool = False
    indent: int = 0


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
        suppress_summary_when_open: bool = False,
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
            suppress_summary_when_open: Hide the header summary while the
                body is visible (TASK-31662, AC#3). Opt-in, and the
                default is off, because whether a summary duplicates its
                own rows is a property of the CALLER's data, not of this
                grammar: the Environment section's "branch ±counts" is
                literally its first two rows, while the fleet section's
                "2 working, 1 done" is an aggregate no row restates. The
                columns the summary gives up go back to the title, which
                is what stops "Environment" painting as "Environm…" at
                ``RAIL_CONTENT_WIDTH_MIN``.
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
        # TASK-31664 round-1 review (I1 follow-on, surfaced by that fix's
        # own test): a structural row-set change (PENDING -> the real rows,
        # or any row added/removed) recomposes the WHOLE section -- see
        # `sync_state` -- which rebuilds the tail Button from `compose()`
        # and would silently drop an in-flight "Refreshing…" back to
        # "Refresh" if that state lived only as a live widget mutation.
        # Tracked here so `compose()` can consult it on every rebuild;
        # `set_view_all_busy` updates both this and the live widget.
        self._view_all_busy = False
        self.suppress_summary_when_open = bool(suppress_summary_when_open)
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
                if self._summary_is_suppressed():
                    summary.styles.display = "none"
                yield summary
            if self.collapsible:
                toggle = Button(
                    self._toggle_label(),
                    id=self._toggle_id,
                    classes="console-inspector-section-toggle",
                    compact=True,
                )
                toggle.tooltip = self._toggle_tooltip()
                toggle.styles.width = SECTION_TOGGLE_WIDTH
                toggle.styles.min_width = SECTION_TOGGLE_WIDTH
                toggle.styles.max_width = SECTION_TOGGLE_WIDTH
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
                VIEW_ALL_BUSY_LABEL if self._view_all_busy else self.view_all_label,
                id=self._view_all_id,
                classes="console-inspector-section-view-all",
                compact=True,
            )
            # TASK-31665 AC#5: name the SCOPE. A bare "Refresh" in a rail of
            # stacked sections does not say what it refreshes, and the
            # critique found it reading as a rail-wide (or app-wide)
            # control. The section owns the tail, so the section's title is
            # the honest scope.
            view_all.tooltip = self._view_all_tooltip()
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

    def _summary_is_suppressed(self) -> bool:
        """Whether the header summary should currently be hidden."""
        return self.suppress_summary_when_open and self.open

    def _toggle_label(self) -> str:
        return resolve_glyph(GLYPH_EXPANDED if self.open else GLYPH_COLLAPSED)

    def _toggle_tooltip(self) -> str:
        return f"Collapse {self.title}" if self.open else f"Expand {self.title}"

    def set_open(self, open: bool) -> None:
        """Show or hide the body without recomposing the section.

        The header and its chevron stay painted either way; the summary
        follows ``suppress_summary_when_open`` -- this is the second half
        of that pair, ``compose`` being the first, and skipping it would
        leave the summary in whatever state the section was BUILT in for
        the rest of its life.

        Posts ``CollapseToggled`` so a caller can persist the preference;
        a no-op call (``open`` already matches) posts nothing.

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
            if self.suppress_summary_when_open:
                try:
                    summary = self.query_one(f"#{self._summary_id}", Static)
                except (NoMatches, QueryError):
                    pass  # no summary text -> no Static to reveal
                else:
                    summary.styles.display = (
                        "none" if self._summary_is_suppressed() else "block"
                    )
        self.post_message(self.CollapseToggled(self.section_id, open))

    def set_view_all_busy(self, busy: bool) -> None:
        """Flip the "view all" tail button to a transient acknowledgment.

        TASK-31664 AC#3. Deliberately bypasses ``sync_state``'s rows/summary
        equality guard (see ``VIEW_ALL_BUSY_LABEL``'s module comment): this
        sets the mounted ``Button.label`` directly, so it works even when
        the landed state is byte-identical to what is already painted --
        exactly the case that left Refresh looking dead.

        Round-1 review follow-on: ``busy`` is recorded on ``self`` FIRST,
        unconditionally (even with no tail, or before/without a mount) --
        not just applied to the live widget -- because a structural row-set
        change (e.g. the first real landing after PENDING) recomposes the
        WHOLE section mid-acknowledgment, which rebuilds the tail Button
        from ``compose()`` and would otherwise silently drop "Refreshing…"
        back to "Refresh" the moment anything else about the section
        changed shape. ``compose()`` consults ``self._view_all_busy`` on
        every rebuild, so the label survives.

        Args:
            busy: ``True`` shows ``VIEW_ALL_BUSY_LABEL``; ``False`` restores
                ``view_all_label``. Idempotent either way.
        """
        self._view_all_busy = busy
        if not self.view_all_label or not self.is_mounted:
            return
        try:
            button = self.query_one(f"#{self._view_all_id}", Button)
        except (NoMatches, QueryError):
            return
        button.label = VIEW_ALL_BUSY_LABEL if busy else self.view_all_label
        # Both halves (AC#5): `compose` sets the tooltip on a fresh build,
        # this sets it on a live one, and the two must agree about scope.
        button.tooltip = self._view_all_tooltip()

    def _view_all_tooltip(self) -> str:
        """Tooltip naming what the tail button's action applies to.

        TASK-31665 AC#5. The tail carries no scope in its label ("Refresh",
        "View all"), and the Inspect rail stacks several sections that each
        own one -- so the label alone leaves a user guessing whether the
        control is section-scoped or rail-scoped. Reads the busy state so
        the acknowledgment window says what is being refreshed too.

        Returns:
            ``"<label> — <section title>"``, e.g. ``"Refresh — Environment"``.
        """
        label = VIEW_ALL_BUSY_LABEL if self._view_all_busy else self.view_all_label
        return f"{label} — {self.title}"

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
        order, each in the same LINE SHAPE, with the same summary presence
        (shown vs. hidden) -- they differ at most in row/summary text, row
        status, or row clickability, all of which are safe to patch in
        place.

        The line shape is part of the key (TASK-31662) because it decides
        the row's mounted DOM: a pair that fits shares one line inside a
        ``Horizontal``, a pair that does not is two stacked Statics.
        Patching new text into the wrong shape would either squeeze both
        halves into ``SINGLE_LINE_ROW_BUDGET`` columns or leave a row two
        lines tall to say one thing -- so a shape change recomposes.

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
            tuple(
                (
                    row.row_id,
                    row_fits_one_line(
                        row.primary_text,
                        row.secondary_text,
                        indent=max(0, row.indent) * ROW_INDENT_COLUMNS,
                    ),
                    # TASK-31665 AC#3: the indent is inline padding written
                    # in `ConsoleInspectorSectionRow.__init__`, so an
                    # in-place patch never revisits it -- a row whose depth
                    # changed while its id and line shape did not would keep
                    # the OLD indent forever. Part of the key, therefore.
                    max(0, row.indent),
                )
                for row in rows
            ),
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
            # TASK-31664 I3: resolved at this render seam, not in the
            # (pure) projection that built `row.primary_text` -- see
            # `InspectorSectionRow.primary_text`'s docstring.
            self.query_one(f"#{self._row_primary_id(index)}", Static).update(
                resolve_glyph_text(row.primary_text)
            )
        if row.secondary_text != previous_row.secondary_text:
            secondary = self.query_one(f"#{self._row_secondary_id(index)}", Static)
            secondary.update(row.secondary_text)
            # Same mounted shape (the structural key guarantees it), but an
            # empty secondary is HIDDEN rather than blank -- so a row that
            # gains or loses its detail text has to gain or lose the line
            # too, or the new text renders into a `display: none` Static.
            secondary.styles.display = "none" if not row.secondary_text else "block"
        if row.status != previous_row.status:
            if previous_row.status:
                row_widget.remove_class(
                    f"console-inspector-section-row-{previous_row.status}"
                )
            if row.status:
                row_widget.add_class(f"console-inspector-section-row-{row.status}")


class ConsoleInspectorSectionRow(Vertical):
    """One row inside a ``ConsoleInspectorSection`` body.

    Always mounts BOTH Statics -- consumers query the secondary by id
    whether or not the row has detail text -- but spends a second LINE on
    the secondary only when it needs one (TASK-31662). Three cases:

    * no ``secondary_text``: one line, the secondary Static hidden. This
      row used to render a blank second line; 25% of the Environment
      section's row-lines were exactly that.
    * a pair that fits ``SINGLE_LINE_ROW_BUDGET``: one line, primary
      ``1fr`` and secondary ``auto`` inside a ``Horizontal`` -- the same
      shape the section HEADER uses for title + summary, so the secondary
      lands flush right.
    * a pair that does not fit: the original two-line stack, uncut. The
      fleet section's rows ("glyph name · elapsed" over "task · N tok")
      live here.

    Which case a row is in is part of ``ConsoleInspectorSection``'s
    structural key, so a row that changes shape recomposes rather than
    being patched into the wrong one. ``clickable`` can still change
    across an in-place patch (re-synced unconditionally by
    ``ConsoleInspectorSection._apply_row_update``).
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
        # TASK-31664 I3: resolved here (render time), not by whatever pure
        # projection built `row.primary_text` -- see
        # `InspectorSectionRow.primary_text`'s docstring.
        self._primary_text = resolve_glyph_text(row.primary_text)
        self._secondary_text = row.secondary_text
        self.indent = max(0, row.indent)
        self._one_line = row_fits_one_line(
            row.primary_text,
            row.secondary_text,
            indent=self.indent * ROW_INDENT_COLUMNS,
        )
        self.styles.height = "auto"
        # Inline as well as in the CSS: a bare test harness loads neither
        # the app bundle nor the console-owned split sheet, and geometry
        # has to be right for any host (the same reason the header's
        # summary sets its own width/height inline).
        self.styles.min_height = 1
        if self.indent:
            # TASK-31665 AC#3. MARGIN, not padding: the row's `padding: 0 1`
            # lives in the console-owned sheet, which a bare harness never
            # loads, so an inline `padding` write would have had to restate
            # a value it cannot see -- and would then shift an unstyled row
            # by a different amount than a styled one. Margin is additive to
            # whatever padding the host supplies, so the indent is exactly
            # `ROW_INDENT_COLUMNS` per level in BOTH hosts. Inline because
            # the depth is per-row DATA, not a class.
            self.styles.margin = (0, 0, 0, self.indent * ROW_INDENT_COLUMNS)
            # Deliberately unstyled today (round-1 review M2): the class is
            # the styling hook containment theming would need -- a guide
            # rule, a muted child colour -- and stamping it here is what
            # makes that a CSS-only change later. The INDENT itself is the
            # margin above, never this class, so no stylesheet is required
            # for the containment cue to work.
            self.add_class("console-inspector-section-row-child")
        if row.status:
            self.add_class(f"console-inspector-section-row-{row.status}")

    @property
    def _primary_id(self) -> str:
        return f"console-inspector-section-{self.section_id}-row-{self._index}-primary"

    @property
    def _secondary_id(self) -> str:
        return (
            f"console-inspector-section-{self.section_id}-row-{self._index}-secondary"
        )

    def _make_primary(self) -> Static:
        primary = Static(
            self._primary_text,
            id=self._primary_id,
            classes="console-inspector-section-row-primary",
            markup=False,
        )
        primary.styles.height = 1
        return primary

    def _make_secondary(self) -> Static:
        secondary = Static(
            self._secondary_text,
            id=self._secondary_id,
            classes="console-inspector-section-row-secondary",
            markup=False,
        )
        secondary.styles.height = 1
        return secondary

    def compose(self) -> ComposeResult:
        if self._one_line:
            line = Horizontal(classes="console-inspector-section-row-line")
            line.styles.height = 1
            line.styles.width = "100%"
            with line:
                primary = self._make_primary()
                # `1fr` + `auto` is the header's own title/summary split:
                # the primary takes everything the secondary does not, which
                # is what puts the secondary flush against the right edge.
                primary.styles.width = "1fr"
                primary.styles.min_width = 0
                yield primary
                secondary = self._make_secondary()
                secondary.styles.width = "auto"
                secondary.styles.min_width = 0
                yield secondary
            return
        yield self._make_primary()
        secondary = self._make_secondary()
        if not self._secondary_text:
            # Mounted (consumers query it by id) but costing no line.
            secondary.styles.display = "none"
        yield secondary

    def on_mouse_down(self, _event: events.MouseDown) -> None:
        """Activate before an owning rail can reflow the row under the pointer."""

        if _event.button == 1 and self.clickable:
            self.post_message(
                ConsoleInspectorSection.RowActivated(self.section_id, self.row_id)
            )

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
