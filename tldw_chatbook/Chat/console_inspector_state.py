"""Pure display state for Console Inspector sections (no Textual).

Moved out of ``Widgets/Console/console_inspector_section.py`` (TASK-36944.4) so
non-widget modules can import the value objects and width budgets without
loading Textual. The widget module re-exports every name here.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.cells import cell_len


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
        wrap_secondary: Wrap essential recovery instructions instead of
            truncating them to the usual one-line detail preview.
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
    wrap_secondary: bool = False


@dataclass(frozen=True)
class ConsoleInspectorSectionState:
    """Atomic rows, header summary, and optional wrapped notice.

    Passed as ONE value so the two dimensions can never drift independently
    -- a caller that wants to refresh only the rows still passes the
    summary it already had (its own held state, unchanged), rather than
    the previous kwargs shape where omitting ``summary`` silently meant
    "clear it" (task-3 review round 1 finding). Mirrors
    ``ConsoleInspectorState`` (`console_display_state.py`), the same
    full-snapshot-per-sync pattern ``ConsoleRunInspector`` already uses.

    **Rows and summary are REQUIRED -- neither has a default** (task-3 review
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
        notice: Optional wrapped explanation after the preview rows.
    """

    rows: tuple[InspectorSectionRow, ...]
    summary: str
    notice: str = ""
