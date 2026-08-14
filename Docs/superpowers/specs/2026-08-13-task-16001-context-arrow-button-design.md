# TASK-16001: Full-width Context arrow button

## Summary

Render the horizontal collapsed Console Context handle as the exact ASCII label
`Context--->`. The existing button remains the only interactive element, its
outer and content widths remain unchanged, and the final `>` cell opens the
Context rail.

## Current behavior and constraint

The Context handle is 13 cells wide outside and 11 cells wide inside its frame.
Its canonical state label is `Context ▸`, which the horizontal Console handle
currently renders unchanged. Textual's Button reserves one line-padding cell on
each side, leaving nine cells for visible text.

`Context--->` is exactly 11 terminal cells: seven letters plus three hyphens and
one greater-than sign. It therefore fits the existing content width only when
the horizontal left Console button clears Textual's default `line_pad=1`.

## Approved design

- The horizontal left Console handle renders `Context--->` only when its label
  is the canonical Context label.
- The existing `#console-context-rail-open` Button owns all 11 visible cells.
- The final `>` cell is part of the same hit target and opens the Context rail.
- The button keeps the fixed `Open Context rail` tooltip.
- The handle remains 13 cells wide with an 11-cell framed content region.
- The vertical option continues to render `Context` one character per row.
- Noncanonical left-side labels continue to render unchanged.
- The Inspector keeps `Inspect->`; its geometry, badge, and behavior do not
  change.
- No CSS-file rule, additional widget, new ID/class, runtime width calculation,
  or layout-structure change is introduced.

## Component boundary

Keep the change inside `ConsoleRailHandle`, which owns Console-specific rail
vocabulary. `DestinationRailHandle` remains destination-agnostic and unchanged.
The existing horizontal-left Button branch will clear its inline line padding,
and `_display_label()` will translate only the canonical Context label to the
approved ASCII display literal. The vertical branch remains first so it keeps
normalizing away the canonical disclosure glyph before stacking the word.

## Interaction and responsive behavior

Clicking any cell of `Context--->`, including its last cell, follows the current
Context open path and preserves persistence, focus, and close behavior. Compact
layouts, automatic rail visibility, settings-driven vertical labels, and the
open Context rail are unchanged. The label must remain one composited row at
representative supported widths and must never push the handle outside the
workspace.

## Alternatives rejected

### Dynamic dash fill

Computing the arrow from runtime width would couple copy to layout and add
unneeded responsive logic for a fixed-width control.

### Separate arrow widget

A child arrow control would split the hit target, complicate event routing, and
violate the requirement that the entire affordance be one button.

### Shorter `Context->` or `Context-->`

Both fit, but neither fills the approved 11-cell content surface. The user chose
the exact full-width `Context--->` treatment.

## Verification contract

Focused tests will prove:

1. Component-level horizontal Context display is exactly `Context--->`, while
   vertical Context, Inspector, and noncanonical labels remain unchanged.
2. The mounted Button owns the exact label, retains `Open Context rail`, and
   remains inside the unchanged 13/11 geometry.
3. A selector-relative click at `(button.width - 1, button.height // 2)` opens
   the Context rail; moving the test click to `button.width` must fail.
4. A real-Console compositor sweep forces the Context rail collapsed and
   requires exactly one painted `Context--->` row at representative compact,
   intermediate, and wide viewports.
5. Existing settings, compact-access, Context open/close, Inspector, CSS
   integrity, and terminal-frame regressions remain green.
6. Ruff lint, Ruff format, duplicate-task-ID, and diff checks pass. Per user
   instruction, no full repository suite is run.

## ADR check

ADR required: no.

ADR path: N/A.

Reason: this is a reversible presentation refinement inside the existing
Console-specific rail display seam. It changes no ownership boundary, public
interface, persistence model, dependency, or long-lived application structure.
