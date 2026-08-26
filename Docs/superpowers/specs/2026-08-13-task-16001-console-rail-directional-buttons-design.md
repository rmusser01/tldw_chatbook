# TASK-16001: Directional full-button Console rail controls

## Summary

Make both Console rails communicate direction correctly in both states. The
collapsed handles use inward-pointing ASCII arrows because they open content
toward the transcript. The open rails replace their separate title and tiny
arrow with one full-width header Button whose ASCII arrow points outward
toward the edge where the rail will collapse.

## Corrected intent

The prior TASK-16001 design changed the collapsed Inspector handle to
`Inspect->` and proposed changing the collapsed Context handle to
`Context--->`. That targeted the wrong controls and made the right-side
Inspector arrow point at the terminal wall even though activating it expands
the Inspector inward. This design supersedes that specification and its plan.

The user-approved labels are:

| State | Context / left rail | Inspector / right rail |
| --- | --- | --- |
| Collapsed, opens inward | `Context->` | `<-Inspect` |
| Open, collapses outward | `<---------|Context` | `Inspect|--------->` |

## Approved design

### Collapsed handles

- The horizontal Context handle renders exactly `Context->` on one row.
- The horizontal Inspector handle renders exactly `<-Inspect` on one row.
- Both strings are nine terminal cells and fit the existing handles without a
  width change.
- The entire existing handle Button remains clickable, not only its arrow.
- The fixed tooltips remain `Open Context rail` and `Open Inspector rail`.
- Opt-in vertical handles remain stacked `Context` and `Inspector`; they do
  not gain horizontal ASCII arrows.
- Noncanonical labels passed to `ConsoleRailHandle` remain unchanged.

The canonical state labels in `console_rail_state.py` remain the existing
shared-glyph vocabulary. Only `ConsoleRailHandle._display_label()` translates
the canonical horizontal labels into the approved Console display copy. This
preserves vertical normalization and avoids changing shared
`DestinationRailHandle` behavior governed by ADR-034.

### Open rail headers

- The Context header contains one Button with the exact visible label
  `<---------|Context`. It spans the full one-row header, aligns its text to
  the right, and collapses the rail to the left.
- The Inspector header contains one Button with the exact visible label
  `Inspect|--------->`. It spans the full one-row header, aligns its text to
  the left, and collapses the rail to the right.
- The existing IDs remain `#console-context-rail-collapse` and
  `#console-inspector-rail-collapse`, so current handlers, persistence, focus
  tours, and automation continue through the same event path.
- The fixed tooltips remain `Collapse Console context rail` and
  `Collapse Inspector rail`.
- The separate `Static` title siblings are removed from these two Console
  headers. The visible title is part of the Button, making every painted title,
  separator, dash, and arrow cell one hit target.
- The header remains one row high. The exact fixed strings are used at every
  width; the dash count does not change on resize.

## Interaction and accessibility

The full-width header Button is keyboard focusable through the existing F6
rail tour and keeps the current focus styling. Clicking the title end of either
open header must collapse its rail, proving the user does not need to target
the arrow. Clicking any part of a collapsed handle opens the corresponding
rail. Recompose-sensitive tests re-query the mounted controls after each state
change.

The arrows describe the result of activation:

- left collapsed `Context->` points right, into the workspace;
- right collapsed `<-Inspect` points left, into the workspace;
- left open `<---------|Context` points left, toward its collapse edge;
- right open `Inspect|--------->` points right, toward its collapse edge.

## Responsive and visual behavior

Existing rail widths, minimums, frames, body composition, compact-collapse
thresholds, single-pane fallback, stored preferences, and automatic Inspector
opening remain unchanged. The 18-cell open-header strings fit the rails'
existing minimum width. At compact widths where a rail is hidden or collapsed,
the existing responsive policy still decides which control is mounted.

The open-header Button fills the same header region formerly shared by a title
and a three-cell Button. It introduces no extra row, margin, border, or child
widget. Focus styling continues to use the incumbent Console rail vocabulary.

## Component boundary

Keep the change in the three Console-owned presentation sites:

1. `ConsoleRailHandle` owns the two canonical horizontal collapsed labels.
2. `ConsoleLeftRail.compose()` owns the Context full-width collapse Button.
3. `ConsoleInspectorRail.compose()` owns the Inspector full-width collapse Button.

Do not change `DestinationRailHandle`, shared disclosure glyph constants,
Console rail-state persistence, screen handlers, rail bodies, section headers,
Personas/Lab rails, or responsive layout rules. Use the existing Button IDs and
messages rather than adding event plumbing.

## Alternatives rejected

### Dynamic dash fill

Recomputing the number of hyphens on every resize would add measurement state,
relabeling, and repaint behavior for no functional gain. A fixed label inside
a full-width Button gives the requested visual direction and hit target.

### Clickable header container with separate children

Keeping a `Static` title plus a tiny Button and proxying container clicks would
require custom mouse and keyboard behavior, split the accessible control, and
retain ambiguity about which cells are interactive.

### Shared rail abstraction

Generalizing these labels or header layouts into `DestinationRailHandle` or a
new cross-destination component would alter Lab/Personas behavior and violate
the Console-specific scope.

## Verification contract

Focused tests must prove:

1. Horizontal canonical collapsed labels paint exactly one row as `Context->`
   and `<-Inspect`; vertical and noncanonical controls remain unchanged.
2. Each open rail header has exactly one full-width Button, no separate title
   Static, the exact approved label, the existing ID and tooltip, and a region
   contained by/equal to the one-row header content.
3. Selector-relative clicks on the non-arrow title end of each open header
   collapse the correct rail; out-of-bounds mutation coordinates fail.
4. Collapsed-handle clicks open the correct rail and preserve persisted state.
5. A real-Console compositor sweep shows the exact four directional labels in
   their applicable states across representative supported viewports, with no
   wrapping, clipping, extra row, transcript loss, or responsive-policy drift.
6. Existing F6 focus-tour, compact-access, rail open/collapse, settings-driven
   vertical labels, Inspector badge, terminal-frame, and CSS-bundle integrity
   tests remain green. TASK-15783 keeps its geometry/frame/badge contract and
   test provenance while updating only its two collapsed Inspector copy
   assertions from the superseded `Inspect->` to `<-Inspect`.
7. Ruff lint/format, duplicate-task-ID, and diff checks pass. Per user
   instruction, do not run the full repository test suite.

The obsolete RED expectations committed under the superseded TASK-16001 design
must be replaced, not layered over or left in history as active assertions.

## ADR check

ADR required: no.

ADR path: N/A.

Reason: this is a reversible Console-only presentation and hit-target
correction using existing widgets, IDs, handlers, state, and layout boundaries.
It does not change shared glyph ownership under ADR-034 or introduce a durable
architecture, persistence, dependency, security, or service decision.
