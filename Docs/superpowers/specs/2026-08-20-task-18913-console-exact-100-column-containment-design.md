# TASK-18913: Console Exact-100-Column Containment Design

## Summary

Keep the Console's current Context-only layout at exactly 100 terminal columns,
but let the transcript yield its standard minimum-width constraint at that one
responsive boundary. This reuses the existing compact-override mechanism that
already keeps explicitly opened rails solvable at narrower widths. It changes no
rail visibility, preference, ordering, label, focus, or mobile behavior. Exact
100 also becomes its own resize-deduplication state so cold start and live
resize cannot diverge at the new waiver boundary.

## Problem

At 100x30 on current `origin/dev`, a fresh Context-only Console can expand far
beyond the viewport: the Context rail measures roughly 255 columns, the
transcript begins around x=257, and the collapsed Inspector handle lands around
x=1362.

The source-level width budget explains the boundary failure:

- The framed workspace grid has 96 content columns at a 100-column viewport
  after its two border cells and two horizontal padding cells.
- The displayed children require Context's 30-column minimum, the transcript's
  56-column standard minimum, and the horizontal Inspector handle's 11 columns.
- Those minimums total 97, one column more than the grid can provide.
- Textual 8.x does not degrade this fractional row by one cell; its min-clamped
  fractional resolution expands the row dramatically instead.

The same layout is already safe below 100 because Context is responsively
collapsed by default, and safe in compact-override states because the
transcript minimum is waived. The gap is the exact 100-column boundary: Context
is correctly open there, but the existing waiver is defined only below the
boundary.

## Users and desired outcome

- First-time technical and non-technical users keep both the visible Context
  rail and an immediately usable transcript instead of encountering a mostly
  blank/off-screen Console.
- Regular technical and non-technical power users retain the exact stored rail
  state and current keyboard/focus behavior across resize.
- At wider, narrower, stacked-handle, Inspector-open, and single-pane states,
  the current product contract remains unchanged.

## Decision

Make the left-rail compact override inclusive at its 100-column boundary while
leaving the left-rail collapse rule exclusive below 100.

In practical terms, a Context rail that renders open at exactly 100 columns
receives the same existing `left_compact_override` / `compact_override`
authority as an explicitly opened Context rail below 100. The screen's two
existing compose-time and live-sync paths already translate that authority into
`#console-main-column` having a zero minimum width. Textual can then allocate the
actual remaining transcript width (about 55 columns) and keep the complete row
inside the 96-column grid content region.

This deliberately broadens the meaning of the override flag from “an explicit
open below the compact threshold” to “an open rail at or below the compact
boundary whose row may require the transcript minimum-width waiver.” It does
not broaden persistence authority and does not change which rail is open.

Give exactly 100 columns its own semantic width-band key in
`console_rail_width_band()`. The live-resize handler already recomputes effective
rail state whenever that key changes; separating 100 from the existing
100–117 band makes 101→100 acquire the waiver and 100→101 remove it. This is
preferred over recomputing on every resize event or adding a second deduplication
mechanism because it preserves the existing bounded resize architecture.

## Alternatives considered

### Fix Context to 30 columns at exactly 100

This also gives the transcript the remaining width, but introduces a second
boundary-specific rail sizing mode and requires both compose and live-sync code
to keep width units synchronized. Rejected as more brittle than reusing the
existing waiver seam.

### Collapse Context at 100 columns

Changing the collapse condition from below 100 to 100-and-below would avoid the
overflow. Rejected because it changes the visible responsive contract and
removes useful Context information where a usable 55-column transcript fits.

### Reduce the transcript minimum from 56 to 55 everywhere

This removes the one-cell deficit but weakens the standard layout at every
width and still relies on Textual's fractional min-clamp behavior. Rejected as
unnecessarily global and less robust.

## Scope

### In scope

- Pure rail-state classification for the exact 100-column, Context-open state.
- Width-band classification that makes crossing into or out of exact 100 a
  live-recomputation boundary.
- Production-CSS compositor coverage of the full workspace hierarchy.
- Cold-start and resize containment evidence.
- Focus, selected-message, transcript-anchor, and no-persistence evidence for
  the affected resize transitions.
- Regression coverage at adjacent and representative widths.
- Comments and documentation that describe the inclusive override boundary.

### Out of scope

- Changing the 100-column Context collapse threshold.
- Changing 70/74-column explicit-open floors, single-pane behavior, or
  Inspector priority.
- Moving or regrouping Context/Inspector content.
- Changing rail widths, labels, handles, badges, focus, or persistence.
- Changing the established focus result of a manual rail collapse; this task
  covers responsive resize continuity only.
- Phone, touch, hover, pointer, soft-keyboard, or served-browser work owned by
  TASK-18911.
- Adding overlays, new settings, or a second layout system.

## State and layout flow

1. `build_console_rail_state()` receives the available terminal width and
   current stored preferences.
2. Existing collapse logic continues to keep Context open at 100 and collapse
   it by default below 100.
3. When Context is open at 100, the existing left/aggregate compact-override
   flags are set.
4. Exact 100 resolves to its own resize-deduplication band, so entering or
   leaving that width always rebuilds effective rail state.
5. Existing Console compose and visibility-sync paths read the aggregate flag
   and waive the transcript minimum width.
6. Textual resolves Context, transcript, and collapsed Inspector handle inside
   the framed grid.
7. Responsive focus handoff, selected-message identity, and transcript reading
   position remain intact; no preference save is called because of viewport
   width.

There is no new error state or recovery path. If width is unavailable, the
current standard state remains unchanged.

## Verification design

Follow test-driven development:

1. Add a production-hierarchy, exact-production-CSS compositor regression at
   100x30. It must fail on current `origin/dev` because displayed grid children
   escape the grid and viewport. Exercise all four stored rail-preference
   combinations and assert these effective states:

   | Stored Context | Stored Inspector | Displayed left | Displayed right | Compact override |
   | --- | --- | --- | --- | --- |
   | open | closed | Context rail | Inspector handle | left / aggregate |
   | closed | closed | Context handle | Inspector handle | none |
   | closed | open | Context handle | Inspector rail | right / aggregate |
   | open | open | Context handle | Inspector rail | right / aggregate |

   The final row preserves ADR-043's Inspector priority from 100 through 149
   columns; no stored preference is rewritten.
2. For every matrix row, assert that each expected displayed child has positive
   geometry and remains within both grid-content and screen bounds. Measure the
   two distinct width contracts explicitly:

   - In the default Context-open, Inspector-closed row,
     `#console-main-column.region.width` is at least 55 columns. This is the
     outer allocation and includes the transcript region's own chrome.
   - In every row, `#console-native-transcript.content_region.width` is at
     least ADR-043's 40-column usable-content floor. Borders and padding do not
     count toward this readable width.

   Configure a ready provider or intentionally dismiss the setup modal before
   paint assertions, then prove the expected transcript, rail, and handle are
   painted/hittable so positive geometry cannot conceal clipping or an overlay.
3. Add pure state coverage proving Context remains open at 100, its stored
   preference is untouched, and the compact override table is default Context
   collapsed/no override at 99, open/override at 100, and open/no override at
   101.
4. Give exact 100 a distinct width-band expectation and retain the existing
   expectations at 99 and 101.
5. Pin adjacent 99/101-column controls plus 80x24, 120x30, 160x45, and 235x52
   representative layouts.
6. Cover cold start plus 101→100, 100→101, 99→100, and 100→99 live transitions.
   Assert containment and both width contracts after every transition. Focus
   remains on the same control for 101↔100; the established Context
   handle-to-collapse-control handoff applies for 99→100, with its inverse for
   100→99.
7. Use a populated transcript to prove selected-message identity and
   tail-follow/released-anchor state survive the four transitions. Spy on
   `_save_console_rail_preferences` and assert zero calls during cold start and
   responsive resize, rather than relying only on deep-equal stored values.
8. Mutation-check the regression by reverting the inclusive boundary and the
   exact-100 resize band independently: the relevant cold-start or live-resize
   test must fail, then pass again when restored.
9. Update state, compose-time, and live-sync comments to call compact override
   layout-minimum-waiver authority and to use the shipped 30-column Context
   minimum, 11-column horizontal Inspector handle, and four cells of grid
   border/padding in the exact-100 arithmetic.
10. Run the directly reachable Console rail/state/resize/compositor suites,
   import provenance, CSS bundle integrity, Ruff, formatting, duplicate task-ID,
   and diff checks. Record any unchanged repository baseline failures rather
   than claiming they are green.

## Accessibility and NNG alignment

- **Visibility of system status:** the active layout no longer disappears
  beyond the viewport at a common breakpoint.
- **User control and freedom:** stored rail choices and collapse controls remain
  intact and reversible.
- **Consistency and standards:** the fix uses the existing compact-width
  resolution behavior rather than introducing a special overlay or alternate
  rail.
- **Error prevention:** the production-CSS boundary regression prevents a
  one-cell budget mismatch from becoming a catastrophic layout expansion.
- **Flexibility and efficiency:** Context remains visible for power users while
  first-time users retain a readable transcript.
- **Recognition and continuity:** cold start and every adjacent resize direction
  converge without losing the user's focused control, selected message, or
  transcript reading position.

## Architecture decision record

ADR required: no

ADR path: `backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md`

Reason: this is a direct defect correction within ADR-043's existing
minimum-width-waiver and grid-resolution mechanism. It does not change storage,
persistence, rail ownership, responsive priority, or long-lived application
structure.
