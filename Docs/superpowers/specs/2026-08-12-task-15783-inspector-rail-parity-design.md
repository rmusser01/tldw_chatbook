# TASK-15783: Collapsed Inspector rail parity design

## Goal

Make the collapsed Console Inspector rail read as the right-side counterpart
to the collapsed Context rail: the rail surface fills the workspace height and
the compact visible label is centered vertically. The horizontal handle uses
the shorter action label `Inspect` so it stays on one line. Preserve the
current width, `Open Inspector rail` tooltip, canonical Inspector terminology,
badge vocabulary, vertical-label mode, and open/collapse behavior.

## Current behavior and root cause

`DestinationRailHandle` gives collapsed handles a full-height, bordered,
panel-background base treatment. Three Inspector-side overrides undo that
treatment:

- `.console-rail-handle-right` changes the container to an auto-height,
  three-to-six-row transparent block with no border.
- `DestinationRailHandle.compose()` and
  `.console-rail-handle-button-right` fix the right-side button at three rows.
- `ChatScreen.compose_content()` frames `#console-inspector-rail-handle` with
  `variant="quiet"`, which writes a borderless inline frame that outranks TCSS.

The open Inspector rail is not involved. The mismatch belongs entirely to the
collapsed handle.

## Approaches considered

### A. Full-height parity (selected)

Restore the right-side handle's full-height panel and border treatment, then
let its button consume the available vertical space. When a badge exists, it
keeps its natural row at the bottom and the label centers within the remaining
button area.

This directly meets the request and reuses the existing handle structure.

### B. Filled wrapper with the compact button retained

Fill the column background but keep the three-row button at the top. This is a
smaller geometry change, but the label remains top-weighted and does not meet
the requested vertical centering.

### C. Narrow vertically stacked label

Use the opt-in vertical-label presentation developed for TASK-1335. This
recovers horizontal space, but changes reading direction and is a different
interaction/design decision from visual parity with the Context rail.

### Follow-up copy refinement

The initial `Inspector` button copy wraps inside the bordered nine-column
content region. Three shorter labels were considered after reviewing the live
render:

- `Inspect` (selected) is a specific action, keeps the established Inspector
  meaning, and fits the horizontal handle on one line.
- `Details` fits but broadens the control beyond inspection.
- `Info` is shortest but too vague for an action control.

Only the horizontal button's visible copy changes. The canonical rail name,
DOM ids, class names, tooltip, and opt-in vertically stacked presentation keep
`Inspector`.

## Design

The existing `ConsoleRailHandle` and `DestinationRailHandle` boundaries stay
unchanged. `ConsoleRailHandle` adds an Inspector-specific class when its side
is `right`; the new class carries the full-height panel geometry so the shared
`.console-rail-handle-right` behavior used by Lab and Personas is unchanged.
`ConsoleRailHandle.compose()` provides the matching Console-only Python
override for the right-side button because the base widget's fixed inline
styles outrank TCSS: it sets the button to `1fr` height with a zero minimum and
no fixed maximum, and sizes it to the bordered handle's content width rather
than the base widget's 11-column outer width. This keeps an optional badge's
natural row at the bottom without overlap or horizontal overflow while the
handle itself remains 11 columns wide. The Console compose call uses the
normal solid frame instead of the quiet frame, matching the existing Context
handle's frame.

No ids, messages, state builders, persistence keys, rail widths, responsive
thresholds, tooltips, or badge abbreviations change. The only copy change is
the horizontal right-handle display label from `Inspector` to `Inspect`. The
open Inspector rail, vertically stacked label mode, and Context rail are
untouched.

The component stylesheet remains the source of truth. The generated modular
stylesheet must be rebuilt with the repository CSS generator.

## Verification

Add a mounted handle regression in a harness that loads the production
`tldw_cli_modular.tcss` bundle. It must fail against the current implementation
and prove:

- left and right collapsed handles occupy the exact harness content height;
- the right label's button height equals the handle content height minus the
  badge height and uses `content-align: center middle`;
- the right handle has a non-transparent panel background and a solid border;
- deterministic unbadged and `3 approvals` states both render correctly;
- the badge's visible `3 appr` text does not overlap the button and its bottom
  and right edges stay within the handle content bounds.
- the button's left and right edges stay within `handle.content_region`,
  proving the new solid border does not cause horizontal overflow.
- the horizontal right-handle button displays `Inspect` without a newline,
  while its tooltip remains `Open Inspector rail` and vertical-label mode
  continues to render the canonical Inspector name.

The one-line check must inspect the button's actual composited region and
require exactly one non-empty painted row equal to `Inspect`. A whole-SVG
substring assertion is insufficient: the screenshot title contains
`Inspector` (and therefore the substring `Inspect`) even when the rail button
is absent, wrapped, or wrong. Keep the tooltip and vertical-label assertions
as focused component contracts so each invariant fails independently.

The regression inventory includes every current horizontal collapsed-label
expectation, not only the TASK-15783 sweep:

- `Tests/UI/test_console_rail_handle.py`
- `Tests/UI/test_destination_rail.py`
- `Tests/UI/test_console_shell_regions.py`
- `Tests/UI/test_settings_console_rail_labels.py`
- `Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py`

Update only expectations that observe the horizontal collapsed button. Do not
change vertical `I\nn\ns\np\ne\nc\nt\no\nr` expectations, Inspector headings,
settings copy, tooltips, or unrelated destination text. Run these modules with
the existing TASK-15783 focused suite.

Update the existing Console decomposition assertion from a quiet/no-border
Inspector handle to a solid/all-edge frame. Add a non-Console right-handle
assertion proving the Lab/Personas shared default remains compact and quiet.

Then run the focused destination/Console rail tests, CSS build integrity check,
and a live Console render at representative terminal widths. The visual check
must inspect both the deterministic unbadged and badged Inspector states.

## ADR check

ADR required: no

ADR path: backlog/decisions/017-console-left-rail-usability.md

Reason: Existing ADR-017 applies. This is a reversible presentation refinement
inside its established text-only Console rail visual language and introduces
no new architectural decision, storage, state ownership, service contract,
security boundary, dependency, or long-lived application structure.
